//! Grammar-aware selection before top-k. The unmasked collector stays unchanged.
//! Upload storage and results belong to the single CUDA owner. Callers agree
//! mask/policy/position across ranks before entering any GPU collective.
use crate::{
    candidates::{LOCAL, payload_bytes},
    cuda::{Buffer, Device, Kernel, Pinned, Stream, arg},
    front::load_native,
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};

pub(crate) const MASK_WORDS: usize = sampling::VOCAB / 32;
fn word(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}
pub(crate) fn decode(bytes: &[u8], k: usize) -> Result<Vec<Candidate>> {
    let stride = payload_bytes(k)?;
    ensure!(bytes.len() == stride * 4, "masked candidate payload length");
    let mut result = Vec::with_capacity(k * 4);
    for (rank, row) in bytes.chunks_exact(stride).enumerate() {
        let count = word(row, k * 8 + 4) as usize;
        ensure!(
            word(row, k * 8) == 0 && count <= k,
            "masked candidate shard failure"
        );
        let mut previous = None;
        for j in 0..k {
            let bits = word(row, j * 8);
            let id = word(row, j * 8 + 4);
            if j >= count {
                ensure!(bits == 0 && id == 0, "masked candidate padding");
                continue;
            }
            let value = f32::from_bits(bits);
            ensure!(
                value.is_finite()
                    && bits & 0xffff == 0
                    && (rank * LOCAL..(rank + 1) * LOCAL).contains(&(id as usize))
                    && previous.is_none_or(|p| id > p),
                "invalid masked candidate payload"
            );
            previous = Some(id);
            result.push(Candidate { id, value });
        }
    }
    ensure!(!result.is_empty(), "grammar permits no token");
    Ok(result)
}
fn launch(kernel: &Kernel, stream: &Stream, words: &mut [u64]) -> Result<()> {
    let mut args = words.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { kernel.launch(stream, [1, 1, 1], &mut args) }
}
pub(crate) struct MaskedCandidates {
    select: Kernel,
    unpack: Kernel,
    draw: Kernel,
    pub local: Buffer,
    pub gathered: Buffer,
    mask: Buffer,
    values: Buffer,
    ids: Buffer,
    ints: Buffer,
    params: Buffer,
    result: Buffer,
    status: Buffer,
    upload: Pinned,
    mask_upload: Pinned,
}
impl MaskedCandidates {
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        let path = native.join("masked_candidates.cubin");
        Ok(Self {
            select: load_native(device, &path, "masked_candidates", 256)?,
            unpack: load_native(device, &path, "masked_candidates_unpack", 256)?,
            draw: load_native(
                device,
                &native.join("sampling.cubin"),
                "sampling_choose",
                256,
            )?,
            local: device.alloc(payload_bytes(LOCAL)?)?,
            gathered: device.alloc(payload_bytes(LOCAL)? * 4)?,
            mask: device.alloc(MASK_WORDS * 4)?,
            values: device.alloc(256 * 4)?,
            ids: device.alloc(256 * 4)?,
            ints: device.alloc(32)?,
            params: device.alloc(16)?,
            result: device.alloc(8)?,
            status: device.alloc(4)?,
            upload: Pinned::new(device, 48)?,
            mask_upload: Pinned::new(device, MASK_WORDS * 4)?,
        })
    }
    /// Prior upload must be complete. Mask storage remains owned until sync.
    pub unsafe fn upload_mask(&mut self, stream: &Stream, words: &[i32]) -> Result<()> {
        ensure!(words.len() == MASK_WORDS, "grammar vocabulary mask length");
        for (dst, word) in self
            .mask_upload
            .bytes_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(words)
        {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        unsafe {
            self.mask_upload
                .upload_to(stream, &self.mask, 0, 0, MASK_WORDS * 4)
        }
    }
    pub fn local(
        &self,
        stream: &Stream,
        logits: &Buffer,
        rank: usize,
        policy: Sampling,
    ) -> Result<()> {
        ensure!(
            logits.bytes == LOCAL * 2 && rank < 4,
            "masked vocabulary shard"
        );
        let k = policy.local_count();
        payload_bytes(k)?;
        launch(
            &self.select,
            stream,
            &mut [
                logits.ptr,
                self.mask.ptr,
                k as u64,
                (rank * LOCAL) as u64,
                policy.greedy() as u64,
                self.local.ptr,
            ],
        )
    }
    /// No CPU candidate/count readback on the common k20 path. The unpacker
    /// preserves the original fixed width with zero-mass padding for the
    /// already-qualified GPU sampler's reduction order.
    pub unsafe fn gpu(&mut self, stream: &Stream, s: Sampling, position: u64) -> Result<()> {
        let k = s.local_count();
        ensure!(s.gpu_fits(4 * k), "masked GPU draw shape");
        for (dst, word) in self
            .upload
            .bytes_mut()
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip([
                0,
                0,
                s.seed,
                position,
                s.temperature.to_bits(),
                s.top_p.to_bits(),
            ])
        {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        unsafe {
            self.upload.upload_to(stream, &self.ints, 0, 0, 32)?;
            self.upload.upload_to(stream, &self.params, 0, 32, 16)?;
        }
        launch(
            &self.unpack,
            stream,
            &mut [
                self.gathered.ptr,
                k as u64,
                self.values.ptr,
                self.ids.ptr,
                self.ints.ptr,
                s.top_k as u64,
                self.status.ptr,
            ],
        )?;
        launch(
            &self.draw,
            stream,
            &mut [
                self.values.ptr,
                self.ids.ptr,
                self.ints.ptr,
                self.params.ptr,
                self.result.ptr,
            ],
        )
    }
    pub fn read_gpu(&self, s: Sampling) -> Result<(u32, usize)> {
        let mut status = [0; 4];
        self.status.read_into(0, &mut status)?;
        ensure!(status == [0; 4], "invalid gathered masked candidates");
        let mut out = [0; 8];
        self.result.read_into(0, &mut out)?;
        let result = (word(&out, 0), word(&out, 4) as usize);
        ensure!(
            (result.0 as usize) < sampling::VOCAB
                && result.1 > 0
                && result.1 <= s.kept(4 * s.local_count()),
            "invalid masked sampled result"
        );
        Ok(result)
    }
    pub fn read_host(&self, s: Sampling, position: u64) -> Result<(u32, usize)> {
        let k = s.local_count();
        sampling::choose_masked(
            &decode(&self.gathered.read(0, payload_bytes(k)? * 4)?, k)?,
            position,
            s,
            4 * k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn payload(k: usize, counts: [usize; 4]) -> Vec<u8> {
        let stride = payload_bytes(k).unwrap();
        let mut result = vec![0; stride * 4];
        for (rank, count) in counts.into_iter().enumerate() {
            let base = rank * stride;
            for i in 0..count {
                result[base + i * 8..base + i * 8 + 4].copy_from_slice(&1f32.to_le_bytes());
                result[base + i * 8 + 4..base + i * 8 + 8]
                    .copy_from_slice(&((rank * LOCAL + i) as u32).to_le_bytes());
            }
            result[base + k * 8 + 4..base + k * 8 + 8]
                .copy_from_slice(&(count as u32).to_le_bytes());
        }
        result
    }
    #[test]
    fn empty_shards_are_valid_but_global_empty_and_bad_padding_reject() {
        let good = payload(2, [0, 1, 0, 2]);
        let selected = decode(&good, 2).unwrap();
        assert_eq!(
            selected.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![LOCAL as u32, (3 * LOCAL) as u32, (3 * LOCAL + 1) as u32]
        );
        assert!(decode(&payload(2, [0; 4]), 2).is_err());
        for (offset, value) in [
            (0, 1u32),
            (16, 1),
            (20, 3),
            (24, f32::INFINITY.to_bits()),
            (28, 0),
        ] {
            let mut bad = good.clone();
            bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(decode(&bad, 2).is_err(), "offset {offset}");
        }
        assert!(decode(&good[..good.len() - 1], 2).is_err());
    }
    #[test]
    fn single_allowed_token_preserves_sampler_contract() {
        let input = decode(&payload(28, [0, 0, 1, 0]), 28).unwrap();
        for temperature in [0.0, 1e-8, 1.0, 2.0] {
            for top_k in [0, 1, 20, sampling::VOCAB] {
                let p = Sampling::new(19, temperature, top_k, 0.95).unwrap();
                assert_eq!(
                    sampling::choose_masked(&input, 1048575, p, 112).unwrap(),
                    ((2 * LOCAL) as u32, 1)
                );
            }
        }
    }
}
