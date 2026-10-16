//! Bounded finite-logit collection and seeded draw. The owner supplies all-rank
//! agreement, communication and stream completion before consuming any result.
use crate::{
    cuda::{Buffer, Device, Kernel, Pinned, Stream, arg},
    front::load_native,
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};

pub(crate) const LOCAL: usize = sampling::VOCAB / 4;
pub(crate) fn payload_bytes(k: usize) -> Result<usize> {
    ensure!((1..=LOCAL).contains(&k), "candidate width");
    Ok((k + 1) * 8)
}
fn word(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}
pub(crate) fn decode(bytes: &[u8], k: usize) -> Result<Vec<Candidate>> {
    let stride = payload_bytes(k)?;
    ensure!(bytes.len() == stride * 4, "candidate payload length");
    let mut result = Vec::with_capacity(k * 4);
    for (rank, row) in bytes.chunks_exact(stride).enumerate() {
        ensure!(
            word(row, k * 8) == 0 && word(row, k * 8 + 4) as usize == k,
            "candidate shard failure"
        );
        let mut previous = None;
        for j in 0..k {
            let bits = word(row, j * 8);
            let value = f32::from_bits(bits);
            let id = word(row, j * 8 + 4);
            ensure!(
                value.is_finite()
                    && bits & 0xffff == 0
                    && (rank * LOCAL..(rank + 1) * LOCAL).contains(&(id as usize))
                    && previous.is_none_or(|p| id > p),
                "invalid candidate payload"
            );
            previous = Some(id);
            result.push(Candidate { id, value });
        }
    }
    Ok(result)
}
fn launch(kernel: &Kernel, stream: &Stream, words: &mut [u64]) -> Result<()> {
    let mut args = words.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { kernel.launch(stream, [1, 1, 1], &mut args) }
}
pub(crate) struct Candidates {
    select: Kernel,
    unpack: Kernel,
    draw: Kernel,
    pub local: Buffer,
    pub gathered: Buffer,
    values: Buffer,
    ids: Buffer,
    ints: Buffer,
    params: Buffer,
    result: Buffer,
    status: Buffer,
    upload: Pinned,
}
impl Candidates {
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        Ok(Self {
            select: load_native(device, &native.join("candidates.cubin"), "candidates", 256)?,
            unpack: load_native(
                device,
                &native.join("candidates.cubin"),
                "candidates_unpack",
                256,
            )?,
            draw: load_native(
                device,
                &native.join("sampling.cubin"),
                "sampling_choose",
                256,
            )?,
            local: device.alloc(payload_bytes(LOCAL)?)?,
            gathered: device.alloc(payload_bytes(LOCAL)? * 4)?,
            values: device.alloc(256 * 4)?,
            ids: device.alloc(256 * 4)?,
            ints: device.alloc(32)?,
            params: device.alloc(16)?,
            result: device.alloc(8)?,
            status: device.alloc(4)?,
            upload: Pinned::new(device, 48)?,
        })
    }
    pub fn local(&self, stream: &Stream, logits: &Buffer, rank: usize, k: usize) -> Result<()> {
        ensure!(logits.bytes == LOCAL * 2, "single-row vocabulary shard");
        self.local_offset(stream, logits, 0, rank, k)
    }
    pub fn local_offset(
        &self,
        stream: &Stream,
        logits: &Buffer,
        offset: usize,
        rank: usize,
        k: usize,
    ) -> Result<()> {
        payload_bytes(k)?;
        ensure!(
            rank < 4
                && offset.is_multiple_of(LOCAL * 2)
                && offset
                    .checked_add(LOCAL * 2)
                    .is_some_and(|n| n <= logits.bytes),
            "TP4 vocabulary shard"
        );
        launch(
            &self.select,
            stream,
            &mut [
                logits.ptr + offset as u64,
                k as u64,
                (rank * LOCAL) as u64,
                self.local.ptr,
            ],
        )
    }
    /// The prior upload must have completed. Do not modify/drop this owner or
    /// referenced storage until the stream completes; it also owns capture life.
    pub unsafe fn gpu(&mut self, stream: &Stream, s: Sampling, position: u64) -> Result<()> {
        let k = s.local_count();
        ensure!(
            !s.greedy() && s.gpu_fits(4 * k),
            "stochastic GPU draw shape"
        );
        let policy = self.upload.bytes_mut();
        for (i, word) in [
            4 * k as u64,
            s.kept(4 * k) as u64,
            s.seed,
            position,
            s.temperature.to_bits(),
            s.top_p.to_bits(),
        ]
        .iter()
        .enumerate()
        {
            policy[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }
        unsafe {
            self.upload.upload_to(stream, &self.ints, 0, 0, 32)?;
            self.upload.upload_to(stream, &self.params, 0, 32, 16)?;
            self.status.zero_prefix(stream, 4)?;
        }
        launch(
            &self.unpack,
            stream,
            &mut [
                self.gathered.ptr,
                k as u64,
                self.values.ptr,
                self.ids.ptr,
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
    /// Consume only after the owner's stream and transport status are checked.
    pub fn read_gpu(&self, s: Sampling) -> Result<(u32, usize)> {
        let mut status = [0; 4];
        self.status.read_into(0, &mut status)?;
        ensure!(status == [0; 4], "invalid gathered candidates");
        let mut out = [0; 8];
        self.result.read_into(0, &mut out)?;
        let result = (word(&out, 0), word(&out, 4) as usize);
        ensure!(
            (result.0 as usize) < sampling::VOCAB
                && result.1 > 0
                && result.1 <= s.kept(4 * s.local_count()),
            "invalid sampled result"
        );
        Ok(result)
    }
    pub fn read_host(&self, s: Sampling, position: u64) -> Result<(u32, usize)> {
        ensure!(
            !s.greedy(),
            "greedy uses the separately qualified numeric selector"
        );
        let k = s.local_count();
        let bytes = self.gathered.read(0, payload_bytes(k)? * 4)?;
        sampling::choose(&decode(&bytes, k)?, position, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_payload_rejects_corruption() {
        let k = 2;
        let mut good = Vec::new();
        for rank in 0..4 {
            for j in 0..k {
                good.extend(0.0f32.to_le_bytes());
                good.extend(((rank * LOCAL + j) as u32).to_le_bytes());
            }
            good.extend(0u32.to_le_bytes());
            good.extend((k as u32).to_le_bytes());
        }
        assert_eq!(decode(&good, k).unwrap().len(), 8);
        for (offset, value) in [
            (0, f32::NAN.to_bits()),
            (0, 1),
            (4, LOCAL as u32),
            (12, 0),
            (16, 1),
            (20, 3),
        ] {
            let mut bad = good.clone();
            bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(decode(&bad, k).is_err());
        }
        assert!(decode(&good[..good.len() - 1], k).is_err());
        assert!(payload_bytes(0).is_err());
        assert!(payload_bytes(LOCAL + 1).is_err());
    }
}
