//! Bounded all-row verifier sampling: one gather and one host readback. The
//! owner agrees shape/policy before calling and checks transport before read.
use crate::{
    cuda::{Buffer, Device, Kernel, Pinned, Stream, arg},
    front::load_native,
    packed_candidates_layout::{Layout, MAX_LOCAL_BYTES, ROWS},
    sampling::Sampling,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};
fn launch<const N: usize>(
    kernel: &Kernel,
    stream: &Stream,
    rows: usize,
    mut words: [u64; N],
) -> Result<()> {
    let mut args = std::array::from_fn::<_, N, _>(|i| arg(&mut words[i]));
    unsafe { kernel.launch(stream, [rows as u32, 1, 1], &mut args) }
}
#[derive(Default, serde::Serialize)]
struct Stats {
    calls: u64,
    rows: u64,
    gathered_bytes: u64,
    host_fallbacks: u64,
    readbacks: u64,
}
pub(crate) struct Packed {
    stats: Stats,
    select: Kernel,
    unpack: Kernel,
    draw: Kernel,
    pub local: Buffer,
    pub gathered: Buffer,
    values: Buffer,
    ids: Buffer,
    policy: Buffer,
    result: Buffer,
    upload: Pinned,
    host: Vec<u8>,
}
impl Drop for Packed {
    fn drop(&mut self) {
        if let Ok(stats) = serde_json::to_string(&self.stats) {
            eprintln!("verifier_packed_sample={stats}");
        }
    }
}
impl Packed {
    pub fn configured(
        device: &Rc<Device>,
        comm: &crate::comm::Nccl,
        stream: &Stream,
        native: &Path,
    ) -> Result<Option<Self>> {
        let parsed = match std::env::var("ALASKA_VERIFY_PACKED_SAMPLE") {
            Ok(v) => crate::packed_candidates_layout::enabled(Some(&v)),
            Err(std::env::VarError::NotPresent) => crate::packed_candidates_layout::enabled(None),
            Err(e) => Err(e.into()),
        };
        let mut record = [0u8; 16];
        record[..8].copy_from_slice(b"ALPACK01");
        record[8] = u8::from(parsed.is_ok());
        record[9] = u8::from(*parsed.as_ref().unwrap_or(&false));
        let all = comm.exchange_host(device, stream, &record)?;
        ensure!(
            record[8] == 1 && all.iter().all(|r| r == &record),
            "packed sampler policy differs/invalid on a rank"
        );
        let owner = if parsed? {
            Self::new(device, native).map(Some)
        } else {
            Ok(None)
        };
        record[10] = u8::from(owner.is_ok());
        let all = comm.exchange_host(device, stream, &record)?;
        ensure!(
            record[10] == 1 && all.iter().all(|r| r == &record),
            "packed sampler setup failed on a rank"
        );
        owner
    }
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        // The isolated GB10 build is pinned by front; construction fails closed
        // if the exact compiled artifact is absent or changed.
        Ok(Self {
            stats: Stats::default(),
            select: load_native(
                device,
                &native.join("packed_candidates.cubin"),
                "packed_candidates",
                256,
            )?,
            unpack: load_native(
                device,
                &native.join("packed_candidates.cubin"),
                "packed_candidates_unpack",
                256,
            )?,
            draw: load_native(
                device,
                &native.join("sampling.cubin"),
                "sampling_choose",
                256,
            )?,
            local: device.alloc(MAX_LOCAL_BYTES)?,
            gathered: device.alloc(MAX_LOCAL_BYTES * 4)?,
            values: device.alloc(ROWS * 256 * 4)?,
            ids: device.alloc(ROWS * 256 * 4)?,
            policy: device.alloc(ROWS * 48)?,
            result: device.alloc(ROWS * 12)?,
            upload: Pinned::new(device, ROWS * 48)?,
            host: vec![0; MAX_LOCAL_BYTES * 4],
        })
    }
    pub fn local(
        &self,
        stream: &Stream,
        logits: &Buffer,
        rank: usize,
        layout: Layout,
    ) -> Result<()> {
        let ptr = layout.logits(logits.ptr, logits.bytes, rank)?;
        ensure!(
            layout.rank_bytes <= self.local.bytes && layout.gathered_bytes <= self.gathered.bytes,
            "packed candidate storage"
        );
        launch(
            &self.select,
            stream,
            layout.rows,
            [
                ptr,
                layout.k as u64,
                (rank * crate::packed_candidates_layout::LOCAL) as u64,
                self.local.ptr,
            ],
        )
    }
    /// Prior stream work/upload must have completed before this owner is reused.
    /// Caller gathers exactly layout.rank_bytes per rank before this launch.
    pub unsafe fn gpu(
        &mut self,
        stream: &Stream,
        layout: Layout,
        policy: Sampling,
        first: u64,
    ) -> Result<()> {
        layout.encode_policy(
            policy,
            first,
            &mut self.upload.bytes_mut()[..layout.rows * 48],
        )?;
        unsafe { self.launch_gpu(stream, layout) }
    }
    /// As for gpu: the prior upload and work must have completed. Each row
    /// supplies its own request policy and absolute target-sampling position.
    pub unsafe fn gpu_policies(
        &mut self,
        stream: &Stream,
        layout: Layout,
        policies: &[(Sampling, u64)],
    ) -> Result<()> {
        layout.encode_policies(policies, &mut self.upload.bytes_mut()[..layout.rows * 48])?;
        unsafe { self.launch_gpu(stream, layout) }
    }
    unsafe fn launch_gpu(&self, stream: &Stream, layout: Layout) -> Result<()> {
        unsafe {
            self.upload
                .upload_to(stream, &self.policy, 0, 0, layout.rows * 48)?;
        }
        // All result/status words are overwritten or zeroed before consumption.
        // Draw produces the first rows*8 bytes; unpack owns rows*4 status bytes.
        unsafe {
            self.result.zero_prefix(stream, layout.rows * 12)?;
        }
        launch(
            &self.unpack,
            stream,
            layout.rows,
            [
                self.gathered.ptr,
                layout.k as u64,
                layout.rows as u64,
                self.values.ptr,
                self.ids.ptr,
                self.result.ptr + (layout.rows * 8) as u64,
            ],
        )?;
        launch(
            &self.draw,
            stream,
            layout.rows,
            [
                self.values.ptr,
                self.ids.ptr,
                self.policy.ptr,
                self.policy.ptr + (layout.rows * 32) as u64,
                self.result.ptr,
            ],
        )
    }
    /// Call after both transport and stream completion have been checked. No
    /// rows escape this method if any row/rank payload or result is invalid.
    pub fn read(
        &mut self,
        layout: Layout,
        policy: Sampling,
        first: u64,
    ) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            layout == Layout::for_policy(layout.rows, policy, first)?,
            "packed read policy"
        );
        let gpu = policy.gpu_fits(4 * layout.k);
        let result = if gpu {
            let mut bytes = [0; ROWS * 12];
            self.result.read_into(0, &mut bytes[..layout.rows * 12])?;
            layout.results(&bytes[..layout.rows * 12], policy)
        } else {
            let bytes = &mut self.host[..layout.gathered_bytes];
            self.gathered.read_into(0, bytes)?;
            layout.choose(bytes, policy, first)
        }?;
        self.stats.calls += 1;
        self.stats.rows += layout.rows as u64;
        self.stats.gathered_bytes += layout.gathered_bytes as u64;
        self.stats.readbacks += 1;
        self.stats.host_fallbacks += u64::from(!gpu);
        Ok(result)
    }
    pub fn read_policies(
        &mut self,
        layout: Layout,
        policies: &[(Sampling, u64)],
    ) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            policies.len() == layout.rows
                && policies.iter().all(|&(p, at)| {
                    Layout::for_policy(1, p, at).is_ok() && p.local_count() == layout.k
                }),
            "independent packed read policy"
        );
        let gpu = policies.iter().all(|&(p, _)| p.gpu_fits(4 * layout.k));
        let result = if gpu {
            let mut bytes = [0; ROWS * 12];
            self.result.read_into(0, &mut bytes[..layout.rows * 12])?;
            layout.results_policies(&bytes[..layout.rows * 12], policies)
        } else {
            let bytes = &mut self.host[..layout.gathered_bytes];
            self.gathered.read_into(0, bytes)?;
            layout.choose_policies(bytes, policies)
        }?;
        self.stats.calls += 1;
        self.stats.rows += layout.rows as u64;
        self.stats.gathered_bytes += layout.gathered_bytes as u64;
        self.stats.readbacks += 1;
        self.stats.host_fallbacks += u64::from(!gpu);
        Ok(result)
    }
}
