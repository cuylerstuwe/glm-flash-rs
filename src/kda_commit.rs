//! All-layer recurrent-state replay with a pinned CUDA kernel.
use crate::{
    cuda::{Buffer, Device, Kernel, Pinned, Stream, arg},
    kda_commit_plan::{LAYERS, Source, TABLES, table},
    kda_window::HEADS,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};

pub(crate) struct Batch {
    kernel: Kernel,
    tables: Vec<Buffer>,
    keep: Buffer,
    host_keep: Pinned,
}
impl Batch {
    pub(crate) fn new(
        device: &Rc<Device>,
        native: &Path,
        pointers: &[Vec<[u64; 6]>],
    ) -> Result<Self> {
        ensure!(pointers.len() == TABLES, "batched KDA slot/parity count");
        let tables = pointers
            .iter()
            .map(|p| {
                let bytes = table(p)?;
                let buffer = device.alloc(bytes.len())?;
                buffer.write(0, &bytes)?;
                Ok(buffer)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            kernel: crate::front::load_native(
                device,
                &native.join("kda.cubin"),
                "replay_slots_kernel",
                1024,
            )?,
            tables,
            keep: device.alloc(LAYERS * 4)?,
            host_keep: Pinned::new(device, LAYERS * 4)?,
        })
    }

    /// All pointer owners and this owner must outlive stream completion. The
    /// caller must have drained the previous invocation before rewriting pinned
    /// counters, and every save must be from the current exclusive proposal.
    /// Publication remains the caller's existing post-acceptance transaction.
    pub(crate) unsafe fn replay(
        &mut self,
        stream: &Stream,
        slot: usize,
        parity: usize,
        rows: usize,
        keep: usize,
    ) -> Result<()> {
        ensure!(
            slot < TABLES / 2 && parity < 2 && Source::new(rows, keep)? == Source::Replay,
            "batched KDA replay requires a partial accepted prefix"
        );
        for bytes in self.host_keep.bytes_mut().as_chunks_mut::<4>().0 {
            bytes.copy_from_slice(&(keep as i32).to_le_bytes());
        }
        unsafe {
            self.host_keep
                .upload_to(stream, &self.keep, 0, 0, LAYERS * 4)?;
            let mut args = [
                HEADS as u64,
                self.tables[slot * 2 + parity].ptr,
                self.keep.ptr,
                self.keep.ptr, // run[] is not read when skip_full is zero.
                0,             // grid-x has one layer; every grid-y is a layer.
                0,             // Partial replay only; no full-accept skip needed.
            ];
            self.kernel.launch(
                stream,
                [HEADS as u32, LAYERS as u32, 1],
                &mut args.each_mut().map(arg),
            )
        }
    }
}

#[derive(Default, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) rounds: u64,
    pub(crate) unchanged: u64,
    pub(crate) full: u64,
    pub(crate) partial: u64,
    pub(crate) replay_launches: u64,
    pub(crate) state_copy_bytes: u64,
    pub(crate) avoided_state_copy_bytes: u64,
}
