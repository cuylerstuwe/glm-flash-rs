//! Batched exact byte movement for per-round verifier bookkeeping: index undo
//! rows and the all-layer KDA tail/state commit, one launch each instead of
//! hundreds of individual device copies. Tables are kernel parameters.
use crate::{
    cuda::{Device, Kernel, Stream},
    front::load_native,
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, path::Path, rc::Rc};

pub(crate) const INDEX_LAYERS: usize = 16;
pub(crate) const KDA_LAYERS: usize = 34;
const STATE_BLOCKS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct IndexUndo {
    pub ring_keys: [u64; INDEX_LAYERS],
    pub ring_gates: [u64; INDEX_LAYERS],
    pub undo_keys: [u64; INDEX_LAYERS],
    pub undo_gates: [u64; INDEX_LAYERS],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct KdaCommit {
    pub conv: [u64; KDA_LAYERS],
    pub proj: [u64; KDA_LAYERS],
    pub tail: [u64; KDA_LAYERS],
    pub state_src: [u64; KDA_LAYERS],
    pub state_dst: [u64; KDA_LAYERS],
}
impl Default for KdaCommit {
    fn default() -> Self {
        Self {
            conv: [0; KDA_LAYERS],
            proj: [0; KDA_LAYERS],
            tail: [0; KDA_LAYERS],
            state_src: [0; KDA_LAYERS],
            state_dst: [0; KDA_LAYERS],
        }
    }
}
fn aligned(pointers: &[u64]) -> bool {
    pointers.iter().all(|p| *p != 0 && p.is_multiple_of(16))
}
pub(crate) struct BatchCopy {
    index: Kernel,
    kda: Kernel,
}
impl BatchCopy {
    pub(crate) fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        let path = native.join("batch_copy.cubin");
        Ok(Self {
            index: load_native(device, &path, "index_undo", 32)?,
            kda: load_native(device, &path, "kda_commit", 256)?,
        })
    }
    /// Save (restore = false) rows `first..rows` of the window from each ring to
    /// its undo buffers, or restore them. Same bytes as per-row copies.
    /// # Safety
    /// Pointers must be live, non-overlapping allocations of the stated geometry
    /// through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn index_undo(
        &self,
        stream: &Stream,
        table: &IndexUndo,
        layers: usize,
        position: usize,
        first: usize,
        rows: usize,
        ring_rows: usize,
        row_bytes: usize,
        restore: bool,
    ) -> Result<()> {
        ensure!(
            layers <= INDEX_LAYERS
                && first <= rows
                && rows <= crate::kda_window::MAX_ROWS
                && rows <= ring_rows
                && row_bytes.is_multiple_of(16)
                && row_bytes > 0
                && aligned(&table.ring_keys[..layers])
                && aligned(&table.ring_gates[..layers])
                && aligned(&table.undo_keys[..layers])
                && aligned(&table.undo_gates[..layers]),
            "batched index undo bounds/alignment"
        );
        if layers == 0 || first == rows {
            return Ok(());
        }
        let mut t = *table;
        let mut position = (position % ring_rows) as u32;
        let mut first = first as u32;
        let mut ring_rows32 = ring_rows as u32;
        let mut row_bytes = row_bytes as u32;
        let mut restore = u32::from(restore);
        let mut args = [
            (&mut t as *mut IndexUndo).cast::<c_void>(),
            (&mut position as *mut u32).cast::<c_void>(),
            (&mut first as *mut u32).cast::<c_void>(),
            (&mut ring_rows32 as *mut u32).cast::<c_void>(),
            (&mut row_bytes as *mut u32).cast::<c_void>(),
            (&mut restore as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.index
                .launch(stream, [rows as u32 - first, layers as u32, 2], &mut args)
        }
    }
    /// Rebuild every layer's retained tail from the accepted prefix and, when
    /// `state_bytes` is given, copy the chosen recurrent state, exactly as the
    /// per-layer ordered copies did.
    /// # Safety
    /// As for `index_undo`; `tail` must not alias `conv`/`proj`, and each
    /// state source must not alias its destination.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn kda_commit(
        &self,
        stream: &Stream,
        table: &KdaCommit,
        keep: usize,
        row_bytes: usize,
        proj_row_bytes: usize,
        state_bytes: Option<usize>,
    ) -> Result<()> {
        let copy_state = state_bytes.is_some();
        ensure!(
            (1..=crate::kda_window::MAX_ROWS).contains(&keep)
                && [row_bytes, proj_row_bytes, state_bytes.unwrap_or(16)]
                    .iter()
                    .all(|b| *b > 0 && b.is_multiple_of(16))
                && aligned(&table.conv)
                && aligned(&table.proj)
                && aligned(&table.tail)
                && (!copy_state || (aligned(&table.state_src) && aligned(&table.state_dst)))
                && (0..KDA_LAYERS).all(|l| {
                    table.tail[l] != table.conv[l]
                        && table.tail[l] != table.proj[l]
                        && (!copy_state || table.state_src[l] != table.state_dst[l])
                }),
            "batched KDA commit bounds/alignment"
        );
        let mut t = *table;
        let mut keep = keep as u32;
        let mut row_bytes = row_bytes as u32;
        let mut proj_row_bytes = proj_row_bytes as u32;
        let mut state_bytes = state_bytes.unwrap_or(0) as u32;
        let mut args = [
            (&mut t as *mut KdaCommit).cast::<c_void>(),
            (&mut keep as *mut u32).cast::<c_void>(),
            (&mut row_bytes as *mut u32).cast::<c_void>(),
            (&mut proj_row_bytes as *mut u32).cast::<c_void>(),
            (&mut state_bytes as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            let blocks = 1 + if copy_state { STATE_BLOCKS } else { 0 };
            self.kda
                .launch(stream, [blocks, KDA_LAYERS as u32, 1], &mut args)
        }
    }
}
