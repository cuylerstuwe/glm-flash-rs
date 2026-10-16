//! Compact DFlash context snapshots. Only the valid sliding window is copied;
//! target state and cache accounting/format integration remain the caller's job.
use super::{Buffer, Drafter, Engine};
use anyhow::{Result, ensure};
use std::rc::Rc;

const RING: usize = 4096;
const WINDOW: usize = 2047;
const HEAD_BYTES: usize = 128 * 2;
const HEADS: usize = 2;
const LAYERS: usize = 5;

pub(in crate::model) struct Snapshot {
    pub(in crate::model) device: Rc<crate::cuda::Device>,
    pub(in crate::model) model: String,
    pub(in crate::model) end: usize,
    pub(in crate::model) lo: usize,
    // Layer-major K followed by layer-major V, each [2, end-lo, 128] BF16.
    pub(in crate::model) buffers: Vec<Buffer>,
}
impl Snapshot {
    pub(in crate::model) fn gpu_bytes(&self) -> usize {
        self.buffers.iter().map(|v| v.bytes).sum()
    }
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        // Same owner/lifetime fence as target snapshots. Never free a copy
        // source while its native stream can still refer to it.
        let _ = self.device.sync();
    }
}

/// (ring offset, packed offset, bytes), preserving head-major tensor layout.
fn ranges(lo: usize, end: usize) -> Result<Vec<(usize, usize, usize)>> {
    ensure!(
        lo <= end && end <= 1048576 && end - lo <= WINDOW,
        "drafter snapshot window"
    );
    let rows = end - lo;
    let first = rows.min(RING - lo % RING);
    let mut result = Vec::with_capacity(4);
    for head in 0..HEADS {
        if first != 0 {
            result.push((
                (head * RING + lo % RING) * HEAD_BYTES,
                head * rows * HEAD_BYTES,
                first * HEAD_BYTES,
            ));
        }
        if first < rows {
            result.push((
                head * RING * HEAD_BYTES,
                (head * rows + first) * HEAD_BYTES,
                (rows - first) * HEAD_BYTES,
            ));
        }
    }
    Ok(result)
}

impl Drafter {
    pub(in crate::model) fn capture_context(&self, e: &Engine<'_>) -> Result<Snapshot> {
        e.sync()?;
        let end = self.end();
        let stored_lo = i64::from_le_bytes(self.context().lo.read(0, 8)?.try_into().unwrap());
        ensure!(
            stored_lo >= 0 && stored_lo as usize <= end,
            "drafter snapshot lower bound"
        );
        ensure!(
            self.context().pos.read(0, 8)? == (end as i64).to_le_bytes(),
            "drafter snapshot cursor"
        );
        let lo = (stored_lo as usize).max(end.saturating_sub(WINDOW));
        let ranges = ranges(lo, end)?;
        let bytes = (end - lo) * HEADS * HEAD_BYTES;
        let buffers = if bytes == 0 {
            Vec::new()
        } else {
            (0..LAYERS * 2)
                .map(|_| e.device.alloc(bytes))
                .collect::<Result<Vec<_>>>()?
        };
        let snapshot = Snapshot {
            device: e.device.clone(),
            model: self.p.model_identity()?,
            end,
            lo,
            buffers,
        };
        for (src, dst) in self
            .context()
            .kc
            .iter()
            .chain(&self.context().vc)
            .zip(&snapshot.buffers)
        {
            for &(ring, packed, bytes) in &ranges {
                unsafe {
                    dst.copy_from(&e.stream, packed, src, ring, bytes)?;
                }
            }
        }
        e.sync()?;
        Ok(snapshot)
    }

    pub(in crate::model) fn restore_context(
        &mut self,
        e: &Engine<'_>,
        saved: &Snapshot,
    ) -> Result<()> {
        // Complete all validation before any live state is changed. The eventual
        // distributed caller must agree success and fail-stop a partial copy.
        ensure!(
            Rc::ptr_eq(&e.device, &saved.device) && saved.model == self.p.model_identity()?,
            "drafter snapshot owner/model"
        );
        let ranges = ranges(saved.lo, saved.end)?;
        let bytes = (saved.end - saved.lo) * HEADS * HEAD_BYTES;
        ensure!(
            saved.buffers.len() == if bytes == 0 { 0 } else { LAYERS * 2 }
                && saved.buffers.iter().all(|b| b.bytes == bytes),
            "drafter snapshot payload"
        );
        e.sync()?;
        for (dst, src) in self
            .context()
            .kc
            .iter()
            .chain(&self.context().vc)
            .zip(&saved.buffers)
        {
            for &(ring, packed, bytes) in &ranges {
                unsafe {
                    dst.copy_from(&e.stream, ring, src, packed, bytes)?;
                }
            }
        }
        // Rows outside [lo,end) may belong to a previous owner. The attention
        // kernel's existing lower bound and window mask must exclude them.
        e.sync()?;
        self.context()
            .lo
            .write(0, &(saved.lo as i64).to_le_bytes())?;
        self.context()
            .pos
            .write(0, &(saved.end as i64).to_le_bytes())?;
        self.contexts[self.active].end = saved.end;
        Ok(())
    }
}

pub(super) fn probe(e: &Engine<'_>, draft: &mut Drafter) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let hash =
        |b: &Buffer| -> Result<String> { Ok(format!("{:x}", Sha256::digest(b.read(0, b.bytes)?))) };
    let policy = crate::sampling::Sampling::new(731, 1., 20, 0.95)?;
    let mut cases = Vec::new();
    for (base, count) in [
        (0usize, 0usize),
        (0, 128),
        (3900, 320),
        (0, 2304),
        (1048192, 320),
    ] {
        draft.bind(e, 0)?;
        draft.start_context(e, base)?;
        for row in (0..count).step_by(64) {
            let n = (count - row).min(64);
            let bytes: Vec<_> = (0..n * 20480)
                .flat_map(|i| {
                    half::bf16::from_f32(
                        (((base + row + i / 20480) * 13 + i % 20480 * 7) % 127) as f32 / 64. - 1.,
                    )
                    .to_bits()
                    .to_le_bytes()
                })
                .collect();
            e.sync()?;
            draft.taps.write(0, &bytes)?;
            draft.add_taps(e, n)?;
        }
        let saved = draft.capture_context(e)?;
        let payload = saved.buffers.iter().map(hash).collect::<Result<Vec<_>>>()?;
        let expected = if count != 0 {
            Some((
                draft.propose(e, 198, policy)?,
                hash(&draft.logits)?,
                hash(&draft.projected)?,
            ))
        } else {
            None
        };
        for slot in [1usize, 3, 0] {
            draft.bind(e, slot)?;
            // Populate all destination rows with unrelated valid BF16 bytes.
            // Restoring must replace the valid window and mask every other row.
            e.sync()?;
            for b in draft.context().kc.iter().chain(&draft.context().vc) {
                b.write(0, &vec![0x3f; b.bytes])?;
            }
            draft.restore_context(e, &saved)?;
            let restored = draft.capture_context(e)?;
            ensure!(
                restored.end == saved.end
                    && restored.lo == saved.lo
                    && restored
                        .buffers
                        .iter()
                        .map(hash)
                        .collect::<Result<Vec<_>>>()?
                        == payload,
                "drafter compact snapshot bytes/cursor changed"
            );
            if let Some((tokens, logits, projected)) = &expected {
                ensure!(
                    draft.propose(e, 198, policy)? == *tokens
                        && hash(&draft.logits)? == *logits
                        && hash(&draft.projected)? == *projected,
                    "drafter snapshot changed graph/proposal"
                );
            }
        }
        ensure!(
            saved.gpu_bytes() == (saved.end - saved.lo) * HEADS * HEAD_BYTES * LAYERS * 2,
            "drafter snapshot byte charge"
        );
        cases.push(serde_json::json!({"base":base,"end":saved.end,"lo":saved.lo,"gpu_bytes":saved.gpu_bytes(),"destinations":3,"exact_payload_cursor_proposal_and_logits":true}));
    }
    Ok(
        serde_json::json!({"cases":cases,"scope":"Compact valid-window capture/restore across three poisoned retained slots and existing graph owners. RAM/SSD format and native request integration remain separate."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_ranges_cover_each_head_and_window_once() -> Result<()> {
        for lo in [0usize, 1, 1984, 2047, 4095, 4096, 1048384] {
            for rows in [0usize, 1, 64, 128, 2047] {
                if lo + rows > 1048576 {
                    continue;
                }
                let ranges = ranges(lo, lo + rows)?;
                let mut covered = vec![false; HEADS * rows];
                for (ring, packed, bytes) in ranges {
                    assert!(ring + bytes <= HEADS * RING * HEAD_BYTES);
                    assert!(packed + bytes <= HEADS * rows * HEAD_BYTES);
                    for i in 0..bytes / HEAD_BYTES {
                        let out = packed / HEAD_BYTES + i;
                        assert!(!covered[out]);
                        covered[out] = true;
                        let head = out / rows;
                        assert_eq!(
                            ring / HEAD_BYTES + i,
                            head * RING + (lo + out % rows) % RING
                        );
                    }
                }
                assert!(covered.into_iter().all(|v| v));
            }
        }
        for (lo, end) in [(1, 0), (0, 2048), (0, 1048577), (usize::MAX, usize::MAX)] {
            assert!(ranges(lo, end).is_err());
        }
        Ok(())
    }
}
