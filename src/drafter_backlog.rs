//! Retained device features while lookup delays DFlash context updates.
//! Copies and updates are enqueued on the existing engine stream. The native
//! owner must stay unhealthy until its ordinary drained, all-rank commit; these
//! methods deliberately add no per-append/flush collective or synchronization.
use super::{Drafter, Engine};
use crate::{
    accepted_features::{self, Backlog, Lease, STORAGE_BYTES, Status, Storage, WIDTH},
    cuda::{Buffer, Device},
};
use anyhow::{Result, ensure};
use std::rc::Rc;

const _: () = assert!(WIDTH == crate::target_taps::WIDTH);
const _: () = assert!(accepted_features::ROWS == super::MAX);
const _: () = assert!(crate::kda_window::MAX_ROWS == 16);
const _: () = assert!(crate::cache_pages::CONTEXT == 1 << 20);
const _: () = assert!(crate::cache_pages::SLOTS == 4);

impl Storage for Buffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

pub(in crate::model) struct Retained {
    state: Backlog<Buffer>,
    device: Rc<Device>,
}

impl Retained {
    pub(in crate::model) fn new(d: &Rc<Device>, slot: usize, cursor: usize) -> Result<Self> {
        Ok(Self {
            state: Backlog::new(slot, cursor, d.alloc(STORAGE_BYTES)?)?,
            device: d.clone(),
        })
    }

    pub(in crate::model) fn lease(&self) -> Lease {
        self.state.lease()
    }

    pub(in crate::model) fn status(&self) -> Status {
        self.state.status()
    }

    pub(in crate::model) fn available(&self, lease: Lease) -> Result<usize> {
        self.state.available(lease)
    }

    /// The caller has already agreed the clipped accepted prefix and owner on
    /// all ranks. Verifier taps use source_row=0; serial capture can select its
    /// one committed row in the shared ring. The source must remain allocated
    /// until stream completion and may be overwritten only in stream order.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn append(
        &mut self,
        e: &Engine<'_>,
        lease: Lease,
        position: usize,
        source: &Buffer,
        source_row: usize,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &e.device),
            "accepted-feature device owner"
        );
        ensure!(
            rows <= crate::kda_window::MAX_ROWS,
            "accepted-feature source rows"
        );
        let source_offset = source_row
            .checked_mul(WIDTH)
            .ok_or_else(|| anyhow::anyhow!("accepted-feature source offset overflow"))?;
        ensure!(
            source_offset <= source.bytes && rows * WIDTH <= source.bytes - source_offset,
            "accepted-feature source range"
        );
        self.state
            .append(lease, position, rows, |storage, offset, rows| {
                unsafe {
                    storage.copy_from(
                        &e.stream,
                        offset * WIDTH,
                        source,
                        source_offset,
                        rows * WIDTH,
                    )?;
                }
                Ok(())
            })
    }

    /// Enqueue one contiguous copy and one at-most-64-row drafter update. As
    /// with Drafter::add_taps, local cursor advance is provisional until the
    /// enclosing native work boundary drains and agrees successful completion.
    pub(in crate::model) fn flush(
        &mut self,
        e: &Engine<'_>,
        draft: &mut Drafter,
        lease: Lease,
    ) -> Result<usize> {
        ensure!(
            Rc::ptr_eq(&self.device, &e.device) && draft.active == self.status().slot,
            "accepted-feature drafter owner"
        );
        self.state.flush(lease, draft.end(), |storage, _, rows| {
            unsafe {
                draft
                    .taps
                    .copy_from(&e.stream, 0, storage, 0, rows * WIDTH)?;
            }
            draft.add_taps(e, rows)
        })
    }

    /// `drain` is the existing whole-engine native hold boundary; do not add a
    /// separate stream/collective fence for each slot at the serving callsite.
    pub(in crate::model) fn pause(
        &mut self,
        lease: Lease,
        drain: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.state.pause(lease, drain)
    }

    pub(in crate::model) fn resume(&mut self, lease: Lease) -> Result<()> {
        self.state.resume(lease)
    }

    pub(in crate::model) fn replace(
        &mut self,
        lease: Lease,
        cursor: usize,
        replace: impl FnOnce() -> Result<()>,
    ) -> Result<Lease> {
        self.state.replace(lease, cursor, replace)
    }

    pub(in crate::model) fn snapshot_cursor(&self, lease: Lease) -> Result<usize> {
        self.state.snapshot_cursor(lease)
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        // A failed round may have enqueued a copy before returning an error.
        // Fence all references before the uniquely owned Buffer is released.
        let _ = self.device.sync();
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Reference {
    end: usize,
    state: Vec<String>,
    tokens: Vec<i32>,
    logits: String,
    projected: String,
}

fn feature_bytes(slot: usize, position: usize, rows: usize) -> Vec<u8> {
    (0..rows * (WIDTH / 2))
        .flat_map(|i| {
            half::bf16::from_f32(
                (((position + i / (WIDTH / 2)) * 13 + i % (WIDTH / 2) * 7 + slot * 17) % 127)
                    as f32
                    / 64.
                    - 1.,
            )
            .to_bits()
            .to_le_bytes()
        })
        .collect()
}

fn capture(e: &Engine<'_>, draft: &mut Drafter, slot: usize) -> Result<Reference> {
    use sha2::{Digest, Sha256};
    let hash =
        |b: &Buffer| -> Result<String> { Ok(format!("{:x}", Sha256::digest(b.read(0, b.bytes)?))) };
    let tokens = draft.propose(
        e,
        198 + slot as i32,
        crate::sampling::Sampling::new(731, 1., 20, 0.95)?,
    )?;
    Ok(Reference {
        end: draft.end(),
        state: draft
            .context()
            .kc
            .iter()
            .chain(&draft.context().vc)
            .chain([&draft.context().pos, &draft.context().lo])
            .map(hash)
            .collect::<Result<_>>()?,
        tokens,
        logits: hash(&draft.logits)?,
        projected: hash(&draft.projected)?,
    })
}

/// Synthetic accepted target features isolate lifetime/order/batching from the
/// verifier. Compare real GPU K/V and graph proposals with immediate flushing.
pub(super) fn probe(e: &Engine<'_>, draft: &mut Drafter) -> Result<serde_json::Value> {
    let bases = [128usize, 2050, 4090, 1048320];
    // Two exactly-full backlogs; partial accepts must exclude poisoned tails.
    let accepted = [16usize, 4, 16, 1, 16, 11, 16, 7, 16, 9, 16];
    // Declare capture storage before the retained owners so their Drop fences
    // run before this source is released, including a partial-copy error path.
    let source = e.device.alloc(17 * WIDTH)?;
    let mut owners = bases
        .iter()
        .enumerate()
        .map(|(slot, &base)| Retained::new(&e.device, slot, base))
        .collect::<Result<Vec<_>>>()?;
    // One leading guard row also exercises the serial ring's nonzero source
    // offset. Every unaccepted row is finite poison, never a valid next feature.
    let fill = |slot: usize, position: usize, rows: usize| -> Result<()> {
        let mut bytes = vec![0x3f; source.bytes];
        bytes[WIDTH..(rows + 1) * WIDTH].copy_from_slice(&feature_bytes(slot, position, rows));
        e.sync()?;
        source.write(0, &bytes)
    };
    let mut reference: Vec<Vec<Reference>> = Vec::new();
    for slot in 0..4 {
        draft.bind(e, slot)?;
        draft.start_context(e, bases[slot])?;
        let owner = &mut owners[slot];
        let lease = owner.lease();
        let mut saved = Vec::new();
        for &rows in &accepted {
            let position = owner.status().end;
            fill(slot, position, rows)?;
            owner.append(e, lease, position, &source, 1, rows)?;
            ensure!(
                owner.flush(e, draft, lease)? == rows,
                "immediate feature flush count"
            );
            e.sync()?;
            ensure!(
                draft.taps.read(0, rows * WIDTH)? == feature_bytes(slot, position, rows),
                "immediate accepted feature bytes"
            );
            if [bases[slot] + 64, bases[slot] + 128].contains(&draft.end()) {
                saved.push(capture(e, draft, slot)?);
            }
        }
        ensure!(saved.len() == 2, "backlog reference checkpoints");
        reference.push(saved);
    }

    for slot in 0..4 {
        draft.bind(e, slot)?;
        let owner = &mut owners[slot];
        owner.replace(owner.lease(), bases[slot], || {
            draft.start_context(e, bases[slot])
        })?;
    }
    let mut pauses = 0;
    let mut flushes = 0;
    for (round, &rows) in accepted.iter().enumerate() {
        for slot in [2usize, 0, 3, 1] {
            draft.bind(e, slot)?;
            let owner = &mut owners[slot];
            let lease = owner.lease();
            if round != 0 {
                owner.resume(lease)?;
            }
            if owner.available(lease)? < rows {
                let start = owner.status().start;
                ensure!(
                    owner.flush(e, draft, lease)? == 64,
                    "bounded backlog flush count"
                );
                e.sync()?;
                ensure!(
                    draft.taps.read(0, 64 * WIDTH)? == feature_bytes(slot, start, 64),
                    "deferred accepted feature bytes"
                );
                ensure!(
                    capture(e, draft, slot)? == reference[slot][0],
                    "first deferred K/V or graph proposal changed"
                );
                flushes += 1;
            }
            let position = owner.status().end;
            fill(slot, position, rows)?;
            owner.append(e, lease, position, &source, 1, rows)?;
            ensure!(
                owner.snapshot_cursor(lease).is_err(),
                "nonempty backlog snapshot accepted"
            );
            owner.pause(lease, || e.sync())?;
            let held = owner.status();
            ensure!(held.end > held.start, "hold did not retain queued features");
            ensure!(
                owner.flush(e, draft, lease).is_err() && owner.status() == held,
                "paused backlog changed or executed"
            );
            // The shared capture owner is overwritten before any next visit.
            // Retained data must already live in the slot's own device Buffer.
            source.write(0, &vec![0x40; source.bytes])?;
            pauses += 1;
        }
    }
    let mut cases = Vec::new();
    for slot in [3usize, 1, 0, 2] {
        draft.bind(e, slot)?;
        let owner = &mut owners[slot];
        let lease = owner.lease();
        owner.resume(lease)?;
        let start = owner.status().start;
        ensure!(
            owner.flush(e, draft, lease)? == 64,
            "final backlog flush count"
        );
        e.sync()?;
        ensure!(
            draft.taps.read(0, 64 * WIDTH)? == feature_bytes(slot, start, 64),
            "final accepted feature bytes"
        );
        ensure!(
            capture(e, draft, slot)? == reference[slot][1],
            "final deferred K/V or graph proposal changed"
        );
        ensure!(
            owner.snapshot_cursor(lease)? == draft.end(),
            "backlog snapshot cursor"
        );
        flushes += 1;
        cases.push(
            serde_json::json!({"slot":slot,"base":bases[slot],"end":draft.end(),
            "accepted_rows":128,"deferred_flush_rows":[64,64],
            "context_state_sha256":&reference[slot][1].state,
            "logits_sha256":&reference[slot][1].logits,
            "selector_sha256":&reference[slot][1].projected,
            "sampled_proposal":&reference[slot][1].tokens,
            "exact_accepted_features_kv_cursor_logits_and_graph_proposal":true}),
        );
    }
    for slot in [0usize, 2, 1, 3, 2, 0, 3, 1] {
        draft.bind(e, slot)?;
        ensure!(
            capture(e, draft, slot)? == reference[slot][1],
            "retained backlog graph replay changed state or proposal"
        );
    }
    Ok(
        serde_json::json!({"cases":cases,"retained_slots":4,"bytes_per_slot":STORAGE_BYTES,
        "total_backlog_bytes":STORAGE_BYTES * 4,"nonempty_pause_resume_pairs":pauses,
        "deferred_flushes":flushes,"retained_graph_replays":8,
        "per_append_flush_collectives":0,"per_append_flush_synchronizations":0,
        "scope":"Actual GPU-owned accepted-feature backlogs with synthetic features; immediate-vs-deferred K/V and graph proposals. Not request serving or snapshot format integration."}),
    )
}
