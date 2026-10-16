//! Bounded, retained accepted-feature ownership for one model slot.
//!
//! The backing store is owned, not borrowed from shared capture scratch. GPU
//! adapters provide exactly STORAGE_BYTES and implement the operations below
//! with ordered device copies/updates at accepted-prefix commit boundaries.
use anyhow::{Result, ensure};

pub const ROWS: usize = 64;
pub const WIDTH: usize = 5 * 4096 * 2;
pub const STORAGE_BYTES: usize = ROWS * WIDTH;
const COMMIT_ROWS: usize = 16;
const CONTEXT: usize = 1 << 20;
const SLOTS: usize = 4;

/// Implement on the uniquely owned GPU allocation when registering this module.
pub trait Storage {
    fn bytes(&self) -> usize;
}

/// Identifies retained slot state, not a request. A live follow-up may retain
/// this lease; replacing/restoring its model state must change the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lease {
    slot: usize,
    epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Paused,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    pub slot: usize,
    pub epoch: u64,
    /// Drafter cursor; rows before it have already been consumed.
    pub start: usize,
    /// Target cursor; [start, end) is retained in backing rows [0, len).
    pub end: usize,
    pub phase: Phase,
}

pub struct Backlog<S> {
    storage: S,
    state: Status,
}

impl<S: Storage> Backlog<S> {
    /// The caller has already established the drafter/target at `cursor`.
    pub fn new(slot: usize, cursor: usize, storage: S) -> Result<Self> {
        ensure!(
            slot < SLOTS && cursor <= CONTEXT,
            "accepted-feature owner bounds"
        );
        ensure!(
            storage.bytes() == STORAGE_BYTES,
            "accepted-feature allocation bound"
        );
        Ok(Self {
            storage,
            state: Status {
                slot,
                epoch: 0,
                start: cursor,
                end: cursor,
                phase: Phase::Ready,
            },
        })
    }

    pub fn lease(&self) -> Lease {
        Lease {
            slot: self.state.slot,
            epoch: self.state.epoch,
        }
    }

    pub fn status(&self) -> Status {
        self.state
    }

    fn owned(&self, lease: Lease) -> Result<()> {
        ensure!(lease == self.lease(), "stale accepted-feature owner");
        ensure!(
            self.state.phase != Phase::Poisoned,
            "failed accepted-feature owner"
        );
        Ok(())
    }

    fn ready(&self, lease: Lease) -> Result<()> {
        self.owned(lease)?;
        ensure!(
            self.state.phase == Phase::Ready,
            "accepted-feature owner is paused"
        );
        Ok(())
    }

    /// Check before launching a verifier round. Flush if the maximum possible
    /// accepted prefix would exceed this capacity; never drop accepted rows.
    pub fn available(&self, lease: Lease) -> Result<usize> {
        self.ready(lease)?;
        Ok(ROWS - (self.state.end - self.state.start))
    }

    /// Copy ONLY the commonly accepted/clipped prefix into the owned allocation.
    /// `copy` receives destination row offset and count. For a GPU adapter it
    /// may enqueue stream-ordered work; the caller must keep the containing
    /// native owner unhealthy until its existing drain/success agreement. These
    /// local cursors are provisional until that boundary, not an external
    /// publication. Do not add a separate collective for each append.
    ///
    /// Validation failures do not call `copy` or mutate anything. Once `copy`
    /// starts, any error (or panic) poisons this owner because device side effects
    /// may be partial. The containing Long must also remain unhealthy on failure.
    pub fn append(
        &mut self,
        lease: Lease,
        position: usize,
        rows: usize,
        copy: impl FnOnce(&mut S, usize, usize) -> Result<()>,
    ) -> Result<()> {
        let available = self.available(lease)?;
        ensure!(position == self.state.end, "accepted-feature cursor gap");
        ensure!(
            (1..=COMMIT_ROWS).contains(&rows) && rows <= available,
            "accepted-feature append bound"
        );
        ensure!(rows <= CONTEXT - position, "accepted-feature context bound");
        let offset = self.state.end - self.state.start;
        self.state.phase = Phase::Poisoned;
        copy(&mut self.storage, offset, rows)?;
        self.state.end += rows;
        self.state.phase = Phase::Ready;
        Ok(())
    }

    /// Consume every queued row in one contiguous drafter update (at most 64).
    /// The caller checks the actual retained Drafter::end(), not a cached guess.
    /// `consume` may enqueue work on the same stream; external publication still
    /// requires the containing owner's existing drain/all-rank success boundary.
    pub fn flush(
        &mut self,
        lease: Lease,
        drafter_end: usize,
        consume: impl FnOnce(&S, usize, usize) -> Result<()>,
    ) -> Result<usize> {
        self.ready(lease)?;
        ensure!(
            drafter_end == self.state.start,
            "accepted-feature drafter cursor mismatch"
        );
        let rows = self.state.end - self.state.start;
        if rows == 0 {
            return Ok(0);
        }
        self.state.phase = Phase::Poisoned;
        consume(&self.storage, self.state.start, rows)?;
        self.state.start = self.state.end;
        self.state.phase = Phase::Ready;
        Ok(rows)
    }

    /// Retain the backlog exactly; a hold does not force drafter catch-up.
    /// Drain every engine/copy stream and coordinate success in `drain` before
    /// the native owner acknowledges the hold. Shared capture scratch may then
    /// be reused by another slot without affecting this allocation.
    pub fn pause(&mut self, lease: Lease, drain: impl FnOnce() -> Result<()>) -> Result<()> {
        self.ready(lease)?;
        self.state.phase = Phase::Poisoned;
        drain()?;
        self.state.phase = Phase::Paused;
        Ok(())
    }

    pub fn resume(&mut self, lease: Lease) -> Result<()> {
        self.owned(lease)?;
        ensure!(
            self.state.phase == Phase::Paused,
            "accepted-feature owner was not paused"
        );
        self.state.phase = Phase::Ready;
        Ok(())
    }

    /// Replace state only as part of a coordinated target/drafter reset/restore
    /// (or completed prefill update). `replace` drains old references and proves
    /// both cursors now equal `cursor`; it must preserve the fixed allocation.
    /// Retained live continuation uses the existing lease instead. Replacement
    /// preserves a native hold; only an explicit resume makes it runnable.
    pub fn replace(
        &mut self,
        lease: Lease,
        cursor: usize,
        replace: impl FnOnce() -> Result<()>,
    ) -> Result<Lease> {
        self.owned(lease)?;
        ensure!(
            cursor <= CONTEXT,
            "accepted-feature replacement cursor bound"
        );
        let epoch = self
            .state
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("accepted-feature epoch overflow"))?;
        let phase = self.state.phase;
        self.state.phase = Phase::Poisoned;
        replace()?;
        self.state.epoch = epoch;
        self.state.start = cursor;
        self.state.end = cursor;
        self.state.phase = phase;
        Ok(self.lease())
    }

    /// Pure-prefix snapshots must first consume the entire backlog and include
    /// the retained drafter K/V at this cursor. This is not a snapshot by itself.
    pub fn snapshot_cursor(&self, lease: Lease) -> Result<usize> {
        self.ready(lease)?;
        ensure!(
            self.state.start == self.state.end,
            "accepted features must be flushed before snapshot"
        );
        Ok(self.state.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Memory([u32; ROWS]);
    impl Storage for Memory {
        fn bytes(&self) -> usize {
            STORAGE_BYTES
        }
    }
    fn owner(slot: usize, cursor: usize) -> Backlog<Memory> {
        Backlog::new(slot, cursor, Memory([0; ROWS])).unwrap()
    }
    fn append(b: &mut Backlog<Memory>, n: usize) -> Result<()> {
        let start = b.status().end;
        b.append(b.lease(), start, n, |s, off, rows| {
            for i in 0..rows {
                s.0[off + i] = (start + i) as u32;
            }
            Ok(())
        })
    }
    fn flush(b: &mut Backlog<Memory>) -> Result<Vec<u32>> {
        let mut values = Vec::new();
        b.flush(b.lease(), b.status().start, |s, start, rows| {
            values.extend_from_slice(&s.0[..rows]);
            assert_eq!(
                values,
                (start as u32..(start + rows) as u32).collect::<Vec<_>>()
            );
            Ok(())
        })?;
        Ok(values)
    }

    #[test]
    fn bounded_batches_equal_serial_and_reuse_without_compaction() -> Result<()> {
        let (mut serial, mut batched) = (owner(0, 2050), owner(1, 2050));
        for _ in 0..64 {
            append(&mut serial, 1)?;
        }
        for _ in 0..4 {
            append(&mut batched, 16)?;
        }
        assert_eq!(serial.available(serial.lease())?, 0);
        assert!(append(&mut serial, 1).is_err());
        assert_eq!(flush(&mut serial)?, flush(&mut batched)?);
        assert_eq!(serial.snapshot_cursor(serial.lease())?, 2114);
        append(&mut serial, 7)?;
        assert_eq!(flush(&mut serial)?, (2114..2121).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn four_paused_slots_preserve_independent_rows_and_cursors() -> Result<()> {
        let mut slots = (0..4).map(|i| owner(i, 8190 + i * 100)).collect::<Vec<_>>();
        for (i, slot) in slots.iter_mut().enumerate() {
            append(slot, 4 + i)?;
            slot.pause(slot.lease(), || Ok(()))?;
        }
        for slot in slots.iter_mut().rev() {
            let before = slot.status();
            assert!(append(slot, 1).is_err());
            assert!(flush(slot).is_err());
            assert!(slot.snapshot_cursor(slot.lease()).is_err());
            assert_eq!(slot.status(), before);
            slot.resume(slot.lease())?;
            append(slot, 16)?;
            flush(slot)?;
        }
        Ok(())
    }

    #[test]
    fn invalid_operations_never_mutate_or_invoke_callbacks() -> Result<()> {
        let mut b = owner(0, CONTEXT - 8);
        let other = owner(1, CONTEXT - 8).lease();
        let lease = b.lease();
        let original = b.status();
        for (owner, pos, n) in [
            (other, CONTEXT - 8, 1),
            (lease, 0, 1),
            (lease, CONTEXT - 8, 0),
            (lease, CONTEXT - 8, 17),
            (lease, CONTEXT - 8, 9),
        ] {
            assert!(
                b.append(owner, pos, n, |_, _, _| panic!("invalid copy called"))
                    .is_err()
            );
            assert_eq!(b.status(), original);
        }
        assert!(
            b.flush(lease, 0, |_, _, _| panic!("invalid flush called"))
                .is_err()
        );
        assert!(
            b.replace(lease, CONTEXT + 1, || panic!("invalid reset called"))
                .is_err()
        );
        assert!(b.resume(lease).is_err());
        assert_eq!(b.status(), original);
        append(&mut b, 8)?;
        assert_eq!(flush(&mut b)?.len(), 8);
        assert_eq!(b.snapshot_cursor(lease)?, CONTEXT);
        assert!(append(&mut b, 1).is_err());
        Ok(())
    }

    #[test]
    fn replacement_invalidates_stale_leases_but_live_followup_keeps_rows() -> Result<()> {
        let mut b = owner(0, 128);
        let old = b.lease();
        append(&mut b, 16)?;
        // A different request using the exact live continuation retains this
        // slot epoch. Request identity is checked by the surrounding queue.
        append(&mut b, 1)?;
        flush(&mut b)?;
        b.pause(old, || Ok(()))?;
        let new = b.replace(old, 8192, || Ok(()))?;
        assert_ne!(old, new);
        assert!(
            b.append(old, 8192, 1, |_, _, _| panic!("stale copy called"))
                .is_err()
        );
        assert!(b.pause(old, || panic!("stale drain called")).is_err());
        assert_eq!(b.status().phase, Phase::Paused);
        assert!(append(&mut b, 1).is_err());
        b.resume(new)?;
        append(&mut b, 16)?;
        assert_eq!(flush(&mut b)?.len(), 16);
        Ok(())
    }

    #[test]
    fn operation_failures_poison_owner_without_publishing_new_cursor() -> Result<()> {
        for operation in 0..4 {
            let mut b = owner(0, 128);
            append(&mut b, 4)?;
            let before = b.status();
            let lease = b.lease();
            let fail = || anyhow::bail!("coordinated device fault");
            let result = match operation {
                0 => b.append(lease, 132, 4, |s, off, _| {
                    s.0[off] = 999;
                    fail()
                }),
                1 => b.flush(lease, 128, |_, _, _| fail()).map(|_| ()),
                2 => b.pause(lease, fail),
                _ => b.replace(lease, 8192, fail).map(|_| ()),
            };
            assert!(result.is_err());
            assert_eq!(
                b.status(),
                Status {
                    phase: Phase::Poisoned,
                    ..before
                }
            );
            assert!(b.available(lease).is_err());
            assert!(b.resume(lease).is_err());
            assert!(
                b.replace(lease, 0, || panic!("poison reset called"))
                    .is_err()
            );
            assert!(b.snapshot_cursor(lease).is_err());
        }
        Ok(())
    }

    #[test]
    fn snapshot_and_empty_flush_require_exact_drafter_cursor() -> Result<()> {
        let mut b = owner(0, 2051);
        let lease = b.lease();
        assert_eq!(
            b.flush(lease, 2051, |_, _, _| panic!("empty flush called"))?,
            0
        );
        append(&mut b, 1)?;
        assert!(b.snapshot_cursor(lease).is_err());
        assert!(
            b.flush(lease, 2052, |_, _, _| panic!("wrong cursor called"))
                .is_err()
        );
        flush(&mut b)?;
        assert_eq!(b.snapshot_cursor(lease)?, 2052);
        Ok(())
    }

    #[test]
    fn allocation_and_epoch_limits_are_checked_before_side_effects() -> Result<()> {
        struct Sized(usize);
        impl Storage for Sized {
            fn bytes(&self) -> usize {
                self.0
            }
        }
        for bytes in [0, STORAGE_BYTES - 1, STORAGE_BYTES + 1, usize::MAX] {
            assert!(Backlog::new(0, 0, Sized(bytes)).is_err());
        }
        assert!(Backlog::new(4, 0, Sized(STORAGE_BYTES)).is_err());
        assert!(Backlog::new(0, CONTEXT + 1, Sized(STORAGE_BYTES)).is_err());
        let mut b = owner(0, 128);
        b.state.epoch = u64::MAX;
        let before = b.status();
        assert!(
            b.replace(b.lease(), 0, || panic!("overflow reset called"))
                .is_err()
        );
        assert_eq!(b.status(), before);
        append(&mut b, 1)?;
        Ok(())
    }
}
