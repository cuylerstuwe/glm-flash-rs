//! Rank-zero priority decisions, all-rank device drain before acknowledgement.
//! The HTTP scheduler applies authenticated leases through this native boundary.
use super::{Long, store::Control};
use crate::{
    cache_pages::{Phase, SLOTS},
    priority::Snapshot,
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
pub(super) struct Boundary {
    control: Control,
    tick: u64,
    revision: u64,
    owned: u8,
}
impl Boundary {
    pub fn new(long: &Long<'_>) -> Result<Self> {
        Ok(Self {
            control: Control::new(&long.engine.device)?,
            tick: 0,
            revision: 0,
            owned: 0,
        })
    }
    pub fn apply(&mut self, long: &mut Long<'_>, head: Option<Snapshot>) -> Result<u64> {
        long.ready()?;
        long.healthy = false;
        let result = self.apply_inner(long, head);
        if result.is_ok() {
            long.healthy = true;
        }
        result
    }
    fn apply_inner(&mut self, long: &mut Long<'_>, head: Option<Snapshot>) -> Result<u64> {
        let mut record = [0u8; 64];
        record[..8].copy_from_slice(b"ALPRIO01");
        record[8..16].copy_from_slice(&self.tick.to_le_bytes());
        let valid = (long.engine.rank == 0) == head.is_some();
        record[16] = u8::from(valid);
        if let Some(head) = head {
            record[17..25].copy_from_slice(&head.revision.to_le_bytes());
            record[25] = u8::from(
                head.hold
                    || long.sessions.iter().enumerate().any(|(i, s)| {
                        !s.background
                            && long
                                .memory
                                .book()
                                .slot(i)
                                .is_ok_and(|s| s.phase() == Phase::Active)
                    }),
            );
        }
        let all = self.control.exchange(long, &record)?;
        ensure!(
            all.iter().all(|r| r[..17] == record[..17]) && valid,
            "priority control sequence/leader disagreement"
        );
        let revision = u64::from_le_bytes(all[0][17..25].try_into().unwrap());
        ensure!(
            revision >= self.revision && all[0][25] <= 1,
            "priority revision/hold invalid"
        );
        let hold = all[0][25] == 1;
        let mut digest = Sha256::new();
        digest.update(b"priority-progress-v2");
        let speculation = long.speculation_digest();
        digest.update(speculation.as_ref().copied().unwrap_or([0; 32]));
        digest.update([self.owned, u8::from(long.priority_hold)]);
        let mut owned = 0u8;
        for slot in 0..SLOTS {
            let session = &long.sessions[slot];
            let phase = long.memory.book().slot(slot)?.phase();
            digest.update([
                match phase {
                    Phase::Idle => 0,
                    Phase::Active => 1,
                    Phase::Paused => 2,
                    Phase::Restoring => 3,
                },
                u8::from(session.background),
                u8::from(session.mixed),
            ]);
            for n in [session.tokens.len(), session.prompt_end, session.total_end] {
                digest.update((n as u64).to_le_bytes());
            }
            digest.update(session.pending.unwrap_or(-1).to_le_bytes());
            if hold
                && session.background
                && (phase == Phase::Active
                    || (phase == Phase::Paused && self.owned & (1 << slot) != 0))
            {
                owned |= 1 << slot;
            }
        }
        let hash: [u8; 32] = digest.finalize().into();
        ensure!(
            self.control.agree(long, 700, hash, speculation.is_ok())?,
            "priority state disagreement"
        );
        speculation?;
        // All streams must be drained for a hold, including future preparation
        // and transfer streams. Release need only drain the model stream.
        let drained = if hold {
            if let Some(gate) = long.copy_gate.as_ref().and_then(std::sync::Weak::upgrade) {
                gate.hold(true);
            }
            long.engine.device.sync().and_then(|()| {
                ensure!(
                    !long
                        .copy_gate
                        .as_ref()
                        .and_then(std::sync::Weak::upgrade)
                        .is_some_and(|g| g.fatal.load(std::sync::atomic::Ordering::Acquire)),
                    "SSD copy stream failed before native hold"
                );
                Ok(())
            })
        } else {
            long.engine.sync()
        };
        ensure!(
            self.control.agree(long, 701, hash, drained.is_ok())?,
            "priority device drain failed"
        );
        drained?;
        let installed = (|| -> Result<()> {
            long.priority_hold = hold;
            for slot in 0..SLOTS {
                let phase = long.memory.book().slot(slot)?.phase();
                if owned & (1 << slot) != 0 && phase == Phase::Active {
                    long.memory.pause(&long.engine.stream, slot)?;
                } else if !hold
                    && self.owned & (1 << slot) != 0
                    && phase == Phase::Paused
                    && long.sessions[slot].background
                {
                    long.memory.resume(slot)?;
                }
            }
            long.sync_speculation_holds()
        })();
        // A proof containing the exact common decision is exchanged after every
        // rank has installed its mask. Never acknowledge merely receiving it.
        record[16] = u8::from(installed.is_ok());
        record[17..25].copy_from_slice(&revision.to_le_bytes());
        record[25] = u8::from(hold);
        record[26] = owned;
        ensure!(
            self.control
                .exchange(long, &record)?
                .iter()
                .all(|r| r == &record && r[16] == 1),
            "priority applied proof disagreement"
        );
        installed?;
        self.owned = owned;
        self.tick = self
            .tick
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("priority boundary overflow"))?;
        self.revision = revision;
        Ok(revision)
    }
}
#[path = "native_priority_tests.rs"]
mod tests;
pub use tests::run;
