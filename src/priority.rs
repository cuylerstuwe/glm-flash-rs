//! Bounded, monotonic native lease state. Only the CUDA owner acknowledges a
//! captured revision after an all-rank drain; receipt of a control update is not
//! acknowledgement. Expired/released owners remain epoch tombstones.
use anyhow::{Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};
const OWNERS: usize = 1024;
const LOCALS: usize = 1024;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Snapshot {
    pub revision: u64,
    pub hold: bool,
}
struct Lease {
    epoch: u64,
    foreground: bool,
    expires: Duration,
}
#[derive(Default)]
pub(crate) struct Gate {
    leases: BTreeMap<String, Lease>,
    locals: BTreeSet<u64>,
    serial: u64,
    revision: u64,
    applied: u64,
    last: Duration,
}
impl Gate {
    fn advance(&mut self, now: Duration) -> Result<()> {
        ensure!(now >= self.last, "priority monotonic clock went backwards");
        self.last = now;
        Ok(())
    }
    pub fn update(
        &mut self,
        owner: &str,
        epoch: u64,
        foreground: bool,
        seconds: f64,
        now: Duration,
    ) -> Result<u64> {
        ensure!(
            !owner.is_empty() && owner.chars().count() <= 160,
            "invalid lease owner"
        );
        ensure!(
            seconds.is_finite() && (1.0..=300.0).contains(&seconds),
            "lease TTL must be 1..300 seconds"
        );
        if let Some(old) = self.leases.get(owner) {
            ensure!(epoch >= old.epoch, "stale lease epoch");
            ensure!(
                epoch != old.epoch || foreground == old.foreground,
                "conflicting lease epoch"
            );
        } else {
            ensure!(self.leases.len() < OWNERS, "lease owner capacity");
        }
        let expires = now
            .checked_add(Duration::from_secs_f64(seconds))
            .ok_or_else(|| anyhow::anyhow!("lease clock overflow"))?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("priority revision overflow"))?;
        self.advance(now)?;
        self.leases.insert(
            owner.to_owned(),
            Lease {
                epoch,
                foreground,
                expires,
            },
        );
        self.revision = revision;
        Ok(revision)
    }
    /// A ticket owns one local foreground admission/preparation lifetime.
    pub fn enter(&mut self, now: Duration) -> Result<u64> {
        ensure!(self.locals.len() < LOCALS, "foreground ticket capacity");
        let ticket = self
            .serial
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("foreground ticket overflow"))?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("priority revision overflow"))?;
        self.advance(now)?;
        self.locals.insert(ticket);
        self.serial = ticket;
        self.revision = revision;
        Ok(ticket)
    }
    pub fn leave(&mut self, ticket: u64, now: Duration) -> Result<()> {
        ensure!(
            self.locals.contains(&ticket),
            "unknown/released foreground ticket"
        );
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("priority revision overflow"))?;
        self.advance(now)?;
        self.locals.remove(&ticket);
        self.revision = revision;
        Ok(())
    }
    pub fn snapshot(&mut self, now: Duration, queued_or_live_foreground: bool) -> Result<Snapshot> {
        self.advance(now)?;
        Ok(Snapshot {
            revision: self.revision,
            hold: queued_or_live_foreground
                || !self.locals.is_empty()
                || self
                    .leases
                    .values()
                    .any(|v| v.foreground && v.expires > now),
        })
    }
    /// Called ONLY with a revision returned by a completed native boundary.
    /// A newer update arriving during drain remains unacknowledged.
    pub fn acknowledge(&mut self, drained_revision: u64) -> Result<()> {
        ensure!(
            drained_revision >= self.applied && drained_revision <= self.revision,
            "invalid drained revision"
        );
        self.applied = drained_revision;
        Ok(())
    }
    /// Include lease identities and ticket ownership, not just aggregate hold.
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"native-gate-v1");
        for n in [
            self.serial,
            self.revision,
            self.applied,
            self.locals.len() as u64,
            self.leases.len() as u64,
        ] {
            h.update(n.to_le_bytes());
        }
        for t in &self.locals {
            h.update(t.to_le_bytes());
        }
        for (name, l) in &self.leases {
            h.update((name.len() as u64).to_le_bytes());
            h.update(name.as_bytes());
            h.update(l.epoch.to_le_bytes());
            h.update([u8::from(l.foreground)]);
            h.update(l.expires.as_nanos().to_le_bytes());
        }
        h.finalize().into()
    }
    pub fn local_count(&self) -> usize {
        self.locals.len()
    }
    pub fn applied(&self) -> u64 {
        self.applied
    }
    pub fn wait(&mut self, now: Duration) -> Result<Duration> {
        self.advance(now)?;
        Ok(self
            .leases
            .values()
            .filter(|v| v.foreground && v.expires > now)
            .map(|v| v.expires - now)
            .min()
            .unwrap_or(Duration::from_secs(1))
            .min(Duration::from_secs(1)))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(n: u64) -> Duration {
        Duration::from_millis(n)
    }
    #[test]
    fn lease_epochs_expiry_and_tombstones() -> Result<()> {
        let mut g = Gate::default();
        let r = g.update("router", 9, true, 1.0, t(0))?;
        assert!(g.snapshot(t(999), false)?.hold);
        assert_eq!(g.wait(t(999))?, t(1));
        assert!(!g.snapshot(t(1000), false)?.hold);
        assert!(g.update("router", 8, false, 1.0, t(1000)).is_err());
        assert!(g.update("router", 9, false, 1.0, t(1000)).is_err());
        g.update("router", 9, true, 1.0, t(1000))?;
        assert_eq!(g.applied(), 0);
        g.acknowledge(r)?;
        assert_eq!(g.applied(), r);
        assert!(g.applied() < g.snapshot(t(1000), false)?.revision);
        g.update("router", 10, false, 1.0, t(1000))?;
        assert!(!g.snapshot(t(1000), false)?.hold);
        Ok(())
    }
    #[test]
    fn independent_owners_local_tickets_and_queue_compose() -> Result<()> {
        let mut g = Gate::default();
        g.update("a", 0, true, 1.0, t(0))?;
        g.update("b", 0, true, 2.0, t(0))?;
        let a = g.enter(t(0))?;
        let b = g.enter(t(0))?;
        g.leave(a, t(1000))?;
        assert!(g.leave(a, t(1000)).is_err());
        assert!(g.snapshot(t(2000), false)?.hold);
        g.leave(b, t(2000))?;
        assert!(!g.snapshot(t(2000), false)?.hold);
        assert!(g.snapshot(t(2000), true)?.hold);
        assert!(g.snapshot(t(1999), false).is_err());
        Ok(())
    }
    #[test]
    fn limits_validation_and_ack_fail_closed() -> Result<()> {
        let mut g = Gate::default();
        for x in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.999, 300.001] {
            assert!(g.update("a", 0, true, x, t(0)).is_err());
        }
        assert!(g.update("", 0, true, 1.0, t(0)).is_err());
        assert!(g.update(&"🦀".repeat(161), 0, true, 1.0, t(0)).is_err());
        g.update(&"🦀".repeat(160), 0, true, 300.0, t(0))?;
        for n in 1..OWNERS {
            g.update(&format!("{n}"), 0, false, 1.0, t(0))?;
        }
        assert!(g.update("extra", 0, false, 1.0, t(400000)).is_err());
        g.update("1", 1, true, 1.0, t(400000))?;
        assert_eq!(g.leases.len(), OWNERS);
        assert!(g.acknowledge(u64::MAX).is_err());
        g.acknowledge(1)?;
        assert!(g.acknowledge(0).is_err());
        for _ in 0..LOCALS {
            g.enter(t(400000))?;
        }
        assert!(g.enter(t(400000)).is_err());
        Ok(())
    }
}
