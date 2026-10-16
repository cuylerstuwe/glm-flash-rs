//! Owner-clock accounting for a cooperative job. Native holds suspend the
//! progress deadline; request deadlines remain independent wall-clock limits.
use anyhow::{Result, ensure};
use std::time::Duration;
#[derive(Default)]
pub(crate) struct Watch {
    last: Option<Duration>,
    allowed: bool,
    progress: (usize, usize),
    idle: Duration,
    cancelling: Option<Duration>,
}
impl Watch {
    pub fn observe(
        &mut self,
        now: Duration,
        allowed: bool,
        progress: (usize, usize),
    ) -> Result<()> {
        if let Some(last) = self.last {
            ensure!(now >= last, "cache watchdog clock went backwards");
            if allowed && self.allowed {
                let elapsed = now - last;
                self.idle = self.idle.saturating_add(elapsed);
                if let Some(age) = &mut self.cancelling {
                    *age = age.saturating_add(elapsed);
                }
            }
        }
        if progress != self.progress {
            self.idle = Duration::ZERO;
            self.progress = progress;
        }
        self.last = Some(now);
        self.allowed = allowed;
        Ok(())
    }
    pub fn stalled(&self, limit: Duration) -> bool {
        self.cancelling.is_none() && self.idle >= limit
    }
    pub fn cancel(&mut self) {
        self.cancelling.get_or_insert(Duration::ZERO);
    }
    pub fn expired(&self, grace: Duration) -> bool {
        self.cancelling.is_some_and(|age| age >= grace)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(n: u64) -> Duration {
        Duration::from_secs(n)
    }
    #[test]
    fn progress_and_holds_bound_cancellation_without_replacing_workers() -> Result<()> {
        let mut w = Watch::default();
        w.observe(t(0), true, (0, 0))?;
        w.observe(t(9), true, (0, 0))?;
        assert!(!w.stalled(t(10)));
        w.observe(t(10), true, (8, 1))?;
        assert!(!w.stalled(t(10)));
        w.observe(t(11), false, (8, 1))?;
        w.observe(t(500), false, (8, 1))?;
        w.observe(t(501), true, (8, 1))?;
        assert!(!w.stalled(t(10)));
        w.observe(t(511), true, (8, 1))?;
        assert!(w.stalled(t(10)));
        w.cancel();
        assert!(!w.expired(t(3)));
        w.observe(t(512), true, (9, 2))?;
        w.cancel(); // Repeated cancellation cannot reset the grace deadline.
        w.observe(t(514), true, (9, 2))?;
        assert!(w.expired(t(3)));
        assert!(!w.stalled(t(10)));
        assert!(w.observe(t(513), true, (9, 2)).is_err());
        Ok(())
    }
    #[test]
    fn parked_job_excludes_a_long_gap_without_intermediate_observations() -> Result<()> {
        let mut w = Watch::default();
        w.observe(t(0), true, (8, 1))?;
        w.observe(t(5), true, (8, 1))?;
        w.cancel();
        w.observe(t(5), false, (8, 1))?;
        w.observe(t(500), true, (8, 1))?;
        assert!(!w.expired(t(30)));
        w.observe(t(529), true, (8, 1))?;
        assert!(!w.expired(t(30)));
        w.observe(t(530), true, (8, 1))?;
        assert!(w.expired(t(30)));
        Ok(())
    }
}
