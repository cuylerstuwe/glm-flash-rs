//! CPU image work retained under one native preparation ticket.
//! The native owner supplies hold, cancellation and deadline decisions.
use anyhow::{Context, Result, ensure};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Accounting {
    used: usize,
    failed: bool,
}
pub(crate) struct Budget {
    cap: usize,
    state: Mutex<Accounting>,
}
impl Budget {
    pub(crate) fn new(cap: usize) -> Result<Arc<Self>> {
        ensure!(
            cap > 0 && cap <= 4 * 1024 * 1024 * 1024,
            "image owner budget"
        );
        Ok(Arc::new(Self {
            cap,
            state: Mutex::new(Accounting::default()),
        }))
    }
    /// Reserve before extracting encoded bytes. Queued, executing and ready
    /// images keep the same reservation until the model owner releases them.
    pub(crate) fn reserve(self: &Arc<Self>, ticket: u64, bytes: usize) -> Result<Lease> {
        ensure!(ticket != 0 && bytes != 0, "image native ticket/reservation");
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image accounting poisoned"))?;
        ensure!(
            !state.failed && bytes <= self.cap.saturating_sub(state.used),
            "image admission unavailable"
        );
        state.used += bytes;
        Ok(Lease {
            owner: self.clone(),
            ticket,
            bytes,
        })
    }
    pub(crate) fn used(&self) -> Result<usize> {
        Ok(self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image accounting poisoned"))?
            .used)
    }
}
pub(crate) struct Lease {
    owner: Arc<Budget>,
    ticket: u64,
    bytes: usize,
}
impl Lease {
    pub(crate) fn ticket(&self) -> u64 {
        self.ticket
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    /// Unexpected loss of an unretired worker fails admission closed. The
    /// allowance cannot be recycled while detached native work may still own it.
    pub(crate) fn abandon(self) {
        if let Ok(mut state) = self.owner.state.lock() {
            state.failed = true;
        }
        std::mem::forget(self);
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if let Ok(mut state) = self.owner.state.lock() {
            state.used = state
                .used
                .checked_sub(self.bytes)
                .expect("image reservation owner");
        }
    }
}

#[derive(Default)]
struct Progress {
    held: bool,
    waiting: bool,
    cancelled: bool,
    boundaries: u64,
    paused: Duration,
}
pub(crate) struct Control {
    state: Mutex<Progress>,
    changed: Condvar,
}
impl Control {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(Progress::default()),
            changed: Condvar::new(),
        })
    }
    pub(crate) fn hold(&self, held: bool) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image control poisoned"))?;
        ensure!(!state.cancelled, "image preparation cancelled");
        state.held = held;
        self.changed.notify_all();
        Ok(())
    }
    pub(crate) fn cancel(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.cancelled = true;
            state.held = false;
            self.changed.notify_all();
        }
    }
    /// Called only by the actual CPU worker at completed native/row boundaries.
    /// A hold retains stack, buffers and cursor; it never means retry/cancel.
    pub(crate) fn check(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image control poisoned"))?;
        state.boundaries = state
            .boundaries
            .checked_add(1)
            .context("image progress exhausted")?;
        while state.held && !state.cancelled {
            state.waiting = true;
            self.changed.notify_all();
            let start = Instant::now();
            state = self
                .changed
                .wait(state)
                .map_err(|_| anyhow::anyhow!("image control poisoned"))?;
            state.paused = state
                .paused
                .checked_add(start.elapsed())
                .context("image paused time overflow")?;
        }
        state.waiting = false;
        ensure!(!state.cancelled, "image preparation cancelled");
        Ok(())
    }
    pub(crate) fn status(&self) -> Result<(bool, bool, bool, u64)> {
        let s = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image control poisoned"))?;
        Ok((s.held, s.waiting, s.cancelled, s.boundaries))
    }
    fn paused(&self) -> Result<Duration> {
        Ok(self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image control poisoned"))?
            .paused)
    }
}

/// The shared source budget excludes time the worker actually waits at a
/// native hold boundary. An in-flight DNS/socket operation is not suspended:
/// its bounded network deadline continues and its result is retained afterward.
pub(crate) struct ActiveDeadline {
    start: Instant,
    paused: Duration,
    budget: Duration,
}
impl ActiveDeadline {
    pub(crate) fn new(control: &Control, budget: Duration) -> Result<Self> {
        ensure!(!budget.is_zero(), "image source time budget");
        Ok(Self {
            start: Instant::now(),
            paused: control.paused()?,
            budget,
        })
    }
    pub(crate) fn deadline(&self, control: &Control) -> Result<Instant> {
        let held = control
            .paused()?
            .checked_sub(self.paused)
            .context("image paused clock regressed")?;
        self.start
            .checked_add(self.budget)
            .and_then(|v| v.checked_add(held))
            .context("image deadline overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queued_and_ready_tickets_share_a_bounded_reservation() {
        let budget = Budget::new(100).unwrap();
        let a = budget.reserve(1, 60).unwrap();
        assert!(budget.reserve(2, 41).is_err());
        let b = budget.reserve(2, 40).unwrap();
        assert_eq!(budget.used().unwrap(), 100);
        assert_eq!((a.ticket(), a.bytes()), (1, 60));
        drop(a);
        assert_eq!(budget.used().unwrap(), 40);
        drop(b);
        assert_eq!(budget.used().unwrap(), 0);
        budget.reserve(3, 10).unwrap().abandon();
        assert!(budget.reserve(4, 1).is_err());
        assert_eq!(budget.used().unwrap(), 10);
    }
    #[test]
    fn held_boundary_retains_progress_then_resumes_once_or_cancels() {
        let control = Control::new();
        control.hold(true).unwrap();
        let worker = control.clone();
        let thread = std::thread::spawn(move || worker.check());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !control.status().unwrap().1 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(control.status().unwrap(), (true, true, false, 1));
        control.hold(false).unwrap();
        thread.join().unwrap().unwrap();
        assert_eq!(control.status().unwrap(), (false, false, false, 1));
        control.hold(true).unwrap();
        control.cancel();
        assert!(control.check().is_err());
        assert!(control.hold(false).is_err());
    }
    #[test]
    fn only_actual_native_wait_extends_shared_source_deadline() {
        let control = Control::new();
        let clock = ActiveDeadline::new(&control, Duration::from_millis(40)).unwrap();
        let original = clock.deadline(&control).unwrap();
        control.hold(true).unwrap();
        // Merely requesting a hold cannot extend an in-flight network timeout.
        assert_eq!(clock.deadline(&control).unwrap(), original);
        let worker = control.clone();
        let thread = std::thread::spawn(move || worker.check());
        let until = Instant::now() + Duration::from_secs(2);
        while !control.status().unwrap().1 {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(80));
        control.hold(false).unwrap();
        thread.join().unwrap().unwrap();
        let moved = clock.deadline(&control).unwrap();
        assert!(moved.duration_since(original) >= Duration::from_millis(80));
        assert!(moved > Instant::now());
        // Re-reading it does not count held time a second time.
        assert_eq!(clock.deadline(&control).unwrap(), moved);
    }
}
