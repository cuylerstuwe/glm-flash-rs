//! Admission and drain for one immutable snapshot's independent copy stream.
//! Closing the gate waits for CUDA only, never for filesystem I/O.
use anyhow::{Result, ensure};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
struct State {
    held: bool,
    cancelled: bool,
    copying: bool,
}
pub struct Gate {
    state: Mutex<State>,
    wake: Condvar,
    pub fatal: AtomicBool,
    pub bytes: AtomicUsize,
    pub calls: AtomicUsize,
    pub slices: AtomicUsize,
}
impl Gate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                held: true,
                ..State::default()
            }),
            wake: Condvar::new(),
            fatal: AtomicBool::new(false),
            bytes: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            slices: AtomicUsize::new(0),
        })
    }
    pub fn enter(&self) -> Result<Permit<'_>> {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.held && !s.cancelled {
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        ensure!(!s.cancelled, "SSD copy cancelled");
        ensure!(!s.copying, "SSD copy gate has multiple workers");
        s.copying = true;
        Ok(Permit(self))
    }
    pub fn hold(&self, held: bool) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.held = held;
        self.wake.notify_all();
        while held && s.copying {
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }
    /// Rare model graph captures use CUDA's global capture mode. Temporarily
    /// close this independent stream without changing a native hold's state.
    pub fn pause(self: &Arc<Self>) -> Pause {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let was_held = s.held;
        s.held = true;
        while s.copying {
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        Pause {
            gate: self.clone(),
            was_held,
        }
    }
    pub fn cancel(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.cancelled = true;
        self.wake.notify_all();
        while s.copying {
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }
    pub fn progress(&self) -> (usize, usize) {
        (
            self.bytes.load(Ordering::Relaxed),
            self.slices.load(Ordering::Relaxed),
        )
    }
}
pub struct Pause {
    gate: Arc<Gate>,
    was_held: bool,
}
impl Drop for Pause {
    fn drop(&mut self) {
        self.gate.hold(self.was_held);
    }
}
pub struct Permit<'a>(&'a Gate);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        s.copying = false;
        self.0.wake.notify_all();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};
    #[test]
    fn hold_drains_admitted_copy_and_blocks_the_next() {
        let gate = Gate::new();
        gate.hold(false);
        let (started, seen) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let g = gate.clone();
        let worker = thread::spawn(move || {
            let p = g.enter().unwrap();
            started.send(()).unwrap();
            wait.recv().unwrap();
            drop(p);
        });
        seen.recv_timeout(Duration::from_secs(1)).unwrap();
        let g = gate.clone();
        let (ack, receipt) = mpsc::channel();
        let holder = thread::spawn(move || {
            g.hold(true);
            ack.send(()).unwrap();
        });
        assert!(receipt.recv_timeout(Duration::from_millis(20)).is_err());
        release.send(()).unwrap();
        receipt.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
        holder.join().unwrap();
        let g = gate.clone();
        let (ran, recv) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _p = g.enter().unwrap();
            ran.send(()).unwrap();
        });
        assert!(recv.recv_timeout(Duration::from_millis(20)).is_err());
        gate.hold(false);
        recv.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
    }
    #[test]
    fn cancellation_wakes_a_held_worker_and_cannot_be_released() {
        let gate = Gate::new();
        let g = gate.clone();
        let worker = thread::spawn(move || g.enter().is_err());
        gate.cancel();
        gate.hold(false);
        assert!(worker.join().unwrap());
        assert!(gate.enter().is_err());
    }
    #[test]
    fn capture_pause_preserves_native_hold_and_sticky_cancellation() {
        let gate = Gate::new();
        {
            let _capture = gate.pause();
        }
        assert!(gate.state.lock().unwrap().held);
        gate.hold(false);
        {
            let _capture = gate.pause();
            assert!(gate.state.lock().unwrap().held);
        }
        assert!(!gate.state.lock().unwrap().held);
        {
            let _capture = gate.pause();
            gate.cancel();
        }
        assert!(gate.enter().is_err());
    }
}
