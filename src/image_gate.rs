//! Head-only preparation ownership for the separate vision encoder. A native
//! hold closes background admission before waiting for its last admitted encode.
//! No CUDA object crosses threads; the encoder's completed reply proves drain.
use crate::vision_request::Control;
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
struct State {
    held: bool,
    failed: bool,
    jobs: BTreeMap<u64, (bool, Arc<Control>)>,
    encoding: Option<(u64, bool)>,
    started: u64,
    completed: u64,
}
#[derive(Default)]
pub(crate) struct Gate {
    state: Mutex<State>,
    changed: Condvar,
}
pub(crate) struct Job {
    gate: Arc<Gate>,
    id: u64,
    pub control: Arc<Control>,
}
pub(crate) struct Encode<'a> {
    job: &'a Job,
    drained: bool,
}
impl Gate {
    pub fn status(&self) -> serde_json::Value {
        let s = self.state.lock().unwrap();
        serde_json::json!({"preparing":s.jobs.len(),"held":s.held,"failed":s.failed,
            "encoding":s.encoding.is_some(),"background_encoding":s.encoding.is_some_and(|v|v.1),
            "encodes_started":s.started,"encodes_completed":s.completed})
    }
    pub fn register(self: &Arc<Self>, id: u64, background: bool) -> Result<Job> {
        let mut s = self.state.lock().unwrap();
        ensure!(
            !s.failed && s.jobs.len() < 16,
            "image preparation unavailable"
        );
        ensure!(!s.jobs.contains_key(&id), "image preparation already owned");
        let control = Control::new();
        control.hold(background && s.held)?;
        s.jobs.insert(id, (background, control.clone()));
        Ok(Job {
            gate: self.clone(),
            id,
            control,
        })
    }
    /// Update from the authoritative native receipt. Before requesting a hold,
    /// quiesce() must run so a separate CUDA process cannot outlive its ack.
    pub fn set_hold(&self, held: bool) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        ensure!(!s.failed, "vision encoder drain unproven");
        if s.held != held {
            s.held = held;
            for (background, control) in s.jobs.values() {
                if *background && !control.status()?.2 {
                    control.hold(held)?;
                }
            }
            self.changed.notify_all();
        }
        Ok(())
    }
    pub fn quiesce(&self, timeout: Duration) -> Result<()> {
        self.set_hold(true)?;
        let end = Instant::now() + timeout;
        let mut s = self.state.lock().unwrap();
        while s.encoding.is_some_and(|(_, background)| background) {
            ensure!(!s.failed, "vision encoder drain unproven");
            let wait = end.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                s.failed = true;
                anyhow::bail!("vision encoder did not drain before native hold");
            }
            s = self.changed.wait_timeout(s, wait).unwrap().0;
        }
        ensure!(!s.failed, "vision encoder drain unproven");
        Ok(())
    }
}
impl Job {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn encode(&self) -> Result<Encode<'_>> {
        let mut s = self.gate.state.lock().unwrap();
        loop {
            ensure!(
                !s.failed && !self.control.status()?.2,
                "image preparation cancelled"
            );
            let background = s.jobs.get(&self.id).context("retired image preparation")?.0;
            if s.encoding.is_none() && (!background || !s.held) {
                s.encoding = Some((self.id, background));
                s.started += 1;
                return Ok(Encode {
                    job: self,
                    drained: false,
                });
            }
            // Cancellation belongs to the retained native client, and can wake
            // its source control independently of this encoder condition variable.
            s = self
                .gate
                .changed
                .wait_timeout(s, Duration::from_millis(10))
                .unwrap()
                .0;
        }
    }
}
impl Encode<'_> {
    /// Call only after a complete successful response or a decoded-image
    /// rejection (which the sidecar sends before launching tower work).
    pub fn drained(mut self) {
        self.drained = true;
    }
}
impl Drop for Encode<'_> {
    fn drop(&mut self) {
        let mut s = self.job.gate.state.lock().unwrap();
        assert_eq!(s.encoding.map(|v| v.0), Some(self.job.id));
        if !self.drained {
            // A socket timeout is not proof the other process stopped its GPU.
            // Fail closed; never acknowledge a native hold after losing that proof.
            s.failed = true;
            for (_, c) in s.jobs.values() {
                c.cancel();
            }
        } else {
            s.completed += 1;
        }
        s.encoding = None;
        self.job.gate.changed.notify_all();
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        let mut s = self.gate.state.lock().unwrap();
        assert_ne!(s.encoding.map(|v| v.0), Some(self.id));
        s.jobs.remove(&self.id);
        self.control.cancel();
        self.gate.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_hold_waits_for_admitted_encode_and_blocks_later_background() {
        let gate = Arc::new(Gate::default());
        let job = gate.register(1, true).unwrap();
        let encoding = job.encode().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let g = gate.clone();
        let waiter =
            std::thread::spawn(move || tx.send(g.quiesce(Duration::from_secs(2))).unwrap());
        while !gate.state.lock().unwrap().held {
            std::thread::yield_now();
        }
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
        encoding.drained();
        rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        waiter.join().unwrap();
        let next = gate.register(2, true).unwrap();
        assert!(next.control.status().unwrap().0);
        let foreground = gate.register(3, false).unwrap();
        foreground.encode().unwrap().drained();
        gate.set_hold(false).unwrap();
        next.encode().unwrap().drained();
        assert_eq!(next.control.status().unwrap().3, 0); // no replayed preparation
    }
    #[test]
    fn lost_encoder_reply_cannot_be_reported_as_a_drained_native_hold() {
        let gate = Arc::new(Gate::default());
        let job = gate.register(1, true).unwrap();
        drop(job.encode().unwrap());
        assert!(gate.quiesce(Duration::from_millis(10)).is_err());
        assert!(gate.register(2, false).is_err());
    }
    #[test]
    fn cancelled_waiter_never_enters_encoder() {
        let gate = Arc::new(Gate::default());
        gate.quiesce(Duration::from_millis(10)).unwrap();
        let job = gate.register(1, true).unwrap();
        job.control.cancel();
        assert!(job.encode().is_err());
        assert!(gate.state.lock().unwrap().encoding.is_none());
    }
}
