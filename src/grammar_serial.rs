//! Retained serial grammar owners with asynchronous native mask preparation.
//! Local worker completion is not collective authority: the scheduler must
//! exchange status and agree the ready mask before model selection/commit.
use crate::grammar_runtime::{Constraint, Machine, Masks, Pending, Pool, VOCAB};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, time::Duration};
const OWNERS: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Idle,
    Pending,
    Ready,
    Terminal,
    Failed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Status {
    pub(crate) stage: Stage,
    pub(crate) progress: [u8; 32],
    pub(crate) emitted: usize,
    pub(crate) mask: Option<[u8; 32]>,
}
enum Work<M> {
    Idle(Constraint<M>),
    Pending(Pending<M>),
    Ready(Constraint<M>, Option<Masks>),
    Terminal(Constraint<M>),
    // Keep native ownership until explicit retirement even after failure.
    Failed(Option<Constraint<M>>),
}
struct Entry<M> {
    work: Work<M>,
    progress: [u8; 32],
    emitted: usize,
}
pub(crate) struct Serial<M: Machine + 'static> {
    pool: Pool<M>,
    entries: BTreeMap<u64, Entry<M>>,
    closed: bool,
}
impl<M: Machine + 'static> Serial<M> {
    pub(crate) fn new(workers: usize) -> Result<Self> {
        Ok(Self {
            pool: Pool::new(workers)?,
            entries: BTreeMap::new(),
            closed: false,
        })
    }
    pub(crate) fn insert(&mut self, id: u64, owner: Constraint<M>) -> Result<()> {
        ensure!(
            !self.closed && id != 0 && self.entries.len() < OWNERS,
            "serial grammar owner bound"
        );
        ensure!(
            !self.entries.contains_key(&id),
            "serial grammar owner already exists"
        );
        ensure!(
            owner.emitted() == 0 && !owner.finished()?,
            "serial grammar initial progress"
        );
        self.entries.insert(
            id,
            Entry {
                progress: owner.digest(),
                emitted: 0,
                work: Work::Idle(owner),
            },
        );
        Ok(())
    }
    /// Start before the next model call, preferably while prefill is pending.
    /// A full worker pool leaves the exact owner Idle for a later common retry.
    /// Row-zero token 0 is a placeholder, never accepted by Constraint::cut.
    pub(crate) fn begin(&mut self, id: u64) -> Result<bool> {
        ensure!(!self.closed, "serial grammar is closed");
        let e = self
            .entries
            .get_mut(&id)
            .context("missing serial grammar owner")?;
        ensure!(
            matches!(e.work, Work::Idle(_)),
            "serial grammar already scheduled or terminal"
        );
        let Work::Idle(mut owner) = std::mem::replace(&mut e.work, Work::Failed(None)) else {
            unreachable!()
        };
        if !owner.active() {
            e.work = Work::Ready(owner, None);
            return Ok(true);
        }
        let window = match owner.cut(&[0]) {
            Ok(window) => window,
            Err(error) => {
                e.work = Work::Failed(Some(owner));
                return Err(error);
            }
        };
        ensure!(
            window.tokens() == [0] && window.rows() == [0] && window.cut() == 0,
            "serial grammar window shape"
        );
        match self.pool.start(owner, window) {
            Ok(job) => {
                e.work = Work::Pending(job);
                Ok(true)
            }
            Err(rejected) => {
                ensure!(
                    rejected.window.tokens() == [0] && rejected.window.cut() == 0,
                    "serial grammar rejected window shape"
                );
                e.work = Work::Idle(rejected.constraint);
                Ok(false)
            }
        }
    }
    /// Poll without waiting on native C++. Local Ready must be compared with
    /// all peers; a slower peer's Pending is normal, not a metadata mismatch.
    pub(crate) fn poll(&mut self, id: u64) -> Result<Status> {
        let e = self
            .entries
            .get_mut(&id)
            .context("missing serial grammar owner")?;
        if let Work::Pending(job) = &mut e.work {
            match job.poll(Duration::ZERO) {
                Ok(None) => {}
                Ok(Some(done)) => {
                    if done.constraint.digest() != e.progress
                        || done.constraint.emitted() != e.emitted
                    {
                        e.work = Work::Failed(Some(done.constraint));
                    } else {
                        e.work = match done.masks {
                            Ok(mask)
                                if mask.window().tokens() == [0]
                                    && mask.window().rows() == [0]
                                    && mask.window().cut() == 0 =>
                            {
                                Work::Ready(done.constraint, Some(mask))
                            }
                            Ok(_) => Work::Failed(Some(done.constraint)),
                            Err(_) => Work::Failed(Some(done.constraint)),
                        };
                    }
                }
                Err(_) => {
                    e.work = Work::Failed(None);
                }
            }
        }
        self.status(id)
    }
    pub(crate) fn status(&self, id: u64) -> Result<Status> {
        let e = self
            .entries
            .get(&id)
            .context("missing serial grammar owner")?;
        let (stage, mask) = match &e.work {
            Work::Idle(_) => (Stage::Idle, None),
            Work::Pending(_) => (Stage::Pending, None),
            Work::Ready(_, mask) => {
                let mut hash = Sha256::new();
                hash.update(b"alaska-serial-grammar-selection-v1");
                hash.update(id.to_le_bytes());
                hash.update(e.progress);
                hash.update([u8::from(mask.is_some())]);
                if let Some(mask) = mask {
                    hash.update(mask.digest());
                }
                (Stage::Ready, Some(hash.finalize().into()))
            }
            Work::Terminal(_) => (Stage::Terminal, None),
            Work::Failed(_) => (Stage::Failed, None),
        };
        Ok(Status {
            stage,
            progress: e.progress,
            emitted: e.emitted,
            mask,
        })
    }
    pub(crate) fn mask(&self, id: u64) -> Result<Option<&[i32]>> {
        let e = self
            .entries
            .get(&id)
            .context("missing serial grammar owner")?;
        let Work::Ready(_, masks) = &e.work else {
            anyhow::bail!("serial grammar mask is not ready")
        };
        Ok(masks.as_ref().and_then(|m| m.row(0)))
    }
    /// Called after common successful selection and before publication. `terminal`
    /// represents the already-validated output cap/EOS/visible stop. Native EOS
    /// also terminates independently. No next mask is started implicitly: a hold
    /// must be able to interpose before any new native worker call.
    pub(crate) fn commit(&mut self, id: u64, token: i32, terminal: bool) -> Result<()> {
        ensure!(
            (0..VOCAB as i32).contains(&token),
            "serial grammar output token bounds"
        );
        let e = self
            .entries
            .get_mut(&id)
            .context("missing serial grammar owner")?;
        ensure!(
            matches!(e.work, Work::Ready(_, _)),
            "serial grammar commit without ready selection"
        );
        let Work::Ready(mut owner, mask) = std::mem::replace(&mut e.work, Work::Failed(None))
        else {
            unreachable!()
        };
        if let Some(bits) = mask.as_ref().and_then(|m| m.row(0))
            && (bits[token as usize / 32] as u32 & (1 << (token as usize % 32))) == 0
        {
            e.work = Work::Failed(Some(owner));
            anyhow::bail!("serial grammar selected a masked token");
        }
        if let Err(error) = owner.advance(&[token]) {
            e.work = Work::Failed(Some(owner));
            return Err(error);
        }
        let done = match owner.finished() {
            Ok(done) => terminal || done,
            Err(error) => {
                e.work = Work::Failed(Some(owner));
                return Err(error);
            }
        };
        e.progress = owner.digest();
        e.emitted = owner.emitted();
        e.work = if done {
            Work::Terminal(owner)
        } else {
            Work::Idle(owner)
        };
        Ok(())
    }
    /// Logical progress excludes nondeterministic worker readiness/timing, which
    /// is exchanged separately. Failed/terminal states must still be agreed.
    pub(crate) fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"alaska-serial-grammar-owners-v1");
        hash.update((self.entries.len() as u64).to_le_bytes());
        for (&id, e) in &self.entries {
            hash.update(id.to_le_bytes());
            hash.update(e.progress);
            hash.update((e.emitted as u64).to_le_bytes());
            hash.update([match &e.work {
                Work::Terminal(_) => 1,
                Work::Failed(_) => 2,
                _ => 0,
            }]);
        }
        hash.finalize().into()
    }
    /// Dropping a pending receiver cannot fabricate a drained worker. This
    /// invalidates the logical owner; native retirement is observed separately.
    pub(crate) fn remove(&mut self, id: u64) -> bool {
        self.entries.remove(&id).is_some()
    }
    pub(crate) fn in_flight(&self) -> usize {
        self.pool.in_flight()
    }
    pub(crate) fn count(&self) -> usize {
        self.entries.len()
    }
    pub(crate) fn close(&mut self) {
        self.closed = true;
        self.entries.clear();
        self.pool.close();
    }
    pub(crate) fn retired(&mut self) -> Result<bool> {
        ensure!(self.closed, "close serial grammar before retirement");
        self.pool.retired()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar_runtime::WORDS;
    use std::{
        sync::{Arc, Condvar, Mutex, mpsc},
        time::Instant,
    };
    #[derive(Default)]
    struct Fake {
        accepted: Vec<i32>,
        gate: Option<Arc<(Mutex<bool>, Condvar)>>,
        entered: Option<mpsc::Sender<()>>,
        fail: bool,
    }
    impl Machine for Fake {
        fn words(&self) -> usize {
            WORDS
        }
        fn accept(&mut self, token: i32) -> Result<bool> {
            let accepted = token == self.accepted.len() as i32 + 1;
            if accepted {
                self.accepted.push(token);
            }
            Ok(accepted)
        }
        fn fill(&mut self, bits: &mut [i32]) -> Result<bool> {
            if let Some(entered) = self.entered.take() {
                let _ = entered.send(());
            }
            if let Some(gate) = &self.gate {
                let mut ready = gate.0.lock().unwrap();
                while !*ready {
                    ready = gate.1.wait(ready).unwrap();
                }
            }
            ensure!(!self.fail, "injected native mask failure");
            bits.fill(0);
            bits[0] = 1 << (self.accepted.len() + 1);
            Ok(true)
        }
        fn rollback(&mut self, n: usize) -> Result<()> {
            self.accepted.truncate(self.accepted.len() - n);
            Ok(())
        }
        fn terminated(&self) -> Result<bool> {
            Ok(self.accepted.len() == 3)
        }
    }
    fn owner(fake: Fake, active: bool, id: u8) -> Constraint<Fake> {
        Constraint::new(
            fake,
            active,
            42,
            vec![false; VOCAB].into(),
            131072,
            [id; 32],
        )
        .unwrap()
    }
    fn ready(s: &mut Serial<Fake>, id: u64) -> Status {
        let end = Instant::now() + Duration::from_secs(2);
        loop {
            let status = s.poll(id).unwrap();
            if status.stage != Stage::Pending {
                return status;
            }
            assert!(Instant::now() < end);
            std::thread::yield_now();
        }
    }
    fn retire(s: &mut Serial<Fake>) {
        s.close();
        let end = Instant::now() + Duration::from_secs(2);
        while !s.retired().unwrap() {
            assert!(Instant::now() < end);
            std::thread::yield_now();
        }
        assert_eq!(s.in_flight(), 0);
        assert_eq!(s.count(), 0);
    }
    #[test]
    fn four_interleaved_owners_keep_exact_masks_progress_and_terminal() {
        let mut a = Serial::new(4).unwrap();
        let mut b = Serial::new(1).unwrap();
        for id in 1..=4 {
            a.insert(id, owner(Fake::default(), true, id as u8))
                .unwrap();
            b.insert(id, owner(Fake::default(), true, id as u8))
                .unwrap();
        }
        for token in 1..=3 {
            let before = a.digest();
            for id in 1..=4 {
                assert!(a.begin(id).unwrap());
            }
            assert_eq!(a.digest(), before); // Worker start is not logical progress.
            for id in 1..=4 {
                let am = ready(&mut a, id);
                while !b.begin(id).unwrap() {
                    std::thread::yield_now();
                }
                let bm = ready(&mut b, id);
                assert_eq!(am, bm);
                assert_eq!(a.mask(id).unwrap(), b.mask(id).unwrap());
                assert_eq!(a.mask(id).unwrap().unwrap()[0], 1 << token);
                a.commit(id, token, false).unwrap();
                b.commit(id, token, false).unwrap();
                assert_eq!(a.status(id).unwrap().emitted, token as usize);
                assert_eq!(a.digest(), b.digest());
            }
        }
        for id in 1..=4 {
            assert_eq!(a.status(id).unwrap().stage, Stage::Terminal);
            assert!(a.begin(id).is_err());
            assert!(a.commit(id, 4, false).is_err());
            assert!(a.remove(id));
        }
        retire(&mut a);
        retire(&mut b);
    }
    #[test]
    fn reasoning_close_activates_answer_mask_and_visible_stop_stays_terminal() {
        let mut s = Serial::new(1).unwrap();
        s.insert(7, owner(Fake::default(), false, 7)).unwrap();
        for token in [11, 12, 42] {
            assert!(s.begin(7).unwrap());
            assert_eq!(s.status(7).unwrap().stage, Stage::Ready);
            assert_eq!(s.mask(7).unwrap(), None);
            assert_eq!(s.in_flight(), 0);
            s.commit(7, token, false).unwrap();
        }
        assert!(s.begin(7).unwrap());
        assert_eq!(ready(&mut s, 7).stage, Stage::Ready);
        assert_eq!(s.mask(7).unwrap().unwrap()[0], 2);
        s.commit(7, 1, true).unwrap();
        assert_eq!(s.status(7).unwrap().emitted, 4);
        assert_eq!(s.status(7).unwrap().stage, Stage::Terminal);
        assert!(s.begin(7).is_err());
        retire(&mut s);
    }
    #[test]
    fn hold_and_cancel_do_not_replay_or_fabricate_native_retirement() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered, seen) = mpsc::channel();
        let mut s = Serial::new(1).unwrap();
        s.insert(
            1,
            owner(
                Fake {
                    gate: Some(gate.clone()),
                    entered: Some(entered),
                    ..Fake::default()
                },
                true,
                1,
            ),
        )
        .unwrap();
        let before = s.digest();
        s.begin(1).unwrap();
        seen.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(s.poll(1).unwrap().stage, Stage::Pending);
        assert_eq!(s.digest(), before);
        assert_eq!(s.in_flight(), 1);
        assert!(s.commit(1, 1, false).is_err());
        assert_eq!(s.digest(), before);
        assert!(s.remove(1));
        assert_eq!(s.count(), 0);
        assert_eq!(s.in_flight(), 1);
        s.close();
        assert!(!s.retired().unwrap());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        retire(&mut s);
    }
    #[test]
    fn owner_capacity_failed_masks_and_invalid_selection_are_isolated() {
        let mut s = Serial::new(1).unwrap();
        assert!(s.insert(0, owner(Fake::default(), true, 0)).is_err());
        for id in 1..=16 {
            s.insert(
                id,
                owner(
                    Fake {
                        fail: id == 2,
                        ..Fake::default()
                    },
                    true,
                    id as u8,
                ),
            )
            .unwrap();
        }
        assert!(s.insert(17, owner(Fake::default(), true, 17)).is_err());
        assert!(s.insert(1, owner(Fake::default(), true, 1)).is_err());
        s.begin(1).unwrap();
        ready(&mut s, 1);
        assert!(s.commit(1, 2, false).is_err());
        assert_eq!(s.status(1).unwrap().stage, Stage::Failed);
        assert_eq!(s.status(1).unwrap().emitted, 0);
        s.begin(2).unwrap();
        assert_eq!(ready(&mut s, 2).stage, Stage::Failed);
        s.begin(3).unwrap();
        ready(&mut s, 3);
        assert!(s.commit(3, -1, false).is_err());
        assert_eq!(s.status(3).unwrap().stage, Stage::Ready);
        s.commit(3, 1, false).unwrap();
        assert_eq!(s.status(3).unwrap().emitted, 1);
        retire(&mut s);
    }
    #[test]
    fn worker_capacity_preserves_idle_owner_for_retry() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered, seen) = mpsc::channel();
        let mut s = Serial::new(1).unwrap();
        for id in 1..=5 {
            s.insert(
                id,
                owner(
                    Fake {
                        gate: Some(gate.clone()),
                        entered: if id == 1 { Some(entered.clone()) } else { None },
                        ..Fake::default()
                    },
                    true,
                    id as u8,
                ),
            )
            .unwrap();
        }
        for id in 1..=4 {
            assert!(s.begin(id).unwrap());
        }
        seen.recv_timeout(Duration::from_secs(2)).unwrap();
        let before = s.status(5).unwrap();
        assert!(!s.begin(5).unwrap());
        assert_eq!(s.status(5).unwrap(), before);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        for id in 1..=4 {
            assert_eq!(ready(&mut s, id).stage, Stage::Ready);
        }
        while !s.begin(5).unwrap() {
            std::thread::yield_now();
        }
        assert_eq!(ready(&mut s, 5).stage, Stage::Ready);
        retire(&mut s);
    }
}
