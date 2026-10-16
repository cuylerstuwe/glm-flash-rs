//! Retained grammar progress and bounded exclusive native mask workers.
//! cut/fill temporarily walk drafts and always roll back; only advance commits.
use crate::grammar_native::Matcher;
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
pub(crate) const VOCAB: usize = 154880;
pub(crate) const WORDS: usize = VOCAB / 32;
pub(crate) const ROWS: usize = 16;
const OWNERS: usize = 4;
const OUTPUT: usize = 131072;

/// The trait keeps state-machine tests independent of a native dynamic library.
/// Production uses the pinned Matcher implementation below, exclusively owned.
pub(crate) trait Machine: Send {
    fn words(&self) -> usize;
    fn accept(&mut self, token: i32) -> Result<bool>;
    fn fill(&mut self, bits: &mut [i32]) -> Result<bool>;
    fn rollback(&mut self, count: usize) -> Result<()>;
    fn terminated(&self) -> Result<bool>;
}
impl Machine for Matcher {
    fn words(&self) -> usize {
        Matcher::mask_words(self)
    }
    fn accept(&mut self, token: i32) -> Result<bool> {
        Matcher::accept(self, token)
    }
    fn fill(&mut self, bits: &mut [i32]) -> Result<bool> {
        Matcher::fill(self, bits)
    }
    fn rollback(&mut self, count: usize) -> Result<()> {
        Matcher::rollback(self, count)
    }
    fn terminated(&self) -> Result<bool> {
        Matcher::terminated(self)
    }
}
pub(crate) struct Constraint<M = Matcher> {
    matcher: M,
    active: bool,
    think_end: i32,
    never: Arc<[bool]>,
    output: usize,
    emitted: usize,
    hash: Sha256,
    poisoned: bool,
}
#[derive(Debug)]
pub(crate) struct Window {
    tokens: Vec<i32>,
    rows: Vec<usize>,
    cut: usize,
    at: [u8; 32],
}
impl Window {
    pub(crate) fn tokens(&self) -> &[i32] {
        &self.tokens
    }
    pub(crate) fn rows(&self) -> &[usize] {
        &self.rows
    }
    pub(crate) fn cut(&self) -> usize {
        self.cut
    }
}
pub(crate) struct Masks {
    window: Window,
    bits: Vec<i32>,
    digest: [u8; 32],
}
impl Masks {
    pub(crate) fn window(&self) -> &Window {
        &self.window
    }
    /// Full vocabulary bitmask, not a post-top-k filter. None means unconstrained.
    pub(crate) fn row(&self, row: usize) -> Option<&[i32]> {
        self.window
            .rows
            .binary_search(&row)
            .ok()
            .map(|i| &self.bits[i * WORDS..(i + 1) * WORDS])
    }
    pub(crate) fn digest(&self) -> [u8; 32] {
        self.digest
    }
}
impl<M: Machine> Constraint<M> {
    /// identity must bind request ID, grammar kind/text, tokenizer identity and
    /// actual prompt. never excludes added/padded tokens except configured EOS.
    pub(crate) fn new(
        matcher: M,
        active: bool,
        think_end: i32,
        never: Arc<[bool]>,
        output: usize,
        identity: [u8; 32],
    ) -> Result<Self> {
        ensure!(
            matcher.words() == WORDS
                && never.len() == VOCAB
                && (0..VOCAB as i32).contains(&think_end)
                && (1..=OUTPUT).contains(&output),
            "grammar owner bounds"
        );
        let mut hash = Sha256::new();
        hash.update(b"alaska-retained-grammar-v1");
        hash.update(identity);
        hash.update([u8::from(active)]);
        hash.update(think_end.to_le_bytes());
        hash.update((output as u64).to_le_bytes());
        Ok(Self {
            matcher,
            active,
            think_end,
            never,
            output,
            emitted: 0,
            hash,
            poisoned: false,
        })
    }
    pub(crate) fn digest(&self) -> [u8; 32] {
        let mut hash = self.hash.clone();
        hash.update([u8::from(self.active), u8::from(self.poisoned)]);
        hash.update((self.emitted as u64).to_le_bytes());
        hash.finalize().into()
    }
    pub(crate) fn emitted(&self) -> usize {
        self.emitted
    }
    pub(crate) fn active(&self) -> bool {
        self.active
    }
    fn ready(&self) -> Result<()> {
        ensure!(!self.poisoned, "grammar owner poisoned");
        Ok(())
    }
    pub(crate) fn finished(&self) -> Result<bool> {
        self.ready()?;
        Ok(self.active && self.matcher.terminated()?)
    }
    fn cleanup<T>(&mut self, result: Result<T>, accepted: usize) -> Result<T> {
        let rollback = self.matcher.rollback(accepted);
        if result.is_err() || rollback.is_err() {
            self.poisoned = true;
        }
        rollback.context("grammar temporary rollback failed")?;
        result
    }
    /// Row zero is the pending model input already followed by this grammar.
    /// Its prediction is constrained by the current matcher state. At first
    /// prefill, use any common in-range row-zero token; it is never accepted here.
    pub(crate) fn cut(&mut self, tokens: &[i32]) -> Result<Window> {
        self.ready()?;
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= ROWS
                && tokens.iter().all(|&t| (0..VOCAB as i32).contains(&t)),
            "grammar window bounds"
        );
        let at = self.digest();
        if self.finished()? {
            return Ok(Window {
                tokens: tokens.to_vec(),
                rows: vec![],
                cut: 0,
                at,
            });
        }
        let mut accepted = 0;
        let result = (|| {
            let mut active = self.active;
            let mut rows = if active { vec![0] } else { vec![] };
            let mut keep = 1;
            for (r, &token) in tokens.iter().enumerate().skip(1) {
                if active {
                    if self.never[token as usize] || !self.matcher.accept(token)? {
                        break;
                    }
                    accepted += 1;
                    if self.matcher.terminated()? {
                        break;
                    }
                } else {
                    active = token == self.think_end;
                }
                keep += 1;
                if active {
                    rows.push(r);
                }
            }
            Ok(Window {
                tokens: tokens[..keep].to_vec(),
                rows,
                cut: tokens.len() - keep,
                at,
            })
        })();
        self.cleanup(result, accepted)
    }
    pub(crate) fn fill(&mut self, window: Window) -> Result<Masks> {
        self.ready()?;
        ensure!(
            window.at == self.digest(),
            "stale or foreign grammar window"
        );
        let mut bits = vec![0; window.rows.len() * WORDS];
        let mut accepted = 0;
        let result = (|| {
            if !window.rows.is_empty() {
                let mut active = self.active;
                let mut j = 0;
                for (r, &token) in window.tokens.iter().enumerate() {
                    if r != 0 {
                        if active {
                            ensure!(self.matcher.accept(token)?, "grammar rejected kept draft");
                            accepted += 1;
                        } else {
                            active = token == self.think_end;
                        }
                    }
                    if active {
                        ensure!(
                            window.rows.get(j) == Some(&r),
                            "grammar constrained rows changed"
                        );
                        let row = &mut bits[j * WORDS..(j + 1) * WORDS];
                        self.matcher.fill(row)?;
                        ensure!(row.iter().any(|&v| v != 0), "grammar allows no token");
                        j += 1;
                    }
                }
                ensure!(
                    j == window.rows.len(),
                    "grammar constrained row count changed"
                );
            }
            Ok(())
        })();
        self.cleanup(result, accepted)?;
        let mut hash = Sha256::new();
        hash.update(b"alaska-grammar-masks-v1");
        hash.update(window.at);
        hash.update((window.tokens.len() as u64).to_le_bytes());
        for &token in &window.tokens {
            hash.update(token.to_le_bytes());
        }
        hash.update((window.cut as u64).to_le_bytes());
        hash.update((window.rows.len() as u64).to_le_bytes());
        for &row in &window.rows {
            hash.update((row as u64).to_le_bytes());
        }
        for &word in &bits {
            hash.update(word.to_le_bytes());
        }
        Ok(Masks {
            window,
            bits,
            digest: hash.finalize().into(),
        })
    }
    /// Follow only output actually published under the masks. Never advance
    /// speculative proposals, resumed prompt bytes or a decoded cache snapshot.
    /// Failure poisons this owner, so partially advanced state cannot be reused.
    pub(crate) fn advance(&mut self, tokens: &[i32]) -> Result<()> {
        self.ready()?;
        ensure!(
            tokens.len() <= ROWS && tokens.iter().all(|&t| (0..VOCAB as i32).contains(&t)),
            "grammar accepted path bounds"
        );
        ensure!(
            self.emitted
                .checked_add(tokens.len())
                .is_some_and(|n| n <= self.output),
            "grammar output bound"
        );
        let result = (|| {
            for &token in tokens {
                if self.active {
                    ensure!(
                        !self.matcher.terminated()?,
                        "grammar committed output after EOS"
                    );
                    ensure!(
                        !self.never[token as usize] && self.matcher.accept(token)?,
                        "grammar rejected chosen token"
                    );
                } else {
                    self.active = token == self.think_end;
                }
                self.emitted += 1;
                self.hash.update(token.to_le_bytes());
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

pub(crate) struct Complete<M = Matcher> {
    pub(crate) constraint: Constraint<M>,
    pub(crate) masks: Result<Masks>,
}
struct Work<M> {
    constraint: Constraint<M>,
    window: Window,
    reply: mpsc::SyncSender<Complete<M>>,
}
pub(crate) struct Pending<M = Matcher> {
    reply: mpsc::Receiver<Complete<M>>,
    taken: bool,
}
impl<M> Pending<M> {
    /// Timeout retains the original pending handle and exclusive matcher owner.
    pub(crate) fn poll(&mut self, timeout: Duration) -> Result<Option<Complete<M>>> {
        ensure!(!self.taken, "grammar result already consumed");
        match self.reply.recv_timeout(timeout) {
            Ok(result) => {
                self.taken = true;
                Ok(Some(result))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("grammar worker exited before returning ownership")
            }
        }
    }
}
pub(crate) struct Rejected<M = Matcher> {
    pub(crate) constraint: Constraint<M>,
    pub(crate) window: Window,
}
pub(crate) struct Pool<M = Matcher> {
    send: Option<mpsc::SyncSender<Work<M>>>,
    workers: Vec<JoinHandle<()>>,
    busy: Arc<AtomicUsize>,
    limit: usize,
    _exclusive: PhantomData<Cell<()>>,
}
struct Busy(Arc<AtomicUsize>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
impl<M: Machine + 'static> Pool<M> {
    /// Construct on the loader/preparation thread so worker CPU affinity does
    /// not accidentally inherit a later single-core CUDA owner affinity.
    pub(crate) fn new(workers: usize) -> Result<Self> {
        ensure!((1..=OWNERS).contains(&workers), "grammar worker bound");
        let (send, receive) = mpsc::sync_channel::<Work<M>>(OWNERS);
        let receive = Arc::new(Mutex::new(receive));
        let busy = Arc::new(AtomicUsize::new(0));
        let mut pool = Self {
            send: Some(send),
            workers: vec![],
            busy,
            limit: OWNERS,
            _exclusive: PhantomData,
        };
        for i in 0..workers {
            let receive = receive.clone();
            let busy = pool.busy.clone();
            let handle = thread::Builder::new()
                .name(format!("alaska-grammar-{i}"))
                .spawn(move || {
                    crate::cpu_pin::helper_thread();
                    loop {
                        let work = receive.lock().unwrap_or_else(|p| p.into_inner()).recv();
                        let Ok(Work {
                            mut constraint,
                            window,
                            reply,
                        }) = work
                        else {
                            break;
                        };
                        let _busy = Busy(busy.clone());
                        let masks = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            constraint.fill(window)
                        }))
                        .unwrap_or_else(|_| {
                            constraint.poisoned = true;
                            Err(anyhow::anyhow!("grammar worker panicked"))
                        });
                        // A one-entry result channel never waits for the model
                        // owner. Cancellation may drop it, but only actual worker
                        // completion releases busy/Constraint ownership.
                        let _ = reply.send(Complete { constraint, masks });
                    }
                })?;
            pool.workers.push(handle);
        }
        Ok(pool)
    }
    pub(crate) fn start(
        &mut self,
        constraint: Constraint<M>,
        window: Window,
    ) -> std::result::Result<Pending<M>, Box<Rejected<M>>> {
        let Some(send) = &self.send else {
            return Err(Box::new(Rejected { constraint, window }));
        };
        if self.busy.load(Ordering::Acquire) >= self.limit {
            return Err(Box::new(Rejected { constraint, window }));
        }
        let (reply, result) = mpsc::sync_channel(1);
        self.busy.fetch_add(1, Ordering::AcqRel);
        let work = Work {
            constraint,
            window,
            reply,
        };
        if let Err(error) = send.try_send(work) {
            self.busy.fetch_sub(1, Ordering::AcqRel);
            let work = match error {
                mpsc::TrySendError::Full(w) | mpsc::TrySendError::Disconnected(w) => w,
            };
            return Err(Box::new(Rejected {
                constraint: work.constraint,
                window: work.window,
            }));
        }
        Ok(Pending {
            reply: result,
            taken: false,
        })
    }
    pub(crate) fn in_flight(&self) -> usize {
        self.busy.load(Ordering::Acquire)
    }
    pub(crate) fn close(&mut self) {
        self.send.take();
    }
    /// True only after every worker exited and every native call drained.
    pub(crate) fn retired(&mut self) -> Result<bool> {
        ensure!(self.send.is_none(), "close grammar pool before retirement");
        if !self.workers.iter().all(JoinHandle::is_finished) {
            return Ok(false);
        }
        for worker in self.workers.drain(..) {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("grammar worker exit panic"))?;
        }
        ensure!(self.in_flight() == 0, "grammar worker accounting drift");
        Ok(true)
    }
}
impl<M> Drop for Pool<M> {
    fn drop(&mut self) {
        self.send.take();
        // The normal supervisor path polls retired under its own deadline.
        // Destruction must not detach native work or free its live handles.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Condvar;
    #[derive(Default)]
    struct Fake {
        accepted: Vec<i32>,
        empty: bool,
        panic: bool,
        gate: Option<Arc<(Mutex<bool>, Condvar)>>,
        entered: Option<mpsc::Sender<()>>,
    }
    impl Machine for Fake {
        fn words(&self) -> usize {
            WORDS
        }
        fn accept(&mut self, t: i32) -> Result<bool> {
            if self.accepted.len() < 4 && t == self.accepted.len() as i32 + 1 {
                self.accepted.push(t);
                Ok(true)
            } else {
                Ok(false)
            }
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
            assert!(!self.panic, "injected worker panic");
            bits.fill(0);
            if !self.empty && self.accepted.len() < 4 {
                bits[0] = 1 << (self.accepted.len() + 1);
            }
            Ok(true)
        }
        fn rollback(&mut self, n: usize) -> Result<()> {
            ensure!(n <= self.accepted.len(), "fake rollback bound");
            self.accepted.truncate(self.accepted.len() - n);
            Ok(())
        }
        fn terminated(&self) -> Result<bool> {
            Ok(self.accepted.len() == 4)
        }
    }
    fn con(active: bool) -> Constraint<Fake> {
        let mut never = vec![false; VOCAB];
        never[99] = true;
        Constraint::new(Fake::default(), active, 42, never.into(), OUTPUT, [7; 32]).unwrap()
    }
    fn check(bits: &[i32], id: i32) -> bool {
        bits[id as usize / 32] as u32 & (1 << (id % 32)) != 0
    }
    fn complete(pending: &mut Pending<Fake>) -> Complete<Fake> {
        pending
            .poll(Duration::from_secs(2))
            .unwrap()
            .expect("worker completion")
    }
    fn retire(pool: &mut Pool<Fake>) {
        pool.close();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !pool.retired().unwrap() {
            assert!(
                std::time::Instant::now() < deadline,
                "worker retirement deadline"
            );
            thread::yield_now();
        }
        assert_eq!(pool.in_flight(), 0);
    }
    #[test]
    fn kept_drafts_cut_before_forward_and_masks_are_next_row_distributions() {
        let mut c = con(true);
        let before = c.digest();
        let win = c.cut(&[0, 1, 2, 9]).unwrap();
        assert_eq!(win.tokens(), [0, 1, 2]);
        assert_eq!(win.rows(), [0, 1, 2]);
        assert_eq!(win.cut(), 1);
        assert_eq!(c.digest(), before);
        assert!(c.matcher.accepted.is_empty());
        let masks = c.fill(win).unwrap();
        for row in 0..3 {
            assert!(check(masks.row(row).unwrap(), row as i32 + 1));
        }
        assert!(masks.row(3).is_none());
        assert_eq!(masks.window().cut(), 1);
        assert_eq!(c.digest(), before);
        assert!(c.matcher.accepted.is_empty());
        let mut same = con(true);
        let win = same.cut(&[0, 1, 2, 9]).unwrap();
        assert_eq!(same.fill(win).unwrap().digest(), masks.digest());
    }
    #[test]
    fn eos_draft_drops_itself_and_tail_but_parent_may_choose_eos() {
        let mut c = con(true);
        let win = c.cut(&[0, 1, 2, 3, 4, 8]).unwrap();
        assert_eq!(win.tokens(), [0, 1, 2, 3]);
        assert_eq!(win.cut(), 2);
        let masks = c.fill(win).unwrap();
        assert!(check(masks.row(3).unwrap(), 4));
        c.advance(&[1, 2, 3, 4]).unwrap();
        assert!(c.finished().unwrap());
        assert_eq!(c.emitted(), 4);
        let win = c.cut(&[4, 9]).unwrap();
        assert!(win.rows().is_empty());
        assert!(c.fill(win).unwrap().row(0).is_none());
    }
    #[test]
    fn trailing_committed_output_after_eos_is_poisoned_not_silently_dropped() {
        let mut c = con(true);
        c.advance(&[1, 2, 3]).unwrap();
        let window = c.cut(&[3, 4, 9]).unwrap();
        assert_eq!(window.tokens(), [3]);
        let masks = c.fill(window).unwrap();
        assert!(check(masks.row(0).unwrap(), 4));
        assert!(c.advance(&[4, 9]).is_err());
        assert_eq!(c.emitted(), 4);
        assert!(c.finished().is_err());
        let mut c = con(true);
        c.advance(&[1, 2, 3, 4]).unwrap();
        assert_eq!(c.emitted(), 4);
        assert!(c.advance(&[4]).is_err());
        assert_eq!(c.emitted(), 4);
    }
    #[test]
    fn think_close_activates_its_prediction_without_accepting_delimiter() {
        let mut c = con(false);
        let before = c.digest();
        let win = c.cut(&[10, 7, 42, 1, 2]).unwrap();
        assert_eq!(win.rows(), [2, 3, 4]);
        let masks = c.fill(win).unwrap();
        assert!(masks.row(0).is_none());
        assert!(masks.row(1).is_none());
        assert!(check(masks.row(2).unwrap(), 1));
        assert!(check(masks.row(4).unwrap(), 3));
        assert_eq!(before, c.digest());
        assert!(!c.active());
        c.advance(&[7, 42, 1]).unwrap();
        assert!(c.active());
        assert_eq!(c.emitted(), 3);
        assert_eq!(c.matcher.accepted, [1]);
        let win = c.cut(&[1]).unwrap();
        assert!(check(c.fill(win).unwrap().row(0).unwrap(), 2));
    }
    #[test]
    fn unconstrained_thinking_has_no_masks_or_native_accepts() {
        let mut c = con(false);
        let win = c.cut(&[6, 5, 8]).unwrap();
        assert!(win.rows().is_empty());
        assert!(c.fill(win).unwrap().row(0).is_none());
        c.advance(&[6, 5, 8]).unwrap();
        assert_eq!(c.emitted(), 3);
        assert!(c.matcher.accepted.is_empty());
    }
    #[test]
    fn stale_window_and_bounds_fail_before_native_mutation() {
        let mut c = con(true);
        let before = c.digest();
        for ids in [vec![], vec![-1], vec![VOCAB as i32], vec![1; ROWS + 1]] {
            assert!(c.cut(&ids).is_err());
        }
        assert_eq!(c.digest(), before);
        let win = c.cut(&[0]).unwrap();
        c.advance(&[1]).unwrap();
        assert!(c.fill(win).is_err());
        assert_eq!(c.matcher.accepted, [1]);
        let fresh = c.cut(&[1]).unwrap();
        assert!(c.fill(fresh).is_ok());
        c.output = 1;
        assert!(c.advance(&[2]).is_err());
        assert_eq!(c.matcher.accepted, [1]);
    }
    #[test]
    fn forbidden_draft_is_cut_without_native_accept() {
        let mut c = con(true);
        let win = c.cut(&[0, 99, 1]).unwrap();
        assert_eq!(win.tokens(), [0]);
        assert_eq!(win.cut(), 2);
        assert!(c.matcher.accepted.is_empty());
    }
    #[test]
    fn chosen_rejection_and_empty_mask_poison_only_the_affected_owner() {
        let mut c = con(true);
        assert!(c.advance(&[1, 9]).is_err());
        assert!(c.cut(&[0]).is_err());
        let mut empty = con(true);
        empty.matcher.empty = true;
        let win = empty.cut(&[0, 1]).unwrap();
        assert!(empty.fill(win).is_err());
        assert!(empty.matcher.accepted.is_empty());
        assert!(empty.finished().is_err());
        let mut other = con(true);
        let win = other.cut(&[0]).unwrap();
        assert!(other.fill(win).is_ok());
    }
    #[test]
    fn exclusive_worker_returns_original_progress_and_same_masks() {
        let mut expected = con(true);
        let win = expected.cut(&[0, 1, 2]).unwrap();
        let masks = expected.fill(win).unwrap();
        let mut c = con(true);
        let before = c.digest();
        let win = c.cut(&[0, 1, 2]).unwrap();
        let mut pool = Pool::new(2).unwrap();
        let mut job = pool
            .start(c, win)
            .unwrap_or_else(|_| panic!("unexpected queue rejection"));
        let done = complete(&mut job);
        assert_eq!(done.constraint.digest(), before);
        assert_eq!(done.masks.unwrap().digest(), masks.digest());
        assert!(job.poll(Duration::ZERO).is_err());
        let mut returned = done.constraint;
        returned.advance(&[1, 2]).unwrap();
        assert_eq!(returned.emitted(), 2);
        retire(&mut pool);
        let win = returned.cut(&[2]).unwrap();
        let rejected = match pool.start(returned, win) {
            Err(v) => v,
            Ok(_) => panic!("closed pool admitted"),
        };
        let mut returned = rejected.constraint;
        assert!(returned.fill(rejected.window).is_ok());
    }
    #[test]
    fn dropped_pending_cannot_fake_hold_drain_or_exceed_four_owner_bound() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered, receive) = mpsc::channel();
        let mut pool = Pool::new(1).unwrap();
        let mut pending = Vec::new();
        for i in 0..4 {
            let mut c = con(true);
            c.matcher.gate = Some(gate.clone());
            if i == 0 {
                c.matcher.entered = Some(entered.clone());
            }
            let win = c.cut(&[0]).unwrap();
            pending.push(
                pool.start(c, win)
                    .unwrap_or_else(|_| panic!("capacity rejection")),
            );
        }
        receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(pool.in_flight(), 4);
        assert!(pending[0].poll(Duration::ZERO).unwrap().is_none());
        let mut fifth = con(true);
        let win = fifth.cut(&[0]).unwrap();
        assert!(pool.start(fifth, win).is_err());
        drop(pending);
        assert_eq!(pool.in_flight(), 4);
        pool.close();
        assert!(!pool.retired().unwrap());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        retire(&mut pool);
    }
    #[test]
    fn worker_panic_returns_poisoned_owner_and_worker_remains_usable() {
        let mut pool = Pool::new(1).unwrap();
        let mut c = con(true);
        c.matcher.panic = true;
        let win = c.cut(&[0]).unwrap();
        let mut job = pool.start(c, win).unwrap_or_else(|_| panic!("rejected"));
        let mut failed = complete(&mut job);
        assert!(failed.masks.is_err());
        assert!(failed.constraint.cut(&[0]).is_err());
        let mut c = con(true);
        let win = c.cut(&[0]).unwrap();
        let mut job = pool.start(c, win).unwrap_or_else(|_| panic!("rejected"));
        assert!(complete(&mut job).masks.is_ok());
        retire(&mut pool);
        assert!(Pool::<Fake>::new(0).is_err());
        assert!(Pool::<Fake>::new(5).is_err());
    }
}
