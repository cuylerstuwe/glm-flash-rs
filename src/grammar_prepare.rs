//! CPU compiler actor. Compiler/Compiled never cross threads; only exclusively
//! owned native matchers do. Compiler caches and outstanding handles are bounded.
use crate::{
    grammar_bound::Bound,
    grammar_native::{Compiler, Kind, Matcher, Native},
    grammar_request::CompileKind,
    grammar_runtime::{Constraint, Machine},
    grammar_vocab::Views,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const QUEUED: usize = 4;
const OWNERS: usize = 16;
const ACTIVE_BYTES: usize = 512 << 20;
#[derive(Default)]
struct Budget {
    bytes: AtomicUsize,
    owners: AtomicUsize,
    cached: AtomicUsize,
}
struct Lease {
    budget: Arc<Budget>,
    bytes: usize,
}
// Keep the same acquire/release reservation on both the Mac's Rust 1.95 and
// GB10's Rust 1.99, where fetch_update was renamed to try_update.
fn reserve(counter: &AtomicUsize, amount: usize, limit: usize) -> bool {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let Some(next) = current.checked_add(amount).filter(|&n| n <= limit) else {
            return false;
        };
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
}
impl Lease {
    fn new(budget: &Arc<Budget>, bytes: usize) -> Result<Self> {
        ensure!(
            bytes > 0 && bytes <= 128 << 20,
            "grammar active object byte bound"
        );
        ensure!(
            reserve(&budget.owners, 1, OWNERS),
            "grammar active owner limit"
        );
        if !reserve(&budget.bytes, bytes, ACTIVE_BYTES) {
            budget.owners.fetch_sub(1, Ordering::AcqRel);
            anyhow::bail!("grammar active byte limit");
        }
        Ok(Self {
            budget: budget.clone(),
            bytes,
        })
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.budget.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        self.budget.owners.fetch_sub(1, Ordering::AcqRel);
    }
}
pub(crate) struct LeasedMatcher {
    matcher: Matcher,
    _lease: Lease,
}
impl Machine for LeasedMatcher {
    fn words(&self) -> usize {
        self.matcher.mask_words()
    }
    fn accept(&mut self, token: i32) -> Result<bool> {
        self.matcher.accept(token)
    }
    fn fill(&mut self, bits: &mut [i32]) -> Result<bool> {
        self.matcher.fill(bits)
    }
    fn rollback(&mut self, count: usize) -> Result<()> {
        self.matcher.rollback(count)
    }
    fn terminated(&self) -> Result<bool> {
        self.matcher.terminated()
    }
}
pub(crate) type Owner = Constraint<LeasedMatcher>;
pub(crate) struct Ready {
    matcher: LeasedMatcher,
    bound: Bound,
    never: Arc<[bool]>,
    views: [u8; 32],
    bytes: usize,
}
impl Ready {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn bound_digest(&self) -> [u8; 32] {
        self.bound.digest()
    }
    pub(crate) fn bind(self, request_id: u64, output: usize) -> Result<Owner> {
        let mut hash = Sha256::new();
        hash.update(b"alaska-grammar-request-owner-v1");
        hash.update(request_id.to_le_bytes());
        hash.update(self.bound.digest());
        hash.update(self.views);
        Constraint::new(
            self.matcher,
            self.bound.active(),
            crate::output_decode::THINK_END,
            self.never,
            output,
            hash.finalize().into(),
        )
    }
}
struct Job {
    bound: Bound,
    reply: mpsc::SyncSender<Result<Ready>>,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
}
pub(crate) struct Pending {
    reply: mpsc::Receiver<Result<Ready>>,
    cancel: Arc<AtomicBool>,
    taken: bool,
    done: Arc<AtomicBool>,
    received: Option<Result<Ready>>,
}
impl Pending {
    /// Cancellation is a request, not proof that native compilation stopped.
    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    /// Drain unpublished native handles before releasing the preparation ticket.
    /// A worker racing cancellation may already have queued its Ready result.
    pub(crate) fn retired(&mut self) -> Result<bool> {
        ensure!(
            self.cancel.load(Ordering::Acquire),
            "cancel grammar preparation before retirement"
        );
        if !self.done.load(Ordering::Acquire) {
            return Ok(false);
        }
        if !self.taken {
            if let Some(value) = self.received.take() {
                drop(value);
                self.taken = true;
                return Ok(true);
            }
            match self.reply.try_recv() {
                Ok(value) => {
                    drop(value);
                    self.taken = true;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.taken = true;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    anyhow::bail!("retired compiler has unclosed receipt")
                }
            }
        }
        Ok(true)
    }
    pub(crate) fn poll(&mut self, wait: Duration) -> Result<Option<Result<Ready>>> {
        ensure!(!self.taken, "grammar preparation already consumed");
        ensure!(
            !self.cancel.load(Ordering::Acquire),
            "grammar preparation cancelled"
        );
        let deadline = Instant::now()
            .checked_add(wait)
            .context("grammar poll deadline overflow")?;
        if self.received.is_none() {
            match self.reply.recv_timeout(wait) {
                Ok(value) => self.received = Some(value),
                Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("grammar compiler exited before receipt")
                }
            }
        }
        // A reply can arrive just before Busy releases the compiler-job slot.
        // Never let a Ready staging entry disappear while that slot is still
        // counted: peers could otherwise observe different admission capacity.
        while !self.done.load(Ordering::Acquire) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            thread::sleep(remaining.min(Duration::from_micros(50)));
        }
        self.taken = true;
        Ok(self.received.take())
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}
pub(crate) struct Preparer {
    send: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    busy: Arc<AtomicUsize>,
    budget: Arc<Budget>,
}
struct Busy {
    count: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
}
impl Drop for Busy {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
        self.done.store(true, Ordering::Release);
    }
}
fn publish_or_discard<T>(result: T, reply: mpsc::SyncSender<T>, cancel: &AtomicBool, busy: Busy) {
    if !cancel.load(Ordering::Acquire) {
        let _ = reply.send(result);
    } else {
        // Native destruction can be substantial. It must finish before done
        // allows the preparation ticket and collective capacity to be released.
        drop(result);
    }
    drop(reply);
    drop(busy);
}
fn prepare(
    compilers: &mut [Compiler; 2],
    bound: Bound,
    never: &[Arc<[bool]>; 2],
    views: [u8; 32],
    budget: &Arc<Budget>,
) -> Result<Ready> {
    let view = usize::from(bound.tools());
    let kind = match bound.kind() {
        CompileKind::JsonSchema => Kind::JsonSchema,
        CompileKind::Regex => Kind::Regex,
        CompileKind::Ebnf => Kind::Ebnf,
        CompileKind::StructuralTag => Kind::StructuralTag,
    };
    let compiled = compilers[view]
        .compile(kind, bound.source())
        .context("request grammar cannot be enforced")?;
    let bytes = usize::try_from(compiled.bytes()?)?;
    let cached = compilers
        .iter()
        .try_fold(0usize, |sum, c| -> Result<usize> {
            sum.checked_add(usize::try_from(c.cache_bytes()?)?)
                .context("grammar cache byte overflow")
        })?;
    budget.cached.store(cached, Ordering::Release);
    let lease = Lease::new(budget, bytes)?;
    let matcher = compiled.matcher()?;
    Ok(Ready {
        matcher: LeasedMatcher {
            matcher,
            _lease: lease,
        },
        bound,
        never: never[view].clone(),
        views,
        bytes,
    })
}
impl Preparer {
    pub(crate) fn new(native: Native, views: Arc<Views>) -> Result<Self> {
        let (send, receive) = mpsc::sync_channel::<Job>(QUEUED);
        let (boot, ready) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicUsize::new(0));
        let budget = Arc::new(Budget::default());
        let worker_busy = busy.clone();
        let worker_budget = budget.clone();
        let worker = thread::Builder::new()
            .name("alaska-grammar-compile".into())
            .spawn(move || {
                crate::cpu_pin::helper_thread();
                // Created/used/destroyed on this one CPU thread; no unsafe compiler
                // Send/Sync assertion and no native compiler mutex on the GPU owner.
                let setup = (|| -> Result<[Compiler; 2]> {
                    Ok([
                        native.compiler(&views.text, true, views.add_prefix_space, &views.stop)?,
                        native.compiler(&views.tools, true, views.add_prefix_space, &views.stop)?,
                    ])
                })();
                let mut compilers = match setup {
                    Ok(value) => {
                        let _ = boot.send(Ok(()));
                        value
                    }
                    Err(error) => {
                        let _ = boot.send(Err(error));
                        return;
                    }
                };
                let never = [&views.text, &views.tools].map(|view| {
                    Arc::from(
                        view.iter()
                            .enumerate()
                            .map(|(id, token)| {
                                token.is_empty() && !views.stop.contains(&(id as i32))
                            })
                            .collect::<Vec<_>>(),
                    )
                });
                while let Ok(job) = receive.recv() {
                    // Drop reply before publishing done, including the early
                    // cancelled branch. Retirement then observes a closed reply
                    // or an already-queued result it can safely destroy.
                    let busy = Busy {
                        count: worker_busy.clone(),
                        done: job.done.clone(),
                    };
                    let reply = job.reply;
                    if job.cancel.load(Ordering::Acquire) {
                        drop(reply);
                        drop(busy);
                        continue;
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        prepare(
                            &mut compilers,
                            job.bound,
                            &never,
                            views.identity,
                            &worker_budget,
                        )
                    }))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("grammar compiler worker panicked")));
                    // Cancellation cannot stop an in-flight native compiler call.
                    // It can discard its unpublished result after the call returns.
                    publish_or_discard(result, reply, &job.cancel, busy);
                }
            })?;
        let mut owner = Self {
            send: Some(send),
            worker: Some(worker),
            busy,
            budget,
        };
        if let Err(error) = ready
            .recv()
            .context("grammar compiler initialization failed")?
        {
            owner.close();
            return Err(error);
        }
        Ok(owner)
    }
    /// Rejection returns the original cheap Arc-backed bound value for retry.
    pub(crate) fn start(&mut self, bound: Bound) -> std::result::Result<Pending, Bound> {
        let Some(send) = &self.send else {
            return Err(bound);
        };
        if self.busy.load(Ordering::Acquire) >= QUEUED {
            return Err(bound);
        }
        let (reply, receive) = mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        self.busy.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = send.try_send(Job {
            bound,
            reply,
            cancel: cancel.clone(),
            done: done.clone(),
        }) {
            self.busy.fetch_sub(1, Ordering::AcqRel);
            return Err(match error {
                mpsc::TrySendError::Full(job) | mpsc::TrySendError::Disconnected(job) => job.bound,
            });
        }
        Ok(Pending {
            reply: receive,
            cancel,
            taken: false,
            done,
            received: None,
        })
    }
    pub(crate) fn in_flight(&self) -> usize {
        self.busy.load(Ordering::Acquire)
    }
    pub(crate) fn active_bytes(&self) -> usize {
        self.budget.bytes.load(Ordering::Acquire)
    }
    pub(crate) fn active_owners(&self) -> usize {
        self.budget.owners.load(Ordering::Acquire)
    }
    pub(crate) fn cached_bytes(&self) -> usize {
        self.budget.cached.load(Ordering::Acquire)
    }
    pub(crate) fn close(&mut self) {
        self.send.take();
    }
    pub(crate) fn retired(&mut self) -> Result<bool> {
        ensure!(
            self.send.is_none(),
            "close grammar compiler before retirement"
        );
        if self
            .worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
        {
            return Ok(false);
        }
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("grammar compiler exit panic"))?;
        }
        ensure!(self.in_flight() == 0, "grammar compiler accounting drift");
        Ok(true)
    }
}
impl Drop for Preparer {
    fn drop(&mut self) {
        self.send.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_unpublished_result_dies_before_job_retirement() {
        struct DestructionGate {
            entered: mpsc::SyncSender<()>,
            release: mpsc::Receiver<()>,
        }
        impl Drop for DestructionGate {
            fn drop(&mut self) {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
        }
        let (entered, observe) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        let (reply, receive) = mpsc::sync_channel(1);
        let count = Arc::new(AtomicUsize::new(1));
        let done = Arc::new(AtomicBool::new(false));
        let busy = Busy {
            count: count.clone(),
            done: done.clone(),
        };
        let worker = thread::spawn(move || {
            crate::cpu_pin::helper_thread();
            publish_or_discard(
                DestructionGate {
                    entered,
                    release: wait,
                },
                reply,
                &AtomicBool::new(true),
                busy,
            );
        });
        observe.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!done.load(Ordering::Acquire));
        assert_eq!(count.load(Ordering::Acquire), 1);
        assert!(matches!(receive.try_recv(), Err(mpsc::TryRecvError::Empty)));
        release.send(()).unwrap();
        worker.join().unwrap();
        assert!(done.load(Ordering::Acquire));
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert!(matches!(
            receive.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }
    #[test]
    fn active_native_handles_have_raii_count_and_byte_bounds() {
        let budget = Arc::new(Budget::default());
        assert!(Lease::new(&budget, 0).is_err());
        assert!(Lease::new(&budget, (128 << 20) + 1).is_err());
        let mut leases = Vec::new();
        for _ in 0..16 {
            leases.push(Lease::new(&budget, 1).unwrap());
        }
        assert!(Lease::new(&budget, 1).is_err());
        assert_eq!(budget.bytes.load(Ordering::Acquire), 16);
        leases.clear();
        assert_eq!(budget.owners.load(Ordering::Acquire), 0);
        for _ in 0..4 {
            leases.push(Lease::new(&budget, 128 << 20).unwrap());
        }
        assert!(Lease::new(&budget, 1).is_err());
        assert_eq!(budget.owners.load(Ordering::Acquire), 4);
        drop(leases);
        assert_eq!(budget.bytes.load(Ordering::Acquire), 0);
    }
    #[test]
    fn cancelled_receipt_waits_for_actual_worker_exit_and_drains_queued_result() {
        let (send, receive) = mpsc::sync_channel(1);
        let done = Arc::new(AtomicBool::new(false));
        let mut p = Pending {
            reply: receive,
            cancel: Arc::new(AtomicBool::new(false)),
            taken: false,
            done: done.clone(),
            received: None,
        };
        assert!(p.retired().is_err());
        p.cancel();
        assert!(!p.retired().unwrap());
        assert!(send.send(Err(anyhow::anyhow!("compile error"))).is_ok());
        assert!(!p.retired().unwrap());
        drop(send);
        done.store(true, Ordering::Release);
        assert!(p.retired().unwrap());
        assert!(p.taken);
        assert!(p.retired().unwrap());
        assert!(p.poll(Duration::ZERO).is_err());
    }
    #[test]
    fn cancelled_before_execution_requires_closed_reply_and_done_flag() {
        let (send, receive) = mpsc::sync_channel(1);
        let done = Arc::new(AtomicBool::new(false));
        let mut p = Pending {
            reply: receive,
            cancel: Arc::new(AtomicBool::new(true)),
            taken: false,
            done: done.clone(),
            received: None,
        };
        drop(send);
        assert!(!p.retired().unwrap());
        done.store(true, Ordering::Release);
        assert!(p.retired().unwrap());
    }
    #[test]
    fn reply_is_not_published_before_actual_job_capacity_is_released() {
        for cancelled in [false, true] {
            let (send, receive) = mpsc::sync_channel(1);
            let done = Arc::new(AtomicBool::new(false));
            let mut p = Pending {
                reply: receive,
                cancel: Arc::new(AtomicBool::new(false)),
                taken: false,
                done: done.clone(),
                received: None,
            };
            assert!(send.send(Err(anyhow::anyhow!("result is ready"))).is_ok());
            assert!(p.poll(Duration::ZERO).unwrap().is_none());
            assert!(p.received.is_some() && !p.taken);
            if cancelled {
                p.cancel();
                assert!(!p.retired().unwrap());
            }
            drop(send);
            done.store(true, Ordering::Release);
            if cancelled {
                assert!(p.retired().unwrap());
            } else {
                assert!(p.poll(Duration::ZERO).unwrap().unwrap().is_err());
            }
            assert!(p.taken && p.received.is_none());
        }
    }
}
