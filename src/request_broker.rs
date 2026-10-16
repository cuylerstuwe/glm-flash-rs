//! Bounded CPU ingress. No CUDA values cross this interface. Disconnect is a
//! flag on a reserved entry, so a full control queue cannot lose cancellation.
use crate::{
    cache_pages::{CONTEXT, MAX_OUTPUT},
    request_queue::{End, REQUESTS},
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    time::{Duration, Instant},
};
pub(crate) const OUTPUT_BUFFER: usize = 256;
const CONTROLS: usize = 32;
#[derive(Clone)]
pub(crate) enum Command {
    Tick,
    Prepare {
        background: bool,
    },
    Release(u64),
    Stop,
    Submit {
        tokens: Arc<[i32]>,
        output: usize,
        background: bool,
        deadline: Duration,
        policy: Option<crate::output_policy::Policy>,
    },
    SubmitGrammar {
        key: u64,
        tokens: Arc<[i32]>,
        output: usize,
        background: bool,
        deadline: Duration,
        policy: Option<crate::output_policy::Policy>,
    },
    CancelGrammar(u64),
    Cancel(u64),
    Lease {
        owner: String,
        epoch: u64,
        foreground: bool,
        seconds: f64,
    },
}
#[derive(Clone, Debug)]
pub(crate) struct Receipt {
    pub applied: u64,
    pub held: bool,
}
#[derive(Debug)]
pub(crate) struct ControlFailure {
    pub rejected: bool,
    message: String,
}
impl std::fmt::Display for ControlFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl From<&str> for ControlFailure {
    fn from(message: &str) -> Self {
        Self {
            rejected: false,
            message: message.into(),
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Token(i32),
    End(End),
}
#[derive(Debug)]
pub(crate) struct InvalidGrammar;
impl std::fmt::Display for InvalidGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("request grammar cannot be enforced")
    }
}
impl std::error::Error for InvalidGrammar {}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    New,
    Preparing,
    Ready,
    Compiling,
    Submitted,
    Retired,
}
struct Entry {
    background: bool,
    deadline: Duration,
    phase: Phase,
    connected: bool,
    ticket: Option<u64>,
    request: Option<u64>,
    assigned: Option<u64>,
    input: Option<(Arc<[i32]>, usize, Option<crate::output_policy::Policy>)>,
    output: VecDeque<i32>,
    end: Option<End>,
    error: Option<String>,
    invalid_grammar: bool,
    cancel: bool,
    // Cancellation may only begin asynchronous native retirement (e.g. SSD
    // restore). Retain the request but do not flood control-only Cancel ticks.
    cancel_applied: bool,
    receipt: Option<Receipt>,
    timed_out: bool,
    cached_tokens: usize,
    /// Rank 0's image rows for this request's prompt, if it has images.
    vision: Option<Arc<crate::vision_service::Table>>,
    image_preparation: Option<Arc<crate::vision_request::Control>>,
}
struct Control {
    command: Command,
    reply: SyncSender<std::result::Result<Receipt, ControlFailure>>,
    expires: Instant,
}
struct State {
    file_ready: bool,
    entries: BTreeMap<u64, Entry>,
    controls: VecDeque<Control>,
    serial: u64,
    stopping: bool,
    ready: bool,
    failed: Option<String>,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    start: Instant,
    telemetry: Mutex<serde_json::Value>,
    images: Arc<crate::image_gate::Gate>,
}
#[derive(Clone)]
pub(crate) struct Broker {
    shared: Arc<Shared>,
}
pub(crate) struct Client {
    broker: Broker,
    id: u64,
}
#[derive(Clone)]
pub(crate) struct Cancellation {
    broker: Broker,
    id: u64,
}
impl Cancellation {
    pub fn cancel(&self) {
        let mut s = self.broker.shared.state.lock().unwrap();
        if let Some(e) = s.entries.get_mut(&self.id) {
            e.cancel = true;
            if let Some(c) = &e.image_preparation {
                c.cancel();
            }
        }
        self.broker.shared.changed.notify_all();
    }
}
pub(crate) struct Owner {
    broker: Broker,
}
/// Coalesced CPU notification only. It neither admits CUDA work nor bypasses
/// the normal native priority/cancellation/collective boundary.
#[derive(Clone)]
pub(crate) struct FileWake(std::sync::Weak<Shared>);
impl FileWake {
    pub fn notify(&self) {
        if let Some(shared) = self.0.upgrade() {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.file_ready = true;
            shared.changed.notify_all();
        }
    }
}
pub(crate) enum Action {
    Prepare(u64, Command),
    Submit(u64, Command),
    Release(u64, Command),
    Cancel(u64, Command),
    CancelGrammar(u64, Command),
    Control(
        Command,
        SyncSender<std::result::Result<Receipt, ControlFailure>>,
    ),
    Tick,
    Stop,
}
impl Action {
    pub fn command(&self) -> Command {
        match self {
            Self::Prepare(_, c)
            | Self::Submit(_, c)
            | Self::Release(_, c)
            | Self::Cancel(_, c)
            | Self::CancelGrammar(_, c)
            | Self::Control(c, _) => c.clone(),
            Self::Tick => Command::Tick,
            Self::Stop => Command::Stop,
        }
    }
    pub fn control_only(&self) -> bool {
        !matches!(self, Self::Tick)
    }
}
impl Broker {
    pub fn new() -> (Self, Owner) {
        let b = Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    file_ready: false,
                    entries: BTreeMap::new(),
                    controls: VecDeque::new(),
                    serial: 0,
                    stopping: false,
                    ready: true,
                    failed: None,
                }),
                changed: Condvar::new(),
                start: Instant::now(),
                telemetry: Mutex::new(serde_json::Value::Null),
                images: Arc::new(crate::image_gate::Gate::default()),
            }),
        };
        (b.clone(), Owner { broker: b })
    }
    /// Rank 0's image rows for a scheduler request: its entry records the
    /// request ID once the submission is applied; before that, in the same
    /// step, it still holds the submitted prompt (virtual row IDs derive from
    /// image content, so any entry with this exact prompt carries equal rows).
    pub fn vision(
        &self,
        request: u64,
        tokens: &[i32],
    ) -> Option<Arc<crate::vision_service::Table>> {
        let s = self.shared.state.lock().unwrap();
        s.entries
            .values()
            .filter(|e| e.vision.is_some())
            .find(|e| {
                e.request == Some(request)
                    || e.input.as_ref().is_some_and(|i| i.0.as_ref() == tokens)
            })
            .and_then(|e| e.vision.clone())
    }
    pub fn ready(&self, ready: bool) {
        self.shared.state.lock().unwrap().ready = ready;
    }
    pub fn now(&self) -> Duration {
        self.shared.start.elapsed()
    }
    pub fn status(&self) -> serde_json::Value {
        let s = self.shared.state.lock().unwrap();
        serde_json::json!({"available":s.ready && !s.stopping && s.failed.is_none(), "clients":s.entries.len(), "pending_controls":s.controls.len(), "failed":s.failed.is_some(), "uptime_ns":self.now().as_nanos(), "native":self.shared.telemetry.lock().unwrap().clone(), "images":self.shared.images.status()})
    }
    /// Reserve BEFORE allocating a body or starting template/tokenizer work.
    pub fn begin(&self, background: bool, timeout: Duration) -> Result<Client> {
        ensure!(
            !timeout.is_zero() && timeout <= Duration::from_secs(86400),
            "request timeout bounds"
        );
        let mut s = self.shared.state.lock().unwrap();
        ensure!(!s.stopping && s.failed.is_none(), "owner unavailable");
        ensure!(
            s.entries.len() < REQUESTS - usize::from(background),
            "ingress capacity / foreground reservation"
        );
        let id = s.serial.checked_add(1).context("ingress IDs exhausted")?;
        s.serial = id;
        s.entries.insert(
            id,
            Entry {
                invalid_grammar: false,
                background,
                deadline: self.now() + timeout,
                phase: Phase::New,
                connected: true,
                ticket: None,
                request: None,
                assigned: None,
                input: None,
                output: VecDeque::with_capacity(OUTPUT_BUFFER),
                end: None,
                error: None,
                cancel: false,
                cancel_applied: false,
                receipt: None,
                timed_out: false,
                cached_tokens: 0,
                vision: None,
                image_preparation: None,
            },
        );
        self.shared.changed.notify_all();
        Ok(Client {
            broker: self.clone(),
            id,
        })
    }
    /// Trusted internal control; future HTTP adapter must authenticate first.
    pub fn lease(
        &self,
        owner: &str,
        epoch: u64,
        foreground: bool,
        seconds: f64,
    ) -> Result<Receiver<std::result::Result<Receipt, ControlFailure>>> {
        ensure!(
            !owner.is_empty()
                && owner.chars().count() <= 160
                && seconds.is_finite()
                && (1.0..=300.0).contains(&seconds),
            "invalid lease"
        );
        let mut s = self.shared.state.lock().unwrap();
        ensure!(!s.stopping && s.failed.is_none(), "owner unavailable");
        ensure!(s.controls.len() < CONTROLS, "control queue full");
        let (tx, rx) = mpsc::sync_channel(1);
        s.controls.push_back(Control {
            command: Command::Lease {
                owner: owner.into(),
                epoch,
                foreground,
                seconds,
            },
            reply: tx,
            expires: Instant::now() + Duration::from_secs(25),
        });
        self.shared.changed.notify_all();
        Ok(rx)
    }
    /// Stops admission and cancels all owned lifetimes; owner still drains releases.
    pub fn shutdown(&self) {
        let mut s = self.shared.state.lock().unwrap();
        s.stopping = true;
        for c in s.controls.drain(..) {
            let _ = c.reply.try_send(Err("owner shutting down".into()));
        }
        for e in s.entries.values_mut() {
            e.cancel = true;
            if let Some(c) = &e.image_preparation {
                c.cancel();
            }
        }
        self.shared.changed.notify_all();
    }
}
pub(crate) struct ImagePreparation {
    broker: Broker,
    id: u64,
    pub job: Option<crate::image_gate::Job>,
}
impl Drop for ImagePreparation {
    fn drop(&mut self) {
        drop(self.job.take());
        let mut s = self.broker.shared.state.lock().unwrap();
        if let Some(e) = s.entries.get_mut(&self.id) {
            e.image_preparation = None;
        }
        self.broker.shared.changed.notify_all();
    }
}
impl Client {
    pub fn prepare_image(&self) -> Result<ImagePreparation> {
        let mut s = self.broker.shared.state.lock().unwrap();
        let e = s
            .entries
            .get_mut(&self.id)
            .context("retired image client")?;
        ensure!(
            !e.cancel && e.phase == Phase::Preparing && e.image_preparation.is_none(),
            "image preparation unavailable"
        );
        let job = self.broker.shared.images.register(self.id, e.background)?;
        e.image_preparation = Some(job.control.clone());
        Ok(ImagePreparation {
            broker: self.broker.clone(),
            id: self.id,
            job: Some(job),
        })
    }
    pub fn cancellation(&self) -> Cancellation {
        Cancellation {
            broker: self.broker.clone(),
            id: self.id,
        }
    }
    /// Foreground callers must wait for the native drain before preparation.
    pub fn prepared(&self, timeout: Duration) -> Result<Receipt> {
        let end = Instant::now() + timeout;
        let mut s = self.broker.shared.state.lock().unwrap();
        loop {
            let e = s.entries.get_mut(&self.id).context("retired client")?;
            if self.broker.now() >= e.deadline && !e.cancel {
                e.cancel = true;
                e.timed_out = true;
                self.broker.shared.changed.notify_all();
            }
            ensure!(
                e.error.is_none() && !e.cancel,
                "preparation cancelled: {:?}",
                e.error
            );
            if let Some(receipt) = &e.receipt {
                return Ok(receipt.clone());
            }
            let wait = end.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                e.cancel = true;
                self.broker.shared.changed.notify_all();
                anyhow::bail!("preparation wait timed out");
            }
            s = self.broker.shared.changed.wait_timeout(s, wait).unwrap().0;
        }
    }
    /// Transfers immutable packed tokens. The body/tokenizer allocation has its
    /// own adapter limit; this component accepts no text or unbounded event data.
    pub fn submit(&self, tokens: Arc<[i32]>, output: usize) -> Result<()> {
        self.submit_policy(tokens, output, None)
    }
    pub fn submit_policy(
        &self,
        tokens: Arc<[i32]>,
        output: usize,
        policy: Option<crate::output_policy::Policy>,
    ) -> Result<()> {
        let invalid = tokens.is_empty()
            || output == 0
            || output > MAX_OUTPUT
            || tokens.len() > CONTEXT - output
            || tokens
                .iter()
                .any(|&t| !crate::vision_input::prompt_token(t))
            || policy.as_ref().is_some_and(|p| p.validate().is_err());
        let mut s = self.broker.shared.state.lock().unwrap();
        let e = s.entries.get_mut(&self.id).context("retired client")?;
        if invalid {
            e.cancel = true;
            e.error = Some("invalid input/output bounds or policy".into());
            self.broker.shared.changed.notify_all();
            anyhow::bail!("invalid input/output bounds or policy");
        }
        ensure!(
            e.phase == Phase::Preparing
                && e.receipt.is_some()
                && !e.cancel
                && self.broker.now() < e.deadline,
            "request not prepared or expired"
        );
        e.input = Some((tokens, output, policy));
        e.phase = Phase::Ready;
        self.broker.shared.changed.notify_all();
        Ok(())
    }
    /// The prompt's image rows; must precede submission.
    pub fn attach_vision(&self, table: Arc<crate::vision_service::Table>) -> Result<()> {
        let mut s = self.broker.shared.state.lock().unwrap();
        let e = s.entries.get_mut(&self.id).context("retired client")?;
        ensure!(e.input.is_none(), "vision attached after submission");
        e.vision = Some(table);
        Ok(())
    }
    /// Stable process-local scheduler identity, only after common submission.
    pub fn submitted(&self, timeout: Duration) -> Result<u64> {
        let end = Instant::now() + timeout;
        let mut s = self.broker.shared.state.lock().unwrap();
        loop {
            let e = s.entries.get_mut(&self.id).context("retired client")?;
            if let Some(id) = e.assigned {
                return Ok(id);
            }
            if e.invalid_grammar {
                return Err(InvalidGrammar.into());
            }
            ensure!(
                e.error.is_none() && !e.cancel,
                "submission cancelled: {:?}",
                e.error
            );
            let wait = end.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                e.cancel = true;
                self.broker.shared.changed.notify_all();
                anyhow::bail!("submission wait timed out");
            }
            s = self.broker.shared.changed.wait_timeout(s, wait).unwrap().0;
        }
    }
    /// A timeout is a keepalive opportunity, not an inference cancellation.
    pub fn receive(&self, timeout: Duration) -> Result<Option<Event>> {
        let end = Instant::now() + timeout;
        let mut s = self.broker.shared.state.lock().unwrap();
        loop {
            let e = s.entries.get_mut(&self.id).context("retired client")?;
            if let Some(t) = e.output.pop_front() {
                return Ok(Some(Event::Token(t)));
            }
            if let Some(error) = &e.error {
                anyhow::bail!("{error}");
            }
            if let Some(reason) = e.end {
                return Ok(Some(Event::End(reason)));
            }
            let wait = end.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                return Ok(None);
            }
            s = self.broker.shared.changed.wait_timeout(s, wait).unwrap().0;
        }
    }
    pub fn cached_tokens(&self) -> Result<usize> {
        Ok(self
            .broker
            .shared
            .state
            .lock()
            .unwrap()
            .entries
            .get(&self.id)
            .context("retired client")?
            .cached_tokens)
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        let mut s = self.broker.shared.state.lock().unwrap();
        if let Some(e) = s.entries.get_mut(&self.id) {
            e.connected = false;
            e.cancel = true;
            if let Some(c) = &e.image_preparation {
                c.cancel();
            }
        }
        self.broker.shared.changed.notify_all();
    }
}
impl Owner {
    pub fn file_wake(&self) -> FileWake {
        FileWake(Arc::downgrade(&self.broker.shared))
    }
    pub fn image_boundary(&self, action: &Action) -> Result<()> {
        if matches!(
            action,
            Action::Prepare(_, Command::Prepare { background: false })
                | Action::Control(
                    Command::Lease {
                        foreground: true,
                        ..
                    },
                    _
                )
                | Action::Stop
        ) {
            self.broker.shared.images.quiesce(Duration::from_secs(25))?;
        }
        Ok(())
    }
    pub fn image_hold(&self, held: bool) -> Result<()> {
        self.broker.shared.images.set_hold(held)
    }
    pub fn telemetry(&self, value: serde_json::Value) {
        *self.broker.shared.telemetry.lock().unwrap() = value;
    }
    pub fn now(&self) -> Duration {
        self.broker.now()
    }
    /// One single consumer. Pending disconnect/release cannot be starved by
    /// flooded control traffic. Condvar releases the lock while idle.
    pub fn next(&self, wait: Duration) -> Action {
        let end = Instant::now() + wait;
        let mut s = self.broker.shared.state.lock().unwrap();
        loop {
            let now = self.now();
            for e in s.entries.values_mut() {
                if now >= e.deadline && e.end.is_none() && !e.cancel {
                    e.cancel = true;
                    e.timed_out = true;
                    if let Some(c) = &e.image_preparation {
                        c.cancel();
                    }
                }
            }
            s.entries.retain(|_, e| {
                e.connected
                    || e.ticket.is_some()
                    || e.request.is_some()
                    || e.phase == Phase::Compiling
                    || e.image_preparation.is_some()
            });
            for (&id, e) in &mut s.entries {
                if e.cancel && e.phase == Phase::Compiling {
                    return Action::CancelGrammar(id, Command::CancelGrammar(id));
                }
                if let Some(ticket) = e.ticket
                    && e.image_preparation.is_none()
                    && (e.cancel || e.phase == Phase::Submitted || e.phase == Phase::Retired)
                {
                    return Action::Release(id, Command::Release(ticket));
                }
                if e.cancel {
                    if let Some(c) = &e.image_preparation {
                        c.cancel();
                        continue;
                    }
                    e.input = None;
                    if let Some(request) = e.request {
                        if !e.timed_out && !e.cancel_applied {
                            return Action::Cancel(id, Command::Cancel(request));
                        }
                        // A deadline is propagated by the native scheduler;
                        // an acknowledged cancel is already sticky there.
                        // Allow foreground/control actions and ordinary ticks
                        // to drain the owned worker before publishing its end.
                        continue;
                    }
                    e.phase = Phase::Retired;
                    e.end.get_or_insert(if e.timed_out {
                        End::TimedOut
                    } else {
                        End::Cancelled
                    });
                }
            }
            // Foreground preparation is prioritized before leases and token work.
            if let Some((&id, e)) = s
                .entries
                .iter()
                .filter(|(_, e)| !e.cancel && e.phase == Phase::New)
                .min_by_key(|(id, e)| (e.background, **id))
            {
                return Action::Prepare(
                    id,
                    Command::Prepare {
                        background: e.background,
                    },
                );
            }
            while let Some(c) = s.controls.pop_front() {
                if Instant::now() >= c.expires {
                    let _ = c
                        .reply
                        .try_send(Err("control expired before application".into()));
                    continue;
                }
                return Action::Control(c.command, c.reply);
            }
            if let Some((&id, e)) = s
                .entries
                .iter()
                .filter(|(_, e)| !e.cancel && e.phase == Phase::Ready)
                .min_by_key(|(id, e)| (e.background, **id))
            {
                let (tokens, output, policy) = e.input.as_ref().unwrap();
                let command = if policy.as_ref().is_some_and(|p| p.grammar().is_some()) {
                    Command::SubmitGrammar {
                        key: id,
                        tokens: tokens.clone(),
                        output: *output,
                        background: e.background,
                        deadline: e.deadline,
                        policy: policy.clone(),
                    }
                } else {
                    Command::Submit {
                        tokens: tokens.clone(),
                        output: *output,
                        background: e.background,
                        deadline: e.deadline,
                        policy: policy.clone(),
                    }
                };
                return Action::Submit(id, command);
            }
            if s.stopping
                && s.entries.values().all(|e| {
                    e.ticket.is_none()
                        && e.request.is_none()
                        && e.phase != Phase::Compiling
                        && e.image_preparation.is_none()
                })
            {
                return Action::Stop;
            }
            if std::mem::take(&mut s.file_ready) {
                return Action::Tick;
            }
            let wait = end.saturating_duration_since(Instant::now()).min(
                s.entries
                    .values()
                    .filter(|e| e.end.is_none())
                    .map(|e| e.deadline.saturating_sub(now))
                    .min()
                    .unwrap_or(Duration::from_secs(1)),
            );
            self.broker.shared.changed.notify_all();
            if wait.is_zero() {
                return Action::Tick;
            }
            s = self.broker.shared.changed.wait_timeout(s, wait).unwrap().0;
        }
    }
    /// Called only AFTER the all-rank native proof, on a control-only scheduler
    /// tick. The reply is visible before the owner dispatches unrelated work.
    pub fn applied(
        &self,
        action: Action,
        prepared: Option<u64>,
        submitted: Option<u64>,
        receipt: Receipt,
        rejected: Option<String>,
    ) -> Result<()> {
        let mut s = self.broker.shared.state.lock().unwrap();
        match action {
            Action::Control(_, reply) => {
                let _ = reply.try_send(rejected.map_or(Ok(receipt), |message| {
                    Err(ControlFailure {
                        rejected: true,
                        message,
                    })
                }));
            }
            Action::Prepare(id, _) => {
                let e = s.entries.get_mut(&id).context("lost preparation owner")?;
                e.ticket = prepared;
                e.phase = Phase::Preparing;
                e.receipt = Some(receipt);
                if let Some(error) = rejected {
                    e.error = Some(error);
                    e.cancel = true;
                }
            }
            Action::Submit(id, _) => {
                let e = s.entries.get_mut(&id).context("lost submit owner")?;
                e.input = None;
                e.request = submitted;
                e.assigned = submitted;
                e.phase = Phase::Submitted;
                if let Some(error) = rejected {
                    e.error = Some(error);
                    e.cancel = true;
                }
            }
            Action::Release(id, _) => {
                ensure!(rejected.is_none(), "preparation release failed");
                s.entries.get_mut(&id).context("lost release owner")?.ticket = None;
            }
            Action::Cancel(id, _) => {
                if rejected.is_some() {
                    anyhow::bail!("request cancellation lost owner");
                }
                s.entries
                    .get_mut(&id)
                    .context("lost cancel owner")?
                    .cancel_applied = true;
            }
            Action::CancelGrammar(id, _) => {
                ensure!(
                    s.entries
                        .get(&id)
                        .is_some_and(|e| e.phase == Phase::Compiling),
                    "lost grammar cancellation owner"
                );
                ensure!(
                    rejected.is_none(),
                    "grammar cancellation lost native preparation"
                );
            }
            Action::Tick => {}
            Action::Stop => {
                ensure!(rejected.is_none(), "owner refused stop");
            }
        }
        self.broker.shared.changed.notify_all();
        Ok(())
    }
    /// Native compilation is deferred; retain the existing foreground ticket
    /// and broker entry until all ranks finish or cancel actual native work.
    pub(crate) fn defer_submission(&self, action: Action, key: u64) -> Result<()> {
        let Action::Submit(id, Command::SubmitGrammar { key: sent, .. }) = action else {
            anyhow::bail!("deferred receipt for non-grammar submission")
        };
        ensure!(
            id == key && sent == key,
            "deferred grammar preparation key differs"
        );
        let mut state = self.broker.shared.state.lock().unwrap();
        let e = state
            .entries
            .get_mut(&id)
            .context("lost deferred preparation owner")?;
        ensure!(e.phase == Phase::Ready, "grammar preparation phase changed");
        e.input = None;
        e.phase = Phase::Compiling;
        self.broker.shared.changed.notify_all();
        Ok(())
    }
    pub(crate) fn grammar_applied(
        &self,
        key: u64,
        submitted: Option<u64>,
        rejected: Option<String>,
        invalid_grammar: bool,
    ) -> Result<()> {
        ensure!(
            !invalid_grammar || rejected.is_some(),
            "invalid grammar receipt without rejection"
        );
        ensure!(
            submitted.is_some() != rejected.is_some(),
            "invalid grammar completion receipt"
        );
        let mut state = self.broker.shared.state.lock().unwrap();
        let e = state
            .entries
            .get_mut(&key)
            .context("lost compiled grammar owner")?;
        ensure!(
            e.phase == Phase::Compiling,
            "grammar completion phase changed"
        );
        e.request = submitted;
        e.assigned = submitted;
        e.phase = Phase::Submitted;
        e.invalid_grammar = invalid_grammar;
        if let Some(error) = rejected {
            e.error = Some(error);
            e.cancel = true;
        }
        self.broker.shared.changed.notify_all();
        Ok(())
    }
    /// Publishing never waits for a client. Overflow cancels just that request
    /// at the next common boundary, explicitly reporting a slow consumer.
    pub fn publish(&self, request: u64, token: Option<i32>, end: Option<End>) -> Result<()> {
        self.publish_many(request, token.as_slice(), end)
    }
    /// Publish one commonly committed boundary atomically. Overflow is an
    /// explicit request error, never an apparently successful partial batch.
    pub fn publish_many(&self, request: u64, tokens: &[i32], end: Option<End>) -> Result<()> {
        ensure!(
            tokens.len() <= crate::request_queue::DECODE_COMMIT_ROWS
                && tokens.iter().all(|t| (0..154880).contains(t)),
            "output publication bound"
        );
        let mut s = self.broker.shared.state.lock().unwrap();
        let e = s
            .entries
            .values_mut()
            .find(|e| e.request == Some(request))
            .context("unowned model output")?;
        if !tokens.is_empty() && !e.cancel {
            if e.output.len() + tokens.len() > OUTPUT_BUFFER {
                e.error = Some("output buffer full: slow consumer".into());
                e.cancel = true;
            } else {
                e.output.extend(tokens.iter().copied());
            }
        }
        if let Some(end) = end {
            e.request = None;
            e.end
                .get_or_insert(if e.timed_out { End::TimedOut } else { end });
            e.phase = Phase::Retired;
        }
        self.broker.shared.changed.notify_all();
        Ok(())
    }
    pub fn cached(&self, request: u64, tokens: usize) -> Result<()> {
        let mut s = self.broker.shared.state.lock().unwrap();
        let e = s
            .entries
            .values_mut()
            .find(|e| e.request == Some(request))
            .context("unowned cache admission")?;
        e.cached_tokens = tokens;
        Ok(())
    }
    pub fn fail(&self, error: &str) {
        let mut s = self.broker.shared.state.lock().unwrap();
        s.failed = Some(error.into());
        s.stopping = true;
        for e in s.entries.values_mut() {
            if e.end.is_none() {
                e.error = Some("model owner failed; request cannot resume".into());
                e.cancel = true;
                if let Some(c) = &e.image_preparation {
                    c.cancel();
                }
            }
        }
        for c in s.controls.drain(..) {
            let _ = c.reply.try_send(Err("model owner failed".into()));
        }
        self.broker.shared.changed.notify_all();
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.fail("owner stopped");
    }
}
#[cfg(test)]
#[path = "request_broker_tests.rs"]
mod tests;
