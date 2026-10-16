//! Collective deferred grammar admission and serial mask execution.
use super::*;
use crate::{
    grammar_native::Native,
    grammar_prepare::{LeasedMatcher, Pending, Preparer, Ready},
    grammar_serial::{Serial, Stage},
    grammar_vocab::Views,
    output_policy::Policy,
};
const PREPARATIONS: usize = 4;
pub(super) struct Submission {
    pub(super) tokens: Arc<[i32]>,
    pub(super) output: usize,
    pub(super) background: bool,
    pub(super) deadline: Duration,
    pub(super) policy: Option<Policy>,
}
struct Staged {
    tokens: Arc<[i32]>,
    output: usize,
    background: bool,
    deadline: Duration,
    policy: Policy,
    bound: [u8; 32],
    policy_digest: [u8; 32],
    job: Pending,
    ready: Option<Ready>,
    cancelling: Option<String>,
    cancelled_at: Option<Duration>,
    invalid_grammar: bool,
}
pub(super) struct State {
    preparer: Preparer,
    serial: Serial<LeasedMatcher>,
    staged: BTreeMap<u64, Staged>,
    identity: [u8; 32],
    pub(super) stopping: bool,
    masked: Option<crate::masked_candidates::MaskedCandidates>,
    masked_calls: u64,
}
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Receipt {
    pub(crate) key: u64,
    pub(crate) submitted: Option<u64>,
    pub(crate) rejected: Option<String>,
    pub(crate) invalid_grammar: bool,
}
impl State {
    pub(super) fn new(native: Native, views: Arc<Views>, library_sha256: &str) -> Result<Self> {
        ensure!(
            library_sha256.len() == 64
                && library_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "grammar library pin encoding"
        );
        let mut hash = Sha256::new();
        hash.update(b"alaska-native-grammar-serving-v1");
        hash.update(library_sha256.as_bytes());
        hash.update(views.identity);
        Ok(Self {
            preparer: Preparer::new(native, views)?,
            serial: Serial::new(4)?,
            staged: BTreeMap::new(),
            identity: hash.finalize().into(),
            stopping: false,
            masked: None,
            masked_calls: 0,
        })
    }
    fn stage(&mut self, key: u64, submission: Submission, now: Duration) -> Result<()> {
        let Submission {
            tokens,
            output,
            background,
            deadline,
            policy,
        } = submission;
        let policy = policy.context("grammar policy absent")?;
        policy.validate()?;
        ensure!(
            !self.stopping
                && key != 0
                && self.staged.len() < PREPARATIONS
                && !self.staged.contains_key(&key),
            "grammar preparation capacity/key"
        );
        ensure!(
            !tokens.is_empty()
                && output > 0
                && output <= crate::cache_pages::MAX_OUTPUT
                && tokens.len() <= crate::cache_pages::CONTEXT - output
                && tokens.iter().all(|&t| crate::vision_input::prompt_token(t))
                && deadline > now,
            "grammar submission bounds/deadline"
        );
        let bound = policy
            .grammar()
            .context("grammar submission has no constraint")?
            .clone();
        let identity = bound.digest();
        let policy_digest = policy.digest();
        let job = self
            .preparer
            .start(bound)
            .map_err(|_| anyhow::anyhow!("grammar compiler capacity unavailable"))?;
        self.staged.insert(
            key,
            Staged {
                tokens,
                output,
                background,
                deadline,
                policy,
                bound: identity,
                policy_digest,
                job,
                ready: None,
                cancelling: None,
                cancelled_at: None,
                invalid_grammar: false,
            },
        );
        Ok(())
    }
    fn cancel(&mut self, key: u64, message: &str, now: Duration) -> Result<()> {
        let staged = self
            .staged
            .get_mut(&key)
            .context("missing grammar preparation")?;
        staged.cancelling.get_or_insert_with(|| message.into());
        staged.cancelled_at.get_or_insert(now);
        staged.ready.take();
        staged.job.cancel();
        Ok(())
    }
    fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.identity);
        hash.update(self.serial.digest());
        hash.update([u8::from(self.stopping)]);
        for (&key, s) in &self.staged {
            hash.update(key.to_le_bytes());
            hash.update(s.bound);
            hash.update(s.policy_digest);
            hash.update((s.output as u64).to_le_bytes());
            hash.update(s.deadline.as_nanos().to_le_bytes());
            hash.update([u8::from(s.background), u8::from(s.cancelling.is_some())]);
            hash.update(s.cancelled_at.map_or(0, |t| t.as_nanos()).to_le_bytes());
        }
        hash.finalize().into()
    }
    fn wait(&self) -> Duration {
        if !self.staged.is_empty() || self.serial.in_flight() != 0 || self.stopping {
            Duration::from_millis(1)
        } else {
            Duration::from_secs(1)
        }
    }
    fn empty(&self) -> bool {
        self.staged.is_empty() && self.serial.count() == 0
    }
}
impl Scheduler {
    pub(super) fn install_grammar(
        &mut self,
        long: &Long<'_>,
        mut state: State,
        native: &Path,
    ) -> Result<()> {
        ensure!(
            self.grammar.is_none() && self.queue.count() == 0 && state.empty(),
            "grammar installation after admission"
        );
        let sampler = crate::masked_candidates::MaskedCandidates::new(&long.engine.device, native);
        ensure!(
            self.wire
                .control
                .agree(long, 850, state.identity, sampler.is_ok())?,
            "grammar native library/tokenizer identity differs"
        );
        state.masked = Some(sampler?);
        self.grammar = Some(state);
        Ok(())
    }
    pub(super) fn grammar_wait(&self) -> Duration {
        self.grammar
            .as_ref()
            .map_or(Duration::from_secs(1), State::wait)
    }
    pub(super) fn grammar_digest(&self) -> [u8; 32] {
        self.grammar.as_ref().map_or([0; 32], State::digest)
    }
    pub(super) fn grammar_remove(&mut self, id: u64) {
        if let Some(grammar) = &mut self.grammar {
            grammar.serial.remove(id);
        }
    }
    pub(super) fn grammar_pending(&self) -> bool {
        self.grammar.as_ref().is_some_and(|g| !g.staged.is_empty())
    }
    pub(super) fn grammar_stage(
        &mut self,
        long: &Long<'_>,
        key: u64,
        submission: Submission,
    ) -> Result<bool> {
        let result = (|| {
            ensure!(
                !self.audit || (submission.tokens.len() <= 16384 && submission.output <= 64),
                "diagnostic request limit"
            );
            self.grammar
                .as_mut()
                .context("native grammar preparation unavailable")?
                .stage(key, submission, self.stamp)
        })();
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALGMSTG1");
        record[8..16].copy_from_slice(&key.to_le_bytes());
        record[16] = u8::from(result.is_ok());
        let all = self.wire.control.exchange(long, &record)?;
        ensure!(
            all.iter()
                .all(|r| r[..16] == record[..16] && r[16] == record[16]),
            "grammar preparation start differed between ranks"
        );
        if result.is_err() {
            return Ok(false);
        }
        result?;
        Ok(true)
    }
    pub(super) fn grammar_cancel(&mut self, long: &Long<'_>, key: u64) -> Result<()> {
        let cancelled = self
            .grammar
            .as_mut()
            .context("grammar unavailable")
            .and_then(|g| g.cancel(key, "grammar preparation cancelled", self.stamp));
        let mut identity = [0; 32];
        identity[..8].copy_from_slice(&key.to_le_bytes());
        ensure!(
            self.wire
                .control
                .agree(long, 857, identity, cancelled.is_ok())?,
            "grammar preparation cancellation failed on a rank"
        );
        cancelled
    }
    /// One completion per tick preserves the broker's bounded receipt protocol.
    pub(super) fn grammar_poll(
        &mut self,
        long: &Long<'_>,
        now: Duration,
    ) -> Result<Option<Receipt>> {
        let Some(grammar) = &mut self.grammar else {
            return Ok(None);
        };
        let keys: Vec<u64> = grammar.staged.keys().copied().collect();
        for key in keys {
            if now >= grammar.staged[&key].deadline {
                grammar.cancel(key, "grammar preparation deadline elapsed", now)?;
            }
            let s = grammar.staged.get_mut(&key).unwrap();
            let state = if s.cancelling.is_some() {
                match s.job.retired() {
                    Ok(true) => 3,
                    Ok(false)
                        if s.cancelled_at
                            .is_some_and(|at| now.saturating_sub(at) > Duration::from_secs(5)) =>
                    {
                        4
                    }
                    Ok(false) => 0,
                    Err(_) => 2,
                }
            } else if s.ready.is_some() {
                1
            } else {
                match s.job.poll(Duration::ZERO) {
                    Ok(None) => 0,
                    Ok(Some(Ok(ready))) => {
                        s.ready = Some(ready);
                        1
                    }
                    Ok(Some(Err(_))) | Err(_) => 2,
                }
            };
            let mut record = [0u8; 64];
            record[..8].copy_from_slice(b"ALGRAM01");
            record[8..16].copy_from_slice(&key.to_le_bytes());
            record[16..48].copy_from_slice(&s.bound);
            record[48] = state;
            record[49] = u8::from(s.cancelling.is_some());
            let all = self.wire.control.exchange(long, &record)?;
            ensure!(
                all.iter()
                    .all(|r| r[..48] == record[..48] && r[48] <= 3 && r[49] == record[49]),
                "grammar preparation ownership differs"
            );
            if all.iter().any(|r| r[48] == 2) {
                ensure!(
                    s.cancelling.is_none(),
                    "grammar compiler retirement failed on a rank"
                );
                s.invalid_grammar = true;
                grammar.cancel(key, "request grammar cannot be enforced", now)?;
                continue;
            }
            if all.iter().all(|r| r[48] == 3) {
                let s = grammar.staged.remove(&key).unwrap();
                return Ok(Some(Receipt {
                    key,
                    submitted: None,
                    rejected: s.cancelling,
                    invalid_grammar: s.invalid_grammar,
                }));
            }
            if !all.iter().all(|r| r[48] == 1) {
                continue;
            }
            let mut s = grammar.staged.remove(&key).unwrap();
            let ready = s.ready.take();
            ensure!(
                self.wire.control.agree(
                    long,
                    853,
                    s.bound,
                    ready
                        .as_ref()
                        .is_some_and(|r| r.bound_digest() == s.bound && r.bytes() > 0)
                )?,
                "compiled grammar identity differs"
            );
            let ready = ready.unwrap();
            let stream = (|| -> Result<_> {
                Ok(if s.policy.has_stops() {
                    Some(crate::output::Stream::new(
                        self.output_table
                            .clone()
                            .context("stop decoder unavailable")?,
                        s.policy.thinking(),
                        s.policy.tools(),
                        &s.policy.stops(),
                        s.output,
                    )?)
                } else {
                    None
                })
            })();
            ensure!(
                self.wire
                    .control
                    .agree(long, 854, s.policy_digest, stream.is_ok())?,
                "compiled grammar output stream unavailable on a rank"
            );
            let stream = stream?;
            let submitted = self.queue.submit_policy(
                s.tokens,
                s.output,
                s.background,
                s.deadline,
                now,
                Some(&s.policy),
            );
            let mut record = [0; 64];
            record[..8].copy_from_slice(b"ALGMQAD1");
            record[8..16].copy_from_slice(&key.to_le_bytes());
            record[16..24].copy_from_slice(&submitted.as_ref().copied().unwrap_or(0).to_le_bytes());
            record[24] = u8::from(submitted.is_ok());
            let all = self.wire.control.exchange(long, &record)?;
            ensure!(
                all.iter().all(|r| r[..25] == record[..25]),
                "compiled grammar queue admission differs"
            );
            match submitted {
                Ok(id) => {
                    let installed = ready
                        .bind(id, s.output)
                        .and_then(|owner| grammar.serial.insert(id, owner));
                    ensure!(
                        self.wire
                            .control
                            .agree(long, 856, s.bound, installed.is_ok())?,
                        "compiled grammar owner binding failed on a rank"
                    );
                    installed?;
                    if let Some(stream) = stream {
                        self.output_streams.insert(id, stream);
                    }
                    return Ok(Some(Receipt {
                        key,
                        submitted: Some(id),
                        rejected: None,
                        invalid_grammar: false,
                    }));
                }
                Err(error) => {
                    return Ok(Some(Receipt {
                        key,
                        submitted: None,
                        rejected: Some(error.to_string()),
                        invalid_grammar: false,
                    }));
                }
            }
        }
        Ok(None)
    }
    /// Must precede model execution, while logits still belong to the selected
    /// request. Worker timing is allowed to differ; common ready masks are not.
    pub(super) fn grammar_mask_ready(&mut self, long: &Long<'_>, id: u64) -> Result<bool> {
        let status = (|| -> Result<_> {
            let g = self
                .grammar
                .as_mut()
                .context("constrained request lost grammar manager")?;
            ensure!(
                g.masked.is_some(),
                "constrained request lost masked sampler"
            );
            if g.serial.status(id)?.stage == Stage::Idle {
                g.serial.begin(id)?;
            }
            g.serial.poll(id)
        })();
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALGMASK1");
        record[8..16].copy_from_slice(&id.to_le_bytes());
        record[16..48].copy_from_slice(&status.as_ref().map_or([0; 32], |s| s.progress));
        record[48] = match status.as_ref().map(|s| s.stage) {
            Ok(Stage::Idle | Stage::Pending) => 0,
            Ok(Stage::Ready) => 1,
            _ => 2,
        };
        let all = self.wire.control.exchange(long, &record)?;
        ensure!(
            all.iter().all(|r| r[..48] == record[..48] && r[48] <= 1),
            "grammar mask failure or ownership disagreement"
        );
        if !all.iter().all(|r| r[48] == 1) {
            return Ok(false);
        }
        let status = status?;
        ensure!(
            self.wire.control.agree(
                long,
                852,
                status.mask.unwrap_or([0; 32]),
                status.mask.is_some()
            )?,
            "grammar masks differ"
        );
        Ok(true)
    }
    pub(super) fn grammar_sample(
        &mut self,
        long: &mut Long<'_>,
        id: u64,
        policy: Option<crate::sampling::Sampling>,
        fallback: i32,
    ) -> Result<i32> {
        let grammar = self.grammar.as_mut().context("grammar unavailable")?;
        let mask = grammar.serial.mask(id)?;
        if let Some(mask) = mask {
            let policy = policy.unwrap_or(crate::sampling::Sampling::new(0, 0., 20, 0.95)?);
            let next_call = grammar
                .masked_calls
                .checked_add(1)
                .context("grammar masked call counter overflow")?;
            let token = long
                .sample_masked(
                    grammar
                        .masked
                        .as_mut()
                        .context("masked sampler unavailable")?,
                    policy,
                    mask,
                )
                .map(|r| r.0 as i32)?;
            grammar.masked_calls = next_call;
            return Ok(token);
        }
        if let Some(policy) = policy.filter(|s| !s.greedy()) {
            long.sample(
                self.sampler.as_mut().context("sampler unavailable")?,
                policy,
            )
            .map(|r| r.0 as i32)
        } else {
            Ok(fallback)
        }
    }
    pub(super) fn grammar_commit(&mut self, id: u64, token: i32, terminal: bool) -> Result<()> {
        self.grammar
            .as_mut()
            .context("grammar unavailable")?
            .serial
            .commit(id, token, terminal)
    }
    pub(super) fn grammar_drain_masks(&mut self, long: &Long<'_>) -> Result<()> {
        let Some(g) = &self.grammar else {
            return Ok(());
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut record = [0; 64];
            record[..8].copy_from_slice(b"ALGMDRA1");
            record[8] = u8::from(g.serial.in_flight() == 0);
            record[9] = u8::from(Instant::now() < deadline);
            let all = self.wire.control.exchange(long, &record)?;
            ensure!(
                all.iter().all(|r| r[..8] == record[..8] && r[9] == 1),
                "grammar hold drain deadline"
            );
            if all.iter().all(|r| r[8] == 1) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    pub(super) fn grammar_stop(&mut self, long: &Long<'_>) -> Result<bool> {
        let Some(g) = &mut self.grammar else {
            return Ok(true);
        };
        let retired = (|| -> Result<bool> {
            ensure!(g.empty(), "grammar owners remain at stop");
            if !g.stopping {
                g.preparer.close();
                g.serial.close();
                g.stopping = true;
            }
            let compiler = g.preparer.retired()?;
            let masks = g.serial.retired()?;
            Ok(compiler && masks)
        })();
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALGMSTOP");
        record[8] = match retired {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(_) => 2,
        };
        let all = self.wire.control.exchange(long, &record)?;
        ensure!(
            all.iter().all(|r| r[..8] == record[..8] && r[8] <= 1),
            "grammar retirement protocol differs"
        );
        Ok(all.iter().all(|r| r[8] == 1))
    }
    pub(super) fn grammar_telemetry(&self) -> serde_json::Value {
        self.grammar.as_ref().map_or(serde_json::Value::Null,|g|serde_json::json!({
            "preparing":g.staged.len(),"compile_in_flight":g.preparer.in_flight(),
            "active_owners":g.preparer.active_owners(),"active_bytes":g.preparer.active_bytes(),
            "compiler_cache_bytes":g.preparer.cached_bytes(),"mask_in_flight":g.serial.in_flight(),
            "requests":g.serial.count(),"stopping":g.stopping,"masked_calls":g.masked_calls
        }))
    }
}
