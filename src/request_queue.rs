//! Bounded request ownership and accounting, independent of CUDA and wall clocks.
//! The owner supplies monotonic time. GPU work is charged explicitly, never as
//! elapsed wall time between tokens (which would include native pauses).
use crate::{
    cache_pages::{CONTEXT, MAX_OUTPUT, SLOTS},
    priority::Gate,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::Duration,
};
pub(crate) const REQUESTS: usize = 16;
/// Maximum accepted inputs published by one target-verification round.
pub(crate) const DECODE_COMMIT_ROWS: usize = 16;
const HISTORY: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum End {
    Complete,
    Cancelled,
    TimedOut,
    Rejected,
    OutputError,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Completion {
    Length,
    Eos,
    Stop,
}
pub(crate) struct Request {
    pub id: u64,
    pub tokens: Arc<[i32]>,
    pub output: usize,
    pub background: bool,
    pub slot: Option<usize>,
    pub done: usize,
    pub cached_tokens: usize,
    pub restoring: bool,
    cancel_forwarded: bool,
    pub outputs: Vec<i32>,
    output_hash: Sha256,
    pub completion: Option<Completion>,
    policy: Option<[u8; 32]>,
    sampling: Option<crate::sampling::Sampling>,
    thinking: Option<crate::thinking_budget::State>,
    string_stops: bool,
    has_grammar: bool,
    drafts: bool,
    stop_eos: bool,
    pub paused: bool,
    cancel: Option<End>,
    ticket: Option<u64>,
    submitted: Duration,
    admitted: Option<Duration>,
    deadline: Duration,
    paused_at: Option<Duration>,
    pause_time: Duration,
    work_time: Duration,
    prefill_time: Duration,
    decode_time: Duration,
    prefill_calls: u64,
    pub decode_calls: u64,
    pub speculative_calls: u64,
    /// Computed decode inputs, excluding the pending token emitted by prefill.
    pub decode_tokens: usize,
    pause_count: u64,
    resume_count: u64,
    /// Pure-prefill positions that end a slab and save a snapshot, planned at
    /// the first prefill dispatch (None until then).
    pub marks: Option<Vec<usize>>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Status {
    pub id: u64,
    pub background: bool,
    pub slot: Option<usize>,
    pub phase: &'static str,
    pub prompt_tokens: usize,
    pub prefill_done: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    pub admission_count: u8,
    pub prefill_calls: u64,
    pub decode_calls: u64,
    pub speculative_calls: u64,
    pub decode_tokens: usize,
    pub pause_count: u64,
    pub resume_count: u64,
    pub queued_ns: u128,
    pub active_ns: u128,
    pub prefill_active_ns: u128,
    pub decode_active_ns: u128,
    pub paused_ns: u128,
    pub age_ns: u128,
}
#[derive(serde::Serialize)]
pub(crate) struct Completed {
    pub status: Status,
    pub reason: End,
    pub outputs: Vec<i32>,
    pub completion: Option<Completion>,
}
impl Request {
    pub fn stop_eos(&self) -> bool {
        self.stop_eos
    }
    pub fn stops_at(&self, token: i32) -> bool {
        self.stop_eos && crate::output_decode::EOS.contains(&(token as u32))
    }
    pub fn has_grammar(&self) -> bool {
        self.has_grammar
    }
    pub fn drafts(&self) -> bool {
        self.drafts
    }
    pub fn sampling(&self) -> Option<crate::sampling::Sampling> {
        self.sampling
    }
    /// The next pending output is selected only after its model input has
    /// completed. This state is owned by the request, including native holds.
    pub fn next_output(&self) -> crate::thinking_budget::Next {
        self.thinking
            .map_or(crate::thinking_budget::Next::Sample, |state| state.next())
    }
    /// Pure bounded preview over already target-accepted samples. Request state
    /// advances only in worked_many after the common native commit.
    pub fn sampled_prefix(&self, tokens: &[i32]) -> Result<usize> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= DECODE_COMMIT_ROWS
                && tokens.iter().all(|t| (0..154880).contains(t)),
            "sampled output preview bound"
        );
        let limit = tokens
            .len()
            .min(self.output.saturating_sub(self.outputs.len()));
        if let Some(state) = self.thinking {
            ensure!(
                state.emitted() == self.outputs.len(),
                "thinking/output preview cursor"
            );
            Ok(state
                .preview_sampled(&tokens[..limit], |id| self.stops_at(id))
                .map_err(anyhow::Error::msg)?
                .kept)
        } else {
            Ok(tokens[..limit]
                .iter()
                .position(|id| self.stops_at(*id))
                .map_or(limit, |at| at + 1))
        }
    }
    fn preview_outputs(&self, tokens: &[i32]) -> Result<Option<crate::thinking_budget::State>> {
        let mut next = self.thinking;
        if let Some(state) = &mut next {
            ensure!(
                state.emitted() == self.outputs.len(),
                "thinking/output cursor diverged"
            );
            for &token in tokens {
                state
                    .accept(token, self.stops_at(token))
                    .map_err(anyhow::Error::msg)?;
            }
        }
        Ok(next)
    }
    pub fn cancellation(&self) -> Option<End> {
        self.cancel
    }
    fn status(&self, now: Duration) -> Status {
        Status {
            id: self.id,
            background: self.background,
            slot: self.slot,
            phase: if self.slot.is_none() {
                "queued"
            } else if self.paused {
                "paused"
            } else if self.restoring {
                "restoring"
            } else if self.done < self.tokens.len() {
                "prefill"
            } else {
                "decoding"
            },
            prefill_done: self.done,
            prompt_tokens: self.tokens.len(),
            cached_tokens: self.cached_tokens,
            output_tokens: self.outputs.len(),
            admission_count: u8::from(self.admitted.is_some()),
            prefill_calls: self.prefill_calls,
            decode_calls: self.decode_calls,
            speculative_calls: self.speculative_calls,
            decode_tokens: self.decode_tokens,
            pause_count: self.pause_count,
            resume_count: self.resume_count,
            queued_ns: (self.admitted.unwrap_or(now) - self.submitted).as_nanos(),
            active_ns: self.work_time.as_nanos(),
            prefill_active_ns: self.prefill_time.as_nanos(),
            decode_active_ns: self.decode_time.as_nanos(),
            paused_ns: (self.pause_time + self.paused_at.map_or(Duration::ZERO, |t| now - t))
                .as_nanos(),
            age_ns: (now - self.submitted).as_nanos(),
        }
    }
}
pub(crate) struct Queue {
    pub gate: Gate,
    requests: BTreeMap<u64, Request>,
    slots: [Option<u64>; SLOTS],
    pub completed: VecDeque<Completed>,
    totals: crate::request_metrics::Counters,
    serial: u64,
    last: Duration,
    cursor: usize,
}
impl Default for Queue {
    fn default() -> Self {
        Self {
            gate: Gate::default(),
            requests: BTreeMap::new(),
            slots: [None; SLOTS],
            completed: VecDeque::new(),
            totals: Default::default(),
            serial: 0,
            last: Duration::ZERO,
            cursor: 0,
        }
    }
}
impl Queue {
    pub fn now(&mut self, now: Duration) -> Result<()> {
        ensure!(now >= self.last, "request clock went backwards");
        self.last = now;
        Ok(())
    }
    pub fn submit(
        &mut self,
        tokens: Arc<[i32]>,
        output: usize,
        background: bool,
        deadline: Duration,
        now: Duration,
    ) -> Result<u64> {
        self.submit_policy(tokens, output, background, deadline, now, None)
    }
    pub fn submit_policy(
        &mut self,
        tokens: Arc<[i32]>,
        output: usize,
        background: bool,
        deadline: Duration,
        now: Duration,
        policy: Option<&crate::output_policy::Policy>,
    ) -> Result<u64> {
        ensure!(
            self.requests.len() < REQUESTS - usize::from(background),
            "request capacity / foreground queue reservation"
        );
        ensure!(
            !tokens.is_empty()
                && output > 0
                && output <= MAX_OUTPUT
                && tokens.len() <= CONTEXT - output
                && tokens.iter().all(|&t| crate::vision_input::prompt_token(t)),
            "request token/output bounds"
        );
        ensure!(deadline > now, "request deadline elapsed");
        if let Some(policy) = policy {
            policy.validate()?;
        }
        let thinking = policy
            .map(|p| {
                crate::thinking_budget::State::new(
                    p.thinking_budget(),
                    output,
                    tokens.len(),
                    crate::output_decode::THINK_END,
                )
                .map_err(anyhow::Error::msg)
            })
            .transpose()?;
        let id = self
            .serial
            .checked_add(1)
            .context("request IDs exhausted")?;
        self.now(now)?;
        let ticket = if background {
            None
        } else {
            Some(self.gate.enter(now)?)
        };
        self.requests.insert(
            id,
            Request {
                id,
                tokens,
                output,
                background,
                slot: None,
                done: 0,
                cached_tokens: 0,
                restoring: false,
                cancel_forwarded: false,
                outputs: Vec::with_capacity(output),
                output_hash: Sha256::new(),
                completion: None,
                policy: policy.map(|p| p.digest()),
                sampling: policy.and_then(|p| p.sampling()),
                thinking,
                string_stops: policy.is_some_and(|p| p.has_stops()),
                has_grammar: policy.is_some_and(|p| p.grammar().is_some()),
                drafts: policy.is_some_and(|p| p.drafts()),
                stop_eos: policy.is_some_and(|p| !p.ignore_eos()),
                paused: false,
                cancel: None,
                ticket,
                submitted: now,
                admitted: None,
                deadline,
                paused_at: None,
                pause_time: Duration::ZERO,
                work_time: Duration::ZERO,
                prefill_time: Duration::ZERO,
                decode_time: Duration::ZERO,
                prefill_calls: 0,
                decode_calls: 0,
                speculative_calls: 0,
                decode_tokens: 0,
                pause_count: 0,
                resume_count: 0,
                marks: None,
            },
        );
        self.serial = id;
        Ok(id)
    }
    pub fn request(&self, id: u64) -> Result<&Request> {
        self.requests.get(&id).context("unknown request")
    }
    pub fn status(&self, id: u64, now: Duration) -> Result<Status> {
        ensure!(now >= self.last, "status clock went backwards");
        Ok(self.request(id)?.status(now))
    }
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(r) = self.requests.get_mut(&id) {
            r.cancel.get_or_insert(End::Cancelled);
            true
        } else {
            false
        }
    }
    pub fn cancellation(&mut self, now: Duration) -> Result<Option<(u64, End)>> {
        self.now(now)?;
        for r in self.requests.values_mut() {
            if now >= r.deadline {
                r.cancel.get_or_insert(End::TimedOut);
            }
        }
        Ok(self.requests.values().find_map(|r| {
            r.cancel
                .filter(|_| !r.cancel_forwarded)
                .map(|end| (r.id, end))
        }))
    }
    pub fn foreground(&self) -> bool {
        self.requests.values().any(|r| !r.background)
    }
    pub fn queued(&self, held: bool) -> Option<u64> {
        self.requests
            .values()
            .filter(|r| {
                r.slot.is_none()
                    && r.cancel.is_none()
                    && (!r.background
                        || (!held
                            && self
                                .requests
                                .values()
                                .filter(|a| a.background && a.slot.is_some())
                                .count()
                                < SLOTS - 1))
            })
            .min_by_key(|r| (r.background, r.id))
            .map(|r| r.id)
    }
    pub fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }
    pub fn slot_free(&self, slot: usize) -> bool {
        self.slots.get(slot).is_some_and(Option::is_none)
    }
    #[cfg(test)]
    pub fn admit(&mut self, id: u64, slot: usize, now: Duration) -> Result<()> {
        self.admit_cached(id, slot, 0, now)
    }
    pub fn admit_cached(
        &mut self,
        id: u64,
        slot: usize,
        cached: usize,
        now: Duration,
    ) -> Result<()> {
        ensure!(
            slot < SLOTS && self.slots[slot].is_none(),
            "request slot owned"
        );
        let r = self.request(id)?;
        ensure!(
            cached < r.tokens.len(),
            "cache must leave fresh prompt logits"
        );
        ensure!(
            !r.background
                || self
                    .requests
                    .values()
                    .filter(|r| r.background && r.slot.is_some())
                    .count()
                    < SLOTS - 1,
            "background slot reservation"
        );
        ensure!(
            r.slot.is_none() && r.cancel.is_none(),
            "request already admitted/cancelled"
        );
        self.now(now)?;
        let r = self.requests.get_mut(&id).unwrap();
        r.slot = Some(slot);
        r.done = cached;
        r.cached_tokens = cached;
        r.admitted = Some(now);
        self.slots[slot] = Some(id);
        Ok(())
    }
    pub fn hold(&mut self, held: bool, now: Duration) -> Result<()> {
        self.now(now)?;
        for r in self.requests.values_mut() {
            let paused = held && r.background && r.slot.is_some();
            if paused && !r.paused {
                r.paused_at = Some(now);
                r.pause_count += 1;
            }
            if !paused && r.paused {
                r.pause_time += now - r.paused_at.take().unwrap();
                r.resume_count += 1;
            }
            r.paused = paused;
        }
        Ok(())
    }
    pub fn reserve_restore(&mut self, id: u64, slot: usize, now: Duration) -> Result<()> {
        self.admit_cached(id, slot, 0, now)?;
        self.requests.get_mut(&id).unwrap().restoring = true;
        Ok(())
    }
    pub fn forward_restore_cancel(&mut self, id: u64) -> Result<()> {
        let r = self
            .requests
            .get_mut(&id)
            .context("unknown restore owner")?;
        ensure!(
            r.restoring && r.cancel.is_some(),
            "restore cancellation not requested"
        );
        r.cancel_forwarded = true;
        Ok(())
    }
    /// Called only after all ranks commit or discard the destination and the
    /// controller releases the file worker. No partial state earns cache credit.
    pub fn finish_restore(&mut self, id: u64, cached: usize) -> Result<()> {
        let r = self
            .requests
            .get_mut(&id)
            .context("unknown restore owner")?;
        ensure!(
            r.restoring
                && r.done == 0
                && r.outputs.is_empty()
                && cached < r.tokens.len()
                && (r.cancel.is_none() || cached == 0),
            "invalid restore commit"
        );
        r.restoring = false;
        r.done = cached;
        r.cached_tokens = cached;
        Ok(())
    }
    pub fn set_marks(&mut self, id: u64, marks: Vec<usize>) -> Result<()> {
        let r = self.requests.get_mut(&id).context("unknown request")?;
        ensure!(
            r.marks.is_none()
                && marks.windows(2).all(|w| w[0] < w[1])
                && marks.iter().all(|&m| r.done < m && m < r.tokens.len()),
            "snapshot marks outside the prefill"
        );
        r.marks = Some(marks);
        Ok(())
    }
    /// The other admitted requests (fork points and prefix-share partners).
    pub fn admitted(&self, except: u64) -> impl Iterator<Item = &Request> {
        self.requests
            .values()
            .filter(move |r| r.id != except && r.slot.is_some())
    }
    pub(crate) fn requests_for_slot(&self, slot: usize) -> Option<&Request> {
        self.slots
            .get(slot)
            .copied()
            .flatten()
            .and_then(|id| self.requests.get(&id))
    }
    /// Select without spending the turn; a batched cohort can include peers
    /// while advancing only its first request's normal round-robin turn.
    pub(crate) fn peek_runnable(&self) -> Option<u64> {
        // Foreground work always outranks background; round-robin among equals.
        let foreground = self
            .requests
            .values()
            .any(|r| r.slot.is_some() && !r.background && r.cancel.is_none());
        for offset in 0..SLOTS {
            let slot = (self.cursor + offset) % SLOTS;
            if let Some(id) = self.slots[slot] {
                let r = &self.requests[&id];
                if !r.restoring && !r.paused && r.cancel.is_none() && (!foreground || !r.background)
                {
                    return Some(id);
                }
            }
        }
        None
    }
    pub fn runnable(&mut self) -> Option<u64> {
        let id = self.peek_runnable()?;
        self.cursor = (self.requests[&id].slot.unwrap() + 1) % SLOTS;
        Some(id)
    }
    pub fn worked(&mut self, id: u64, inputs: usize, next: i32, elapsed: Duration) -> Result<bool> {
        self.worked_batch(id, inputs, &[next], elapsed, false)
    }
    /// A nonterminal prefill piece may omit its unused head. Final prefill must
    /// still provide exactly one pending token; no placeholder ID is invented.
    pub fn worked_prefill(
        &mut self,
        id: u64,
        inputs: usize,
        next: Option<i32>,
        elapsed: Duration,
    ) -> Result<bool> {
        let r = self.request(id)?;
        ensure!(r.done < r.tokens.len(), "prefill already complete");
        self.worked_batch(id, inputs, next.as_slice(), elapsed, false)
    }
    /// Publish one already-committed model-work boundary. Prefill supplies one
    /// candidate next token (ignored until its final piece). Decode supplies one
    /// new output per committed input, including the last still-pending token.
    /// The existing pending output is the first input consumed by this round;
    /// after publication outputs.len() == decode_tokens + 1.
    ///
    /// This does not choose speculative acceptance. The native owner must bound
    /// acceptance by EOS, string stops, grammar, thinking/output budgets before
    /// committing GPU state, then agree it on all ranks before calling this API.
    /// All rejection checks precede mutation. Call once per request in a cohort.
    pub fn worked_many(
        &mut self,
        id: u64,
        inputs: usize,
        next: &[i32],
        elapsed: Duration,
    ) -> Result<bool> {
        self.worked_batch(id, inputs, next, elapsed, true)
    }
    fn worked_batch(
        &mut self,
        id: u64,
        inputs: usize,
        next: &[i32],
        elapsed: Duration,
        speculative: bool,
    ) -> Result<bool> {
        let r = self.requests.get_mut(&id).context("unknown work owner")?;
        ensure!(
            r.slot
                .is_some_and(|slot| self.slots.get(slot) == Some(&Some(id)))
                && !r.restoring
                && !r.paused
                && r.cancel.is_none()
                && r.completion.is_none()
                && next.len() <= DECODE_COMMIT_ROWS
                && next.iter().all(|t| (0..154880).contains(t)),
            "unrunnable request"
        );
        ensure!(r.outputs.len() < r.output, "request output exhausted");
        let work_time = r
            .work_time
            .checked_add(elapsed)
            .context("request timer overflow")?;
        if r.done < r.tokens.len() {
            ensure!(
                inputs > 0
                    && inputs <= 8192
                    && inputs <= r.tokens.len() - r.done
                    && (next.len() == 1
                        || (next.is_empty() && r.done + inputs < r.tokens.len() && !speculative))
                    && r.outputs.is_empty()
                    && r.decode_calls == 0
                    && r.decode_tokens == 0,
                "prefill progress bounds"
            );
            let calls = r
                .prefill_calls
                .checked_add(1)
                .context("prefill count overflow")?;
            let time = r
                .prefill_time
                .checked_add(elapsed)
                .context("prefill timer overflow")?;
            let thinking = if r.done + inputs == r.tokens.len() {
                r.preview_outputs(next)?
            } else {
                r.thinking
            };
            r.done += inputs;
            r.prefill_calls = calls;
            r.prefill_time = time;
            if r.done == r.tokens.len() {
                r.outputs.push(next[0]);
                r.output_hash.update(next[0].to_le_bytes());
            }
            r.thinking = thinking;
        } else {
            ensure!(
                r.done == r.tokens.len()
                    && !next.is_empty()
                    && inputs == next.len()
                    && next.len() <= r.output - r.outputs.len()
                    && r.decode_tokens.checked_add(1) == Some(r.outputs.len()),
                "decode progress bounds"
            );
            if r.stop_eos {
                ensure!(
                    r.outputs
                        .last()
                        .is_none_or(|t| !crate::output_decode::EOS.contains(&(*t as u32)))
                        && !next[..next.len() - 1]
                            .iter()
                            .any(|t| crate::output_decode::EOS.contains(&(*t as u32))),
                    "decode commit crosses EOS"
                );
            }
            let calls = r
                .decode_calls
                .checked_add(1)
                .context("decode count overflow")?;
            let speculative_calls = r
                .speculative_calls
                .checked_add(u64::from(speculative))
                .context("speculative count overflow")?;
            let tokens = r
                .decode_tokens
                .checked_add(inputs)
                .context("decode token overflow")?;
            let time = r
                .decode_time
                .checked_add(elapsed)
                .context("decode timer overflow")?;
            let thinking = r.preview_outputs(next)?;
            r.decode_calls = calls;
            r.speculative_calls = speculative_calls;
            r.decode_tokens = tokens;
            r.decode_time = time;
            r.outputs.extend_from_slice(next);
            for token in next {
                r.output_hash.update(token.to_le_bytes());
            }
            r.thinking = thinking;
        }
        r.work_time = work_time;
        let done = r.outputs.len() == r.output;
        if done {
            r.completion = Some(Completion::Length);
        }
        Ok(done)
    }
    /// Only the common native post-token decision can shorten a successful
    /// request. Clients cannot inject a completion command or a bare slot ID.
    pub fn terminal(&mut self, id: u64, completion: Completion) -> Result<()> {
        let r = self
            .requests
            .get_mut(&id)
            .context("unknown terminal owner")?;
        ensure!(
            r.policy.is_some()
                && r.slot.is_some()
                && !r.paused
                && r.cancel.is_none()
                && r.done == r.tokens.len()
                && !r.outputs.is_empty(),
            "unready terminal request"
        );
        ensure!(
            match completion {
                Completion::Length => r.outputs.len() == r.output,
                Completion::Eos => r.stops_at(*r.outputs.last().unwrap()),
                Completion::Stop => r.string_stops,
            },
            "invalid terminal reason"
        );
        if let Some(state) = &mut r.thinking {
            state.finish().map_err(anyhow::Error::msg)?;
        }
        r.completion = Some(completion);
        Ok(())
    }
    /// Charge a collectively rejected parser round without claiming committed
    /// tokens/calls. Its native slot has already been drained and discarded.
    pub fn output_rejected(&mut self, id: u64, elapsed: Duration) -> Result<()> {
        let r = self
            .requests
            .get_mut(&id)
            .context("unknown rejected output owner")?;
        ensure!(
            r.policy.is_some()
                && !r.outputs.is_empty()
                && r.slot.is_some()
                && !r.paused
                && !r.restoring
                && r.cancel.is_none()
                && r.done == r.tokens.len(),
            "invalid rejected output owner"
        );
        let work = r
            .work_time
            .checked_add(elapsed)
            .context("request timer overflow")?;
        let decode = r
            .decode_time
            .checked_add(elapsed)
            .context("decode timer overflow")?;
        r.work_time = work;
        r.decode_time = decode;
        r.cancel = Some(End::OutputError);
        r.completion = None;
        Ok(())
    }
    pub fn output_failed(&mut self, id: u64) -> Result<()> {
        let r = self.requests.get_mut(&id).context("unknown output owner")?;
        ensure!(
            r.policy.is_some() && !r.outputs.is_empty(),
            "invalid output failure"
        );
        r.cancel = Some(End::OutputError);
        r.completion = None;
        Ok(())
    }
    pub fn output_policy(&self, id: u64) -> Result<bool> {
        Ok(self.request(id)?.policy.is_some())
    }
    /// Retire only after the CUDA owner has completed/released the matching slot.
    pub fn finish(&mut self, id: u64, reason: End, now: Duration) -> Result<()> {
        self.now(now)?;
        let r = self.request(id)?;
        ensure!(!r.restoring, "restore worker still owns destination");
        ensure!(
            reason != End::Complete || r.completion.is_some(),
            "unfinished output"
        );
        if let Some(slot) = r.slot {
            ensure!(self.slots[slot] == Some(id), "request lost slot ownership");
        }
        if let Some(ticket) = r.ticket {
            self.gate.leave(ticket, now)?;
        }
        let r = self.requests.remove(&id).unwrap();
        let mut status = r.status(now);
        status.phase = match reason {
            End::Complete => "complete",
            End::Cancelled => "cancelled",
            End::TimedOut => "timed_out",
            End::Rejected => "rejected",
            End::OutputError => "output_error",
        };
        if let Some(slot) = r.slot {
            self.slots[slot] = None;
        }
        if self.completed.len() == HISTORY {
            self.completed.pop_front();
        }
        self.totals.finish(&status, reason, now.as_nanos());
        self.completed.push_back(Completed {
            status,
            reason,
            outputs: r.outputs,
            completion: if reason == End::Complete {
                r.completion
            } else {
                None
            },
        });
        Ok(())
    }
    /// Duration until actual model work or request maintenance is possible.
    /// No round-robin state is advanced by this read-only readiness query.
    pub fn wait(&self, now: Duration, held: bool) -> Duration {
        if self.requests.values().any(|r| {
            ((!r.cancel_forwarded) && (r.cancel.is_some() || r.deadline <= now))
                || (r.slot.is_some() && !r.restoring && !r.paused && r.cancel.is_none())
        }) || (self.free_slot().is_some() && self.queued(held).is_some())
        {
            return Duration::ZERO;
        }
        self.requests
            .values()
            .filter(|r| !r.cancel_forwarded)
            .map(|r| r.deadline.saturating_sub(now))
            .min()
            .unwrap_or(Duration::from_secs(1))
            .min(Duration::from_secs(1))
    }
    pub fn telemetry(&self, now: Duration) -> serde_json::Value {
        serde_json::json!({"requests":self.requests.values().map(|r|r.status(now)).collect::<Vec<_>>(),"totals":self.totals,
            "recent":self.completed.iter().map(|c|serde_json::json!({"status":c.status,"reason":c.reason,"completion":c.completion})).collect::<Vec<_>>()})
    }
    pub fn count(&self) -> usize {
        self.requests.len()
    }
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"request-owners-v10");
        for r in self.requests.values() {
            for n in [
                r.id,
                r.slot.map_or(u64::MAX, |s| s as u64),
                r.tokens.len() as u64,
                r.output as u64,
                r.done as u64,
                r.cached_tokens as u64,
                r.outputs.len() as u64,
                r.prefill_calls,
                r.decode_calls,
                r.speculative_calls,
                r.decode_tokens as u64,
                r.pause_count,
                r.resume_count,
            ] {
                h.update(n.to_le_bytes());
            }
            h.update([
                u8::from(r.background),
                u8::from(r.restoring),
                u8::from(r.cancel_forwarded),
                u8::from(r.paused),
                r.cancel.map_or(0, |e| e as u8 + 1),
                r.completion.map_or(0, |c| c as u8 + 1),
                u8::from(r.policy.is_some()),
                u8::from(r.string_stops),
                u8::from(r.has_grammar),
                u8::from(r.drafts),
                u8::from(r.stop_eos),
                u8::from(r.sampling.is_some()),
                u8::from(r.thinking.is_some()),
            ]);
            if let Some(policy) = r.policy {
                h.update(policy);
            }
            if let Some(sampling) = r.sampling {
                h.update(sampling.encode());
            }
            if let Some(state) = r.thinking {
                h.update(state.digest_bytes());
            }
            h.update(r.outputs.last().unwrap_or(&-1).to_le_bytes());
            if let Some(marks) = &r.marks {
                h.update((marks.len() as u64).to_le_bytes());
                for m in marks {
                    h.update((*m as u64).to_le_bytes());
                }
            } else {
                h.update(u64::MAX.to_le_bytes());
            }
            // Incremental hashing covers middle tokens of an atomic commit
            // without rescanning a potentially 128K-token response every round.
            h.update(r.output_hash.clone().finalize());
        }
        h.finalize().into()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn publication_snapshot(q: &Queue) -> ([u8; 32], serde_json::Value) {
        (q.digest(), q.telemetry(q.last))
    }
    #[test]
    fn ignore_eos_continues_atomic_commits_holds_and_cap_without_false_terminal() -> Result<()> {
        let base = crate::output_policy::Policy::new(false, false, &serde_json::json!("STOP"))?;
        let policy = base.clone().with_ignore_eos(true)?;
        for eos in crate::output_decode::EOS {
            let mut queue = Queue::default();
            let id = queue.submit_policy(
                vec![17].into(),
                6,
                true,
                Duration::from_secs(10),
                Duration::ZERO,
                Some(&policy),
            )?;
            queue.admit(id, 0, Duration::ZERO)?;
            assert!(!queue.worked(id, 1, 20, Duration::ZERO)?);
            assert!(!queue.request(id)?.stop_eos());
            assert_eq!(
                queue
                    .request(id)?
                    .sampled_prefix(&[eos as i32, 21, eos as i32])?,
                3
            );
            assert!(!queue.worked_many(id, 3, &[eos as i32, 21, eos as i32], Duration::ZERO)?);
            assert!(queue.terminal(id, Completion::Eos).is_err());
            let before = publication_snapshot(&queue);
            assert!(
                queue
                    .worked_many(id, 3, &[22, 23, 24], Duration::ZERO)
                    .is_err()
            );
            assert_eq!(publication_snapshot(&queue), before);
            queue.hold(true, Duration::from_secs(1))?;
            assert_eq!(
                queue.request(id)?.next_output(),
                crate::thinking_budget::Next::Sample
            );
            assert!(queue.worked(id, 1, 22, Duration::ZERO).is_err());
            queue.hold(false, Duration::from_secs(2))?;
            assert_eq!(queue.request(id)?.sampled_prefix(&[22, eos as i32, 24])?, 2);
            assert!(queue.worked_many(id, 2, &[22, eos as i32], Duration::ZERO)?);
            assert_eq!(queue.request(id)?.completion, Some(Completion::Length));
            assert!(queue.terminal(id, Completion::Eos).is_err());
            assert_eq!(
                queue.request(id)?.outputs,
                [20, eos as i32, 21, eos as i32, 22, eos as i32]
            );
            assert_eq!(queue.request(id)?.decode_tokens, 5);
        }
        let mut normal = Queue::default();
        let mut ignore = Queue::default();
        normal.submit_policy(
            vec![17].into(),
            6,
            false,
            Duration::from_secs(10),
            Duration::ZERO,
            Some(&base),
        )?;
        ignore.submit_policy(
            vec![17].into(),
            6,
            false,
            Duration::from_secs(10),
            Duration::ZERO,
            Some(&policy),
        )?;
        assert_ne!(normal.digest(), ignore.digest());
        let grammar = crate::grammar_bound::Bound::from_request(
            &serde_json::json!({"guided_regex":"x"}),
            &[],
        )?
        .unwrap();
        let conflict = policy.with_grammar(Some(grammar));
        let before = ignore.digest();
        assert!(
            ignore
                .submit_policy(
                    vec![17].into(),
                    6,
                    false,
                    Duration::from_secs(10),
                    Duration::ZERO,
                    Some(&conflict)
                )
                .is_err()
        );
        assert_eq!(ignore.digest(), before);
        Ok(())
    }
    #[test]
    fn explicit_draft_disable_and_grammar_are_retained_in_dispatch_identity() -> Result<()> {
        let policy = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?;
        let submit = |policy: &crate::output_policy::Policy| -> Result<Queue> {
            let mut q = Queue::default();
            q.submit_policy(vec![7; 64].into(), 16, false, t(100), t(0), Some(policy))?;
            Ok(q)
        };
        let enabled = submit(&policy)?;
        let disabled = submit(&policy.clone().with_drafts(false))?;
        assert!(enabled.request(1)?.drafts());
        assert!(!disabled.request(1)?.drafts());
        assert_ne!(enabled.digest(), disabled.digest());
        assert!(!enabled.request(1)?.has_grammar());
        let bound = crate::grammar_bound::Bound::from_request(
            &serde_json::json!({"guided_regex":"x"}),
            &[],
        )?;
        let constrained = submit(&policy.with_grammar(bound))?;
        assert!(constrained.request(1)?.has_grammar());
        assert_ne!(enabled.digest(), constrained.digest());
        Ok(())
    }
    fn thinking_owner(budget: usize, output: usize, prompt: usize) -> Result<(Queue, u64)> {
        let policy = crate::output_policy::Policy::new(true, false, &serde_json::json!("STOP"))?
            .with_thinking_budget(Some(budget))?
            .with_sampling(crate::sampling::Sampling::new(99, 1.0, 20, 0.95)?)?;
        let mut q = Queue::default();
        let id = q.submit_policy(
            vec![7; prompt].into(),
            output,
            true,
            t(1000),
            t(0),
            Some(&policy),
        )?;
        q.admit(id, 0, t(0))?;
        Ok((q, id))
    }
    #[test]
    fn speculative_preview_is_pure_and_stops_at_budget_eos_and_output_cap() -> Result<()> {
        let (mut q, id) = thinking_owner(3, 8, 1)?;
        q.worked(id, 1, 10, t(1))?;
        let before = publication_snapshot(&q);
        assert_eq!(q.request(id)?.sampled_prefix(&[11, 12, 13, 14])?, 2);
        assert_eq!(q.request(id)?.sampled_prefix(&[11, 154827, 13])?, 2);
        assert_eq!(publication_snapshot(&q), before);
        q.worked_many(id, 2, &[11, 12], t(2))?;
        let before = publication_snapshot(&q);
        assert_eq!(q.request(id)?.sampled_prefix(&[13, 14])?, 0);
        assert_eq!(publication_snapshot(&q), before);
        let (mut q, id) = thinking_owner(100, 3, 1)?;
        q.worked(id, 1, 10, t(1))?;
        assert_eq!(q.request(id)?.sampled_prefix(&[11, 12, 13])?, 2);
        assert!(q.request(id)?.sampled_prefix(&[]).is_err());
        assert!(q.request(id)?.sampled_prefix(&[-1]).is_err());
        Ok(())
    }
    #[test]
    fn parser_rejection_charges_time_without_publishing_or_advancing_tokens() -> Result<()> {
        let (mut q, id) = thinking_owner(100, 8, 1)?;
        let before = publication_snapshot(&q);
        assert!(q.output_rejected(id, t(2)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.worked(id, 1, 10, t(1))?;
        let hash = q.request(id)?.output_hash.clone().finalize();
        q.output_rejected(id, t(2))?;
        let r = q.request(id)?;
        assert_eq!(r.outputs, vec![10]);
        assert_eq!(r.output_hash.clone().finalize(), hash);
        assert_eq!((r.decode_calls, r.decode_tokens), (0, 0));
        assert_eq!(r.decode_time, t(2));
        assert_eq!(r.work_time, t(3));
        assert_eq!(r.cancellation(), Some(End::OutputError));
        let before = publication_snapshot(&q);
        assert!(q.output_rejected(id, t(2)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.finish(id, End::OutputError, t(3))?;
        assert_eq!(q.completed[0].status.output_tokens, 1);
        Ok(())
    }
    #[test]
    fn thinking_zero_forces_first_published_token_without_extra_work() -> Result<()> {
        use crate::{output_decode::THINK_END, thinking_budget::Next};
        let (mut q, id) = thinking_owner(0, 3, 65)?;
        assert_eq!(q.request(id)?.next_output(), Next::Force(THINK_END));
        // Intermediate prefill selections are not generated output, even EOS.
        assert!(!q.worked(id, 64, 154828, t(2))?);
        assert_eq!(q.request(id)?.thinking.unwrap().emitted(), 0);
        let before = publication_snapshot(&q);
        assert!(q.worked(id, 1, 19, t(3)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        assert!(!q.worked(id, 1, THINK_END, t(3))?);
        assert_eq!(q.request(id)?.outputs, [THINK_END]);
        assert_eq!(q.request(id)?.next_output(), Next::Sample);
        assert!(q.worked_many(id, 2, &[20, 21], t(4))?);
        let r = q.request(id)?;
        assert_eq!(
            (r.prefill_calls, r.decode_calls, r.decode_tokens),
            (2, 1, 2)
        );
        assert_eq!(r.thinking.unwrap().emitted(), 3);
        assert_eq!(r.next_output(), Next::End);
        q.finish(id, End::Complete, t(20))?;
        assert_eq!(q.completed[0].outputs, [THINK_END, 20, 21]);
        assert_eq!(q.completed[0].completion, Some(Completion::Length));
        assert_eq!(q.totals.completion_tokens, 3);
        Ok(())
    }
    #[test]
    fn thinking_atomic_window_cannot_publish_past_forced_boundary() -> Result<()> {
        use crate::{output_decode::THINK_END, thinking_budget::Next};
        let (mut q, id) = thinking_owner(3, 6, 1)?;
        q.worked(id, 1, 19, t(1))?;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 3, &[20, 21, 22], t(3)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.worked_many(id, 2, &[20, 21], t(2))?;
        assert_eq!(q.request(id)?.next_output(), Next::Force(THINK_END));
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 1, &[154828], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        // A caller that has committed the actual forced path may publish it
        // atomically; the delimiter is counted once and disables later forcing.
        assert!(q.worked_many(id, 3, &[THINK_END, 22, 23], t(3))?);
        let r = q.request(id)?;
        assert_eq!(r.outputs, [19, 20, 21, THINK_END, 22, 23]);
        assert_eq!((r.decode_calls, r.decode_tokens), (2, 5));
        assert_eq!(r.next_output(), Next::End);
        let before = publication_snapshot(&q);
        assert!(q.worked(id, 1, 24, t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        Ok(())
    }
    #[test]
    fn thinking_natural_close_and_total_output_cap_win() -> Result<()> {
        use crate::{output_decode::THINK_END, thinking_budget::Next};
        for budget in [1, 3] {
            let (mut q, id) = thinking_owner(budget, 5, 1)?;
            q.worked(id, 1, THINK_END, t(1))?;
            q.worked_many(id, 3, &[20, 21, 22], t(3))?;
            assert_eq!(q.request(id)?.next_output(), Next::Sample);
            assert!(q.worked(id, 1, 23, t(1))?);
            assert_eq!(q.request(id)?.next_output(), Next::End);
        }
        for budget in [0, 1, usize::MAX] {
            let (mut q, id) = thinking_owner(budget, 1, 1)?;
            let selected = if budget == 0 { THINK_END } else { 19 };
            assert!(q.worked(id, 1, selected, t(1))?);
            assert_eq!(q.request(id)?.next_output(), Next::End);
            assert_eq!(q.request(id)?.outputs, [selected]);
            assert_eq!(q.request(id)?.completion, Some(Completion::Length));
        }
        Ok(())
    }
    #[test]
    fn thinking_eos_and_visible_stop_terminate_before_due_delimiter() -> Result<()> {
        use crate::thinking_budget::Next;
        for (token, completion) in [(154828, Completion::Eos), (20, Completion::Stop)] {
            let (mut q, id) = thinking_owner(2, 8, 1)?;
            q.worked(id, 1, 19, t(1))?;
            q.worked(id, 1, token, t(1))?;
            q.terminal(id, completion)?;
            assert_eq!(q.request(id)?.next_output(), Next::End);
            assert_eq!(q.request(id)?.thinking.unwrap().emitted(), 2);
            let before = publication_snapshot(&q);
            assert!(
                q.worked(id, 1, crate::output_decode::THINK_END, t(1))
                    .is_err()
            );
            assert_eq!(publication_snapshot(&q), before);
            q.finish(id, End::Complete, t(3))?;
            assert_eq!(q.completed[0].outputs, [19, token]);
            assert_eq!(q.completed[0].completion, Some(completion));
        }
        Ok(())
    }
    #[test]
    fn thinking_native_hold_other_owner_and_cancel_preserve_pending_budget() -> Result<()> {
        use crate::{output_decode::THINK_END, thinking_budget::Next};
        let (mut q, bg) = thinking_owner(1, 8, 1)?;
        q.worked(bg, 1, 19, t(1))?;
        let state = q.request(bg)?.thinking;
        assert_eq!(q.request(bg)?.next_output(), Next::Force(THINK_END));
        q.hold(true, t(10))?;
        let fg = q.submit(vec![8].into(), 1, false, t(1000), t(10))?;
        q.admit(fg, 1, t(10))?;
        let before = publication_snapshot(&q);
        assert!(q.worked(bg, 1, THINK_END, t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        assert!(q.worked(fg, 1, 30, t(1))?);
        q.finish(fg, End::Complete, t(11))?;
        assert_eq!(q.request(bg)?.thinking, state);
        assert_eq!(q.request(bg)?.outputs, [19]);
        q.hold(false, t(110))?;
        q.worked(bg, 1, THINK_END, t(1))?;
        let forced = q.request(bg)?.thinking;
        assert_eq!(q.request(bg)?.next_output(), Next::Sample);
        q.hold(true, t(120))?;
        q.hold(false, t(220))?;
        assert_eq!(q.request(bg)?.thinking, forced);
        assert_eq!(q.request(bg)?.outputs, [19, THINK_END]);
        assert!(q.cancel(bg));
        let before = publication_snapshot(&q);
        assert!(q.worked(bg, 1, 20, t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.finish(bg, End::Cancelled, t(221))?;
        assert_eq!(q.completed[1].outputs, [19, THINK_END]);
        assert_eq!(q.completed[1].status.paused_ns, t(200).as_nanos());
        Ok(())
    }
    #[test]
    fn thinking_digest_covers_closure_and_rejects_counter_divergence_atomically() -> Result<()> {
        let (mut q, id) = thinking_owner(3, 8, 1)?;
        q.worked(id, 1, 19, t(1))?;
        let original = q.request(id)?.thinking.unwrap();
        let digest = q.digest();
        let mut closed =
            crate::thinking_budget::State::new(Some(3), 8, 1, crate::output_decode::THINK_END)
                .map_err(anyhow::Error::msg)?;
        closed
            .accept(crate::output_decode::THINK_END, false)
            .map_err(anyhow::Error::msg)?;
        q.requests.get_mut(&id).unwrap().thinking = Some(closed);
        assert_eq!(original.emitted(), closed.emitted());
        assert_ne!(q.digest(), digest);
        q.requests.get_mut(&id).unwrap().thinking = Some(original);
        assert_eq!(q.digest(), digest);
        let mut advanced = original;
        advanced.accept(20, false).map_err(anyhow::Error::msg)?;
        q.requests.get_mut(&id).unwrap().thinking = Some(advanced);
        let before = publication_snapshot(&q);
        assert!(q.worked(id, 1, 20, t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        Ok(())
    }
    #[test]
    fn atomic_decode_preserves_serial_tokens_cursor_and_total_work() -> Result<()> {
        let mut serial = Queue::default();
        let mut multi = Queue::default();
        for q in [&mut serial, &mut multi] {
            let id = q.submit(vec![7; 65].into(), 5, true, t(1000), t(0))?;
            q.admit(id, 0, t(0))?;
            assert!(!q.worked(id, 64, 18, t(2))?);
            assert!(!q.worked(id, 1, 19, t(3))?);
            assert_eq!(q.request(id)?.outputs, [19]);
            assert_eq!(q.request(id)?.decode_tokens, 0);
        }
        for (i, token) in [20, 21, 22, 23].into_iter().enumerate() {
            assert_eq!(serial.worked(1, 1, token, t(i as u64 + 1))?, i == 3);
        }
        assert!(multi.worked_many(1, 4, &[20, 21, 22, 23], t(10))?);
        let a = serial.request(1)?;
        let b = multi.request(1)?;
        assert_eq!(a.outputs, b.outputs);
        assert_eq!((a.done, b.done), (65, 65));
        assert_eq!((a.decode_tokens, b.decode_tokens), (4, 4));
        assert_eq!((a.decode_calls, b.decode_calls), (4, 1));
        assert_eq!(b.outputs.len(), b.decode_tokens + 1);
        assert_eq!(b.outputs.last(), Some(&23));
        assert_eq!(a.work_time, b.work_time);
        assert_eq!(a.prefill_time, b.prefill_time);
        assert_eq!(a.decode_time, b.decode_time);
        assert_eq!(b.completion, Some(Completion::Length));
        // Output hashing is independent of publication grouping. The sole
        // ownership-digest difference is the legitimate round-count difference.
        assert_ne!(serial.digest(), multi.digest());
        multi.requests.get_mut(&1).unwrap().decode_calls = 4;
        multi.requests.get_mut(&1).unwrap().speculative_calls = 0;
        assert_eq!(serial.digest(), multi.digest());
        multi.requests.get_mut(&1).unwrap().decode_calls = 1;
        multi.requests.get_mut(&1).unwrap().speculative_calls = 1;
        for q in [&mut serial, &mut multi] {
            q.finish(1, End::Complete, t(20))?;
            assert_eq!(q.totals.completion_tokens, 5);
            assert_eq!(q.totals.prefill_ns, t(5).as_nanos());
            assert_eq!(q.totals.decode_ns, t(10).as_nanos());
            assert_eq!(q.completed[0].status.decode_tokens, 4);
        }
        assert_eq!(serial.totals.decode_rounds, 4);
        assert_eq!(multi.totals.decode_rounds, 1);
        Ok(())
    }
    #[test]
    fn atomic_decode_rejects_invalid_commits_without_any_progress() -> Result<()> {
        let mut q = Queue::default();
        let id = q.submit(vec![7; 65].into(), 20, true, t(1000), t(0))?;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 1, &[19], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.admit(id, 0, t(0))?;
        for (inputs, next) in [
            (1, vec![]),
            (1, vec![18, 19]),
            (8193, vec![19]),
            (usize::MAX, vec![19]),
        ] {
            let before = publication_snapshot(&q);
            assert!(q.worked_many(id, inputs, &next, t(1)).is_err());
            assert_eq!(publication_snapshot(&q), before);
        }
        q.worked(id, 65, 19, t(2))?;
        for (inputs, next) in [
            (0, vec![]),
            (0, vec![20]),
            (1, vec![20, 21]),
            (2, vec![20]),
            (17, vec![20; 17]),
            (3, vec![20, -1, 21]),
            (3, vec![20, 154880, 21]),
        ] {
            let before = publication_snapshot(&q);
            assert!(q.worked_many(id, inputs, &next, t(1)).is_err());
            assert_eq!(publication_snapshot(&q), before);
        }
        assert!(!q.worked_many(id, DECODE_COMMIT_ROWS, &[20; DECODE_COMMIT_ROWS], t(4))?);
        assert_eq!(q.request(id)?.decode_tokens, 16);
        assert_eq!(q.request(id)?.decode_calls, 1);
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 4, &[21; 4], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        assert!(q.worked_many(id, 3, &[21; 3], t(1))?);
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 1, &[22], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        assert_eq!(q.request(id)?.decode_tokens, 19);
        assert_eq!(q.request(id)?.outputs.len(), 20);
        Ok(())
    }
    #[test]
    fn atomic_decode_pause_cancel_and_other_owners_preserve_pending_state() -> Result<()> {
        let mut q = Queue::default();
        let bg = q.submit(vec![7].into(), 8, true, t(1000), t(0))?;
        let fg = q.submit(vec![8].into(), 2, false, t(1000), t(0))?;
        q.admit(bg, 0, t(0))?;
        q.admit(fg, 1, t(0))?;
        q.worked(bg, 1, 19, t(2))?;
        q.worked(fg, 1, 30, t(1))?;
        q.worked_many(bg, 2, &[20, 21], t(3))?;
        q.hold(true, t(10))?;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(bg, 3, &[22, 23, 24], t(4)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        assert_eq!(q.request(bg)?.outputs.last(), Some(&21));
        assert!(q.worked_many(fg, 1, &[31], t(1))?);
        q.finish(fg, End::Complete, t(12))?;
        q.hold(false, t(110))?;
        q.worked_many(bg, 3, &[22, 23, 24], t(4))?;
        let status = q.status(bg, t(110))?;
        assert_eq!(status.decode_calls, 2);
        assert_eq!(status.decode_tokens, 5);
        assert_eq!(status.output_tokens, 6);
        assert_eq!(status.active_ns, t(9).as_nanos());
        assert_eq!(status.decode_active_ns, t(7).as_nanos());
        assert_eq!(status.paused_ns, t(100).as_nanos());
        assert!(q.cancel(bg));
        let before = publication_snapshot(&q);
        assert!(q.worked_many(bg, 1, &[25], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.finish(bg, End::Cancelled, t(111))?;
        assert_eq!(q.completed[1].outputs, [19, 20, 21, 22, 23, 24]);
        Ok(())
    }
    #[test]
    fn atomic_decode_digest_covers_middle_tokens_and_committed_count() -> Result<()> {
        let mut a = Queue::default();
        let mut b = Queue::default();
        for q in [&mut a, &mut b] {
            let id = q.submit(vec![7].into(), 8, true, t(1000), t(0))?;
            q.admit(id, 0, t(0))?;
            q.worked(id, 1, 19, t(1))?;
        }
        a.worked_many(1, 3, &[20, 21, 22], t(2))?;
        b.worked_many(1, 3, &[20, 99, 22], t(2))?;
        assert_eq!(a.request(1)?.outputs.last(), b.request(1)?.outputs.last());
        assert_ne!(a.digest(), b.digest());
        let before = a.digest();
        a.requests.get_mut(&1).unwrap().decode_tokens += 1;
        assert_ne!(a.digest(), before);
        let invalid = publication_snapshot(&a);
        assert!(a.worked_many(1, 1, &[23], t(1)).is_err());
        assert_eq!(publication_snapshot(&a), invalid);
        Ok(())
    }
    #[test]
    fn atomic_decode_checks_arithmetic_before_mutating() -> Result<()> {
        let mut q = Queue::default();
        let id = q.submit(vec![7; 2].into(), 8, true, t(1000), t(0))?;
        q.admit(id, 0, t(0))?;
        q.requests.get_mut(&id).unwrap().prefill_calls = u64::MAX;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 1, &[19], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.requests.get_mut(&id).unwrap().prefill_calls = 0;
        q.worked(id, 2, 19, t(1))?;
        q.requests.get_mut(&id).unwrap().decode_calls = u64::MAX;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 2, &[20, 21], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        q.requests.get_mut(&id).unwrap().decode_calls = 0;
        q.requests.get_mut(&id).unwrap().work_time = Duration::MAX;
        let before = publication_snapshot(&q);
        assert!(q.worked_many(id, 2, &[20, 21], t(1)).is_err());
        assert_eq!(publication_snapshot(&q), before);
        Ok(())
    }
    #[test]
    fn atomic_decode_respects_policy_eos_but_preserves_fixed_length_diagnostics() -> Result<()> {
        let policy = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?;
        for with_policy in [false, true] {
            let mut q = Queue::default();
            let id = q.submit_policy(
                vec![7].into(),
                8,
                true,
                t(1000),
                t(0),
                with_policy.then_some(&policy),
            )?;
            q.admit(id, 0, t(0))?;
            q.worked(id, 1, 19, t(1))?;
            let before = publication_snapshot(&q);
            let crossing = q.worked_many(id, 3, &[20, 154827, 21], t(2));
            if with_policy {
                assert!(crossing.is_err());
                assert_eq!(publication_snapshot(&q), before);
                assert!(!q.worked_many(id, 2, &[20, 154827], t(2))?);
                let before = publication_snapshot(&q);
                assert!(q.worked_many(id, 1, &[21], t(1)).is_err());
                assert_eq!(publication_snapshot(&q), before);
                assert!(q.finish(id, End::Complete, t(3)).is_err());
                q.terminal(id, Completion::Eos)?;
                q.finish(id, End::Complete, t(3))?;
                assert_eq!(q.completed[0].outputs, [19, 20, 154827]);
                assert_eq!(q.completed[0].status.decode_tokens, 2);
                assert_eq!(q.completed[0].status.decode_calls, 1);
            } else {
                assert!(!crossing?);
                assert_eq!(q.request(id)?.outputs, [19, 20, 154827, 21]);
                assert!(q.terminal(id, Completion::Eos).is_err());
            }
        }
        Ok(())
    }
    #[test]
    fn omitted_intermediate_heads_preserve_progress_and_publication() -> Result<()> {
        let mut reference = Queue::default();
        let mut candidate = Queue::default();
        let now = Duration::ZERO;
        for q in [&mut reference, &mut candidate] {
            let id = q.submit(vec![7; 16385].into(), 2, true, Duration::from_secs(60), now)?;
            q.admit(id, 0, now)?;
        }
        let id = reference.requests.keys().next().copied().unwrap();
        reference.worked(id, 8192, 17, Duration::from_millis(3))?;
        candidate.worked_prefill(id, 8192, None, Duration::from_millis(3))?;
        assert_eq!(reference.digest(), candidate.digest());
        assert!(candidate.request(id)?.outputs.is_empty());
        for q in [&mut reference, &mut candidate] {
            q.hold(true, now)?;
        }
        let before = publication_snapshot(&candidate);
        assert!(
            candidate
                .worked_prefill(id, 8192, None, Duration::from_millis(4))
                .is_err()
        );
        assert_eq!(publication_snapshot(&candidate), before);
        for q in [&mut reference, &mut candidate] {
            q.hold(false, now)?;
        }
        reference.worked(id, 8192, 18, Duration::from_millis(4))?;
        candidate.worked_prefill(id, 8192, None, Duration::from_millis(4))?;
        assert_eq!(reference.digest(), candidate.digest());
        let before = publication_snapshot(&candidate);
        assert!(
            candidate
                .worked_prefill(id, 1, None, Duration::from_millis(1))
                .is_err()
        );
        assert_eq!(publication_snapshot(&candidate), before);
        reference.worked(id, 1, 21, Duration::from_millis(1))?;
        candidate.worked_prefill(id, 1, Some(21), Duration::from_millis(1))?;
        assert_eq!(reference.digest(), candidate.digest());
        let before = publication_snapshot(&candidate);
        assert!(
            candidate
                .worked_prefill(id, 1, Some(22), Duration::from_millis(2))
                .is_err()
        );
        assert!(
            candidate
                .worked_many(id, 0, &[], Duration::from_millis(2))
                .is_err()
        );
        assert_eq!(publication_snapshot(&candidate), before);
        for q in [&mut reference, &mut candidate] {
            assert!(q.worked(id, 1, 22, Duration::from_millis(2))?);
        }
        assert_eq!(reference.digest(), candidate.digest());
        assert_eq!(candidate.request(id)?.outputs, [21, 22]);
        Ok(())
    }
    #[test]
    fn large_prefill_keeps_exact_progress_and_pause_accounting() -> Result<()> {
        let mut q = Queue::default();
        let now = Duration::ZERO;
        let id = q.submit(vec![7; 16385].into(), 2, true, Duration::from_secs(60), now)?;
        q.admit(id, 0, now)?;
        let before = q.digest();
        assert!(q.worked(id, 8193, 19, Duration::from_millis(1)).is_err());
        assert_eq!(q.digest(), before);
        assert!(!q.worked(id, 8192, 19, Duration::from_millis(3))?);
        assert_eq!(q.request(id)?.done, 8192);
        assert!(q.request(id)?.outputs.is_empty());
        q.hold(true, now)?;
        assert!(q.worked(id, 8192, 20, Duration::from_millis(4)).is_err());
        assert_eq!(q.request(id)?.done, 8192);
        q.hold(false, now)?;
        assert!(!q.worked(id, 8192, 20, Duration::from_millis(4))?);
        assert!(!q.worked(id, 1, 21, Duration::from_millis(1))?);
        assert_eq!(q.request(id)?.outputs, [21]);
        assert!(q.worked(id, 1, 22, Duration::from_millis(2))?);
        let r = q.request(id)?;
        assert_eq!((r.done, r.prefill_calls, r.decode_calls), (16385, 3, 1));
        assert_eq!(r.prefill_time, Duration::from_millis(8));
        assert_eq!(r.decode_time, Duration::from_millis(2));
        assert_eq!(r.work_time, Duration::from_millis(10));
        assert_eq!(r.outputs, [21, 22]);
        Ok(())
    }
    #[test]
    fn lifetime_totals_survive_history_eviction_and_failed_retirement() -> Result<()> {
        let mut q = Queue::default();
        let t = Duration::ZERO;
        for _ in 0..24 {
            let id = q.submit(vec![7; 1025].into(), 2, true, Duration::from_secs(60), t)?;
            q.admit_cached(id, 0, 1024, t)?;
            q.worked(id, 1, 19, Duration::from_millis(2))?;
            q.worked(id, 1, 20, Duration::from_millis(3))?;
            q.finish(id, End::Complete, t)?;
            assert!(q.finish(id, End::Complete, t).is_err());
        }
        assert_eq!(q.completed.len(), HISTORY);
        assert_eq!(q.totals.requests, 24);
        assert_eq!(q.totals.prompt_tokens, 24 * 1025);
        assert_eq!(q.totals.cached_tokens, 24 * 1024);
        assert_eq!(q.totals.completion_tokens, 48);
        assert_eq!(q.totals.decode_rounds, 24);
        assert_eq!(q.totals.prefill_ns, 48_000_000);
        assert_eq!(q.totals.decode_ns, 72_000_000);
        for reason in [
            End::Cancelled,
            End::TimedOut,
            End::Rejected,
            End::OutputError,
        ] {
            let id = q.submit(vec![7; 64].into(), 1, true, Duration::from_secs(60), t)?;
            q.finish(id, reason, t)?;
        }
        assert_eq!(q.totals.requests, 28);
        assert_eq!(q.totals.errors, 2);
        assert_eq!(
            (q.totals.cancelled, q.totals.timed_out, q.totals.rejected),
            (1, 1, 1)
        );
        assert_eq!(q.totals.completion_tokens, 48);
        assert_eq!(q.telemetry(t)["totals"]["requests"], 28);
        Ok(())
    }
    #[test]
    fn restoring_slots_cannot_run_or_retire_and_forwarded_cancel_does_not_spin() -> Result<()> {
        let mut q = Queue::default();
        let t = Duration::ZERO;
        let bg = q.submit(vec![7; 1025].into(), 4, true, Duration::from_secs(10), t)?;
        q.reserve_restore(bg, 0, t)?;
        assert!(!q.slot_free(0));
        assert_eq!(q.runnable(), None);
        assert_eq!(q.status(bg, t)?.phase, "restoring");
        assert_eq!(q.status(bg, t)?.cached_tokens, 0);
        assert!(q.worked(bg, 1, 8, Duration::ZERO).is_err());
        assert!(q.finish(bg, End::Cancelled, t).is_err());
        q.hold(true, t)?;
        q.cancel(bg);
        assert_eq!(q.cancellation(t)?, Some((bg, End::Cancelled)));
        q.forward_restore_cancel(bg)?;
        assert_eq!(q.cancellation(t)?, None);
        assert_eq!(q.wait(t, true), Duration::from_secs(1));
        let fg = q.submit(vec![9; 64].into(), 1, false, Duration::from_secs(20), t)?;
        q.admit(fg, 1, t)?;
        assert_eq!(q.runnable(), Some(fg));
        q.worked(fg, 64, 9, Duration::from_millis(1))?;
        q.finish(fg, End::Complete, t)?;
        assert!(q.request(bg)?.restoring);
        q.finish_restore(bg, 0)?;
        q.finish(bg, End::Cancelled, t)?;
        assert!(q.slot_free(0));
        let next = q.submit(vec![7; 1025].into(), 1, true, Duration::from_secs(20), t)?;
        q.reserve_restore(next, 0, t)?;
        q.hold(false, t)?;
        q.finish_restore(next, 1024)?;
        assert_eq!(q.runnable(), Some(next));
        assert_eq!(q.status(next, t)?.cached_tokens, 1024);
        assert!(q.worked(next, 1, 7, Duration::from_millis(1))?);
        q.finish(next, End::Complete, t)?;
        Ok(())
    }
    #[test]
    fn cached_admission_preserves_output_and_execution_accounting() -> Result<()> {
        let mut q = Queue::default();
        let id = q.submit(
            vec![7; 1025].into(),
            2,
            true,
            Duration::from_secs(60),
            Duration::ZERO,
        )?;
        let digest = q.digest();
        assert!(q.admit_cached(id, 0, 1025, Duration::ZERO).is_err());
        assert_eq!(q.digest(), digest);
        q.admit_cached(id, 2, 1024, Duration::ZERO)?;
        assert!(!q.slot_free(2));
        assert_eq!(q.status(id, Duration::ZERO)?.cached_tokens, 1024);
        assert!(q.request(id)?.outputs.is_empty());
        assert!(!q.worked(id, 1, 19, Duration::from_millis(2))?);
        q.hold(true, Duration::from_secs(1))?;
        assert!(q.worked(id, 1, 20, Duration::ZERO).is_err());
        q.hold(false, Duration::from_secs(3))?;
        assert!(q.worked(id, 1, 20, Duration::from_millis(1))?);
        q.finish(id, End::Complete, Duration::from_secs(4))?;
        let c = q.completed.back().unwrap();
        assert_eq!(
            (
                c.status.cached_tokens,
                c.status.prefill_calls,
                c.status.decode_calls
            ),
            (1024, 1, 1)
        );
        assert_eq!(c.status.active_ns, 3_000_000);
        assert_eq!(c.status.paused_ns, 2_000_000_000);
        assert_eq!(c.outputs, vec![19, 20]);
        assert!(q.slot_free(2));
        Ok(())
    }
    #[test]
    fn sampling_owner_digest_covers_actual_policy_and_survives_pause() -> Result<()> {
        let p = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?
            .with_sampling(crate::sampling::Sampling::new(99, 1.0, 20, 0.95)?)?;
        let mut q = Queue::default();
        let id = q.submit_policy(
            vec![7].into(),
            8,
            true,
            Duration::from_secs(90),
            Duration::ZERO,
            Some(&p),
        )?;
        q.admit(id, 0, Duration::ZERO)?;
        q.worked(id, 1, 19, Duration::from_millis(5))?;
        let digest = q.digest();
        q.requests
            .get_mut(&id)
            .unwrap()
            .sampling
            .as_mut()
            .unwrap()
            .seed = 100;
        assert_ne!(q.digest(), digest);
        q.requests.get_mut(&id).unwrap().sampling = p.sampling();
        assert_eq!(q.digest(), digest);
        q.hold(true, Duration::from_secs(1))?;
        assert_eq!(q.request(id)?.sampling(), p.sampling());
        assert_eq!(q.request(id)?.outputs, vec![19]);
        q.hold(false, Duration::from_secs(2))?;
        assert_eq!(q.request(id)?.sampling(), p.sampling());
        assert_eq!(q.request(id)?.outputs, vec![19]);
        Ok(())
    }
    #[test]
    fn successful_terminal_policy_and_eos_validation() -> Result<()> {
        let policy = crate::output_policy::Policy::new(false, false, &serde_json::json!("stop"))?;
        for (token, cause) in [
            (154820, Completion::Eos),
            (154827, Completion::Eos),
            (154829, Completion::Eos),
            (154828, Completion::Eos),
            (19, Completion::Stop),
        ] {
            let mut q = Queue::default();
            let id = q.submit_policy(
                vec![1].into(),
                4,
                false,
                Duration::from_secs(10),
                Duration::ZERO,
                Some(&policy),
            )?;
            assert!(q.terminal(id, cause).is_err());
            q.admit(id, 0, Duration::ZERO)?;
            q.worked(id, 1, token, Duration::ZERO)?;
            assert!(q.finish(id, End::Complete, Duration::ZERO).is_err());
            q.terminal(id, cause)?;
            assert!(q.worked(id, 1, 7, Duration::ZERO).is_err());
            q.finish(id, End::Complete, Duration::ZERO)?;
            assert_eq!(q.completed[0].completion, Some(cause));
            assert_eq!(q.completed[0].outputs, vec![token]);
            assert_eq!(q.gate.local_count(), 0);
        }
        let mut q = Queue::default();
        let id = q.submit(
            vec![1].into(),
            4,
            true,
            Duration::from_secs(10),
            Duration::ZERO,
        )?;
        q.admit(id, 0, Duration::ZERO)?;
        q.worked(id, 1, 154827, Duration::ZERO)?;
        assert!(q.terminal(id, Completion::Eos).is_err());
        assert!(q.finish(id, End::Complete, Duration::ZERO).is_err());
        Ok(())
    }
    #[test]
    fn output_error_is_not_success_or_another_owners_cancel() -> Result<()> {
        let policy = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?;
        let mut q = Queue::default();
        let a = q.submit_policy(
            vec![1].into(),
            4,
            true,
            Duration::from_secs(10),
            Duration::ZERO,
            Some(&policy),
        )?;
        let b = q.submit(
            vec![1].into(),
            4,
            true,
            Duration::from_secs(10),
            Duration::ZERO,
        )?;
        q.admit(a, 0, Duration::ZERO)?;
        q.admit(b, 1, Duration::ZERO)?;
        q.worked(a, 1, 19, Duration::ZERO)?;
        assert!(q.terminal(a, Completion::Eos).is_err());
        assert!(q.terminal(a, Completion::Stop).is_err());
        q.output_failed(a)?;
        q.finish(a, End::OutputError, Duration::ZERO)?;
        assert_eq!(q.completed[0].completion, None);
        assert!(q.request(b)?.cancel.is_none());
        assert!(q.terminal(a, Completion::Stop).is_err());
        Ok(())
    }
    fn t(n: u64) -> Duration {
        Duration::from_millis(n)
    }
    #[test]
    fn time_progress_freeze_and_foreground_order() -> Result<()> {
        let mut q = Queue::default();
        let bg = q.submit(vec![1; 513].into(), 2, true, t(10000), t(0))?;
        q.admit(bg, 0, t(5))?;
        q.worked(bg, 512, 4, t(2))?;
        q.hold(true, t(10))?;
        let a = q.status(bg, t(100))?;
        assert_eq!(a.active_ns, t(2).as_nanos());
        assert_eq!(a.paused_ns, t(90).as_nanos());
        assert_eq!(a.prefill_done, 512);
        assert_eq!(a.queued_ns, t(5).as_nanos());
        assert!(q.worked(bg, 1, 4, t(2)).is_err());
        assert!(q.runnable().is_none());
        let fg = q.submit(vec![2; 64].into(), 1, false, t(10000), t(100))?;
        assert_eq!(q.queued(true), Some(fg));
        q.admit(fg, 1, t(101))?;
        assert_eq!(q.runnable(), Some(fg));
        assert!(q.worked(fg, 64, 5, t(3))?);
        q.finish(fg, End::Complete, t(104))?;
        q.hold(false, t(110))?;
        assert_eq!(q.runnable(), Some(bg));
        assert!(!q.worked(bg, 1, 6, t(2))?);
        assert!(q.worked(bg, 1, 7, t(1))?);
        q.finish(bg, End::Complete, t(115))?;
        let b = &q.completed.back().unwrap().status;
        assert_eq!(b.active_ns, t(5).as_nanos());
        assert_eq!(b.paused_ns, t(100).as_nanos());
        assert_eq!((b.pause_count, b.resume_count), (1, 1));
        assert_eq!(b.decode_calls, 1);
        Ok(())
    }
    #[test]
    fn cancellation_timeout_identity_and_tickets() -> Result<()> {
        let mut q = Queue::default();
        let a = q.submit(vec![1; 64].into(), 2, false, t(20), t(0))?;
        assert!(q.gate.snapshot(t(0), false)?.hold);
        assert!(q.cancel(a));
        assert_eq!(q.cancellation(t(2))?, Some((a, End::Cancelled)));
        q.finish(a, End::Cancelled, t(3))?;
        assert!(!q.gate.snapshot(t(3), false)?.hold);
        let b = q.submit(vec![1; 64].into(), 2, true, t(20), t(4))?;
        assert!(b > a);
        q.admit(b, 0, t(5))?;
        q.hold(true, t(6))?;
        assert!(!q.cancel(a));
        assert!(q.cancellation(t(19))?.is_none());
        assert_eq!(q.cancellation(t(20))?, Some((b, End::TimedOut)));
        q.finish(b, End::TimedOut, t(20))?;
        assert_eq!(q.free_slot(), Some(0));
        assert_eq!(
            q.completed.back().unwrap().status.paused_ns,
            t(14).as_nanos()
        );
        Ok(())
    }
    #[test]
    fn foreground_reserves_queue_entry_and_model_slot() -> Result<()> {
        let mut q = Queue::default();
        for _ in 0..4 {
            q.submit(vec![1; 64].into(), 4, true, t(1000), t(0))?;
        }
        for slot in 0..3 {
            q.admit(slot as u64 + 1, slot, t(1))?;
        }
        assert_eq!(q.queued(false), None);
        assert!(q.admit(4, 3, t(1)).is_err());
        let fg = q.submit(vec![2; 64].into(), 4, false, t(1000), t(1))?;
        assert_eq!(q.queued(true), Some(fg));
        q.admit(fg, 3, t(1))?;
        assert_eq!(q.runnable(), Some(fg));
        q.finish(fg, End::Cancelled, t(2))?;
        q.finish(1, End::Cancelled, t(2))?;
        assert_eq!(q.queued(false), Some(4));
        Ok(())
    }
    #[test]
    fn capacity_bounds_history_and_invalid_transitions() -> Result<()> {
        let mut q = Queue::default();
        for _ in 0..REQUESTS - 1 {
            q.submit(vec![1; 1].into(), 1, true, t(1000), t(0))?;
        }
        q.submit(vec![1; 1].into(), 1, false, t(1000), t(0))?;
        assert!(q.submit(vec![1].into(), 1, true, t(1000), t(0)).is_err());
        assert!(q.admit(1, SLOTS, t(0)).is_err());
        q.admit(1, 0, t(0))?;
        assert!(q.admit(1, 1, t(0)).is_err());
        assert!(q.admit(2, 0, t(0)).is_err());
        assert!(q.finish(1, End::Complete, t(0)).is_err());
        for id in 1..=REQUESTS as u64 {
            q.finish(id, End::Rejected, t(1))?;
        }
        assert_eq!(q.completed.len(), HISTORY);
        assert!(
            q.submit(vec![1].into(), MAX_OUTPUT + 1, true, t(1000), t(1))
                .is_err()
        );
        assert!(q.submit(vec![1].into(), 0, true, t(1000), t(1)).is_err());
        assert!(
            q.submit(vec![154880].into(), 1, true, t(1000), t(1))
                .is_err()
        );
        assert!(q.submit(vec![1].into(), 1, true, t(1), t(1)).is_err());
        assert!(q.now(t(0)).is_err());
        let id = q.submit(vec![1].into(), 1, true, t(1000), t(1))?;
        q.finish(id, End::Cancelled, t(2))?;
        assert_eq!(q.completed.len(), HISTORY);
        assert_eq!(q.completed.front().unwrap().status.id, 2);
        Ok(())
    }
}
