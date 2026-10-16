//! Lifetime counters, independent of the bounded request-history ring.
use crate::request_queue::{End, Status};

#[derive(Default, serde::Serialize)]
pub(crate) struct Counters {
    pub requests: u64,
    pub errors: u64,
    pub cancelled: u64,
    pub timed_out: u64,
    pub rejected: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub decode_rounds: u64,
    pub prefill_ns: u128,
    pub decode_ns: u128,
    pub last_finished_ns: Option<u128>,
}
impl Counters {
    pub fn finish(&mut self, s: &Status, reason: End, now: u128) {
        self.requests = self.requests.saturating_add(1);
        self.errors = self.errors.saturating_add(u64::from(matches!(
            reason,
            End::OutputError | End::TimedOut
        )));
        self.cancelled = self
            .cancelled
            .saturating_add(u64::from(reason == End::Cancelled));
        self.timed_out = self
            .timed_out
            .saturating_add(u64::from(reason == End::TimedOut));
        self.rejected = self
            .rejected
            .saturating_add(u64::from(reason == End::Rejected));
        self.prompt_tokens = self.prompt_tokens.saturating_add(s.prompt_tokens as u64);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(s.output_tokens as u64);
        self.cached_tokens = self.cached_tokens.saturating_add(s.cached_tokens as u64);
        self.decode_rounds = self.decode_rounds.saturating_add(s.decode_calls);
        self.prefill_ns = self.prefill_ns.saturating_add(s.prefill_active_ns);
        self.decode_ns = self.decode_ns.saturating_add(s.decode_active_ns);
        self.last_finished_ns = Some(now);
    }
}
