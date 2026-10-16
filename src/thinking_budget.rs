//! Opt-in thinking-token budget, independent of CUDA and output rendering.
//!
//! Request parsing and policy transport feed retained native-owner state.
//! The owner resolves the optional field once, preserves state through a native
//! hold, and publishes a preview only after all-rank model/output commit succeeds.

const MAX_OUTPUT: usize = 131_072;
const CONTEXT: usize = 1_048_576;
use anyhow::{Context, Result, ensure};
use serde_json::Value;

/// Resolve budget fields after template effort overrides.
/// Integer strings use ASCII decimal syntax. Budget conversion permits booleans
/// and truncates floats, unlike the strict sampling/length fields.
pub(crate) fn parse(body: &Value, chat: bool, thinking: bool) -> Result<Option<usize>> {
    if !chat || !thinking {
        return Ok(None);
    }
    let selected = body
        .get("thinking_token_budget")
        .filter(|v| !v.is_null())
        .map(|v| ("thinking_token_budget", v))
        .or_else(|| {
            body.get("thinking_budget")
                .filter(|v| crate::chat_json::truthy(v))
                .map(|v| ("thinking_budget", v))
        });
    selected
        .map(|(name, value)| integer(value).with_context(|| format!("invalid {name}")))
        .transpose()
}

fn decimal(text: &str) -> Result<usize> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let bytes = digits.as_bytes();
    ensure!(!bytes.is_empty(), "budget integer required");
    let mut magnitude = 0usize;
    let mut count = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'_' {
            ensure!(
                i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit(),
                "budget integer underscore"
            );
        } else {
            ensure!(
                byte.is_ascii_digit(),
                "ASCII decimal budget integer required"
            );
            count += 1;
            // Bound decimal conversion to 4300 digits.
            ensure!(count <= 4300, "budget integer exceeds 4300 digits");
            magnitude = magnitude
                .saturating_mul(10)
                .saturating_add((byte - b'0') as usize);
        }
    }
    Ok(if negative { 0 } else { magnitude })
}

fn integer(value: &Value) -> Result<usize> {
    match value {
        Value::Bool(b) => Ok(usize::from(*b)),
        Value::String(s) => decimal(s),
        Value::Number(n) => {
            let text = n.to_string();
            if text.contains(['.', 'e', 'E']) {
                let x: f64 = text.parse()?;
                ensure!(x.is_finite(), "finite thinking budget required");
                Ok(x.max(0.0).trunc() as usize)
            } else {
                decimal(&text)
            }
        }
        _ => anyhow::bail!("thinking budget integer required"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Next {
    Sample,
    Force(i32),
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct State {
    budget: Option<usize>,
    max_output: usize,
    think_end: i32,
    emitted: usize,
    end_seen: bool,
    terminal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Speculative preview; serial serving uses accept.
pub(crate) struct Preview {
    pub state: State,
    pub kept: usize,
}

impl State {
    /// `budget` is already resolved from the request. None disables forcing;
    /// Some(0) immediately emits the forced delimiter. Budgets larger than the
    /// output limit are valid and cannot force a token beyond that limit.
    pub(crate) fn new(
        budget: Option<usize>,
        max_output: usize,
        prompt_tokens: usize,
        think_end: i32,
    ) -> std::result::Result<Self, &'static str> {
        if !(1..=MAX_OUTPUT).contains(&max_output)
            || prompt_tokens
                .checked_add(max_output)
                .is_none_or(|n| n > CONTEXT)
            || !(0..crate::sampling::VOCAB as i32).contains(&think_end)
        {
            return Err("thinking budget output/token bound");
        }
        Ok(Self {
            budget,
            max_output,
            think_end,
            emitted: 0,
            end_seen: false,
            terminal: false,
        })
    }

    pub(crate) fn next(&self) -> Next {
        if self.terminal || self.emitted == self.max_output {
            Next::End
        } else if !self.end_seen && self.budget.is_some_and(|cap| self.emitted >= cap) {
            Next::Force(self.think_end)
        } else {
            Next::Sample
        }
    }

    pub(crate) fn emitted(&self) -> usize {
        self.emitted
    }

    /// Visible-stop rendering may finish an already published token later in
    /// the same owner boundary. Do not accept or count that token a second time.
    pub(crate) fn finish(&mut self) -> std::result::Result<(), &'static str> {
        if self.emitted == 0 {
            return Err("termination before output");
        }
        self.terminal = true;
        Ok(())
    }

    /// Stable versioned bytes for all-rank queue agreement, not persistence.
    pub(crate) fn digest_bytes(&self) -> [u8; 32] {
        let mut out = [0; 32];
        out[0] = 1;
        out[1] = u8::from(self.budget.is_some());
        out[2..10].copy_from_slice(&(self.budget.unwrap_or(0) as u64).to_le_bytes());
        out[10..18].copy_from_slice(&(self.max_output as u64).to_le_bytes());
        out[18..22].copy_from_slice(&self.think_end.to_le_bytes());
        out[22..30].copy_from_slice(&(self.emitted as u64).to_le_bytes());
        out[30] = u8::from(self.end_seen);
        out[31] = u8::from(self.terminal);
        out
    }

    /// Publish one selected token, including a forced delimiter, after model
    /// agreement. The caller identifies EOS/visible-stop completion; it takes
    /// precedence over forcing another token. Errors leave the state unchanged.
    pub(crate) fn accept(
        &mut self,
        token: i32,
        terminal: bool,
    ) -> std::result::Result<(), &'static str> {
        if !(0..crate::sampling::VOCAB as i32).contains(&token) {
            return Err("output token outside vocabulary");
        }
        match self.next() {
            Next::End => return Err("output after termination"),
            Next::Force(expected) if token != expected => {
                return Err("thinking budget requires forced delimiter");
            }
            Next::Sample | Next::Force(_) => {}
        }
        self.emitted += 1;
        self.end_seen |= token == self.think_end;
        self.terminal = terminal;
        Ok(())
    }

    /// Clip only a serial-equivalent sampled path; never predictions following
    /// a rejected draft. The original state is unchanged. A zero-length preview
    /// is legitimate when a forced token or termination is already due, but the
    /// current verifier clip hook requires >=1, so the owner must handle those
    /// states before launching a verification window.
    ///
    /// A delimiter chosen naturally at/before the boundary disables forcing.
    /// A sampled delimiter after the boundary is still replaced by the forced
    /// token policy: no extra sampled output is published beyond the cap.
    #[allow(dead_code)] // Speculative preview; serial serving uses accept.
    pub(crate) fn preview_sampled(
        &self,
        tokens: &[i32],
        mut terminal: impl FnMut(i32) -> bool,
    ) -> std::result::Result<Preview, &'static str> {
        let mut state = *self;
        let mut kept = 0;
        for &token in tokens {
            if state.next() != Next::Sample {
                break;
            }
            state.accept(token, terminal(token))?;
            kept += 1;
        }
        Ok(Preview { state, kept })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    const END: i32 = 7;
    const EOS: i32 = 9;

    #[test]
    fn actual_field_precedence_and_zero_alias_distinction() -> Result<()> {
        for (body, expected) in [
            (json!({}), None),
            (json!({"thinking_token_budget":null}), None),
            (json!({"thinking_budget":0}), None),
            (json!({"thinking_budget":false}), None),
            (json!({"thinking_budget":[]}), None),
            (json!({"thinking_budget":{}}), None),
            (json!({"thinking_budget":""}), None),
            (json!({"thinking_budget":"0"}), Some(0)),
            (json!({"thinking_budget":true}), Some(1)),
            (json!({"thinking_token_budget":0}), Some(0)),
            (json!({"thinking_token_budget":false}), Some(0)),
            (
                json!({"thinking_token_budget":null,"thinking_budget":23}),
                Some(23),
            ),
            (
                json!({"thinking_token_budget":0,"thinking_budget":23}),
                Some(0),
            ),
            (
                json!({"thinking_token_budget":7,"thinking_budget":{}}),
                Some(7),
            ),
        ] {
            assert_eq!(parse(&body, true, true)?, expected, "{body}");
        }
        Ok(())
    }

    #[test]
    fn budget_integer_conversion_and_errors() -> Result<()> {
        for (value, expected) in [
            (json!(-999), 0),
            (json!(-0.9), 0),
            (json!(3.9), 3),
            (json!(" +2_048 "), 2048),
            (json!("-9_123"), 0),
            (json!("184467440737095516160"), usize::MAX),
            (json!(1e100), usize::MAX),
        ] {
            assert_eq!(
                parse(&json!({"thinking_token_budget":value}), true, true)?,
                Some(expected)
            );
        }
        for value in [
            json!(""),
            json!("+"),
            json!("2.5"),
            json!("2e3"),
            json!("NaN"),
            json!("_2"),
            json!("2_"),
            json!("2__3"),
            json!([1]),
            json!({}),
            json!("9".repeat(4301)),
        ] {
            let body = json!({"thinking_token_budget":value});
            assert!(parse(&body, true, true).is_err(), "{body}");
            assert_eq!(parse(&body, false, true)?, None);
            assert_eq!(parse(&body, true, false)?, None);
        }
        assert_eq!(
            parse(
                &json!({"thinking_token_budget":"9".repeat(4300)}),
                true,
                true
            )?,
            Some(usize::MAX)
        );
        let nonfinite: Value = serde_json::from_str("{\"thinking_token_budget\":1e999}")?;
        assert!(parse(&nonfinite, true, true).is_err());
        Ok(())
    }

    #[test]
    fn zero_forces_before_first_sample_and_counts_usage() {
        let mut state = State::new(Some(0), 3, 100, END).unwrap();
        assert_eq!(state.next(), Next::Force(END));
        assert_eq!(state.preview_sampled(&[1, 2], |_| false).unwrap().kept, 0);
        let before = state;
        assert!(state.accept(1, false).is_err());
        assert_eq!(state, before);
        state.accept(END, false).unwrap();
        assert_eq!((state.next(), state.emitted()), (Next::Sample, 1));
        state.accept(2, false).unwrap();
        state.accept(3, false).unwrap();
        assert_eq!((state.next(), state.emitted()), (Next::End, 3));
    }

    #[test]
    fn overall_limit_wins_and_disabled_budget_never_forces() {
        for budget in [None, Some(3), Some(4), Some(usize::MAX)] {
            let mut state = State::new(budget, 3, 100, END).unwrap();
            for token in [1, 2, 3] {
                assert_eq!(state.next(), Next::Sample);
                state.accept(token, false).unwrap();
            }
            assert_eq!(state.next(), Next::End);
            let before = state;
            assert!(state.accept(END, false).is_err());
            assert_eq!(state, before);
        }
    }

    #[test]
    fn natural_close_before_or_at_cap_never_reopens_thinking() {
        for tokens in [[END, 2, 3, 4], [1, END, 3, 4], [1, 2, END, 4]] {
            let state = State::new(Some(3), 5, 100, END).unwrap();
            let preview = state.preview_sampled(&tokens, |_| false).unwrap();
            assert_eq!(preview.kept, 4);
            assert_eq!(preview.state.next(), Next::Sample);
            assert_eq!(state.emitted(), 0);
        }
    }

    #[test]
    fn sampled_window_clips_before_force_without_mutating_owner() {
        let state = State::new(Some(3), 8, 100, END).unwrap();
        let preview = state
            .preview_sampled(&[1, 2, 3, END, 5], |_| false)
            .unwrap();
        assert_eq!((preview.kept, preview.state.next()), (3, Next::Force(END)));
        assert_eq!(state.emitted(), 0);
        let mut committed = preview.state;
        committed.accept(END, false).unwrap();
        let resumed = committed
            .preview_sampled(&[5, 6, 7, 8, 9], |_| false)
            .unwrap();
        assert_eq!((resumed.kept, resumed.state.next()), (4, Next::End));
        assert_eq!(resumed.state.emitted(), 8);
    }

    #[test]
    fn eos_and_visible_stops_count_but_prevent_forcing() {
        for stop_token in [EOS, 2] {
            let state = State::new(Some(3), 8, 100, END).unwrap();
            let preview = state
                .preview_sampled(&[1, stop_token, 3, 4], |id| id == stop_token)
                .unwrap();
            assert_eq!((preview.kept, preview.state.next()), (2, Next::End));
        }
        let state = State::new(Some(3), 8, 100, END).unwrap();
        let preview = state
            .preview_sampled(&[1, 2, EOS, 4], |id| id == EOS)
            .unwrap();
        assert_eq!((preview.kept, preview.state.next()), (3, Next::End));
    }

    #[test]
    fn invalid_inputs_and_native_hold_do_not_change_state() {
        assert!(State::new(None, 0, 100, END).is_err());
        assert!(State::new(None, MAX_OUTPUT + 1, 100, END).is_err());
        assert!(State::new(None, 1, 100, -1).is_err());
        assert!(State::new(None, 1, 100, crate::sampling::VOCAB as i32).is_err());
        assert!(State::new(None, 1, CONTEXT, END).is_err());
        assert!(State::new(None, 1, usize::MAX, END).is_err());
        let state = State::new(Some(3), 8, 100, END).unwrap();
        assert!(state.preview_sampled(&[1, -1], |_| false).is_err());
        assert!(
            state
                .preview_sampled(&[1, crate::sampling::VOCAB as i32], |_| false)
                .is_err()
        );
        assert_eq!(state.emitted(), 0);
        let held = state.preview_sampled(&[1, 2], |_| false).unwrap().state;
        let resumed = held.preview_sampled(&[3, 4], |_| false).unwrap();
        let uninterrupted = state.preview_sampled(&[1, 2, 3, 4], |_| false).unwrap();
        assert_eq!(resumed.state, uninterrupted.state);
    }

    #[test]
    fn forced_delimiter_uses_reserved_output_and_context_capacity() {
        let mut state = State::new(Some(0), 1, CONTEXT - 1, END).unwrap();
        state.accept(END, false).unwrap();
        assert_eq!((state.emitted(), state.next()), (1, Next::End));
        let mut state = State::new(Some(0), MAX_OUTPUT, CONTEXT - MAX_OUTPUT, END).unwrap();
        state.accept(END, false).unwrap();
        for _ in 1..MAX_OUTPUT {
            state.accept(1, false).unwrap();
        }
        assert_eq!((state.emitted(), state.next()), (MAX_OUTPUT, Next::End));
        assert!(State::new(Some(0), MAX_OUTPUT, CONTEXT - MAX_OUTPUT + 1, END).is_err());
    }

    #[test]
    fn retained_digest_covers_policy_and_every_state_transition() {
        let base = State::new(None, 8, 100, END).unwrap();
        for other in [
            State::new(Some(0), 8, 100, END).unwrap(),
            State::new(Some(7), 8, 100, END).unwrap(),
            State::new(None, 7, 100, END).unwrap(),
            State::new(None, 8, 100, END + 1).unwrap(),
        ] {
            assert_ne!(base.digest_bytes(), other.digest_bytes());
        }
        let mut sampled = base;
        sampled.accept(1, false).unwrap();
        let mut closed = base;
        closed.accept(END, false).unwrap();
        let mut stopped = base;
        stopped.accept(1, true).unwrap();
        assert_ne!(base.digest_bytes(), sampled.digest_bytes());
        assert_ne!(sampled.digest_bytes(), closed.digest_bytes());
        assert_ne!(sampled.digest_bytes(), stopped.digest_bytes());
        let before = base;
        let mut unstarted = base;
        assert!(unstarted.finish().is_err());
        assert_eq!(before, unstarted);
        sampled.finish().unwrap();
        assert_eq!(sampled, stopped);
        let finished = sampled;
        sampled.finish().unwrap();
        assert_eq!(sampled, finished);
    }
}
