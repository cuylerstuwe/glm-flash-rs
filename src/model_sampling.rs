//! Low-level sampled model continuation. Request-policy/wire integration remains
//! separate: the caller must agree the same policy/owner on every rank first.
use super::{Engine, Long, ROWS};
use crate::{
    candidates::{Candidates, LOCAL, payload_bytes},
    prepared::Prepared,
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{net::SocketAddr, path::Path};

impl Long<'_> {
    /// Serial constrained draw. The owner must collectively validate the
    /// exact mask, policy, slot and cursor before calling this method, then
    /// collectively commit its result before exposing the pending token.
    #[allow(dead_code)] // Direct sampling diagnostic entry point.
    pub(super) fn sample_masked(
        &mut self,
        sampler: &mut crate::masked_candidates::MaskedCandidates,
        policy: Sampling,
        mask: &[i32],
    ) -> Result<(u32, usize)> {
        self.ready_active()?;
        let session = &self.sessions[self.active];
        ensure!(
            self.latest_logits == Some(self.active)
                && session.pending.is_some()
                && session.tokens.len() >= session.prompt_end
                && mask.len() == crate::masked_candidates::MASK_WORDS,
            "masked sample requires current logits and canonical vocabulary mask"
        );
        let position = session.tokens.len() as u64;
        let k = policy.local_count();
        payload_bytes(k)?;
        self.healthy = false;
        unsafe {
            sampler.upload_mask(&self.engine.stream, mask)?;
        }
        sampler.local(
            &self.engine.stream,
            &self.engine.logits,
            self.engine.rank,
            policy,
        )?;
        unsafe {
            self.engine
                .gather_prefix(&sampler.local, &sampler.gathered, payload_bytes(k)?)?;
        }
        let gpu = policy.gpu_fits(4 * k);
        if gpu {
            unsafe {
                sampler.gpu(&self.engine.stream, policy, position)?;
            }
        }
        self.engine.sync()?;
        let result = if gpu {
            sampler.read_gpu(policy)?
        } else {
            sampler.read_host(policy, position)?
        };
        ensure!(
            (mask[result.0 as usize / 32] as u32 >> (result.0 % 32)) & 1 == 1,
            "masked sampler selected a forbidden token"
        );
        self.sessions[self.active].pending = Some(result.0 as i32);
        self.healthy = true;
        Ok(result)
    }
    /// Replace the just-computed pending selection at the same cursor. The
    /// next model step consumes this token; no input is replayed and no random
    /// draw is performed for a policy-forced thinking delimiter.
    pub(super) fn force_pending(&mut self, token: i32) -> Result<()> {
        self.ready_active()?;
        let session = &mut self.sessions[self.active];
        ensure!(
            self.latest_logits == Some(self.active)
                && session.pending.is_some()
                && session.tokens.len() >= session.prompt_end
                && session.tokens.len() < session.total_end
                && (0..sampling::VOCAB as i32).contains(&token),
            "forced output requires current completed prompt/decode logits"
        );
        session.pending = Some(token);
        Ok(())
    }
    /// Draw only after a completed model head for this active owner. A pause
    /// retains the already selected pending token; other owners may overwrite
    /// shared logits, so resuming must consume pending before drawing again.
    pub(super) fn sample(
        &mut self,
        sampler: &mut Candidates,
        policy: Sampling,
    ) -> Result<(u32, usize)> {
        self.ready_active()?;
        let session = &self.sessions[self.active];
        ensure!(
            self.latest_logits == Some(self.active)
                && session.pending.is_some()
                && session.tokens.len() >= session.prompt_end,
            "sample requires current completed prompt/decode logits"
        );
        if policy.greedy() {
            let token = self.engine.selection.read()?;
            self.sessions[self.active].pending = Some(token);
            return Ok((token as u32, 1));
        }
        let position = session.tokens.len() as u64;
        let k = policy.local_count();
        payload_bytes(k)?;
        self.healthy = false;
        sampler.local(
            &self.engine.stream,
            &self.engine.logits,
            self.engine.rank,
            k,
        )?;
        unsafe {
            self.engine
                .gather_prefix(&sampler.local, &sampler.gathered, payload_bytes(k)?)?;
        }
        let gpu = policy.gpu_fits(4 * k);
        if gpu {
            unsafe {
                sampler.gpu(&self.engine.stream, policy, position)?;
            }
        }
        self.engine.sync()?;
        let result = if gpu {
            sampler.read_gpu(policy)?
        } else {
            sampler.read_host(policy, position)?
        };
        self.sessions[self.active].pending = Some(result.0 as i32);
        self.healthy = true;
        Ok(result)
    }
}
// Independent CPU selection from audit logits, using the frozen ATen key/tie
// rule validated against native fixtures. Does not read GPU candidate buffers.
fn reference(long: &Long<'_>, policy: Sampling) -> Result<(u32, usize)> {
    let bytes = long
        .engine
        .all_logits
        .read(0, long.engine.all_logits.bytes)?;
    let mut selected = Vec::new();
    for (rank, row) in bytes.as_chunks::<{ LOCAL * 2 }>().0.iter().enumerate() {
        let mut values: Vec<_> = row
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(id, b)| {
                let b = u16::from_le_bytes(*b);
                let key = if b & 0x8000 != 0 {
                    b ^ 0xffff
                } else {
                    b ^ 0x8000
                };
                (key, id, b)
            })
            .collect();
        values.sort_unstable_by_key(|&(key, id, _)| (std::cmp::Reverse(key), id));
        for &(_, id, b) in values.iter().take(policy.local_count()) {
            selected.push(Candidate {
                id: (rank * LOCAL + id) as u32,
                value: f32::from_bits((b as u32) << 16),
            });
        }
    }
    sampling::choose(
        &selected,
        long.sessions[long.active].tokens.len() as u64,
        policy,
    )
}
pub(super) fn checked_sample(
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    policy: Sampling,
) -> Result<i32> {
    let expected = reference(long, policy)?;
    let actual = long.sample(sampler, policy)?;
    ensure!(
        actual == expected,
        "sampled model differs from full-logit host reference"
    );
    Ok(actual.0 as i32)
}
#[derive(Clone, serde::Serialize, PartialEq, Eq)]
struct Trace {
    tokens: Vec<i32>,
    logits: Vec<String>,
    prefill_state: String,
    final_state: String,
}
fn cold(
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    prompt: &[i32],
    policy: Sampling,
    output: usize,
) -> Result<Trace> {
    long.start(prompt.len(), output)?;
    ensure!(
        long.sample(sampler, policy).is_err(),
        "sample before prefill accepted"
    );
    for (n, chunk) in prompt.chunks(ROWS).enumerate() {
        long.prefill(n * ROWS, chunk)?;
    }
    let mut trace = Trace {
        tokens: Vec::new(),
        logits: Vec::new(),
        prefill_state: long.state(prompt.len())?,
        final_state: String::new(),
    };
    for step in 0..output {
        if step > 0 {
            long.step(prompt.len() + step - 1, trace.tokens[step - 1])?;
        }
        trace.tokens.push(checked_sample(long, sampler, policy)?);
        trace.logits.push(long.logits()?);
    }
    trace.final_state = long.state(prompt.len() + output - 1)?;
    long.finish()?;
    Ok(trace)
}
#[allow(clippy::too_many_arguments)] // Same diagnostic fleet ABI as other model controls.
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    ensure!(
        !token_sets.is_empty() && token_sets.iter().all(|t| !t.is_empty()) && new_tokens >= 8,
        "sample model fixture shape"
    );
    let short = token_sets.iter().min_by_key(|v| v.len()).unwrap();
    let short = &short[..short.len().min(400)];
    let long_prompt = token_sets.iter().max_by_key(|v| v.len()).unwrap();
    let output = 8;
    let prompts = [short, long_prompt.as_slice(), short, short];
    let policies = [
        Sampling::new(450, 1.0, 20, 0.95)?,
        Sampling::new(1234, 0.7, 20, 0.95)?,
        Sampling::new(u64::MAX, 1.0, 504, 0.5)?,
        Sampling::new(0, 1.0, 154880, 0.95)?,
    ];
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(short[0])?;
    engine.enable_prefill(root, native)?;
    let mut sampler = Candidates::new(&engine.device, native)?;
    let mut long = Long::new(engine, root)?;
    let mut traces = Vec::new();
    for i in 0..4 {
        eprintln!("sampled cold case {i}, prompt={}", prompts[i].len());
        traces.push(cold(
            &mut long,
            &mut sampler,
            prompts[i],
            policies[i],
            output,
        )?);
    }
    ensure!(
        cold(&mut long, &mut sampler, prompts[0], policies[0], output)? == traces[0],
        "seeded cold replay changed"
    );
    // Four policy owners, prefill and decode interleaved, including a native hold
    // on all background work. This exercises the model boundary directly;
    // request-queue integration is covered by separate serving tests.
    for (slot, prompt) in prompts.iter().enumerate() {
        long.bind(slot)?;
        long.start_mode(prompt.len(), output, slot < 3)?;
        long.pause()?;
        ensure!(
            long.sample(&mut sampler, policies[slot]).is_err(),
            "paused sample accepted"
        );
    }
    let mut emitted = [0; 4];
    while emitted.iter().any(|&n| n < output) {
        for slot in 0..4 {
            if emitted[slot] == output {
                continue;
            }
            long.bind(slot)?;
            long.priority_hold = true;
            if slot < 3 {
                ensure!(long.resume().is_err(), "background resumed through hold");
            }
            long.priority_hold = false;
            long.resume()?;
            let position = long.sessions[slot].tokens.len();
            if position < prompts[slot].len() {
                let end = (position + ROWS).min(prompts[slot].len());
                long.prefill(position, &prompts[slot][position..end])?;
                if end < prompts[slot].len() {
                    ensure!(
                        long.sample(&mut sampler, policies[slot]).is_err(),
                        "partial prompt sampled"
                    );
                    long.pause()?;
                    continue;
                }
                ensure!(
                    long.state(end)? == traces[slot].prefill_state,
                    "interleaved prefill state changed"
                );
            } else {
                let pending = long.sessions[slot].pending.unwrap();
                ensure!(
                    pending == traces[slot].tokens[emitted[slot] - 1],
                    "paused sampled token changed"
                );
                long.step(position, pending)?;
            }
            let n = emitted[slot];
            ensure!(
                checked_sample(&mut long, &mut sampler, policies[slot])? == traces[slot].tokens[n]
                    && long.logits()? == traces[slot].logits[n],
                "interleaved sampled output/logits changed"
            );
            emitted[slot] += 1;
            if emitted[slot] == output {
                ensure!(
                    long.state(prompts[slot].len() + output - 1)? == traces[slot].final_state,
                    "interleaved sampled state changed"
                );
                long.finish()?;
            } else {
                long.pause()?;
            }
        }
    }
    // Pure cache restoration does not restore logits or mutable RNG. The caller
    // retains the seed; consuming the remaining prompt establishes absolute position.
    long.bind(0)?;
    long.start(long_prompt.len(), output)?;
    let cut = ((long_prompt.len() / 2) / 64 * 64).max(64);
    ensure!(cut < long_prompt.len(), "prefix fixture needs >64 tokens");
    for (n, chunk) in long_prompt[..cut].chunks(ROWS).enumerate() {
        long.prefill(n * ROWS, chunk)?;
    }
    let checkpoint = long.checkpoint()?;
    long.cancel()?;
    let _ = cold(&mut long, &mut sampler, prompts[3], policies[3], output)?;
    ensure!(
        long.restore_prefix(&checkpoint, long_prompt, output)? == cut,
        "restored prefix length"
    );
    ensure!(
        long.sample(&mut sampler, policies[1]).is_err(),
        "restored stale logits accepted"
    );
    for (n, chunk) in long_prompt[cut..].chunks(ROWS).enumerate() {
        long.prefill(cut + n * ROWS, chunk)?;
    }
    for n in 0..output {
        if n > 0 {
            long.step(long_prompt.len() + n - 1, traces[1].tokens[n - 1])?;
        }
        ensure!(
            checked_sample(&mut long, &mut sampler, policies[1])? == traces[1].tokens[n]
                && long.logits()? == traces[1].logits[n],
            "restored sampled result changed"
        );
    }
    ensure!(
        long.state(long_prompt.len() + output - 1)? == traces[1].final_state,
        "restored sampled state changed"
    );
    long.cancel()?;
    ensure!(
        cold(&mut long, &mut sampler, prompts[0], policies[0], output)? == traces[0],
        "cancelled owner reuse changed"
    );
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_lengths":prompts.map(<[i32]>::len),
        "output_tokens_per_case":output,"traces":traces,"sampled_tokens_match_independent_full_logits":true,
        "cold_repeat_exact":true,"four_slot_interleave_exact":true,"paused_pending_token_retained":true,
        "background_hold_enforced":true,"pure_restore_exact":true,"prefix_tokens":cut,"cancel_reuse_exact":true,
        "python_executed":false,"scope":"low-level fixed-length sampled model and retained slots; no sampled request/wire or HTTP integration"}),
    )
}
