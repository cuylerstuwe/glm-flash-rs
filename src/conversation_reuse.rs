//! Slot-local mixed-history continuation and diagnostic replay qualification.
use super::{Engine, Long, ROWS};
use crate::{
    cache_pages::{CONTEXT, MAX_OUTPUT, PAGE, Phase},
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use std::{net::SocketAddr, path::Path, time::Instant};

pub(super) fn eligible(
    history: &[i32],
    mixed: bool,
    pages: usize,
    request: &[i32],
    output: usize,
) -> Result<()> {
    ensure!(
        output > 0 && output <= MAX_OUTPUT && request.len() <= CONTEXT - output,
        "continuation limits"
    );
    ensure!(
        !history.is_empty() && (mixed || history.len().is_multiple_of(64)),
        "live prefix arithmetic/grid"
    );
    ensure!(
        pages >= history.len().div_ceil(PAGE),
        "live prefix was spilled"
    );
    ensure!(
        request.len() > history.len()
            && request.starts_with(history)
            && request
                .iter()
                .all(|&t| crate::vision_input::prompt_token(t)),
        "live prefix does not match request"
    );
    Ok(())
}
impl Long<'_> {
    pub(super) fn continue_request(
        &mut self,
        request: &[i32],
        output: usize,
        background: bool,
    ) -> Result<usize> {
        self.ready()?;
        let live = self.memory.book().slot(self.active)?;
        ensure!(
            live.phase() == Phase::Idle,
            "continuation requires a completed idle request"
        );
        let session = &self.sessions[self.active];
        eligible(
            &session.tokens,
            session.mixed,
            live.pages().len(),
            request,
            output,
        )?;
        let n = session.tokens.len();
        let plan = self.admission(request.len(), output, n, background)?;
        self.healthy = false;
        self.commit_admission(&plan, n)?;
        self.engine.sync()?;
        let session = &mut self.sessions[self.active];
        session.pending = None;
        session.prompt_end = request.len();
        session.total_end = request.len() + output;
        session.background = background;
        self.latest_logits = None;
        self.sync_speculation_holds()?;
        self.healthy = self.speculation.is_none();
        Ok(n)
    }
}

struct Operation {
    fast: bool,
    ids: Vec<i32>,
    next: i32,
    logits: String,
}
fn record(
    long: &Long<'_>,
    operations: &mut Vec<Operation>,
    fast: bool,
    ids: &[i32],
    next: i32,
) -> Result<()> {
    operations.push(Operation {
        fast,
        ids: ids.to_vec(),
        next,
        logits: long.logits()?,
    });
    Ok(())
}
fn suffix(long: &mut Long<'_>, ids: &[i32], operations: &mut Vec<Operation>) -> Result<i32> {
    let mut offset = 0;
    let mut next = 0;
    // Only a mixed prefix may take this exact head. Pure fast-prefix resume is
    // constrained to its 64-token grid, as in the retained engine.
    if !long.sessions[long.active].tokens.len().is_multiple_of(64) {
        ensure!(
            long.sessions[long.active].mixed,
            "unaligned pure fast prefix"
        );
        while offset < ids.len() && !long.sessions[long.active].tokens.len().is_multiple_of(64) {
            next = long.exact_prompt_token(ids[offset])?;
            record(long, operations, false, &ids[offset..offset + 1], next)?;
            offset += 1;
        }
    }
    for chunk in ids[offset..].chunks(ROWS) {
        next = long.prefill(long.sessions[long.active].tokens.len(), chunk)?;
        record(long, operations, true, chunk, next)?;
    }
    Ok(next)
}
fn generate(
    long: &mut Long<'_>,
    mut next: i32,
    operations: &mut Vec<Operation>,
) -> Result<Vec<i32>> {
    // Deliberately continue through stop IDs: this is an internal state test,
    // not a user-visible server response or stop-policy implementation.
    let mut outputs = vec![next];
    for _ in 0..3 {
        let input = next;
        next = long.step(long.sessions[long.active].tokens.len(), input)?;
        record(long, operations, false, &[input], next)?;
        outputs.push(next);
    }
    Ok(outputs)
}
fn cross_slot(long: &mut Long<'_>, tokens: &[i32]) -> Result<serde_json::Value> {
    let cut = (tokens.len().saturating_sub(1).min(2112) / 64) * 64;
    ensure!(cut > 0, "cross-slot fixture too short");
    long.bind(0)?;
    long.start(tokens.len(), 4)?;
    let mut ignored = Vec::new();
    suffix(long, &tokens[..cut], &mut ignored)?;
    let prefix_state = long.state(cut)?;
    let checkpoint = long.checkpoint()?;
    let next = suffix(long, &tokens[cut..], &mut ignored)?;
    let prefill_state = long.state(tokens.len())?;
    let prefill_logits = long.logits()?;
    let outputs = generate(long, next, &mut ignored)?;
    let reference = long.state(tokens.len() + 3)?;
    let reference_logits: Vec<_> = ignored
        .iter()
        .rev()
        .take(3)
        .map(|o| o.logits.clone())
        .collect();
    long.finish()?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[7; 64])?;
    let protected = long.state(64)?;
    long.pause()?;
    long.bind(1)?;
    long.start(64, 4)?;
    long.prefill(0, &[8; 64])?;
    long.finish()?;
    let reused = long.restore_prefix(&checkpoint, tokens, 4)?;
    ensure!(
        reused == cut && long.state(cut)? == prefix_state,
        "cross-slot restored prefix state"
    );
    let mut actual = Vec::new();
    let next = suffix(long, &tokens[cut..], &mut actual)?;
    ensure!(
        long.state(tokens.len())? == prefill_state && long.logits()? == prefill_logits,
        "cross-slot prefill differs"
    );
    ensure!(
        generate(long, next, &mut actual)? == outputs,
        "cross-slot output differs"
    );
    ensure!(
        actual
            .iter()
            .rev()
            .take(3)
            .map(|o| o.logits.clone())
            .collect::<Vec<_>>()
            == reference_logits
            && long.state(tokens.len() + 3)? == reference,
        "cross-slot decode/state differs"
    );
    long.finish()?;
    long.bind(0)?;
    ensure!(
        long.state(tokens.len() + 3)? == reference,
        "restoration overwrote source slot"
    );
    long.bind(2)?;
    ensure!(
        long.state(64)? == protected,
        "restoration overwrote paused slot"
    );
    long.cancel()?;
    long.memory.book().check()?;
    Ok(
        serde_json::json!({"prompt_tokens":tokens.len(),"reused_model_tokens":reused,"source_and_paused_owner_unchanged":true,"logits_and_state_exact":true,"generated_tokens":outputs}),
    )
}
fn live(long: &mut Long<'_>, tokens: &[i32]) -> Result<serde_json::Value> {
    let initial = &tokens[..tokens.len().min(8192)];
    long.bind(0)?;
    long.start(initial.len(), 4)?;
    let mut operations = Vec::new();
    let next = suffix(long, initial, &mut operations)?;
    generate(long, next, &mut operations)?;
    long.finish()?;
    let mut turns = Vec::new();
    for added in [17, 129, 63] {
        long.bind(0)?;
        let history = long.sessions[0].tokens.clone();
        let n = history.len();
        let state = long.state(n)?;
        let pages = long.memory.book().slot(0)?.pages().to_vec();
        let mut request = history.clone();
        request.push(long.sessions[0].pending.unwrap());
        request.extend(std::iter::repeat_n(7, added - 1));
        let mut wrong = request.clone();
        wrong[0] = (wrong[0] + 1) % 154880;
        ensure!(
            long.continue_request(&wrong, 4, false).is_err(),
            "changed live prefix accepted"
        );
        ensure!(
            long.continue_request(&history, 4, false).is_err(),
            "non-extending prefix accepted"
        );
        ensure!(
            long.state(n)? == state && long.memory.book().slot(0)?.phase() == Phase::Idle,
            "invalid live continuation mutated owner"
        );
        let began = Instant::now();
        let reused = long.continue_request(&request, 4, false)?;
        let admission_seconds = began.elapsed().as_secs_f64();
        ensure!(
            reused == n && long.state(n)? == state && long.memory.book().slot(0)?.pages() == pages,
            "live continuation lost retained state/pages"
        );
        long.pause()?;
        long.bind(3)?;
        long.start(64, 4)?;
        long.prefill(0, &[9; 64])?;
        let protected = long.state(64)?;
        long.pause()?;
        long.bind(0)?;
        long.resume()?;
        let before = operations.len();
        let next = suffix(long, &request[n..], &mut operations)?;
        let prompt_state = long.state(request.len())?;
        let prompt_logits = long.logits()?;
        let outputs = generate(long, next, &mut operations)?;
        let final_state = long.state(request.len() + 3)?;
        ensure!(
            long.checkpoint().is_err(),
            "mixed conversation entered pure prefix namespace"
        );
        long.finish()?;
        // Replay the same arithmetic schedule from empty state in another slot,
        // forcing the recorded inputs. No cached state is used by this reference.
        long.bind(1)?;
        long.start(request.len() + 3, 1)?;
        for (i, op) in operations.iter().enumerate() {
            let position = long.sessions[1].tokens.len();
            let next = if op.fast {
                long.prefill(position, &op.ids)?
            } else {
                long.exact_prompt_token(op.ids[0])?
            };
            ensure!(
                next == op.next && long.logits()? == op.logits,
                "live replay logits mismatch operation={i}"
            );
            if long.sessions[1].tokens.len() == request.len() {
                ensure!(
                    long.state(request.len())? == prompt_state && long.logits()? == prompt_logits,
                    "live replay prefill state mismatch"
                );
            }
        }
        ensure!(
            long.state(request.len() + 3)? == final_state,
            "live replay final state mismatch"
        );
        long.finish()?;
        long.bind(3)?;
        ensure!(
            long.state(64)? == protected,
            "continued request overwrote paused owner"
        );
        long.cancel()?;
        let exact_head = operations[before..operations.len() - 3]
            .iter()
            .filter(|o| !o.fast)
            .count();
        turns.push(serde_json::json!({"reused_model_tokens":reused,"new_prompt_tokens":added,"exact_head_tokens":exact_head,"admission_seconds":admission_seconds,"generated_tokens":outputs,"prefill_state_sha256":prompt_state,"final_state_sha256":final_state,"all_replay_logits_exact":true,"retained_pages_unchanged":true,"paused_owner_unchanged":true}));
        long.memory.book().check()?;
    }
    Ok(
        serde_json::json!({"initial_prompt_tokens":initial.len(),"turns":turns,"mixed_prefix_never_shared":true,"scope":"Slot-local live continuation, exact head then fast grid, compared with empty-state replay of the same arithmetic schedule. Not a cold all-fast arithmetic equivalence or production speed claim."}),
    )
}
#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    ensure!(
        !sets.is_empty() && (4..=MAX_OUTPUT).contains(&new_tokens),
        "reuse harness shape"
    );
    let tokens = sets.last().unwrap();
    ensure!(
        tokens.len() >= 128
            && tokens.len() < CONTEXT - 1024
            && tokens.iter().all(|t| (0..154880).contains(t)),
        "reuse fixture shape"
    );
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    eprintln!("cross-slot prefix qualification");
    let cross = cross_slot(&mut long, tokens)?;
    eprintln!("live mixed conversation qualification");
    let live = live(&mut long, tokens)?;
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"cross_slot":cross,"live":live}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_reuse_requires_exact_extending_prefix_and_retained_pages() {
        let history = vec![7; 4099];
        let mut request = history.clone();
        request.push(8);
        assert!(eligible(&history, true, 17, &request, 4).is_ok());
        assert!(eligible(&history, true, 16, &request, 4).is_err());
        assert!(eligible(&history, true, 17, &history, 4).is_err());
        request[0] = 8;
        assert!(eligible(&history, true, 17, &request, 4).is_err());
    }
    #[test]
    fn pure_off_grid_and_invalid_limits_are_not_reused() {
        assert!(eligible(&[7; 63], false, 1, &[7; 64], 1).is_err());
        assert!(eligible(&[7; 64], false, 1, &[7; 65], 1).is_ok());
        assert!(eligible(&[], true, 1, &[7; 65], 1).is_err());
        assert!(eligible(&[7; 64], true, 1, &[7; 65], 0).is_err());
        assert!(eligible(&[7; 64], true, 1, &[7; 65], MAX_OUTPUT + 1).is_err());
        assert!(eligible(&[7; 64], true, 1, &vec![7; CONTEXT], 1).is_err());
        let mut request = vec![7; 65];
        request[64] = -1;
        assert!(eligible(&[7; 64], true, 1, &request, 1).is_err());
    }
}
