//! Isolated exact-state and paired-timing control for unused prefill heads.
use super::*;

#[allow(clippy::too_many_arguments)]
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
        !token_sets.is_empty() && (4..=256).contains(&new_tokens),
        "head probe shape"
    );
    for ids in token_sets {
        ensure!(
            (8193..=65536).contains(&ids.len()) && ids.iter().all(|id| (0..154880).contains(id)),
            "head probe token bounds"
        );
    }
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.batch = Some(super::super::batch::Batch::new_with_rows(
        &engine.device,
        &engine,
        root,
        native,
        8192,
    )?);
    let batch = engine.batch.as_mut().unwrap();
    batch.hc_slabs = true;
    batch.hc_pipeline = true;
    batch.overlap = true;
    batch.prepare_pipe(&engine.device)?;
    let mut long = Long::new(engine, root)?;
    let mut cases = Vec::new();
    for ids in token_sets {
        let mut states = Vec::new();
        let mut final_logits = String::new();
        let mut first = None;
        for candidate in [false, true] {
            long.start(ids.len(), new_tokens)?;
            for (chunk, tokens) in ids.chunks(8192).enumerate() {
                let end = chunk * 8192 + tokens.len();
                let next = if candidate {
                    long.prefill_request(chunk * 8192, tokens)?
                } else {
                    Some(long.prefill(chunk * 8192, tokens)?)
                };
                let state = long.state(end)?;
                if candidate {
                    ensure!(state == states[chunk], "intermediate prefill state changed");
                    if end < ids.len() {
                        ensure!(
                            next.is_none()
                                && long.latest_logits.is_none()
                                && long.sessions[long.active].pending.is_none()
                                && long.logits().is_err(),
                            "unused head exposed stale output"
                        );
                    } else {
                        ensure!(
                            next == first && long.logits()? == final_logits,
                            "final prefill head changed"
                        );
                    }
                } else {
                    states.push(state);
                    if end == ids.len() {
                        first = next;
                        final_logits = long.logits()?;
                    }
                }
            }
        }
        let mut reference_output = Vec::new();
        let mut reference_state = String::new();
        let mut trials = Vec::new();
        for candidate in [false, true, true, false] {
            long.start(ids.len(), new_tokens)?;
            let begin = Instant::now();
            let mut next = None;
            for (chunk, tokens) in ids.chunks(8192).enumerate() {
                next = if candidate {
                    long.prefill_request(chunk * 8192, tokens)?
                } else {
                    Some(long.prefill(chunk * 8192, tokens)?)
                };
            }
            let prefill_seconds = begin.elapsed().as_secs_f64();
            ensure!(
                next == first
                    && long.state(ids.len())? == *states.last().unwrap()
                    && long.logits()? == final_logits,
                "timed prefill changed state/logits"
            );
            let mut next = next.ok_or_else(|| anyhow::anyhow!("final head missing"))?;
            let mut output = vec![next];
            while output.len() < new_tokens {
                next = long.step(ids.len() + output.len() - 1, next)?;
                output.push(next);
            }
            let final_state = long.state(ids.len() + new_tokens - 1)?;
            if reference_output.is_empty() {
                reference_output = output.clone();
                reference_state = final_state.clone();
            } else {
                ensure!(
                    output == reference_output && final_state == reference_state,
                    "decode after omitted heads changed"
                );
            }
            trials.push(serde_json::json!({"final_prefill_head_only":candidate,
                "prefill_seconds":prefill_seconds,"prefill_tps":ids.len() as f64/prefill_seconds,
                "head_calls":if candidate {1} else {ids.len().div_ceil(8192)},
                "output":output,"final_state_sha256":final_state}));
        }
        cases.push(serde_json::json!({"prompt_tokens":ids.len(),
            "intermediate_state_sha256":states,"final_logits_sha256":final_logits,
            "stale_intermediate_logits_refused":true,"trials":trials}));
    }
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce {"roce"} else {"nccl"},
        "audit_logits":true,"cases":cases,
        "scope":"Intermediate/full final model state, full logits and generated tokens. ABBA prefill timing includes identical diagnostic head gathering. Does not measure HTTP, cache or priority overhead."}),
    )
}
