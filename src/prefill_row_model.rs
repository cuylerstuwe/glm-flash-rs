//! Paired TP4 ownership model control, independently crossed with BF16 outputs.
use super::super::{
    batch::{Batch, RowStats},
    prefill_split_pipe::plan::{Plan, Policy},
};
use super::*;

fn prepare(engine: &mut Engine<'_>, root: &Path, native: &Path, row: bool) -> Result<Batch> {
    engine.row_prefill = Policy::parse(Some(if row { "1" } else { "0" }))?;
    let mut batch = Batch::new_with_rows(&engine.device, engine, root, native, 8192)?;
    batch.hc_slabs = true;
    batch.overlap = true;
    batch.hc_pipeline = true;
    batch.prepare_pipe(&engine.device)?;
    Ok(batch)
}
fn expected_counts(n: usize, rank: usize, row_split: bool) -> Result<RowStats> {
    let mut expected = RowStats::default();
    for first in (0..n).step_by(8192) {
        let rows = (n - first).min(8192);
        if !row_split || rows <= 1024 {
            expected.fallback_batches += 1;
            continue;
        }
        expected.batches += 1;
        for offset in (0..rows).step_by(1024) {
            let count = (rows - offset).min(1024);
            let own = Plan::new(count, rank)?.own();
            expected.embedded_rows += own.count as u64;
            expected.post_rows += (own.count * 90) as u64;
            expected.partial_exchanges += 90;
            expected.shared_pieces += 90;
            let end = offset + own.first + own.count;
            let begin = (offset + own.first).max(rows.saturating_sub(2048)).min(end);
            expected.captured_rows += ((end - begin) * 5) as u64;
        }
    }
    Ok(expected)
}

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
        !token_sets.is_empty() && (4..=32).contains(&new_tokens),
        "row probe shape"
    );
    for ids in token_sets {
        ensure!(
            (1..=32769).contains(&ids.len()) && ids.iter().all(|x| (0..154880).contains(x)),
            "row probe token bounds"
        );
    }
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    ensure!(
        !engine.row_prefill.enabled() && !engine.direct_prefill.enabled(),
        "paired row probe requires default-off environment"
    );
    engine.batch = None;
    engine.batch = Some(prepare(&mut engine, root, native, false)?);
    engine.taps = Some(crate::target_taps::Taps::new(&engine.device, native, 2048)?);
    let mut long = Long::new(engine, root)?;
    let mut references = vec![None; token_sets.len()];
    let mut trials = vec![Vec::new(); token_sets.len()];
    for direct in [false, true] {
        // At most two 8K Batch owners, never all four policy combinations.
        // Allocation is outside diagnostic timing and is not a serving footprint.
        if direct {
            long.engine.sync()?;
            long.engine.batch = None;
            long.engine.direct_prefill = crate::prefill_partial::Policy::parse(Some("1"))?;
            long.engine.batch = Some(prepare(&mut long.engine, root, native, false)?);
        }
        let mut alternative = Some(prepare(&mut long.engine, root, native, true)?);
        long.engine.row_prefill = Policy::default();
        for (case, ids) in token_sets.iter().enumerate() {
            for row_split in [false, true, true, false] {
                if long.engine.row_prefill.enabled() != row_split {
                    long.engine.sync()?;
                    std::mem::swap(&mut long.engine.batch, &mut alternative);
                    long.engine.row_prefill =
                        Policy::parse(Some(if row_split { "1" } else { "0" }))?;
                }
                let batch = long.engine.batch.as_ref().unwrap();
                batch.row_stats.set(RowStats::default());
                batch
                    .partial_stats
                    .set(crate::prefill_partial::Stats::default());
                long.engine.taps.as_ref().unwrap().rows.zero()?;
                long.start(ids.len(), new_tokens)?;
                let mut boundaries = Vec::new();
                let mut next = 0;
                let mut seconds = 0.;
                for (chunk, tokens) in ids.chunks(8192).enumerate() {
                    let started = Instant::now();
                    next = long.prefill(chunk * 8192, tokens)?;
                    seconds += started.elapsed().as_secs_f64();
                    let end = chunk * 8192 + tokens.len();
                    let taps = long.engine.taps.as_ref().unwrap();
                    let taps_hash =
                        format!("{:x}", Sha256::digest(taps.rows.read(0, taps.rows.bytes)?));
                    let residual_hash = format!(
                        "{:x}",
                        Sha256::digest(long.engine.x.read(0, long.engine.x.bytes)?)
                    );
                    boundaries.push(serde_json::json!({"position":end,"state_sha256":long.state(end)?,"logits_sha256":long.logits()?,"taps_sha256":taps_hash,"residual_sha256":residual_hash}));
                }
                let mut generated = vec![next];
                let mut logits = vec![long.logits()?];
                while generated.len() < new_tokens {
                    next = long.step(ids.len() + generated.len() - 1, next)?;
                    generated.push(next);
                    logits.push(long.logits()?);
                }
                let reference = serde_json::json!({"boundaries":boundaries,"generated":generated,"decode_logits":logits,"final_state_sha256":long.state(ids.len()+new_tokens-1)?});
                if let Some(control) = &references[case] {
                    ensure!(
                        control == &reference,
                        "row-owned prefill mismatch prompt={} direct={direct} split={row_split}",
                        ids.len()
                    );
                } else {
                    references[case] = Some(reference);
                }
                let batch = long.engine.batch.as_ref().unwrap();
                let observed = batch.row_stats.get();
                ensure!(
                    serde_json::to_value(observed)?
                        == serde_json::to_value(expected_counts(ids.len(), rank, row_split)?)?,
                    "row-owned path counter mismatch"
                );
                let partial = batch.partial_stats.get();
                let elements = (ids.len() * 45 * 4096) as u64;
                ensure!(
                    partial.fp32_cast_elements == if direct { 0 } else { elements * 2 },
                    "row partial cast counter"
                );
                ensure!(
                    partial.direct_slot_elements == if direct { elements } else { 0 },
                    "row partial direct counter"
                );
                ensure!(
                    partial.bf16_copy_elements == if direct { elements } else { 0 },
                    "row partial copy counter"
                );
                trials[case].push(serde_json::json!({"direct_bf16":direct,"row_split":row_split,"exact":true,"prefill_seconds":seconds,"row_ownership":observed,"partial_storage":partial}));
                eprintln!(
                    "row-owned prompt={} direct={direct} split={row_split} exact",
                    ids.len()
                );
            }
        }
        long.engine.sync()?;
        drop(alternative);
    }
    let cases=token_sets.iter().enumerate().map(|(i,ids)|serde_json::json!({"prompt_tokens":ids.len(),"reference":references[i],"trials":trials[i]})).collect::<Vec<_>>();
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce {"roce"}else{"nccl"},"audit_logits":true,"cases":cases,"scope":"Paired rank-owned/full-row prefill crossed with direct BF16, exact state/logits/taps/residual/decode and path counters. Diagnostic only; no serving/cache/native-hold or speed claim."}),
    )
}
