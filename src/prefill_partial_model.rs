//! Same-binary, paired partial-storage comparison. Diagnostic only.
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
        !token_sets.is_empty() && (4..=32).contains(&new_tokens),
        "partial probe shape"
    );
    for ids in token_sets {
        ensure!(
            (1..=32769).contains(&ids.len()) && ids.iter().all(|id| (0..154880).contains(id)),
            "partial probe token bounds"
        );
    }
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    ensure!(
        !engine.direct_prefill.enabled(),
        "paired partial probe requires default-off environment"
    );
    // Both sets of prepared weights/scratch are allocated before timed work.
    // This is a diagnostic memory cost; a service retains only one Batch.
    engine.direct_prefill = crate::prefill_partial::Policy::parse(Some("1"))?;
    let mut alternative = Some(super::super::batch::Batch::new_with_rows(
        &engine.device,
        &engine,
        root,
        native,
        8192,
    )?);
    engine.direct_prefill = crate::prefill_partial::Policy::default();
    engine.batch = Some(super::super::batch::Batch::new_with_rows(
        &engine.device,
        &engine,
        root,
        native,
        8192,
    )?);
    for batch in [&mut engine.batch, &mut alternative] {
        let batch = batch.as_mut().unwrap();
        batch.hc_slabs = true;
        batch.prepare_pipe(&engine.device)?;
    }
    engine.taps = Some(crate::target_taps::Taps::new(&engine.device, native, 2048)?);
    let mut long = Long::new(engine, root)?;
    let mut cases = Vec::new();
    for ids in token_sets {
        let mut control: Option<serde_json::Value> = None;
        let mut trials = Vec::new();
        for overlap in [false, true] {
            // Repeated B and repeated begin/drain cycles expose send-slot reuse.
            for direct in [false, true, true, false] {
                if long.engine.direct_prefill.enabled() != direct {
                    long.engine.sync()?;
                    std::mem::swap(&mut long.engine.batch, &mut alternative);
                    long.engine.direct_prefill =
                        crate::prefill_partial::Policy::parse(Some(if direct {
                            "1"
                        } else {
                            "0"
                        }))?;
                }
                let batch = long.engine.batch.as_mut().unwrap();
                batch.overlap = overlap;
                batch.hc_pipeline = overlap;
                batch
                    .partial_stats
                    .set(crate::prefill_partial::Stats::default());
                long.engine.taps.as_ref().unwrap().rows.zero()?;
                long.start(ids.len(), new_tokens)?;
                let mut boundaries = Vec::new();
                let mut next = 0;
                let mut prefill_seconds = 0.;
                for (chunk, tokens) in ids.chunks(8192).enumerate() {
                    let begin = Instant::now();
                    next = long.prefill(chunk * 8192, tokens)?;
                    prefill_seconds += begin.elapsed().as_secs_f64();
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
                // Fixed diagnostic work: EOS does not change the comparison.
                while generated.len() < new_tokens {
                    next = long.step(ids.len() + generated.len() - 1, next)?;
                    generated.push(next);
                    logits.push(long.logits()?);
                }
                let result = serde_json::json!({"boundaries":boundaries,"generated":generated,"decode_logits":logits,"final_state_sha256":long.state(ids.len()+new_tokens-1)?});
                if let Some(reference) = &control {
                    ensure!(
                        &result == reference,
                        "direct partial model/logit/tap mismatch direct={direct} overlap={overlap} rows={}",
                        ids.len()
                    );
                } else {
                    control = Some(result.clone());
                }
                let stats = long.engine.batch.as_ref().unwrap().partial_stats.get();
                let partials = (ids.len() * 45 * 4096) as u64;
                ensure!(
                    stats.fp32_cast_elements == if direct { 0 } else { partials * 2 },
                    "FP32 partial counter"
                );
                ensure!(
                    stats.direct_slot_elements == if direct && overlap { partials } else { 0 },
                    "direct slot counter"
                );
                ensure!(
                    stats.bf16_copy_elements
                        == if direct {
                            partials * if overlap { 1 } else { 2 }
                        } else {
                            0
                        },
                    "BF16 copy counter"
                );
                trials.push(serde_json::json!({"direct_bf16":direct,"overlap":overlap,"prefill_seconds":prefill_seconds,"prefill_tps":ids.len() as f64/prefill_seconds,"exact":true,"partial_storage":stats}));
                eprintln!(
                    "partial storage prompt={} direct={} overlap={} exact",
                    ids.len(),
                    direct,
                    overlap
                );
            }
        }
        cases.push(
            serde_json::json!({"prompt_tokens":ids.len(),"reference":control,"trials":trials}),
        );
    }
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce {"roce"} else {"nccl"},"audit_logits":true,"cases":cases,"scope":"Same-binary FP32+cast versus direct BF16, whole intermediate state/logits/taps and following decode, repeated two-slot reuse with and without overlap. Diagnostic prefill timings only; excludes serving, cache and native-hold overhead."}),
    )
}
