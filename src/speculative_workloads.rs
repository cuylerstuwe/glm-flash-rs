//! Contiguous sampled generation on real chat inputs. This isolates model
//! execution from HTTP/SSD; end-to-end serving needs separate measurement.
#[path = "speculative_calibration.rs"]
mod calibration;
use super::{Candidates, Decision, Engine, Long, Prepared, Sampling, Verifier};
use crate::{
    chat::{Policy, Preparer},
    target_taps::Taps,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::Path, time::Instant};

fn tap_hash(taps: &Taps, end: usize) -> Result<String> {
    let n = end.min(taps.cap);
    let at = (end - n) % taps.cap;
    let first = n.min(taps.cap - at);
    let mut h = Sha256::new();
    h.update(taps.rows.read(
        at * crate::target_taps::WIDTH,
        first * crate::target_taps::WIDTH,
    )?);
    if first < n {
        h.update(taps.rows.read(0, (n - first) * crate::target_taps::WIDTH)?);
    }
    Ok(format!("{:x}", h.finalize()))
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    manifest: &Path,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    tokenizer: &Path,
    workloads: &Path,
) -> Result<Value> {
    let raw: Value = serde_json::from_slice(&std::fs::read(workloads)?)?;
    let inputs = raw.as_array().context("workload array")?;
    ensure!(!inputs.is_empty() && inputs.len() <= 16, "workload count");
    let mut prepare = Preparer::new(
        tokenizer,
        Policy {
            thinking: true,
            effort: Some("high"),
            clear: Some(false),
        },
    )?;
    let decoder = crate::tokenizer::load(tokenizer)?;
    let mut engine = Engine::new(p, rank, root, native, nccl, master, false, true)?;
    engine.taps = Some(Taps::new(&engine.device, native, 2048)?);
    engine.batch = Some(super::super::super::batch::Batch::new_with_rows(
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
    let mut verifier = Verifier::new(&long, root, native)?;
    verifier.batch_dsa = true;
    verifier.use_graphs = true;
    verifier.stop_eos = true;
    verifier.audit = false;
    verifier.taps = Some(Taps::new(
        &long.engine.device,
        native,
        crate::kda_window::MAX_ROWS,
    )?);
    let mut draft =
        super::super::super::drafter::Drafter::new(&long.engine, manifest, root, native)?;
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let (adaptive, lookup_enabled) = match std::env::var("ALASKA_SPECULATION_POLICY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("fixed") => (false, false),
        Ok("adaptive") => (true, false),
        Ok("lookup") => (true, true),
        _ => anyhow::bail!("ALASKA_SPECULATION_POLICY must be fixed/adaptive/lookup"),
    };
    let mut agreed_costs = None;
    if adaptive {
        let calibration_prompt = prepare
            .prepare(&serde_json::to_vec(&inputs[0]["request"])?, 512)?
            .ids;
        agreed_costs = Some(calibration::calibrate(
            &mut long,
            &mut verifier,
            &mut draft,
            &mut sampler,
            &calibration_prompt,
            if lookup_enabled { 16 } else { 8 },
        )?);
    }
    let mut cases = Vec::new();
    for input in inputs {
        let name = input["name"].as_str().context("workload name")?;
        let budget = input["output_tokens"].as_u64().unwrap_or(512) as usize;
        ensure!((32..=2048).contains(&budget), "bounded workload output");
        let prepared = prepare.prepare(&serde_json::to_vec(&input["request"])?, budget)?;
        ensure!(
            prepared.rendered.thinking,
            "benchmark expects thinking enabled"
        );
        let prefix = prepared.ids;
        ensure!(prefix.len() <= 32768, "bounded workload prompt");
        for seed in [731u64, 917] {
            let policy = Sampling::new(seed, 1., 20, 0.95)?;
            let mut reference: Option<(Vec<i32>, String, String)> = None;
            let mut trials = Vec::new();
            // Cold then warm speculation, bracketed by identical serial work.
            // Hashes/readbacks happen after the whole measured generation.
            for (trial, speculative) in [false, true, true, false].into_iter().enumerate() {
                long.start(prefix.len(), budget)?;
                let start = Instant::now();
                for (i, chunk) in prefix.chunks(8192).enumerate() {
                    long.prefill(i * 8192, chunk)?;
                }
                let first = long.sample(&mut sampler, policy)?.0 as i32;
                long.engine.sync()?;
                let prefill_seconds = start.elapsed().as_secs_f64();
                let setup = Instant::now();
                if speculative {
                    draft.bind(&long.engine, 0)?;
                    draft.start_context(&long.engine, prefix.len().saturating_sub(2047))?;
                    draft.append_history(
                        &long.engine,
                        long.engine.taps.as_ref().unwrap(),
                        prefix.len(),
                    )?;
                    long.engine.sync()?;
                }
                let draft_context_setup_seconds = setup.elapsed().as_secs_f64();
                let mut generated = vec![first];
                let mut rounds = Vec::new();
                let mut draft_seconds = 0.;
                let mut verify_seconds = 0.;
                let mut context_seconds = 0.;
                let mut graph_build_seconds = 0.;
                let mut lookup_seconds = 0.;
                let mut depth = agreed_costs
                    .clone()
                    .map(crate::depth_policy::DepthPolicy::new)
                    .transpose()?;
                let mut lookup = lookup_enabled
                    .then(|| crate::prompt_lookup::Lookup::new(&prefix))
                    .transpose()?;
                let start = Instant::now();
                if speculative {
                    while generated.len() < budget
                        && !crate::output_decode::EOS.contains(&(*generated.last().unwrap() as u32))
                    {
                        let pending = long.sessions[long.active].pending.unwrap();
                        let room = (budget - generated.len()).min(crate::kda_window::MAX_ROWS);
                        let phase = Instant::now();
                        let look = if let Some(lookup) = &mut lookup {
                            lookup.plan(&generated, room - 1, depth.as_ref().unwrap())?
                        } else {
                            Vec::new()
                        };
                        lookup_seconds += phase.elapsed().as_secs_f64();
                        let use_lookup = !look.is_empty();
                        let mut backlog = 0;
                        let proposals = if use_lookup {
                            look
                        } else {
                            let phase = Instant::now();
                            let end = prefix.len() + generated.len() - 1;
                            backlog = end - draft.end();
                            if backlog > 0 {
                                draft.append_history(
                                    &long.engine,
                                    long.engine.taps.as_ref().unwrap(),
                                    end,
                                )?;
                                long.engine.sync()?;
                            }
                            context_seconds += phase.elapsed().as_secs_f64();
                            let phase = Instant::now();
                            let proposals = draft.propose_scored(&long.engine, pending, policy)?;
                            draft_seconds += phase.elapsed().as_secs_f64();
                            let most = (room - 1).min(7);
                            let k = if most > 0 {
                                if let Some(depth) = &mut depth {
                                    depth.dflash(&proposals.probabilities[..most])?
                                } else {
                                    most
                                }
                            } else {
                                0
                            };
                            proposals.tokens[..k].to_vec()
                        };
                        let n = proposals.len() + 1;
                        let mut tokens = [0; crate::kda_window::MAX_ROWS];
                        tokens[0] = pending;
                        tokens[1..n].copy_from_slice(&proposals);
                        let phase = Instant::now();
                        let out =
                            verifier.execute(&mut long, &tokens[..n], Decision::Sampled(policy))?;
                        let elapsed = phase.elapsed().as_secs_f64();
                        verify_seconds += elapsed;
                        if out.graph_built {
                            graph_build_seconds += elapsed;
                        }
                        if n > 1
                            && let Some(depth) = &mut depth
                        {
                            depth.record(
                                use_lookup,
                                n,
                                if lookup_enabled { backlog } else { out.keep },
                                out.keep,
                            )?;
                        }
                        if use_lookup {
                            lookup.as_mut().unwrap().record(n, out.keep)?;
                        }
                        generated.extend_from_slice(&out.sampled[..out.keep]);
                        let phase = Instant::now();
                        let end = prefix.len() + generated.len() - 1;
                        // Deferred catch-up saves drafter work during lookup runs.
                        // Drain before another full window could overwrite unread taps.
                        if !lookup_enabled || end - draft.end() > 2048 - crate::kda_window::MAX_ROWS
                        {
                            draft.append_history(
                                &long.engine,
                                long.engine.taps.as_ref().unwrap(),
                                end,
                            )?;
                            long.engine.sync()?;
                        }
                        context_seconds += phase.elapsed().as_secs_f64();
                        rounds
                            .push(json!({"rows":n,"keep":out.keep,"graph_built":out.graph_built,"lookup":use_lookup,"backlog":backlog}));
                    }
                } else {
                    while generated.len() < budget
                        && !crate::output_decode::EOS.contains(&(*generated.last().unwrap() as u32))
                    {
                        long.step(
                            prefix.len() + generated.len() - 1,
                            *generated.last().unwrap(),
                        )?;
                        generated.push(long.sample(&mut sampler, policy)?.0 as i32);
                    }
                }
                long.engine.sync()?;
                let decode_seconds = start.elapsed().as_secs_f64();
                let state = long.state(prefix.len() + generated.len() - 1)?;
                let taps = tap_hash(
                    long.engine.taps.as_ref().unwrap(),
                    prefix.len() + generated.len() - 1,
                )?;
                if let Some((tokens, expected, expected_taps)) = &reference {
                    ensure!(
                        &generated == tokens && &state == expected && &taps == expected_taps,
                        "workload output/state drift name={name} seed={seed} trial={trial}"
                    );
                } else {
                    reference = Some((generated.clone(), state, taps));
                }
                let timed_tokens = generated.len() - 1;
                eprintln!(
                    "workload {name} seed={seed} trial={trial} speculative={speculative} tokens={timed_tokens} decode_tok_s={:.2}",
                    timed_tokens as f64 / decode_seconds
                );
                trials.push(json!({"trial":trial,"speculative":speculative,
                    "prefill_seconds":prefill_seconds,"draft_context_setup_seconds":draft_context_setup_seconds,
                    "timed_generated_tokens":timed_tokens,"decode_seconds":decode_seconds,
                    "decode_tok_s":timed_tokens as f64 / decode_seconds,
                    "draft_seconds":draft_seconds,"verify_seconds":verify_seconds,
                    "lookup_seconds":lookup_seconds,
                    "context_seconds":context_seconds,"verifier_graph_build_round_seconds":graph_build_seconds,
                    "rounds":rounds,"exact_tokens_and_final_state":true,"exact_committed_target_taps":true}));
            }
            let tokens = reference.unwrap().0;
            let mut phase_profile = None;
            if seed == 731 && tokens.len() > 1 {
                long.start(prefix.len(), budget)?;
                for (i, chunk) in prefix.chunks(8192).enumerate() {
                    long.prefill(i * 8192, chunk)?;
                }
                let first = long.sample(&mut sampler, policy)?.0 as i32;
                draft.start_context(&long.engine, prefix.len().saturating_sub(2047))?;
                draft.append_history(
                    &long.engine,
                    long.engine.taps.as_ref().unwrap(),
                    prefix.len(),
                )?;
                let proposals = draft.propose(&long.engine, first, policy)?;
                let n = (tokens.len() - 1).min(8);
                let mut input = [first; 8];
                input[1..n].copy_from_slice(&proposals[..n - 1]);
                verifier.trace = Some(crate::phase_profile::Trace::new(&long.engine.device)?);
                let out = verifier.execute(&mut long, &input[..n], Decision::Sampled(policy))?;
                ensure!(
                    out.sampled[..out.keep] == tokens[1..=out.keep],
                    "profile changed sampled output"
                );
                let mut trace = verifier.trace.take().unwrap();
                trace.mark(&long.engine.stream, "end", 45)?;
                long.engine.sync()?;
                phase_profile = Some(trace.finish()?);
            }
            let output = decoder
                .decode(&tokens.iter().map(|t| *t as u32).collect::<Vec<_>>(), false)
                .map_err(|e| anyhow::anyhow!("workload decode: {e}"))?;
            cases.push(
                json!({"name":name,"seed":seed,"top_k":20,"temperature":1.,"top_p":0.95,
                "prompt_tokens":prefix.len(),"output_tokens":tokens.len(),"generated":tokens,
                "text":output,"thinking":true,"trials":trials,"eager_diagnostic_phases":phase_profile}),
            );
        }
    }
    Ok(
        json!({"rank":rank,"cases":cases,"adaptive":adaptive,"lookup":lookup_enabled,"calibrated_costs":agreed_costs,"scope":"Real High chat prompts, contiguous sampled model execution, exact serial outputs/final state. Serial/speculative/speculative/serial order; cold graph creation included. No HTTP/SSD/native priority overhead. EOS truncates the accepted path and rolls back excess proposal state; no serial-length oracle. End-to-end serving performance requires separate measurement."}),
    )
}
