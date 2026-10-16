//! Request ownership, exact sampling and terminal/priority behavior together.
use super::termination::{completed, drive, submit, tick, trace};
use super::*;
use crate::{
    candidates::Candidates,
    output_policy::Policy,
    request_broker::{Broker, Event},
    sampling::Sampling,
};
use serde_json::{Value, json};

fn policy(s: Sampling, stop: Value) -> Result<Policy> {
    Policy::new(false, false, &stop)?.with_sampling(s)
}
pub(super) fn reference(
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    tokens: &[i32],
    sampling: Sampling,
    count: usize,
) -> Result<Value> {
    long.start(tokens.len(), 8)?;
    super::super::store::prefill_to(long, tokens)?;
    let mut out = Vec::new();
    let mut trace = Trace {
        prefill: Some(long.state(tokens.len())?),
        logits: Vec::new(),
        final_state: None,
    };
    for n in 0..count {
        if n > 0 {
            long.step(long.sessions[long.active].tokens.len(), out[n - 1])?;
        }
        let next = super::super::sampling::checked_sample(long, sampler, sampling)?;
        out.push(next);
        trace.logits.push(long.logits()?);
        if crate::output_decode::EOS.contains(&(next as u32)) {
            break;
        }
    }
    trace.final_state = Some(long.state(long.sessions[long.active].tokens.len())?);
    long.finish()?;
    Ok(trace.value(&out))
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
    _new_tokens: usize,
    use_roce: bool,
) -> Result<Value> {
    let path = std::env::var("ALASKA_TOKENIZER_JSON")?;
    let tok = crate::tokenizer::load(Path::new(&path))?;
    let table = Arc::new(Table::new(&tok)?);
    let mut prep = crate::chat::Preparer::new(
        Path::new(&path),
        crate::chat::Policy {
            thinking: false,
            effort: Some("high"),
            clear: Some(false),
        },
    )?;
    let short=prep.prepare(&serde_json::to_vec(&json!({"messages":[{"role":"user","content":"What is 2 + 2? Answer with one numeral."}]}))?,8)?.ids;
    let large: Arc<[i32]> = sets.last().context("16K fixture missing")?.clone().into();
    ensure!(large.len() == 16384, "sampled request fixture must be 16K");
    let wide: Arc<[i32]> = large[..400].to_vec().into();
    let normal = Sampling::new(450, 1.0, 20, 0.95)?;
    let background = Sampling::new(1234, 0.7, 20, 0.95)?;
    let wide_policy = Sampling::new(u64::MAX, 1.0, 504, 0.5)?;
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(short[0])?;
    engine.enable_prefill(root, native)?;
    let mut sampler = Candidates::new(&engine.device, native)?;
    let mut long = Long::new(engine, root)?;
    let one = reference(&mut long, &mut sampler, &short, normal, 1)?;
    let eos = reference(&mut long, &mut sampler, &short, normal, 8)?;
    let long_two = reference(&mut long, &mut sampler, &large, background, 2)?;
    let wide_ref = reference(&mut long, &mut sampler, &wide, wide_policy, 8)?;
    let wide_math = reference(&mut long, &mut sampler, &short, wide_policy, 8)?;
    ensure!(
        one["outputs"] == json!([19]) && eos["outputs"] == json!([19, 154827]),
        "sampled math fixture changed"
    );
    let long_ids: Vec<u32> = serde_json::from_value(long_two["outputs"].clone())?;
    ensure!(
        long_ids.len() == 2
            && long_ids
                .iter()
                .all(|id| !crate::output_decode::EOS.contains(id)),
        "long stop fixture"
    );
    let stop = tok
        .decode(&long_ids, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        !stop.is_empty()
            && !tok
                .decode(&long_ids[..1], false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .contains(&stop),
        "stop must span tokens"
    );
    let clock = Instant::now();
    let mut s = Scheduler::new(&long, true)?;
    s.install_output_table(&long, table.clone())?;
    let missing = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: short.clone(),
            output: 8,
            background: true,
            deadline: clock.elapsed() + Duration::from_secs(90),
            policy: Some(policy(normal, Value::Null)?),
        },
        &clock,
    )?;
    ensure!(
        missing.rejected.is_some() && s.queue.count() == 0 && long.healthy,
        "uninstalled sampler accepted or poisoned"
    );
    s.install_sampler(&long, native)?;
    // Largest legal command plus maximum stop bytes and all exact sampling bits.
    let mut wire = Wire::new(&long)?;
    let full = policy(
        Sampling::new(
            u64::MAX,
            f64::MIN_POSITIVE,
            154880,
            f64::from_bits(0.5f64.to_bits() - 1),
        )?,
        json!("\0".repeat(crate::output_policy::STOP_BYTES)),
    )?;
    let tokens: Arc<[i32]> = vec![7; crate::cache_pages::CONTEXT - 1].into();
    let (got, _, control) = wire.receive(
        &long,
        (rank == 0).then(|| {
            (
                Command::Submit {
                    tokens: tokens.clone(),
                    output: 1,
                    background: true,
                    deadline: Duration::from_secs(90),
                    policy: Some(full.clone()),
                },
                Duration::ZERO,
                true,
            )
        }),
    )?;
    let Command::Submit {
        tokens: received,
        policy: received_policy,
        ..
    } = got
    else {
        anyhow::bail!("wire command kind")
    };
    ensure!(
        control && received == tokens && received_policy == Some(full),
        "sampled full policy wire changed"
    );
    let bg = submit(
        &mut s,
        &mut long,
        &clock,
        large.clone(),
        true,
        policy(background, json!(stop))?,
    )?;
    while s.queue.request(bg)?.outputs.is_empty() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "sampled".into(),
            epoch: 1,
            foreground: true,
            seconds: 30.0,
        },
        &clock,
    )?;
    let before = s.queue.status(bg, s.stamp)?;
    let bg_slot = s.queue.request(bg)?.slot.unwrap();
    let pending = long.sessions[bg_slot].pending;
    ensure!(
        before.pause_count == 1 && before.output_tokens == 1,
        "background not paused at sampled boundary"
    );
    let fg = submit(
        &mut s,
        &mut long,
        &clock,
        wide.clone(),
        false,
        policy(wide_policy, Value::Null)?,
    )?;
    drive(&mut s, &mut long, fg, &clock)?;
    ensure!(trace(&s, fg)? == wide_ref, "wide foreground differs");
    for _ in 0..4 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    let held = s.queue.status(bg, s.stamp)?;
    ensure!(
        held.output_tokens == 1
            && held.active_ns == before.active_ns
            && long.sessions[bg_slot].pending == pending
            && s.queue.request(bg)?.sampling() == Some(background)
            && s.output_streams[&bg].ids.len() == 1,
        "paused sampler advanced"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "sampled".into(),
            epoch: 2,
            foreground: false,
            seconds: 30.0,
        },
        &clock,
    )?;
    drive(&mut s, &mut long, bg, &clock)?;
    ensure!(
        trace(&s, bg)? == long_two
            && completed(&s, bg)?.completion == Some(Completion::Stop)
            && completed(&s, bg)?.status.resume_count == 1,
        "sampled stop resume differs"
    );
    let math = submit(
        &mut s,
        &mut long,
        &clock,
        short.clone(),
        false,
        policy(normal, Value::Null)?,
    )?;
    drive(&mut s, &mut long, math, &clock)?;
    ensure!(
        trace(&s, math)? == eos && completed(&s, math)?.completion == Some(Completion::Eos),
        "sampled EOS differs"
    );
    let stop_one = submit(
        &mut s,
        &mut long,
        &clock,
        short.clone(),
        false,
        policy(normal, json!("4"))?,
    )?;
    ensure!(
        trace(&s, stop_one)? == one
            && completed(&s, stop_one)?.completion == Some(Completion::Stop),
        "first sampled stop did extra work"
    );
    // End a request's decoder before admission; only that request may fail.
    let command = Command::Submit {
        tokens: short.clone(),
        output: 8,
        background: true,
        deadline: clock.elapsed() + Duration::from_secs(90),
        policy: Some(policy(normal, json!("4"))?),
    };
    let bad = s
        .pump(
            &mut long,
            (rank == 0).then(|| (command, clock.elapsed(), true)),
        )?
        .submitted
        .context("output fault submission")?;
    ensure!(
        s.output_streams.get_mut(&bad).unwrap().push(19)?.1,
        "ended decoder fixture"
    );
    drive(&mut s, &mut long, bad, &clock)?;
    ensure!(
        completed(&s, bad)?.reason == End::OutputError
            && long.healthy
            && !s.output_streams.contains_key(&bad),
        "sampled output error leaked or poisoned"
    );
    let cancelled = submit(
        &mut s,
        &mut long,
        &clock,
        wide.clone(),
        true,
        policy(wide_policy, Value::Null)?,
    )?;
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "sampled-cancel".into(),
            epoch: 1,
            foreground: true,
            seconds: 30.0,
        },
        &clock,
    )?;
    let paused_outputs = s.queue.request(cancelled)?.outputs.clone();
    tick(&mut s, &mut long, Command::Cancel(cancelled), &clock)?;
    ensure!(
        completed(&s, cancelled)?.reason == End::Cancelled
            && completed(&s, cancelled)?.outputs == paused_outputs,
        "paused cancellation advanced sample"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "sampled-cancel".into(),
            epoch: 2,
            foreground: false,
            seconds: 30.0,
        },
        &clock,
    )?;
    let replay = submit(
        &mut s,
        &mut long,
        &clock,
        wide.clone(),
        false,
        policy(wide_policy, Value::Null)?,
    )?;
    drive(&mut s, &mut long, replay, &clock)?;
    ensure!(
        trace(&s, replay)? == wide_ref,
        "sampled cancel/reuse differs"
    );
    let serial = serde_json::to_value(&s.queue.completed)?;
    // Concurrent broker lifetimes carry independent exact policies to the owner.
    let (broker, owner) = Broker::new();
    let join = if rank == 0 {
        let b = broker.clone();
        let short = short.clone();
        let wide = wide.clone();
        Some(std::thread::spawn(move || -> Result<Value> {
            let result = (|| {
                let mut jobs = Vec::new();
                for (tokens, p) in [
                    (short, policy(normal, Value::Null)?),
                    (wide.clone(), policy(wide_policy, Value::Null)?),
                    (wide, policy(wide_policy, Value::Null)?),
                ] {
                    let b = b.clone();
                    jobs.push(std::thread::spawn(move || -> Result<Vec<i32>> {
                        let client = b.begin(false, Duration::from_secs(60))?;
                        client.prepared(Duration::from_secs(10))?;
                        client.submit_policy(tokens, 8, Some(p))?;
                        let mut outputs = Vec::new();
                        loop {
                            match client.receive(Duration::from_secs(30))? {
                                Some(Event::Token(t)) => outputs.push(t),
                                Some(Event::End(End::Complete)) => break,
                                other => anyhow::bail!("unexpected sampled client event {other:?}"),
                            }
                        }
                        Ok(outputs)
                    }));
                }
                let outputs = jobs
                    .into_iter()
                    .map(|j| {
                        j.join()
                            .map_err(|_| anyhow::anyhow!("sampled client panic"))?
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(json!(outputs))
            })();
            b.shutdown();
            result
        }))
    } else {
        None
    };
    let mut concurrent = Scheduler::new(&long, true)?;
    concurrent.install_output_table(&long, table.clone())?;
    concurrent.install_sampler(&long, native)?;
    concurrent.serve(&mut long, (rank == 0).then_some(&owner))?;
    let clients = join
        .map(|j| j.join().map_err(|_| anyhow::anyhow!("client owner panic")))
        .transpose()?
        .transpose()?;
    ensure!(
        concurrent.queue.completed.len() == 3
            && concurrent.queue.count() == 0
            && concurrent.queue.gate.local_count() == 0
            && concurrent.output_streams.is_empty(),
        "sampled concurrent ownership leak"
    );
    for c in &concurrent.queue.completed {
        ensure!(
            c.reason == End::Complete
                && trace(&concurrent, c.status.id)?
                    == if c.outputs.len() == 2 {
                        eos.clone()
                    } else {
                        wide_ref.clone()
                    },
            "concurrent sampled trace differs"
        );
    }
    if let Some(clients) = &clients {
        ensure!(
            clients == &json!([eos["outputs"], wide_ref["outputs"], wide_ref["outputs"]]),
            "client outputs changed"
        );
    }
    let concurrent_completed = serde_json::to_value(&concurrent.queue.completed)?;
    let (http_broker, http_owner) = Broker::new();
    let http_join = if rank == 0 {
        let server =
            crate::http_probe::start(Path::new(&path), http_broker.clone(), table.clone())?;
        let broker = http_broker.clone();
        let eos = eos["outputs"].clone();
        let wide = wide_math["outputs"].clone();
        Some(std::thread::spawn(move || {
            let result = crate::http_sampling_probe::exercise(server, &broker, &eos, &wide);
            broker.shutdown();
            result
        }))
    } else {
        None
    };
    let mut http = Scheduler::new(&long, true)?;
    http.install_output_table(&long, table)?;
    http.install_sampler(&long, native)?;
    http.serve(&mut long, (rank == 0).then_some(&http_owner))?;
    let http_result = (|| {
        http_join
            .map(|j| {
                j.join()
                    .map_err(|_| anyhow::anyhow!("HTTP sampled probe panic"))
            })
            .transpose()?
            .transpose()
    })();
    if let Err(error) = &http_result {
        eprintln!("sampled HTTP probe failed: {error:#}");
    }
    ensure!(
        http.wire.control.agree(
            &long,
            825,
            Sha256::digest(b"http-probe-result-v1").into(),
            http_result.is_ok()
        )?,
        "sampled HTTP probe failed on a rank"
    );
    let http_result = http_result?;
    ensure!(
        http.queue.completed.len() == 11
            && http.queue.count() == 0
            && http.queue.gate.local_count() == 0
            && http.output_streams.is_empty()
            && !long.priority_hold,
        "sampled HTTP ownership leak"
    );
    for c in &http.queue.completed {
        let got = trace(&http, c.status.id)?;
        ensure!(
            c.reason == End::Complete && (got == one || got == eos || got == wide_math),
            "sampled HTTP full native trace differs"
        );
    }
    if let Some(result) = &http_result {
        for record in result["results"].as_array().unwrap() {
            let id = record["response"]["tensorfold"]["native_request_id"]
                .as_u64()
                .context("HTTP native ID")?;
            let expected = match record["reference"].as_str().unwrap() {
                "one" => &one,
                "eos" => &eos,
                "wide" => &wide_math,
                _ => unreachable!(),
            };
            ensure!(
                &trace(&http, id)? == expected,
                "HTTP request/reference association changed"
            );
        }
    }
    let http_completed = serde_json::to_value(&http.queue.completed)?;
    // Deliberately inject a rank-local seed difference before dispatch. The
    // ownership digest must fail before any model call or candidate collective.
    let mut fault = Scheduler::new(&long, true)?;
    fault.install_sampler(&long, native)?;
    let p = policy(
        Sampling::new(if rank == 3 { 451 } else { 450 }, 1.0, 20, 0.95)?,
        Value::Null,
    )?;
    let id = fault.queue.submit_policy(
        short,
        8,
        true,
        Duration::from_secs(3600),
        Duration::ZERO,
        Some(&p),
    )?;
    // This deliberately bypasses the head's Submit wire. Use a shared clock,
    // not each process's Instant origin: a backwards-clock error on one rank
    // would leave its peers entering the ownership collective alone.
    let failure = tick(&mut fault, &mut long, Command::Tick, &clock)
        .err()
        .context("policy disagreement accepted")?;
    ensure!(
        failure.to_string() == "session rank decision/digest disagreement"
            && !long.healthy
            && fault.queue.request(id)?.done == 0
            && fault.queue.completed.is_empty(),
        "policy disagreement reached GPU work"
    );
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":16384,"one_token_reference":one,"eos_reference":eos,
        "long_stop_reference":long_two,"wide_reference":wide_ref,"wide_math_reference":wide_math,"serial_completed":serial,"concurrent_completed":concurrent_completed,"client_outputs":clients,
        "http_loopback":http_result,"http_completed":http_completed,
        "wire_prompt_tokens":1048575,"wire_stop_bytes":crate::output_policy::STOP_BYTES,"sampler_bits_exact":true,
        "uninstalled_sampler_refused":true,"paused_pending_policy_and_counters_fixed":true,"wide_foreground_exact":true,
        "first_token_stop_no_extra_work":true,"sampled_eos_exact":true,"output_failure_isolated":true,"paused_cancel_reuse_exact":true,
        "concurrent_sampled_clients_exact":true,"policy_disagreement_before_work":true,"production_ready":false,"sampled_http":true}),
    )
}
