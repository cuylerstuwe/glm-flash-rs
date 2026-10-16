use super::*;
use crate::{output_policy::Policy, request_broker::Broker};
use serde_json::{Value, json};
use std::sync::Mutex;

pub(super) fn tick(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    command: Command,
    clock: &Instant,
) -> Result<Report> {
    s.tick(
        long,
        (long.engine.rank == 0).then(|| (command, clock.elapsed())),
    )
}
pub(super) fn submit(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    clock: &Instant,
    tokens: Arc<[i32]>,
    background: bool,
    policy: Policy,
) -> Result<u64> {
    tick(
        s,
        long,
        Command::Submit {
            tokens,
            output: 8,
            background,
            deadline: clock.elapsed() + Duration::from_secs(90),
            policy: Some(policy),
        },
        clock,
    )?
    .submitted
    .context("termination fixture submission")
}
fn finished(s: &Scheduler, id: u64) -> bool {
    s.queue.completed.iter().any(|c| c.status.id == id)
}
pub(super) fn drive(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    id: u64,
    clock: &Instant,
) -> Result<()> {
    for _ in 0..80 {
        if finished(s, id) {
            return Ok(());
        }
        tick(s, long, Command::Tick, clock)?;
    }
    anyhow::bail!("termination did not finish")
}
pub(super) fn completed(s: &Scheduler, id: u64) -> Result<&crate::request_queue::Completed> {
    s.queue
        .completed
        .iter()
        .find(|c| c.status.id == id)
        .context("missing terminal completion")
}
pub(super) fn trace(s: &Scheduler, id: u64) -> Result<Value> {
    Ok(s.traces
        .get(&id)
        .context("missing terminal trace")?
        .value(&completed(s, id)?.outputs))
}
fn reference(long: &mut Long<'_>, tokens: &[i32], n: usize) -> Result<Value> {
    long.start(tokens.len(), 8)?;
    let mut next = super::super::store::prefill_to(long, tokens)?;
    let mut trace = Trace {
        prefill: Some(long.state(tokens.len())?),
        logits: vec![long.logits()?],
        final_state: None,
    };
    let mut out = vec![next];
    for _ in 1..n {
        next = long.step(long.sessions[long.active].tokens.len(), next)?;
        out.push(next);
        trace.logits.push(long.logits()?);
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
    let path = std::env::var("ALASKA_TOKENIZER_JSON").context("ALASKA_TOKENIZER_JSON required")?;
    let tok = crate::tokenizer::load(Path::new(&path))?;
    let table = Arc::new(Table::new(&tok)?);
    let make_preparer = || {
        crate::chat::Preparer::new(
            Path::new(&path),
            crate::chat::Policy {
                thinking: false,
                effort: Some("high"),
                clear: Some(false),
            },
        )
    };
    let body =
        json!({"messages":[{"role":"user","content":"What is 2 + 2? Answer with one numeral."}]});
    let short = make_preparer()?
        .prepare(&serde_json::to_vec(&body)?, 8)?
        .ids;
    let large: Arc<[i32]> = sets.last().context("16K fixture missing")?.clone().into();
    ensure!(large.len() == 16384, "termination fixture must be 16K");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(short[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let one = reference(&mut long, &short, 1)?;
    let eos = reference(&mut long, &short, 2)?;
    let long_two = reference(&mut long, &large, 2)?;
    ensure!(
        one["outputs"] == json!([19]) && eos["outputs"] == json!([19, 154827]),
        "original short output changed"
    );
    ensure!(
        long_two["outputs"] == json!([5095, 12]),
        "retained long output changed"
    );
    let long_stop = tok
        .decode(&[5095, 12], false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        !long_stop.is_empty()
            && !tok
                .decode(&[5095], false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .contains(&long_stop),
        "invalid long stop fixture"
    );

    let clock = Instant::now();
    let mut s = Scheduler::new(&long, true)?;
    s.install_output_table(&long, table.clone())?;
    // Largest prompt with one output plus the full binary stop-string budget.
    let mut wire = Wire::new(&long)?;
    let policy = Policy::new(
        true,
        true,
        &json!("\0".repeat(crate::output_policy::STOP_BYTES)),
    )?;
    let input: Arc<[i32]> = vec![7; crate::cache_pages::CONTEXT - 1].into();
    let (received, _, control) = wire.receive(
        &long,
        (rank == 0).then(|| {
            (
                Command::Submit {
                    tokens: input.clone(),
                    output: 1,
                    background: true,
                    deadline: Duration::from_secs(90),
                    policy: Some(policy.clone()),
                },
                Duration::ZERO,
                true,
            )
        }),
    )?;
    let Command::Submit {
        tokens,
        policy: received_policy,
        ..
    } = received
    else {
        anyhow::bail!("wire policy kind")
    };
    ensure!(
        control && tokens == input && received_policy == Some(policy),
        "generation wire changed payload/policy"
    );

    // A native pause between the two stop-completing tokens; foreground work
    // finishes with EOS while the exact background matcher/model state stays put.
    let bg = submit(
        &mut s,
        &mut long,
        &clock,
        large.clone(),
        true,
        Policy::new(false, false, &json!(long_stop))?,
    )?;
    while s.queue.request(bg)?.outputs.is_empty() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "termination".into(),
            epoch: 1,
            foreground: true,
            seconds: 30.0,
        },
        &clock,
    )?;
    let before = s.queue.status(bg, s.stamp)?;
    ensure!(
        before.pause_count == 1 && before.output_tokens == 1,
        "background not paused before second output"
    );
    let fg = submit(
        &mut s,
        &mut long,
        &clock,
        short.clone(),
        false,
        Policy::new(false, false, &Value::Null)?,
    )?;
    drive(&mut s, &mut long, fg, &clock)?;
    ensure!(
        trace(&s, fg)? == eos && completed(&s, fg)?.completion == Some(Completion::Eos),
        "foreground EOS trace differs"
    );
    for _ in 0..4 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    let held = s.queue.status(bg, s.stamp)?;
    ensure!(
        held.output_tokens == 1
            && held.active_ns == before.active_ns
            && s.output_streams[&bg].ids.len() == 1,
        "paused output/matcher advanced"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "termination".into(),
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
        "paused stop changed state/logits/tokens"
    );

    // Output failure is request-local. A deliberately ended decoder rejects its
    // next token, is collectively retired, and a later owner still runs exactly.
    let faulty = submit(
        &mut s,
        &mut long,
        &clock,
        short.clone(),
        true,
        Policy::new(false, false, &json!("4"))?,
    )?;
    // submit() currently performs one prefill tick, so use a control-only Submit
    // below for faults and cancellation that must precede any generated output.
    ensure!(
        finished(&s, faulty)
            && completed(&s, faulty)?.completion == Some(Completion::Stop)
            && trace(&s, faulty)? == one,
        "first-token stop did extra work"
    );
    let command = Command::Submit {
        tokens: short.clone(),
        output: 8,
        background: true,
        deadline: clock.elapsed() + Duration::from_secs(90),
        policy: Some(Policy::new(false, false, &json!("4"))?),
    };
    let bad = s
        .pump(
            &mut long,
            (rank == 0).then(|| (command.clone(), clock.elapsed(), true)),
        )?
        .submitted
        .context("fault fixture")?;
    ensure!(
        s.output_streams.get_mut(&bad).unwrap().push(19)?.1,
        "fault fixture did not end decoder"
    );
    drive(&mut s, &mut long, bad, &clock)?;
    ensure!(
        completed(&s, bad)?.reason == End::OutputError
            && completed(&s, bad)?.completion.is_none()
            && long.healthy
            && !s.output_streams.contains_key(&bad),
        "output failure poisoned or succeeded"
    );
    let replay = submit(
        &mut s,
        &mut long,
        &clock,
        short.clone(),
        true,
        Policy::new(false, false, &Value::Null)?,
    )?;
    drive(&mut s, &mut long, replay, &clock)?;
    ensure!(
        trace(&s, replay)? == eos,
        "replay after decoder failure changed model"
    );
    let cancel = s
        .pump(
            &mut long,
            (rank == 0).then(|| (command.clone(), clock.elapsed(), true)),
        )?
        .submitted
        .context("cancel fixture")?;
    tick(&mut s, &mut long, Command::Cancel(cancel), &clock)?;
    ensure!(
        completed(&s, cancel)?.reason == End::Cancelled && !s.output_streams.contains_key(&cancel),
        "cancel leaked output matcher"
    );
    let serial_completed = serde_json::to_value(&s.queue.completed)?;

    // Real concurrent text preparation -> native owner -> token stream -> final
    // reply. Each client waits for the native receipt and then receives its own
    // successful terminal event before finalization/history insertion.
    let (broker, owner) = Broker::new();
    let prep = Arc::new(Mutex::new(make_preparer()?));
    let join = if rank == 0 {
        let b = broker.clone();
        let table = table.clone();
        Some(std::thread::spawn(move || -> Result<Vec<Value>> {
            let result = {
                let mut jobs = Vec::new();
                for (stop, limit) in [(Value::Null, 8), (json!("4"), 8), (Value::Null, 1)] {
                    let b = b.clone();
                    let prep = prep.clone();
                    let table = table.clone();
                    let mut body = body.clone();
                    body["stop"] = stop;
                    jobs.push(std::thread::spawn(move || -> Result<Value> {
                        let client = b.begin(false, Duration::from_secs(30))?;
                        let bytes = serde_json::to_vec(&body)?;
                        let (client, prepared) = crate::chat_ingress::submit(
                            client,
                            bytes.as_slice(),
                            &prep,
                            limit,
                            Duration::from_secs(10),
                        )?;
                        let mut streamed = String::new();
                        let mut reply = crate::chat_egress::consume(
                            client,
                            prepared,
                            table,
                            &prep,
                            limit,
                            |delta| {
                                if let Some(d) = delta {
                                    streamed.push_str(&d.content);
                                }
                                Ok(true)
                            },
                            &mut || Ok("call_000000000000000000000001".into()),
                        )?;
                        streamed.push_str(reply["final"]["content"].as_str().unwrap());
                        ensure!(
                            streamed == reply["content"],
                            "streamed/final content differs"
                        );
                        reply["streamed"] = json!(streamed);
                        Ok(reply)
                    }));
                }
                jobs.into_iter()
                    .map(|j| j.join().map_err(|_| anyhow::anyhow!("client panic"))?)
                    .collect::<Result<Vec<_>>>()
            };
            b.shutdown();
            result
        }))
    } else {
        None
    };
    let mut concurrent = Scheduler::new(&long, true)?;
    concurrent.install_output_table(&long, table.clone())?;
    concurrent.serve(&mut long, (rank == 0).then_some(&owner))?;
    let replies = join
        .map(|j| j.join().map_err(|_| anyhow::anyhow!("clients panic")))
        .transpose()?
        .transpose()?;
    if let Some(replies) = &replies {
        ensure!(
            replies[0]["content"] == "4"
                && replies[0]["finish"] == "stop"
                && replies[0]["completion_tokens"] == 2
                && replies[1]["content"] == ""
                && replies[1]["finish"] == "stop"
                && replies[1]["completion_tokens"] == 1
                && replies[2]["content"] == "4"
                && replies[2]["finish"] == "length"
                && replies[2]["completion_tokens"] == 1,
            "concurrent reply mismatch"
        );
    }
    ensure!(
        concurrent.queue.completed.len() == 3
            && concurrent.queue.count() == 0
            && concurrent.queue.gate.local_count() == 0
            && concurrent.output_streams.is_empty()
            && !long.priority_hold,
        "completed clients leaked owner/hold"
    );
    for c in &concurrent.queue.completed {
        ensure!(
            c.reason == End::Complete
                && trace(&concurrent, c.status.id)?
                    == if c.outputs.len() == 2 {
                        eos.clone()
                    } else {
                        one.clone()
                    },
            "concurrent terminal model differs"
        );
    }
    let concurrent_completed = serde_json::to_value(&concurrent.queue.completed)?;
    long.memory.book().check()?;
    let (http_broker, http_owner) = Broker::new();
    let http_join = if rank == 0 {
        let server =
            crate::http_probe::start(Path::new(&path), http_broker.clone(), table.clone())?;
        let broker = http_broker.clone();
        Some(std::thread::spawn(move || {
            let result = crate::http_probe::exercise(server, &broker, false);
            broker.shutdown();
            result
        }))
    } else {
        None
    };
    let mut http_scheduler = Scheduler::new(&long, true)?;
    http_scheduler.install_output_table(&long, table.clone())?;
    http_scheduler.serve(&mut long, (rank == 0).then_some(&http_owner))?;
    let http_result = (|| {
        http_join
            .map(|j| j.join().map_err(|_| anyhow::anyhow!("HTTP probe panic")))
            .transpose()?
            .transpose()
    })();
    if let Err(error) = &http_result {
        eprintln!("HTTP probe failed: {error:#}");
    }
    ensure!(
        http_scheduler.wire.control.agree(
            &long,
            825,
            Sha256::digest(b"http-probe-result-v1").into(),
            http_result.is_ok()
        )?,
        "HTTP probe failed on a rank"
    );
    let http_result = http_result?;
    for c in &http_scheduler.queue.completed {
        if c.reason == End::Complete {
            ensure!(
                trace(&http_scheduler, c.status.id)?
                    == if c.outputs.len() == 2 {
                        eos.clone()
                    } else {
                        one.clone()
                    },
                "HTTP model trace differs"
            );
        }
    }
    ensure!(
        http_scheduler.queue.count() == 0
            && http_scheduler.queue.gate.local_count() == 0
            && http_scheduler.output_streams.is_empty(),
        "HTTP leaked native ownership"
    );
    let http_completed = serde_json::to_value(&http_scheduler.queue.completed)?;
    // A rank-local decoder disagreement must fail before any subsequent model
    // step or successful completion. This is deliberately the final operation.
    let mut fault = Scheduler::new(&long, true)?;
    fault.install_output_table(&long, table.clone())?;
    let id = fault
        .pump(
            &mut long,
            (rank == 0).then(|| (command, clock.elapsed(), true)),
        )?
        .submitted
        .context("disagreement fixture")?;
    if rank == 3 {
        fault.output_streams.insert(
            id,
            crate::output::Stream::new(table, false, false, &json!("absent"), 8)?,
        );
    }
    let mismatch = tick(&mut fault, &mut long, Command::Tick, &clock);
    ensure!(
        mismatch.is_err()
            && !long.healthy
            && fault.queue.completed.is_empty()
            && fault.queue.request(id)?.outputs.len() == 1,
        "terminal disagreement did not fail before extra token"
    );
    Ok(
        json!({"milestone":"native_eos_and_visible_stop_termination","rank":rank,"transport":if use_roce{"roce"}else{"nccl"},
        "prompt_tokens":large.len(),"short_prompt_tokens":short.len(),"one_token_reference":one,"eos_reference":eos,"long_stop_reference":long_two,
        "serial_completed":serial_completed,"concurrent_completed":concurrent_completed,"replies":replies,"http_loopback":http_result,"http_completed":http_completed,
        "wire_prompt_tokens":1048575,"wire_stop_bytes":crate::output_policy::STOP_BYTES,
        "paused_matcher_and_counters_fixed":true,"first_token_stop_no_extra_work":true,"native_eos_exact":true,
        "output_failure_isolated":true,"cancellation_retires_matcher":true,"concurrent_preparation_to_reply":true,
        "decoder_disagreement_failstop":true,"production_ready":false,"http":false,"automatic_cache_policy":false}),
    )
}
