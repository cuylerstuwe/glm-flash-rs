use super::super::termination::{completed, tick, trace};
use super::*;
use serde_json::{Value, json};
fn submit(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    clock: &Instant,
    tokens: &[i32],
    background: bool,
) -> Result<u64> {
    tick(
        s,
        long,
        Command::Submit {
            tokens: tokens.into(),
            output: 4,
            background,
            deadline: clock.elapsed() + Duration::from_secs(250),
            policy: None,
        },
        clock,
    )?
    .submitted
    .context("cache test submission")
}
fn drive(s: &mut Scheduler, long: &mut Long<'_>, id: u64, clock: &Instant) -> Result<()> {
    for _ in 0..200 {
        if s.queue.completed.iter().any(|c| c.status.id == id) {
            return Ok(());
        }
        tick(s, long, Command::Tick, clock)?;
    }
    anyhow::bail!("cache request did not finish")
}
fn mixed_reference(long: &mut Long<'_>, base: &[i32], request: &[i32]) -> Result<Value> {
    long.bind(3)?;
    long.start(base.len(), 4)?;
    let mut next = super::super::super::store::prefill_to(long, base)?;
    for _ in 0..3 {
        next = long.step(long.sessions[3].tokens.len(), next)?;
    }
    long.finish()?;
    long.continue_request(request, 4, false)?;
    while long.sessions[3].tokens.len() < request.len() {
        let n = long.sessions[3].tokens.len();
        next = if !n.is_multiple_of(64) {
            long.exact_prompt_token(request[n])?
        } else {
            long.prefill(n, &request[n..(n + ROWS).min(request.len())])?
        };
    }
    let mut trace = Trace {
        prefill: Some(long.state(request.len())?),
        logits: vec![long.logits()?],
        final_state: None,
    };
    let mut output = vec![next];
    for _ in 0..3 {
        next = long.step(long.sessions[3].tokens.len(), next)?;
        output.push(next);
        trace.logits.push(long.logits()?);
    }
    trace.final_state = Some(long.state(long.sessions[3].tokens.len())?);
    long.finish()?;
    Ok(trace.value(&output))
}
fn http(s: &mut Scheduler, long: &mut Long<'_>, native: &Path, clock: &Instant) -> Result<Value> {
    use crate::{candidates::Candidates, http_probe, request_broker::Broker, sampling::Sampling};
    let path = std::env::var("ALASKA_TOKENIZER_JSON")?;
    let table = Arc::new(Table::new(&crate::tokenizer::load(Path::new(&path))?)?);
    let mut body = http_probe::chat();
    body["messages"] = json!([{"role":"system","content":"Reference material follows. ".repeat(700)},{"role":"user","content":"What is 2 + 2? Answer with one numeral."}]);
    body["temperature"] = json!(1.0);
    body["seed"] = json!(450);
    body["max_tokens"] = json!(4);
    let mut prep = crate::chat::Preparer::new(
        Path::new(&path),
        crate::chat::Policy {
            thinking: false,
            effort: Some("high"),
            clear: Some(false),
        },
    )?;
    let ids = prep.prepare(&serde_json::to_vec(&body)?, 4)?.ids;
    ensure!(
        ids.len() > 512 && !ids.len().is_multiple_of(64),
        "HTTP cache fixture must have off-grid tail"
    );
    let cached = ids.len() / ROWS * ROWS;
    long.bind(3)?;
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let reference = super::super::sampled::reference(
        long,
        &mut sampler,
        &ids,
        Sampling::new(450, 1.0, 20, 0.95)?,
        4,
    )?;
    s.install_sampler(long, native)?;
    s.install_output_table(long, table.clone())?;
    let (broker, owner) = Broker::new();
    let job = if long.engine.rank == 0 {
        let server = http_probe::start(Path::new(&path), broker.clone(), table)?;
        let broker = broker.clone();
        let expected = reference["outputs"].clone();
        Some(std::thread::spawn(move || {
            let result = (|| -> Result<Value> {
                let mut records = Vec::new();
                for i in 0..3 {
                    if i == 2 {
                        body["stream"] = json!(true);
                        body["stream_options"] = json!({"include_usage":true});
                    }
                    let (status, text) = http_probe::call(
                        server.address,
                        "POST",
                        "/v1/chat/completions",
                        &http_probe::auth(),
                        &body,
                    )?;
                    ensure!(status == 200, "cache HTTP failed {status}: {text}");
                    let v: Value = if i == 2 {
                        http_probe::events(&text)?
                            .last()
                            .context("missing cached SSE usage")?
                            .clone()
                    } else {
                        serde_json::from_str(&text)?
                    };
                    let want = if i == 0 { 0 } else { cached };
                    ensure!(
                        v["usage"]["prompt_tokens_details"]["cached_tokens"] == want
                            && v["tensorfold"]["cached"] == want
                            && v["tensorfold"]["token_ids"] == expected,
                        "cache HTTP usage/output changed"
                    );
                    records.push(v);
                }
                http_probe::empty(&broker)?;
                drop(server);
                Ok(json!({"results":records,"cached_tokens":cached}))
            })();
            broker.shutdown();
            result
        }))
    } else {
        None
    };
    s.serve(long, (long.engine.rank == 0).then_some(&owner))?;
    let result = job
        .map(|j| {
            j.join()
                .map_err(|_| anyhow::anyhow!("cache HTTP client panic"))
        })
        .transpose()?
        .transpose();
    if let Err(e) = &result {
        eprintln!("cache HTTP failure: {e:#}");
    }
    ensure!(
        s.wire.control.agree(
            long,
            827,
            Sha256::digest(b"cached-http-result-v1").into(),
            result.is_ok()
        )?,
        "cache HTTP failed on a rank"
    );
    let result = result?;
    ensure!(
        s.queue.completed.len() == 3 && s.queue.count() == 0 && s.queue.gate.local_count() == 0,
        "cached HTTP ownership leak"
    );
    for c in &s.queue.completed {
        ensure!(
            c.reason == End::Complete && trace(s, c.status.id)? == reference,
            "cached HTTP full model trace differs"
        );
    }
    let socket_completed = serde_json::to_value(&s.queue.completed)?;
    let policy = crate::output_policy::Policy::new(false, false, &Value::Null)?
        .with_sampling(Sampling::new(450, 1.0, 20, 0.95)?)?;
    let bg = tick(
        s,
        long,
        Command::Submit {
            tokens: ids.clone(),
            output: 4,
            background: true,
            deadline: clock.elapsed() + Duration::from_secs(90),
            policy: Some(policy.clone()),
        },
        clock,
    )?
    .submitted
    .context("cached sampled submission")?;
    while s.queue.request(bg)?.outputs.is_empty() {
        tick(s, long, Command::Tick, clock)?;
    }
    tick(
        s,
        long,
        Command::Lease {
            owner: "cached-sampled".into(),
            epoch: 1,
            foreground: true,
            seconds: 30.0,
        },
        clock,
    )?;
    let before = s.queue.status(bg, s.stamp)?;
    let slot = s.queue.request(bg)?.slot.unwrap();
    let pending = long.sessions[slot].pending;
    ensure!(
        before.cached_tokens == cached && before.output_tokens == 1,
        "sampled cached prefix not exercised"
    );
    let fg = tick(
        s,
        long,
        Command::Submit {
            tokens: ids,
            output: 4,
            background: false,
            deadline: clock.elapsed() + Duration::from_secs(90),
            policy: Some(policy),
        },
        clock,
    )?
    .submitted
    .context("cached sampled foreground")?;
    drive(s, long, fg, clock)?;
    let after = s.queue.status(bg, s.stamp)?;
    ensure!(
        before.active_ns == after.active_ns
            && before.output_tokens == after.output_tokens
            && pending == long.sessions[slot].pending
            && trace(s, fg)? == reference,
        "sampled pending changed under cache foreground"
    );
    tick(
        s,
        long,
        Command::Lease {
            owner: "cached-sampled".into(),
            epoch: 2,
            foreground: false,
            seconds: 30.0,
        },
        clock,
    )?;
    drive(s, long, bg, clock)?;
    ensure!(
        trace(s, bg)? == reference
            && completed(s, bg)?.status.pause_count == 1
            && completed(s, bg)?.status.resume_count == 1,
        "sampled cached resume differs"
    );
    Ok(
        json!({"socket":result,"reference":reference,"completed":socket_completed,"sampled_cached_pause_resume_exact":true,"sampled_completed":s.queue.completed}),
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
    _new_tokens: usize,
    use_roce: bool,
) -> Result<Value> {
    let tokens = sets.last().context("cache fixture")?;
    ensure!(tokens.len() == 16384, "cache fixture must be 16K");
    let base = &tokens[..8192];
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    long.start(tokens.len(), 4)?;
    let cold = super::super::super::store::finish_reference(&mut long, tokens)?;
    long.start(base.len(), 4)?;
    let base_ref = super::super::super::store::finish_reference(&mut long, base)?;
    // Remove diagnostic retained state before automatic admission.
    long.start(1, 1)?;
    long.cancel()?;
    let clock = Instant::now();
    let mut s = Scheduler::new(&long, true)?;
    s.install_cache(&long, 256 << 20)?;
    let first = submit(&mut s, &mut long, &clock, base, false)?;
    drive(&mut s, &mut long, first, &clock)?;
    ensure!(
        trace(&s, first)? == base_ref && completed(&s, first)?.status.cached_tokens == 0,
        "cold cache producer changed"
    );
    ensure!(
        s.cache
            .as_ref()
            .unwrap()
            .find(&long, tokens)?
            .is_some_and(|(_, n)| n == 8192),
        "foreground snapshot absent"
    );
    let first_slot = completed(&s, first)?.status.slot.unwrap();
    let history = long.sessions[first_slot].tokens.clone();
    let mut follow = history.clone();
    follow.push(long.sessions[first_slot].pending.unwrap());
    follow.extend(std::iter::repeat_n(7, 128));
    let expected = mixed_reference(&mut long, base, &follow)?;
    // Clear reference slot so it cannot satisfy the live selection itself.
    long.start(1, 1)?;
    long.cancel()?;
    let live = submit(&mut s, &mut long, &clock, &follow, false)?;
    drive(&mut s, &mut long, live, &clock)?;
    ensure!(
        trace(&s, live)? == expected
            && completed(&s, live)?.status.slot == Some(first_slot)
            && completed(&s, live)?.status.cached_tokens == history.len(),
        "live continuation replay differs"
    );
    let ram = submit(&mut s, &mut long, &clock, tokens, false)?;
    drive(&mut s, &mut long, ram, &clock)?;
    ensure!(
        trace(&s, ram)? == cold && completed(&s, ram)?.status.cached_tokens == 8192,
        "RAM continuation differs"
    );
    // A partial cached background request must keep its exact state under hold.
    let mut branch = base.to_vec();
    branch.extend(std::iter::repeat_n(8, 4096));
    let bg = submit(&mut s, &mut long, &clock, &branch, true)?;
    let bg_slot = s.queue.request(bg)?.slot.context("background admission")?;
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "cached-background".into(),
            epoch: 1,
            foreground: true,
            seconds: 30.0,
        },
        &clock,
    )?;
    let before = s.queue.status(bg, s.stamp)?;
    long.bind(bg_slot)?;
    let protected = long.state(long.sessions[bg_slot].tokens.len())?;
    ensure!(
        s.cache.as_mut().unwrap().save(&mut long).is_err() && long.healthy,
        "held background snapshot allowed"
    );
    let fg = submit(&mut s, &mut long, &clock, tokens, false)?;
    drive(&mut s, &mut long, fg, &clock)?;
    ensure!(
        trace(&s, fg)? == cold && completed(&s, fg)?.status.cached_tokens == 8192,
        "foreground restore under hold differs"
    );
    let held = s.queue.status(bg, s.stamp)?;
    long.bind(bg_slot)?;
    ensure!(
        held.prefill_done == before.prefill_done
            && held.active_ns == before.active_ns
            && long.state(long.sessions[bg_slot].tokens.len())? == protected,
        "cached background advanced under hold"
    );
    tick(&mut s, &mut long, Command::Cancel(bg), &clock)?;
    ensure!(
        completed(&s, bg)?.reason == End::Cancelled && long.sessions[bg_slot].tokens.is_empty(),
        "cached cancellation did not discard live state"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "cached-background".into(),
            epoch: 2,
            foreground: false,
            seconds: 30.0,
        },
        &clock,
    )?;
    long.bind(3)?;
    long.start(branch.len(), 4)?;
    let branch_reference = super::super::super::store::finish_reference(&mut long, &branch)?;
    let resumed = submit(&mut s, &mut long, &clock, &branch, true)?;
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "cached-background".into(),
            epoch: 3,
            foreground: true,
            seconds: 30.0,
        },
        &clock,
    )?;
    let paused = s.queue.status(resumed, s.stamp)?;
    for _ in 0..3 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    ensure!(
        s.queue.status(resumed, s.stamp)?.prefill_done == paused.prefill_done,
        "cached prefill advanced under hold"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "cached-background".into(),
            epoch: 4,
            foreground: false,
            seconds: 30.0,
        },
        &clock,
    )?;
    drive(&mut s, &mut long, resumed, &clock)?;
    ensure!(
        trace(&s, resumed)? == branch_reference
            && completed(&s, resumed)?.status.cached_tokens == 8192
            && completed(&s, resumed)?.status.pause_count == 1
            && completed(&s, resumed)?.status.resume_count == 1,
        "cached pause/resume changed replay"
    );
    // A 64-grid prefix need not align to the 512-row slab origin. Preserve the
    // shifted prefill schedule and snapshot its actual final full slab.
    let mut shifted = vec![16; 9001];
    let producer = submit(&mut s, &mut long, &clock, &shifted[..8256], false)?;
    drive(&mut s, &mut long, producer, &clock)?;
    // The uniform fixture generates more of its own token. Force a divergence
    // immediately after the pure snapshot so the longer mixed live prefix is
    // ineligible; otherwise a cold arithmetic reference is inappropriate.
    shifted[8256] = (completed(&s, producer)?.outputs[0] + 1) % 154880;
    let reference_slot = (completed(&s, producer)?.status.slot.unwrap() + 1) % SLOTS;
    long.bind(reference_slot)?;
    long.start(shifted.len(), 4)?;
    let shifted_reference = super::super::super::store::finish_reference(&mut long, &shifted)?;
    let shifted_id = submit(&mut s, &mut long, &clock, &shifted, false)?;
    drive(&mut s, &mut long, shifted_id, &clock)?;
    eprintln!(
        "shifted diagnostics: {}",
        json!({"cached":completed(&s,shifted_id)?.status.cached_tokens,
        "stored":s.cache.as_ref().unwrap().find(&long,&shifted)?,"producer":completed(&s,producer)?.outputs,
        "actual":trace(&s,shifted_id)?,"reference":shifted_reference})
    );
    ensure!(
        completed(&s, shifted_id)?.status.cached_tokens == 8256
            && trace(&s, shifted_id)? == shifted_reference
            && s.cache
                .as_ref()
                .unwrap()
                .find(&long, &shifted)?
                .is_some_and(|(_, n)| n == 8768),
        "shifted RAM snapshot/replay differs"
    );
    let shifted_result = json!({"cached_tokens":completed(&s,shifted_id)?.status.cached_tokens,
        "saved_tokens":s.cache.as_ref().unwrap().find(&long,&shifted)?.unwrap().1,
        "reference":shifted_reference,"actual":trace(&s,shifted_id)?});
    // Force real bounded RAM eviction; immutable restore already detached from it.
    for token in [11, 12] {
        let unrelated = vec![token; 16384];
        let id = submit(&mut s, &mut long, &clock, &unrelated, false)?;
        drive(&mut s, &mut long, id, &clock)?;
    }
    ensure!(
        s.cache.as_ref().unwrap().find(&long, tokens)?.is_none(),
        "cache pressure did not evict old prefix"
    );
    let miss = submit(&mut s, &mut long, &clock, tokens, false)?;
    drive(&mut s, &mut long, miss, &clock)?;
    ensure!(
        trace(&s, miss)? == cold && completed(&s, miss)?.status.cached_tokens == 0,
        "evicted prefix did not replay cold exactly"
    );
    let records = serde_json::to_value(&s.queue.completed)?;
    let mut socket_scheduler = Scheduler::new(&long, true)?;
    socket_scheduler.install_cache(&long, 192 << 20)?;
    let http_result = http(&mut socket_scheduler, &mut long, native, &clock)?;
    // Selection disagreement must fail before binding/admission/model mutation.
    let id = s
        .queue
        .submit(base.into(), 4, true, Duration::from_secs(3600), s.stamp)?;
    let active = long.active;
    let state = long.state(long.sessions[active].tokens.len())?;
    if rank == 3 {
        s.cache = None;
    }
    let failure = tick(&mut s, &mut long, Command::Tick, &clock)
        .err()
        .context("divergent cache accepted")?;
    ensure!(
        failure.to_string() == "session rank decision/digest disagreement"
            && !long.healthy
            && s.queue.request(id)?.done == 0
            && s.queue.request(id)?.slot.is_none()
            && long.active == active,
        "cache divergence reached work"
    );
    // `state` is retained evidence; state() deliberately refuses poisoned owners.
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":16384,
        "cold_reference":cold,"base_reference":base_ref,"mixed_reference":expected,"completed":records,
        "live_reused_tokens":history.len(),"ram_reused_tokens":8192,"ram_budget":256u64<<20,
        "foreground_save_and_restore_under_hold":true,"paused_state_exact":true,"held_background_save_refused":true,
        "cancel_discard_exact":true,"cached_pause_resume_exact":true,"shifted_snapshot_and_restore_exact":true,"eviction_cold_fallback_exact":true,"cache_disagreement_before_work":true,
        "state_before_fault":state,"shifted":shifted_result,"http":http_result,"production_ready":false}),
    )
}
