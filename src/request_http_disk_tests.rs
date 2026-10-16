//! Real HTTP/SSE use of SSD-only prefixes under fresh native owners.
use super::*;
pub(super) fn fixture(long: &mut Long<'_>, native: &Path, root: &Path) -> Result<Value> {
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
    let cached = ids.len() / ROWS * ROWS;
    ensure!(
        cached > 0 && cached < ids.len(),
        "SSD HTTP strict prefix fixture"
    );
    long.bind(3)?;
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let expected = super::super::super::sampled::reference(
        long,
        &mut sampler,
        &ids,
        Sampling::new(450, 1.0, 20, 0.95)?,
        4,
    )?;
    let mut seed = Scheduler::new(long, true)?;
    seed.install_cache(long, 256 << 20)?;
    seed.install_disk(long, root, 256 << 20)?;
    let clock = Instant::now();
    drain(&mut seed, long, &clock)?;
    let producer = submit(&mut seed, long, &clock, &ids[..cached], true)?;
    drive_restore(&mut seed, long, producer, &clock)?;
    clear_memory(&mut seed, long, &clock)?;
    ensure!(
        tick(&mut seed, long, Command::Stop, &clock)?.stopped,
        "HTTP seed stop"
    );
    drop(seed);
    let mut results = Vec::new();
    for stream in [false, true] {
        // A genuinely fresh owner/clock/RAM catalog. Only rank-local SSD files
        // survive; no live mixed session or RAM snapshot can satisfy this request.
        for slot in 0..4 {
            long.bind(slot)?;
            long.start(1, 1)?;
            long.cancel()?;
        }
        let mut s = Scheduler::new(long, true)?;
        s.install_cache(long, 256 << 20)?;
        s.install_disk(long, root, 256 << 20)?;
        s.install_sampler(long, native)?;
        s.install_output_table(long, table.clone())?;
        let (broker, owner) = Broker::new();
        let end = Instant::now() + Duration::from_secs(20);
        while s.disk.as_ref().unwrap().controller.busy() {
            s.pump(
                long,
                (long.engine.rank == 0).then(|| (Command::Tick, owner.now(), false)),
            )?;
            ensure!(Instant::now() < end, "HTTP catalog readiness");
            std::thread::sleep(Duration::from_millis(2));
        }
        ensure!(
            s.disk.as_ref().unwrap().controller.enabled(),
            "HTTP cache unavailable"
        );
        let job = if long.engine.rank == 0 {
            let server = http_probe::start(Path::new(&path), broker.clone(), table.clone())?;
            let mut body = body.clone();
            let expected_ids = expected["outputs"].clone();
            body["stream"] = json!(stream);
            if stream {
                body["stream_options"] = json!({"include_usage":true});
            }
            Some(std::thread::spawn(move || {
                let result = (|| -> Result<Value> {
                    let (status, text) = http_probe::call(
                        server.address,
                        "POST",
                        "/v1/chat/completions",
                        &http_probe::auth(),
                        &body,
                    )?;
                    ensure!(status == 200, "SSD HTTP status {status}: {text}");
                    let v: Value = if stream {
                        http_probe::events(&text)?
                            .last()
                            .context("missing SSD usage")?
                            .clone()
                    } else {
                        serde_json::from_str(&text)?
                    };
                    ensure!(
                        v["usage"]["prompt_tokens_details"]["cached_tokens"] == cached
                            && v["tensorfold"]["cached"] == cached
                            && v["tensorfold"]["token_ids"] == expected_ids,
                        "SSD HTTP cache credit/output differs"
                    );
                    http_probe::empty(&broker)?;
                    drop(server);
                    Ok(v)
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
                    .map_err(|_| anyhow::anyhow!("SSD HTTP client panic"))
            })
            .transpose()?
            .transpose();
        if let Err(e) = &result {
            eprintln!("SSD HTTP: {e:#}");
        }
        ensure!(
            s.wire.control.agree(long, 839, [0; 32], result.is_ok())?,
            "SSD HTTP failed"
        );
        let reply = result?;
        ensure!(
            s.queue.completed.len() == 1 && s.queue.count() == 0 && s.queue.gate.local_count() == 0,
            "SSD HTTP ownership leak"
        );
        let done = &s.queue.completed[0];
        ensure!(
            done.reason == End::Complete
                && done.status.cached_tokens == cached
                && trace(&s, done.status.id)? == expected,
            "SSD HTTP model trace differs"
        );
        ensure!(
            s.disk
                .as_ref()
                .unwrap()
                .records
                .iter()
                .any(|r| r["kind"] == "Read" && r["result"]["restored_tokens"] == cached),
            "HTTP request did not exercise SSD restore"
        );
        results.push(
            json!({"stream":stream,"reply":reply,"completed":s.queue.completed,
            "disk_records":s.disk.as_ref().unwrap().records}),
        );
    }
    Ok(
        json!({"cached_tokens":cached,"prompt_tokens":ids.len(),"reference":expected,
        "results":results,"fresh_owner_ssd_only":true,"json_and_sse_exact":true}),
    )
}

/// Inject rank-local elapsed-time observations while a real file job owns a
/// partial destination. This checks consensus/cleanup, without wedging the OS.
pub(super) fn watchdog(
    long: &mut Long<'_>,
    root: &Path,
    tokens: &[i32],
    expected: &Value,
) -> Result<Value> {
    use crate::cache_watch::Watch;
    let mut s = Scheduler::new(long, true)?;
    s.install_cache(long, 256 << 20)?;
    s.install_disk(long, root, 256 << 20)?;
    let clock = Instant::now();
    drain(&mut s, long, &clock)?;
    let source = submit(&mut s, long, &clock, &tokens[..8192], true)?;
    drive_restore(&mut s, long, source, &clock)?;
    clear_memory(&mut s, long, &clock)?;
    let id = submit(&mut s, long, &clock, tokens, true)?;
    partial(&mut s, long, id, &clock)?;
    let offset = Duration::from_secs(121);
    let future = s.stamp + offset;
    let d = s.disk.as_mut().unwrap();
    let mut watch = Watch::default();
    watch.observe(
        if long.engine.rank == 2 {
            Duration::ZERO
        } else {
            future
        },
        true,
        d.controller.progress(),
    )?;
    d.watch = Some(watch);
    d.next_watch = Duration::ZERO;
    let end = Instant::now() + Duration::from_secs(30);
    while !s.queue.completed.iter().any(|c| c.status.id == id) {
        s.pump(
            long,
            (long.engine.rank == 0).then(|| (Command::Tick, clock.elapsed() + offset, false)),
        )?;
        ensure!(Instant::now() < end, "watchdog fallback deadline");
        std::thread::sleep(Duration::from_millis(2));
    }
    let done = super::super::super::termination::completed(&s, id)?;
    ensure!(
        done.reason == End::Complete
            && done.status.cached_tokens == 0
            && trace(&s, id)? == *expected
            && s.disk.as_ref().unwrap().stalled_jobs == 1,
        "one-rank stall did not cause common cold fallback"
    );
    let completed = serde_json::to_value(done)?;
    // Let automatic recovery start a new optional write. The prior reader must
    // have actually retired before this new controller owner can exist.
    loop {
        let d = s.disk.as_ref().unwrap();
        if d.controller.busy() && d.reading.is_none() {
            break;
        }
        s.pump(
            long,
            (long.engine.rank == 0).then(|| (Command::Tick, clock.elapsed() + offset, false)),
        )?;
        ensure!(Instant::now() < end, "no recovery job for grace fixture");
        std::thread::sleep(Duration::from_millis(2));
    }
    let future = s.stamp + Duration::from_secs(31);
    let d = s.disk.as_mut().unwrap();
    let mut watch = Watch::default();
    watch.observe(
        if long.engine.rank == 1 {
            Duration::ZERO
        } else {
            future
        },
        true,
        d.controller.progress(),
    )?;
    watch.cancel();
    d.watch = Some(watch);
    d.next_watch = Duration::ZERO;
    let error = s
        .pump(
            long,
            (long.engine.rank == 0).then(|| {
                (
                    Command::Tick,
                    clock.elapsed() + offset + Duration::from_secs(31),
                    false,
                )
            }),
        )
        .err()
        .context("expired worker grace accepted")?;
    ensure!(
        error.to_string()
            == "SSD worker did not retire after cancellation; supervisor recovery required"
            && !long.healthy
            && s.disk.as_ref().unwrap().controller.busy(),
        "worker grace did not fail closed before replacement"
    );
    Ok(
        json!({"rank_local_elapsed_time_injection":true,"one_rank_stall_cold_replay_exact":true,
        "worker_grace_collective_fail_stop":true,"completed":completed,"error":error.to_string()}),
    )
}
