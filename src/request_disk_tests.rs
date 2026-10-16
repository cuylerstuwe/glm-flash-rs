use super::super::termination::{drive, tick, trace};
use super::*;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
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
            deadline: clock.elapsed() + Duration::from_secs(240),
            policy: None,
        },
        clock,
    )?
    .submitted
    .context("disk request submission")
}
fn drain(s: &mut Scheduler, long: &mut Long<'_>, clock: &Instant) -> Result<()> {
    let end = Instant::now() + Duration::from_secs(30);
    while s
        .disk
        .as_ref()
        .is_some_and(|d| d.controller.busy() || d.pending.is_some())
    {
        tick(s, long, Command::Tick, clock)?;
        ensure!(Instant::now() < end, "request SSD drain deadline");
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}
fn hold(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    clock: &Instant,
    epoch: u64,
    foreground: bool,
) -> Result<()> {
    let command = Command::Lease {
        owner: "request-ssd".into(),
        epoch,
        foreground,
        seconds: 30.0,
    };
    let report = s.pump(
        long,
        (long.engine.rank == 0).then(|| (command, clock.elapsed(), true)),
    )?;
    ensure!(
        report.worked.is_none() && report.held == foreground,
        "SSD hold receipt"
    );
    Ok(())
}
fn reference(long: &mut Long<'_>, tokens: &[i32]) -> Result<Value> {
    long.bind(3)?;
    long.start(tokens.len(), 4)?;
    super::super::super::store::finish_reference(long, tokens)
}
fn restore(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    tokens: &[i32],
    expected: &Value,
) -> Result<Value> {
    long.bind(3)?;
    let disk = s.disk.as_mut().unwrap();
    ensure!(
        disk.pending.is_none() && !disk.controller.busy(),
        "restore diagnostic while writer active"
    );
    ensure!(
        disk.controller.read(long, tokens.into(), 4, true)?,
        "written prefix read admission"
    );
    let end = Instant::now() + Duration::from_secs(30);
    let report = loop {
        if let Some(c) =
            disk.controller
                .step(long, s.cache.as_mut().unwrap(), false, true, false)?
        {
            break serde_json::to_value(c)?;
        }
        ensure!(Instant::now() < end, "written prefix restore deadline");
        std::thread::sleep(Duration::from_millis(2));
    };
    ensure!(
        report["result"]["restored_tokens"] == 8192,
        "written prefix length changed: {report}"
    );
    let actual = super::super::super::store::finish_reference(long, tokens)?;
    ensure!(
        &actual == expected,
        "written request prefix changed full model replay"
    );
    Ok(json!({"read":report,"actual":actual,"reference":expected}))
}
fn owner_fixture(long: &mut Long<'_>, root: &Path) -> Result<Value> {
    use crate::request_broker::{Broker, Event};
    let input = vec![23; 8192];
    let expected = reference(long, &input)?;
    let mut s = Scheduler::new(long, true)?;
    s.install_cache(long, 256 << 20)?;
    s.install_disk(long, root, 256 << 20)?;
    let (broker, owner) = Broker::new();
    let job = if long.engine.rank == 0 {
        Some(std::thread::spawn(move || {
            let result = (|| -> Result<Vec<i32>> {
                let client = broker.begin(true, Duration::from_secs(30))?;
                client.prepared(Duration::from_secs(5))?;
                client.submit(input.into(), 4)?;
                let mut outputs = Vec::new();
                loop {
                    match client.receive(Duration::from_secs(20))? {
                        Some(Event::Token(t)) => outputs.push(t),
                        Some(Event::End(End::Complete)) => break,
                        other => anyhow::bail!("write-through owner reply {other:?}"),
                    }
                }
                drop(client);
                // No ingress wakes the owner during this interval. The finite
                // SSD maintenance wakeup must continue the pending save.
                std::thread::sleep(Duration::from_secs(5));
                Ok(outputs)
            })();
            broker.shutdown();
            result
        }))
    } else {
        None
    };
    let ticks = s.serve(long, (long.engine.rank == 0).then_some(&owner))?;
    let result = job
        .map(|j| {
            j.join()
                .map_err(|_| anyhow::anyhow!("SSD owner client panic"))
        })
        .transpose()?
        .transpose();
    ensure!(
        s.wire.control.agree(long, 830, [0; 32], result.is_ok())?,
        "SSD owner client failed"
    );
    let outputs = result?;
    let disk = s.disk.as_ref().unwrap();
    ensure!(
        s.queue.completed.len() == 1
            && trace(&s, s.queue.completed[0].status.id)? == expected
            && !disk.controller.busy()
            && disk.started == 1
            && disk
                .records
                .iter()
                .any(|r| r["result"]["outcome"] == "saved"),
        "idle owner failed to complete write-through before shutdown"
    );
    Ok(
        json!({"ticks":ticks,"outputs":outputs,"reference":expected,"records":disk.records,
        "idle_write_completed":true,"shutdown_retired":true}),
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
    let tokens = sets.last().context("disk request fixture")?;
    ensure!(tokens.len() == 16384, "disk request fixture length");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let mut s = Scheduler::new(&long, true)?;
    s.install_cache(&long, 256 << 20)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut r = [0; 64];
    r[..16].copy_from_slice(&stamp.to_le_bytes());
    let all = s.wire.control.exchange(&long, &r)?;
    let stamp = u128::from_le_bytes(all[0][..16].try_into().unwrap());
    let path = std::path::PathBuf::from(format!("evidence/request-disk-{stamp}-rank{rank}"));
    s.install_disk(&mut long, &path, 256 << 20)?;
    let clock = Instant::now();
    // No work is admitted while the lifecycle job publishes its common catalog.
    drain(&mut s, &mut long, &clock)?;
    ensure!(
        s.disk.as_ref().unwrap().controller.enabled(),
        "catalog did not open"
    );
    let cold = reference(&mut long, tokens)?;
    let base = &tokens[..8192];
    let base_ref = reference(&mut long, base)?;
    let id = submit(&mut s, &mut long, &clock, base, true)?;
    drive(&mut s, &mut long, id, &clock)?;
    ensure!(
        trace(&s, id)? == base_ref,
        "write-through changed request trace"
    );
    // Freeze once every rank has made source-copy progress; independent file
    // readiness must not change collective decisions or source ownership.
    let end = Instant::now() + Duration::from_secs(20);
    loop {
        let disk = s.disk.as_ref().unwrap();
        ensure!(
            disk.controller.busy(),
            "writer completed before partial-hold fixture"
        );
        if s.wire
            .control
            .agree(&long, 829, [0; 32], disk.controller.progress().0 > 0)?
        {
            break;
        }
        tick(&mut s, &mut long, Command::Tick, &clock)?;
        ensure!(Instant::now() < end, "writer did not progress");
        std::thread::sleep(Duration::from_millis(2));
    }
    hold(&mut s, &mut long, &clock, 1, true)?;
    let progress = s.disk.as_ref().unwrap().controller.progress();
    let records = s.disk.as_ref().unwrap().records.len();
    let pinned = s
        .cache
        .as_ref()
        .unwrap()
        .find(&long, tokens)?
        .context("pinned prefix missing")?
        .0;
    // Foreground snapshots create real bounded cache pressure while the older
    // source is pinned. The live source slot is reused by automatic admission.
    let mut foreground = Vec::new();
    for token in [13, 14] {
        let input = vec![token; 16384];
        let expected = reference(&mut long, &input)?;
        let fg = submit(&mut s, &mut long, &clock, &input, false)?;
        drive(&mut s, &mut long, fg, &clock)?;
        ensure!(
            trace(&s, fg)? == expected,
            "held writer changed foreground output"
        );
        foreground.push(expected);
        let d = s.disk.as_ref().unwrap();
        ensure!(
            d.controller.progress() == progress
                && d.records.len() == records
                && d.controller.busy()
                && s.cache.as_ref().unwrap().contains(pinned),
            "held writer advanced or lost pin"
        );
    }
    let held_pending = s.disk.as_ref().unwrap().pending;
    ensure!(
        held_pending.is_some(),
        "foreground write-through not queued"
    );
    hold(&mut s, &mut long, &clock, 2, false)?;
    drain(&mut s, &mut long, &clock)?;
    let replay = restore(&mut s, &mut long, tokens, &cold)?;
    let writes = serde_json::to_value(&s.disk.as_ref().unwrap().records)?;
    ensure!(
        writes
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["kind"] == "Write" && v["result"]["outcome"] == "saved"),
        "automatic write never completed"
    );
    // An evicted pending source is skipped normally, before attempting a pin.
    let dropped = s.disk.as_ref().unwrap().dropped;
    s.disk.as_mut().unwrap().pending = Some((u64::MAX, true));
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        s.disk.as_ref().unwrap().dropped == dropped + 1 && long.healthy,
        "stale pending write poisoned owner"
    );
    // Duplicate saves may finish on the file thread without a new source copy.
    // Even completed jobs must remain owned until a foreground hold is released.
    let duplicate = s
        .cache
        .as_ref()
        .unwrap()
        .find(&long, &[vec![14; 16384], vec![15]].concat())?
        .context("duplicate source absent")?
        .0;
    s.disk.as_mut().unwrap().pending = Some((duplicate, true));
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        s.disk.as_ref().unwrap().controller.busy(),
        "duplicate writer not admitted"
    );
    hold(&mut s, &mut long, &clock, 3, true)?;
    let held_digest = s.cache.as_ref().unwrap().digest();
    let count = s.disk.as_ref().unwrap().records.len();
    for _ in 0..32 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
        std::thread::sleep(Duration::from_millis(2));
    }
    ensure!(
        s.disk.as_ref().unwrap().controller.busy()
            && s.disk.as_ref().unwrap().records.len() == count
            && s.cache.as_ref().unwrap().digest() == held_digest,
        "finished worker retired under hold"
    );
    hold(&mut s, &mut long, &clock, 4, false)?;
    drain(&mut s, &mut long, &clock)?;
    ensure!(
        s.disk.as_ref().unwrap().records.back().unwrap()["result"]["outcome"] == "duplicate",
        "duplicate fixture missed"
    );
    // Shutdown while a write owns its source must drain cancellation before
    // dropping the scheduler or releasing its accounted RAM owner.
    let input = vec![15; 8256];
    hold(&mut s, &mut long, &clock, 5, true)?;
    let fg = submit(&mut s, &mut long, &clock, &input, false)?;
    drive(&mut s, &mut long, fg, &clock)?;
    hold(&mut s, &mut long, &clock, 6, false)?;
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        s.disk.as_ref().unwrap().controller.busy(),
        "shutdown writer not admitted"
    );
    let mut shutdown_ticks = 0;
    loop {
        let report = tick(&mut s, &mut long, Command::Stop, &clock)?;
        shutdown_ticks += 1;
        if report.stopped {
            break;
        }
        ensure!(shutdown_ticks < 1000, "shutdown did not drain writer");
        std::thread::sleep(Duration::from_millis(2));
    }
    ensure!(
        !s.disk.as_ref().unwrap().controller.busy()
            && s.disk.as_ref().unwrap().pending.is_none()
            && long.healthy,
        "shutdown lost ownership"
    );
    let final_records = serde_json::to_value(&s.disk.as_ref().unwrap().records)?;
    drop(s.disk.take());
    let owner = owner_fixture(&mut long, &path)?;
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":16384,
        "base_reference":base_ref,"cold_reference":cold,"foreground_references":foreground,"replay":replay,
        "write_records":writes,"final_records":final_records,"owner":owner,
        "partial_writer_frozen":true,"pin_survived_cache_pressure":true,"foreground_exact":true,
        "stale_pending_skipped":true,"completed_writer_retained_under_hold":true,"shutdown_drained":true,
        "shutdown_ticks":shutdown_ticks,"automatic_restores":false,"production_ready":false}),
    )
}

fn drive_restore(s: &mut Scheduler, long: &mut Long<'_>, id: u64, clock: &Instant) -> Result<()> {
    let end = s.stamp + Duration::from_secs(45);
    while !s.queue.completed.iter().any(|c| c.status.id == id) {
        tick(s, long, Command::Tick, clock)?;
        ensure!(s.stamp < end, "automatic restore deadline");
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}
fn clear_memory(s: &mut Scheduler, long: &mut Long<'_>, clock: &Instant) -> Result<()> {
    drain(s, long, clock)?;
    tick(s, long, Command::Tick, clock)?;
    ensure!(
        s.queue.count() == 0 && !s.disk.as_ref().unwrap().controller.busy(),
        "clear with owned state"
    );
    s.disk.as_mut().unwrap().pending = None;
    s.cache = Some(super::super::super::store::Store::new(long, 256 << 20)?);
    for slot in 0..4 {
        long.bind(slot)?;
        long.start(1, 1)?;
        long.cancel()?;
    }
    Ok(())
}
fn partial(s: &mut Scheduler, long: &mut Long<'_>, id: u64, clock: &Instant) -> Result<()> {
    let end = s.stamp + Duration::from_secs(20);
    loop {
        ensure!(s.queue.request(id)?.restoring, "reader ran before fixture");
        if s.wire.control.agree(
            long,
            835,
            [0; 32],
            s.disk.as_ref().unwrap().controller.progress().0 > 0,
        )? {
            break;
        }
        tick(s, long, Command::Tick, clock)?;
        ensure!(s.stamp < end, "no restore copy progress");
        std::thread::sleep(Duration::from_millis(2));
    }
    let r = s.queue.request(id)?;
    ensure!(
        r.cached_tokens == 0 && r.done == 0 && r.outputs.is_empty(),
        "partial restore exposed progress"
    );
    let slot = r.slot.unwrap();
    ensure!(
        long.bind(slot).is_err()
            && long.memory.book().slot(slot)?.phase() == crate::cache_pages::Phase::Restoring,
        "partial destination executable"
    );
    Ok(())
}
fn recover(s: &mut Scheduler, long: &mut Long<'_>, clock: &Instant) -> Result<()> {
    let end = s.stamp + Duration::from_secs(30);
    while !s.disk.as_ref().unwrap().controller.enabled()
        || s.disk.as_ref().unwrap().controller.busy()
    {
        tick(s, long, Command::Tick, clock)?;
        ensure!(s.stamp < end, "automatic disk recovery deadline");
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub fn run_restore(
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
    use super::super::termination::completed;
    let tokens = sets.last().context("automatic restore fixture")?;
    ensure!(tokens.len() == 16384, "restore fixture size");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let cold = reference(&mut long, tokens)?;
    let mut s = Scheduler::new(&long, true)?;
    s.install_cache(&long, 256 << 20)?;
    s.install_sampler(&long, native)?;
    let mut stamp = [0; 64];
    stamp[..16].copy_from_slice(
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_le_bytes(),
    );
    let all = s.wire.control.exchange(&long, &stamp)?;
    let stamp = u128::from_le_bytes(all[0][..16].try_into().unwrap());
    let path = std::path::PathBuf::from(format!("evidence/request-restore-{stamp}-rank{rank}"));
    s.install_disk(&mut long, &path, 256 << 20)?;
    let clock = Instant::now();
    drain(&mut s, &mut long, &clock)?;
    let namespace = s.disk.as_ref().unwrap().controller.namespace()?;
    let source = submit(&mut s, &mut long, &clock, &tokens[..8192], true)?;
    drive_restore(&mut s, &mut long, source, &clock)?;
    clear_memory(&mut s, &mut long, &clock)?;
    let bg = submit(&mut s, &mut long, &clock, tokens, true)?;
    partial(&mut s, &mut long, bg, &clock)?;
    hold(&mut s, &mut long, &clock, 1, true)?;
    let progress = s.disk.as_ref().unwrap().controller.progress();
    let old_slot = s.queue.request(bg)?.slot.unwrap();
    let cancellation = Command::Cancel(bg);
    s.pump(
        &mut long,
        (rank == 0).then(|| (cancellation, clock.elapsed(), true)),
    )?;
    ensure!(
        s.wait(s.stamp)? >= Duration::from_millis(100),
        "held cancellation spins"
    );
    // Same prompt, occupied reader, empty RAM: foreground must run cold rather
    // than wait for retirement under its own hold.
    let fg = submit(&mut s, &mut long, &clock, tokens, false)?;
    drive_restore(&mut s, &mut long, fg, &clock)?;
    ensure!(
        trace(&s, fg)? == cold
            && completed(&s, fg)?.status.cached_tokens == 0
            && s.queue.request(bg)?.restoring
            && s.disk.as_ref().unwrap().controller.progress() == progress,
        "held cancelled restore blocked/corrupted foreground"
    );
    hold(&mut s, &mut long, &clock, 2, false)?;
    drive_restore(&mut s, &mut long, bg, &clock)?;
    ensure!(
        completed(&s, bg)?.reason == End::Cancelled
            && completed(&s, bg)?.outputs.is_empty()
            && long.memory.book().slot(old_slot)?.phase() == crate::cache_pages::Phase::Idle
            && long.memory.book().slot(old_slot)?.pages().is_empty(),
        "cancelled restore retained destination"
    );
    recover(&mut s, &mut long, &clock)?;
    clear_memory(&mut s, &mut long, &clock)?;
    // Foreground restoration must progress through its own native hold.
    let fg_read = submit(&mut s, &mut long, &clock, tokens, false)?;
    ensure!(
        s.queue.request(fg_read)?.restoring && long.priority_hold,
        "foreground reader not exercised"
    );
    drive_restore(&mut s, &mut long, fg_read, &clock)?;
    ensure!(
        trace(&s, fg_read)? == cold && completed(&s, fg_read)?.status.cached_tokens == 8192,
        "foreground SSD replay differs"
    );
    clear_memory(&mut s, &mut long, &clock)?;
    // Sampled policy is owned by the request; SSD files contain no pending token,
    // logits or RNG. Compare with a fresh full-logit cold sampling reference.
    let sampling = crate::sampling::Sampling::new(450, 1.0, 20, 0.95)?;
    long.bind(3)?;
    let mut sampler = crate::candidates::Candidates::new(&long.engine.device, native)?;
    let sampled = super::super::sampled::reference(&mut long, &mut sampler, tokens, sampling, 4)?;
    long.start(1, 1)?;
    long.cancel()?;
    let policy =
        crate::output_policy::Policy::new(false, false, &Value::Null)?.with_sampling(sampling)?;
    let sampled_id = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: tokens.clone().into(),
            output: 4,
            background: true,
            deadline: clock.elapsed() + Duration::from_secs(90),
            policy: Some(policy),
        },
        &clock,
    )?
    .submitted
    .context("sampled SSD request")?;
    drive_restore(&mut s, &mut long, sampled_id, &clock)?;
    ensure!(
        trace(&s, sampled_id)? == sampled
            && completed(&s, sampled_id)?.status.cached_tokens == 8192,
        "SSD sampler changed"
    );
    clear_memory(&mut s, &mut long, &clock)?;
    // Reconstruct the catalog/controller and RAM ownership using the same files.
    drop(s.disk.take());
    s.install_disk(&mut long, &path, 256 << 20)?;
    drain(&mut s, &mut long, &clock)?;
    let reopened = submit(&mut s, &mut long, &clock, tokens, true)?;
    drive_restore(&mut s, &mut long, reopened, &clock)?;
    ensure!(
        trace(&s, reopened)? == cold && completed(&s, reopened)?.status.cached_tokens == 8192,
        "reopen request differs"
    );
    clear_memory(&mut s, &mut long, &clock)?;
    // Expiry during a held partial read must not reuse the destination early.
    let expired = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: tokens.clone().into(),
            output: 4,
            background: true,
            deadline: clock.elapsed() + Duration::from_secs(1),
            policy: None,
        },
        &clock,
    )?
    .submitted
    .context("expiring restore")?;
    partial(&mut s, &mut long, expired, &clock)?;
    hold(&mut s, &mut long, &clock, 3, true)?;
    std::thread::sleep(Duration::from_millis(1100));
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        s.queue.request(expired)?.cancellation() == Some(End::TimedOut)
            && s.queue.request(expired)?.restoring,
        "deadline dropped restore ownership"
    );
    hold(&mut s, &mut long, &clock, 4, false)?;
    drive_restore(&mut s, &mut long, expired, &clock)?;
    ensure!(
        completed(&s, expired)?.reason == End::TimedOut,
        "restore timeout reason lost"
    );
    recover(&mut s, &mut long, &clock)?;
    clear_memory(&mut s, &mut long, &clock)?;
    let (_, keys) = crate::ssd_format::keys(&namespace, &tokens[..8192])?;
    let bad = path
        .join("pages")
        .join(format!("{}.page", crate::ssd_format::hex(&keys[6])));
    let corrupt = (|| -> Result<()> {
        if rank == 2 {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&bad)?;
            f.seek(SeekFrom::End(-1))?;
            let mut b = [0];
            f.read_exact(&mut b)?;
            b[0] ^= 1;
            f.seek(SeekFrom::End(-1))?;
            f.write_all(&b)?;
            f.sync_all()?;
        }
        Ok(())
    })();
    ensure!(
        s.wire.control.agree(&long, 836, [0; 32], corrupt.is_ok())?,
        "corruption fixture failed"
    );
    corrupt?;
    let corrupt_id = submit(&mut s, &mut long, &clock, tokens, true)?;
    drive_restore(&mut s, &mut long, corrupt_id, &clock)?;
    ensure!(
        trace(&s, corrupt_id)? == cold && completed(&s, corrupt_id)?.status.cached_tokens == 0,
        "corrupt SSD did not cold replay"
    );
    let failed = s
        .disk
        .as_ref()
        .unwrap()
        .records
        .iter()
        .rev()
        .find(|r| r["kind"] == "Read" && r["result"]["outcome"] == "discarded")
        .context("no rejected read")?
        .clone();
    if rank == 2 {
        ensure!(
            failed["result"]["copy_bytes"].as_u64().unwrap() >= 5 * 11 * (256 * 528 + 64 * 256),
            "corruption not after partial H2D"
        );
    }
    recover(&mut s, &mut long, &clock)?;
    drain(&mut s, &mut long, &clock)?;
    let records = serde_json::to_value(&s.queue.completed)?;
    let disk_records = serde_json::to_value(&s.disk.as_ref().unwrap().records)?;
    let end = tick(&mut s, &mut long, Command::Stop, &clock)?;
    ensure!(end.stopped && long.healthy, "restore owner did not stop");
    drop(s);
    let http = http::fixture(
        &mut long,
        native,
        &std::path::PathBuf::from(format!("{}.http", path.display())),
    )?;
    let watchdog = http::watchdog(
        &mut long,
        &std::path::PathBuf::from(format!("{}.watch", path.display())),
        tokens,
        &cold,
    )?;
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":16384,"cache_root":path,
        "cold_reference":cold,"sampled_reference":sampled,"completed":records,"disk_records":disk_records,"late_failure":failed,
        "http":http,"watchdog":watchdog,"restored_tokens":8192,"partial_state_unrunnable":true,"held_cancel_does_not_spin":true,
        "foreground_busy_fallback_exact":true,"cancel_discard_exact":true,"foreground_restore_under_hold_exact":true,
        "sampled_restore_exact":true,"reopen_restore_exact":true,"held_deadline_cleanup_exact":true,
        "late_corruption_cold_fallback_exact":true,"automatic_tier_recovery":true,"production_ready":false}),
    )
}

#[path = "request_http_disk_tests.rs"]
mod http;
