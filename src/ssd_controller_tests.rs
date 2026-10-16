use super::*;
use std::time::{Duration, Instant};
fn native_hold(
    controller: &mut Controller,
    long: &mut Long<'_>,
    ram: &mut Ram,
    partial: bool,
) -> Result<()> {
    use super::super::super::priority::Boundary;
    use crate::priority::Snapshot;
    fn progress(c: &Controller) -> (usize, usize) {
        match &c.state {
            State::Writing(w) => w.progress(),
            State::Reading(r) => r.progress(),
            _ => (0, 0),
        }
    }
    let mut boundary = Boundary::new(long)?;
    if partial {
        let control = Control::new(&long.engine.device)?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            ensure!(
                controller
                    .step(long, ram, false, progress(controller).0 == 0, false)?
                    .is_none(),
                "job retired before hold fixture"
            );
            if control.agree(long, 639, [0; 32], progress(controller).0 > 0)? {
                break;
            }
            ensure!(Instant::now() < deadline, "partial native hold deadline");
        }
    }
    let head = (long.engine.rank == 0).then_some(Snapshot {
        revision: 1,
        hold: true,
    });
    boundary.apply(long, head)?;
    let before = progress(controller);
    for _ in 0..64 {
        ensure!(
            controller.step(long, ram, false, true, false)?.is_none(),
            "native hold retired job"
        );
        ensure!(
            progress(controller) == before,
            "native hold allowed new SSD copies"
        );
    }
    ensure!(
        controller.open(long).is_err(),
        "native hold admitted lifecycle work"
    );
    let head = (long.engine.rank == 0).then_some(Snapshot {
        revision: 2,
        hold: false,
    });
    boundary.apply(long, head)?;
    ensure!(!long.priority_hold, "native fixture failed to release");
    Ok(())
}
fn finish(controller: &mut Controller, long: &mut Long<'_>, ram: &mut Ram) -> Result<Completion> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut tick = 0u64;
    loop {
        if let Some(report) = controller.step(
            long,
            ram,
            false,
            !tick.is_multiple_of(long.engine.rank as u64 + 2),
            false,
        )? {
            return Ok(report);
        }
        tick += 1;
        ensure!(Instant::now() < deadline, "controller fixture deadline");
        std::thread::yield_now();
    }
}
fn foreground_lane(
    controller: &mut Controller,
    long: &mut Long<'_>,
    ram: &mut Ram,
    tokens: &[i32],
    expected_state: &str,
    reference: &serde_json::Value,
) -> Result<serde_json::Value> {
    if controller.spare.is_none() {
        return Ok(serde_json::Value::Null);
    }
    let control = Control::new(&long.engine.device)?;
    long.bind(1)?;
    ensure!(
        controller.read(long, tokens.into(), 4, true)?,
        "background lane fixture"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        ensure!(
            controller
                .step(long, ram, false, controller.progress().0 == 0, false)?
                .is_none(),
            "background completed before parking"
        );
        if control.agree(long, 640, [0; 32], controller.progress().0 > 0)? {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "foreground lane initial read deadline"
        );
    }
    let mut boundary = super::super::super::priority::Boundary::new(long)?;
    let head = long.engine.rank == 0;
    boundary.apply(
        long,
        head.then_some(crate::priority::Snapshot {
            revision: 1,
            hold: true,
        }),
    )?;
    let before = controller.progress();
    ensure!(
        controller.foreground_hint(tokens)? == Some(8256),
        "parked controller lost foreground hint"
    );
    long.bind(3)?;
    ensure!(
        controller.read(long, tokens.into(), 4, false)? && controller.parked(),
        "foreground lane admission"
    );
    let foreground = serde_json::to_value(finish(controller, long, ram)?)?;
    ensure!(
        !controller.parked() && controller.progress() == before,
        "foreground advanced parked background bytes"
    );
    long.bind(3)?;
    ensure!(
        long.state(8256)? == expected_state && &finish_reference(long, tokens)? == reference,
        "foreground lane full state/logits/output"
    );
    // Cancel a second foreground copy while retaining the original background
    // worker and partial destination; all ranks must retire the correct owner.
    ensure!(
        controller.read(long, tokens.into(), 4, false)?,
        "foreground cancel lane admission"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let cancelled = loop {
        if let Some(done) = controller.step(long, ram, false, true, true)? {
            break serde_json::to_value(done)?;
        }
        ensure!(
            Instant::now() < deadline,
            "foreground lane cancellation deadline"
        );
    };
    ensure!(
        cancelled["result"]["restored_tokens"].is_null() && controller.progress() == before,
        "foreground cancel changed background"
    );
    boundary.apply(
        long,
        head.then_some(crate::priority::Snapshot {
            revision: 2,
            hold: false,
        }),
    )?;
    let background = serde_json::to_value(finish(controller, long, ram)?)?;
    long.bind(1)?;
    ensure!(
        background["result"]["restored_tokens"] == 8256
            && long.state(8256)? == expected_state
            && &finish_reference(long, tokens)? == reference,
        "background lane resume full state/logits/output"
    );
    Ok(
        serde_json::json!({"foreground":foreground,"cancelled":cancelled,"background":background,"held_progress":before,"exact":true}),
    )
}
fn open_report(value: Completion, enabled: bool, entries: usize) -> Result<serde_json::Value> {
    let Completion::Open(ref report) = value else {
        anyhow::bail!("expected open completion");
    };
    ensure!(
        report.enabled == enabled && report.entries == entries,
        "catalog result mismatch"
    );
    Ok(serde_json::to_value(value)?)
}
fn foreground_writer_lane(
    controller: &mut Controller,
    long: &mut Long<'_>,
    ram: &mut Ram,
    tokens: &[i32],
    expected_state: &str,
    reference: &serde_json::Value,
) -> Result<serde_json::Value> {
    if controller.spare.is_none() {
        return Ok(serde_json::Value::Null);
    }
    long.bind(0)?;
    long.start(tokens.len(), 4)?;
    prefill_to(long, &tokens[..8320])?;
    let id = ram.save(long)?.id.context("foreground writer source")?;
    long.cancel()?;
    ensure!(
        controller.write_mode(long, ram, id, false)?,
        "optional foreground writer admission"
    );
    let control = Control::new(&long.engine.device)?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        ensure!(
            controller.step(long, ram, false, true, false)?.is_none(),
            "foreground writer finished before partial-copy fixture"
        );
        let State::Writing(writer) = &mut controller.state else {
            anyhow::bail!("foreground writer fixture lost owner");
        };
        writer.pause_for_read(long)?;
        if control.agree(long, 641, [0; 32], controller.progress().0 > 0)? {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "foreground writer partial-copy deadline"
        );
    }
    let before = controller.progress();
    // The recurrent snapshot alone exceeds this bound. Keep a demonstrably
    // partial save so cancellation cannot have published the extra test entry.
    ensure!(
        control.agree(long, 642, [0; 32], before.0 > 0 && before.0 < (20 << 20))?,
        "writer fixture did not stop at a partial copy"
    );
    let mut boundary = super::super::super::priority::Boundary::new(long)?;
    let head = long.engine.rank == 0;
    boundary.apply(
        long,
        head.then_some(crate::priority::Snapshot {
            revision: 1,
            hold: true,
        }),
    )?;
    ensure!(
        controller.foreground_hint(tokens)? == Some(8256),
        "foreground save hid completed prefix"
    );
    long.bind(3)?;
    ensure!(
        controller.read(long, tokens.into(), 4, false)? && controller.parked(),
        "foreground writer lane admission"
    );
    let foreground = serde_json::to_value(finish(controller, long, ram)?)?;
    ensure!(
        !controller.parked() && controller.progress() == before,
        "restore advanced optional writer DMA"
    );
    ensure!(
        long.state(8256)? == expected_state && &finish_reference(long, tokens)? == reference,
        "writer lane full state/logits/output"
    );
    ensure!(
        controller.read(long, tokens.into(), 4, false)?,
        "writer lane cancellation admission"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let cancelled = loop {
        if let Some(done) = controller.step(long, ram, false, true, true)? {
            break serde_json::to_value(done)?;
        }
        ensure!(
            Instant::now() < deadline,
            "writer lane reader cancellation deadline"
        );
    };
    ensure!(
        cancelled["result"]["restored_tokens"].is_null() && controller.progress() == before,
        "reader cancellation changed parked writer"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let writer_cancelled = loop {
        if let Some(done) = controller.step(long, ram, false, false, true)? {
            break serde_json::to_value(done)?;
        }
        ensure!(
            Instant::now() < deadline,
            "parked writer cancellation deadline"
        );
    };
    ensure!(
        writer_cancelled["kind"] == "Write" && writer_cancelled["result"]["outcome"].is_null(),
        "partial writer published after cancellation"
    );
    boundary.apply(
        long,
        head.then_some(crate::priority::Snapshot {
            revision: 2,
            hold: false,
        }),
    )?;
    Ok(
        serde_json::json!({"foreground":foreground,"cancelled":cancelled,"writer_cancelled":writer_cancelled,"held_progress":before,"exact":true}),
    )
}
fn await_inventory(
    controller: &mut Controller,
    long: &mut Long<'_>,
    ram: &mut Ram,
    control: &Control,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        ensure!(
            controller.step(long, ram, true, false, false)?.is_none(),
            "foreground published catalog"
        );
        let ready = matches!(&controller.state,State::Opening(job) if job.order.is_some()&&job.phase==0&&job.job.is_some());
        if control.agree(long, 630, [0; 32], ready)? {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "catalog fixture inventory deadline"
        );
    }
}
fn cancel_open(
    controller: &mut Controller,
    long: &mut Long<'_>,
    ram: &mut Ram,
    control: &Control,
) -> Result<serde_json::Value> {
    controller.open(long)?;
    await_inventory(controller, long, ram, control)?;
    ensure!(
        controller
            .step(long, ram, true, false, long.engine.rank == 0)?
            .is_none(),
        "foreground retired catalog"
    );
    open_report(finish(controller, long, ram)?, false, 0)
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
) -> Result<serde_json::Value> {
    let tokens = sets.last().context("controller fixture")?;
    ensure!(tokens.len() == 16384, "controller fixture length");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let control = Control::new(&long.engine.device)?;
    let mut stamp = [0; 64];
    stamp[..16].copy_from_slice(
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_le_bytes(),
    );
    let all = control.exchange(&long, &stamp)?;
    let stamp = u128::from_le_bytes(all[0][..16].try_into().unwrap());
    let path = PathBuf::from(format!("evidence/ssd-controller-{stamp}-rank{rank}"));
    let mut ram = Ram::new(&long, 256 << 20)?;
    let mut controller = Controller::new(&mut long, &path, 256 << 20)?;
    ensure!(
        controller.open(&mut long).is_err()
            && controller.write(&mut long, &mut ram, 1).is_err()
            && controller
                .read(&mut long, vec![1; 64].into(), 4, false)
                .is_err(),
        "busy controller allowed second owner"
    );
    await_inventory(&mut controller, &mut long, &mut ram, &control)?;
    native_hold(&mut controller, &mut long, &mut ram, false)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    let foreground_tokens = vec![45; 2177];
    long.bind(0)?;
    long.start(foreground_tokens.len(), 4)?;
    prefill_to(&mut long, &foreground_tokens[..2176])?;
    ensure!(
        controller
            .step(&mut long, &mut ram, true, true, false)?
            .is_none(),
        "foreground open committed"
    );
    let foreground = finish_reference(&mut long, &foreground_tokens)?;
    let initial = open_report(finish(&mut controller, &mut long, &mut ram)?, true, 0)?;
    let namespace = controller.idle()?.id.namespace;
    let id = controller.idle()?.id.clone();
    long.start(tokens.len(), 4)?;
    let mut states = Vec::new();
    let mut ids = Vec::new();
    for n in [8192, 8256] {
        prefill_to(&mut long, &tokens[..n])?;
        states.push(long.state(n)?);
        ids.push(ram.save(&mut long)?.id.context("RAM source")?);
    }
    let reference = finish_reference(&mut long, tokens)?;
    let mut writes = Vec::new();
    for source in ids {
        ensure!(
            controller.write(&mut long, &mut ram, source)?,
            "controller writer admission"
        );
        native_hold(&mut controller, &mut long, &mut ram, true)?;
        let report = finish(&mut controller, &mut long, &mut ram)?;
        let value = serde_json::to_value(report)?;
        ensure!(
            value["kind"] == "Write" && value["result"]["outcome"] == "saved",
            "controller write result"
        );
        writes.push(value);
    }
    controller.open(&mut long)?;
    await_inventory(&mut controller, &mut long, &mut ram, &control)?;
    long.start(foreground_tokens.len(), 4)?;
    prefill_to(&mut long, &foreground_tokens[..2176])?;
    ensure!(
        controller
            .step(&mut long, &mut ram, true, false, false)?
            .is_none(),
        "foreground reopen committed"
    );
    ensure!(
        finish_reference(&mut long, &foreground_tokens)? == foreground,
        "catalog changed foreground model"
    );
    let reopened = open_report(finish(&mut controller, &mut long, &mut ram)?, true, 2)?;
    long.bind(1)?;
    ensure!(
        controller.read(&mut long, tokens.clone().into(), 4, true)?,
        "controller reader admission"
    );
    long.bind(0)?;
    native_hold(&mut controller, &mut long, &mut ram, true)?;
    let read = serde_json::to_value(finish(&mut controller, &mut long, &mut ram)?)?;
    ensure!(
        read["kind"] == "Read" && read["result"]["restored_tokens"] == 8256,
        "controller reader length"
    );
    long.bind(1)?;
    ensure!(
        long.state(8256)? == states[1] && finish_reference(&mut long, tokens)? == reference,
        "controller model restore"
    );
    // Foreground restoration must be allowed while the lease is held; otherwise
    // the request that caused the hold would deadlock on its own cache lookup.
    let mut priority = super::super::super::priority::Boundary::new(&long)?;
    priority.apply(
        &mut long,
        (rank == 0).then_some(crate::priority::Snapshot {
            revision: 1,
            hold: true,
        }),
    )?;
    ensure!(
        controller.read(&mut long, tokens.clone().into(), 4, false)?,
        "foreground restore admission while held"
    );
    let foreground_read = serde_json::to_value(finish(&mut controller, &mut long, &mut ram)?)?;
    ensure!(
        foreground_read["result"]["restored_tokens"] == 8256
            && finish_reference(&mut long, tokens)? == reference,
        "foreground cache/model during native hold"
    );
    priority.apply(
        &mut long,
        (rank == 0).then_some(crate::priority::Snapshot {
            revision: 2,
            hold: false,
        }),
    )?;
    let foreground_lane = foreground_lane(
        &mut controller,
        &mut long,
        &mut ram,
        tokens,
        &states[1],
        &reference,
    )?;
    let foreground_writer_lane = foreground_writer_lane(
        &mut controller,
        &mut long,
        &mut ram,
        tokens,
        &states[1],
        &reference,
    )?;
    long.bind(0)?;
    let cancelled = cancel_open(&mut controller, &mut long, &mut ram, &control)?;
    // A genuine rank-local lock conflict must disable only the file tier.
    let held = if rank == 1 {
        Some(Store::open(&path, id.clone(), 256 << 20)?)
    } else {
        None
    };
    ensure!(control.agree(&long, 631, [0; 32], true)?, "lock fixture");
    controller.open(&mut long)?;
    let open_failed = open_report(finish(&mut controller, &mut long, &mut ram)?, false, 0)?;
    drop(held);
    controller.open(&mut long)?;
    let recovered = open_report(finish(&mut controller, &mut long, &mut ram)?, true, 2)?;
    // Cancel to release every catalog, then remove a valid entry on one rank.
    cancel_open(&mut controller, &mut long, &mut ram, &control)?;
    let key = format::keys(&namespace, &tokens[..8256])?.0;
    let removed = (|| -> Result<()> {
        if rank == 2 {
            let mut store = Store::open(&path, id.clone(), 256 << 20)?;
            store.forget(&key)?;
        }
        Ok(())
    })();
    ensure!(
        control.agree(&long, 632, [0; 32], removed.is_ok())?,
        "missing entry fixture"
    );
    removed?;
    controller.open(&mut long)?;
    let intersection = open_report(finish(&mut controller, &mut long, &mut ram)?, true, 1)?;
    // After all scans finish, fail a real retain_order write on rank 1.
    controller.open(&mut long)?;
    await_inventory(&mut controller, &mut long, &mut ram, &control)?;
    let kept_key = format::keys(&namespace, &tokens[..8192])?.0;
    let bad = path
        .join("entries")
        .join(format!("{}.touch.tmp", format::hex(&kept_key)));
    let blocked = if rank == 1 {
        fs::create_dir(&bad)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 633, [0; 32], blocked.is_ok())?,
        "touch fault fixture"
    );
    blocked?;
    let apply_failed = open_report(finish(&mut controller, &mut long, &mut ram)?, false, 0)?;
    let repaired = if rank == 1 {
        fs::remove_dir(&bad)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 634, [0; 32], repaired.is_ok())?,
        "touch fixture repair"
    );
    repaired?;
    controller.open(&mut long)?;
    let final_open = open_report(finish(&mut controller, &mut long, &mut ram)?, true, 1)?;
    long.bind(3)?;
    ensure!(
        controller.read(&mut long, tokens.clone().into(), 4, false)?,
        "retained reader admission"
    );
    long.bind(0)?;
    let retained = serde_json::to_value(finish(&mut controller, &mut long, &mut ram)?)?;
    ensure!(
        retained["result"]["restored_tokens"] == 8192,
        "intersection kept wrong prefix"
    );
    long.bind(3)?;
    ensure!(long.state(8192)? == states[0], "retained full tail changed");
    prefill_to(&mut long, &tokens[..8256])?;
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "recovered model changed"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "lifecycle changed paused model");
    long.resume()?;
    long.finish()?;
    long.memory.book().check()?;
    let digest = format::hex(&controller.idle()?.digest());
    // A divergent control decision is fatal before any extra reconciliation work.
    controller.open(&mut long)?;
    ensure!(
        controller
            .step(&mut long, &mut ram, rank == 3, false, false)
            .is_err()
            && !long.healthy
            && long.start(64, 4).is_err(),
        "lifecycle control mismatch not fail-stop"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"snapshot_tokens":[8192,8256],"snapshot_states":states,"reference":reference,"foreground_reference":foreground,"foreground_lane":foreground_lane,"foreground_writer_lane":foreground_writer_lane,"namespace":format::hex(&namespace),"cache_root":path,"final_catalog_digest":digest,"initial":initial,"writes":writes,"reopened":reopened,"read":read,"cancelled":cancelled,"open_failed":open_failed,"recovered":recovered,"intersection":intersection,"apply_failed":apply_failed,"final_open":final_open,"retained":retained,"foreground_model_during_catalog_hold_exact":true,"paused_model_exact":true,"exclusive_unified_lifecycle_read_write":true,"rank_local_open_and_mutation_errors_disable_tier":true,"common_intersection_recovers_exact_model":true,"rank_local_cancel_waits_for_workers":true,"control_disagreement_poisons_owner":true,"catalog_file_io_off_owner":true,"native_internal_holds_freeze_lifecycle_writer_reader":true,"foreground_restore_during_native_hold_exact":true,"native_priority_qualified":false,"pinned_stage_bytes":CHUNK,"codec_stage_bytes":CHUNK,"exchange_stage_bytes":CHUNK,"max_catalog_keys":MAX_KEYS}),
    )
}
