use super::super::{
    Engine,
    store::{finish_reference, prefill_to},
};
use super::*;
use crate::{prepared::Prepared, priority::Gate};
use anyhow::Context;
use std::{net::SocketAddr, path::Path, time::Duration};
fn apply(
    boundary: &mut Boundary,
    long: &mut Long<'_>,
    gate: &mut Gate,
    ms: u64,
    queued: bool,
) -> Result<()> {
    let head = if long.engine.rank == 0 {
        Some(gate.snapshot(Duration::from_millis(ms), queued)?)
    } else {
        None
    };
    let revision = boundary.apply(long, head)?;
    if long.engine.rank == 0 {
        gate.acknowledge(revision)?;
        ensure!(gate.applied() == revision, "unacknowledged boundary");
    }
    Ok(())
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
    let tokens = sets.last().context("priority fixture")?;
    ensure!(tokens.len() == 16384, "priority fixture must be 16K");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let mut boundary = Boundary::new(&long)?;
    let mut gate = Gate::default();
    let mut ram = super::super::store::Store::new(&long, 256 << 20)?;
    long.start(tokens.len(), 4)?;
    let reference = finish_reference(&mut long, tokens)?;
    long.bind(2)?;
    long.start_mode(64, 4, true)?;
    long.prefill(0, &[9; 64])?;
    let independent = long.state(64)?;
    long.pause()?;
    long.bind(1)?;
    long.start_mode(64, 4, true)?;
    long.prefill(0, &[8; 64])?;
    long.bind(0)?;
    long.start_mode(tokens.len(), 4, true)?;
    prefill_to(&mut long, &tokens[..8192])?;
    let paused = long.state(8192)?;
    let pending = long.sessions[0].pending;
    if rank == 0 {
        gate.update("router-a", 10, true, 1.0, Duration::ZERO)?;
        gate.update("router-b", 20, true, 2.0, Duration::ZERO)?;
    }
    apply(&mut boundary, &mut long, &mut gate, 0, false)?;
    ensure!(
        long.prefill(8192, &tokens[8192..8256]).is_err()
            && long.resume().is_err()
            && long.start_mode(64, 4, true).is_err(),
        "background bypassed hold"
    );
    ensure!(
        ram.save(&mut long).is_err() && long.checkpoint().is_err() && long.healthy,
        "held snapshot save bypassed priority or poisoned engine"
    );
    long.bind(3)?;
    long.start(2177, 4)?;
    let foreground_tokens = vec![45; 2177];
    let foreground = finish_reference(&mut long, &foreground_tokens)?;
    // Releasing one owner cannot release another. Expiry at exactly TTL releases.
    if rank == 0 {
        gate.update("router-a", 11, false, 1.0, Duration::from_millis(1))?;
    }
    apply(&mut boundary, &mut long, &mut gate, 1, false)?;
    long.bind(0)?;
    ensure!(
        long.state(8192)? == paused && long.sessions[0].pending == pending,
        "paused prefill changed"
    );
    ensure!(
        long.memory.book().slot(0)?.phase() == Phase::Paused,
        "second lease lost"
    );
    // Cancel and immediately reuse an owned paused slot for an independently
    // paused foreground request, without an intervening priority boundary.
    long.bind(1)?;
    long.cancel()?;
    long.start(64, 4)?;
    long.prefill(0, &[7; 64])?;
    let reused = long.state(64)?;
    long.pause()?;
    long.bind(0)?;
    if rank == 0 {
        ensure!(
            gate.wait(Duration::from_millis(1999))? == Duration::from_millis(1),
            "lease wake deadline"
        );
    }
    apply(&mut boundary, &mut long, &mut gate, 2000, false)?;
    ensure!(
        long.memory.book().slot(2)?.phase() == Phase::Paused,
        "native release resumed independent pause"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "resumed prefill reference mismatch"
    );
    long.start_mode(tokens.len(), 4, true)?;
    let mut next = prefill_to(&mut long, tokens)?;
    let prefill_state = long.state(tokens.len())?;
    let mut logits = vec![long.logits()?];
    let mut outputs = vec![next];
    next = long.step(tokens.len(), next)?;
    logits.push(long.logits()?);
    outputs.push(next);
    let decode_state = long.state(tokens.len() + 1)?;
    let ticket = if rank == 0 {
        Some(gate.enter(Duration::from_millis(2001))?)
    } else {
        None
    };
    apply(&mut boundary, &mut long, &mut gate, 2001, false)?;
    ensure!(
        long.step(tokens.len() + 1, next).is_err(),
        "held decoder ran"
    );
    long.bind(3)?;
    long.start(2177, 4)?;
    ensure!(
        finish_reference(&mut long, &foreground_tokens)? == foreground,
        "foreground reference changed"
    );
    if let Some(ticket) = ticket {
        gate.leave(ticket, Duration::from_millis(2002))?;
    }
    apply(&mut boundary, &mut long, &mut gate, 2002, true)?;
    long.bind(0)?;
    ensure!(
        long.memory.book().slot(0)?.phase() == Phase::Paused,
        "queued foreground did not hold"
    );
    apply(&mut boundary, &mut long, &mut gate, 2003, false)?;
    ensure!(
        long.state(tokens.len() + 1)? == decode_state && long.sessions[0].pending == Some(next),
        "decode state/pending changed"
    );
    for _ in 0..2 {
        next = long.step(long.sessions[0].tokens.len(), next)?;
        outputs.push(next);
        logits.push(long.logits()?);
    }
    let final_state = long.state(tokens.len() + 3)?;
    long.finish()?;
    let resumed = serde_json::json!({"prefill_state":prefill_state,"final_state":final_state,"logits":logits,"outputs":outputs});
    ensure!(resumed == reference, "resumed decoder reference mismatch");
    long.bind(1)?;
    ensure!(
        long.memory.book().slot(1)?.phase() == Phase::Paused && long.state(64)? == reused,
        "native release resumed a replacement foreground owner"
    );
    long.resume()?;
    long.finish()?;
    long.bind(2)?;
    ensure!(
        long.state(64)? == independent,
        "independent paused owner changed"
    );
    long.resume()?;
    long.finish()?;
    long.memory.book().check()?;
    let boundaries = boundary.tick;
    // A stale/different progress record must poison all ranks before mask changes.
    if rank == 3 {
        long.sessions[2].pending = Some(42);
    }
    let head = if rank == 0 {
        Some(gate.snapshot(Duration::from_millis(2004), false)?)
    } else {
        None
    };
    ensure!(
        boundary.apply(&mut long, head).is_err() && !long.healthy && long.start(64, 4).is_err(),
        "priority disagreement did not fail closed"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"reference":reference,"foreground_reference":foreground,"boundaries":boundaries,"prefill_resume_exact":true,"decode_resume_exact":true,"independent_pause_preserved":true,"paused_cancel_exact":true,"replacement_foreground_pause_preserved":true,"lease_expiry_and_composition":true,"all_rank_drain_before_ack":true,"disagreement_poisons":true,"production_native_priority_qualified":false}),
    )
}
