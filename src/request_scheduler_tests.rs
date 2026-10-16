use super::super::store::finish_reference;
use super::*;
struct Clock {
    start: Instant,
    offset: Duration,
}
impl Clock {
    fn now(&self) -> Duration {
        self.start.elapsed() + self.offset
    }
}
fn tick(s: &mut Scheduler, long: &mut Long<'_>, command: Command, clock: &Clock) -> Result<Report> {
    s.tick(
        long,
        (long.engine.rank == 0).then(|| (command, clock.now())),
    )
}
fn submit(
    s: &mut Scheduler,
    long: &mut Long<'_>,
    tokens: &[i32],
    background: bool,
    clock: &Clock,
) -> Result<u64> {
    tick(
        s,
        long,
        Command::Submit {
            tokens: tokens.into(),
            output: 4,
            background,
            deadline: clock.now() + Duration::from_secs(300),
            policy: None,
        },
        clock,
    )?
    .submitted
    .context("fixture submission rejected")
}
fn completion(s: &Scheduler, id: u64) -> Result<&crate::request_queue::Completed> {
    s.queue
        .completed
        .iter()
        .find(|c| c.status.id == id)
        .context("fixture missing completion")
}
fn reference(s: &Scheduler, id: u64) -> Result<serde_json::Value> {
    let c = completion(s, id)?;
    ensure!(c.reason == End::Complete, "fixture did not complete");
    Ok(s.traces
        .get(&id)
        .context("fixture missing trace")?
        .value(&c.outputs))
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
    let tokens = sets.last().context("request fixture")?;
    ensure!(tokens.len() == 16384, "request fixture must be 16K");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    long.start(tokens.len(), 4)?;
    let cold = finish_reference(&mut long, tokens)?;
    let foreground_tokens = vec![45; 2177];
    long.start(foreground_tokens.len(), 4)?;
    let foreground = finish_reference(&mut long, &foreground_tokens)?;
    let mut s = Scheduler::new(&long, true)?;
    let mut clock = Clock {
        start: Instant::now(),
        offset: Duration::ZERO,
    };
    let bg = submit(&mut s, &mut long, tokens, true, &clock)?;
    while s.queue.request(bg)?.done < 8192 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    let second = submit(&mut s, &mut long, tokens, true, &clock)?;
    let active_timeout = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: vec![7; 64].into(),
            output: 4,
            background: true,
            deadline: clock.now() + Duration::from_secs(60),
            policy: None,
        },
        &clock,
    )?
    .submitted
    .context("active deadline submit")?;
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        s.queue.request(active_timeout)?.outputs.len() == 1,
        "third background slot did not prefill"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "fixture".into(),
            epoch: 1,
            foreground: true,
            seconds: 300.0,
        },
        &clock,
    )?;
    let before = s.queue.status(bg, s.stamp)?;
    let bg_slot = s.queue.request(bg)?.slot.unwrap();
    long.bind(bg_slot)?;
    let state = long.state(before.prefill_done)?;
    let fg = submit(&mut s, &mut long, &foreground_tokens, false, &clock)?;
    while s.queue.request(fg).is_ok() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    let after = s.queue.status(bg, s.stamp)?;
    ensure!(
        before.prefill_done == after.prefill_done
            && before.output_tokens == after.output_tokens
            && before.active_ns == after.active_ns
            && after.paused_ns > before.paused_ns,
        "held progress/timers changed"
    );
    long.bind(bg_slot)?;
    ensure!(
        long.state(before.prefill_done)? == state,
        "held model changed"
    );
    ensure!(
        reference(&s, fg)? == foreground,
        "queued foreground reference changed"
    );
    tick(&mut s, &mut long, Command::Cancel(second), &clock)?;
    ensure!(
        completion(&s, second)?.reason == End::Cancelled,
        "paused cancel"
    );
    // Queue a request under the lease, then cancel without ever admitting it.
    let queued = submit(&mut s, &mut long, tokens, true, &clock)?;
    ensure!(
        s.queue.request(queued)?.slot.is_none(),
        "held queued request admitted"
    );
    tick(&mut s, &mut long, Command::Cancel(queued), &clock)?;
    ensure!(
        completion(&s, queued)?.status.admission_count == 0,
        "queued cancel admitted"
    );
    let timed = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: vec![7; 64].into(),
            output: 4,
            background: true,
            deadline: clock.now() + Duration::from_secs(1),
            policy: None,
        },
        &clock,
    )?
    .submitted
    .context("timed submission")?;
    clock.offset += Duration::from_secs(62);
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        completion(&s, active_timeout)?.reason == End::TimedOut
            && completion(&s, active_timeout)?.status.admission_count == 1,
        "active paused timeout"
    );
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        completion(&s, timed)?.reason == End::TimedOut
            && completion(&s, timed)?.status.admission_count == 0,
        "queued deadline"
    );
    tick(
        &mut s,
        &mut long,
        Command::Lease {
            owner: "fixture".into(),
            epoch: 2,
            foreground: false,
            seconds: 1.0,
        },
        &clock,
    )?;
    while s.queue.request(bg)?.outputs.is_empty() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    // Foreground arrival while decoding preserves the sampled pending token.
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    let before_decode = s.queue.status(bg, s.stamp)?;
    ensure!(
        before_decode.decode_calls == 1,
        "decode pause fixture cursor"
    );
    let fg2 = submit(&mut s, &mut long, &foreground_tokens, false, &clock)?;
    while s.queue.request(fg2).is_ok() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    let after_decode = s.queue.status(bg, s.stamp)?;
    ensure!(
        before_decode.output_tokens == after_decode.output_tokens
            && before_decode.active_ns == after_decode.active_ns,
        "decode advanced during foreground"
    );
    while s.queue.count() > 0 {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    // Apply the final foreground leave and verify there is no stale ticket/hold.
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    ensure!(
        !long.priority_hold && !s.queue.gate.snapshot(s.stamp, false)?.hold,
        "foreground ticket leaked"
    );
    ensure!(
        reference(&s, bg)? == cold && reference(&s, fg2)? == foreground,
        "queued/resumed full reference"
    );
    let completed = serde_json::to_value(&s.queue.completed)?;
    let bg_stats = &completion(&s, bg)?.status;
    ensure!(
        bg_stats.admission_count == 1
            && bg_stats.prefill_done == tokens.len()
            && bg_stats.output_tokens == 4
            && bg_stats.decode_calls == 3
            && bg_stats.pause_count >= 2
            && bg_stats.resume_count == bg_stats.pause_count
            && bg_stats.active_ns > 0
            && bg_stats.paused_ns > 0,
        "request accounting"
    );
    let rejected = tick(
        &mut s,
        &mut long,
        Command::Submit {
            tokens: vec![1].into(),
            output: 0,
            background: false,
            deadline: clock.now() + Duration::from_secs(1),
            policy: None,
        },
        &clock,
    )?;
    ensure!(
        rejected.submitted.is_none() && rejected.rejected.is_some() && !long.priority_hold,
        "invalid request/ticket"
    );
    // A duplicate/stale cancellation cannot cancel the new occupant of a slot.
    let reused = submit(&mut s, &mut long, &foreground_tokens, false, &clock)?;
    ensure!(
        tick(&mut s, &mut long, Command::Cancel(second), &clock)?
            .rejected
            .is_some(),
        "stale cancellation accepted"
    );
    while s.queue.request(reused).is_ok() {
        tick(&mut s, &mut long, Command::Tick, &clock)?;
    }
    ensure!(
        reference(&s, reused)? == foreground,
        "stale cancel changed replacement"
    );
    tick(&mut s, &mut long, Command::Tick, &clock)?;
    long.memory.book().check()?;
    let commands = s.wire.serial;
    // Detect a rank-local ownership discrepancy before any next admission/work.
    if rank == 3 {
        s.queue.submit(
            vec![1].into(),
            1,
            true,
            clock.now() + Duration::from_secs(60),
            s.stamp,
        )?;
    }
    ensure!(
        tick(&mut s, &mut long, Command::Tick, &clock).is_err() && !long.healthy,
        "request state disagreement not fail-stop"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"reference":cold,"foreground_reference":foreground,"completed":completed,"commands":commands,"stable_request_ownership":true,"foreground_prefill_decode_preemption_exact":true,"paused_and_queued_cancellation":true,"queued_timeout":true,"active_paused_timeout":true,"three_background_slots_leave_foreground_capacity":true,"active_time_excludes_pause":true,"invalid_request_no_ticket_leak":true,"stale_cancel_preserves_new_owner":true,"rank_disagreement_before_work":true,"automatic_cache_policy":false,"concurrent_ingress_and_http":false,"production_ready":false}),
    )
}
