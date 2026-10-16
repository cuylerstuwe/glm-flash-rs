//! Optional host-side phase timing of scheduler steps (ALASKA_ROUND_TIMING=1).
//! Diagnostics only: marks read a monotonic clock and never change scheduling.
use std::{
    cell::RefCell,
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Receive,
    Control,
    Disk,
    Admit,
    Dispatch,
    Plan,
    Draft,
    DraftRead,
    VerifyDispatch,
    VerifyLaunch,
    VerifyWait,
    VerifyPost,
    Complete,
    Timing,
    Progress,
    Idle,
}
const NAMES: [&str; 16] = [
    "receive",
    "control",
    "disk",
    "admit",
    "dispatch",
    "plan",
    "draft",
    "draft_read",
    "verify_dispatch",
    "verify_launch",
    "verify_wait",
    "verify_post",
    "complete",
    "timing",
    "progress",
    "idle",
];
struct Timer {
    on: bool,
    last: Option<Instant>,
    phase: usize,
    totals: [Duration; 16],
    rounds: u64,
    /// Rank exchanges by the tag word of their record: count and wall time.
    exchanges: BTreeMap<u32, (u64, Duration)>,
}
thread_local! {
    static TIMER: RefCell<Timer> = RefCell::new(Timer {
        on: std::env::var("ALASKA_ROUND_TIMING").as_deref() == Ok("1"),
        last: None,
        phase: Phase::Idle as usize,
        totals: [Duration::ZERO; 16],
        rounds: 0,
        exchanges: BTreeMap::new(),
    });
}
/// Charges one rank exchange (an agreement's tag, or a raw record's tag word) to `tag`.
pub(crate) fn exchange(tag: u32, elapsed: Duration) {
    TIMER.with(|t| {
        let mut t = t.borrow_mut();
        if t.on {
            let e = t.exchanges.entry(tag).or_default();
            e.0 += 1;
            e.1 += elapsed;
        }
    });
}
/// Charges the time since the previous mark to the running phase, then starts `next`.
pub(crate) fn enter(next: Phase) {
    TIMER.with(|t| {
        let mut t = t.borrow_mut();
        if !t.on {
            return;
        }
        let now = Instant::now();
        if let Some(last) = t.last {
            let p = t.phase;
            t.totals[p] += now - last;
        }
        t.last = Some(now);
        t.phase = next as usize;
    });
}
/// Ends a speculative decode round; prints a summary every 4,096 rounds.
pub(crate) fn round() {
    enter(Phase::Idle);
    let summary = TIMER.with(|t| {
        let mut t = t.borrow_mut();
        if !t.on {
            return None;
        }
        t.rounds += 1;
        t.rounds.is_multiple_of(4096).then(|| summary(&t))
    });
    if let Some(s) = summary {
        eprintln!("round_timing={s}");
    }
}
pub(crate) fn report() {
    let summary = TIMER.with(|t| {
        let t = t.borrow();
        (t.on && t.rounds > 0).then(|| summary(&t))
    });
    if let Some(s) = summary {
        eprintln!("round_timing={s}");
    }
}
fn summary(t: &Timer) -> String {
    let per = |d: Duration| d.as_secs_f64() * 1e6 / t.rounds as f64;
    let mut fields = vec![format!("\"rounds\":{}", t.rounds)];
    fields.extend(
        NAMES
            .iter()
            .zip(t.totals)
            .map(|(n, d)| format!("\"{n}_us\":{:.1}", per(d))),
    );
    let exchanges = t
        .exchanges
        .iter()
        .map(|(tag, (n, d))| {
            format!(
                "\"{tag}\":[{:.3},{:.1}]",
                *n as f64 / t.rounds as f64,
                per(*d)
            )
        })
        .collect::<Vec<_>>();
    fields.push(format!(
        "\"exchanges_per_round_and_us\":{{{}}}",
        exchanges.join(",")
    ));
    format!("{{{}}}", fields.join(","))
}
