//! Real-continuation calibration, independent of the timed workload trials.
//! Reconcile each fastest warmed span cost using the slowest of all four ranks.
use super::{Candidates, Decision, Long, Sampling, Verifier};
use crate::{depth_policy::Costs, model::drafter::Drafter};
use anyhow::{Result, ensure};
use std::time::Instant;

pub(super) fn calibrate(
    long: &mut Long<'_>,
    verifier: &mut Verifier,
    draft: &mut Drafter,
    sampler: &mut Candidates,
    prefix: &[i32],
    width: usize,
) -> Result<Costs> {
    ensure!(matches!(width, 8 | 16), "calibration window bound");
    let policy = Sampling::new(731, 1., 20, 0.95)?;
    long.start(prefix.len(), 64)?;
    for (i, chunk) in prefix.chunks(8192).enumerate() {
        long.prefill(i * 8192, chunk)?;
    }
    let mut continuation = vec![long.sample(sampler, policy)?.0 as i32];
    for row in 0..3 * width - 1 {
        long.step(prefix.len() + row, continuation[row])?;
        continuation.push(long.sample(sampler, policy)?.0 as i32);
    }
    let mut local = Vec::new();
    for span in 0..3 {
        let mut history = prefix.to_vec();
        history.extend_from_slice(&continuation[..span * width]);
        long.start(history.len() + width, 1)?;
        for (i, chunk) in history.chunks(8192).enumerate() {
            long.prefill(i * 8192, chunk)?;
        }
        let mut best = vec![f64::INFINITY; width];
        for pass in 0..9 {
            for n in 1..=width {
                let start = Instant::now();
                verifier.execute(
                    long,
                    &continuation[span * width..span * width + n],
                    Decision::Forced(0),
                )?;
                if pass >= 2 {
                    best[n - 1] = best[n - 1].min(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
        local.extend(best);
    }
    // The last prefix's taps are committed; measure stock DFlash and its native
    // tap catch-up separately. Resets and graph construction are outside minima.
    let end = long.sessions[long.active].tokens.len();
    draft.bind(&long.engine, long.active)?;
    draft.start_context(&long.engine, end.saturating_sub(2047))?;
    draft.append_history(&long.engine, long.engine.taps.as_ref().unwrap(), end)?;
    let mut block = f64::INFINITY;
    for pass in 0..9 {
        let start = Instant::now();
        draft.propose(&long.engine, continuation[2 * width], policy)?;
        if pass >= 2 {
            block = block.min(start.elapsed().as_secs_f64() * 1000.);
        }
    }
    let mut taps = f64::INFINITY;
    for pass in 0..9 {
        draft.start_context(&long.engine, end - 8)?;
        long.engine.sync()?;
        let start = Instant::now();
        draft.append_history(&long.engine, long.engine.taps.as_ref().unwrap(), end)?;
        long.engine.sync()?;
        if pass >= 2 {
            taps = taps.min(start.elapsed().as_secs_f64() * 1000. / 8.);
        }
    }
    local.extend([block, taps]);
    let bytes: Vec<_> = local.iter().flat_map(|x| x.to_le_bytes()).collect();
    let e = &long.engine;
    let ranks = e.comm.exchange_host(&e.device, &e.stream, &bytes)?;
    let mut agreed = vec![0f64; local.len()];
    for bytes in ranks {
        ensure!(bytes.len() == local.len() * 8, "rank calibration length");
        for (i, b) in bytes.as_chunks::<8>().0.iter().enumerate() {
            let t = f64::from_le_bytes(*b);
            ensure!(t.is_finite() && t > 0., "invalid rank calibration time");
            agreed[i] = agreed[i].max(t);
        }
    }
    let average: Vec<_> = (0..width)
        .map(|r| (0..3).map(|s| agreed[s * width + r]).sum::<f64>() / 3.)
        .collect();
    let mut slopes = Vec::new();
    for i in 1..width {
        for j in i + 1..width {
            slopes.push((average[j] - average[i]) / (j - i) as f64);
        }
    }
    let median = |mut xs: Vec<f64>| {
        xs.sort_by(f64::total_cmp);
        let n = xs.len();
        (xs[(n - 1) / 2] + xs[n / 2]) * 0.5
    };
    let slope = median(slopes);
    let base = median(
        (1..width)
            .map(|i| average[i] - slope * (i + 1) as f64)
            .collect(),
    );
    let mut verify = vec![average[0]];
    verify.extend((2..=width).map(|r| base + slope * r as f64));
    // Use common integer microseconds in the shared request header.
    for x in &mut verify {
        *x = (*x * 1000.).round().max(1.) / 1000.;
    }
    let costs = Costs {
        verify,
        block: agreed[3 * width],
        taps_row: agreed[3 * width + 1],
        round: 0.,
    };
    costs.validate()?;
    eprintln!("common real-continuation calibration {:?}", costs);
    Ok(costs)
}
