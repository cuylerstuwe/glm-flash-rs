//! Bounded host oracle for the canonical selector policy. This proposes only;
//! the exact target sampler decides every emitted token. GPU residency follows.
use crate::sampling::{Candidate, Sampling, uniform};
use anyhow::{Result, ensure};
pub(crate) struct Proposal {
    pub tokens: Vec<i32>,
    pub probabilities: Vec<f64>,
}
#[cfg(test)]
pub(crate) fn choose(
    rows: &[Vec<Candidate>],
    projected: &[f32],
    anchor: i32,
    first: usize,
    policy: Sampling,
    code: impl Fn(bool, u32, usize) -> f64,
) -> Result<Vec<i32>> {
    Ok(choose_scored(rows, projected, anchor, first, policy, code)?.tokens)
}
pub(crate) fn choose_scored(
    rows: &[Vec<Candidate>],
    projected: &[f32],
    anchor: i32,
    first: usize,
    policy: Sampling,
    code: impl Fn(bool, u32, usize) -> f64,
) -> Result<Proposal> {
    ensure!(
        (1..=7).contains(&rows.len())
            && projected.len() == rows.len() * 256
            && (0..154880).contains(&anchor)
            && projected.iter().all(|v| v.is_finite()),
        "drafter chain inputs"
    );
    let mut previous = anchor as u32;
    let mut result = Vec::with_capacity(rows.len());
    let mut probabilities = Vec::with_capacity(rows.len());
    let temperature = if policy.greedy() {
        1.
    } else {
        policy.temperature
    };
    for (row, candidates) in rows.iter().enumerate() {
        ensure!(
            candidates.len() == 16
                && candidates
                    .iter()
                    .all(|c| c.id < 154880 && c.value.is_finite()),
            "drafter candidates"
        );
        let edge_input: [f64; 256] =
            std::array::from_fn(|k| code(true, previous, k) * projected[row * 256 + k] as f64);
        let mut confidence = [0.; 16];
        let mut selected = 0;
        let mut best = f64::NEG_INFINITY;
        let mut token = None;
        for (j, candidate) in candidates.iter().enumerate() {
            let mut edge = 0.;
            for (k, value) in edge_input.iter().enumerate() {
                edge += code(false, candidate.id, k) * value;
            }
            let mut score = (candidate.value as f64 + 0.6 * edge) / temperature;
            confidence[j] = score;
            if !policy.greedy() {
                let u = uniform(policy.seed, (first + row) as u64, candidate.id);
                let noise = -(-u.ln()).ln();
                // Confidence describes this selection: same noise weight as below.
                confidence[j] = (score + 0.7 * noise) / 1.5;
                score += 0.7 * noise;
            }
            ensure!(!score.is_nan(), "nonfinite drafter selector");
            if token.is_none() || score > best {
                best = score;
                token = Some(candidate.id);
                selected = j;
            }
        }
        let max = confidence.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = confidence.map(|s| (s - max).exp());
        probabilities.push(weights[selected] / weights.iter().sum::<f64>());
        previous = token.unwrap();
        result.push(previous as i32);
    }
    Ok(Proposal {
        tokens: result,
        probabilities,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_edges_follow_the_previous_pick() {
        let candidates = (0..16)
            .map(|id| Candidate { id, value: 0. })
            .collect::<Vec<_>>();
        let rows = vec![candidates; 3];
        let p = Sampling::new(31, 0., 20, 0.95).unwrap();
        let uniform = choose_scored(&rows, &vec![0.; 768], 2, 100, p, |_, _, _| 0.).unwrap();
        assert_eq!(uniform.tokens, vec![0; 3]);
        assert_eq!(uniform.probabilities, vec![1. / 16.; 3]);
        let out = choose(&rows, &vec![1.; 768], 2, 100, p, |pred, id, k| {
            if pred {
                f64::from(k == id as usize)
            } else {
                f64::from(k + 1 == id as usize)
            }
        })
        .unwrap();
        assert_eq!(out, vec![3, 4, 5]);
        let sampled = Sampling::new(731, 1., 20, 0.95).unwrap();
        let first = choose(&rows, &vec![0.; 768], 2, 100, sampled, |_, _, _| 0.).unwrap();
        assert_eq!(
            first,
            choose(&rows, &vec![0.; 768], 2, 100, sampled, |_, _, _| 0.).unwrap()
        );
        assert!(choose(&[], &[], 2, 100, p, |_, _, _| 0.).is_err());
    }
}
