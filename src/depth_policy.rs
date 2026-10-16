//! Deterministic speculative cost policy, with no clocks or rank-local decisions.
//! Every request must receive the same immutable measured cost table.
use anyhow::{Result, ensure};
use std::collections::VecDeque;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Costs {
    /// Entry j is a verifier window containing the pending token and j drafts.
    pub verify: Vec<f64>,
    pub block: f64,
    pub taps_row: f64,
    pub round: f64,
}
impl Costs {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=16).contains(&self.verify.len()),
            "verifier cost table bounds"
        );
        ensure!(
            self.verify.iter().all(|x| x.is_finite() && *x > 0.)
                && [self.block, self.taps_row, self.round]
                    .iter()
                    .all(|x| x.is_finite() && *x >= 0.),
            "invalid measured costs"
        );
        Ok(())
    }
}

pub fn best_depth(qs: &[f64], costs: &Costs, rate: f64, fixed: f64, row: f64) -> (usize, f64) {
    let mut product = 1.;
    let mut expected = 1.;
    let mut best = (0, f64::NEG_INFINITY);
    for (j, &q) in qs.iter().take(costs.verify.len() - 1).enumerate() {
        product *= q;
        expected += product;
        let score = expected - rate * (fixed + costs.verify[j + 1] + row * expected);
        if score > best.1 + 1e-12 {
            best = (j + 1, score);
        }
    }
    best
}

pub struct DepthPolicy {
    costs: Costs,
    kept: [f64; 15],
    prob: [f64; 15],
    rounds: VecDeque<(usize, f64)>,
    used: Vec<f64>,
}
impl DepthPolicy {
    pub fn new(costs: Costs) -> Result<Self> {
        costs.validate()?;
        Ok(Self {
            costs,
            kept: [0.; 15],
            prob: [0.; 15],
            rounds: VecDeque::with_capacity(16),
            used: Vec::with_capacity(7),
        })
    }
    pub fn rate(&self) -> f64 {
        if self.rounds.is_empty() {
            2. / (self.costs.verify[1] + self.costs.block + self.costs.round)
        } else {
            self.rounds.iter().map(|x| x.0 as f64).sum::<f64>()
                / self.rounds.iter().map(|x| x.1).sum::<f64>()
        }
    }
    pub fn dflash(&mut self, probabilities: &[f64]) -> Result<usize> {
        ensure!(
            !probabilities.is_empty()
                && probabilities.len() <= 7
                && probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0. ..=1.).contains(p)),
            "draft confidence bounds"
        );
        let qs: Vec<_> = probabilities
            .iter()
            .enumerate()
            .map(|(j, p)| {
                let factor = ((self.kept[j] + 2.) / (self.prob[j] + 2.)).clamp(0.2, 3.);
                (p * factor).min(0.995)
            })
            .collect();
        let k = best_depth(&qs, &self.costs, self.rate(), 0., 0.).0.max(1);
        self.used.clear();
        self.used.extend_from_slice(&probabilities[..k]);
        Ok(k)
    }
    pub fn lookup(&self, probability: f64, most: usize) -> Result<(usize, f64)> {
        ensure!(
            probability.is_finite() && (0. ..=1.).contains(&probability) && most <= 15,
            "lookup confidence bounds"
        );
        Ok(best_depth(
            &vec![probability.min(0.995); most],
            &self.costs,
            self.rate(),
            self.costs.round,
            self.costs.taps_row,
        ))
    }
    /// Record the actual fixed-width proposal used by a shared target batch.
    /// Its row count is chosen by the batch scheduler, rather than dflash().
    pub fn fixed_window(&mut self, probabilities: &[f64]) -> Result<()> {
        ensure!(
            !probabilities.is_empty()
                && probabilities.len() <= 7
                && probabilities.len() < self.costs.verify.len()
                && probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0. ..=1.).contains(p)),
            "fixed draft confidence bounds"
        );
        self.used.clear();
        self.used.extend_from_slice(probabilities);
        Ok(())
    }
    /// rows includes pending; keep includes the pending row, not the sampled
    /// output left pending for the next round. No evidence after first rejection.
    pub fn record(&mut self, lookup: bool, rows: usize, backlog: usize, keep: usize) -> Result<()> {
        ensure!(
            rows >= 1 && rows <= self.costs.verify.len() && keep >= 1 && keep <= rows,
            "depth outcome bounds"
        );
        if !lookup {
            ensure!(
                self.used.len() + 1 == rows,
                "depth outcome differs from proposal"
            );
            for (j, &p) in self.used.iter().take(keep).enumerate() {
                self.kept[j] = self.kept[j] * 0.97 + f64::from(j + 1 < keep);
                self.prob[j] = self.prob[j] * 0.97 + p;
            }
        }
        self.used.clear();
        let cost = self.costs.verify[rows - 1]
            + self.costs.round
            + if lookup {
                0.
            } else {
                self.costs.block + self.costs.taps_row * backlog as f64
            };
        self.rounds.push_back((keep, cost));
        if self.rounds.len() > 16 {
            self.rounds.pop_front();
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn costs() -> Costs {
        Costs {
            verify: vec![20., 23., 26., 29., 32., 35., 38., 41.],
            block: 4.,
            taps_row: 0.1,
            round: 0.5,
        }
    }
    #[test]
    fn confidence_and_costs_choose_different_depths() {
        let mut x = DepthPolicy::new(costs()).unwrap();
        assert_eq!(x.dflash(&[0.01; 7]).unwrap(), 1);
        assert_eq!(x.dflash(&[0.99; 7]).unwrap(), 7);
        let (k, s) = x.lookup(0.9, 7).unwrap();
        assert!(k > 1 && s > 0.);
        let mut slow = costs();
        slow.verify[2..].fill(1000.);
        assert_eq!(
            DepthPolicy::new(slow).unwrap().dflash(&[0.99; 7]).unwrap(),
            1
        );
    }
    #[test]
    fn ties_shorter_rejection_reach_and_bounded_rate_history() {
        let c = Costs {
            verify: vec![1.; 8],
            block: 1.,
            taps_row: 0.,
            round: 0.,
        };
        assert_eq!(best_depth(&[0.; 7], &c, 1., 0., 0.).0, 1);
        let mut x = DepthPolicy::new(c).unwrap();
        for _ in 0..64 {
            let k = x.dflash(&[0.99; 7]).unwrap();
            x.record(false, k + 1, 3, 1).unwrap();
        }
        assert_eq!(x.rounds.len(), 16);
        assert_eq!(x.kept, [0.; 15]);
        assert!(x.prob[0] > 0. && x.prob[1..].iter().all(|p| *p == 0.));
        assert_eq!(x.rate(), 0.5);
        assert!(x.record(false, 2, 0, 2).is_err());
        assert!(x.dflash(&[f64::NAN]).is_err());
        assert!(x.dflash(&[]).is_err());
    }
    #[test]
    fn full_acceptance_updates_every_reached_position() {
        let mut x = DepthPolicy::new(costs()).unwrap();
        let k = x.dflash(&[0.99; 7]).unwrap();
        x.record(false, k + 1, k + 1, k + 1).unwrap();
        assert!(x.kept[..7].iter().all(|p| *p == 1.));
        assert!(x.prob[..7].iter().all(|p| *p == 0.99));
        let rate = x.rate();
        x.record(true, 8, 0, 8).unwrap();
        assert!(x.rate() > rate);
    }
    #[test]
    fn fixed_batch_window_records_the_executed_depth_without_replanning() {
        let mut x = DepthPolicy::new(costs()).unwrap();
        x.fixed_window(&[0.5; 7]).unwrap();
        x.record(false, 8, 3, 4).unwrap();
        assert_eq!(&x.kept[..4], &[1., 1., 1., 0.]);
        assert_eq!(&x.prob[..4], &[0.5; 4]);
        x.fixed_window(&[0.9; 3]).unwrap();
        assert!(x.record(false, 8, 0, 4).is_err());
        x.record(false, 4, 0, 4).unwrap();
        assert!(x.fixed_window(&[]).is_err());
        assert!(x.fixed_window(&[f64::NAN]).is_err());
        assert!(x.fixed_window(&[1.1]).is_err());
    }
}
