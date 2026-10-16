//! Packed exact suffix lookup with one owner per
//! retained slot; integer back-links and fixed-size keys, no per-token objects.
use anyhow::{Result, ensure};
use std::collections::HashMap;
const LIMIT: usize = 1_048_576;
const CANDIDATES: usize = 64;
const EXTEND: usize = 64;
const VIRTUAL_BASE: i32 = 1 << 24;

pub struct SuffixIndex {
    n: usize,
    tokens: Vec<i32>,
    previous: Vec<i32>,
    ends: HashMap<[i32; 8], i32>,
    indexed: usize,
}
impl SuffixIndex {
    pub fn new(n: usize) -> Result<Self> {
        ensure!((1..=8).contains(&n), "lookup n-gram bound");
        Ok(Self {
            n,
            tokens: Vec::new(),
            previous: Vec::new(),
            ends: HashMap::new(),
            indexed: 0,
        })
    }
    pub fn extend(&mut self, tokens: &[i32]) -> Result<()> {
        ensure!(
            tokens.len() <= LIMIT - self.tokens.len(),
            "lookup context bound"
        );
        self.tokens.try_reserve(tokens.len())?;
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }
    fn key(&self, end: usize) -> [i32; 8] {
        let mut key = [0; 8];
        key[..self.n].copy_from_slice(&self.tokens[end - self.n..end]);
        key
    }
    fn index(&mut self) -> Result<()> {
        let first = self.indexed.max(self.n);
        let last = self.tokens.len();
        if first < last {
            self.ends.try_reserve(last - first)?;
            self.previous
                .try_reserve(last.saturating_sub(self.previous.len()))?;
            self.previous.resize(first, -1);
            for end in first..last {
                let key = self.key(end);
                self.previous
                    .push(self.ends.insert(key, end as i32).unwrap_or(-1));
            }
        }
        self.indexed = self.indexed.max(last);
        Ok(())
    }
    pub fn find(&mut self) -> Result<Option<(usize, usize)>> {
        let last = self.tokens.len();
        if last <= self.n {
            return Ok(None);
        }
        self.index()?;
        let mut end = *self.ends.get(&self.key(last)).unwrap_or(&-1);
        let mut best = None;
        for _ in 0..CANDIDATES {
            if end < 0 {
                break;
            }
            let at = end as usize;
            let mut len = self.n;
            while len < EXTEND.min(at) && self.tokens[at - 1 - len] == self.tokens[last - 1 - len] {
                len += 1;
            }
            if best.is_none_or(|(length, _)| len > length) {
                best = Some((len, at));
                if len >= EXTEND {
                    break;
                }
            }
            end = self.previous[at];
        }
        Ok(best)
    }
    pub fn truncate(&mut self, length: usize) {
        if length >= self.tokens.len() {
            return;
        }
        for end in (length.max(self.n)..self.indexed).rev() {
            let key = self.key(end);
            let previous = self.previous[end];
            if previous < 0 {
                self.ends.remove(&key);
            } else {
                self.ends.insert(key, previous);
            }
        }
        self.indexed = self.indexed.min(length);
        self.previous.truncate(length);
        self.tokens.truncate(length);
    }
    /// Legal only when this slot has no paused/in-flight owner.
    pub fn replace_completed(&mut self, prompt: &[i32]) -> Result<()> {
        ensure!(prompt.len() <= LIMIT, "lookup context bound");
        let keep = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        self.truncate(keep);
        self.extend(&prompt[keep..])
    }
    pub fn propose(&mut self, most: usize, least: usize) -> Result<(Vec<i32>, usize)> {
        ensure!(
            most <= 15 && (1..=64).contains(&least),
            "lookup proposal bound"
        );
        let Some((length, end)) = self.find()? else {
            return Ok((Vec::new(), 0));
        };
        if length < least {
            return Ok((Vec::new(), length));
        }
        let mut out = Vec::with_capacity(most);
        for i in 0..most {
            let pos = end + i;
            let token = if pos < self.tokens.len() {
                self.tokens[pos]
            } else {
                out[pos - self.tokens.len()]
            };
            if token >= VIRTUAL_BASE {
                break;
            }
            out.push(token);
        }
        Ok((out, length))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reference(t: &[i32], n: usize) -> Option<(usize, usize)> {
        if t.len() <= n {
            return None;
        }
        let last = t.len();
        let mut best = None;
        for at in (n..last)
            .rev()
            .filter(|at| t[at - n..*at] == t[last - n..])
            .take(64)
        {
            let len = (n..EXTEND.min(at))
                .take_while(|j| t[at - 1 - j] == t[last - 1 - j])
                .count()
                + n;
            if best.is_none_or(|(length, _)| len > length) {
                best = Some((len, at));
            }
        }
        best
    }
    #[test]
    fn branch_trim_extend_matches_fresh_bounded_search() {
        for n in 1..=8 {
            let mut x = SuffixIndex::new(n).unwrap();
            let mut seed = 97u32;
            for step in 0..1000 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                if step % 7 == 0 {
                    x.truncate(seed as usize % (x.tokens.len() + 1));
                }
                let block: Vec<_> = (0..seed % 17 + 1)
                    .map(|i| ((seed >> 8) + i) % 7)
                    .map(|x| x as i32)
                    .collect();
                x.extend(&block).unwrap();
                assert_eq!(x.find().unwrap(), reference(&x.tokens, n));
                let (proposal, _) = x.propose(15, n).unwrap();
                if let Some((_, at)) = reference(&x.tokens, n) {
                    let period = &x.tokens[at..];
                    assert_eq!(
                        proposal,
                        (0..15)
                            .map(|i| period[i % period.len()])
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn overlap_recency_and_virtual_stop() {
        let mut x = SuffixIndex::new(3).unwrap();
        x.extend(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]).unwrap();
        assert_eq!(x.propose(6, 3).unwrap().0, vec![4, 1, 2, 3, 4, 1]);
        x.replace_completed(&[1, 2, 3, VIRTUAL_BASE, 1, 2, 3])
            .unwrap();
        assert_eq!(x.propose(6, 3).unwrap().0, Vec::<i32>::new());
        x.truncate(0);
        assert!(x.ends.is_empty() && x.previous.is_empty());
        x.extend(&[8; 100]).unwrap();
        assert_eq!(x.find().unwrap(), Some((64, 99)));
        assert!(x.propose(16, 4).is_err());
    }
}

/// Acceptance gate for the deployed cost-based lookup arm. The owner retains
/// this index across its own rounds; no global cache or ownership cycle exists.
pub struct Lookup {
    index: SuffixIndex,
    synced: usize,
    accepted: [f64; 4],
    tried: [f64; 4],
    skipped: [usize; 4],
    band: usize,
}
impl Lookup {
    pub fn new(prompt: &[i32]) -> Result<Self> {
        let mut index = SuffixIndex::new(4)?;
        index.replace_completed(prompt)?;
        Ok(Self {
            index,
            synced: 0,
            accepted: [0.; 4],
            tried: [0.; 4],
            skipped: [0; 4],
            band: 0,
        })
    }
    pub fn plan(
        &mut self,
        generated: &[i32],
        room: usize,
        depth: &crate::depth_policy::DepthPolicy,
    ) -> Result<Vec<i32>> {
        self.plan_with_eos(generated, room, depth, true)
    }
    pub fn plan_with_eos(
        &mut self,
        generated: &[i32],
        room: usize,
        depth: &crate::depth_policy::DepthPolicy,
        stop_eos: bool,
    ) -> Result<Vec<i32>> {
        ensure!(
            generated.len() >= self.synced,
            "lookup output moved backwards"
        );
        self.index.extend(&generated[self.synced..])?;
        self.synced = generated.len();
        let (mut drafts, length) = self.index.propose(room.min(15), 4)?;
        // Prompt image rows are not vocabulary tokens and cannot be drafted.
        if let Some(i) = drafts.iter().position(|t| !(0..154880).contains(t)) {
            drafts.truncate(i);
        }
        if stop_eos
            && let Some(i) = drafts
                .iter()
                .position(|t| crate::output_decode::EOS.contains(&(*t as u32)))
        {
            drafts.truncate(i);
        }
        if drafts.is_empty() {
            return Ok(drafts);
        }
        self.band = if length >= 32 {
            3
        } else if length >= 16 {
            2
        } else if length >= 8 {
            1
        } else {
            0
        };
        let b = self.band;
        let prior = [0.45, 0.7, 0.8, 0.9][b];
        let probability = (self.accepted[b] + 2. * prior) / (self.tried[b] + 2.);
        let (k, surplus) = depth.lookup(probability, drafts.len())?;
        if k == 0 || surplus <= 0. {
            self.skipped[b] += 1;
            if k == 0 || b < 2 || self.skipped[b] < 8 {
                return Ok(Vec::new());
            }
        }
        self.skipped[b] = 0;
        drafts.truncate(k);
        Ok(drafts)
    }
    pub fn record(&mut self, rows: usize, keep: usize) -> Result<()> {
        ensure!(
            (2..=16).contains(&rows) && (1..=rows).contains(&keep),
            "lookup accepted rows"
        );
        let kept = keep - 1;
        self.accepted[self.band] = self.accepted[self.band] * 0.85 + kept as f64;
        self.tried[self.band] = self.tried[self.band] * 0.85 + kept as f64 + f64::from(keep < rows);
        Ok(())
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::depth_policy::{Costs, DepthPolicy};
    #[test]
    fn ignored_eos_remains_an_ordinary_lookup_proposal() -> Result<()> {
        let depth = DepthPolicy::new(Costs {
            verify: vec![20., 21., 22., 23., 24., 25., 26., 27.],
            block: 4.,
            taps_row: 0.,
            round: 0.,
        })?;
        for eos in crate::output_decode::EOS {
            let mut prompt: Vec<i32> = (10..90).collect();
            prompt[21] = eos as i32;
            prompt.extend(10..30);
            let mut stopped = Lookup::new(&prompt)?;
            assert!(stopped.plan(&[30], 7, &depth)?.is_empty());
            let mut ignored = Lookup::new(&prompt)?;
            let proposals = ignored.plan_with_eos(&[30], 7, &depth, false)?;
            assert!(!proposals.is_empty());
            assert_eq!(proposals[0], eos as i32);
        }
        Ok(())
    }
    #[test]
    fn exact_suffix_is_cost_gated_and_long_skips_are_probed() {
        let costs = Costs {
            verify: vec![20., 23., 26., 29., 32., 35., 38., 41.],
            block: 4.,
            taps_row: 0.1,
            round: 0.,
        };
        let depth = DepthPolicy::new(costs).unwrap();
        let text: Vec<_> = (10..90).cycle().take(180).collect();
        let mut lookup = Lookup::new(&text).unwrap();
        let drafts = lookup.plan(&[30], 7, &depth).unwrap();
        assert!(!drafts.is_empty());
        assert_eq!(drafts[0], 31);
        lookup.record(drafts.len() + 1, drafts.len() + 1).unwrap();
        assert!(lookup.accepted[3] > 0.);
        // Make all deeper windows unprofitable; the long-match probe still
        // explores after eight skips, without changing any target decision.
        let mut slow = DepthPolicy::new(Costs {
            verify: vec![1., 1000., 2000., 3000.],
            block: 0.,
            taps_row: 0.,
            round: 0.,
        })
        .unwrap();
        // Establish a fast common rate from a one-row round.
        slow.record(true, 1, 0, 1).unwrap();
        for _ in 0..7 {
            assert!(lookup.plan(&[30], 7, &slow).unwrap().is_empty());
        }
        assert!(!lookup.plan(&[30], 7, &slow).unwrap().is_empty());
        assert!(lookup.plan(&[], 7, &depth).is_err());
    }
}
