//! Checked rank-major verifier payloads. Arithmetic and RNG stay in sampling.rs.
use crate::sampling::{self, Candidate, Sampling};
use anyhow::{Result, ensure};
use std::ops::Range;
pub(crate) const ROWS: usize = 16;
pub(crate) const LOCAL: usize = sampling::VOCAB / 4;
pub(crate) const MAX_LOCAL_BYTES: usize = ROWS * (LOCAL + 1) * 8;
pub(crate) fn enabled(value: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(value, None | Some("0") | Some("1")),
        "ALASKA_VERIFY_PACKED_SAMPLE must be 0/1"
    );
    Ok(value == Some("1"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub rows: usize,
    pub k: usize,
    pub row_bytes: usize,
    pub rank_bytes: usize,
    pub gathered_bytes: usize,
}
impl Layout {
    pub fn new(rows: usize, k: usize) -> Result<Self> {
        ensure!(
            (1..=ROWS).contains(&rows) && (1..=LOCAL).contains(&k),
            "packed candidate shape"
        );
        let row_bytes = (k + 1) * 8;
        let rank_bytes = rows * row_bytes;
        Ok(Self {
            rows,
            k,
            row_bytes,
            rank_bytes,
            gathered_bytes: 4 * rank_bytes,
        })
    }
    pub fn for_policy(rows: usize, policy: Sampling, first: u64) -> Result<Self> {
        ensure!(
            !policy.greedy(),
            "packed stochastic candidates require positive temperature"
        );
        let result = Self::new(rows, policy.local_count())?;
        ensure!(
            first.checked_add(rows as u64 - 1).is_some(),
            "packed candidate position overflow"
        );
        Ok(result)
    }
    pub fn range(self, rank: usize, row: usize) -> Result<Range<usize>> {
        ensure!(rank < 4 && row < self.rows, "packed candidate rank/row");
        let start = rank * self.rank_bytes + row * self.row_bytes;
        Ok(start..start + self.row_bytes)
    }
    pub fn logits(self, ptr: u64, bytes: usize, rank: usize) -> Result<u64> {
        let needed = self.rows * LOCAL * 2;
        ensure!(
            rank < 4 && needed <= bytes && ptr.checked_add(needed as u64).is_some(),
            "packed logits bounds"
        );
        Ok(ptr)
    }
    pub fn decode_row(self, bytes: &[u8], row: usize) -> Result<Vec<Candidate>> {
        ensure!(
            bytes.len() == self.gathered_bytes,
            "packed candidate payload length"
        );
        let mut candidates = Vec::with_capacity(4 * self.k);
        for rank in 0..4 {
            let data = &bytes[self.range(rank, row)?];
            ensure!(
                word(data, self.k * 8) == 0 && word(data, self.k * 8 + 4) as usize == self.k,
                "packed candidate shard failure"
            );
            let mut previous = None;
            for j in 0..self.k {
                let bits = word(data, j * 8);
                let value = f32::from_bits(bits);
                let id = word(data, j * 8 + 4);
                ensure!(
                    value.is_finite()
                        && bits & 0xffff == 0
                        && (rank * LOCAL..(rank + 1) * LOCAL).contains(&(id as usize))
                        && previous.is_none_or(|p| id > p),
                    "invalid packed candidate payload"
                );
                previous = Some(id);
                candidates.push(Candidate { id, value });
            }
        }
        Ok(candidates)
    }
    /// Publish no partial row outputs on any malformed gathered shard.
    pub fn choose(
        self,
        bytes: &[u8],
        policy: Sampling,
        first: u64,
    ) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            self == Self::for_policy(self.rows, policy, first)?,
            "packed host policy shape"
        );
        let mut out = [(0, 0); ROWS];
        for (row, result) in out.iter_mut().take(self.rows).enumerate() {
            *result = sampling::choose(&self.decode_row(bytes, row)?, first + row as u64, policy)?;
        }
        Ok(out)
    }
    /// Rows of the unchanged sampling_choose ABI, in a single upload: all
    /// 32-byte integer records followed by all 16-byte floating-point records.
    pub fn encode_policy(self, policy: Sampling, first: u64, out: &mut [u8]) -> Result<()> {
        ensure!(
            self == Self::for_policy(self.rows, policy, first)?
                && policy.gpu_fits(4 * self.k)
                && out.len() == self.rows * 48,
            "packed GPU policy shape"
        );
        for row in 0..self.rows {
            for (i, value) in [
                4 * self.k as u64,
                policy.kept(4 * self.k) as u64,
                policy.seed,
                first + row as u64,
            ]
            .into_iter()
            .enumerate()
            {
                out[row * 32 + i * 8..row * 32 + (i + 1) * 8].copy_from_slice(&value.to_le_bytes());
            }
            for (i, bits) in [policy.temperature.to_bits(), policy.top_p.to_bits()]
                .into_iter()
                .enumerate()
            {
                let off = self.rows * 32 + row * 16 + i * 8;
                out[off..off + 8].copy_from_slice(&bits.to_le_bytes());
            }
        }
        Ok(())
    }
    /// Independent request seeds and absolute positions, sharing only the
    /// candidate payload width. No request uses another request's RNG cursor.
    pub fn encode_policies(self, policies: &[(Sampling, u64)], out: &mut [u8]) -> Result<()> {
        ensure!(
            policies.len() == self.rows
                && out.len() == self.rows * 48
                && policies.iter().all(|&(p, at)| {
                    Self::for_policy(1, p, at).is_ok()
                        && p.local_count() == self.k
                        && p.gpu_fits(4 * self.k)
                }),
            "independent packed GPU policy shape"
        );
        for (row, &(policy, position)) in policies.iter().enumerate() {
            for (i, value) in [
                4 * self.k as u64,
                policy.kept(4 * self.k) as u64,
                policy.seed,
                position,
            ]
            .into_iter()
            .enumerate()
            {
                out[row * 32 + i * 8..row * 32 + (i + 1) * 8].copy_from_slice(&value.to_le_bytes());
            }
            for (i, bits) in [policy.temperature.to_bits(), policy.top_p.to_bits()]
                .into_iter()
                .enumerate()
            {
                let off = self.rows * 32 + row * 16 + i * 8;
                out[off..off + 8].copy_from_slice(&bits.to_le_bytes());
            }
        }
        Ok(())
    }
    pub fn results_policies(
        self,
        bytes: &[u8],
        policies: &[(Sampling, u64)],
    ) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            bytes.len() == self.rows * 12
                && policies.len() == self.rows
                && policies.iter().all(|&(p, _)| {
                    p.local_count() == self.k && !p.greedy() && p.gpu_fits(4 * self.k)
                }),
            "independent packed GPU result shape"
        );
        ensure!(
            bytes[self.rows * 8..].iter().all(|b| *b == 0),
            "invalid independent packed gathered candidates"
        );
        let mut out = [(0, 0); ROWS];
        for (row, &(policy, _)) in policies.iter().enumerate() {
            let id = word(bytes, row * 8);
            let kept = word(bytes, row * 8 + 4) as usize;
            ensure!(
                (id as usize) < sampling::VOCAB && kept > 0 && kept <= policy.kept(4 * self.k),
                "invalid independent packed sampled result"
            );
            out[row] = (id, kept);
        }
        Ok(out)
    }
    pub fn choose_policies(
        self,
        bytes: &[u8],
        policies: &[(Sampling, u64)],
    ) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            policies.len() == self.rows
                && policies.iter().all(|&(p, at)| {
                    Self::for_policy(1, p, at).is_ok() && p.local_count() == self.k
                }),
            "independent packed host policy shape"
        );
        let mut out = [(0, 0); ROWS];
        for (row, &(policy, position)) in policies.iter().enumerate() {
            out[row] = sampling::choose(&self.decode_row(bytes, row)?, position, policy)?;
        }
        Ok(out)
    }
    pub fn results(self, bytes: &[u8], policy: Sampling) -> Result<[(u32, usize); ROWS]> {
        ensure!(
            bytes.len() == self.rows * 12
                && policy.local_count() == self.k
                && !policy.greedy()
                && policy.gpu_fits(4 * self.k),
            "packed GPU result shape"
        );
        ensure!(
            bytes[self.rows * 8..].iter().all(|b| *b == 0),
            "invalid packed gathered candidates"
        );
        let mut out = [(0, 0); ROWS];
        for (row, result) in out.iter_mut().take(self.rows).enumerate() {
            let id = word(bytes, row * 8);
            let kept = word(bytes, row * 8 + 4) as usize;
            ensure!(
                (id as usize) < sampling::VOCAB && kept > 0 && kept <= policy.kept(4 * self.k),
                "invalid packed sampled result"
            );
            *result = (id, kept);
        }
        Ok(out)
    }
}
fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_policy_rows_preserve_uniform_abi_and_rng_ownership() -> Result<()> {
        let policy = Sampling::new(731, 1., 20, 0.95)?;
        let layout = Layout::for_policy(4, policy, 101)?;
        let uniform: Vec<_> = (0..4).map(|row| (policy, 101 + row)).collect();
        let mut old = vec![0; 4 * 48];
        let mut new = old.clone();
        layout.encode_policy(policy, 101, &mut old)?;
        layout.encode_policies(&uniform, &mut new)?;
        assert_eq!(old, new);
        let policies = [
            (Sampling::new(31, 1., 20, 0.95)?, 400),
            (Sampling::new(73, 0.8, 20, 0.8)?, 16001),
            (Sampling::new(59, 1.2, 20, 1.)?, 262145),
            (Sampling::new(91, 1., 20, 0.9)?, 1048000),
        ];
        layout.encode_policies(&policies, &mut new)?;
        for (row, &(p, at)) in policies.iter().enumerate() {
            assert_eq!(
                u64::from_le_bytes(new[row * 32 + 16..row * 32 + 24].try_into()?),
                p.seed
            );
            assert_eq!(
                u64::from_le_bytes(new[row * 32 + 24..row * 32 + 32].try_into()?),
                at
            );
        }
        let (bytes, rows) = fixture(layout);
        let picked = layout.choose_policies(&bytes, &policies)?;
        for (row, &(p, at)) in policies.iter().enumerate() {
            assert_eq!(picked[row], sampling::choose(&rows[row], at, p)?);
        }
        let mut invalid = policies;
        invalid[2].0 = Sampling::new(59, 0., 20, 1.)?;
        let untouched = new.clone();
        assert!(layout.encode_policies(&invalid, &mut new).is_err());
        assert_eq!(new, untouched);
        Ok(())
    }
    fn fixture(l: Layout) -> (Vec<u8>, Vec<Vec<Candidate>>) {
        // Fill the complete maximum allocation with poison, then populate only
        // the compact logical prefix. Wrong max-stride reads hit poison.
        let mut bytes = vec![0xa5; MAX_LOCAL_BYTES * 4];
        let mut rows = vec![Vec::new(); l.rows];
        for rank in 0..4 {
            for (row, values) in rows.iter_mut().enumerate() {
                let at = l.range(rank, row).unwrap().start;
                for j in 0..l.k {
                    let id = (rank * LOCAL + j) as u32;
                    let value = if j % 5 == 0 {
                        -0.0
                    } else {
                        ((row + rank + j) % 11) as f32
                    };
                    bytes[at + j * 8..at + j * 8 + 4].copy_from_slice(&value.to_le_bytes());
                    bytes[at + j * 8 + 4..at + j * 8 + 8].copy_from_slice(&id.to_le_bytes());
                    values.push(Candidate { id, value });
                }
                bytes[at + l.k * 8..at + l.k * 8 + 4].copy_from_slice(&0u32.to_le_bytes());
                bytes[at + l.k * 8 + 4..at + l.row_bytes]
                    .copy_from_slice(&(l.k as u32).to_le_bytes());
            }
        }
        bytes.truncate(l.gathered_bytes);
        (bytes, rows)
    }
    #[test]
    fn logical_strides_and_all_rows_match_independent_scalar_sampling() -> Result<()> {
        for rows in [1, 2, 7, 16] {
            for top in [0, 1, 20, 56, 128, 504] {
                let s = Sampling::new(u64::MAX, 0.7, top, 0.95)?;
                let l = Layout::for_policy(rows, s, 500001)?;
                let (bytes, expected) = fixture(l);
                let actual = l.choose(&bytes, s, 500001)?;
                for (row, expected) in expected.iter().enumerate() {
                    let decoded = l.decode_row(&bytes, row)?;
                    assert_eq!(
                        decoded
                            .iter()
                            .map(|x| (x.id, x.value.to_bits()))
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|x| (x.id, x.value.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        actual[row],
                        sampling::choose(expected, 500001 + row as u64, s)?
                    );
                }
                assert!(l.rank_bytes <= MAX_LOCAL_BYTES);
                assert_eq!(l.range(3, rows - 1)?.end, l.gathered_bytes);
                assert!(actual[rows..].iter().all(|v| *v == (0, 0)));
            }
        }
        Ok(())
    }
    #[test]
    fn every_rank_and_tail_row_rejects_nonfinite_bad_ids_status_and_length() -> Result<()> {
        let l = Layout::new(16, 28)?;
        let (good, _) = fixture(l);
        for rank in 0..4 {
            for row in [0, 7, 15] {
                let base = l.range(rank, row)?.start;
                for (off, word) in [
                    (0, 0x7f800000u32),
                    (0, 0xff800000),
                    (0, 0x7fc10000),
                    (0, 1),
                    (4, ((rank + 1) % 4 * LOCAL) as u32),
                    (12, (rank * LOCAL) as u32),
                    (28 * 8, 1),
                    (28 * 8 + 4, 27),
                ] {
                    let mut bad = good.clone();
                    bad[base + off..base + off + 4].copy_from_slice(&word.to_le_bytes());
                    assert!(l.decode_row(&bad, row).is_err());
                    assert!(
                        l.choose(&bad, Sampling::new(450, 1., 20, 0.95)?, 900)
                            .is_err()
                    );
                }
            }
        }
        assert!(l.decode_row(&good[..good.len() - 1], 0).is_err());
        assert!(l.decode_row(&good, 16).is_err());
        Ok(())
    }
    #[test]
    fn positions_policy_bits_and_results_share_one_bounded_buffer() -> Result<()> {
        let s = Sampling::new(u64::MAX, 1e-12, 20, -0.)?;
        let l = Layout::for_policy(16, s, u64::MAX - 15)?;
        let mut upload = [0xa5; ROWS * 48];
        l.encode_policy(s, u64::MAX - 15, &mut upload)?;
        let read = |off| u64::from_le_bytes(upload[off..off + 8].try_into().unwrap());
        for row in 0..16 {
            assert_eq!(
                [
                    read(row * 32),
                    read(row * 32 + 8),
                    read(row * 32 + 16),
                    read(row * 32 + 24)
                ],
                [112, 20, u64::MAX, u64::MAX - 15 + row as u64]
            );
            assert_eq!(read(16 * 32 + row * 16), s.temperature.to_bits());
            assert_eq!(read(16 * 32 + row * 16 + 8), (-0f64).to_bits());
        }
        let mut bytes = vec![0; 16 * 12];
        for row in 0..16 {
            bytes[row * 8..row * 8 + 4].copy_from_slice(&(row as u32).to_le_bytes());
            bytes[row * 8 + 4..row * 8 + 8].copy_from_slice(&20u32.to_le_bytes());
        }
        assert_eq!(l.results(&bytes, s)?[15], (15, 20));
        for off in [16 * 8, 16 * 12 - 1] {
            let mut bad = bytes.clone();
            bad[off] = 1;
            assert!(l.results(&bad, s).is_err());
        }
        for (off, val) in [(0, 154880u32), (15 * 8 + 4, 0), (7 * 8 + 4, 21)] {
            let mut bad = bytes.clone();
            bad[off..off + 4].copy_from_slice(&val.to_le_bytes());
            assert!(l.results(&bad, s).is_err());
        }
        assert!(Layout::for_policy(16, s, u64::MAX - 14).is_err());
        assert!(l.encode_policy(s, 0, &mut upload[..47]).is_err());
        assert!(l.results(&bytes[..191], s).is_err());
        Ok(())
    }
    #[test]
    fn shapes_fallback_and_pointer_bounds_fail_before_launch() -> Result<()> {
        assert!(!enabled(None)? && !enabled(Some("0"))? && enabled(Some("1"))?);
        for value in ["", "true", "false", "2"] {
            assert!(enabled(Some(value)).is_err());
        }
        for (r, k) in [(0, 28), (17, 28), (1, 0), (1, LOCAL + 1)] {
            assert!(Layout::new(r, k).is_err());
        }
        let l = Layout::new(16, LOCAL)?;
        assert_eq!(l.rank_bytes, MAX_LOCAL_BYTES);
        assert!(l.logits(u64::MAX, 16 * LOCAL * 2, 0).is_err());
        assert!(l.logits(100, 16 * LOCAL * 2 - 1, 0).is_err());
        assert!(l.logits(100, 16 * LOCAL * 2, 4).is_err());
        assert_eq!(l.logits(100, 16 * LOCAL * 2, 3)?, 100);
        assert!(l.range(4, 0).is_err());
        assert!(!Sampling::new(0, 1., 128, 0.95)?.gpu_fits(4 * 136));
        assert!(Layout::for_policy(1, Sampling::new(0, -0., 20, 0.95)?, 0).is_err());
        Ok(())
    }
}
