//! Serial sampler semantics shared by native requests and diagnostics.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{fmt::Write, io::Read};

pub(crate) const VOCAB: usize = 154880;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sampling {
    pub seed: u64,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
}
// Policies are identities of exact wire bits, including the sign of zero.
impl PartialEq for Sampling {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
            && self.top_k == other.top_k
            && self.temperature.to_bits() == other.temperature.to_bits()
            && self.top_p.to_bits() == other.top_p.to_bits()
    }
}
impl Eq for Sampling {}
impl Sampling {
    pub fn new(seed: u64, temperature: f64, top_k: usize, top_p: f64) -> Result<Self> {
        ensure!(
            temperature.is_finite() && top_p.is_finite(),
            "nonfinite sampling policy"
        );
        // Larger k has identical behavior: every target vocabulary candidate is kept.
        Ok(Self {
            seed,
            temperature,
            top_k: top_k.min(VOCAB),
            top_p,
        })
    }
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
    pub fn local_count(&self) -> usize {
        if self.greedy() {
            1
        } else {
            (self.top_k + 8).min(VOCAB / 4)
        }
    }
    pub fn kept(&self, width: usize) -> usize {
        if self.top_k == 0 {
            width
        } else {
            self.top_k.min(width)
        }
        .max(1)
    }
    pub fn gpu_fits(&self, width: usize) -> bool {
        width > 0 && width <= 256 && (self.greedy() || self.kept(width) <= 128)
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut out = b"ALSAM001".to_vec();
        out.extend(self.seed.to_le_bytes());
        out.extend(self.temperature.to_le_bytes());
        out.extend((self.top_k as u64).to_le_bytes());
        out.extend(self.top_p.to_le_bytes());
        out
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == 40 && bytes.starts_with(b"ALSAM001"),
            "sampler wire header"
        );
        let word = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let k = usize::try_from(word(24))?;
        ensure!(k <= VOCAB, "noncanonical sampler top-k");
        let result = Self::new(
            word(8),
            f64::from_bits(word(16)),
            k,
            f64::from_bits(word(32)),
        )?;
        ensure!(result.encode() == bytes, "noncanonical sampler encoding");
        Ok(result)
    }
}
pub(crate) fn random_seed() -> Result<u64> {
    let mut bytes = [0; 8];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes) & ((1 << 63) - 1))
}
pub(crate) fn seed_for(tokens: &[i32], salt: i64) -> u64 {
    // Stream the canonical decimal representation; no million-token String.
    let mut hash = Sha256::new();
    let mut text = String::with_capacity(24);
    for (i, token) in tokens.iter().enumerate() {
        if i != 0 {
            hash.update(b",");
        }
        text.clear();
        write!(&mut text, "{token}").unwrap();
        hash.update(text.as_bytes());
    }
    text.clear();
    write!(&mut text, "|{salt}").unwrap();
    hash.update(text.as_bytes());
    u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap()) & ((1 << 63) - 1)
}
pub(crate) fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
pub(crate) fn uniform(seed: u64, position: u64, id: u32) -> f64 {
    let x = mix(seed.wrapping_add(0x9E3779B97F4A7C15));
    let x = mix(x ^ position.wrapping_mul(0xD1B54A32D192ED03));
    let x = mix(x ^ id as u64);
    // Preserve the expression's rounding, including its possible result of 1.0.
    (x >> 11) as f64 * 2.0f64.powi(-53) + 2.0f64.powi(-54)
}
/// NumPy float64 pairwise sum (PW_BLOCKSIZE=128), including recursive large k.
fn pairwise(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(-0.0, |a, &b| a + b);
    }
    if n > 128 {
        let mut half = n / 2;
        half -= half % 8;
        return pairwise(&values[..half]) + pairwise(&values[half..]);
    }
    let mut a: [f64; 8] = values[..8].try_into().unwrap();
    for chunk in values[8..n - n % 8].as_chunks::<8>().0 {
        for j in 0..8 {
            a[j] += chunk[j];
        }
    }
    let sum = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    values[n - n % 8..].iter().fold(sum, |a, &b| a + b)
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub id: u32,
    pub value: f32,
}
/// Bounded host fallback/oracle. Candidate collection is a separate GPU concern.
pub(crate) fn choose(input: &[Candidate], position: u64, s: Sampling) -> Result<(u32, usize)> {
    ensure!(!input.is_empty() && input.len() <= VOCAB, "candidate count");
    ensure!(
        input
            .iter()
            .all(|c| (c.id as usize) < VOCAB && c.value.is_finite()),
        "invalid candidate"
    );
    let mut values = input.to_vec();
    values.sort_unstable_by_key(|c| c.id);
    ensure!(
        values.windows(2).all(|w| w[0].id != w[1].id),
        "duplicate candidate id"
    );
    values.sort_unstable_by(|a, b| b.value.partial_cmp(&a.value).unwrap().then(a.id.cmp(&b.id)));
    if s.greedy() {
        return Ok((values[0].id, 1));
    }
    values.truncate(s.kept(values.len()));
    let scaled: Vec<f64> = values
        .iter()
        .map(|c| c.value as f64 / s.temperature.max(1e-6))
        .collect();
    let mut keep = scaled.len();
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let probs: Vec<f64> = scaled.iter().map(|v| (v - scaled[0]).exp()).collect();
        let total = pairwise(&probs);
        let mut cs = 0.0;
        keep = 1;
        for p in probs {
            cs += p / total;
            keep += usize::from(cs < s.top_p);
        }
        keep = keep.min(values.len());
    }
    let mut best = f64::NEG_INFINITY;
    let mut token = values[0].id;
    for i in 0..keep {
        let score = scaled[i] - (-uniform(s.seed, position, values[i].id).ln()).ln();
        if score > best {
            best = score;
            token = values[i].id;
        }
    }
    Ok((token, keep))
}

/// Masked rows use the fixed-width reduction, including zero mass
/// from forbidden top-k fillers. Only finite candidates can ever be published.
pub(crate) fn choose_masked(
    input: &[Candidate],
    position: u64,
    s: Sampling,
    logical_width: usize,
) -> Result<(u32, usize)> {
    ensure!(
        !input.is_empty() && input.len() <= logical_width && logical_width <= VOCAB,
        "masked candidate count"
    );
    ensure!(
        input
            .iter()
            .all(|c| (c.id as usize) < VOCAB && c.value.is_finite()),
        "invalid masked candidate"
    );
    let mut values = input.to_vec();
    values.sort_unstable_by_key(|c| c.id);
    ensure!(
        values.windows(2).all(|w| w[0].id != w[1].id),
        "duplicate candidate id"
    );
    values.sort_unstable_by(|a, b| b.value.partial_cmp(&a.value).unwrap().then(a.id.cmp(&b.id)));
    if s.greedy() {
        return Ok((values[0].id, 1));
    }
    let k = s.kept(logical_width);
    values.truncate(k);
    let scaled: Vec<f64> = values
        .iter()
        .map(|c| c.value as f64 / s.temperature.max(1e-6))
        .collect();
    let mut keep = k;
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mut probs: Vec<f64> = scaled.iter().map(|v| (v - scaled[0]).exp()).collect();
        probs.resize(k, 0.0);
        let total = pairwise(&probs);
        let mut cs = 0.0;
        keep = 1;
        for p in probs {
            cs += p / total;
            keep += usize::from(cs < s.top_p);
        }
        keep = keep.min(k);
    }
    let mut best = f64::NEG_INFINITY;
    let mut token = values[0].id;
    for i in 0..keep.min(values.len()) {
        let score = scaled[i] - (-uniform(s.seed, position, values[i].id).ln()).ln();
        if score > best {
            best = score;
            token = values[i].id;
        }
    }
    Ok((token, keep))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_zero_mass_padding_matches_the_qualified_fixed_width_draw() -> Result<()> {
        // Finite sentinels are so far below this fixture's permitted logits
        // that exp underflows exactly to zero. The unmasked chooser supplies
        // an independent fixed-width oracle for masked sampling.
        for count in [1, 3, 7, 8, 9, 15, 19, 20, 29] {
            let input: Vec<_> = (0..count)
                .map(|i| Candidate {
                    id: i as u32,
                    value: -(i as f32) / 8.0,
                })
                .collect();
            let mut padded = input.clone();
            padded.extend((count..112).map(|i| Candidate {
                id: i as u32,
                value: -1e30,
            }));
            for top_k in [0, 1, 20, 56] {
                for top_p in [0.0, 0.2, 0.5, 0.95, f64::from_bits(1.0f64.to_bits() - 1)] {
                    for position in [0, 16384, 1048575] {
                        let policy = Sampling::new(450, 1.0, top_k, top_p)?;
                        assert_eq!(
                            choose_masked(&input, position, policy, 112)?,
                            choose(&padded, position, policy)?
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn canonical_policy_and_rng() -> Result<()> {
        for seed in [0, 1, u64::MAX, 1 << 63] {
            for temperature in [-1.0, -0.0, 1e-8, 0.7, 1.0] {
                let s = Sampling::new(seed, temperature, 20, 0.95)?;
                assert_eq!(s.encode(), Sampling::decode(&s.encode())?.encode());
                for n in 0..40 {
                    assert!(Sampling::decode(&s.encode()[..n]).is_err());
                }
                for pos in [0, 19, 16384, 1048575, u64::MAX] {
                    for id in [0, 38720, 154879] {
                        let u = uniform(seed, pos, id);
                        assert!(u > 0.0 && u <= 1.0);
                        assert_eq!(u.to_bits(), uniform(seed, pos, id).to_bits());
                    }
                }
            }
        }
        assert!(Sampling::new(0, f64::NAN, 20, 0.95).is_err());
        assert!(Sampling::new(0, 1.0, 20, f64::INFINITY).is_err());
        assert_eq!(Sampling::new(0, 1.0, 0, 0.95)?.local_count(), 8);
        assert_eq!(
            Sampling::new(0, 1.0, usize::MAX, 0.95)?.local_count(),
            38720
        );
        let tokens = [154820, 123, 42];
        let expected = Sha256::digest(b"154820,123,42|-9");
        assert_eq!(
            seed_for(&tokens, -9),
            u64::from_le_bytes(expected[..8].try_into().unwrap()) & ((1 << 63) - 1)
        );
        assert!(random_seed()? < 1 << 63);
        Ok(())
    }
    #[test]
    fn order_cuts_replay_and_rejections() -> Result<()> {
        let input = [
            Candidate { id: 9, value: 0.0 },
            Candidate { id: 3, value: -0.0 },
            Candidate { id: 7, value: 0.0 },
            Candidate { id: 1, value: 0.0 },
        ];
        assert_eq!(choose(&input, 0, Sampling::new(0, 0.0, 20, 0.95)?)?, (1, 1));
        for (p, keep) in [(0.25, 1), (0.5, 2), (0.75, 3), (1.0, 4), (0.0, 4)] {
            let s = Sampling::new(63, 1.0, 0, p)?;
            assert_eq!(choose(&input, 17, s)?.1, keep);
            let mut reversed = input;
            reversed.reverse();
            assert_eq!(choose(&input, 17, s)?, choose(&reversed, 17, s)?);
        }
        let s = Sampling::new(19, 0.7, 20, 0.95)?;
        let before: Vec<_> = (0..128).map(|p| choose(&input, p, s).unwrap()).collect();
        for p in 1..32 {
            let _ = choose(&input, p, Sampling::new(99, 1.1, 3, 0.8)?)?;
        }
        assert_eq!(
            before,
            (0..128)
                .map(|p| choose(&input, p, s).unwrap())
                .collect::<Vec<_>>()
        );
        assert!(choose(&[input[0], input[0]], 0, s).is_err());
        assert!(
            choose(
                &[Candidate {
                    id: 0,
                    value: f32::NAN
                }],
                0,
                s
            )
            .is_err()
        );
        assert!(choose(&[], 0, s).is_err());
        assert_eq!(pairwise(&vec![1.0; 154880]), 154880.0);
        Ok(())
    }
}
