//! Validated serial-generation policy and bounded, canonical wire encoding.
use anyhow::{Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) const MAX_STOPS: usize = 32;
pub(crate) const STOP_BYTES: usize = 16 << 10;
pub(crate) const WIRE_BYTES: usize =
    12 + 40 + 8 + 4 + crate::grammar_bound::WIRE_BYTES + 4 * MAX_STOPS + STOP_BYTES;

pub(crate) fn parse_stops(value: &Value) -> Result<Vec<&str>> {
    let strings: Vec<&str> = match value {
        Value::Null => vec![],
        Value::String(s) => vec![s],
        Value::Array(a) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| anyhow::anyhow!("stop must contain strings"))
            })
            .collect::<Result<_>>()?,
        _ => anyhow::bail!("stop must be a string or string array"),
    };
    ensure!(strings.len() <= MAX_STOPS, "too many stops");
    ensure!(
        strings.iter().map(|s| s.len()).sum::<usize>() <= STOP_BYTES,
        "stop byte bound"
    );
    Ok(strings.into_iter().filter(|s| !s.is_empty()).collect())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    thinking: bool,
    tools: bool,
    stops: Vec<String>,
    sampling: Option<crate::sampling::Sampling>,
    thinking_budget: Option<usize>,
    grammar: Option<crate::grammar_bound::Bound>,
    drafts: bool,
    ignore_eos: bool,
}
impl Policy {
    pub(crate) fn new(thinking: bool, tools: bool, stops: &Value) -> Result<Self> {
        Ok(Self {
            thinking,
            tools,
            stops: parse_stops(stops)?.into_iter().map(str::to_owned).collect(),
            sampling: None,
            thinking_budget: None,
            grammar: None,
            drafts: true,
            ignore_eos: false,
        })
    }
    pub(crate) fn with_sampling(mut self, sampling: crate::sampling::Sampling) -> Result<Self> {
        self.sampling = Some(crate::sampling::Sampling::decode(&sampling.encode())?);
        Ok(self)
    }
    pub(crate) fn sampling(&self) -> Option<crate::sampling::Sampling> {
        self.sampling
    }
    pub(crate) fn with_thinking_budget(mut self, budget: Option<usize>) -> Result<Self> {
        ensure!(
            self.thinking || budget.is_none(),
            "thinking budget without thinking"
        );
        self.thinking_budget = budget;
        Ok(self)
    }
    pub(crate) fn thinking_budget(&self) -> Option<usize> {
        self.thinking_budget
    }
    pub(crate) fn with_grammar(mut self, grammar: Option<crate::grammar_bound::Bound>) -> Self {
        self.grammar = grammar;
        self
    }
    pub(crate) fn grammar(&self) -> Option<&crate::grammar_bound::Bound> {
        self.grammar.as_ref()
    }
    pub(crate) fn with_drafts(mut self, drafts: bool) -> Self {
        self.drafts = drafts;
        self
    }
    pub(crate) fn drafts(&self) -> bool {
        self.drafts
    }
    pub(crate) fn with_ignore_eos(mut self, ignore: bool) -> Result<Self> {
        self.ignore_eos = ignore;
        self.validate()?;
        Ok(self)
    }
    pub(crate) fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.ignore_eos || self.grammar.is_none(),
            "ignore_eos is unavailable with an active grammar"
        );
        Ok(())
    }
    pub(crate) fn thinking(&self) -> bool {
        self.thinking
    }
    pub(crate) fn tools(&self) -> bool {
        self.tools
    }
    pub(crate) fn has_stops(&self) -> bool {
        !self.stops.is_empty()
    }
    pub(crate) fn stops(&self) -> Value {
        serde_json::json!(self.stops)
    }
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = b"ALGEN005".to_vec();
        out.extend_from_slice(&[
            self.thinking as u8,
            self.tools as u8,
            self.stops.len() as u8,
            u8::from(self.sampling.is_some())
                | (u8::from(self.thinking_budget.is_some()) << 1)
                | (u8::from(self.grammar.is_some()) << 2)
                | (u8::from(!self.drafts()) << 3)
                | (u8::from(self.ignore_eos) << 4),
        ]);
        if let Some(sampling) = self.sampling {
            out.extend(sampling.encode());
        }
        if let Some(budget) = self.thinking_budget {
            out.extend_from_slice(&(budget as u64).to_le_bytes());
        }
        if let Some(grammar) = &self.grammar {
            let bytes = grammar.encode();
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend(bytes);
        }
        for s in &self.stops {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }
    pub(crate) fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.encode()).into()
    }
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            (12..=WIRE_BYTES).contains(&bytes.len()) && bytes.starts_with(b"ALGEN005"),
            "generation policy header"
        );
        ensure!(
            bytes[8] <= 1 && bytes[9] <= 1 && bytes[10] as usize <= MAX_STOPS && bytes[11] <= 31,
            "generation policy flags"
        );
        let mut strings = Vec::new();
        let mut at = 12;
        let sampling = if bytes[11] & 1 != 0 {
            ensure!(bytes.len() >= 52, "missing sampling policy");
            at = 52;
            Some(crate::sampling::Sampling::decode(&bytes[12..52])?)
        } else {
            None
        };
        let thinking_budget = if bytes[11] & 2 != 0 {
            ensure!(at + 8 <= bytes.len(), "missing thinking budget");
            let budget = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            Some(usize::try_from(budget)?)
        } else {
            None
        };
        let grammar = if bytes[11] & 4 != 0 {
            ensure!(at + 4 <= bytes.len(), "missing grammar policy length");
            let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            at += 4;
            ensure!(n <= bytes.len() - at, "grammar policy length");
            let grammar = crate::grammar_bound::Bound::decode(&bytes[at..at + n])?;
            at += n;
            Some(grammar)
        } else {
            None
        };
        for _ in 0..bytes[10] {
            ensure!(at + 4 <= bytes.len(), "generation stop length missing");
            let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            at += 4;
            ensure!(n > 0 && n <= bytes.len() - at, "generation stop length");
            strings.push(std::str::from_utf8(&bytes[at..at + n])?.to_owned());
            at += n;
        }
        ensure!(at == bytes.len(), "generation policy trailing bytes");
        let mut policy = Self::new(bytes[8] != 0, bytes[9] != 0, &serde_json::json!(strings))?
            .with_thinking_budget(thinking_budget)?
            .with_grammar(grammar)
            .with_drafts(bytes[11] & 8 == 0)
            .with_ignore_eos(bytes[11] & 16 != 0)?;
        policy.sampling = sampling;
        ensure!(policy.encode() == bytes, "noncanonical generation policy");
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn policy_roundtrip_and_rejections() -> Result<()> {
        for value in [
            Value::Null,
            json!(""),
            json!(["", "🦀\u{0000}", "STOP"]),
            json!("x".repeat(STOP_BYTES)),
        ] {
            let p = Policy::new(true, true, &value)?;
            assert_eq!(Policy::decode(&p.encode())?, p);
            for end in 0..p.encode().len() {
                assert!(Policy::decode(&p.encode()[..end]).is_err());
            }
        }
        for value in [
            json!(3),
            json!([true]),
            json!(vec!["x"; 33]),
            json!("x".repeat(STOP_BYTES + 1)),
        ] {
            assert!(Policy::new(false, false, &value).is_err());
        }
        let p = Policy::new(false, false, &Value::Null)?.encode();
        for (i, b) in [(0, b'X'), (8, 2), (9, 2), (10, 33), (11, 1)] {
            let mut bad = p.clone();
            bad[i] = b;
            assert!(Policy::decode(&bad).is_err());
        }
        let mut bad = p;
        bad.push(0);
        assert!(Policy::decode(&bad).is_err());
        Ok(())
    }
    #[test]
    fn sampling_bits_bounds_and_policy_identity() -> Result<()> {
        use crate::sampling::Sampling;
        let base = Policy::new(true, true, &json!("s".repeat(STOP_BYTES)))?;
        let a = base
            .clone()
            .with_sampling(Sampling::new(u64::MAX, -0.0, 154880, 0.95)?)?;
        assert_eq!(Policy::decode(&a.encode())?, a);
        for n in 12..52 {
            assert!(Policy::decode(&a.encode()[..n]).is_err());
        }
        let b = base
            .clone()
            .with_sampling(Sampling::new(u64::MAX, 0.0, 154880, 0.95)?)?;
        assert_ne!(a.digest(), b.digest());
        assert_ne!(a, b);
        for (off, bits) in [
            (28, f64::NAN.to_bits()),
            (36, u64::MAX),
            (44, f64::INFINITY.to_bits()),
        ] {
            let mut bytes = a.encode();
            bytes[off..off + 8].copy_from_slice(&bits.to_le_bytes());
            assert!(Policy::decode(&bytes).is_err());
        }
        let mut old = a.encode();
        old[..8].copy_from_slice(b"ALGEN001");
        assert!(Policy::decode(&old).is_err());
        assert!(
            base.with_sampling(Sampling {
                seed: 0,
                temperature: f64::NAN,
                top_k: 20,
                top_p: 0.95
            })
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn explicit_budget_wire_flags_bounds_and_identity() -> Result<()> {
        let base = Policy::new(true, true, &json!(["stop", "done"]))?;
        assert_eq!(base.thinking_budget(), None);
        for sampling in [
            None,
            Some(crate::sampling::Sampling::new(73, 1.0, 20, 0.95)?),
        ] {
            let mut plain = base.clone();
            if let Some(sampling) = sampling {
                plain = plain.with_sampling(sampling)?;
            }
            for budget in [0, 1, 131_072, usize::MAX] {
                let policy = plain.clone().with_thinking_budget(Some(budget))?;
                let bytes = policy.encode();
                assert_eq!(policy.thinking_budget(), Some(budget));
                assert_eq!(Policy::decode(&bytes)?, policy);
                assert_ne!(policy.digest(), plain.digest());
                let mut bad = bytes.clone();
                bad[8] = 0;
                assert!(Policy::decode(&bad).is_err());
                bad = bytes.clone();
                bad[11] |= 32;
                assert!(Policy::decode(&bad).is_err());
                bad = bytes.clone();
                bad[..8].copy_from_slice(b"ALGEN002");
                assert!(Policy::decode(&bad).is_err());
                let start = if sampling.is_some() { 52 } else { 12 };
                for end in start..start + 8 {
                    assert!(Policy::decode(&bytes[..end]).is_err());
                }
            }
        }
        assert!(
            Policy::new(false, false, &Value::Null)?
                .with_thinking_budget(Some(0))
                .is_err()
        );
        let full = Policy::new(
            true,
            true,
            &json!(vec!["x".repeat(STOP_BYTES / MAX_STOPS); MAX_STOPS]),
        )?
        .with_sampling(crate::sampling::Sampling::new(73, 1.0, 20, 0.95)?)?
        .with_thinking_budget(Some(usize::MAX))?;
        assert_eq!(
            full.encode().len(),
            WIRE_BYTES - 4 - crate::grammar_bound::WIRE_BYTES
        );
        assert_eq!(Policy::decode(&full.encode())?, full);
        Ok(())
    }
    #[test]
    fn grammar_policy_roundtrip_bound_activation_and_all_flag_combinations() -> Result<()> {
        for sampler in [false, true] {
            for budget in [None, Some(0), Some(32768)] {
                for prompt in [vec![], vec![154841]] {
                    let grammar = crate::grammar_bound::Bound::from_request(
                        &json!({"guided_regex":"a|b"}),
                        &prompt,
                    )?
                    .unwrap();
                    let mut plain =
                        Policy::new(true, true, &json!(["stop"]))?.with_thinking_budget(budget)?;
                    if sampler {
                        plain = plain
                            .with_sampling(crate::sampling::Sampling::new(73, 1., 20, 0.95)?)?;
                    }
                    let policy = plain.clone().with_grammar(Some(grammar.clone()));
                    assert_eq!(policy.grammar(), Some(&grammar));
                    assert_ne!(policy.digest(), plain.digest());
                    let bytes = policy.encode();
                    assert_eq!(Policy::decode(&bytes)?, policy);
                    for end in 0..bytes.len() {
                        assert!(Policy::decode(&bytes[..end]).is_err());
                    }
                    let offset = 12 + usize::from(sampler) * 40 + usize::from(budget.is_some()) * 8;
                    for n in [0u32, 1, u32::MAX] {
                        let mut bad = bytes.clone();
                        bad[offset..offset + 4].copy_from_slice(&n.to_le_bytes());
                        assert!(Policy::decode(&bad).is_err());
                    }
                    let mut old = bytes;
                    old[..8].copy_from_slice(b"ALGEN003");
                    assert!(Policy::decode(&old).is_err());
                }
            }
        }
        // The source cap applies after wrapping, independently of stop bytes.
        let grammar = crate::grammar_bound::Bound::from_request(
            &json!({"guided_regex":"a".repeat(crate::grammar_bound::SOURCE_BYTES)}),
            &[],
        )?
        .unwrap();
        let full = Policy::new(
            true,
            true,
            &json!(vec!["x".repeat(STOP_BYTES / MAX_STOPS); MAX_STOPS]),
        )?
        .with_sampling(crate::sampling::Sampling::new(73, 1., 20, 0.95)?)?
        .with_thinking_budget(Some(usize::MAX))?
        .with_grammar(Some(grammar));
        assert_eq!(full.encode().len(), WIRE_BYTES);
        assert_eq!(Policy::decode(&full.encode())?, full);
        Ok(())
    }
    #[test]
    fn drafts_default_enabled_and_explicit_false_has_wire_identity() -> Result<()> {
        let base = Policy::new(false, false, &Value::Null)?;
        assert!(base.drafts());
        assert_eq!(base.clone().with_drafts(true), base);
        let serial = base.clone().with_drafts(false);
        assert!(!serial.drafts());
        assert_eq!(serial.encode()[11], 8);
        assert_eq!(Policy::decode(&serial.encode())?, serial);
        assert_ne!(serial.digest(), base.digest());
        let grammar =
            crate::grammar_bound::Bound::from_request(&json!({"guided_regex":"x"}), &[])?.unwrap();
        let full = Policy::new(true, true, &json!("stop"))?
            .with_thinking_budget(Some(0))?
            .with_sampling(crate::sampling::Sampling::new(7, 1.0, 20, 0.95)?)?
            .with_grammar(Some(grammar))
            .with_drafts(false);
        assert_eq!(full.encode()[11], 15);
        assert_eq!(Policy::decode(&full.encode())?, full);
        Ok(())
    }
    #[test]
    fn ignore_eos_roundtrip_identity_and_grammar_conflict() -> Result<()> {
        let base = Policy::new(false, false, &json!("AB"))?;
        assert!(!base.ignore_eos());
        assert_eq!(base.clone().with_ignore_eos(false)?, base);
        for drafts in [false, true] {
            let policy = base
                .clone()
                .with_drafts(drafts)
                .with_ignore_eos(true)?
                .with_sampling(crate::sampling::Sampling::new(0, -0., 20, 0.95)?)?;
            assert!(policy.ignore_eos());
            assert_ne!(policy.digest(), base.digest());
            assert_eq!(Policy::decode(&policy.encode())?, policy);
            let mut old = policy.encode();
            old[..8].copy_from_slice(b"ALGEN004");
            assert!(Policy::decode(&old).is_err());
            let mut unknown = policy.encode();
            unknown[11] |= 32;
            assert!(Policy::decode(&unknown).is_err());
        }
        let grammar =
            crate::grammar_bound::Bound::from_request(&json!({"guided_regex":"x"}), &[])?.unwrap();
        assert!(
            base.clone()
                .with_grammar(Some(grammar.clone()))
                .with_ignore_eos(true)
                .is_err()
        );
        // A direct caller can attach a grammar after selecting EOS policy;
        // decoder and admissions must refuse that ordering as well.
        let conflict = base.with_ignore_eos(true)?.with_grammar(Some(grammar));
        assert!(conflict.validate().is_err());
        assert!(Policy::decode(&conflict.encode()).is_err());
        Ok(())
    }
}
