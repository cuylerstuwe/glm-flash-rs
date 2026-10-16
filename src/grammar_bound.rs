//! Bounded prepared grammar wire value. Compilation and matcher state are not
//! serialized; each rank compiles the same source before accepting generation.
use crate::grammar_request::{self, CompileKind};
use anyhow::{Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
pub(crate) const SOURCE_BYTES: usize = 16 << 20;
pub(crate) const WIRE_BYTES: usize = 14 + SOURCE_BYTES;
// Verified against the canonical tokenizer in GRAMMAR-VOCABULARY.md.
const THINK_OPEN: i32 = 154841;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bound {
    kind: CompileKind,
    source: Arc<str>,
    active: bool,
}
impl Bound {
    pub(crate) fn from_request(body: &Value, prompt: &[i32]) -> Result<Option<Self>> {
        let Some(spec) = grammar_request::parse(body)? else {
            return Ok(None);
        };
        ensure!(
            prompt.len() <= 1048576
                && prompt
                    .iter()
                    .all(|&id| (0..crate::sampling::VOCAB as i32).contains(&id)),
            "grammar prompt bounds"
        );
        let (kind, source) = spec.compile_input()?;
        ensure!(
            !source.is_empty() && source.len() <= SOURCE_BYTES,
            "grammar source bounds"
        );
        Ok(Some(Self {
            kind,
            source: source.into(),
            active: grammar_request::think_active(
                prompt,
                THINK_OPEN,
                crate::output_decode::THINK_END,
            ),
        }))
    }
    pub(crate) fn kind(&self) -> CompileKind {
        self.kind
    }
    pub(crate) fn source(&self) -> &str {
        &self.source
    }
    #[allow(dead_code)] // Consumed by the independently qualified native preparer.
    pub(crate) fn tools(&self) -> bool {
        self.kind == CompileKind::StructuralTag
    }
    pub(crate) fn active(&self) -> bool {
        self.active
    }
    fn header(&self) -> [u8; 14] {
        let mut out = [0; 14];
        out[..8].copy_from_slice(b"ALGRM001");
        out[8] = match self.kind() {
            CompileKind::JsonSchema => 0,
            CompileKind::Regex => 1,
            CompileKind::Ebnf => 2,
            CompileKind::StructuralTag => 3,
        };
        out[9] = u8::from(self.active());
        out[10..14].copy_from_slice(&(self.source().len() as u32).to_le_bytes());
        out
    }
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(14 + self.source.len());
        out.extend(self.header());
        out.extend(self.source.as_bytes());
        out
    }
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            (15..=WIRE_BYTES).contains(&bytes.len()) && bytes.starts_with(b"ALGRM001"),
            "grammar wire header"
        );
        ensure!(bytes[8] <= 3 && bytes[9] <= 1, "grammar wire flags");
        let n = u32::from_le_bytes(bytes[10..14].try_into()?) as usize;
        ensure!(n == bytes.len() - 14, "grammar wire length");
        let kind = match bytes[8] {
            0 => CompileKind::JsonSchema,
            1 => CompileKind::Regex,
            2 => CompileKind::Ebnf,
            _ => CompileKind::StructuralTag,
        };
        Ok(Self {
            kind,
            source: std::str::from_utf8(&bytes[14..])?.into(),
            active: bytes[9] != 0,
        })
    }
    #[allow(dead_code)] // Consumed by the independently qualified native preparer.
    pub(crate) fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.header());
        hash.update(self.source.as_bytes());
        hash.finalize().into()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn all_kinds_wire_roundtrip_truncation_flags_and_identity() {
        for body in [
            json!({"guided_json":{}}),
            json!({"guided_regex":"[a-z]+"}),
            json!({"guided_choice":["a","b"]}),
            json!({"tools":[{"function":{"name":"f"}}],"tool_choice":"required"}),
        ] {
            let bound = Bound::from_request(&body, &[THINK_OPEN]).unwrap().unwrap();
            assert!(!bound.active());
            assert!(!bound.source().is_empty());
            assert_eq!(bound.tools(), bound.kind() == CompileKind::StructuralTag);
            let bytes = bound.encode();
            assert_eq!(Bound::decode(&bytes).unwrap(), bound);
            assert_eq!(
                bound.digest(),
                <Sha256 as Digest>::digest(&bytes).as_slice()
            );
            for n in 0..bytes.len() {
                assert!(Bound::decode(&bytes[..n]).is_err());
            }
            for (offset, value) in [(0, b'?'), (8, 4), (9, 2), (10, 255)] {
                let mut bad = bytes.clone();
                bad[offset] = value;
                assert!(Bound::decode(&bad).is_err());
            }
            let mut bad = bytes.clone();
            bad.push(0);
            assert!(Bound::decode(&bad).is_err());
        }
    }
    #[test]
    fn actual_prompt_controls_activation_and_plain_auto_stays_absent() {
        let body = json!({"guided_regex":"x","enable_thinking":false});
        let active = Bound::from_request(&body, &[]).unwrap().unwrap();
        let inactive = Bound::from_request(&body, &[THINK_OPEN]).unwrap().unwrap();
        assert!(active.active());
        assert!(!inactive.active());
        assert_ne!(active.digest(), inactive.digest());
        assert!(
            Bound::from_request(&body, &[THINK_OPEN, crate::output_decode::THINK_END])
                .unwrap()
                .unwrap()
                .active()
        );
        assert!(
            Bound::from_request(&json!({"tools":[{"function":{"name":"f"}}]}), &[])
                .unwrap()
                .is_none()
        );
        assert!(Bound::from_request(&body, &[-1]).is_err());
        assert!(Bound::from_request(&body, &[crate::sampling::VOCAB as i32]).is_err());
    }
    #[test]
    fn upper_bound_utf8_and_empty_source_are_explicit() {
        let full = Bound {
            kind: CompileKind::Regex,
            source: "a".repeat(SOURCE_BYTES).into(),
            active: true,
        };
        assert_eq!(Bound::decode(&full.encode()).unwrap(), full);
        let mut bad = full.encode();
        bad.push(0);
        assert!(Bound::decode(&bad).is_err());
        let mut bad = Bound {
            kind: CompileKind::Regex,
            source: "é".into(),
            active: true,
        }
        .encode();
        bad[14] = 255;
        assert!(Bound::decode(&bad).is_err());
        let empty = Bound {
            kind: CompileKind::Regex,
            source: "".into(),
            active: true,
        };
        assert!(Bound::decode(&empty.encode()).is_err());
    }
}
