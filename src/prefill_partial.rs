//! Store prefill rank partials in their existing BF16 exchange format.
//! This changes neither weights nor the fixed rank-order HC sum.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) fp32_cast_calls: u64,
    pub(crate) fp32_cast_elements: u64,
    pub(crate) bf16_copy_calls: u64,
    pub(crate) bf16_copy_elements: u64,
    pub(crate) direct_slot_calls: u64,
    pub(crate) direct_slot_elements: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Policy(bool);
impl Policy {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self> {
        ensure!(
            matches!(value, None | Some("0") | Some("1")),
            "ALASKA_PREFILL_DIRECT_BF16 must be 0/1"
        );
        Ok(Self(value == Some("1")))
    }
    pub(crate) fn environment() -> Result<Self> {
        match std::env::var("ALASKA_PREFILL_DIRECT_BF16") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn identity(self) -> [u8; 16] {
        let mut bytes = *b"ALPFBF01\0\0\0\0\0\0\0\0";
        bytes[8] = u8::from(self.0);
        bytes
    }
    pub(crate) fn enabled(self) -> bool {
        self.0
    }
    pub(crate) fn element_bytes(self) -> usize {
        if self.0 { 2 } else { 4 }
    }
    pub(crate) fn f32_output(self, name: &str, original: bool) -> bool {
        // Shared expert down remains FP32: combine must consume the same inputs
        // and round only the final weighted sum to BF16.
        original && !(self.0 && matches!(name, "kda.o" | "dsa.o" | "mlp.down"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounds_only_final_communication_partials() -> Result<()> {
        let on = Policy::parse(Some("1"))?;
        let off = Policy::parse(None)?;
        assert_eq!(off, Policy::default());
        assert_eq!(off, Policy::parse(Some("0"))?);
        assert_ne!(on.identity(), off.identity());
        assert_eq!(on.element_bytes(), 2);
        assert_eq!(off.element_bytes(), 4);
        for name in ["kda.o", "dsa.o", "mlp.down"] {
            assert!(off.f32_output(name, true));
            assert!(!on.f32_output(name, true));
        }
        for name in ["moe.shared.down", "another.output"] {
            assert!(on.f32_output(name, true));
        }
        for name in ["kda.proj", "dsa.proj", "mlp.gu", "moe.shared.gu"] {
            assert!(!on.f32_output(name, false));
            assert!(!off.f32_output(name, false));
        }
        for bad in ["", "yes", "true", "2", " 1", "1 "] {
            assert!(Policy::parse(Some(bad)).is_err());
        }
        Ok(())
    }
}
