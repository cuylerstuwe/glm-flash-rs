//! Sampling defaults and bounded numeric request conversion. JSON integers
//! retain their low 64 bits; integral JSON floats first round to binary64.
//! Numeric strings support ASCII decimal syntax.
use crate::sampling::{Sampling, VOCAB};
use anyhow::{Context, Result, ensure};
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
struct Integer {
    negative: bool,
    magnitude: u64, // Saturating magnitude, for policy bounds only.
    low: u64,       // Exact signed value modulo 2^64, for seeded sampling.
}
fn decimal(text: &str) -> Result<Integer> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let bytes = digits.as_bytes();
    ensure!(!bytes.is_empty(), "integer required");
    let (mut magnitude, mut low) = (0u64, 0u64);
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_' {
            ensure!(
                i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit(),
                "integer underscore"
            );
        } else {
            ensure!(b.is_ascii_digit(), "ASCII decimal integer required");
            let d = (b - b'0') as u64;
            magnitude = magnitude.saturating_mul(10).saturating_add(d);
            low = low.wrapping_mul(10).wrapping_add(d);
        }
    }
    Ok(Integer {
        negative: negative && magnitude != 0,
        magnitude,
        low: if negative { low.wrapping_neg() } else { low },
    })
}
fn integral_float(x: f64) -> Result<Integer> {
    ensure!(
        x.is_finite() && x.fract() == 0.0,
        "finite integral number required"
    );
    let a = x.abs();
    // Extract the exact integer represented by binary64 rather than saturating a
    // float-to-u64 cast (which would destroy the seed's wrapping semantics).
    let bits = a.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32 - 1023;
    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let low = if a == 0.0 {
        0
    } else if exponent <= 52 {
        mantissa >> (52 - exponent)
    } else if exponent - 52 >= 64 {
        0
    } else {
        mantissa.wrapping_shl((exponent - 52) as u32)
    };
    Ok(Integer {
        negative: x < 0.0,
        magnitude: a as u64,
        low: if x < 0.0 { low.wrapping_neg() } else { low },
    })
}
fn integer(value: &Value) -> Result<Integer> {
    match value {
        Value::String(s) => decimal(s),
        Value::Number(n) => {
            let text = n.to_string();
            if text.contains(['.', 'e', 'E']) {
                integral_float(text.parse()?)
            } else {
                decimal(&text)
            }
        }
        _ => anyhow::bail!("integer required"),
    }
}
fn finite(value: &Value) -> Result<f64> {
    let text = match value {
        Value::String(s) => s.trim().to_owned(),
        Value::Number(n) => n.to_string(),
        _ => anyhow::bail!("finite number required"),
    };
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_' {
            ensure!(
                i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit(),
                "number underscore"
            );
        }
    }
    let x: f64 = text.replace('_', "").parse()?;
    ensure!(x.is_finite(), "finite number required");
    Ok(x)
}
fn present<'a>(body: &'a Value, name: &str) -> Option<&'a Value> {
    body.get(name).filter(|v| !v.is_null())
}
fn output(value: Integer) -> Result<usize> {
    ensure!(
        !value.negative && value.magnitude >= 1,
        "output limit outside policy"
    );
    Ok(usize::try_from(value.magnitude).unwrap_or(usize::MAX))
}
pub(crate) struct Options {
    /// The client's max_tokens (or max_completion_tokens), before any clamp.
    pub requested: Option<usize>,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: Option<u64>,
}
impl Options {
    pub fn parse(body: &Value) -> Result<Self> {
        let float = |name, default| {
            present(body, name)
                .map(finite)
                .transpose()
                .with_context(|| format!("invalid {name}"))
                .map(|v| v.unwrap_or(default))
        };
        let int = |name| {
            present(body, name)
                .map(integer)
                .transpose()
                .with_context(|| format!("invalid {name}"))
        };
        let temperature = float("temperature", 1.0)?;
        let top_p = float("top_p", 0.95)?;
        let k = int("top_k")?;
        let seed = int("seed")?.map(|v| v.low);
        // Validate both fields, including ones that greedy decoding won't use.
        let a = int("max_tokens")?.map(output).transpose()?;
        let b = int("max_completion_tokens")?.map(output).transpose()?;
        ensure!(
            a.is_none() || b.is_none() || a == b,
            "conflicting output limits"
        );
        // Like the compatible service, any positive limit is
        // accepted; preparation checks it against the prompt and clamps it.
        Ok(Self {
            requested: a.or(b),
            temperature,
            top_p,
            seed,
            top_k: k
                .map(|v| {
                    if v.negative {
                        0
                    } else {
                        v.magnitude.min(VOCAB as u64) as usize
                    }
                })
                .unwrap_or(20),
        })
    }
    /// Called once on the ingress owner. Followers receive the immutable policy;
    /// they never read entropy, reroll a seed, or reparse request JSON.
    pub fn resolve(&self) -> Result<Option<Sampling>> {
        let seed = match self.seed {
            Some(s) => s,
            // Keep an explicit canonical greedy policy so configured target
            // verification can dispatch it. Greedy never consumes entropy.
            None if self.temperature <= 0.0 => 0,
            None => crate::sampling::random_seed()?,
        };
        Ok(Some(Sampling::new(
            seed,
            self.temperature,
            self.top_k,
            self.top_p,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn integer_seed_conversion_preserves_rounding_and_wrap() -> Result<()> {
        for (text, low) in [
            ("18446744073709551615", u64::MAX),
            ("-1", u64::MAX),
            ("18446744073709551616", 0),
            ("18446744073709551617", 1),
            ("-18446744073709551617", u64::MAX),
            ("9007199254740993.0", 9007199254740992),
            ("18446744073709551615.0", 0),
            ("18446744073709555712.0", 4096),
            ("-18446744073709555712.0", 0u64.wrapping_sub(4096)),
            ("1e100", 0),
            ("-0.0", 0),
            ("20e0", 20),
        ] {
            assert_eq!(integer(&serde_json::from_str(text)?)?.low, low, "{text}");
        }
        assert_eq!(
            integer(&json!(" +18_446_744_073_709_551_615 "))?.low,
            u64::MAX
        );
        for v in [
            json!(true),
            json!([]),
            json!(20.5),
            json!("20.0"),
            json!("2e1"),
            json!("1__2"),
            json!("_2"),
            json!("2_"),
            json!(""),
            json!("0x20"),
        ] {
            assert!(integer(&v).is_err(), "{v}");
        }
        Ok(())
    }
    #[test]
    fn defaults_nulls_numbers_and_all_fields_validated() -> Result<()> {
        let p = Options::parse(&json!({"seed":450}))?;
        assert_eq!(p.requested, None);
        assert_eq!(p.resolve()?, Some(Sampling::new(450, 1.0, 20, 0.95)?));
        let p = Options::parse(
            &json!({"temperature":null,"top_p":null,"top_k":null,"seed":null,"max_tokens":null}),
        )?;
        let a = p.resolve()?.unwrap();
        let b = p.resolve()?.unwrap();
        assert!(a.seed < (1 << 63) && b.seed < (1 << 63) && a.seed != b.seed);
        let p = Options::parse(
            &json!({"seed":-1,"temperature":"7e-1","top_k":20.0,"top_p":"0.9_5","max_tokens":"8","max_completion_tokens":8.0}),
        )?;
        assert_eq!(p.resolve()?, Some(Sampling::new(u64::MAX, 0.7, 20, 0.95)?));
        assert_eq!(p.requested, Some(8));
        let p = Options::parse(&json!({"max_tokens":1048576}))?;
        assert_eq!(p.requested, Some(1048576));
        assert_eq!(Options::parse(&json!({"top_k":-20}))?.top_k, 0);
        assert_eq!(
            Options::parse(&serde_json::from_str("{\"top_k\":999999999999999999999}")?)?.top_k,
            VOCAB
        );
        for field in [
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "max_tokens",
            "max_completion_tokens",
        ] {
            let mut b = json!({"temperature":0});
            b[field] = json!(true);
            assert!(Options::parse(&b).is_err(), "{field}");
        }
        for b in [
            json!({"top_p":"nan"}),
            json!({"temperature":"inf"}),
            json!({"seed":"1e2"}),
            json!({"max_tokens":0}),
            json!({"max_tokens":-1}),
            json!({"max_tokens":8,"max_completion_tokens":9}),
            json!({"top_p":"1_.0"}),
        ] {
            assert!(Options::parse(&b).is_err(), "{b}");
        }
        assert_eq!(
            Options::parse(&json!({"temperature":-0.0,"seed":-1}))?.resolve()?,
            Some(Sampling::new(u64::MAX, -0.0, 20, 0.95)?)
        );
        for temperature in [0.0, -0.0, -2.0] {
            let options = Options::parse(&json!({"temperature":temperature}))?;
            let first = options.resolve()?.unwrap();
            assert!(first.greedy());
            assert_eq!(first.seed, 0);
            assert_eq!(options.resolve()?, Some(first));
        }
        Ok(())
    }
}
