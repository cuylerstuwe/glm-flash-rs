//! Canonical JSON for the frozen template: insertion order, Python spacing and
//! finite binary64 spelling. Parsing is bounded and never silently rounds ints.
use anyhow::{Result, bail, ensure};
use serde_json::Value;

pub(crate) const BODY_BYTES: usize = 16 << 20;
pub(crate) const RENDER_BYTES: usize = 32 << 20;
const NODES: usize = 262_144;

pub(crate) fn parse(bytes: &[u8]) -> Result<Value> {
    ensure!(bytes.len() <= BODY_BYTES, "chat JSON exceeds 16 MiB");
    let value: Value = serde_json::from_slice(bytes)?;
    check(&value)?;
    Ok(value)
}
pub(crate) fn check(value: &Value) -> Result<()> {
    fn walk(v: &Value, depth: usize, nodes: &mut usize, bytes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(depth <= 48 && *nodes <= NODES, "chat JSON depth/node bound");
        match v {
            Value::String(s) => *bytes += s.len(),
            Value::Number(n) => {
                // arbitrary_precision prevents serde_json from rounding a large
                // integer into f64 before this explicit refusal.
                ensure!(
                    n.as_i64().is_some()
                        || n.as_u64().is_some()
                        || (n.is_f64() && n.as_f64().is_some_and(f64::is_finite)),
                    "chat JSON number outside supported integer/binary64 range"
                );
            }
            Value::Array(a) => {
                for item in a {
                    walk(item, depth + 1, nodes, bytes)?;
                }
            }
            Value::Object(m) => {
                for (key, item) in m {
                    *bytes += key.len();
                    // A flat raw prompt can contain the full 1M token policy. It
                    // does not expand template/JSON recursion or the general node budget.
                    if depth == 0
                        && key == "prompt"
                        && let Some(ids) = item.as_array()
                    {
                        ensure!(
                            ids.len() <= crate::cache_pages::CONTEXT,
                            "token prompt context bound"
                        );
                        ensure!(
                            ids.iter().all(|id| id.as_u64().is_some()),
                            "token prompt requires unsigned integers"
                        );
                    } else {
                        walk(item, depth + 1, nodes, bytes)?;
                    }
                }
            }
            _ => {}
        }
        ensure!(*bytes <= BODY_BYTES, "chat JSON string bound");
        Ok(())
    }
    walk(value, 0, &mut 0, &mut 0)
}
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}
pub(crate) fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
pub(crate) fn strip(s: &str) -> &str {
    s.trim_matches(whitespace)
}

fn float(value: f64) -> String {
    let mut buffer = ryu::Buffer::new();
    let raw = buffer.format_finite(value);
    let (sign, raw) = raw.strip_prefix('-').map_or(("", raw), |s| ("-", s));
    let (mantissa, exponent) = raw
        .split_once('e')
        .map_or((raw, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
    let mut decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let zeros = digits.bytes().take_while(|c| *c == b'0').count();
    if zeros == digits.len() {
        return format!("{sign}0.0");
    }
    digits.drain(..zeros);
    decimal -= zeros as i32;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let exp = decimal - 1;
    if !(-4..16).contains(&exp) {
        let tail = if digits.len() == 1 {
            String::new()
        } else {
            format!(".{}", &digits[1..])
        };
        format!(
            "{sign}{}{tail}e{}{:02}",
            &digits[..1],
            if exp < 0 { "-" } else { "+" },
            exp.abs()
        )
    } else if decimal <= 0 {
        format!("{sign}0.{}{}", "0".repeat((-decimal) as usize), digits)
    } else if decimal as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(decimal as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..decimal as usize],
            &digits[decimal as usize..]
        )
    }
}

pub(crate) fn json(value: &Value, sorted: bool, compact: bool) -> Result<String> {
    fn push(out: &mut String, s: &str) -> Result<()> {
        ensure!(
            s.len() <= RENDER_BYTES - out.len(),
            "JSON expansion exceeds 32 MiB"
        );
        out.push_str(s);
        Ok(())
    }
    fn write(v: &Value, out: &mut String, sorted: bool, compact: bool) -> Result<()> {
        let sep = if compact { "," } else { ", " };
        match v {
            Value::Null => push(out, "null")?,
            Value::Bool(b) => push(out, if *b { "true" } else { "false" })?,
            Value::String(s) => push(out, &serde_json::to_string(s)?)?,
            Value::Number(n) => {
                if n.is_f64() {
                    let Some(f) = n.as_f64().filter(|v| v.is_finite()) else {
                        bail!("nonfinite JSON number")
                    };
                    push(out, &float(f))?;
                } else {
                    push(out, &n.to_string())?;
                }
            }
            Value::Array(a) => {
                push(out, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        push(out, sep)?;
                    }
                    write(v, out, sorted, compact)?;
                }
                push(out, "]")?;
            }
            Value::Object(m) => {
                push(out, "{")?;
                let mut pairs: Vec<_> = m.iter().collect();
                if sorted {
                    pairs.sort_unstable_by_key(|(k, _)| *k);
                }
                for (i, (k, v)) in pairs.into_iter().enumerate() {
                    if i > 0 {
                        push(out, sep)?;
                    }
                    push(out, &serde_json::to_string(k)?)?;
                    push(out, if compact { ":" } else { ": " })?;
                    write(v, out, sorted, compact)?;
                }
                push(out, "}")?;
            }
        }
        Ok(())
    }
    let mut out = String::new();
    write(value, &mut out, sorted, compact)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_python_json_and_float_boundaries() {
        let raw = r#"{"z":1.0,"中":"<&/\u2028🦀","a":[-0.0,1e-5,1e-4,1e15,1e16,1.25e20,5e-324,1.7976931348623157e308]}"#;
        let v = parse(raw.as_bytes()).unwrap();
        assert_eq!(
            json(&v, false, false).unwrap(),
            "{\"z\": 1.0, \"中\": \"<&/\u{2028}🦀\", \"a\": [-0.0, 1e-05, 0.0001, 1000000000000000.0, 1e+16, 1.25e+20, 5e-324, 1.7976931348623157e+308]}"
        );
        assert_eq!(
            json(&parse(br#"{"z":2,"a":{"y":1,"x":0}}"#).unwrap(), true, true).unwrap(),
            r#"{"a":{"x":0,"y":1},"z":2}"#
        );
    }
    #[test]
    fn bounds_and_large_numbers_fail_without_rounding() {
        assert!(parse(b"18446744073709551615").is_ok());
        assert!(parse(b"18446744073709551616").is_err());
        assert!(parse(b"-9223372036854775809").is_err());
        assert!(parse(b"1e1000").is_err());
        assert!(parse(format!("{}0{}", "[".repeat(49), "]".repeat(49)).as_bytes()).is_err());
        assert!(parse(&vec![b' '; BODY_BYTES + 1]).is_err());
        assert_eq!(strip("\u{1c} café \u{85}"), "café");
    }
    #[test]
    fn duplicate_keys_keep_original_position_and_last_value() {
        let v = parse(br#"{"b":1,"a":2,"b":3}"#).unwrap();
        assert_eq!(json(&v, false, false).unwrap(), r#"{"b": 3, "a": 2}"#);
    }
}

#[cfg(test)]
mod raw_prompt_tests {
    use super::*;
    #[test]
    fn raw_flat_context_does_not_remove_general_json_bounds() -> Result<()> {
        check(&serde_json::json!({"prompt":vec![19; crate::cache_pages::CONTEXT]}))?;
        assert!(
            check(&serde_json::json!({"prompt":vec![19; crate::cache_pages::CONTEXT+1]})).is_err()
        );
        assert!(check(&serde_json::json!({"elsewhere":vec![19; NODES]})).is_err());
        assert!(check(&serde_json::json!({"prompt":[[19]]})).is_err());
        assert!(check(&serde_json::json!({"prompt":[true]})).is_err());
        Ok(())
    }
}
