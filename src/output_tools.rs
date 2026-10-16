//! Final-output tool parsing. Fixed non-backtracking regexes delimit calls;
//! schema coercion is explicit, bounded Rust JSON. No regex runs per token.
use crate::{chat_json as j, output_decode::MAX_BYTES};
use anyhow::{Result, ensure};
use fancy_regex::Regex;
use serde_json::{Map, Value, json};
use std::sync::OnceLock;

struct Patterns {
    calls: Regex,
    function: Regex,
    parameter: Regex,
    glm: Regex,
    trailing: Regex,
}
fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| Patterns {
        calls: Regex::new(r"(?is)<tool_call>[\s\x1c-\x1f]*(.*?)[\s\x1c-\x1f]*</tool_call>").unwrap(),
        function: Regex::new(r"(?is)^[\s\x1c-\x1f]*<funct[iİı]on=([^>\s\x1c-\x1f]+)>[\s\x1c-\x1f]*(.*?)[\s\x1c-\x1f]*</funct[iİı]on>[\s\x1c-\x1f]*$").unwrap(),
        parameter: Regex::new(r"(?is)<parameter=([^>\s\x1c-\x1f]+)>\n?(.*?)\n?</parameter>").unwrap(),
        glm: Regex::new(r"(?s)<arg_key>[\s\x1c-\x1f]*(.*?)[\s\x1c-\x1f]*</arg_key>[\s\x1c-\x1f]*<arg_value>(.*?)</arg_value>").unwrap(),
        trailing: Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap(),
    })
}

fn parsed(text: &str) -> Result<Option<Value>> {
    ensure!(text.len() <= j::BODY_BYTES, "tool JSON exceeds 16 MiB");
    match serde_json::from_str::<Value>(text) {
        Ok(v) => {
            j::check(&v)?;
            Ok(Some(v))
        }
        Err(e) if e.is_syntax() || e.is_eof() => Ok(None),
        Err(e) => Err(e.into()),
    }
}
fn function(tool: &Value) -> &Value {
    if tool["function"].is_object() {
        &tool["function"]
    } else {
        tool
    }
}
fn name(tool: &Value) -> &str {
    j::strip(function(tool)["name"].as_str().unwrap_or(""))
}
const NULL: u8 = 1;
const BOOL: u8 = 2;
const INTEGER: u8 = 4;
const NUMBER: u8 = 8;
const STRING: u8 = 16;
const ARRAY: u8 = 32;
const OBJECT: u8 = 64;
fn kind(s: &str) -> u8 {
    match s {
        "null" => NULL,
        "boolean" => BOOL,
        "integer" => INTEGER,
        "number" => NUMBER,
        "string" => STRING,
        "array" => ARRAY,
        "object" => OBJECT,
        _ => 0,
    }
}
fn value_kind(v: &Value) -> u8 {
    match v {
        Value::Null => NULL,
        Value::Bool(_) => BOOL,
        Value::String(_) => STRING,
        Value::Array(_) => ARRAY,
        Value::Object(_) => OBJECT,
        Value::Number(n) => {
            if n.is_f64() {
                NUMBER
            } else {
                INTEGER
            }
        }
    }
}
fn schema_types(prop: &Value) -> Option<u8> {
    if !prop.is_object() {
        return None;
    }
    if let Some(s) = prop["type"].as_str() {
        return Some(kind(s));
    }
    if let Some(a) = prop["type"]
        .as_array()
        .filter(|a| !a.is_empty() && a.iter().all(Value::is_string))
    {
        return Some(a.iter().fold(0, |bits, v| bits | kind(v.as_str().unwrap())));
    }
    for field in ["anyOf", "oneOf"] {
        if let Some(a) = prop[field].as_array().filter(|a| !a.is_empty()) {
            return a
                .iter()
                .try_fold(0, |bits, v| Some(bits | schema_types(v)?));
        }
    }
    if let Some(a) = prop["enum"].as_array().filter(|a| !a.is_empty()) {
        return Some(a.iter().fold(0, |bits, v| bits | value_kind(v)));
    }
    prop.get("const").map(value_kind)
}
pub(super) fn closed_json(text: &str) -> Option<String> {
    let mut closers = Vec::new();
    let mut string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if string {
            if escaped {
                escaped = false;
            } else {
                escaped = ch == '\\';
                string = ch != '"';
            }
        } else {
            match ch {
                '"' => string = true,
                '[' => closers.push(']'),
                '{' => closers.push('}'),
                ']' | '}' if closers.pop() != Some(ch) => return None,
                _ => {}
            }
        }
    }
    if closers.is_empty() || string {
        None
    } else {
        Some(
            text.trim_end_matches(j::whitespace).to_owned()
                + &closers.into_iter().rev().collect::<String>(),
        )
    }
}
pub(super) fn typed(raw: &str, prop: &Value) -> Result<Value> {
    let Some(types) = schema_types(prop) else {
        return Ok(parsed(raw)?.unwrap_or_else(|| Value::String(raw.to_owned())));
    };
    if types & STRING != 0 {
        return Ok(if types & NULL != 0 && j::strip(raw) == "null" {
            Value::Null
        } else {
            Value::String(raw.to_owned())
        });
    }
    let text = j::strip(raw);
    let mut candidates = Vec::new();
    if let Some(v) = parsed(text)? {
        candidates.push(v);
    }
    if types & (ARRAY | OBJECT) != 0
        && let Some(fixed) = closed_json(text)
        && let Some(v) = parsed(&fixed)?
    {
        candidates.push(v);
    }
    if let Some(s) = candidates.first().and_then(Value::as_str)
        && let Some(v) = parsed(j::strip(s))?
    {
        candidates.push(v);
    }
    match text {
        "True" => candidates.push(Value::Bool(true)),
        "False" => candidates.push(Value::Bool(false)),
        "None" => candidates.push(Value::Null),
        _ => {}
    }
    for mut v in candidates {
        if value_kind(&v) == NUMBER && types & INTEGER != 0 && types & NUMBER == 0 {
            let f = v.as_f64().unwrap();
            if f.fract() == 0.0 {
                ensure!(
                    f >= i64::MIN as f64 && f < 18446744073709551616.0,
                    "coerced integer outside range"
                );
                v = if f < 0.0 {
                    json!(f as i64)
                } else {
                    json!(f as u64)
                };
            }
        }
        let k = value_kind(&v);
        if types & k != 0 || k == INTEGER && types & NUMBER != 0 {
            return Ok(v);
        }
    }
    Ok(Value::String(raw.to_owned()))
}
fn glm(block: &str, tools: &[Value]) -> Result<Option<(String, Value)>> {
    let name = j::strip(block.split_once("<arg_key>").map_or(block, |(n, _)| n));
    if name.is_empty() || name.contains(['<', '\n']) {
        return Ok(None);
    }
    let folded = name.to_lowercase();
    let schema = tools
        .iter()
        .rfind(|t| self::name(t).to_lowercase() == folded)
        .map(|t| &function(t)["parameters"]["properties"]);
    let mut args = Map::new();
    for (i, m) in patterns().glm.captures_iter(block).enumerate() {
        ensure!(i < 4096, "tool parameter count bound");
        let m = m?;
        let key = j::strip(m.get(1).unwrap().as_str());
        let prop = schema.and_then(|s| s.get(key)).unwrap_or(&Value::Null);
        args.insert(key.to_owned(), typed(m.get(2).unwrap().as_str(), prop)?);
    }
    Ok(Some((name.to_owned(), Value::Object(args))))
}
fn block(block: &str, tools: &[Value]) -> Result<Option<(String, Value)>> {
    if let Some(payload) = parsed(block)? {
        if !payload.is_object() {
            return Ok(None);
        }
        let f = function(&payload);
        let mut args = f
            .get("arguments")
            .or_else(|| f.get("parameters"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        if let Some(s) = args.as_str() {
            args = if j::strip(s).is_empty() {
                json!({})
            } else if let Some(v) = parsed(s)? {
                v
            } else {
                return alternate(block, tools);
            };
        }
        return Ok(f["name"]
            .as_str()
            .filter(|n| !n.is_empty())
            .map(|n| (n.to_owned(), args)));
    }
    alternate(block, tools)
}
fn alternate(block: &str, tools: &[Value]) -> Result<Option<(String, Value)>> {
    if let Some(m) = patterns().function.captures(block)? {
        let mut args = Map::new();
        for (i, p) in patterns()
            .parameter
            .captures_iter(m.get(2).unwrap().as_str())
            .enumerate()
        {
            ensure!(i < 4096, "tool parameter count bound");
            let p = p?;
            args.insert(
                j::strip(p.get(1).unwrap().as_str()).to_owned(),
                Value::String(p.get(2).unwrap().as_str().to_owned()),
            );
        }
        Ok(Some((
            j::strip(m.get(1).unwrap().as_str()).to_owned(),
            Value::Object(args),
        )))
    } else {
        glm(block, tools)
    }
}
pub(crate) fn parse(
    text: &str,
    tools: &[Value],
    ids: &mut impl FnMut() -> Result<String>,
) -> Result<(String, Vec<Value>)> {
    ensure!(
        text.len() <= MAX_BYTES && tools.len() <= 1024,
        "tool output bound"
    );
    if tools.is_empty() {
        return Ok((text.to_owned(), vec![]));
    }
    for tool in tools {
        j::check(tool)?;
    }
    let known: std::collections::HashMap<_, _> = tools
        .iter()
        .map(|t| (name(t).to_lowercase(), name(t)))
        .collect();
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0;
    for (i, m) in patterns().calls.captures_iter(text).enumerate() {
        ensure!(i < 256, "tool call block count bound");
        let m = m?;
        let entire = m.get(0).unwrap();
        content.push_str(&text[cursor..entire.start()]);
        cursor = entire.end();
        if let Some((name, args)) = block(j::strip(m.get(1).unwrap().as_str()), tools)?
            && let Some(&canonical) = known.get(&name.to_lowercase())
        {
            let id = ids()?;
            ensure!(
                id.len() == 29
                    && id.starts_with("call_")
                    && id[5..]
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "invalid generated tool ID"
            );
            calls.push(json!({"id":id,"type":"function","function":{"name":canonical,"arguments":j::json(&args,false,true)?}}));
            continue;
        }
        content.push_str(entire.as_str());
    }
    content.push_str(&text[cursor..]);
    Ok((j::strip(&content).to_owned(), calls))
}
pub(crate) fn trailing(text: &str) -> Result<(&str, &str)> {
    ensure!(text.len() <= MAX_BYTES, "reasoning output bound");
    let mut end = text.trim_end_matches(j::whitespace);
    if let Some(s) = end.strip_suffix("</think>") {
        end = s.trim_end_matches(j::whitespace);
    }
    let mut matches = Vec::new();
    for m in patterns().trailing.find_iter(end) {
        ensure!(matches.len() < 256, "reasoning tool block count bound");
        matches.push(m?);
    }
    let mut start = end.len();
    for m in matches.into_iter().rev() {
        if !j::strip(&end[m.end()..start]).is_empty() {
            break;
        }
        start = m.start();
    }
    Ok(if start == end.len() {
        (text, "")
    } else {
        (text[..start].trim_end_matches(j::whitespace), &end[start..])
    })
}
