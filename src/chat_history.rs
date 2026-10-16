//! Compatible tool-history repair and bounded retained reasoning. Reply
//! parsing will call remember only after a complete, validated tool reply.
use crate::chat_json::{check, json, strip};
use anyhow::{Result, ensure};
use serde_json::{Value, json as value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

struct Entry {
    text: Arc<str>,
    keys: Vec<String>,
    chars: usize,
    bytes: usize,
}
pub(crate) struct Memory {
    entries: BTreeMap<u64, Entry>,
    by_key: HashMap<String, u64>,
    lru: VecDeque<u64>,
    serial: u64,
    chars: usize,
    bytes: usize,
    max_entries: usize,
    max_chars: usize,
    max_bytes: usize,
}
impl Memory {
    pub(crate) fn new(max_entries: usize, max_chars: usize, max_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            by_key: HashMap::new(),
            lru: VecDeque::new(),
            serial: 0,
            chars: 0,
            bytes: 0,
            max_entries,
            max_chars,
            max_bytes,
        }
    }
    pub(crate) fn put(&mut self, keys: Vec<String>, text: &str) -> Result<()> {
        ensure!(
            keys.len() <= 257 && keys.iter().all(|k| k.len() <= 128),
            "reasoning key bounds"
        );
        let keys: Vec<_> = keys.into_iter().filter(|k| !k.is_empty()).collect();
        let chars = text.chars().count();
        // Text and both copies of keys, plus conservative fixed node/control
        // accounting; allocator/RSS qualification is still a separate gate.
        let bytes = text.len() + keys.iter().map(|k| 2 * k.len() + 160).sum::<usize>() + 256;
        if keys.is_empty()
            || text.is_empty()
            || self.max_entries == 0
            || chars > self.max_chars
            || bytes > self.max_bytes
        {
            return Ok(());
        }
        let id = self
            .serial
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("reasoning serial exhausted"))?;
        self.serial = id;
        for key in &keys {
            self.by_key.insert(key.clone(), id);
        }
        self.entries.insert(
            id,
            Entry {
                text: Arc::from(text),
                keys,
                chars,
                bytes,
            },
        );
        self.lru.push_back(id);
        self.chars += chars;
        self.bytes += bytes;
        while self.entries.len() > self.max_entries
            || self.chars > self.max_chars
            || self.bytes > self.max_bytes
        {
            let old = self.lru.pop_front().unwrap();
            let entry = self.entries.remove(&old).unwrap();
            for key in entry.keys {
                if self.by_key.get(&key) == Some(&old) {
                    self.by_key.remove(&key);
                }
            }
            self.chars -= entry.chars;
            self.bytes -= entry.bytes;
        }
        Ok(())
    }
    pub(crate) fn get(&mut self, keys: &[String]) -> Option<Arc<str>> {
        let id = keys.iter().find_map(|key| self.by_key.get(key)).copied()?;
        self.lru.retain(|old| *old != id);
        self.lru.push_back(id);
        Some(self.entries[&id].text.clone())
    }
    pub(crate) fn remember(&mut self, user: &str, calls: &[Value], reasoning: &str) -> Result<()> {
        let mut keys: Vec<_> = calls
            .iter()
            .filter_map(|c| c["id"].as_str().map(str::to_owned))
            .collect();
        keys.push(signature(user, calls)?);
        self.put(keys, reasoning)
    }
}
pub(crate) fn text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str().or_else(|| v["text"].as_str()))
            .collect(),
        _ => String::new(),
    }
}
fn args(v: &Value) -> Result<Value> {
    let result = match v {
        Value::Null => value!({}),
        Value::String(s) if strip(s).is_empty() => value!({}),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| v.clone()),
        _ => v.clone(),
    };
    check(&result)?;
    Ok(result)
}
pub(crate) fn signature(user: &str, calls: &[Value]) -> Result<String> {
    let shape: Vec<_> = calls
        .iter()
        .map(|c| {
            let function = if c["function"].is_object() {
                &c["function"]
            } else {
                c
            };
            Ok(value!([
                function["name"].as_str().unwrap_or(""),
                args(&function["arguments"])?
            ]))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "sig:{:x}",
        Sha256::digest(json(&value!([user, shape]), true, true)?)
    ))
}
fn call_id(s: &str) -> bool {
    s.len() == 29
        && s.starts_with("call_")
        && s.as_bytes()[5..]
            .iter()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
}

pub(crate) fn prepare(body: &mut Value, memory: &mut Memory) -> Result<()> {
    if let Some(tools) = body
        .get("tools")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        let choice = &body["tool_choice"];
        if choice
            .as_str()
            .is_some_and(|s| strip(s).eq_ignore_ascii_case("none"))
        {
            body["tools"] = value!([]);
        } else if choice["type"]
            .as_str()
            .is_some_and(|s| s.eq_ignore_ascii_case("function"))
        {
            let name = strip(choice["function"]["name"].as_str().unwrap_or(""));
            ensure!(!name.is_empty(), "named tool_choice needs a function name");
            let kept: Vec<_> = tools
                .iter()
                .filter(|t| {
                    let f = if t["function"].is_object() {
                        &t["function"]
                    } else {
                        t
                    };
                    strip(f["name"].as_str().unwrap_or("")) == name
                })
                .cloned()
                .collect();
            ensure!(!kept.is_empty(), "tool_choice names an unoffered function");
            body["tools"] = Value::Array(kept);
        }
    }
    let messages = body["messages"]
        .as_array_mut()
        .ok_or_else(|| anyhow::anyhow!("messages must be an array"))?;
    // Consume each result block once. Repeated remove/insert would shift the
    // rest of a long history for every malformed call. Parse arguments once.
    let mut source = std::mem::take(messages).into_iter().peekable();
    while let Some(mut message) = source.next() {
        if message["role"] != "assistant" || !message["tool_calls"].is_array() {
            messages.push(message);
            continue;
        }
        let calls = message["tool_calls"].as_array_mut().unwrap();
        let mut bad = Vec::with_capacity(calls.len());
        for call in calls.iter_mut() {
            let parsed = args(&call["function"]["arguments"])?;
            let valid = call["function"].is_object() && parsed.is_object();
            bad.push(!valid);
            if valid {
                call["function"]["arguments"] = parsed;
            }
        }
        if !bad.contains(&true) {
            messages.push(message);
            continue;
        }
        let ids: Vec<_> = calls
            .iter()
            .zip(&bad)
            .filter(|(_, b)| **b)
            .filter_map(|(c, _)| c.get("id"))
            .filter(|v| !v.is_null() && **v != "")
            .cloned()
            .collect();
        let mut results = Vec::new();
        while source.peek().is_some_and(|m| m["role"] == "tool") {
            results.push(source.next().unwrap());
        }
        let by_place = ids.is_empty() && results.len() == calls.len();
        let mut index = 0;
        calls.retain(|_| {
            let keep = !bad[index];
            index += 1;
            keep
        });
        if calls.is_empty() {
            message.as_object_mut().unwrap().shift_remove("tool_calls");
        }
        messages.push(message);
        messages.extend(
            results
                .into_iter()
                .enumerate()
                .filter(|(k, r)| !ids.contains(&r["tool_call_id"]) && !(by_place && bad[*k]))
                .map(|(_, r)| r),
        );
    }
    let mut user = String::new();
    for m in messages {
        if m.get("content").is_some_and(Value::is_null) {
            m["content"] = value!("");
        }
        if m["role"] == "user" {
            user = text(&m["content"]);
        }
        if m["role"] != "assistant" {
            continue;
        }
        if !m["reasoning_content"].is_string() && m["reasoning"].is_string() {
            m["reasoning_content"] = m["reasoning"].clone();
        }
        if !m["reasoning_content"].is_string()
            && !text(&m["content"]).contains("</think>")
            && let Some(calls) = m["tool_calls"].as_array().filter(|a| !a.is_empty())
        {
            let mut keys: Vec<_> = calls
                .iter()
                .filter_map(|c| c["id"].as_str())
                .filter(|s| call_id(s))
                .map(str::to_owned)
                .collect();
            keys.push(signature(&user, calls)?);
            if let Some(reasoning) = memory.get(&keys) {
                m["reasoning_content"] = value!(&*reasoning);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_touch_overwrite_eviction_and_unicode_bounds() {
        let mut m = Memory::new(2, 4, 4096);
        m.put(vec!["old".into(), "same".into()], "🦀🦀").unwrap();
        m.put(vec!["new".into(), "same".into()], "é").unwrap();
        assert_eq!(&*m.get(&["old".into()]).unwrap(), "🦀🦀");
        assert_eq!(&*m.get(&["same".into()]).unwrap(), "é");
        m.put(vec!["third".into()], "x").unwrap();
        assert!(m.get(&["old".into()]).is_none());
        assert_eq!(&*m.get(&["same".into()]).unwrap(), "é");
        m.put(vec!["huge".into()], "abcde").unwrap();
        assert!(m.get(&["huge".into()]).is_none());
        assert!(m.bytes <= m.max_bytes && m.chars <= m.max_chars);
        let mut small = Memory::new(10, 100, 256);
        small.put(vec!["a".into()], "b").unwrap();
        assert!(small.entries.is_empty());
    }
    #[test]
    fn signature_order_and_argument_forms_are_canonical() {
        let a = value!([{"function":{"name":"f","arguments":"{\"z\": 1, \"a\": 2}"}}]);
        let b = value!([{"function":{"name":"f","arguments":{"a":2,"z":1}}}]);
        assert_eq!(
            signature("user", a.as_array().unwrap()).unwrap(),
            signature("user", b.as_array().unwrap()).unwrap()
        );
        assert_ne!(
            signature("other", a.as_array().unwrap()).unwrap(),
            signature("user", b.as_array().unwrap()).unwrap()
        );
    }
}
