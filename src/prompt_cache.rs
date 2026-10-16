//! Exact token-ID reuse at proven added-token boundaries.
//! This is CPU tokenization reuse, not KV-cache usage.
use crate::tokenizer::{self, NativeTokenizer};
use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
    sync::Arc,
    time::Instant,
};

#[derive(Deserialize)]
struct Added {
    id: u32,
    content: String,
    special: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}
struct Rules {
    matcher: AhoCorasick,
    added: Vec<Added>,
    safe: HashMap<u32, String>,
}
impl Rules {
    fn new(added: Vec<Added>) -> Result<Self> {
        let mut ids = HashSet::new();
        let mut contents = HashSet::new();
        ensure!(
            added
                .iter()
                .all(|t| ids.insert(t.id) && contents.insert(t.content.as_str())),
            "ambiguous added-token IDs or spellings"
        );
        ensure!(
            added.iter().all(|t| !t.content.is_empty()
                && !t.single_word
                && !t.lstrip
                && !t.rstrip
                && !t.normalized),
            "added-token flags do not permit exact prefix reuse"
        );
        let mut safe = HashMap::new();
        for token in &added {
            let s = &token.content;
            if !token.special {
                continue;
            }
            let overlap = added.iter().any(|other| {
                s.char_indices().any(|(i, _)| {
                    other.content.len() > s.len() - i && other.content.starts_with(&s[i..])
                }) || other
                    .content
                    .find(s)
                    .is_some_and(|i| i + s.len() < other.content.len())
            });
            if !overlap {
                safe.insert(token.id, s.clone());
            }
        }
        ensure!(!safe.is_empty(), "no safe token boundaries");
        let matcher = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(added.iter().map(|t| t.content.as_str()))?;
        Ok(Self {
            matcher,
            added,
            safe,
        })
    }
    fn cuts(
        &self,
        text: &str,
        ids: &[u32],
        base_byte: usize,
        base_token: usize,
    ) -> Option<Vec<Cut>> {
        let mut matches = self.matcher.find_iter(text).filter_map(|m| {
            let token = &self.added[m.pattern().as_usize()];
            self.safe
                .contains_key(&token.id)
                .then_some((m.end(), token.id))
        });
        let mut cuts = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if self.safe.contains_key(id) {
                let (end, found) = matches.next()?;
                if found != *id {
                    return None;
                }
                cuts.push(Cut {
                    byte: base_byte + end,
                    token: base_token + i + 1,
                });
            }
        }
        if matches.next().is_some() {
            return None;
        }
        Some(cuts)
    }
}
#[derive(Clone, Copy)]
struct Cut {
    byte: usize,
    token: usize,
}
struct Entry {
    text: Arc<str>,
    ids: Arc<[u32]>,
    cuts: Box<[Cut]>,
    bytes: usize,
}
pub struct Encoded {
    pub ids: Arc<[u32]>,
    pub reused_ids: usize,
}
pub struct PromptCache {
    tokenizer: NativeTokenizer,
    rules: Rules,
    entries: VecDeque<Entry>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
}
impl PromptCache {
    pub(crate) fn vocabulary(&self) -> Vec<bool> {
        let mut valid = vec![false; crate::sampling::VOCAB];
        for id in self.tokenizer.get_vocab(true).values() {
            if let Some(v) = valid.get_mut(*id as usize) {
                *v = true;
            }
        }
        valid
    }
    pub fn new(path: &Path, max_entries: usize, max_bytes: usize) -> Result<Self> {
        ensure!(
            max_entries > 0 && max_bytes > 0,
            "positive tokenizer-cache bounds required"
        );
        let tokenizer = tokenizer::load(path)?;
        let spec: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            spec["truncation"].is_null() && spec["padding"].is_null(),
            "prefix reuse requires unpadded, untruncated encoding"
        );
        ensure!(
            spec["post_processor"]
                == serde_json::json!({"type":"ByteLevel",
            "add_prefix_space":true,"trim_offsets":false,"use_regex":true}),
            "prefix reuse requires canonical ID-preserving post-processing"
        );
        let rules = Rules::new(serde_json::from_value(spec["added_tokens"].clone())?)?;
        Ok(Self {
            tokenizer,
            rules,
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            bytes: 0,
        })
    }
    fn full(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode_fast(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    fn store(&mut self, text: &str, ids: Arc<[u32]>, cuts: Option<Vec<Cut>>) {
        let Some(cuts) = cuts.filter(|c| !c.is_empty()) else {
            return;
        };
        // Reject a standalone oversized entry before evicting useful prefixes.
        // Account packed payload and fixed owner/control blocks, not token objects.
        let bytes = text.len()
            + ids.len() * 4
            + cuts.len() * std::mem::size_of::<Cut>()
            + std::mem::size_of::<Entry>()
            + 4 * std::mem::size_of::<usize>();
        if bytes > self.max_bytes {
            return;
        }
        while self.entries.len() >= self.max_entries || self.bytes + bytes > self.max_bytes {
            let entry = self.entries.pop_front().expect("bounded nonempty cache");
            self.bytes -= entry.bytes;
        }
        self.entries.push_back(Entry {
            text: Arc::from(text),
            ids,
            cuts: cuts.into_boxed_slice(),
            bytes,
        });
        self.bytes += bytes;
    }
    pub fn encode(&mut self, text: &str) -> Result<Encoded> {
        if let Some(index) = self.entries.iter().rposition(|e| e.text.as_ref() == text) {
            let entry = self.entries.remove(index).unwrap();
            let result = Encoded {
                ids: entry.ids.clone(),
                reused_ids: entry.ids.len(),
            };
            self.entries.push_back(entry);
            return Ok(result);
        }
        let mut best = None;
        let mut best_byte = 0;
        for (index, entry) in self.entries.iter().enumerate().rev() {
            if entry.cuts.last().unwrap().byte <= best_byte {
                continue;
            }
            let (mut lo, mut hi, mut known) = (0, entry.cuts.len(), 0);
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                let end = entry.cuts[mid - 1].byte;
                if text.as_bytes().get(known..end) == Some(&entry.text.as_bytes()[known..end]) {
                    lo = mid;
                    known = end;
                } else {
                    hi = mid - 1;
                }
            }
            if lo > 0 && entry.cuts[lo - 1].byte > best_byte {
                best_byte = entry.cuts[lo - 1].byte;
                best = Some((index, lo));
            }
        }
        if let Some((index, n)) = best {
            let entry = &self.entries[index];
            let cut = entry.cuts[n - 1];
            let tail = self.full(&text[cut.byte..])?;
            let mut ids = Vec::with_capacity(cut.token + tail.len());
            ids.extend_from_slice(&entry.ids[..cut.token]);
            ids.extend_from_slice(&tail);
            let cuts = self
                .rules
                .cuts(&text[cut.byte..], &tail, cut.byte, cut.token)
                .map(|more| {
                    let mut cuts = entry.cuts[..n].to_vec();
                    cuts.extend(more);
                    cuts
                });
            let ids: Arc<[u32]> = ids.into();
            // Refresh the reused entry's recency, then store the new immutable one.
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
            self.store(text, ids.clone(), cuts);
            Ok(Encoded {
                ids,
                reused_ids: cut.token,
            })
        } else {
            let ids: Arc<[u32]> = self.full(text)?.into();
            let cuts = self.rules.cuts(text, &ids, 0, 0);
            self.store(text, ids.clone(), cuts);
            Ok(Encoded { ids, reused_ids: 0 })
        }
    }
    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// Exact sequential-edit qualification plus warmed full-versus-incremental CPU
/// timing. A cache hit here must never be reported as a model KV-cache hit.
pub fn probe(path: &Path, corpus: &Path) -> Result<serde_json::Value> {
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(corpus)?)?;
    let mut cache = PromptCache::new(path, 32, 256 * 1024 * 1024)?;
    let mut checks = 0usize;
    for (i, body) in texts.iter().enumerate() {
        let parent = format!("[gMASK]<sop><|user|>Document {i}: {body}<|assistant|></think>\n");
        let mut sequence = vec![
            parent.clone(),
            format!("{parent}Done.<|observation|>Tool result: café 🦀 {i}.<|assistant|></think>"),
            format!("{parent}A different branch.<|user|>Continue.<|assistant|><think>"),
            parent.replace("Document", "Edited document"),
            parent.clone(),
        ];
        sequence.extend(["[gMASK]<sop><|user|>Unicode a\u{301}中𐐀<|assistant|>".into(),
            "[gMASK]<sop><|user|>Unicode a\u{301}中𐐀<|assistant|>x<tool_call>{\"x\":1}</tool_call><|observation|>ok<|assistant|>".into()]);
        for text in &sequence {
            let expected = cache.full(text)?;
            let result = cache.encode(text)?;
            ensure!(
                result.ids.as_ref() == expected,
                "incremental tokenizer changed IDs"
            );
            ensure!(
                cache.bytes <= cache.max_bytes && cache.entries.len() <= cache.max_entries,
                "cache bound exceeded"
            );
            checks += 1;
        }
    }
    // Sequential branches, partial added-token spellings and UTF-8 truncations
    // exercise safe-cut selection against a complete encode after every edit.
    cache.clear();
    cache.max_entries = 8;
    let pieces = [
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<think>",
        "</think>",
        "[gMASK]",
        "<sop>",
        "<|",
        "assistant",
        "|>",
        "[",
        "]",
        "café",
        "🦀",
        "\u{301}",
        "中",
        "𐐀",
        "'ſ",
        "\r\n  ",
        "12345",
        "text",
    ];
    let mut random = 0x4a65_c073_d881_092fu64;
    let mut history = vec![String::new()];
    for _ in 0..1000 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let mut next = history[random as usize % history.len()].clone();
        if random & 3 == 0 {
            let end = next
                .char_indices()
                .map(|(i, _)| i)
                .chain([next.len()])
                .nth((random >> 8) as usize % (next.chars().count() + 1))
                .unwrap();
            next.truncate(end);
        }
        for shift in [16, 24, 32] {
            next.push_str(pieces[(random >> shift) as usize % pieces.len()]);
        }
        let expected = cache.full(&next)?;
        ensure!(
            cache.encode(&next)?.ids.as_ref() == expected,
            "randomized incremental tokenizer changed IDs"
        );
        if history.len() == 32 {
            history.remove(0);
        }
        history.push(next);
        checks += 1;
    }
    // A too-large prompt must not flush a useful entry or alter externally held IDs.
    cache.clear();
    cache.max_bytes = 4096;
    cache.max_entries = 2;
    let small = "[gMASK]<sop><|user|>Keep this.<|assistant|>";
    let held = cache.encode(small)?.ids;
    let held_copy = held.to_vec();
    let oversized = format!("[gMASK]<sop><|user|>{}<|assistant|>", "xyz ".repeat(10000));
    cache.encode(&oversized)?;
    ensure!(
        cache.entries.iter().any(|e| e.text.as_ref() == small),
        "oversize evicted useful prefix"
    );
    ensure!(
        held.as_ref() == held_copy && cache.encode(small)?.reused_ids == held.len(),
        "immutable entry changed"
    );
    for i in 0..10 {
        cache.encode(&format!("[gMASK]<sop><|user|>entry {i}<|assistant|>"))?;
    }
    ensure!(
        cache.entries.len() <= 2 && cache.bytes <= 4096,
        "LRU bound failed"
    );
    cache.clear();
    cache.max_bytes = 256 * 1024 * 1024;
    cache.max_entries = 32;
    let mut results = Vec::new();
    for body in texts.iter().filter(|s| s.len() > 100000) {
        let parent = format!("[gMASK]<sop><|user|>{body}<|assistant|></think>\n");
        let child =
            format!("{parent}Done.<|observation|>Small tool follow-up.<|assistant|></think>\n");
        let mut pairs = Vec::new();
        for round in 0..7 {
            cache.clear();
            cache.encode(&parent)?;
            let mut times = [0.; 2];
            let mut reused = 0;
            let mut count = 0;
            for which in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                if which == 0 {
                    std::hint::black_box(cache.full(&child)?);
                } else {
                    let result = cache.encode(&child)?;
                    reused = result.reused_ids;
                    count = result.ids.len();
                }
                times[which] = start.elapsed().as_secs_f64();
            }
            let expected = cache.full(&child)?;
            ensure!(
                cache.encode(&child)?.ids.as_ref() == expected,
                "timed cache result differs"
            );
            pairs.push(serde_json::json!({"full_seconds":times[0],"incremental_seconds":times[1],"reused_ids":reused,"total_ids":count}));
        }
        results.push(serde_json::json!({"utf8_bytes":child.len(),"pairs":pairs}));
    }
    Ok(
        serde_json::json!({"sequential_exactness_checks":checks,"oversize_preserves_useful_prefix":true,
        "immutable_ids_preserved":true,"lru_bounds_pass":true,"timings":results,
        "note":"CPU token-ID reuse only, not GPU KV-cache usage or end-to-end serving speed."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn added(id: u32, s: &str, special: bool) -> Added {
        Added {
            id,
            content: s.into(),
            special,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
        }
    }
    #[test]
    fn only_nonoverlapping_plain_tokens_are_safe_cuts() {
        let rules = Rules::new(vec![
            added(1, "<ok>", true),
            added(2, "[a]", true),
            added(3, "]x", false),
            added(4, "[b]", true),
            added(5, "prefix[b]suffix", false),
            added(6, "[c]", true),
            added(7, "[c]extra", false),
        ])
        .unwrap();
        assert_eq!(rules.safe.len(), 1);
        assert!(rules.safe.contains_key(&1));
        let mut unsafe_token = added(8, "x", true);
        unsafe_token.normalized = true;
        assert!(Rules::new(vec![unsafe_token]).is_err());
    }
    #[test]
    fn cuts_require_exact_id_match_and_use_utf8_byte_offsets() {
        let rules = Rules::new(vec![
            added(1, "<a>", true),
            added(2, "<b>", true),
            added(3, "<tool>", false),
        ])
        .unwrap();
        let text = "🦀<a>é<tool><b>";
        let cuts = rules.cuts(text, &[9, 1, 8, 3, 2], 10, 20).unwrap();
        assert_eq!((cuts[0].byte, cuts[0].token), (17, 22));
        assert_eq!((cuts[1].byte, cuts[1].token), (10 + text.len(), 25));
        assert!(rules.cuts(text, &[9, 2, 8, 3, 1], 0, 0).is_none());
        assert!(rules.cuts(text, &[9, 1], 0, 0).is_none());
    }
    #[test]
    fn ambiguous_added_tokens_are_rejected() {
        assert!(Rules::new(vec![added(1, "<a>", true), added(1, "<b>", true)]).is_err());
        assert!(Rules::new(vec![added(1, "<a>", true), added(2, "<a>", true)]).is_err());
    }
}
