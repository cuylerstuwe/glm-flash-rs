//! Exact scanner for the canonical GLM tokenizer's fixed pre-tokenization rules.
//! BPE, added tokens, byte mapping and decoding remain in the pinned Rust library.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::OnceLock, time::Instant};
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizedString,
    PreTokenizer, SplitDelimiterBehavior, TokenizerImpl, pre_tokenizers::byte_level::ByteLevel,
    tokenizer::pattern::Pattern,
};

pub const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LETTER: u8 = 1;
const NUMBER: u8 = 2;
const SPACE: u8 = 4;

// Derive the Unicode classes once from the same pinned Unicode database as the
// reference. No regex matching or per-character allocation occurs in the scanner.
// Rust char::is_alphabetic includes marks outside \p{L} and is not equivalent.
fn classes() -> &'static [u8] {
    static TABLE: OnceLock<Vec<u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = vec![0; 0x110000];
        for (pattern, bit) in [(r"\p{L}", LETTER), (r"\p{N}", NUMBER), (r"\s", SPACE)] {
            let hir = regex_syntax::Parser::new()
                .parse(pattern)
                .expect("fixed class");
            let regex_syntax::hir::HirKind::Class(regex_syntax::hir::Class::Unicode(c)) =
                hir.kind()
            else {
                panic!("fixed Unicode class expected")
            };
            for range in c.iter() {
                for value in &mut table[range.start() as usize..=range.end() as usize] {
                    *value |= bit;
                }
            }
        }
        table
    })
}

#[derive(Clone, Copy)]
struct Char {
    kind: u8,
    width: usize,
}

#[inline]
fn at(text: &str, i: usize, table: &[u8]) -> Char {
    let byte = text.as_bytes()[i];
    if byte < 128 {
        Char {
            kind: table[byte as usize],
            width: 1,
        }
    } else {
        let ch = text[i..].chars().next().expect("character boundary");
        Char {
            kind: table[ch as usize],
            width: ch.len_utf8(),
        }
    }
}

#[inline]
fn newline(byte: u8) -> bool {
    byte == b'\r' || byte == b'\n'
}

/// Visit contiguous byte spans in the regex's leftmost, alternative-priority order.
/// Each pass makes progress; whitespace lookahead is bounded by the next span.
pub fn spans(text: &str, mut emit: impl FnMut(usize, usize)) {
    let table = classes();
    let bytes = text.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = start;
        // Unicode simple case folding of these ASCII contractions adds only ſ.
        if bytes[start] == b'\'' && start + 1 < bytes.len() {
            let next = bytes[start + 1].to_ascii_lowercase();
            if matches!(next, b's' | b't' | b'm' | b'd') {
                end = start + 2;
            } else if text[start + 1..].starts_with('ſ')
                || (start + 2 < bytes.len()
                    && matches!(
                        (next, bytes[start + 2].to_ascii_lowercase()),
                        (b'r', b'e') | (b'v', b'e') | (b'l', b'l')
                    ))
            {
                end = start + 3;
            }
        }
        let first = at(text, start, table);
        if end == start {
            let letter_start = if first.kind & (LETTER | NUMBER) == 0 && !newline(bytes[start]) {
                start + first.width
            } else {
                start
            };
            if letter_start < bytes.len() && at(text, letter_start, table).kind & LETTER != 0 {
                end = letter_start;
                while end < bytes.len() {
                    let c = at(text, end, table);
                    if c.kind & LETTER == 0 {
                        break;
                    }
                    end += c.width;
                }
            } else if first.kind & NUMBER != 0 {
                end = start;
                for _ in 0..3 {
                    if end == bytes.len() {
                        break;
                    }
                    let c = at(text, end, table);
                    if c.kind & NUMBER == 0 {
                        break;
                    }
                    end += c.width;
                }
            }
        }
        if end == start {
            let punctuation_start = start + usize::from(bytes[start] == b' ');
            if punctuation_start < bytes.len() && at(text, punctuation_start, table).kind == 0 {
                end = punctuation_start;
                while end < bytes.len() {
                    let c = at(text, end, table);
                    if c.kind != 0 {
                        break;
                    }
                    end += c.width;
                }
                while end < bytes.len() && newline(bytes[end]) {
                    end += 1;
                }
            }
        }
        if end == start {
            // \s*[\r\n]+ ends at the last CR/LF in the whitespace run.
            // Otherwise \s+(?!\S) leaves one whitespace before a nonspace.
            let mut last_newline = None;
            let mut last_start = start;
            while end < bytes.len() {
                let c = at(text, end, table);
                if c.kind & SPACE == 0 {
                    break;
                }
                last_start = end;
                if newline(bytes[end]) {
                    last_newline = Some(end + 1);
                }
                end += c.width;
            }
            if let Some(n) = last_newline {
                end = n;
            } else if end < bytes.len() && last_start > start {
                end = last_start;
            }
        }
        assert!(
            end > start,
            "scanner must cover every valid UTF-8 character"
        );
        emit(start, end);
        start = end;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scanner {}

impl Pattern for Scanner {
    fn find_matches(&self, text: &str) -> tokenizers::Result<Vec<((usize, usize), bool)>> {
        let mut matches = Vec::new();
        spans(text, |start, end| matches.push(((start, end), true)));
        Ok(matches)
    }
}

impl PreTokenizer for Scanner {
    fn pre_tokenize(&self, input: &mut PreTokenizedString) -> tokenizers::Result<()> {
        input.split(|_, normalized| normalized.split(Self {}, SplitDelimiterBehavior::Isolated))?;
        ByteLevel::new(false, true, false).pre_tokenize(input)
    }
}

pub type NativeTokenizer =
    TokenizerImpl<ModelWrapper, NormalizerWrapper, Scanner, PostProcessorWrapper, DecoderWrapper>;

pub fn load(path: &Path) -> Result<NativeTokenizer> {
    let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected = serde_json::json!({"type":"Sequence", "pretokenizers":[
        {"type":"Split", "pattern":{"Regex":PATTERN}, "behavior":"Isolated", "invert":false},
        {"type":"ByteLevel", "add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]});
    ensure!(
        value["pre_tokenizer"] == expected,
        "unsupported pre-tokenizer; refusing to silently change token IDs"
    );
    ensure!(value["normalizer"].is_null(), "unsupported normalizer");
    value["pre_tokenizer"] = serde_json::json!({});
    classes();
    Ok(serde_json::from_value(value)?)
}

fn regex_spans(regex: &fancy_regex::Regex, text: &str) -> Result<Vec<(usize, usize)>> {
    regex
        .find_iter(text)
        .map(|m| m.map(|m| (m.start(), m.end())).map_err(Into::into))
        .collect()
}

fn scanner_spans(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    spans(text, |s, e| result.push((s, e)));
    result
}

/// Differential validation and paired microbenchmark; never used by serving.
pub fn bench(path: &Path, corpus: &Path, rounds: usize) -> Result<serde_json::Value> {
    ensure!(rounds >= 3, "at least three paired rounds");
    let load_start = Instant::now();
    let candidate = load(path)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let reference = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let regex = fancy_regex::Regex::new(PATTERN)?;
    let texts: Vec<String> = serde_json::from_reader(std::fs::File::open(corpus)?)?;
    let mut results = Vec::new();
    for text in &texts {
        ensure!(
            scanner_spans(text) == regex_spans(&regex, text)?,
            "split mismatch"
        );
        let expected = reference
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let actual = candidate
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(expected.get_ids() == actual.get_ids(), "token IDs differ");
        ensure!(
            expected.get_offsets() == actual.get_offsets(),
            "offsets differ"
        );
        ensure!(
            expected.get_ids()
                == candidate
                    .encode_fast(text.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .get_ids(),
            "fast IDs differ"
        );
        let mut timings = vec![vec![]; 5];
        for round in 0..rounds {
            // Alternate order so allocator/thermal drift does not favor one path.
            let order: Vec<_> = if round % 2 == 0 {
                (0..5).collect()
            } else {
                (0..5).rev().collect()
            };
            for which in order {
                let start = Instant::now();
                match which {
                    0 => {
                        std::hint::black_box(regex_spans(&regex, text)?);
                    }
                    1 => {
                        std::hint::black_box(scanner_spans(text));
                    }
                    2 => {
                        std::hint::black_box(
                            reference
                                .encode(text.as_str(), false)
                                .map_err(|e| anyhow::anyhow!("{e}"))?,
                        );
                    }
                    3 => {
                        std::hint::black_box(
                            candidate
                                .encode(text.as_str(), false)
                                .map_err(|e| anyhow::anyhow!("{e}"))?,
                        );
                    }
                    _ => {
                        std::hint::black_box(
                            candidate
                                .encode_fast(text.as_str(), false)
                                .map_err(|e| anyhow::anyhow!("{e}"))?,
                        );
                    }
                }
                timings[which].push(start.elapsed().as_secs_f64());
            }
        }
        results.push(serde_json::json!({"utf8_bytes":text.len(), "tokens":expected.len(),
            "token_ids_exact":true,"offsets_exact":true,"seconds":{
                "regex_split":timings[0],"scanner_split":timings[1],
                "reference_encode":timings[2],"scanner_encode":timings[3],"scanner_encode_fast":timings[4]}}));
    }
    Ok(
        serde_json::json!({"candidate_load_seconds":load_seconds,"paired_rounds":rounds,
        "note":"Warmed CPU tokenizer measurements; no GPU work or end-to-end inference claim.","cases":results}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_matches_reference_boundaries() {
        let regex = fancy_regex::Regex::new(PATTERN).unwrap();
        let check =
            |s: &str| assert_eq!(scanner_spans(s), regex_spans(&regex, s).unwrap(), "{s:?}");
        for s in [
            "",
            "we'ſ I'd I'RE you've I'll",
            "\r\n \tfoo",
            "a\u{301}b",
            " \u{2003}a",
            "①Ⅲ１２34567",
            "👨‍👩‍👧‍👦你好 안녕",
            "\0\x7f\u{85}\u{2028}\u{2029}",
            "<|assistant|><think>\n<tool_call>{\"x\": 1}</tool_call>",
        ] {
            check(s);
        }
        let alphabet = [
            'a', 'S', '\'', '1', ' ', '\t', '\r', '\n', '!', 'ſ', 'é', '中', '①', 'Ⅲ', '\u{301}',
            '\u{2003}', '\u{85}', '🦀',
        ];
        // Exhaustive short combinations exercise every alternative and boundary.
        for &a in &alphabet {
            for &b in &alphabet {
                for &c in &alphabet {
                    check(&[a, b, c].iter().collect::<String>());
                }
            }
        }
        let mut seed = 0xb1949c623ad2e541u64;
        for _ in 0..10_000 {
            let mut s = String::new();
            for _ in 0..32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed & 3 == 0 {
                    if let Some(ch) = char::from_u32((seed >> 16) as u32 % 0x110000) {
                        s.push(ch);
                    }
                } else {
                    s.push(alphabet[seed as usize % alphabet.len()]);
                }
            }
            check(&s);
        }
    }

    #[test]
    fn unicode_database_matches_reference_for_every_scalar() {
        let patterns = [r"^\p{L}$", r"^\p{N}$", r"^\s$"];
        let regexes = patterns.map(|p| fancy_regex::Regex::new(p).unwrap());
        let table = classes();
        let mut bytes = [0u8; 4];
        for scalar in 0..0x110000 {
            if let Some(ch) = char::from_u32(scalar) {
                let s = ch.encode_utf8(&mut bytes);
                for (i, re) in regexes.iter().enumerate() {
                    assert_eq!(
                        table[scalar as usize] & (1 << i) != 0,
                        re.is_match(s).unwrap(),
                        "U+{scalar:04X}"
                    );
                }
            }
        }
    }
}
