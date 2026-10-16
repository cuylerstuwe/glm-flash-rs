//! Canonical grammar tokenizer views. No tokenizer inference heuristics:
//! only the already-qualified GLM ByteLevel/null-normalizer layout is accepted.
use crate::{output_decode, tokenizer::NativeTokenizer};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) const VOCAB: usize = 154880;
pub(crate) const THINK_OPEN: i32 = 154841;
pub(crate) const TOOL_MARKERS: [(u32, &str); 6] = [
    (154843, "<tool_call>"),
    (154844, "</tool_call>"),
    (154847, "<arg_key>"),
    (154848, "</arg_key>"),
    (154849, "<arg_value>"),
    (154850, "</arg_value>"),
];
pub(crate) struct Views {
    pub(crate) text: Vec<Vec<u8>>,
    pub(crate) tools: Vec<Vec<u8>>,
    pub(crate) stop: [i32; 4],
    pub(crate) add_prefix_space: bool,
    pub(crate) identity: [u8; 32],
}
impl Views {
    /// The tokenizer must come from tokenizer::load, which validates the exact
    /// scanner contract. Source is its original JSON, before scanner replacement.
    pub(crate) fn new(tok: &NativeTokenizer, source: &Value) -> Result<Self> {
        ensure!(
            source["decoder"]["type"] == "ByteLevel" && source["normalizer"].is_null(),
            "unsupported grammar tokenizer metadata"
        );
        ensure!(
            source["pre_tokenizer"]["type"] == "Sequence"
                && source["pre_tokenizer"]["pretokenizers"]
                    .as_array()
                    .is_some_and(|a| a.len() == 2
                        && a[0]["type"] == "Split"
                        && a[1]["type"] == "ByteLevel"
                        && a[1]["add_prefix_space"] == false),
            "unsupported grammar pre-tokenizer"
        );
        // The official HF detector uses Prepend/Metaspace, not the decoder's
        // add_prefix_space:true. Canonical null-normalizer + this sequence =>false.
        let add_prefix_space = false;
        for (id, marker) in output_decode::EOS
            .into_iter()
            .zip([
                "<|endoftext|>",
                "<|user|>",
                "<|observation|>",
                "<|assistant|>",
            ])
            .chain([
                (THINK_OPEN as u32, "<think>"),
                (output_decode::THINK_END as u32, "</think>"),
            ])
        {
            ensure!(
                tok.id_to_token(id).as_deref() == Some(marker),
                "canonical grammar control token differs"
            );
        }
        for (id, marker) in TOOL_MARKERS {
            ensure!(
                tok.id_to_token(id).as_deref() == Some(marker),
                "canonical tool marker differs"
            );
        }
        let mut text = vec![Vec::new(); VOCAB];
        let mut seen = vec![false; VOCAB];
        let mut bytes = 0;
        for (token, id) in tok.get_vocab(true) {
            let id = id as usize;
            ensure!(
                id < VOCAB && !seen[id],
                "grammar vocabulary ID bound/duplicate"
            );
            seen[id] = true;
            bytes += token.len();
            ensure!(bytes <= 16 << 20, "grammar vocabulary byte bound");
            text[id] = token.into_bytes();
        }
        let mut tools = text.clone();
        let added = source["added_tokens"]
            .as_array()
            .context("grammar added tokens missing")?;
        for token in added {
            let id = token["id"]
                .as_u64()
                .filter(|&id| id < VOCAB as u64)
                .context("grammar added ID bound")? as usize;
            let content = token["content"]
                .as_str()
                .context("grammar added content missing")?;
            ensure!(
                text[id] == content.as_bytes(),
                "grammar added token identity differs"
            );
            text[id].clear();
            if !TOOL_MARKERS
                .iter()
                .any(|&(marker_id, marker)| marker_id as usize == id && marker == content)
            {
                tools[id].clear();
            }
        }
        let stop = output_decode::EOS.map(|id| id as i32);
        let mut hash = Sha256::new();
        hash.update(b"alaska-grammar-bytelevel-views-v1");
        for view in [&text, &tools] {
            for entry in view {
                hash.update((entry.len() as u32).to_le_bytes());
                hash.update(entry);
            }
        }
        for id in stop {
            hash.update(id.to_le_bytes());
        }
        Ok(Self {
            text,
            tools,
            stop,
            add_prefix_space,
            identity: hash.finalize().into(),
        })
    }
}
