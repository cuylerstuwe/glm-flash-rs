//! Immutable canonical ByteLevel token bytes and bounded incremental UTF-8.
//! The pinned tokenizer remains the independent whole-sequence decode oracle.
use crate::tokenizer::NativeTokenizer;
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

// The compatible output policy adds <|assistant|> to the checkpoint's three EOS
// IDs, preventing a second assistant turn from leaking into the same reply.
pub(crate) const EOS: [u32; 4] = [154820, 154827, 154829, 154828];
const EOS_TOKENS: [&str; 4] = [
    "<|endoftext|>",
    "<|user|>",
    "<|observation|>",
    "<|assistant|>",
];
pub(crate) const OBSERVATION: u32 = 154829;
// Verified against the canonical tokenizer's added_tokens; Table::new rejects
// a different mapping before the native owner can force this delimiter.
pub(crate) const THINK_END: i32 = 154842;
pub(crate) const MAX_BYTES: usize = 32 << 20;
const VOCAB: usize = 154880;

pub(crate) struct Table {
    bytes: Vec<Option<Box<[u8]>>>,
    identity: [u8; 32],
}
impl Table {
    pub(crate) fn new(tok: &NativeTokenizer) -> Result<Self> {
        let decoder = serde_json::to_value(tok.get_decoder())?;
        ensure!(decoder["type"] == "ByteLevel", "unsupported output decoder");
        for (id, expected) in EOS.into_iter().zip(EOS_TOKENS) {
            ensure!(
                tok.id_to_token(id).as_deref() == Some(expected),
                "canonical EOS differs"
            );
        }
        ensure!(
            tok.id_to_token(THINK_END as u32).as_deref() == Some("</think>"),
            "canonical thinking delimiter differs"
        );
        let mut inverse = [None; 512];
        let mut next = 256;
        for byte in 0..=255u8 {
            let ch =
                if (b'!'..=b'~').contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae {
                    byte as usize
                } else {
                    next += 1;
                    next - 1
                };
            inverse[ch] = Some(byte);
        }
        let mut bytes = vec![None; VOCAB];
        let mut total = 0;
        for (text, id) in tok.get_vocab(true) {
            ensure!((id as usize) < VOCAB, "output vocabulary ID outside model");
            // ByteLevel falls back to the entire original token if even one
            // character is outside its byte alphabet (including added tokens).
            let decoded: Option<Vec<u8>> = text
                .chars()
                .map(|c| inverse.get(c as usize).copied().flatten())
                .collect();
            let decoded = decoded.unwrap_or_else(|| text.into_bytes());
            total += decoded.len();
            ensure!(total <= 16 << 20, "output vocabulary byte bound");
            ensure!(
                bytes[id as usize].is_none(),
                "duplicate output vocabulary ID"
            );
            bytes[id as usize] = Some(decoded.into_boxed_slice());
        }
        let mut hash = Sha256::new();
        hash.update(b"alaska-output-bytelevel-v1");
        for (id, item) in bytes.iter().enumerate() {
            hash.update((id as u32).to_le_bytes());
            hash.update(
                item.as_ref()
                    .map_or(u64::MAX, |v| v.len() as u64)
                    .to_le_bytes(),
            );
            if let Some(item) = item {
                hash.update(item);
            }
        }
        Ok(Self {
            bytes,
            identity: hash.finalize().into(),
        })
    }
    /// Synthetic immutable token bytes for host-only stream-boundary tests.
    /// EOS IDs retain their marker bytes; this never replaces a runtime table.
    #[cfg(test)]
    pub(crate) fn test_bytes(tokens: &[&[u8]]) -> Self {
        assert!(tokens.len() < EOS[0] as usize);
        let mut bytes = vec![None; VOCAB];
        for (id, token) in tokens.iter().enumerate() {
            bytes[id] = Some(token.to_vec().into_boxed_slice());
        }
        for (id, token) in EOS.into_iter().zip(EOS_TOKENS) {
            bytes[id as usize] = Some(token.as_bytes().to_vec().into_boxed_slice());
        }
        Self {
            bytes,
            identity: [0; 32],
        }
    }
    pub(crate) fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub(crate) fn get(&self, id: u32) -> Result<&[u8]> {
        self.bytes
            .get(id as usize)
            .and_then(Option::as_deref)
            .ok_or_else(|| anyhow::anyhow!("unknown/padded output token ID: {id}"))
    }
}

#[derive(Default)]
pub(crate) struct Utf8 {
    pending: Vec<u8>,
    input: usize,
    output: usize,
}
impl Utf8 {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<String> {
        ensure!(bytes.len() <= MAX_BYTES - self.input, "output byte bound");
        self.input += bytes.len();
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        let mut used = 0;
        while used < self.pending.len() {
            match std::str::from_utf8(&self.pending[used..]) {
                Ok(s) => {
                    text.push_str(s);
                    used = self.pending.len();
                }
                Err(error) => {
                    let end = used + error.valid_up_to();
                    text.push_str(std::str::from_utf8(&self.pending[used..end])?);
                    used = end;
                    if let Some(n) = error.error_len() {
                        text.push('\u{fffd}');
                        used += n;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..used);
        ensure!(
            text.len() <= MAX_BYTES - self.output,
            "decoded output byte bound"
        );
        self.output += text.len();
        Ok(text)
    }
    pub(crate) fn finish(&mut self) -> Result<String> {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        ensure!(
            text.len() <= MAX_BYTES - self.output,
            "decoded output byte bound"
        );
        self.output += text.len();
        self.pending.clear();
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_two_byte_sequence_and_partition() -> Result<()> {
        for first in 0..=255u8 {
            for second in 0..=255u8 {
                let bytes = [first, second];
                let mut d = Utf8::default();
                let actual = d.push(&bytes[..1])? + &d.push(&bytes[1..])? + &d.finish()?;
                assert_eq!(actual, String::from_utf8_lossy(&bytes));
            }
        }
        Ok(())
    }
    #[test]
    fn multibyte_partial_invalid_and_literal_replacement() -> Result<()> {
        let mut bytes = "a🦀中é�".as_bytes().to_vec();
        bytes.extend_from_slice(&[0xf0, 0x90, b'X', 0xe0, 0xa0]);
        for chunk in 1..=bytes.len() {
            let mut d = Utf8::default();
            let mut text = String::new();
            for c in bytes.chunks(chunk) {
                text += &d.push(c)?;
            }
            text += &d.finish()?;
            assert_eq!(text, String::from_utf8_lossy(&bytes));
        }
        Ok(())
    }
}
