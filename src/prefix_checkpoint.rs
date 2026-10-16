//! In-memory, pure fast-prefill checkpoints. Not an SSD or in-flight request format.
use super::{Attention, Long};
use crate::{
    cache_pages::{CONTEXT, MAX_OUTPUT, PAGE},
    context_memory::Packed,
    cuda::{Buffer, Device},
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use std::{rc::Rc, sync::Arc};

pub(super) const ARITHMETIC: &str = "alaska-rust-prefix-v1/fast-b12x4-bf16/kda64/bm32";
#[derive(Clone)]
pub(super) struct Key {
    model: String,
    arithmetic: &'static str,
    pub(super) tokens: Arc<[i32]>,
}
impl Key {
    pub(super) fn new(model: &str, tokens: &[i32], mixed: bool) -> Result<Self> {
        ensure!(
            !mixed && !tokens.is_empty() && tokens.len().is_multiple_of(64),
            "prefix checkpoints require pure fast prefill on the 64-token grid"
        );
        ensure!(
            tokens.len() < CONTEXT && tokens.iter().all(|&t| crate::vision_input::prompt_token(t)),
            "checkpoint token bounds"
        );
        Ok(Self {
            model: model.to_owned(),
            arithmetic: ARITHMETIC,
            tokens: tokens.into(),
        })
    }
    pub(super) fn validate(&self, model: &str, request: &[i32], output: usize) -> Result<()> {
        ensure!(
            self.model == model && self.arithmetic == ARITHMETIC,
            "prefix checkpoint model/arithmetic mismatch"
        );
        ensure!(
            output > 0 && output <= MAX_OUTPUT && request.len() <= CONTEXT - output,
            "checkpoint request context/output bounds"
        );
        // Leave at least one prompt token to recompute the head: logits and
        // sampler/RNG state are deliberately not stored in this prefix format.
        ensure!(
            request.len() > self.tokens.len()
                && request.starts_with(&self.tokens)
                && request
                    .iter()
                    .all(|&t| crate::vision_input::prompt_token(t)),
            "request does not extend checkpoint tokens"
        );
        Ok(())
    }
}

pub(super) enum SavedLayer {
    Kda { conv: Buffer, active: Buffer },
    Dsa { keys: Buffer, gates: Buffer },
}
pub(super) struct PrefixCheckpoint {
    device: Rc<Device>,
    key: Key,
    pages: Packed,
    layers: Vec<SavedLayer>,
    pub bytes: usize,
}
impl Drop for PrefixCheckpoint {
    fn drop(&mut self) {
        // Drain device work before freeing snapshot buffers.
        let _ = self.device.sync();
    }
}
pub(super) fn model_identity(p: &Prepared) -> Result<String> {
    p.model_identity()
}

impl Long<'_> {
    pub(super) fn checkpoint(&self) -> Result<PrefixCheckpoint> {
        self.ready()?;
        ensure!(
            !self.priority_hold,
            "checkpoint save held by native priority"
        );
        let key = Key::new(
            &self.model_identity,
            &self.sessions[self.active].tokens,
            self.sessions[self.active].mixed,
        )?;
        self.engine.sync()?;
        let d = &self.engine.device;
        let s = &self.engine.stream;
        let pages = self
            .memory
            .pack(s, self.active, self.sessions[self.active].tokens.len())?;
        let mut saved = PrefixCheckpoint {
            device: d.clone(),
            bytes: key.tokens.len() * 4
                + pages
                    .families
                    .iter()
                    .map(|f| f.latent.bytes + f.pooled.bytes)
                    .sum::<usize>(),
            key,
            pages,
            layers: Vec::with_capacity(self.engine.attention.len()),
        };
        saved.layers = self.capture_layers()?;
        saved.bytes += saved.layers.iter().map(SavedLayer::bytes).sum::<usize>();
        self.engine.sync()?;
        Ok(saved)
    }
    pub(super) fn restore_prefix(
        &mut self,
        snapshot: &PrefixCheckpoint,
        request: &[i32],
        output: usize,
    ) -> Result<usize> {
        self.ready()?;
        snapshot
            .key
            .validate(&self.model_identity, request, output)?;
        ensure!(
            Rc::ptr_eq(&snapshot.device, &self.engine.device),
            "checkpoint CUDA owner mismatch"
        );
        // Snapshot fields are private: successful construction fixes all tensor
        // shapes to this model. Reject identity/prefix/limits before mutation.
        let n = snapshot.key.tokens.len();
        let keep = n.min(self.memory.book().slot(self.active)?.pages().len() * PAGE);
        let plan = self.admission(
            request.len(),
            output,
            keep,
            self.sessions[self.active].background,
        )?;
        self.healthy = false;
        self.commit_admission(&plan, keep)?;
        let s = &self.engine.stream;
        self.memory.map(s, self.active, n)?;
        self.memory.restore(s, self.active, &snapshot.pages)?;
        self.restore_layers(&snapshot.layers, n)?;
        unsafe {
            self.sessions[self.active].position.set_i32(s, n as i32)?;
        }
        self.engine.sync()?;
        self.sessions[self.active].tokens.clear();
        self.sessions[self.active]
            .tokens
            .extend_from_slice(&snapshot.key.tokens);
        self.sessions[self.active].mixed = false;
        self.sessions[self.active].pending = None;
        self.sessions[self.active].prompt_end = request.len();
        self.sessions[self.active].total_end = request.len() + output;
        self.latest_logits = None;
        self.healthy = true;
        Ok(n)
    }
}

impl SavedLayer {
    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::Kda { conv, active } => conv.bytes + active.bytes,
            Self::Dsa { keys, gates } => keys.bytes + gates.bytes,
        }
    }
}
impl Long<'_> {
    pub(super) fn layer_bytes(&self) -> usize {
        self.sessions[self.active]
            .recurrent
            .iter()
            .map(|a| a.conv.bytes + a.states[0].bytes)
            .sum::<usize>()
            + self.sessions[self.active]
                .index
                .iter()
                .map(|a| a.keys.bytes + a.gates.bytes)
                .sum::<usize>()
    }
    pub(super) fn capture_layers(&self) -> Result<Vec<SavedLayer>> {
        let d = &self.engine.device;
        let s = &self.engine.stream;
        let mut layers = Vec::with_capacity(self.engine.attention.len());
        let copy = |src: &Buffer| -> Result<Buffer> {
            let dst = d.alloc(src.bytes)?;
            unsafe {
                dst.copy_from(s, 0, src, 0, src.bytes)?;
            }
            Ok(dst)
        };
        let mut j = 0;
        for a in &self.engine.attention {
            let layer = match a {
                Attention::Kda(a) => SavedLayer::Kda {
                    conv: copy(&a.conv)?,
                    active: copy(&a.states[self.sessions[self.active].tokens.len() % 2])?,
                },
                Attention::Dsa(_) => {
                    let index = &self.sessions[self.active].index[j];
                    j += 1;
                    SavedLayer::Dsa {
                        keys: copy(&index.keys)?,
                        gates: copy(&index.gates)?,
                    }
                }
            };
            layers.push(layer);
        }

        Ok(layers)
    }
    pub(super) fn restore_layers(&self, layers: &[SavedLayer], n: usize) -> Result<()> {
        ensure!(
            layers.len() == self.engine.attention.len(),
            "snapshot layer count"
        );
        let s = &self.engine.stream;
        let copy = |dst: &Buffer, src: &Buffer| -> Result<()> {
            ensure!(dst.bytes == src.bytes, "checkpoint buffer layout");
            unsafe { dst.copy_from(s, 0, src, 0, src.bytes) }
        };
        let mut j = 0;
        for (dst, src) in self.engine.attention.iter().zip(layers) {
            match (dst, src) {
                (Attention::Kda(a), SavedLayer::Kda { conv, active }) => {
                    copy(&a.conv, conv)?;
                    copy(&a.states[n % 2], active)?;
                }
                (Attention::Dsa(_), SavedLayer::Dsa { keys, gates }) => {
                    copy(&self.sessions[self.active].index[j].keys, keys)?;
                    copy(&self.sessions[self.active].index[j].gates, gates)?;
                    j += 1;
                }
                _ => anyhow::bail!("checkpoint layer layout"),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_and_partial_prefixes_cannot_enter_pure_namespace() {
        assert!(Key::new("a", &[], false).is_err());
        assert!(Key::new("a", &[7; 63], false).is_err());
        assert!(Key::new("a", &[7; 64], true).is_err());
        assert!(Key::new("a", &[-1; 64], false).is_err());
        assert!(Key::new("a", &[154880; 64], false).is_err());
        assert!(Key::new("a", &[7; 64], false).is_ok());
    }
    #[test]
    fn restore_rejects_foreign_changed_truncated_and_exact_without_head() {
        let mut key = Key::new("a", &[7; 64], false).unwrap();
        assert!(key.validate("a", &[7; 65], 1).is_ok());
        assert!(key.validate("b", &[7; 65], 1).is_err());
        assert!(key.validate("a", &[7; 63], 1).is_err());
        assert!(key.validate("a", &[7; 64], 1).is_err());
        assert!(key.validate("a", &[8; 65], 1).is_err());
        assert!(key.validate("a", &[7; 65], 0).is_err());
        assert!(key.validate("a", &[7; 65], MAX_OUTPUT + 1).is_err());
        key.arithmetic = "different";
        assert!(key.validate("a", &[7; 65], 1).is_err());
    }
    #[test]
    fn maximum_total_context_and_valid_suffix() {
        let key = Key::new("a", &[7; 64], false).unwrap();
        let mut request = vec![7; CONTEXT - MAX_OUTPUT];
        assert!(key.validate("a", &request, MAX_OUTPUT).is_ok());
        request.push(7);
        assert!(key.validate("a", &request, MAX_OUTPUT).is_err());
        request.truncate(65);
        request[64] = 154880;
        assert!(key.validate("a", &request, 1).is_err());
    }
}
