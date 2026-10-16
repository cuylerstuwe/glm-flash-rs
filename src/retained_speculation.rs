//! Optional Long-owned speculative state for serving and diagnostics.
//! Local work joins the caller's all-rank work/cache boundary.
pub(in crate::model) use super::verifier::accepted_path::{DiscardProof, ParserFailure};
use super::{
    Long,
    store::Control,
    verifier::{Decision, Verifier},
};
use crate::{
    accepted_features::{Phase as FeaturePhase, STORAGE_BYTES},
    cache_pages::{CONTEXT, SLOTS},
    cuda::Buffer,
    model::drafter::{Drafter, backlog::Retained, snapshot::Snapshot},
    sampling::Sampling,
    target_taps::Taps,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{path::Path, rc::Rc};

const VERSION: &[u8; 8] = b"ALDRF001";
const WINDOW: usize = 2047;
const PER_BUFFER_ROW: usize = 2 * 128 * 2;
const BUFFERS: usize = 10;

#[path = "retained_speculation_probe.rs"]
mod probe;
pub use probe::run;
#[path = "exact_prompt.rs"]
mod exact_prompt;

fn payload_bytes(cursor: usize) -> Result<usize> {
    ensure!(
        cursor > 0 && cursor < CONTEXT && cursor.is_multiple_of(64),
        "speculation snapshot requires a pure-prefix cursor"
    );
    Ok(cursor.min(WINDOW) * PER_BUFFER_ROW * BUFFERS)
}

fn validate_metadata(bytes: &[u8; 64], identity: &[u8; 32], cursor: usize) -> Result<()> {
    ensure!(
        &bytes[..8] == VERSION && &bytes[8..40] == identity,
        "speculation snapshot version/model mismatch"
    );
    ensure!(
        u64::from_le_bytes(bytes[40..48].try_into().unwrap()) == cursor as u64
            && u64::from_le_bytes(bytes[48..56].try_into().unwrap())
                == cursor.saturating_sub(WINDOW) as u64
            && u64::from_le_bytes(bytes[56..64].try_into().unwrap())
                == payload_bytes(cursor)? as u64,
        "speculation snapshot cursor/window/payload mismatch"
    );
    Ok(())
}

/// Pure dispatch validation, before any rank can enter proposal/target
/// collectives. None means DFlash; Some is a bounded target verification input.
#[allow(clippy::too_many_arguments)]
fn dispatch_digest(
    slot: usize,
    cursor: usize,
    prompt_end: usize,
    total_end: usize,
    pending: Option<i32>,
    policy: Sampling,
    tokens: Option<&[i32]>,
) -> Result<[u8; 32]> {
    let pending = pending.context("missing speculative pending token")?;
    ensure!(
        slot < SLOTS
            && cursor > 0
            && cursor >= prompt_end
            && cursor < total_end
            && total_end <= CONTEXT
            && (0..crate::sampling::VOCAB as i32).contains(&pending),
        "speculative dispatch owner/cursor/pending bounds"
    );
    ensure!(
        Sampling::new(policy.seed, policy.temperature, policy.top_k, policy.top_p)? == policy,
        "speculative canonical policy bounds"
    );
    crate::candidates::payload_bytes(policy.local_count())?;
    if let Some(tokens) = tokens {
        ensure!(
            (1..=crate::kda_window::MAX_ROWS).contains(&tokens.len())
                && cursor
                    .checked_add(tokens.len())
                    .is_some_and(|end| end < total_end)
                && tokens.first() == Some(&pending)
                && tokens
                    .iter()
                    .all(|t| (0..crate::sampling::VOCAB as i32).contains(t)),
            "speculative target input/output bounds"
        );
    } else {
        ensure!(cursor <= CONTEXT - 8, "speculative proposal context bound");
    }
    let mut hash = Sha256::new();
    hash.update(b"retained-dispatch-v1");
    hash.update(policy.encode());
    for n in [slot, cursor, prompt_end, total_end] {
        hash.update((n as u64).to_le_bytes());
    }
    hash.update(pending.to_le_bytes());
    hash.update([u8::from(tokens.is_some())]);
    if let Some(tokens) = tokens {
        hash.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hash.update(token.to_le_bytes());
        }
    }
    Ok(hash.finalize().into())
}

pub(in crate::model) struct Saved {
    identity: [u8; 32],
    payload: Snapshot,
}
impl Saved {
    pub(in crate::model) fn metadata(&self) -> [u8; 64] {
        let mut bytes = [0; 64];
        bytes[..8].copy_from_slice(VERSION);
        bytes[8..40].copy_from_slice(&self.identity);
        bytes[40..48].copy_from_slice(&(self.payload.end as u64).to_le_bytes());
        bytes[48..56].copy_from_slice(&(self.payload.lo as u64).to_le_bytes());
        bytes[56..64].copy_from_slice(&(self.gpu_bytes() as u64).to_le_bytes());
        bytes
    }
    pub(in crate::model) fn buffers(&self) -> &[Buffer] {
        &self.payload.buffers
    }
    pub(in crate::model) fn gpu_bytes(&self) -> usize {
        self.payload.gpu_bytes()
    }
}

pub(super) struct Owner {
    // Executables/graphs must retire before their retained pointer owners.
    verifier: Verifier,
    draft: Drafter,
    slots: Vec<Retained>,
    active: usize,
    model: String,
    identity: [u8; 32],
    device: Rc<crate::cuda::Device>,
}
impl Owner {
    fn new(
        long: &mut Long<'_>,
        manifest: &Path,
        root: &Path,
        native: &Path,
        graph_policy: crate::verifier_graph_cache::Policy,
        commit_policy: crate::kda_commit_plan::Policy,
        dsa_window: crate::verifier_dsa_window::Policy,
    ) -> Result<Self> {
        ensure!(
            long.sessions.iter().all(|s| s.tokens.is_empty())
                && long.graphs.is_empty()
                && long.engine.taps.is_none(),
            "enable speculation before target graph/cursor creation"
        );
        let model =
            crate::drafter_weights::open(manifest, long.engine.rank as u64)?.model_identity()?;
        let mut hash = Sha256::new();
        hash.update(VERSION);
        hash.update(model.as_bytes());
        let identity = hash.finalize().into();
        long.engine.taps = Some(Taps::new(&long.engine.device, native, 2048)?);
        let verifier =
            Verifier::new_retained(long, root, native, graph_policy, commit_policy, dsa_window)?;
        let mut draft = Drafter::new(&long.engine, manifest, root, native)?;
        let slots = (0..SLOTS)
            .map(|slot| Retained::new(&long.engine.device, slot, 0))
            .collect::<Result<Vec<_>>>()?;
        for slot in 0..SLOTS {
            draft.bind(&long.engine, slot)?;
            draft.start_context(&long.engine, 0)?;
        }
        draft.bind(&long.engine, 0)?;
        Ok(Self {
            verifier,
            draft,
            slots,
            active: 0,
            model,
            identity,
            device: long.engine.device.clone(),
        })
    }
    fn bind(&mut self, long: &Long<'_>, slot: usize) -> Result<()> {
        ensure!(slot < self.slots.len(), "speculation slot bound");
        if self.active != slot {
            self.draft.bind(&long.engine, slot)?;
            self.active = slot;
        }
        Ok(())
    }
    fn reset(&mut self, long: &Long<'_>) -> Result<()> {
        self.bind(long, long.active)?;
        let slot = &mut self.slots[long.active];
        slot.replace(slot.lease(), 0, || {
            self.draft.start_context(&long.engine, 0)
        })?;
        if slot.status().phase == FeaturePhase::Paused
            && long.memory.book().slot(long.active)?.phase() != crate::cache_pages::Phase::Paused
        {
            slot.resume(slot.lease())?;
        }
        Ok(())
    }
    fn flush(&mut self, long: &Long<'_>, slot: usize) -> Result<()> {
        self.bind(long, slot)?;
        let owner = &mut self.slots[slot];
        owner.flush(&long.engine, &mut self.draft, owner.lease())?;
        Ok(())
    }
    fn prefill(&mut self, long: &Long<'_>, position: usize, rows: usize) -> Result<()> {
        let slot = long.active;
        ensure!(
            rows > 0
                && rows <= long.prefill_rows
                && position <= CONTEXT - rows
                && self.slots[slot].status().end == position,
            "speculation prefill cursor"
        );
        self.flush(long, slot)?;
        let end = position + rows;
        // Every layer captured the same final 2048 absolute feature rows. A
        // large slab needs only the final 2047 for DFlash's sliding window.
        if rows >= WINDOW {
            self.draft.start_context(&long.engine, end - WINDOW)?;
        }
        self.draft.append_history(
            &long.engine,
            long.engine
                .taps
                .as_ref()
                .context("speculation prefill taps")?,
            end,
        )?;
        let owner = &mut self.slots[slot];
        owner.replace(owner.lease(), end, || Ok(()))?;
        Ok(())
    }
    fn serial(&mut self, long: &Long<'_>, position: usize) -> Result<()> {
        let slot = long.active;
        self.bind(long, slot)?;
        ensure!(
            self.slots[slot].status().end == position,
            "speculation serial cursor"
        );
        if self.slots[slot].available(self.slots[slot].lease())? == 0 {
            self.flush(long, slot)?;
        }
        let taps = long
            .engine
            .taps
            .as_ref()
            .context("speculation serial taps")?;
        let owner = &mut self.slots[slot];
        owner.append(
            &long.engine,
            owner.lease(),
            position,
            &taps.rows,
            position % taps.cap,
            1,
        )
    }
    fn capture(&mut self, long: &Long<'_>) -> Result<Saved> {
        let slot = long.active;
        let cursor = long.sessions[slot].tokens.len();
        ensure!(
            !long.sessions[slot].mixed && self.slots[slot].status().end == cursor,
            "speculation snapshot requires complete pure target prefix"
        );
        let bytes = payload_bytes(cursor)?;
        self.flush(long, slot)?;
        ensure!(
            self.slots[slot].snapshot_cursor(self.slots[slot].lease())? == cursor,
            "speculation snapshot backlog cursor"
        );
        let payload = self.draft.capture_context(&long.engine)?;
        ensure!(
            payload.end == cursor
                && payload.lo == cursor.saturating_sub(WINDOW)
                && payload.gpu_bytes() == bytes,
            "speculation snapshot valid window"
        );
        Ok(Saved {
            identity: self.identity,
            payload,
        })
    }
    fn restore(
        &mut self,
        long: &Long<'_>,
        slot: usize,
        saved: &Saved,
        cursor: usize,
    ) -> Result<()> {
        validate_metadata(&saved.metadata(), &self.identity, cursor)?;
        ensure!(
            saved.payload.model == self.model,
            "speculation snapshot prepared model"
        );
        let previous = self.active;
        self.bind(long, slot)?;
        let owner = &mut self.slots[slot];
        owner.replace(owner.lease(), cursor, || {
            self.draft.restore_context(&long.engine, &saved.payload)
        })?;
        // A background SSD restore must not change the foreground target's
        // active slot or leave the shared drafter bound to the background owner.
        self.bind(long, previous)
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.device.sync();
    }
}

impl Long<'_> {
    pub(super) fn multi_greedy_step(&mut self) -> Result<Vec<i32>> {
        self.ready()?;
        let positions: [usize; 4] = std::array::from_fn(|i| self.sessions[i].tokens.len());
        let inputs: [i32; 4] = std::array::from_fn(|i| self.sessions[i].pending.unwrap_or(-1));
        for i in 0..4 {
            ensure!(
                self.memory.book().slot(i)?.phase() == crate::cache_pages::Phase::Active
                    && !self.sessions[i].background
                    && positions[i] > 0
                    && positions[i] < 512
                    && positions[i] + 1 < self.sessions[i].total_end
                    && inputs[i] >= 0,
                "multi request bounds"
            );
            self.memory.map(&self.engine.stream, i, positions[i] + 1)?;
            unsafe {
                self.sessions[i]
                    .position
                    .set_i32(&self.engine.stream, positions[i] as i32)?;
            }
        }
        self.healthy = false;
        let outputs = self
            .with_speculation(|owner, long| {
                for (slot, &position) in positions.iter().enumerate() {
                    ensure!(
                        owner.slots[slot].status().end == position,
                        "multi feature cursor"
                    );
                    if owner.slots[slot].available(owner.slots[slot].lease())? == 0 {
                        owner.flush(long, slot)?;
                    }
                }
                let result = owner.verifier.multi_greedy(long, &positions)?;
                let features = &owner.verifier.accepted_taps()?.rows;
                for (slot, &position) in positions.iter().enumerate() {
                    let retained = &mut owner.slots[slot];
                    retained.append(&long.engine, retained.lease(), position, features, slot, 1)?;
                }
                Ok(result)
            })?
            .context("multi speculation owner absent")?;
        for (slot, &token) in outputs.iter().enumerate() {
            self.sessions[slot].tokens.push(inputs[slot]);
            self.sessions[slot].mixed = true;
            self.sessions[slot].pending = Some(token);
        }
        self.latest_logits = None;
        self.healthy = true;
        Ok(outputs)
    }
    #[allow(dead_code)] // Greedy diagnostic entry point.
    pub(super) fn multi_speculative_step(
        &mut self,
        control: &Control,
        digest: [u8; 32],
        slots: &[usize],
        inputs: &[Vec<i32>],
        stop_eos: &[bool],
        clip: impl FnMut(usize, &[i32]) -> Result<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        let policy = Sampling::new(731, 0., 20, 0.95)?;
        self.multi_policy_step(
            control,
            digest,
            slots,
            inputs,
            stop_eos,
            &vec![policy; slots.len()],
            clip,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn multi_policy_step(
        &mut self,
        control: &Control,
        digest: [u8; 32],
        slots: &[usize],
        inputs: &[Vec<i32>],
        stop_eos: &[bool],
        policies: &[Sampling],
        clip: impl FnMut(usize, &[i32]) -> Result<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        let window = inputs.first().map_or(0, Vec::len);
        let bounds = (|| {
            self.ready()?;
            ensure!(
                matches!(slots.len(), 2..=4)
                    && matches!(window, 2 | 3 | 4 | 8)
                    && slots.len() * window <= crate::kda_window::MAX_ROWS
                    && inputs.len() == slots.len()
                    && stop_eos.len() == slots.len()
                    && policies.len() == slots.len()
                    && policies.iter().all(|p| p.greedy() == policies[0].greedy())
                    && (policies[0].greedy()
                        || policies
                            .iter()
                            .all(|p| p.local_count() == policies[0].local_count()))
                    && slots.iter().all(|&slot| slot < 4)
                    && slots
                        .iter()
                        .enumerate()
                        .all(|(i, slot)| !slots[..i].contains(slot)),
                "multi plan bounds"
            );
            let owner = self.speculation.as_ref().context("multi owner")?;
            for (&slot, tokens) in slots.iter().zip(inputs) {
                let session = &self.sessions[slot];
                let pos = session.tokens.len();
                ensure!(
                    self.memory.book().slot(slot)?.phase() == crate::cache_pages::Phase::Active
                        && !session.background
                        && pos > 0
                        && pos.checked_add(window).is_some_and(|end| end <= CONTEXT)
                        && (pos >= 2051 || pos + window <= 2051)
                        && pos + window < session.total_end
                        && tokens.len() == window
                        && tokens[0] == session.pending.unwrap_or(-1)
                        && tokens
                            .iter()
                            .all(|&t| (0..crate::sampling::VOCAB as i32).contains(&t))
                        && owner.slots[slot].status().end == pos
                        && owner.slots[slot].available(owner.slots[slot].lease())? >= window,
                    "multi ownership/input/feature bounds"
                );
            }
            Ok::<_, anyhow::Error>(())
        })();
        self.agree_speculation_work(control, 909, digest, bounds)?;
        let allocated =
            self.with_speculation(|owner, long| owner.verifier.prepare_multi_spec(long));
        self.agree_speculation_work(control, 907, digest, allocated)?;
        let mapped = (|| -> Result<()> {
            for (&slot, tokens) in slots.iter().zip(inputs) {
                let session = &self.sessions[slot];
                let pos = session.tokens.len();
                ensure!(
                    self.memory.book().slot(slot)?.phase() == crate::cache_pages::Phase::Active
                        && !session.background
                        && pos > 0
                        && pos.checked_add(window).is_some_and(|end| end <= CONTEXT)
                        && (pos >= 2051 || pos + window <= 2051)
                        && pos + window < session.total_end
                        && tokens.len() == window
                        && tokens[0] == session.pending.unwrap_or(-1),
                    "multi ownership/bounds"
                );
                self.memory.map(&self.engine.stream, slot, pos + window)?;
                unsafe {
                    self.sessions[slot]
                        .position
                        .set_i32(&self.engine.stream, pos as i32)?;
                }
            }
            Ok(())
        })();
        self.agree_speculation_work(control, 910, digest, mapped)?;
        let mut plan = Sha256::new();
        plan.update(b"multi-target-plan-v1");
        plan.update(digest);
        for p in policies {
            plan.update(p.encode());
        }
        for (&slot, tokens) in slots.iter().zip(inputs) {
            plan.update((slot as u64).to_le_bytes());
            plan.update((self.sessions[slot].tokens.len() as u64).to_le_bytes());
            plan.update(
                (self.kda_phys(slot, self.sessions[slot].tokens.len()) as u64).to_le_bytes(),
            );
            for token in tokens {
                plan.update(token.to_le_bytes());
            }
        }
        ensure!(
            control.agree(self, 908, plan.finalize().into(), true)?,
            "multi target inputs disagreed"
        );
        self.healthy = false;
        let output = self
            .with_speculation(|owner, long| {
                let result = owner
                    .verifier
                    .multi_speculative(long, slots, inputs, stop_eos, policies, clip)?;
                let features = &owner.verifier.accepted_taps()?.rows;
                for (row, (&slot, tokens)) in slots.iter().zip(&result).enumerate() {
                    let retained = &mut owner.slots[slot];
                    ensure!(
                        retained.status().end == long.sessions[slot].tokens.len(),
                        "multi feature cursor"
                    );
                    if !tokens.is_empty() {
                        retained.append(
                            &long.engine,
                            retained.lease(),
                            retained.status().end,
                            features,
                            row * window,
                            tokens.len(),
                        )?;
                    }
                }
                Ok(result)
            })?
            .context("multi owner missing")?;
        for (row, (&slot, tokens)) in slots.iter().zip(&output).enumerate() {
            if !tokens.is_empty() {
                self.sessions[slot]
                    .tokens
                    .extend_from_slice(&inputs[row][..tokens.len()]);
                self.sessions[slot].mixed = true;
                self.sessions[slot].pending = tokens.last().copied();
            }
        }
        self.engine.sync()?;
        self.latest_logits = None;
        self.healthy = true;
        Ok(output)
    }
    /// Explicit construction seam only: current serving does not call it.
    /// The caller coordinates startup errors before admitting requests.
    pub(super) fn enable_speculation(
        &mut self,
        manifest: &Path,
        root: &Path,
        native: &Path,
    ) -> Result<()> {
        let (graph_policy, commit_policy, dsa_window) = Verifier::diagnostic_policies(self)?;
        self.enable_speculation_with_policies(
            manifest,
            root,
            native,
            graph_policy,
            commit_policy,
            dsa_window,
        )
    }
    pub(super) fn enable_speculation_with_policies(
        &mut self,
        manifest: &Path,
        root: &Path,
        native: &Path,
        graph_policy: crate::verifier_graph_cache::Policy,
        commit_policy: crate::kda_commit_plan::Policy,
        dsa_window: crate::verifier_dsa_window::Policy,
    ) -> Result<()> {
        self.ready()?;
        ensure!(self.speculation.is_none(), "speculation already enabled");
        self.healthy = false;
        self.speculation = Some(Owner::new(
            self,
            manifest,
            root,
            native,
            graph_policy,
            commit_policy,
            dsa_window,
        )?);
        Ok(())
    }
    fn with_speculation<T>(
        &mut self,
        op: impl FnOnce(&mut Owner, &mut Self) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(mut owner) = self.speculation.take() else {
            return Ok(None);
        };
        let result = op(&mut owner, self);
        // Always put the retained pointer owners back, even after local failure.
        // The Long health flag prevents uncertain state from being run again.
        self.speculation = Some(owner);
        if result.is_err() {
            self.healthy = false;
        }
        result.map(Some)
    }
    pub(crate) fn prewarm_speculation_graphs(&mut self) -> Result<usize> {
        let before = self.engine.device.memory()?.0;
        let warmed = self
            .with_speculation(|owner, long| {
                let warmed = owner.verifier.prewarm(long)?;
                let multi = match std::env::var("ALASKA_MULTI_GREEDY").as_deref() {
                    Ok("2") => owner.verifier.prewarm_multi(long, 4)?,
                    Ok("3") => owner.verifier.prewarm_multi(long, 8)?,
                    Ok("4") => owner.verifier.prewarm_multi(long, 3)?,
                    Ok("5") => owner.verifier.prewarm_multi(long, 2)?,
                    Ok("6") => owner.verifier.prewarm_multi(long, 4)?,
                    _ => 0,
                };
                Ok(warmed + multi)
            })?
            .unwrap_or(0);
        let after = self.engine.device.memory()?.0;
        eprintln!(
            "verifier graphs prewarmed: {warmed} graphs, {} MiB",
            before.saturating_sub(after) >> 20
        );
        Ok(warmed)
    }
    pub(super) fn speculation_identity(&self) -> Option<[u8; 32]> {
        self.speculation.as_ref().map(|owner| owner.identity)
    }
    pub(super) fn speculation_snapshot_bytes(&self, cursor: usize) -> Result<usize> {
        if self.speculation.is_some() {
            payload_bytes(cursor)
        } else {
            Ok(0)
        }
    }
    pub(super) fn capture_speculation(&mut self) -> Result<Option<Saved>> {
        self.with_speculation(|owner, long| owner.capture(long))
    }
    pub(super) fn validate_speculation_metadata(
        &self,
        metadata: &[u8; 64],
        cursor: usize,
    ) -> Result<()> {
        let owner = self
            .speculation
            .as_ref()
            .context("speculation snapshot without enabled owner")?;
        validate_metadata(metadata, &owner.identity, cursor)
    }
    pub(super) fn allocate_speculation_restore(
        &self,
        metadata: &[u8; 64],
        cursor: usize,
    ) -> Result<Saved> {
        self.validate_speculation_metadata(metadata, cursor)?;
        let owner = self.speculation.as_ref().unwrap();
        let bytes = payload_bytes(cursor)? / BUFFERS;
        Ok(Saved {
            identity: owner.identity,
            payload: Snapshot {
                device: self.engine.device.clone(),
                model: owner.model.clone(),
                end: cursor,
                lo: cursor.saturating_sub(WINDOW),
                buffers: (0..BUFFERS)
                    .map(|_| self.engine.device.alloc(bytes))
                    .collect::<Result<_>>()?,
            },
        })
    }
    pub(super) fn restore_speculation_at(
        &mut self,
        slot: usize,
        saved: Option<&Saved>,
        cursor: usize,
    ) -> Result<()> {
        ensure!(
            slot < SLOTS && self.speculation.is_some() == saved.is_some(),
            "target/drafter snapshot presence or slot mismatch"
        );
        if let Some(saved) = saved {
            self.with_speculation(|owner, long| owner.restore(long, slot, saved, cursor))?;
        }
        Ok(())
    }
    pub(super) fn reset_speculation(&mut self) -> Result<()> {
        self.with_speculation(|owner, long| owner.reset(long))?;
        Ok(())
    }
    pub(super) fn prefill_speculation(&mut self, position: usize, rows: usize) -> Result<()> {
        self.with_speculation(|owner, long| owner.prefill(long, position, rows))?;
        Ok(())
    }
    pub(super) fn serial_speculation(&mut self, position: usize) -> Result<()> {
        self.with_speculation(|owner, long| owner.serial(long, position))?;
        Ok(())
    }
    /// Include this in the existing native owner/work digest. No tensor readback.
    pub(super) fn speculation_digest(&self) -> Result<[u8; 32]> {
        let mut h = Sha256::new();
        h.update(VERSION);
        if let Some(owner) = &self.speculation {
            h.update([1]);
            // Prepared shard identities are rank-local. They belong in the
            // ordered group cache identity, not an equal-on-every-rank digest.
            for slot in &owner.slots {
                let s = slot.status();
                let draft_end = owner.draft.context_end(s.slot)?;
                ensure!(
                    draft_end == s.start,
                    "retained drafter/backlog cursor mismatch"
                );
                if matches!(
                    self.memory.book().slot(s.slot)?.phase(),
                    crate::cache_pages::Phase::Active | crate::cache_pages::Phase::Paused
                ) {
                    ensure!(
                        s.end == self.sessions[s.slot].tokens.len(),
                        "retained target/backlog cursor mismatch"
                    );
                }
                let target = &self.sessions[s.slot];
                h.update(target.pending.unwrap_or(-1).to_le_bytes());
                h.update((target.tokens.len() as u64).to_le_bytes());
                h.update((target.prompt_end as u64).to_le_bytes());
                h.update((target.total_end as u64).to_le_bytes());
                h.update([u8::from(target.background), u8::from(target.mixed)]);
                for n in [
                    s.slot as u64,
                    s.epoch,
                    s.start as u64,
                    s.end as u64,
                    draft_end as u64,
                ] {
                    h.update(n.to_le_bytes());
                }
                h.update([match s.phase {
                    FeaturePhase::Ready => 0,
                    FeaturePhase::Paused => 1,
                    FeaturePhase::Poisoned => 2,
                }]);
            }
        } else {
            h.update([0]);
        }
        Ok(h.finalize().into())
    }
    /// Local preparation is a separate coordinated boundary before any DFlash
    /// proposal collectives. A caller must not `prepare()?; propose()?` on ranks.
    pub(super) fn prepare_speculation(&mut self, rows: usize, draft: bool) -> Result<()> {
        self.ready_active()?;
        ensure!(
            (1..=crate::kda_window::MAX_ROWS).contains(&rows),
            "speculation preparation rows"
        );
        self.healthy = false;
        self.with_speculation(|owner, long| {
            let slot = long.active;
            owner.bind(long, slot)?;
            ensure!(
                owner.slots[slot].status().end == long.sessions[slot].tokens.len(),
                "speculation preparation cursor"
            );
            if draft || owner.slots[slot].available(owner.slots[slot].lease())? < rows {
                owner.flush(long, slot)?;
            }
            Ok(())
        })?
        .context("speculation owner unavailable")?;
        Ok(())
    }
    /// A local proposal completed. The verifier's dispatch agreement follows
    /// before any other collective or GPU work and confirms it on every rank.
    pub(super) fn drafted(&mut self) {
        self.healthy = true;
    }
    pub(super) fn propose_speculation(
        &mut self,
        control: &Control,
        dispatch: [u8; 32],
        policy: Sampling,
    ) -> Result<crate::draft_chain::Proposal> {
        let pending = self.prepare_dispatch(control, dispatch, policy, None)?;
        self.healthy = false;
        self.with_speculation(|owner, long| {
            owner.draft.propose_scored(&long.engine, pending, policy)
        })?
        .context("speculation owner unavailable")
    }
    pub(super) fn verify_speculation(
        &mut self,
        control: &Control,
        dispatch: [u8; 32],
        tokens: &[i32],
        policy: Sampling,
        clip: impl FnOnce(&[i32]) -> Result<usize>,
    ) -> Result<Vec<i32>> {
        self.verify_speculation_eos(control, dispatch, tokens, policy, true, clip)
    }
    pub(super) fn verify_speculation_eos(
        &mut self,
        control: &Control,
        dispatch: [u8; 32],
        tokens: &[i32],
        policy: Sampling,
        stop_eos: bool,
        clip: impl FnOnce(&[i32]) -> Result<usize>,
    ) -> Result<Vec<i32>> {
        // Agree the exact EOS mode before any target verifier collectives,
        // independently of the serving caller's already-bound queue digest.
        let mut mode = Sha256::new();
        mode.update(b"retained-verification-eos-v1");
        mode.update(dispatch);
        mode.update([u8::from(stop_eos)]);
        let dispatch = mode.finalize().into();
        self.prepare_dispatch(control, dispatch, policy, Some(tokens))?;
        let result = self.with_speculation(|owner, long| {
            let slot = long.active;
            let position = long.sessions[slot].tokens.len();
            let accepted = owner.verifier.execute_with_eos(
                long,
                tokens,
                Decision::Sampled(policy),
                stop_eos,
                clip,
            )?;
            crate::round_timing::enter(crate::round_timing::Phase::VerifyPost);
            long.healthy = false;
            let taps = owner.verifier.accepted_taps()?;
            let backlog = &mut owner.slots[slot];
            backlog.append(
                &long.engine,
                backlog.lease(),
                position,
                &taps.rows,
                0,
                accepted.keep,
            )?;
            Ok(accepted.sampled[..accepted.keep].to_vec())
        });
        self.healthy = false;
        result?.context("speculation owner unavailable")
    }
    /// Only an all-rank, pre-publication parser rejection grants this recovery.
    /// The rejected slot can contain speculative scratch writes, so drain and
    /// discard it without advertising health or replaying any accepted output.
    /// The caller must coordinate this local result before publishing an error.
    pub(super) fn discard_rejected_output(&mut self, proof: &DiscardProof) -> Result<()> {
        ensure!(
            !self.healthy
                && proof.matches(self.active, self.sessions[self.active].tokens.len())
                && self.memory.book().slot(self.active)?.phase()
                    == crate::cache_pages::Phase::Active,
            "output discard proof does not match uncommitted owner"
        );
        self.engine.sync()?;
        self.memory.release(&self.engine.stream, self.active, 0)?;
        let session = &mut self.sessions[self.active];
        session.tokens.clear();
        session.pending = None;
        session.mixed = false;
        session.prompt_end = 0;
        session.total_end = 0;
        self.latest_logits = None;
        self.reset_speculation()?;
        self.sync_speculation_holds()?;
        Ok(())
    }
    /// This is the dispatch agreement, not another append/flush agreement.
    /// Every fallible input/owner/capacity check stays a value until all peers
    /// arrive. The immediately following GPU call cannot observe a changed
    /// owner/input/policy between validation and launch.
    fn prepare_dispatch(
        &mut self,
        control: &Control,
        dispatch: [u8; 32],
        policy: Sampling,
        tokens: Option<&[i32]>,
    ) -> Result<i32> {
        let prepared = (|| {
            self.ready_active()?;
            let target = &self.sessions[self.active];
            let hash = dispatch_digest(
                self.active,
                target.tokens.len(),
                target.prompt_end,
                target.total_end,
                target.pending,
                policy,
                tokens,
            )?;
            let pending = target.pending.unwrap();
            let rows = tokens.map_or(8, <[i32]>::len);
            self.prepare_speculation(rows, tokens.is_none())?;
            let owner = self
                .speculation
                .as_ref()
                .context("speculative dispatch owner")?;
            ensure!(
                owner.active == self.active
                    && owner.slots[self.active].status().end
                        == self.sessions[self.active].tokens.len(),
                "speculative dispatch cursor"
            );
            if tokens.is_none() {
                ensure!(
                    owner.draft.end() == self.sessions[self.active].tokens.len(),
                    "speculative unflushed proposal"
                );
            } else {
                ensure!(
                    owner.slots[self.active].available(owner.slots[self.active].lease())? >= rows,
                    "speculative feature capacity"
                );
            }
            Ok::<_, anyhow::Error>((pending, hash))
        })();
        let mut hash = Sha256::new();
        hash.update(dispatch);
        hash.update(prepared.as_ref().map(|p| p.1).unwrap_or([0; 32]));
        Ok(self
            .agree_speculation_work(control, 851, hash.finalize().into(), prepared)?
            .0)
    }
    /// Replace the caller's existing work agreement with this helper; do not add
    /// a second per-round collective. Local failures stay values until every
    /// rank reaches the same native boundary. Only common success restores health.
    pub(super) fn agree_speculation_work<T>(
        &mut self,
        control: &Control,
        tag: u32,
        digest: [u8; 32],
        work: Result<T>,
    ) -> Result<T> {
        self.healthy = false;
        if let Err(error) = &work {
            eprintln!("work {tag} failed locally: {error:#}");
        }
        let state = self.speculation_digest();
        let mut agreed = Sha256::new();
        agreed.update(b"retained-work-v1");
        agreed.update(digest);
        agreed.update(state.as_ref().copied().unwrap_or([0; 32]));
        ensure!(
            control.agree(
                self,
                tag,
                agreed.finalize().into(),
                work.is_ok() && state.is_ok()
            )?,
            "speculation work failed collectively"
        );
        let result = work?;
        state?;
        self.healthy = true;
        Ok(result)
    }
    /// Called after the existing native drain and before its hold receipt. No
    /// additional GPU work is needed; retained features are not replayed/flushed.
    pub(super) fn sync_speculation_holds(&mut self) -> Result<()> {
        self.with_speculation(|owner, long| {
            for (slot, features) in owner.slots.iter_mut().enumerate() {
                let phase = long.memory.book().slot(slot)?.phase();
                // A cooperative restore has no runnable owner yet. Preserve
                // its retained hold until the target's final publication.
                if phase == crate::cache_pages::Phase::Restoring {
                    continue;
                }
                let paused = phase == crate::cache_pages::Phase::Paused
                    || (phase == crate::cache_pages::Phase::Active
                        && long.sessions[slot].background
                        && long.priority_hold);
                match (paused, features.status().phase) {
                    (true, FeaturePhase::Ready) => features.pause(features.lease(), || Ok(()))?,
                    (false, FeaturePhase::Paused) => features.resume(features.lease())?,
                    (_, FeaturePhase::Poisoned) => {
                        anyhow::bail!("failed retained speculative slot")
                    }
                    _ => {}
                }
            }
            Ok(())
        })?;
        Ok(())
    }
    pub(super) fn speculation_backlog_bytes(&self) -> usize {
        if self.speculation.is_some() {
            SLOTS * STORAGE_BYTES
        } else {
            0
        }
    }
    pub(super) fn speculation_backlog_rows(&self, slot: usize) -> Result<usize> {
        let owner = self
            .speculation
            .as_ref()
            .context("speculation owner unavailable")?;
        let features = owner
            .slots
            .get(slot)
            .context("speculation slot bound")?
            .status();
        Ok(features.end - features.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatch_rejects_incomplete_inputs_and_hashes_exact_owner_policy() -> Result<()> {
        let policy = Sampling::new(731, 1., 20, 0.95)?;
        let valid = dispatch_digest(0, 128, 128, 160, Some(17), policy, Some(&[17, 18, 19]))?;
        ensure!(
            dispatch_digest(0, 128, 128, 160, None, policy, Some(&[17, 18, 19])).is_err(),
            "missing pending"
        );
        for input in [
            vec![],
            vec![18],
            vec![17, -1],
            vec![17, 154880],
            vec![17; 17],
        ] {
            assert!(dispatch_digest(0, 128, 128, 160, Some(17), policy, Some(&input)).is_err());
        }
        assert!(dispatch_digest(0, 128, 129, 160, Some(17), policy, Some(&[17])).is_err());
        assert!(dispatch_digest(0, 128, 128, 131, Some(17), policy, Some(&[17; 3])).is_err());
        assert!(dispatch_digest(0, CONTEXT - 7, 128, CONTEXT, Some(17), policy, None).is_err());
        let greedy = dispatch_digest(
            0,
            128,
            128,
            160,
            Some(17),
            Sampling::new(731, 0., 20, 0.95)?,
            Some(&[17]),
        )?;
        assert_ne!(greedy, valid);
        let negative_zero = dispatch_digest(
            0,
            128,
            128,
            160,
            Some(17),
            Sampling::new(731, -0., 20, 0.95)?,
            Some(&[17]),
        )?;
        assert_ne!(negative_zero, greedy);
        for invalid in [
            Sampling {
                temperature: f64::NAN,
                ..policy
            },
            Sampling {
                top_p: f64::INFINITY,
                ..policy
            },
            Sampling {
                top_k: crate::sampling::VOCAB + 1,
                ..policy
            },
        ] {
            assert!(dispatch_digest(0, 128, 128, 160, Some(17), invalid, Some(&[17])).is_err());
        }
        for changed in [
            dispatch_digest(1, 128, 128, 160, Some(17), policy, Some(&[17, 18, 19]))?,
            dispatch_digest(0, 128, 128, 160, Some(17), policy, Some(&[17, 18, 20]))?,
            dispatch_digest(
                0,
                128,
                128,
                160,
                Some(17),
                Sampling::new(732, 1., 20, 0.95)?,
                Some(&[17, 18, 19]),
            )?,
            dispatch_digest(0, 128, 128, 160, Some(17), policy, None)?,
        ] {
            assert_ne!(valid, changed);
        }
        Ok(())
    }
    #[test]
    fn prefix_metadata_is_exact_bounded_and_rejects_changed_fields() -> Result<()> {
        let identity = [37; 32];
        for cursor in [64usize, 2048, 8192, CONTEXT - 64] {
            let mut bytes = [0; 64];
            bytes[..8].copy_from_slice(VERSION);
            bytes[8..40].copy_from_slice(&identity);
            bytes[40..48].copy_from_slice(&(cursor as u64).to_le_bytes());
            bytes[48..56].copy_from_slice(&(cursor.saturating_sub(WINDOW) as u64).to_le_bytes());
            bytes[56..64].copy_from_slice(&(payload_bytes(cursor)? as u64).to_le_bytes());
            validate_metadata(&bytes, &identity, cursor)?;
            assert_eq!(payload_bytes(cursor)?, cursor.min(2047) * 5120);
            for at in [0usize, 8, 40, 48, 56] {
                let mut changed = bytes;
                changed[at] ^= 1;
                assert!(validate_metadata(&changed, &identity, cursor).is_err());
            }
            assert!(validate_metadata(&bytes, &identity, cursor + 64).is_err());
        }
        for cursor in [0usize, 1, 63, 65, CONTEXT, usize::MAX] {
            assert!(payload_bytes(cursor).is_err());
        }
        Ok(())
    }
}
