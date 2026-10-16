//! Per-session model state; immutable weights and scratch remain shared.
use super::{Attention, Long, ROWS};
use crate::{
    cache_pages::{CONTEXT, Phase},
    context_memory::ContextMemory,
    cuda::{Buffer, Device},
    indexer::IndexCache,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};
pub(super) struct Recurrent {
    pub conv: Rc<Buffer>,
    pub states: [Rc<Buffer>; 2],
}
pub(super) struct Session {
    pub position: Rc<Buffer>,
    pub recurrent: Vec<Recurrent>,
    pub index: Vec<IndexCache>,
    pub tokens: Vec<i32>,
    pub mixed: bool,
    pub pending: Option<i32>,
    pub prompt_end: usize,
    pub total_end: usize,
    pub background: bool,
    /// The KDA state for position p lives in `recurrent[_].states[p % 2]`; a
    /// speculative commit may leave it in the other physical buffer and swap
    /// the two handles. Captured graphs bake physical pointers, so their keys
    /// use `kda_phys`, not `p % 2`.
    pub kda_swapped: bool,
}
impl Session {
    pub fn new(
        d: &Rc<Device>,
        root: &Path,
        memory: &ContextMemory,
        attention: &[Attention],
        slot: usize,
        prefill_rows: usize,
    ) -> Result<Self> {
        let mut recurrent = Vec::new();
        for a in attention {
            if let Attention::Kda(a) = a {
                recurrent.push(if slot == 0 {
                    Recurrent {
                        conv: a.conv.clone(),
                        states: a.states.clone(),
                    }
                } else {
                    Recurrent {
                        conv: Rc::new(d.alloc(a.conv.bytes)?),
                        states: [
                            Rc::new(d.alloc(a.states[0].bytes)?),
                            Rc::new(d.alloc(a.states[1].bytes)?),
                        ],
                    }
                });
            }
        }
        let table = memory.table(slot)?;
        let index = memory
            .families
            .iter()
            .map(|family| {
                IndexCache::shared(
                    d,
                    root,
                    CONTEXT,
                    prefill_rows,
                    family.pooled.clone(),
                    table.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            position: Rc::new(d.alloc(4)?),
            recurrent,
            index,
            tokens: Vec::new(),
            mixed: false,
            pending: None,
            prompt_end: 0,
            total_end: 0,
            background: false,
            kda_swapped: false,
        })
    }
}
impl Long<'_> {
    /// Physical recurrent buffer (0/1 in allocation order) holding position `p`.
    pub(super) fn kda_phys(&self, slot: usize, position: usize) -> usize {
        (position % 2) ^ usize::from(self.sessions[slot].kda_swapped)
    }
    /// The committed state was written into the slot's other physical buffer:
    /// exchange both parities' handles for every KDA layer, then rebind.
    pub(super) fn swap_kda_states(&mut self, slot: usize) -> Result<()> {
        ensure!(slot < self.sessions.len(), "model slot out of range");
        let session = &mut self.sessions[slot];
        for recurrent in &mut session.recurrent {
            recurrent.states.swap(0, 1);
        }
        session.kda_swapped = !session.kda_swapped;
        if slot == self.active {
            let mut k = 0;
            for a in &mut self.engine.attention {
                if let Attention::Kda(a) = a {
                    a.states = self.sessions[slot].recurrent[k].states.clone();
                    k += 1;
                }
            }
        }
        Ok(())
    }
    pub(super) fn bind(&mut self, slot: usize) -> Result<()> {
        self.ready()?;
        ensure!(slot < self.sessions.len(), "model slot out of range");
        ensure!(
            self.memory.book().slot(slot)?.phase() != Phase::Restoring,
            "cannot bind a partially restored session"
        );
        self.healthy = false;
        self.engine.sync()?;
        let session = &self.sessions[slot];
        let table = self.memory.table(slot)?;
        let (mut k, mut j) = (0, 0);
        for a in &mut self.engine.attention {
            match a {
                Attention::Kda(a) => {
                    a.conv = session.recurrent[k].conv.clone();
                    a.states = session.recurrent[k].states.clone();
                    k += 1;
                }
                Attention::Dsa(a) => {
                    a.cache = self.memory.families[j].latent.clone();
                    a.position = session.position.clone();
                    a.page_table = table.clone();
                    a.capacity = CONTEXT;
                    j += 1;
                }
            }
        }
        self.active = slot;
        self.healthy = true;
        Ok(())
    }
    pub(super) fn ready_active(&self) -> Result<()> {
        self.ready()?;
        ensure!(
            !self.priority_hold || !self.sessions[self.active].background,
            "background execution held by native priority"
        );
        ensure!(
            self.memory.book().slot(self.active)?.phase() == Phase::Active,
            "model execution requires an active session"
        );
        Ok(())
    }
    pub(super) fn pause(&mut self) -> Result<()> {
        self.ready_active()?;
        self.healthy = false;
        self.engine.sync()?;
        self.memory.pause(&self.engine.stream, self.active)?;
        self.sync_speculation_holds()?;
        self.healthy = self.speculation.is_none();
        Ok(())
    }
    pub(super) fn resume(&mut self) -> Result<()> {
        self.ready()?;
        ensure!(
            self.memory.book().slot(self.active)?.phase() == Phase::Paused,
            "resume requires a paused session"
        );
        ensure!(
            !self.priority_hold || !self.sessions[self.active].background,
            "background resume held by native priority"
        );
        self.healthy = false;
        self.memory.resume(self.active)?;
        self.sync_speculation_holds()?;
        self.healthy = self.speculation.is_none();
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<()> {
        self.ready_active()?;
        self.healthy = false;
        self.engine.sync()?;
        self.memory.release(
            &self.engine.stream,
            self.active,
            self.sessions[self.active].tokens.len(),
        )?;
        self.sync_speculation_holds()?;
        self.healthy = self.speculation.is_none();
        Ok(())
    }
    pub(super) fn cancel(&mut self) -> Result<()> {
        self.ready()?;
        ensure!(
            matches!(
                self.memory.book().slot(self.active)?.phase(),
                Phase::Active | Phase::Paused
            ),
            "cancel requires live session"
        );
        self.healthy = false;
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
        self.healthy = self.speculation.is_none();
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct Reference {
    generated_tokens: Vec<i32>,
    per_step_logits_sha256: Vec<String>,
    prefill_state_sha256: String,
    final_active_state_sha256: String,
}
/// Deliberately interleave layers' owners only at complete model-call boundaries.
/// This qualifies in-process state ownership, not HTTP priority leases or timing.
pub(super) fn interleave(
    long: &mut Long<'_>,
    prompts: &[Vec<i32>],
    cases: &[serde_json::Value],
    output: usize,
) -> Result<serde_json::Value> {
    const SLOTS: usize = crate::cache_pages::SLOTS;
    let selected = if prompts.len() >= SLOTS {
        [0, prompts.len() - 2, prompts.len() - 1, 1]
    } else {
        [0, prompts.len() - 1, 0, prompts.len() - 1]
    };
    let references = selected
        .iter()
        .map(|&i| serde_json::from_value::<Reference>(cases[i].clone()))
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!("long model interleave four sessions");
    for (slot, &which) in selected.iter().enumerate() {
        long.bind(slot)?;
        long.start_mode(prompts[which].len(), output, slot < 3)?;
        long.pause()?;
        ensure!(
            long.prefill(0, &prompts[which][..1]).is_err(),
            "paused execution accepted"
        );
        ensure!(long.start(64, 1).is_err(), "paused replacement accepted");
    }
    let mut counts = [0usize; SLOTS];
    let mut done = [false; SLOTS];
    let mut rounds = 0;
    while done.iter().any(|&v| !v) {
        rounds += 1;
        for slot in 0..SLOTS {
            if done[slot] {
                continue;
            }
            let prompt = &prompts[selected[slot]];
            let expected = &references[slot];
            long.bind(slot)?;
            long.resume()?;
            let cursor = long.sessions[slot].tokens.len();
            if cursor < prompt.len() {
                let end = (cursor + ROWS).min(prompt.len());
                let next = long.prefill(cursor, &prompt[cursor..end])?;
                if end == prompt.len() {
                    ensure!(
                        next == expected.generated_tokens[0]
                            && long.logits()? == expected.per_step_logits_sha256[0],
                        "interleaved prefill logits slot={slot}"
                    );
                    ensure!(
                        long.state(end)? == expected.prefill_state_sha256,
                        "interleaved prefill state slot={slot}"
                    );
                    counts[slot] = 1;
                }
            } else {
                let i = counts[slot];
                let pending = long.sessions[slot]
                    .pending
                    .ok_or_else(|| anyhow::anyhow!("missing pending token"))?;
                ensure!(
                    pending == expected.generated_tokens[i - 1],
                    "pending token changed while paused"
                );
                let next = long.step(cursor, pending)?;
                ensure!(
                    next == expected.generated_tokens[i]
                        && long.logits()? == expected.per_step_logits_sha256[i],
                    "interleaved decode logits slot={slot} step={i}"
                );
                counts[slot] += 1;
            }
            if counts[slot] == expected.generated_tokens.len() {
                ensure!(
                    long.state(long.sessions[slot].tokens.len())?
                        == expected.final_active_state_sha256,
                    "interleaved final state slot={slot}"
                );
                long.finish()?;
                done[slot] = true;
            } else {
                long.pause()?;
            }
        }
        long.memory.book().check()?;
    }
    // Cancellation before prefill and after an actual paused prefill both release
    // only this owner's pages. Reset/reuse must restore the same output and state.
    long.bind(0)?;
    long.start(64, 2)?;
    long.cancel()?;
    long.start(64, 2)?;
    let ids = vec![7; 64];
    let first = long.prefill(0, &ids)?;
    let logits = long.logits()?;
    let state = long.state(64)?;
    let held: Vec<_> = (1..SLOTS)
        .map(|i| long.memory.book().slot(i).unwrap().pages().to_vec())
        .collect();
    long.pause()?;
    long.cancel()?;
    ensure!(
        long.memory.book().slot(0)?.pages().is_empty() && long.sessions[0].pending.is_none(),
        "cancel retained live request"
    );
    for (slot, pages) in (1..SLOTS).zip(&held) {
        ensure!(
            long.memory.book().slot(slot)?.pages() == pages,
            "cancel released another owner's pages"
        );
    }
    long.start(64, 2)?;
    ensure!(
        long.prefill(0, &ids)? == first && long.logits()? == logits && long.state(64)? == state,
        "cancelled slot failed exact reuse"
    );
    long.step(64, first)?;
    let next = long.sessions[0].pending.unwrap();
    ensure!(
        long.step(65, next).is_err(),
        "request exceeded output limit"
    );
    long.finish()?;
    long.memory.book().check()?;
    Ok(
        serde_json::json!({"slots":SLOTS,"prompt_tokens":selected.map(|i|prompts[i].len()),"generated_tokens":references.iter().map(|r| &r.generated_tokens).collect::<Vec<_>>(),"rounds":rounds,"prefill_and_decode_logits_and_state_exact":true,"paused_execution_and_replacement_rejected":true,"cancel_and_reuse_exact":true,"output_limit_enforced":true,"scope":"In-process model ownership and drained pause/resume; no native lease/HTTP/SSD acknowledgment protocol yet."}),
    )
}
