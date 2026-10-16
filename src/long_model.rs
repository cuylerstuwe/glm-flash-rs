//! Four-slot model adapter with long-context state and retained request ownership.
use super::{Attention, Engine, batch::Inputs};
use crate::{
    cache_pages::{Admission, CONTEXT, MAX_OUTPUT, Phase, SLOTS},
    context_memory::ContextMemory,
    cuda::{Buffer, Graph, Kernel},
    dsa::words,
    front::load_kernel,
    index_projection::Projection,
    prepared::Prepared,
    sparse_attention::SparseAttention,
    sparse_select::Select,
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, net::SocketAddr, path::Path, time::Instant};
const ROWS: usize = 512;
const SELECT_ROWS: usize = 64;
#[path = "prefill_head.rs"]
mod prefill_head;
#[path = "prefill_partial_model.rs"]
mod prefill_partial_model;
#[path = "prefill_row_model.rs"]
mod prefill_row_model;
#[path = "sparse_split.rs"]
mod split;
pub use prefill_head::run as run_prefill_head;
pub use prefill_partial_model::run as run_prefill_partial;
pub use prefill_row_model::run as run_prefill_row;
#[path = "model_verifier.rs"]
mod verifier;
pub use verifier::run as run_verifier;
pub use verifier::run_draft as run_speculative;
pub use verifier::run_sampled as run_verifier_sampled;
pub use verifier::run_workloads as run_speculative_workloads;
#[path = "ssd_model.rs"]
mod disk;
#[path = "prefix_checkpoint.rs"]
mod prefix;
#[path = "native_priority.rs"]
mod priority;
#[path = "request_scheduler.rs"]
mod requests;
pub use requests::run as run_requests;
pub use requests::run_cached_requests;
pub use requests::run_disk_requests;
pub use requests::run_owner;
pub use requests::run_sampled_requests;
pub use requests::run_termination;
#[path = "conversation_reuse.rs"]
mod reuse;
#[path = "model_sampling.rs"]
mod sampling;
#[path = "model_slots.rs"]
mod slots;
pub use priority::run as run_priority;
pub use sampling::run as run_sampling;
#[path = "retained_speculation.rs"]
pub(super) mod speculation;
pub use speculation::run as run_retained_speculation;
#[path = "session_store.rs"]
mod store;
pub use disk::run as run_ssd;
pub use disk::run_controller as run_ssd_controller;
pub use disk::run_cooperative as run_ssd_cooperative;
pub use disk::run_recovery as run_ssd_recovery;
pub use disk::run_restore as run_ssd_restore;
pub use reuse::run as run_reuse;
pub use store::run as run_store;
struct Long<'a> {
    copy_gate: Option<std::sync::Weak<crate::copy_gate::Gate>>,
    graphs: BTreeMap<(usize, usize, usize), Graph>,
    speculation: Option<speculation::Owner>,
    engine: Engine<'a>,
    memory: ContextMemory,
    sessions: Vec<slots::Session>,
    active: usize,
    prefill_rows: usize,
    latest_logits: Option<usize>,
    projection: Projection,
    select: Select,
    split: Option<split::Owner>,
    split_identity: [u8; 32],
    scores: Buffer,
    attention: SparseAttention,
    dense: Kernel,
    dense_rows: Kernel,
    merges: Vec<Kernel>,
    po: Buffer,
    pm: Buffer,
    pl: Buffer,
    healthy: bool,
    priority_hold: bool,
    model_identity: String,
}
impl<'a> Long<'a> {
    fn new(mut engine: Engine<'a>, root: &Path) -> Result<Self> {
        engine.sync()?;
        engine.graphs = None;
        let prefill_rows = engine.batch.as_ref().map_or(ROWS, |b| b.max_rows);
        let attention_rows = engine.batch.as_ref().map_or(ROWS, |b| b.attention_rows);
        let d = &engine.device;
        let s = &engine.stream;
        let memory = ContextMemory::new(d, s, 11)?;
        let sessions = (0..SLOTS)
            .map(|slot| {
                slots::Session::new(d, root, &memory, &engine.attention, slot, attention_rows)
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = Projection::new(d, root, attention_rows)?;
        let select = Select::new(d, root, SELECT_ROWS, CONTEXT / 4)?;
        let scores = d.alloc(SELECT_ROWS * (CONTEXT / 4) * 4)?;
        let attention = SparseAttention::new(d, root, SELECT_ROWS)?;
        let dense = load_kernel(d, root, "prefill_latent_chunks32")?;
        let dense_rows = load_kernel(d, root, "latent_chunks_rows")?;
        let merges = (1..=5)
            .map(|n| load_kernel(d, root, &format!("latent_merge{n}")))
            .collect::<Result<Vec<_>>>()?;
        let po = d.alloc(5 * attention_rows * 16 * 512 * 4)?;
        let pm = d.alloc(5 * attention_rows * 16 * 4)?;
        let pl = d.alloc(5 * attention_rows * 16 * 4)?;
        let model_identity = prefix::model_identity(engine.p)?;
        let (split, split_identity) = split::Owner::from_env(&engine)?;
        let mut result = Self {
            copy_gate: None,
            graphs: BTreeMap::new(),
            speculation: None,
            engine,
            memory,
            sessions,
            active: 0,
            prefill_rows,
            latest_logits: None,
            projection,
            select,
            split,
            split_identity,
            scores,
            attention,
            dense,
            dense_rows,
            merges,
            po,
            pm,
            pl,
            healthy: true,
            priority_hold: false,
            model_identity,
        };
        result.bind(0)?;
        Ok(result)
    }
    fn split_receipt(&self) -> serde_json::Value {
        serde_json::json!({"enabled":self.split.is_some(),
            "identity":self.split_identity.iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "stats":self.split.as_ref().map(|p|p.stats())})
    }
    fn ready(&self) -> Result<()> {
        ensure!(
            self.healthy,
            "long model is poisoned after incomplete GPU operation"
        );
        Ok(())
    }
    fn admission(
        &self,
        prompt: usize,
        output: usize,
        keep: usize,
        background: bool,
    ) -> Result<Admission> {
        self.ready()?;
        ensure!(
            !matches!(
                self.memory.book().slot(self.active)?.phase(),
                Phase::Paused | Phase::Restoring
            ),
            "cannot replace a paused/restoring session"
        );
        ensure!(
            !background || !self.priority_hold,
            "background admission held by native priority"
        );
        // Validate the entire transition before releasing any live reservation.
        let mut proposed = self.memory.book().clone();
        if proposed.slot(self.active)?.phase() != Phase::Idle {
            proposed.release(self.active, keep)?;
        }
        proposed.plan(self.active, prompt, output, keep, background)
    }
    fn commit_admission(&mut self, plan: &Admission, keep: usize) -> Result<()> {
        let s = &self.engine.stream;
        if self.memory.book().slot(self.active)?.phase() != Phase::Idle {
            self.memory.release(s, self.active, keep)?;
        }
        self.memory.commit(s, plan)
    }
    fn start(&mut self, prompt: usize, output: usize) -> Result<()> {
        self.start_mode(prompt, output, false)
    }
    fn start_mode(&mut self, prompt: usize, output: usize, background: bool) -> Result<()> {
        let plan = self.admission(prompt, output, 0, background)?;
        self.healthy = false;
        self.commit_admission(&plan, 0)?;
        let s = &self.engine.stream;
        for a in &self.engine.attention {
            if let Attention::Kda(a) = a {
                unsafe {
                    a.conv.zero_prefix(s, a.conv.bytes)?;
                    for state in &a.states {
                        state.zero_prefix(s, state.bytes)?;
                    }
                }
            }
        }
        for index in &self.sessions[self.active].index {
            unsafe {
                index.keys.zero_prefix(s, index.keys.bytes)?;
                index.gates.zero_prefix(s, index.gates.bytes)?;
            }
        }
        unsafe {
            self.sessions[self.active].position.set_i32(s, 0)?;
        }
        self.engine.sync()?;
        self.sessions[self.active].tokens.clear();
        self.sessions[self.active].mixed = false;
        self.sessions[self.active].pending = None;
        self.sessions[self.active].prompt_end = prompt;
        self.sessions[self.active].total_end = prompt + output;
        self.sessions[self.active].background = background;
        self.latest_logits = None;
        self.reset_speculation()?;
        self.healthy = self.speculation.is_none();
        Ok(())
    }
    fn attend(&self, input: Inputs<'_>, position: usize, fast: bool) -> Result<()> {
        let e = &self.engine;
        let s = &e.stream;
        let rows = input.rows;
        self.projection.run(
            s,
            e.p,
            &e.m,
            input.layer,
            input.norm,
            input.qr,
            rows,
            position,
            fast,
        )?;
        self.attend_projected(
            input,
            position,
            fast,
            self.projection.raw.ptr,
            self.projection.gate.ptr,
            self.projection.query.ptr,
        )
    }
    /// Projection storage may contain one or several rows. Decode's dense ABI
    /// is fixed to one row; wholly sparse windows use the runtime-row kernels.
    fn attend_projected(
        &self,
        input: Inputs<'_>,
        position: usize,
        fast: bool,
        raw: u64,
        gate: u64,
        query: u64,
    ) -> Result<()> {
        self.attend_projected_slot(input, position, fast, raw, gate, query, self.active)
    }
    #[allow(clippy::too_many_arguments)]
    fn attend_projected_slot(
        &self,
        input: Inputs<'_>,
        position: usize,
        fast: bool,
        raw: u64,
        gate: u64,
        query: u64,
        slot: usize,
    ) -> Result<()> {
        let e = &self.engine;
        let s = &e.stream;
        let index = &self.sessions[slot].index[input.layer / 4];
        let rows = input.rows;
        ensure!(
            fast || rows == 1 || position >= 2051 || position + rows <= 2051,
            "multirow mixed dense/sparse decode attention has no compatible ABI"
        );
        index.update(
            s,
            e.p,
            &e.m,
            input.layer,
            raw,
            gate,
            rows,
            position,
            &self.sessions[slot].position,
        )?;
        if position < 2051 {
            let chunks = (position + rows).min(2051).div_ceil(512);
            // BM32 is the retained fast-prefill tile. Decode keeps Dsa's BM16
            // entry and its fixed-R ABI, including at the dense/sparse boundary.
            if fast {
                words(
                    &self.dense,
                    s,
                    [rows.div_ceil(2) as u32, chunks as u32, 1],
                    &[
                        input.qa,
                        input.dsa.cache.ptr,
                        input.dsa.cache.ptr,
                        self.sessions[slot].position.ptr,
                        self.po.ptr,
                        self.pm.ptr,
                        self.pl.ptr,
                        rows as u64,
                        self.memory.table(slot)?.ptr,
                    ],
                )?;
            } else if rows == 1 {
                let k = &input.dsa.kernels["latent_chunks"];
                words(
                    k,
                    s,
                    [1, chunks as u32, 1],
                    &[
                        input.qa,
                        input.dsa.cache.ptr,
                        input.dsa.cache.ptr,
                        self.sessions[slot].position.ptr,
                        self.po.ptr,
                        self.pm.ptr,
                        self.pl.ptr,
                        self.memory.table(slot)?.ptr,
                    ],
                )?;
            } else {
                // A dense verifier window: the same BM16 kernel with R rows, one
                // 16-query tile per row and row r seeing keys through position + r
                // during causal verification; each row equals its one-row call.
                words(
                    &self.dense_rows,
                    s,
                    [rows as u32, chunks as u32, 1],
                    &[
                        input.qa,
                        input.dsa.cache.ptr,
                        input.dsa.cache.ptr,
                        self.sessions[slot].position.ptr,
                        self.po.ptr,
                        self.pm.ptr,
                        self.pl.ptr,
                        rows as u64,
                        self.memory.table(slot)?.ptr,
                    ],
                )?;
            }
            words(
                &self.merges[chunks - 1],
                s,
                [(rows * 16) as u32, 1, 1],
                &[
                    self.po.ptr,
                    self.pm.ptr,
                    self.pl.ptr,
                    input.attended,
                    (rows * 16) as u64,
                ],
            )?;
        }
        if position + rows > 2051 {
            let np = ((position + rows) / 4)
                .next_power_of_two()
                .clamp(1024, CONTEXT / 4);
            if let Some(plan) = split::plan::Plan::new(self.split.is_some(), fast, position, rows)?
            {
                let owner = self.split.as_ref().unwrap();
                let owned = plan.own(e.rank)?;
                for offset in owned.clone().step_by(SELECT_ROWS) {
                    let count = (owned.end - offset).min(SELECT_ROWS);
                    unsafe {
                        self.sessions[slot]
                            .position
                            .set_i32(s, (position + offset) as i32)?;
                    }
                    index.score(
                        s,
                        &self.sessions[slot].position,
                        query + (offset * 4096 * 2) as u64,
                        raw + ((offset * 160 + 128) * 2) as u64,
                        160,
                        &self.scores,
                        count,
                        np,
                        128,
                    )?;
                    self.select.pools_into(
                        s,
                        &self.scores,
                        &self.sessions[slot].position,
                        count,
                        np,
                        &owner.send,
                        (offset - owned.start) * split::plan::ROW_BYTES,
                    )?;
                }
                // The common plan determines the wire stride. Only live IDs are
                // addressed below; each rank's remaining packed bytes are padding.
                unsafe {
                    e.gather_prefix(&owner.send, &owner.recv, plan.stride)?;
                }
                owner.gathered(plan, e.rank)?;
                for offset in (0..rows).step_by(SELECT_ROWS) {
                    let count = (rows - offset).min(SELECT_ROWS);
                    let (at, _) = plan.region(offset, count)?;
                    unsafe {
                        self.sessions[slot]
                            .position
                            .set_i32(s, (position + offset) as i32)?;
                    }
                    self.select.expand_from(
                        s,
                        &owner.recv,
                        at,
                        &self.sessions[slot].position,
                        count,
                    )?;
                    self.attention.run(
                        s,
                        input.qa + (offset * 16 * 512 * 2) as u64,
                        input.dsa.cache.ptr,
                        self.memory.table(slot)?.ptr,
                        self.select.tokens.ptr,
                        self.select.counts.ptr,
                        input.attended + (offset * 16 * 512 * 4) as u64,
                        count,
                        fast,
                    )?;
                }
                unsafe {
                    self.sessions[slot].position.set_i32(s, position as i32)?;
                }
                return Ok(());
            }
            for offset in (0..rows).step_by(SELECT_ROWS) {
                let count = (rows - offset).min(SELECT_ROWS);
                if offset > 0 {
                    unsafe {
                        self.sessions[slot]
                            .position
                            .set_i32(s, (position + offset) as i32)?;
                    }
                }
                index.score(
                    s,
                    &self.sessions[slot].position,
                    query + (offset * 4096 * 2) as u64,
                    raw + ((offset * 160 + 128) * 2) as u64,
                    160,
                    &self.scores,
                    count,
                    np,
                    128,
                )?;
                self.select
                    .run(s, &self.scores, &self.sessions[slot].position, count, np)?;
                self.attention.run(
                    s,
                    input.qa + (offset * 16 * 512 * 2) as u64,
                    input.dsa.cache.ptr,
                    self.memory.table(slot)?.ptr,
                    self.select.tokens.ptr,
                    self.select.counts.ptr,
                    input.attended + (offset * 16 * 512 * 4) as u64,
                    count,
                    fast,
                )?;
            }
            if rows > SELECT_ROWS {
                unsafe {
                    self.sessions[slot].position.set_i32(s, position as i32)?;
                }
            }
        }
        Ok(())
    }
    fn compute(&self, position: usize) -> Result<()> {
        self.engine.compute_with(position % 2, |layer, a, norm| {
            a.run_with(
                &self.engine.stream,
                self.engine.p,
                &self.engine.m,
                layer,
                norm,
                || {
                    self.engine.prefetch_mark(
                        layer,
                        crate::l2_prefetch_plan::Site::AttentionCore,
                        1,
                    )
                },
                |a| {
                    self.attend(
                        Inputs {
                            layer,
                            position,
                            norm,
                            qr: a.qr.ptr,
                            qa: a.qa.ptr,
                            attended: a.attended.ptr,
                            rows: 1,
                            dsa: a,
                        },
                        position,
                        false,
                    )
                },
            )
        })
    }
    fn prefill(&mut self, position: usize, tokens: &[i32]) -> Result<i32> {
        self.prefill_profiled(position, tokens, None)
    }
    fn prefill_profiled(
        &mut self,
        position: usize,
        tokens: &[i32],
        trace: Option<&mut crate::phase_profile::Trace>,
    ) -> Result<i32> {
        self.prefill_output(position, tokens, trace, true)?
            .ok_or_else(|| anyhow::anyhow!("diagnostic prefill head missing"))
    }
    /// Service-only candidate: retain every intermediate model boundary without
    /// computing an output head that the request cannot consume yet.
    fn prefill_request(&mut self, position: usize, tokens: &[i32]) -> Result<Option<i32>> {
        let emit_head =
            position.checked_add(tokens.len()) == Some(self.sessions[self.active].prompt_end);
        self.prefill_output(position, tokens, None, emit_head)
    }
    fn prefill_output(
        &mut self,
        position: usize,
        tokens: &[i32],
        trace: Option<&mut crate::phase_profile::Trace>,
        emit_head: bool,
    ) -> Result<Option<i32>> {
        self.ready_active()?;
        ensure!(
            position == self.sessions[self.active].tokens.len()
                && position.is_multiple_of(64)
                && !tokens.is_empty()
                && tokens.len() <= self.prefill_rows
                && position + tokens.len() <= self.sessions[self.active].prompt_end
                && (emit_head || position + tokens.len() < self.sessions[self.active].prompt_end)
                && tokens.iter().all(|&t| crate::vision_input::prompt_token(t)),
            "long prefill position/token bounds"
        );
        self.healthy = false;
        self.memory
            .map(&self.engine.stream, self.active, position + tokens.len())?;
        if let Some(taps) = &self.engine.taps {
            taps.set_position(&self.engine.stream, position)?;
        }
        let next = self
            .engine
            .batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batch unavailable"))?
            .run_profiled_output(
                &self.engine,
                position,
                tokens,
                CONTEXT,
                |input| {
                    let position = input.position;
                    self.attend(input, position, true)
                },
                trace,
                emit_head,
            )?;
        self.prefill_speculation(position, tokens.len())?;
        self.sessions[self.active].tokens.extend_from_slice(tokens);
        self.sessions[self.active].pending =
            if self.sessions[self.active].tokens.len() == self.sessions[self.active].prompt_end {
                next
            } else {
                None
            };
        // A skipped projection must never expose another cursor's stale logits.
        self.latest_logits = next.map(|_| self.active);
        self.healthy = self.speculation.is_none();
        Ok(next)
    }
    fn step(&mut self, position: usize, token: i32) -> Result<i32> {
        self.ready_active()?;
        ensure!(
            position == self.sessions[self.active].tokens.len()
                && position + 1 < self.sessions[self.active].total_end
                && self.sessions[self.active].pending == Some(token)
                && (0..154880).contains(&token),
            "long model step bounds"
        );
        self.forward_token(position, token)
    }
    /// Exact row-invariant inference for one forced prompt input. Unlike step,
    /// this does not consume output budget or require the sampled pending ID.
    fn exact_prompt_token(&mut self, token: i32) -> Result<i32> {
        self.ready_active()?;
        let position = self.sessions[self.active].tokens.len();
        ensure!(
            position < self.sessions[self.active].prompt_end
                && crate::vision_input::prompt_token(token),
            "exact prompt bounds"
        );
        self.forward_token(position, token)
    }
    fn forward_token(&mut self, position: usize, token: i32) -> Result<i32> {
        self.healthy = false;
        // An image row embeds row 0 and is overwritten by its vision row; it
        // runs eagerly so captured token graphs never contain the overwrite.
        let (ids, map) = self.engine.vision_ids(&[token])?;
        let image = !map.is_empty();
        if image {
            self.engine.vision_map.write(
                0,
                &map.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
        }
        self.memory
            .map(&self.engine.stream, self.active, position + 1)?;
        unsafe {
            self.sessions[self.active]
                .position
                .set_i32(&self.engine.stream, position as i32)?;
            self.engine.id.set_i32(&self.engine.stream, ids[0])?;
        }
        if let Some(taps) = &self.engine.taps {
            taps.set_position(&self.engine.stream, position)?;
        }
        if image {
            self.engine.vision_token.set(true);
            let computed = self.compute(position);
            self.engine.vision_token.set(false);
            computed?;
        } else {
            let shape = if position < 2051 {
                (position + 1).div_ceil(512)
            } else {
                ((position + 1) / 4).next_power_of_two().max(1024)
            };
            let key = (self.active, shape, self.kda_phys(self.active, position));
            if !self.graphs.contains_key(&key) {
                let graph = unsafe { self.engine.stream.capture(|| self.compute(position))? };
                self.graphs.insert(key, graph);
            }
            unsafe {
                self.graphs[&key].launch(&self.engine.stream)?;
            }
        }
        self.engine.sync()?;
        let next = self.engine.selection.read()?;
        self.serial_speculation(position)?;
        self.sessions[self.active].tokens.push(token);
        self.sessions[self.active].mixed = true;
        self.sessions[self.active].pending = (self.sessions[self.active].tokens.len()
            >= self.sessions[self.active].prompt_end)
            .then_some(next);
        self.latest_logits = Some(self.active);
        self.healthy = self.speculation.is_none();
        Ok(next)
    }
    fn logits(&self) -> Result<String> {
        self.ready()?;
        ensure!(
            self.latest_logits == Some(self.active),
            "logits belong to another session or cursor"
        );
        Ok(format!(
            "{:x}",
            Sha256::digest(
                self.engine
                    .all_logits
                    .read(0, self.engine.all_logits.bytes)?
            )
        ))
    }
    fn state(&self, position: usize) -> Result<String> {
        self.ready()?;
        ensure!(
            position == self.sessions[self.active].tokens.len(),
            "state cursor mismatch"
        );
        self.engine.sync()?;
        let mut h = Sha256::new();
        let snapshot = self
            .memory
            .pack(&self.engine.stream, self.active, position)?;
        self.engine.sync()?;
        let mut j = 0;
        for a in &self.engine.attention {
            match a {
                Attention::Kda(a) => {
                    h.update(a.conv.read(0, a.conv.bytes)?);
                    h.update(a.states[position % 2].read(0, a.states[position % 2].bytes)?);
                }
                Attention::Dsa(_) => {
                    h.update(snapshot.families[j].latent.read(0, position * 528)?);
                    h.update(
                        snapshot.families[j]
                            .pooled
                            .read(0, (position / 4) * 128 * 2)?,
                    );
                    let index = &self.sessions[self.active].index[j];
                    h.update(index.keys.read(0, index.keys.bytes)?);
                    h.update(index.gates.read(0, index.gates.bytes)?);
                    j += 1;
                }
            }
        }
        Ok(format!("{:x}", h.finalize()))
    }
}
impl Drop for Long<'_> {
    fn drop(&mut self) {
        if let Some(gate) = self.copy_gate.as_ref().and_then(std::sync::Weak::upgrade) {
            gate.cancel();
        }
        // Drain before graphs and all buffers (including pinned token storage) drop.
        let _ = self.engine.stream.sync();
    }
}
#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    ensure!(
        !token_sets.is_empty() && new_tokens > 0 && new_tokens <= MAX_OUTPUT,
        "long harness shape"
    );
    for tokens in token_sets {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= CONTEXT - new_tokens
                && tokens.iter().all(|t| (0..154880).contains(t)),
            "long context/token bounds"
        );
    }
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(token_sets[0][0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let mut cases = Vec::new();
    for tokens in token_sets {
        eprintln!("long model parity prompt_rows={}", tokens.len());
        long.start(tokens.len(), new_tokens)?;
        let started = Instant::now();
        let mut next = 0;
        for (chunk, ids) in tokens.chunks(ROWS).enumerate() {
            next = long.prefill(chunk * ROWS, ids)?;
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let prefill_state = long.state(tokens.len())?;
        let mut generated = vec![next];
        let mut hashes = vec![long.logits()?];
        while generated.len() < new_tokens && ![154820, 154827, 154828, 154829].contains(&next) {
            next = long.step(tokens.len() + generated.len() - 1, next)?;
            generated.push(next);
            hashes.push(long.logits()?);
        }
        let final_state = long.state(tokens.len() + generated.len() - 1)?;
        long.start(tokens.len(), new_tokens)?;
        for (chunk, ids) in tokens.chunks(64).enumerate() {
            next = long.prefill(chunk * 64, ids)?;
        }
        ensure!(
            next == generated[0] && long.logits()? == hashes[0],
            "long prefill logits partition mismatch"
        );
        ensure!(
            long.state(tokens.len())? == prefill_state,
            "long prefill state partition mismatch"
        );
        for (i, token) in generated.iter().enumerate().skip(1) {
            next = long.step(tokens.len() + i - 1, next)?;
            ensure!(
                next == *token && long.logits()? == hashes[i],
                "long decode partition mismatch step={i}"
            );
        }
        ensure!(
            long.state(tokens.len() + generated.len() - 1)? == final_state,
            "long final state partition mismatch"
        );
        let prefix_reuse = if tokens.len() >= 4096 {
            let prefix = (tokens.len() / 2 / 64) * 64 + 64;
            long.start(tokens.len(), new_tokens)?;
            for (chunk, ids) in tokens[..prefix].chunks(ROWS).enumerate() {
                long.prefill(chunk * ROWS, ids)?;
            }
            let prefix_state = long.state(prefix)?;
            let began = Instant::now();
            let checkpoint = long.checkpoint()?;
            let capture_seconds = began.elapsed().as_secs_f64();
            // Destroy the current live state with an unrelated request. The
            // snapshot must own every byte needed by a future continuation.
            long.start(64, new_tokens)?;
            long.prefill(0, &[7; 64])?;
            let clobbered = long.state(64)?;
            let mut wrong = tokens.clone();
            wrong[0] = (wrong[0] + 1) % 154880;
            ensure!(
                long.restore_prefix(&checkpoint, &wrong, new_tokens)
                    .is_err(),
                "changed prefix accepted"
            );
            ensure!(
                long.start(CONTEXT, 1).is_err(),
                "oversized request accepted"
            );
            ensure!(
                long.state(64)? == clobbered,
                "invalid restore/admission changed state"
            );
            let began = Instant::now();
            let reused = long.restore_prefix(&checkpoint, tokens, new_tokens)?;
            let restore_seconds = began.elapsed().as_secs_f64();
            ensure!(
                reused == prefix && long.state(prefix)? == prefix_state,
                "restored prefix state mismatch"
            );
            for (chunk, ids) in tokens[prefix..].chunks(ROWS).enumerate() {
                next = long.prefill(prefix + chunk * ROWS, ids)?;
            }
            ensure!(
                next == generated[0] && long.logits()? == hashes[0],
                "restored prefill logits mismatch"
            );
            ensure!(
                long.state(tokens.len())? == prefill_state,
                "restored prefill state mismatch"
            );
            for (i, token) in generated.iter().enumerate().skip(1) {
                next = long.step(tokens.len() + i - 1, next)?;
                ensure!(
                    next == *token && long.logits()? == hashes[i],
                    "restored decode mismatch step={i}"
                );
            }
            ensure!(
                long.state(tokens.len() + generated.len() - 1)? == final_state,
                "restored final state mismatch"
            );
            if generated.len() > 1 {
                ensure!(
                    long.checkpoint().is_err(),
                    "mixed decode entered pure prefix namespace"
                );
            }
            Some(
                serde_json::json!({"reused_model_tokens":reused,"owned_payload_bytes":checkpoint.bytes,"capture_seconds":capture_seconds,"restore_seconds":restore_seconds,"clobber_restore_state_exact":true,"cold_vs_restored_logits_and_state_exact":true,"invalid_request_preserves_state":true}),
            )
        } else {
            None
        };
        cases.push(serde_json::json!({"prompt_tokens":tokens.len(),"generated_tokens":generated,"per_step_logits_sha256":hashes,"prefill_seconds":prefill_seconds,"prefill_state_sha256":prefill_state,"final_active_state_sha256":final_state,"chunks_512_vs_64_exact":true,"prefix_reuse":prefix_reuse}));
    }
    let interleaved = slots::interleave(&mut long, token_sets, &cases, new_tokens)?;
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce {"roce"} else {"nccl"},"l2_prefetch":long.engine.prefetch_receipt(),"select_split":long.split_receipt(),"cases":cases,"interleaved":interleaved,"scope":"Experimental four-slot long-context integration; audit readback, greedy only, no serving or production speed/quality claim."}),
    )
}

pub use requests::run_restored_requests;

pub use requests::serve;

/// Bounded diagnostic: normal eager prefill wall timing, then one independently
/// replayed final slab with phase events. Hashes/copies are outside timing.
#[allow(clippy::too_many_arguments)]
pub fn run_profile(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[512, 512, 512],
        &[],
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_profile_batches(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[512, 1024, 1024, 512],
        &[],
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_profile_large_batches(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[512, 2048, 8192, 8192, 2048, 512],
        &[],
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_profile_hc_slabs(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[8192, 8192, 8192, 8192],
        &[false, true, true, false],
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_profile_overlap(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[8192, 8192, 8192, 8192],
        &[true, true, true, true],
        &[false, true, true, false],
        &[],
    )
}
#[allow(clippy::too_many_arguments)]
pub fn run_profile_hc_pipeline(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    profile_inner(
        p,
        rank,
        root,
        native,
        nccl,
        master,
        token_sets,
        new_tokens,
        use_roce,
        &[8192, 8192, 8192, 8192],
        &[true, true, true, true],
        &[true, true, true, true],
        &[false, true, true, false],
    )
}
#[allow(clippy::too_many_arguments)]
fn profile_inner(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    new_tokens: usize,
    use_roce: bool,
    sizes: &[usize],
    hc_slabs: &[bool],
    overlap: &[bool],
    hc_pipeline: &[bool],
) -> Result<serde_json::Value> {
    ensure!(
        !sizes.is_empty()
            && (hc_slabs.is_empty() || hc_slabs.len() == sizes.len())
            && (overlap.is_empty() || overlap.len() == sizes.len())
            && (hc_pipeline.is_empty() || hc_pipeline.len() == sizes.len()),
        "profile configurations"
    );
    ensure!(
        !token_sets.is_empty() && new_tokens > 0 && new_tokens <= 256,
        "profile shape"
    );
    for ids in token_sets {
        ensure!(
            !ids.is_empty()
                && ids.len() <= 65536
                && ids.len().is_multiple_of(ROWS)
                && ids.iter().all(|t| (0..154880).contains(t)),
            "profile aligned prompt bounds"
        );
    }
    let mut engine = Engine::new(p, rank, root, native, nccl, master, false, use_roce)?;
    let capacity = *sizes.iter().max().unwrap();
    engine.batch = Some(super::batch::Batch::new_with_rows(
        &engine.device,
        &engine,
        root,
        native,
        capacity,
    )?);
    if overlap.iter().any(|v| *v) {
        engine
            .batch
            .as_ref()
            .unwrap()
            .prepare_pipe(&engine.device)?;
    }
    let mut long = Long::new(engine, root)?;
    let mut cases = Vec::new();
    for ids in token_sets {
        let mut repeats = Vec::new();
        for (trial, &chunk_rows) in sizes.iter().enumerate() {
            let slab = hc_slabs.get(trial).copied().unwrap_or(false);
            let piped = overlap.get(trial).copied().unwrap_or(false);
            let fused = hc_pipeline.get(trial).copied().unwrap_or(false);
            long.engine.batch.as_mut().unwrap().hc_pipeline = fused;
            long.engine.batch.as_mut().unwrap().overlap = piped;
            long.engine.batch.as_mut().unwrap().hc_slabs = slab;
            long.start(ids.len(), new_tokens)?;
            let begin = Instant::now();
            let mut next = 0;
            for (chunk, tokens) in ids.chunks(chunk_rows).enumerate() {
                next = long.prefill(chunk * chunk_rows, tokens)?;
            }
            let prefill_s = begin.elapsed().as_secs_f64();
            let state = long.state(ids.len())?;
            let first = next;
            // Warm this position's lazy graph outside the timed serial generation.
            // All generated tokens count, even EOS; this is a fixed-work diagnostic.
            let mut generated = vec![next];
            while generated.len() < new_tokens.min(3) {
                next = long.step(ids.len() + generated.len() - 1, next)?;
                generated.push(next);
            }
            let begin = Instant::now();
            while generated.len() < new_tokens {
                next = long.step(ids.len() + generated.len() - 1, next)?;
                generated.push(next);
            }
            let decode_s = begin.elapsed().as_secs_f64();
            eprintln!(
                "profile prompt={} trial={} rows={} prefill_tps={:.2} serial_tps={:.2} state={}",
                ids.len(),
                trial,
                chunk_rows,
                ids.len() as f64 / prefill_s,
                new_tokens.saturating_sub(3) as f64 / decode_s,
                state
            );
            repeats.push(serde_json::json!({"trial":trial,"batch_rows":chunk_rows,"hc_slabs":slab,"overlap":piped,"hc_pipeline":fused,"prefill_s":prefill_s,"prefill_tps":ids.len() as f64/prefill_s,"first_token":first,"prefill_state_sha256":state,"generated":generated,"serial_timed_steps":new_tokens.saturating_sub(3),"serial_s":decode_s}));
        }
        ensure!(
            repeats.windows(2).all(|r| r[0]["prefill_state_sha256"]
                == r[1]["prefill_state_sha256"]
                && r[0]["generated"] == r[1]["generated"]),
            "profile repeat mismatch"
        );
        let mut profiles = Vec::new();
        let unique: std::collections::BTreeSet<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                (
                    n,
                    hc_slabs.get(i).copied().unwrap_or(false),
                    overlap.get(i).copied().unwrap_or(false),
                    hc_pipeline.get(i).copied().unwrap_or(false),
                )
            })
            .collect();
        for (chunk_rows, slab, piped, fused) in unique {
            long.engine.batch.as_mut().unwrap().hc_pipeline = fused;
            long.engine.batch.as_mut().unwrap().overlap = piped;
            long.engine.batch.as_mut().unwrap().hc_slabs = slab;
            long.start(ids.len(), 1)?;
            for (chunk, tokens) in ids[..ids.len() - chunk_rows].chunks(chunk_rows).enumerate() {
                long.prefill(chunk * chunk_rows, tokens)?;
            }
            let pos = ids.len() - chunk_rows;
            let mut trace = crate::phase_profile::Trace::new(&long.engine.device)?;
            let token = long.prefill_profiled(pos, &ids[pos..], Some(&mut trace))?;
            ensure!(
                serde_json::json!(token) == repeats[0]["first_token"],
                "profile instrumentation changed output"
            );
            ensure!(
                serde_json::json!(long.state(ids.len())?) == repeats[0]["prefill_state_sha256"],
                "profile instrumentation changed state"
            );
            profiles.push(serde_json::json!({"batch_rows":chunk_rows,"hc_slabs":slab,"overlap":piped,"hc_pipeline":fused,"last_slab":trace.finish()?}));
        }
        cases.push(serde_json::json!({"prompt_tokens":ids.len(),"scratch_rows":capacity,"repeats":repeats,"profiles":profiles}));
    }
    long.engine.sync()?;
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"audit":false,"speculative":false,"cases":cases,"scope":"diagnostic raw model timing; excludes HTTP, sampling and snapshot writes"}),
    )
}
