//! Bounded pure-prefix GPU storage. Independent of live slots; restore is copy-in.
use super::{
    Engine, Long, ROWS,
    prefix::{ARITHMETIC, Key, SavedLayer},
};
use crate::{
    cache_pages::{PAGE, Phase},
    context_memory::Family,
    cuda::{Buffer, Device, Pinned},
    prepared::Prepared,
    session_index::{EXTENT, Index, Lease, Status},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap, net::SocketAddr, path::Path, rc::Rc};

const LATENT: usize = PAGE * 528;
const POOLED: usize = PAGE / 4 * 128 * 2;
const RAM_BUDGET: u64 = 16 << 30;
const RESERVE: u64 = 2 << 30;
type Hash = [u8; 32];
#[path = "session_export.rs"]
mod export;
pub(super) use export::CopyPlan;
pub(super) use export::Export;
pub(super) struct Control {
    send: Buffer,
    recv: Buffer,
    host: RefCell<Pinned>,
    /// Device address of `host`.
    mapped: u64,
}
impl Control {
    pub(super) fn new(d: &Rc<Device>) -> Result<Self> {
        let host = Pinned::new(d, 320)?;
        let mapped = host.device_ptr()?;
        Ok(Self {
            send: d.alloc(64)?,
            recv: d.alloc(256)?,
            host: RefCell::new(host),
            mapped,
        })
    }
    pub(super) fn exchange(&self, long: &Long<'_>, record: &[u8; 64]) -> Result<Vec<[u8; 64]>> {
        let started = std::time::Instant::now();
        let result = self.exchange_inner(long, record);
        crate::round_timing::exchange(
            u32::from_le_bytes(record[32..36].try_into().unwrap()),
            started.elapsed(),
        );
        result
    }
    fn exchange_inner(&self, long: &Long<'_>, record: &[u8; 64]) -> Result<Vec<[u8; 64]>> {
        long.engine.sync()?;
        let mut host = self.host.borrow_mut();
        host.bytes_mut()[..64].copy_from_slice(record);
        // RoCE reads the record from and writes the gathered records to the
        // page-locked buffer directly, without the two copies per agreement.
        if !unsafe {
            long.engine
                .gather_mapped(self.mapped, self.mapped + 64, 64)?
        } {
            unsafe {
                host.upload_to(&long.engine.stream, &self.send, 0, 0, 64)?;
                long.engine.gather_prefix(&self.send, &self.recv, 64)?;
                host.download_from(&long.engine.stream, 64, &self.recv, 0, 256)?;
            }
        }
        long.engine.sync()?;
        Ok(host.bytes_mut()[64..320].as_chunks::<64>().0.to_vec())
    }
    /// False means an agreed operation cannot fit on at least one rank. A
    /// disagreement is an error, never a local-only eviction or restore.
    pub(super) fn agree(
        &self,
        long: &Long<'_>,
        tag: u32,
        digest: Hash,
        ready: bool,
    ) -> Result<bool> {
        let mut record = [0; 64];
        record[..32].copy_from_slice(&digest);
        record[32..36].copy_from_slice(&tag.to_le_bytes());
        record[36] = u8::from(ready);
        let all = self.exchange(long, &record)?;
        ensure!(
            all.iter().all(|r| r[..36] == record[..36]),
            "session rank decision/digest disagreement"
        );
        Ok(all.iter().all(|r| r[36] == 1))
    }
}
struct Snapshot {
    key: Key,
    layers: Vec<SavedLayer>,
    tail: Vec<Family>,
    draft: Option<super::speculation::Saved>,
}
impl Snapshot {
    fn gpu_bytes(&self) -> usize {
        self.layers.iter().map(SavedLayer::bytes).sum::<usize>()
            + family_bytes(&self.tail)
            + self.draft.as_ref().map_or(0, |s| s.gpu_bytes())
    }
}
fn family_bytes(families: &[Family]) -> usize {
    families
        .iter()
        .map(|f| f.latent.bytes + f.pooled.bytes)
        .sum()
}

/// Coalesce only where both source and destination are contiguous and the
/// destination stays in one allocated slab. Logical pages may skip shared ones.
fn runs(placed: &[(usize, usize)], live: &[u32]) -> Result<Vec<(usize, usize, usize)>> {
    let mut result: Vec<(usize, usize, usize)> = Vec::new();
    for &(logical, dst) in placed {
        let src = *live
            .get(logical)
            .ok_or_else(|| anyhow::anyhow!("snapshot source page missing"))?
            as usize;
        if let Some((previous_src, previous_dst, n)) = result.last_mut()
            && src == *previous_src + *n
            && dst == *previous_dst + *n
            && dst / EXTENT == *previous_dst / EXTENT
        {
            *n += 1;
        } else {
            result.push((src, dst, 1));
        }
    }
    Ok(result)
}

pub(super) struct Store {
    copy_gate: Option<std::sync::Weak<crate::copy_gate::Gate>>,
    device: Rc<Device>,
    stream: usize,
    model: String,
    speculation: Option<Hash>,
    control: Control,
    index: Index,
    slabs: BTreeMap<usize, Vec<Family>>,
    snapshots: BTreeMap<u64, Snapshot>,
    healthy: bool,
}
impl Drop for Store {
    fn drop(&mut self) {
        if let Some(gate) = self.copy_gate.as_ref().and_then(std::sync::Weak::upgrade) {
            gate.cancel();
        }
        let _ = self.device.sync();
    }
}
#[derive(Debug)]
pub(super) struct Saved {
    status: Status,
    pub(super) id: Option<u64>,
    pub(super) evicted: Vec<u64>,
    new_pages: usize,
    copy_runs: usize,
}
impl Store {
    pub(super) fn contains(&self, id: u64) -> bool {
        self.snapshots.contains_key(&id)
    }
    fn pin(&mut self, long: &mut Long<'_>, id: u64) -> Result<Option<Lease>> {
        self.ready(long)?;
        let mut h = Sha256::new();
        h.update(self.index.digest());
        h.update(id.to_le_bytes());
        let digest = h.finalize().into();
        let valid = self.index.can_pin(id);
        self.healthy = false;
        long.healthy = false;
        ensure!(
            self.control.agree(long, 10, digest, valid.is_ok())?,
            "pin validation failed collectively"
        );
        let fits = self.control.agree(long, 11, digest, valid?)?;
        let lease = if fits {
            Some(self.index.pin(id)?)
        } else {
            None
        };
        ensure!(
            self.control.agree(long, 12, self.index.digest(), true)?,
            "pin commit agreement"
        );
        self.healthy = true;
        long.healthy = true;
        Ok(lease)
    }
    fn release(&mut self, long: &mut Long<'_>, lease: &Lease) -> Result<()> {
        self.ready(long)?;
        let mut h = Sha256::new();
        h.update(self.index.digest());
        h.update(lease.entry().to_le_bytes());
        h.update(lease.serial().to_le_bytes());
        let valid = self.index.validate_lease(lease);
        self.healthy = false;
        long.healthy = false;
        // Control::agree drains this CUDA owner before releasing the pin.
        ensure!(
            self.control
                .agree(long, 13, h.finalize().into(), valid.is_ok())?,
            "release validation failed collectively"
        );
        valid?;
        self.index.release(lease)?;
        ensure!(
            self.control.agree(long, 14, self.index.digest(), true)?,
            "release commit agreement"
        );
        self.healthy = true;
        long.healthy = true;
        Ok(())
    }
    pub(super) fn new(long: &Long<'_>, budget: u64) -> Result<Self> {
        long.ready()?;
        ensure!(budget > 0 && budget <= RAM_BUDGET, "session RAM budget");
        let control = Control::new(&long.engine.device)?;
        // All ranks hash the SAME ordered set of rank-specific shard identities.
        let mut record = [0; 64];
        let mut identity = Sha256::new();
        identity.update(long.model_identity.as_bytes());
        if let Some(draft) = long.speculation_identity() {
            identity.update(b"retained-drafter-v1");
            identity.update(draft);
        }
        record[..32].copy_from_slice(&identity.finalize());
        record[32..40].copy_from_slice(&(long.engine.rank as u64).to_le_bytes());
        record[40..48].copy_from_slice(&budget.to_le_bytes());
        record[48..56].copy_from_slice(&(long.layer_bytes() as u64).to_le_bytes());
        record[56..64]
            .copy_from_slice(&(1 + u64::from(long.speculation_identity().is_some())).to_le_bytes());
        let all = control.exchange(long, &record)?;
        for (rank, r) in all.iter().enumerate() {
            ensure!(
                r[32..40] == (rank as u64).to_le_bytes() && r[40..] == record[40..],
                "session group layout/config mismatch"
            );
        }
        let mut h = Sha256::new();
        h.update(ARITHMETIC.as_bytes());
        for record in all {
            h.update(record);
        }
        let index = Index::new(h.finalize().into(), budget);
        ensure!(
            control.agree(long, 0, index.digest(), true)?,
            "session group not ready"
        );
        Ok(Self {
            device: long.engine.device.clone(),
            stream: long.engine.stream.raw() as usize,
            model: long.model_identity.clone(),
            speculation: long.speculation_identity(),
            control,
            index,
            slabs: BTreeMap::new(),
            snapshots: BTreeMap::new(),
            healthy: true,
            copy_gate: None,
        })
    }
    fn ready(&self, long: &Long<'_>) -> Result<()> {
        long.ready()?;
        ensure!(
            self.healthy
                && Rc::ptr_eq(&self.device, &long.engine.device)
                && self.stream == long.engine.stream.raw() as usize
                && self.model == long.model_identity
                && self.speculation == long.speculation_identity(),
            "session store owner/health mismatch"
        );
        Ok(())
    }
    fn gpu_bytes(&self) -> usize {
        self.slabs.values().map(|s| family_bytes(s)).sum::<usize>()
            + self
                .snapshots
                .values()
                .map(Snapshot::gpu_bytes)
                .sum::<usize>()
    }
    fn families(&self, latent: usize, pooled: usize) -> Result<Vec<Family>> {
        (0..11)
            .map(|_| {
                Ok(Family {
                    latent: Rc::new(self.device.alloc(latent)?),
                    pooled: Rc::new(self.device.alloc(pooled)?),
                })
            })
            .collect()
    }
    pub(super) fn save(&mut self, long: &mut Long<'_>) -> Result<Saved> {
        self.save_mode(long, true)
    }
    /// Foreground snapshots may run under their request's own priority hold.
    /// Background maintenance keeps the stricter default `save` contract.
    pub(super) fn save_foreground(&mut self, long: &mut Long<'_>) -> Result<Saved> {
        ensure!(
            !long.sessions[long.active].background,
            "foreground snapshot owner"
        );
        long.ready_active()?;
        self.save_mode(long, false)
    }
    pub(super) fn find(&self, long: &Long<'_>, request: &[i32]) -> Result<Option<(u64, usize)>> {
        self.ready(long)?;
        self.index
            .find(request)
            .map(|id| Ok((id, self.index.entry(id)?.0.len())))
            .transpose()
    }
    pub(super) fn digest(&self) -> Hash {
        self.index.digest()
    }
    pub(super) fn common_prefix(&self, request: &[i32]) -> usize {
        self.index.common_prefix(request)
    }
    pub(super) fn holds(&self, request: &[i32], len: usize) -> bool {
        self.index.holds(request, len)
    }
    fn save_mode(&mut self, long: &mut Long<'_>, background: bool) -> Result<Saved> {
        self.ready(long)?;
        ensure!(
            !background || !long.priority_hold,
            "RAM snapshot save held by native priority"
        );
        let prepared = (|| {
            let session = &long.sessions[long.active];
            let key = Key::new(&self.model, &session.tokens, session.mixed)?;
            ensure!(
                long.memory.book().slot(long.active)?.pages().len()
                    >= key.tokens.len().div_ceil(PAGE),
                "snapshot source spilled"
            );
            let plan = self.index.plan(
                key.tokens.clone(),
                (long.layer_bytes() + long.speculation_snapshot_bytes(key.tokens.len())?) as u64,
            )?;
            let digest = self.index.planned_digest(&plan)?;
            Ok::<_, anyhow::Error>((key, plan, digest))
        })();
        let digest = prepared.as_ref().map(|v| v.2).unwrap_or([0; 32]);
        let agreed = self.control.agree(long, 1, digest, prepared.is_ok());
        if !matches!(agreed, Ok(true)) {
            self.healthy = false;
            long.healthy = false;
            agreed?;
            anyhow::bail!("session save validation failed collectively");
        }
        let (key, plan, digest) = prepared?;
        let growth = self
            .index
            .planned_gpu_bytes(&plan)?
            .saturating_sub(self.index.gpu_bytes());
        let memory = self.device.memory_view();
        let fits = memory
            .as_ref()
            .map(|v| v.usable.saturating_sub(growth) >= RESERVE);
        if let Ok(v) = &memory {
            eprintln!(
                "prefix_memory {}",
                serde_json::json!({"rank":long.engine.rank,
                "tokens":key.tokens.len(),"growth_bytes":growth,
                "immediate_free_fits":v.cuda_free.saturating_sub(growth)>=RESERVE,
                "usable_fits":v.usable.saturating_sub(growth)>=RESERVE,"memory":v})
            );
        }
        let memory_ok = self.control.agree(long, 8, digest, fits.is_ok());
        if !matches!(memory_ok, Ok(true)) {
            self.healthy = false;
            long.healthy = false;
            memory_ok?;
            anyhow::bail!("session memory query failed collectively");
        }
        let guard = self
            .control
            .agree(long, 2, digest, fits.as_ref().copied().unwrap_or(false));
        let fits_all = match guard {
            Ok(v) => v,
            Err(e) => {
                self.healthy = false;
                long.healthy = false;
                return Err(e);
            }
        };
        if !fits_all {
            return Ok(Saved {
                status: Status::Skipped,
                id: None,
                evicted: Vec::new(),
                new_pages: 0,
                copy_runs: 0,
            });
        }
        let mut report = Saved {
            status: plan.status,
            id: plan.entry,
            evicted: plan.evicted.clone(),
            new_pages: plan.new_pages.len(),
            copy_runs: 0,
        };
        self.healthy = false;
        long.healthy = false;
        let operation = (|| -> Result<()> {
            long.engine.sync()?;
            if plan.status == Status::Saved {
                for id in &plan.evicted {
                    ensure!(
                        self.snapshots.remove(id).is_some(),
                        "evicted snapshot missing"
                    );
                }
                let extents = self.index.planned_extents(&plan)?;
                self.slabs.retain(|id, _| extents.binary_search(id).is_ok());
                for extent in extents {
                    if !self.slabs.contains_key(&extent) {
                        self.slabs
                            .insert(extent, self.families(EXTENT * LATENT, EXTENT * POOLED)?);
                    }
                }
                let live = long.memory.book().slot(long.active)?.pages();
                let ranges = runs(&plan.new_pages, live)?;
                report.copy_runs = ranges.len();
                for (src, dst, count) in ranges {
                    let slab = &self.slabs[&(dst / EXTENT)];
                    for (target, source) in slab.iter().zip(&long.memory.families) {
                        unsafe {
                            target.latent.copy_from(
                                &long.engine.stream,
                                (dst % EXTENT) * LATENT,
                                &source.latent,
                                src * LATENT,
                                count * LATENT,
                            )?;
                            target.pooled.copy_from(
                                &long.engine.stream,
                                (dst % EXTENT) * POOLED,
                                &source.pooled,
                                src * POOLED,
                                count * POOLED,
                            )?;
                        }
                    }
                }
                let shared = (key.tokens.len() - 1) / PAGE;
                let tail_rows = key.tokens.len() - shared * PAGE;
                let tail = self.families(tail_rows * 528, tail_rows / 4 * 128 * 2)?;
                let source = live[shared] as usize;
                for (target, src) in tail.iter().zip(&long.memory.families) {
                    unsafe {
                        target.latent.copy_from(
                            &long.engine.stream,
                            0,
                            &src.latent,
                            source * LATENT,
                            target.latent.bytes,
                        )?;
                        target.pooled.copy_from(
                            &long.engine.stream,
                            0,
                            &src.pooled,
                            source * POOLED,
                            target.pooled.bytes,
                        )?;
                    }
                }
                let layers = long.capture_layers()?;
                let draft = long.capture_speculation()?;
                self.snapshots.insert(
                    plan.entry.unwrap(),
                    Snapshot {
                        key,
                        layers,
                        tail,
                        draft,
                    },
                );
            }
            long.engine.sync()?;
            ensure!(
                self.gpu_bytes() as u64 == self.index.planned_gpu_bytes(&plan)?,
                "snapshot GPU bytes disagree with budget"
            );
            Ok(())
        })();
        ensure!(
            self.control.agree(long, 3, digest, operation.is_ok())?,
            "session save failed collectively"
        );
        operation?;
        self.index.commit(plan)?;
        ensure!(
            self.control.agree(long, 4, self.index.digest(), true)?,
            "session save commit agreement"
        );
        self.healthy = true;
        long.healthy = true;
        Ok(report)
    }
    pub(super) fn restore(
        &mut self,
        long: &mut Long<'_>,
        id: u64,
        request: &[i32],
        output: usize,
        background: bool,
    ) -> Result<usize> {
        self.ready(long)?;
        ensure!(
            !background || !long.priority_hold,
            "RAM restore held by native priority"
        );
        let prepared = (|| {
            ensure!(
                long.memory.book().slot(long.active)?.phase() == Phase::Idle,
                "restore destination must be idle"
            );
            let snapshot = self
                .snapshots
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("restore snapshot missing"))?;
            snapshot.key.validate(&self.model, request, output)?;
            let (ids, pages) = self.index.entry(id)?;
            ensure!(ids == snapshot.key.tokens, "index/payload token identity");
            let n = ids.len();
            let keep = n.min(long.memory.book().slot(long.active)?.pages().len() * PAGE);
            let admission = long.admission(request.len(), output, keep, background)?;
            let mut h = Sha256::new();
            h.update(self.index.digest());
            for v in [id, long.active as u64, output as u64, u64::from(background)] {
                h.update(v.to_le_bytes());
            }
            for t in request {
                h.update(t.to_le_bytes());
            }
            // Include the final selected slot's complete admission decision.
            h.update(admission.digest());
            Ok::<_, anyhow::Error>((ids, pages, admission, keep, h.finalize().into()))
        })();
        let digest = prepared.as_ref().map(|p| p.4).unwrap_or([0; 32]);
        let agreed = self.control.agree(long, 5, digest, prepared.is_ok());
        if !matches!(agreed, Ok(true)) {
            self.healthy = false;
            long.healthy = false;
            agreed?;
            anyhow::bail!("session restore validation failed collectively");
        }
        let (ids, pages, admission, keep, digest) = prepared?;
        let n = ids.len();
        self.healthy = false;
        long.healthy = false;
        let operation = (|| -> Result<()> {
            long.commit_admission(&admission, keep)?;
            long.memory.map(&long.engine.stream, long.active, n)?;
            let live = long.memory.book().slot(long.active)?.pages();
            // Use the same coalescer in reverse: live source = restore target.
            let placed: Vec<_> = pages.iter().enumerate().map(|(i, p)| (i, *p)).collect();
            for (dst, src, count) in runs(&placed, live)? {
                let slab = &self.slabs[&(src / EXTENT)];
                for (target, source) in long.memory.families.iter().zip(slab) {
                    unsafe {
                        target.latent.copy_from(
                            &long.engine.stream,
                            dst * LATENT,
                            &source.latent,
                            (src % EXTENT) * LATENT,
                            count * LATENT,
                        )?;
                        target.pooled.copy_from(
                            &long.engine.stream,
                            dst * POOLED,
                            &source.pooled,
                            (src % EXTENT) * POOLED,
                            count * POOLED,
                        )?;
                    }
                }
            }
            let snapshot = &self.snapshots[&id];
            let dst = live[pages.len()] as usize;
            for (target, source) in long.memory.families.iter().zip(&snapshot.tail) {
                unsafe {
                    target.latent.copy_from(
                        &long.engine.stream,
                        dst * LATENT,
                        &source.latent,
                        0,
                        source.latent.bytes,
                    )?;
                    target.pooled.copy_from(
                        &long.engine.stream,
                        dst * POOLED,
                        &source.pooled,
                        0,
                        source.pooled.bytes,
                    )?;
                }
            }
            long.restore_layers(&snapshot.layers, n)?;
            long.restore_speculation_at(long.active, snapshot.draft.as_ref(), n)?;
            unsafe {
                long.sessions[long.active]
                    .position
                    .set_i32(&long.engine.stream, n as i32)?;
            }
            long.engine.sync()?;
            Ok(())
        })();
        ensure!(
            self.control.agree(long, 6, digest, operation.is_ok())?,
            "session restore copy failed collectively"
        );
        operation?;
        let session = &mut long.sessions[long.active];
        session.tokens.clear();
        session.tokens.extend_from_slice(&ids);
        session.mixed = false;
        session.pending = None;
        session.prompt_end = request.len();
        session.total_end = request.len() + output;
        session.background = background;
        long.latest_logits = None;
        self.index.touch(id)?;
        let activated = long.sync_speculation_holds();
        ensure!(
            self.control
                .agree(long, 7, self.index.digest(), activated.is_ok())?,
            "session restore commit agreement"
        );
        activated?;
        self.healthy = true;
        long.healthy = true;
        Ok(n)
    }
}

pub(super) fn prefill_to(long: &mut Long<'_>, tokens: &[i32]) -> Result<i32> {
    let mut next = 0;
    while long.sessions[long.active].tokens.len() < tokens.len() {
        let start = long.sessions[long.active].tokens.len();
        next = long.prefill(start, &tokens[start..(start + ROWS).min(tokens.len())])?;
    }
    Ok(next)
}
pub(super) fn finish_reference(long: &mut Long<'_>, tokens: &[i32]) -> Result<serde_json::Value> {
    let mut next = prefill_to(long, tokens)?;
    let prefill_state = long.state(tokens.len())?;
    let mut logits = vec![long.logits()?];
    let mut outputs = vec![next];
    for _ in 0..3 {
        next = long.step(long.sessions[long.active].tokens.len(), next)?;
        outputs.push(next);
        logits.push(long.logits()?);
    }
    let state = long.state(tokens.len() + 3)?;
    long.finish()?;
    Ok(
        serde_json::json!({"prefill_state":prefill_state,"final_state":state,"logits":logits,"outputs":outputs}),
    )
}
#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    sets: &[Vec<i32>],
    _new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    let tokens = sets
        .last()
        .ok_or_else(|| anyhow::anyhow!("store fixture"))?;
    ensure!(
        (4096..=16384).contains(&tokens.len()),
        "store fixture length"
    );
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    // A deliberately small budget forces real GPU eviction in this harness.
    let cut = if tokens.len() >= 16384 { 8256 } else { 2112 };
    let page_cut = cut - 64;
    let branch_cut = cut + 64;
    let budget = if cut > 4096 {
        256u64 << 20
    } else {
        192u64 << 20
    };
    let mut store = Store::new(&long, budget)?;
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..page_cut])?;
    let page_tail_state = long.state(page_cut)?;
    let full_tail = store.save(&mut long)?;
    ensure!(
        full_tail.new_pages == (page_cut - 1) / PAGE && full_tail.copy_runs < full_tail.new_pages,
        "coalesced page-boundary snapshot"
    );
    prefill_to(&mut long, &tokens[..cut])?;
    let prefix = long.state(cut)?;
    let first = store.save(&mut long)?;
    let first_id = first.id.unwrap();
    ensure!(
        first.status == Status::Saved && first.new_pages == 1,
        "coalesced first snapshot"
    );
    let duplicate = store.save(&mut long)?;
    ensure!(duplicate.status == Status::Duplicate, "duplicate snapshot");
    let reference = finish_reference(&mut long, tokens)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    long.bind(1)?;
    store.restore(&mut long, full_tail.id.unwrap(), tokens, 4, false)?;
    ensure!(
        long.state(page_cut)? == page_tail_state
            && finish_reference(&mut long, tokens)? == reference,
        "full private page-tail restore"
    );
    // Three divergent continuations share all complete prefix pages, but keep
    // distinct recurrent state and private tails.
    let mut reports = Vec::new();
    let mut candidates = Vec::new();
    for branch in 0..3 {
        long.bind(1)?;
        long.start(64, 4)?;
        long.prefill(0, &[8; 64])?;
        long.finish()?;
        let mut branch_tokens = tokens[..branch_cut + 1].to_vec();
        branch_tokens[cut] = 10 + branch;
        ensure!(
            store.restore(&mut long, first_id, &branch_tokens, 4, false)? == cut
                && long.state(cut)? == prefix,
            "store restores original prefix"
        );
        prefill_to(&mut long, &branch_tokens[..branch_cut])?;
        let saved = store.save(&mut long)?;
        ensure!(
            saved.status == Status::Saved && saved.new_pages == 0,
            "shared-prefix snapshot duplicated pages"
        );
        reports.push(serde_json::json!({"evicted":saved.evicted,"new_pages":saved.new_pages,"gpu_bytes":store.gpu_bytes(),"charged_bytes":store.index.used()}));
        let state = long.state(branch_cut)?;
        let result = finish_reference(&mut long, &branch_tokens)?;
        candidates.push((saved.id.unwrap(), branch_tokens, state, result));
    }
    for (id, branch, state, result) in &candidates {
        long.bind(1)?;
        store.restore(&mut long, *id, branch, 4, false)?;
        ensure!(
            long.state(branch_cut)? == *state && finish_reference(&mut long, branch)? == *result,
            "divergent branch restore"
        );
    }
    // Restored live state remains independent even when its source entry is
    // evicted to make room for an unrelated prefix.
    long.bind(0)?;
    store.restore(&mut long, first_id, tokens, 4, false)?;
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "restored model differs from cold reference"
    );
    let source_state = long.state(tokens.len() + 3)?;
    let source_pin = store
        .pin(&mut long, first_id)?
        .ok_or_else(|| anyhow::anyhow!("pin capacity"))?;
    let second_pin = store
        .pin(&mut long, first_id)?
        .ok_or_else(|| anyhow::anyhow!("second pin capacity"))?;
    store.release(&mut long, &source_pin)?;
    for fill in 0..5 {
        long.bind(1)?;
        let input = vec![45 + fill; 2177];
        long.start(input.len(), 4)?;
        prefill_to(&mut long, &input[..2176])?;
        let saved = store.save(&mut long)?;
        ensure!(
            !saved.evicted.contains(&first_id)
                && store.index.validate_lease(&second_pin).is_ok()
                && store.gpu_bytes() as u64 == store.index.gpu_bytes(),
            "pinned source eviction/accounting"
        );
        long.finish()?;
    }
    long.bind(3)?;
    store.restore(&mut long, first_id, tokens, 4, false)?;
    ensure!(
        long.state(cut)? == prefix && finish_reference(&mut long, tokens)? == reference,
        "pinned source changed after live-slot overwrite and eviction pressure"
    );
    store.release(&mut long, &second_pin)?;
    let mut evicted = Vec::new();
    let mut retained = None;
    for fill in 0..5 {
        long.bind(1)?;
        long.start(2177, 4)?;
        let input = vec![30 + fill; 2177];
        prefill_to(&mut long, &input[..2176])?;
        let save = store.save(&mut long)?;
        evicted.extend(save.evicted);
        let result = finish_reference(&mut long, &input)?;
        retained = Some((save.id.unwrap(), input, result));
    }
    ensure!(
        evicted.contains(&first_id),
        "eviction fixture did not evict original prefix"
    );
    let (id, input, result) = retained.unwrap();
    long.bind(3)?;
    store.restore(&mut long, id, &input, 4, false)?;
    ensure!(
        finish_reference(&mut long, &input)? == result,
        "restore after eviction/reallocation"
    );
    long.bind(0)?;
    ensure!(
        long.state(tokens.len() + 3)? == source_state,
        "eviction changed copied-in live owner"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "store changed paused owner");
    long.cancel()?;
    store.index.check()?;
    ensure!(
        store.gpu_bytes() as u64 == store.index.gpu_bytes(),
        "store allocation audit"
    );
    // All ranks reject a deliberately divergent planned decision, without any
    // payload/index mutation. Test the control gate independently of poisoning.
    let before = store.index.digest();
    let mut divergent = before;
    if rank == 3 {
        divergent[0] ^= 1;
    }
    ensure!(
        store.control.agree(&long, 99, divergent, true).is_err() && store.index.digest() == before,
        "rank disagreement not rejected"
    );
    ensure!(
        !store.control.agree(&long, 98, before, rank != 3)? && store.index.digest() == before,
        "one-rank reserve veto"
    );
    long.bind(1)?;
    long.start(64, 4)?;
    long.prefill(0, &[7; 64])?;
    let mut tiny = Store::new(&long, 1)?;
    let skipped = tiny.save(&mut long)?;
    ensure!(
        skipped.status == Status::Skipped
            && tiny.gpu_bytes() == 0
            && store.index.digest() == before,
        "oversized GPU save allocated or evicted"
    );
    long.finish()?;
    // Real save preflight must stop every rank before touching stored payloads
    // if one owner supplies different token metadata. A poisoned owner refuses
    // the next request; restart is required after disagreement.
    let allocated = store.gpu_bytes();
    let mappings = long.memory.book().slot(long.active)?.pages().to_vec();
    if rank == 3 {
        long.sessions[long.active].tokens[0] = 8;
    }
    ensure!(
        store.save(&mut long).is_err()
            && !store.healthy
            && !long.healthy
            && store.index.digest() == before
            && store.gpu_bytes() == allocated
            && long.memory.book().slot(long.active)?.pages() == mappings
            && long.start(64, 4).is_err(),
        "divergent save failed to poison without mutation"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"reused_tokens":cut,"page_boundary_tokens":page_cut,"branch_snapshot_tokens":branch_cut,"initial_page_copy_runs":full_tail.copy_runs,"cold_restored_exact":true,"counted_pins_preserve_overwritten_source_under_eviction":true,"duplicate_exact":true,"full_page_tail_exact":true,"divergent_branch_restore_exact":true,"restore_after_eviction_exact":true,"one_rank_reserve_veto":true,"oversize_no_payload_allocation":true,"branch_saves":reports,"evicted_original":true,"restored_and_paused_live_state_unchanged":true,"rank_disagreement_rejected_before_mutation":true,"divergent_save_poisons_before_mutation":true,"gpu_bytes":store.gpu_bytes(),"charged_bytes":store.index.used(),"budget":budget,"index_digest":format!("{:x}",sha2::digest::generic_array::GenericArray::from(store.index.digest())),"reference":reference,"scope":"Bounded pure-prefix GPU store; no SSD persistence or in-flight request recovery."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coalescing_preserves_holes_fragmentation_and_slab_boundaries() {
        assert_eq!(
            runs(
                &[(0, 14), (1, 15), (2, 16), (4, 17), (5, 18)],
                &[2, 3, 4, 99, 8, 10]
            )
            .unwrap(),
            vec![(2, 14, 2), (4, 16, 1), (8, 17, 1), (10, 18, 1)]
        );
        assert!(runs(&[(2, 0)], &[0]).is_err());
    }
}
