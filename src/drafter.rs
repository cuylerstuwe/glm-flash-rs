//! Native DFlash2 forward state with one bounded 4096-slot ring per owner.
//! Request scheduling and cached speculation state are managed by the caller.
use super::Engine;
use crate::{
    cuda::{Buffer, Kernel},
    draft_blas::Blas,
    draft_q4::{DRAFT_HEAD, Projection, SHAPES},
    drafter_weights,
    dsa::words,
    front::{load_kernel, load_native},
    prepared::{Loaded, Prepared},
};
use anyhow::{Result, ensure};
use std::{collections::BTreeMap, path::Path};
#[path = "drafter_backlog.rs"]
pub(super) mod backlog;
#[path = "drafter_snapshot.rs"]
pub(super) mod snapshot;
const MAX: usize = 64;
/// Context updates of up to this many accepted rows (every decode round) can replay
/// a per-slot graph instead of launching their ~26 small kernels one by one.
const TAP_GRAPH_ROWS: usize = 16;
fn tap_graph_policy() -> Result<bool> {
    match std::env::var("ALASKA_DRAFT_TAP_GRAPHS").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(false),
        Ok("1") => Ok(true),
        _ => anyhow::bail!("ALASKA_DRAFT_TAP_GRAPHS must be 0/1"),
    }
}
/// Draft logits from the 4-bit draft head (default) or the BF16 target
/// head. Drafts only propose; verified output is identical either way.
fn draft_head_policy() -> Result<bool> {
    match std::env::var("ALASKA_DRAFT_HEAD").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("q4") => Ok(true),
        Ok("bf16") => Ok(false),
        _ => anyhow::bail!("ALASKA_DRAFT_HEAD must be q4 or bf16"),
    }
}
struct Context {
    kc: Vec<Buffer>,
    vc: Vec<Buffer>,
    pos: Buffer,
    lo: Buffer,
    end: usize,
}
impl Context {
    fn new(d: &std::rc::Rc<crate::cuda::Device>) -> Result<Self> {
        let pos = d.alloc(8)?;
        pos.zero()?;
        let lo = d.alloc(8)?;
        lo.zero()?;
        Ok(Self {
            kc: (0..5)
                .map(|_| d.alloc(2 * 4096 * 128 * 2))
                .collect::<Result<_>>()?,
            vc: (0..5)
                .map(|_| d.alloc(2 * 4096 * 128 * 2))
                .collect::<Result<_>>()?,
            pos,
            lo,
            end: 0,
        })
    }
}
pub(super) struct Drafter {
    readback: Option<crate::draft_readback::Pair>,
    graphs: [Option<crate::cuda::Graph>; 4],
    /// Per slot, indexed by rows - 1; captured on first use. Every captured pointer
    /// (weights, scratch, the slot's K/V/position) lives as long as the drafter.
    tap_graphs: Option<Vec<Vec<Option<crate::cuda::Graph>>>>,
    contexts: Vec<Context>,
    active: usize,
    select_rows: Kernel,
    cand_local: Buffer,
    cand_gathered: Buffer,
    q4_head: bool,
    blas: Blas,
    p: Prepared,
    m: Loaded,
    kernels: BTreeMap<&'static str, Kernel>,
    q4: Vec<Projection>,
    pub(super) taps: Buffer,
    ids: Buffer,
    x: Buffer,
    norm: Buffer,
    xs: Buffer,
    dyns: Buffer,
    conv: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    row: Buffer,
    gathered: Buffer,
    reduced: Buffer,
    gu: Buffer,
    act: Buffer,
    axs: Buffer,
    part: Buffer,
    ctx: Buffer,
    co: Buffer,
    si: Buffer,
    pub(super) logits: Buffer,
    pub(super) projected: Buffer,
}
impl Drafter {
    pub(super) fn new(e: &Engine<'_>, manifest: &Path, root: &Path, native: &Path) -> Result<Self> {
        let d = &e.device;
        let p = drafter_weights::open(manifest, e.rank as u64)?;
        let m = p.load_selected(d, &["drafter"])?;
        let mut kernels = BTreeMap::new();
        for name in [
            "draft_norm",
            "draft_conv_pre",
            "draft_conv_post",
            "draft_prep_block",
            "draft_prep_context",
            "draft_attention",
            "draft_sums_20480",
            "draft_sums_4096",
            "draft_sums_1024",
            "swiglu",
            "verify_head",
        ] {
            kernels.insert(name, load_kernel(d, root, name)?);
        }
        for name in [
            "draft_rotary",
            "draft_store",
            "draft_advance",
            "draft_embed",
            "draft_rank_sum",
        ] {
            kernels.insert(
                name,
                load_native(d, &native.join("draft_support.cubin"), name, 256)?,
            );
        }
        let alloc = |width, bytes| d.alloc(MAX * width * bytes);
        Ok(Self {
            readback: if crate::draft_readback::enabled()? {
                Some(crate::draft_readback::Pair::new(
                    d,
                    4 * 7 * crate::candidates::payload_bytes(16)?,
                    7 * 256 * 2,
                )?)
            } else {
                None
            },
            graphs: std::array::from_fn(|_| None),
            tap_graphs: tap_graph_policy()?.then(|| {
                (0..4)
                    .map(|_| (0..TAP_GRAPH_ROWS).map(|_| None).collect())
                    .collect()
            }),
            contexts: (0..4).map(|_| Context::new(d)).collect::<Result<_>>()?,
            active: 0,
            select_rows: load_native(d, &native.join("candidates.cubin"), "candidates_rows", 256)?,
            cand_local: d.alloc(7 * crate::candidates::payload_bytes(16)?)?,
            cand_gathered: d.alloc(4 * 7 * crate::candidates::payload_bytes(16)?)?,
            q4_head: draft_head_policy()?,
            blas: Blas::new(d)?,
            p,
            m,
            kernels,
            q4: SHAPES
                .into_iter()
                .map(|q| Projection::new(d, root, q))
                .collect::<Result<_>>()?,
            taps: alloc(20480, 2)?,
            ids: d.alloc(8 * 4)?,
            x: alloc(4096, 2)?,
            norm: alloc(4096, 2)?,
            xs: alloc(320, 4)?,
            dyns: alloc(1024, 2)?,
            conv: alloc(4096, 2)?,
            qkv: alloc(1536, 2)?,
            q: alloc(1024, 2)?,
            k: alloc(256, 2)?,
            v: alloc(256, 2)?,
            attention: alloc(1024, 2)?,
            row: alloc(4096, 4)?,
            gathered: alloc(4 * 4096, 4)?,
            reduced: alloc(4096, 2)?,
            gu: alloc(6144, 2)?,
            act: alloc(3072, 2)?,
            axs: alloc(48, 4)?,
            part: alloc(8 * 6144, 4)?,
            ctx: alloc(4096, 2)?,
            co: alloc(64, 4)?,
            si: alloc(64, 4)?,
            logits: d.alloc(7 * 38720 * 2)?,
            projected: d.alloc(7 * 256 * 2)?,
        })
    }
    fn context(&self) -> &Context {
        &self.contexts[self.active]
    }
    pub(super) fn end(&self) -> usize {
        self.context().end
    }
    pub(super) fn context_end(&self, slot: usize) -> Result<usize> {
        self.contexts
            .get(slot)
            .map(|context| context.end)
            .ok_or_else(|| anyhow::anyhow!("drafter slot bound"))
    }
    /// Shared weights/scratch, retained independent K/V/position and graph per
    /// slot. Switching ownership is only allowed at a drained native boundary.
    pub(super) fn bind(&mut self, e: &Engine<'_>, slot: usize) -> Result<()> {
        ensure!(slot < self.contexts.len(), "drafter slot bound");
        e.sync()?;
        self.active = slot;
        Ok(())
    }
    fn w(&self, name: &str) -> Result<u64> {
        self.m.pointer(&self.p, &format!("drafter.{name}"))
    }
    fn qw(&self, name: &str) -> Result<[u64; 3]> {
        Ok([
            self.w(&format!("{name}.weight"))?,
            self.w(&format!("{name}.scales"))?,
            self.w(&format!("{name}.biases"))?,
        ])
    }
    #[allow(clippy::too_many_arguments)]
    fn mm(
        &self,
        e: &Engine<'_>,
        shape: usize,
        name: &str,
        x: u64,
        sums: u64,
        out: &Buffer,
        rows: usize,
    ) -> Result<()> {
        unsafe {
            self.q4[shape].run(
                &e.stream,
                x,
                SHAPES[shape].k,
                sums,
                self.qw(name)?,
                out,
                &self.part,
                rows,
            )
        }
    }
    fn sums(&self, e: &Engine<'_>, x: u64, k: usize, rows: usize) -> Result<()> {
        words(
            &self.kernels[&*format!("draft_sums_{k}")],
            &e.stream,
            [rows as u32, (k / 64).div_ceil(16) as u32, 1],
            &[x, self.xs.ptr, k as u64],
        )
    }
    fn norm(&self, e: &Engine<'_>, x: u64, weight: u64, rows: usize) -> Result<()> {
        words(
            &self.kernels["draft_norm"],
            &e.stream,
            [rows as u32, 1, 1],
            &[
                x,
                4096,
                weight,
                self.norm.ptr,
                4096,
                self.xs.ptr,
                1e-5f32.to_bits() as u64,
            ],
        )
    }
    fn rotary(&self, e: &Engine<'_>, rows: usize) -> Result<()> {
        words(
            &self.kernels["draft_rotary"],
            &e.stream,
            [(rows * 64).div_ceil(256) as u32, 1, 1],
            &[
                self.context().pos.ptr,
                self.co.ptr,
                self.si.ptr,
                rows as u64,
            ],
        )
    }
    fn prep(&self, e: &Engine<'_>, layer: usize, rows: usize, block: bool) -> Result<()> {
        let heads = if block { 8 } else { 0 };
        let stride = if block { 1536 } else { 512 };
        words(
            &self.kernels[if block {
                "draft_prep_block"
            } else {
                "draft_prep_context"
            }],
            &e.stream,
            [rows as u32, heads + 4, 1],
            &[
                self.qkv.ptr,
                self.w(&format!("layers[{layer}].q_norm"))?,
                self.w(&format!("layers[{layer}].k_norm"))?,
                self.co.ptr,
                self.si.ptr,
                self.q.ptr,
                self.k.ptr,
                self.v.ptr,
                rows as u64,
                stride,
                1e-5f32.to_bits() as u64,
            ],
        )?;
        words(
            &self.kernels["draft_store"],
            &e.stream,
            [(2 * rows * 128).div_ceil(256) as u32, 1, 1],
            &[
                self.k.ptr,
                self.v.ptr,
                self.context().kc[layer].ptr,
                self.context().vc[layer].ptr,
                self.context().pos.ptr,
                rows as u64,
            ],
        )
    }
    pub(super) fn reset(&mut self, e: &Engine<'_>) -> Result<()> {
        e.sync()?;
        self.contexts[self.active].end = 0;
        self.context().pos.zero()?;
        self.context().lo.zero()?;
        for b in self.context().kc.iter().chain(&self.context().vc) {
            b.zero()?;
        }
        Ok(())
    }
    /// `taps` already holds exactly rows committed target mean-stream features.
    pub(super) fn add_taps(&mut self, e: &Engine<'_>, rows: usize) -> Result<()> {
        ensure!(
            (1..=MAX).contains(&rows) && self.end() + rows <= 1048576,
            "draft context bound"
        );
        let active = self.active;
        if rows <= TAP_GRAPH_ROWS
            && let Some(graphs) = &self.tap_graphs
        {
            if graphs[active][rows - 1].is_none() {
                let graph = unsafe { e.stream.capture(|| self.tap_kernels(e, rows))? };
                self.tap_graphs.as_mut().unwrap()[active][rows - 1] = Some(graph);
            }
            unsafe {
                self.tap_graphs.as_ref().unwrap()[active][rows - 1]
                    .as_ref()
                    .unwrap()
                    .launch(&e.stream)?;
            }
        } else {
            self.tap_kernels(e, rows)?;
        }
        self.contexts[active].end += rows;
        Ok(())
    }
    /// The context update's device work only: the position advances on the device.
    fn tap_kernels(&self, e: &Engine<'_>, rows: usize) -> Result<()> {
        self.sums(e, self.taps.ptr, 20480, rows)?;
        self.mm(e, 0, "fc", self.taps.ptr, self.xs.ptr, &self.ctx, rows)?;
        self.norm(e, self.ctx.ptr, self.w("hidden_norm")?, rows)?;
        self.rotary(e, rows)?;
        for i in 0..5 {
            self.mm(
                e,
                3,
                &format!("layers[{i}].kv"),
                self.norm.ptr,
                self.xs.ptr,
                &self.qkv,
                rows,
            )?;
            self.prep(e, i, rows, false)?;
        }
        words(
            &self.kernels["draft_advance"],
            &e.stream,
            [1, 1, 1],
            &[self.context().pos.ptr, rows as u64],
        )
    }
    fn row(
        &self,
        e: &Engine<'_>,
        shape: usize,
        name: &str,
        x: u64,
        sums: u64,
        rows: usize,
    ) -> Result<()> {
        self.mm(e, shape, name, x, sums, &self.row, rows)?;
        unsafe {
            e.gather_prefix(&self.row, &self.gathered, rows * 4096 * 4)?;
        }
        words(
            &self.kernels["draft_rank_sum"],
            &e.stream,
            [(rows * 4096).div_ceil(256) as u32, 1, 1],
            &[self.gathered.ptr, self.reduced.ptr, (rows * 4096) as u64],
        )
    }
    pub(super) fn start_context(&mut self, e: &Engine<'_>, first: usize) -> Result<()> {
        self.reset(e)?;
        self.contexts[self.active].end = first;
        self.context().pos.write(0, &(first as i64).to_le_bytes())?;
        self.context().lo.write(0, &(first as i64).to_le_bytes())
    }
    pub(super) fn append_history(
        &mut self,
        e: &Engine<'_>,
        history: &crate::target_taps::Taps,
        end: usize,
    ) -> Result<()> {
        ensure!(
            end >= self.end() && end - self.end() <= history.cap,
            "draft accepted history range"
        );
        while self.end() < end {
            let n = (end - self.end()).min(MAX);
            history.copy_to(&e.stream, self.end(), n, &self.taps)?;
            self.add_taps(e, n)?;
        }
        Ok(())
    }
    fn collect(&self, e: &Engine<'_>) -> Result<()> {
        self.block(e)?;
        let payload = crate::candidates::payload_bytes(16)?;
        words(
            &self.select_rows,
            &e.stream,
            [7, 1, 1],
            &[
                self.logits.ptr,
                16,
                (e.rank * crate::candidates::LOCAL) as u64,
                self.cand_local.ptr,
                crate::candidates::LOCAL as u64,
                (payload / 4) as u64,
            ],
        )?;
        unsafe { e.gather_prefix(&self.cand_local, &self.cand_gathered, 7 * payload) }
    }
    pub(super) fn propose(
        &mut self,
        e: &Engine<'_>,
        pending: i32,
        policy: crate::sampling::Sampling,
    ) -> Result<Vec<i32>> {
        Ok(self.propose_scored(e, pending, policy)?.tokens)
    }
    pub(super) fn propose_scored(
        &mut self,
        e: &Engine<'_>,
        pending: i32,
        policy: crate::sampling::Sampling,
    ) -> Result<crate::draft_chain::Proposal> {
        ensure!(
            self.end() > 0 && self.end() + 8 <= 1048576,
            "drafter proposal context bound"
        );
        self.set_pending(e, pending)?;
        if self.graphs[self.active].is_none() {
            self.collect(e)?;
            e.sync()?;
            self.graphs[self.active] = Some(unsafe { e.stream.capture(|| self.collect(e))? });
        }
        unsafe {
            self.graphs[self.active]
                .as_ref()
                .unwrap()
                .launch(&e.stream)?;
        }
        let payload = crate::candidates::payload_bytes(16)?;
        let staged = if let Some(readback) = &mut self.readback {
            Some(readback.copy(&e.stream, &self.cand_gathered, &self.projected)?)
        } else {
            None
        };
        // Also checks the transport before any gathered candidate is consumed.
        e.sync()?;
        crate::round_timing::enter(crate::round_timing::Phase::DraftRead);
        // Gathered rank-major: [rank][row][payload]; decode takes one row's
        // four rank payloads contiguously.
        let (all, projected_bytes) = match staged {
            Some((all, projected)) => (all, Some(projected)),
            None => (self.cand_gathered.read(0, 4 * 7 * payload)?, None),
        };
        let rows = (0..7)
            .map(|row| -> Result<_> {
                let mut bytes = Vec::with_capacity(4 * payload);
                for rank in 0..4 {
                    let at = (rank * 7 + row) * payload;
                    bytes.extend_from_slice(&all[at..at + payload]);
                }
                let mut row = crate::candidates::decode(&bytes, 16)?;
                row.sort_by(|a, b| b.value.total_cmp(&a.value).then(a.id.cmp(&b.id)));
                row.truncate(16);
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        let projected_bytes = match projected_bytes {
            Some(bytes) => bytes,
            None => self.projected.read(0, self.projected.bytes)?,
        };
        let projected = crate::front::bf_values(&projected_bytes);
        let codebook = |name: &str| -> Result<&[u8]> {
            let t = self.p.tensor(name)?;
            match &self.m.storages[t.storage] {
                crate::prepared::LoadedStorage::Cpu(b) => Ok(b),
                _ => anyhow::bail!("drafter selector storage"),
            }
        };
        let pred = codebook("drafter.pred")?;
        let succ = codebook("drafter.succ")?;
        crate::draft_chain::choose_scored(
            &rows,
            &projected,
            pending,
            self.end() + 1,
            policy,
            |previous, id, k| {
                let b = if previous { pred } else { succ };
                let off = (id as usize * 256 + k) * 4;
                f32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as f64
            },
        )
    }
    pub(super) fn set_pending(&self, e: &Engine<'_>, id: i32) -> Result<()> {
        ensure!((0..154880).contains(&id), "draft pending token");
        e.sync()?;
        let mut ids = [154856i32; 8];
        ids[0] = id;
        self.ids.write(
            0,
            &ids.into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
    }
    /// Leaves speculative ring rows past `end`; only add_taps advances the cursor.
    pub(super) fn block(&self, e: &Engine<'_>) -> Result<()> {
        ensure!(
            self.end() > 0 && self.end() + 8 <= 1048576,
            "draft block bound"
        );
        let s = &e.stream;
        words(
            &self.kernels["draft_embed"],
            s,
            [128, 1, 1],
            &[
                self.ids.ptr,
                e.m.pointer(e.p, "weights.embed")?,
                self.x.ptr,
                8,
            ],
        )?;
        self.rotary(e, 8)?;
        for i in 0..5 {
            let base = format!("layers[{i}]");
            self.norm(e, self.x.ptr, self.w(&format!("{base}.in_norm"))?, 8)?;
            self.mm(
                e,
                1,
                &format!("{base}.a_kp"),
                self.norm.ptr,
                self.xs.ptr,
                &self.dyns,
                8,
            )?;
            words(
                &self.kernels["draft_conv_pre"],
                s,
                [8, 4, 1],
                &[
                    self.norm.ptr,
                    self.dyns.ptr,
                    self.w(&format!("{base}.a_base"))?,
                    self.x.ptr,
                    self.conv.ptr,
                ],
            )?;
            self.sums(e, self.conv.ptr, 4096, 8)?;
            self.mm(
                e,
                2,
                &format!("{base}.qkv"),
                self.conv.ptr,
                self.xs.ptr,
                &self.qkv,
                8,
            )?;
            self.prep(e, i, 8, true)?;
            words(
                &self.kernels["draft_attention"],
                s,
                [2, 1, 1],
                &[
                    self.q.ptr,
                    self.context().kc[i].ptr,
                    self.context().vc[i].ptr,
                    self.attention.ptr,
                    self.context().pos.ptr,
                    self.context().lo.ptr,
                    2047,
                    (128f32.sqrt().recip()).to_bits() as u64,
                ],
            )?;
            self.sums(e, self.attention.ptr, 1024, 8)?;
            self.row(
                e,
                4,
                &format!("{base}.o"),
                self.attention.ptr,
                self.xs.ptr,
                8,
            )?;
            words(
                &self.kernels["draft_conv_post"],
                s,
                [8, 4, 1],
                &[
                    self.reduced.ptr,
                    self.dyns.ptr,
                    self.w(&format!("{base}.a_base"))?,
                    self.x.ptr,
                    self.x.ptr,
                ],
            )?;
            self.norm(e, self.x.ptr, self.w(&format!("{base}.post_norm"))?, 8)?;
            self.mm(
                e,
                1,
                &format!("{base}.m_kp"),
                self.norm.ptr,
                self.xs.ptr,
                &self.dyns,
                8,
            )?;
            words(
                &self.kernels["draft_conv_pre"],
                s,
                [8, 4, 1],
                &[
                    self.norm.ptr,
                    self.dyns.ptr,
                    self.w(&format!("{base}.m_base"))?,
                    self.x.ptr,
                    self.conv.ptr,
                ],
            )?;
            self.sums(e, self.conv.ptr, 4096, 8)?;
            self.mm(
                e,
                5,
                &format!("{base}.gu"),
                self.conv.ptr,
                self.xs.ptr,
                &self.gu,
                8,
            )?;
            words(
                &self.kernels["swiglu"],
                s,
                [8, 6, 1],
                &[
                    self.gu.ptr,
                    self.act.ptr,
                    self.axs.ptr,
                    1e30f32.to_bits() as u64,
                ],
            )?;
            self.row(e, 6, &format!("{base}.down"), self.act.ptr, self.axs.ptr, 8)?;
            words(
                &self.kernels["draft_conv_post"],
                s,
                [8, 4, 1],
                &[
                    self.reduced.ptr,
                    self.dyns.ptr,
                    self.w(&format!("{base}.m_base"))?,
                    self.x.ptr,
                    self.x.ptr,
                ],
            )?;
        }
        self.norm(e, self.x.ptr + 4096 * 2, self.w("norm")?, 7)?;
        if self.q4_head {
            // draft_norm also left the rows' group sums in xs.
            unsafe {
                self.q4[DRAFT_HEAD].run(
                    s,
                    self.norm.ptr,
                    4096,
                    self.xs.ptr,
                    [
                        e.m.pointer(e.p, "weights.draft_head.weight")?,
                        e.m.pointer(e.p, "weights.draft_head.scales")?,
                        e.m.pointer(e.p, "weights.draft_head.biases")?,
                    ],
                    &self.logits,
                    &self.part,
                    7,
                )?;
            }
        } else {
            words(
                &self.kernels["verify_head"],
                s,
                [1, 605, 1],
                &[
                    self.norm.ptr,
                    e.m.pointer(e.p, "weights.head.weight")?,
                    self.logits.ptr,
                    self.logits.ptr,
                    7,
                    4096,
                ],
            )?;
        }
        unsafe {
            self.blas
                .selector(s, self.norm.ptr, self.w("hproj")?, &self.projected, 7)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn probe(
    p: &Prepared,
    manifest: &Path,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: std::net::SocketAddr,
) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let e = Engine::new(p, rank, root, native, nccl, master, false, true)?;
    let mut draft = Drafter::new(&e, manifest, root, native)?;
    let hash =
        |b: &Buffer| -> Result<String> { Ok(format!("{:x}", Sha256::digest(b.read(0, b.bytes)?))) };
    let mut cases = Vec::new();
    for base in [0usize, 1984, 4032, 1048384] {
        let mut reference = None;
        for block in [1usize, 64] {
            draft.reset(&e)?;
            draft.contexts[draft.active].end = base;
            draft.context().pos.write(0, &(base as i64).to_le_bytes())?;
            draft.context().lo.write(0, &(base as i64).to_le_bytes())?;
            for row in (0..128).step_by(block) {
                let count = (128 - row).min(block);
                let bytes: Vec<u8> = (0..count * 20480)
                    .flat_map(|i| {
                        half::bf16::from_f32(
                            (((base + row + i / 20480) * 13 + i % 20480 * 7) % 127) as f32 / 64.
                                - 1.,
                        )
                        .to_bits()
                        .to_le_bytes()
                    })
                    .collect();
                e.sync()?;
                draft.taps.write(0, &bytes)?;
                draft.add_taps(&e, count)?;
            }
            e.sync()?;
            let state: Vec<_> = draft
                .context()
                .kc
                .iter()
                .chain(&draft.context().vc)
                .map(hash)
                .collect::<Result<_>>()?;
            draft.set_pending(&e, 198)?;
            draft.block(&e)?;
            e.sync()?;
            let logits = hash(&draft.logits)?;
            let projected = hash(&draft.projected)?;
            ensure!(
                crate::front::bf_values(&draft.logits.read(0, draft.logits.bytes)?)
                    .iter()
                    .all(|v| v.is_finite()),
                "nonfinite draft logits"
            );
            if let Some((a, b, c)) = &reference {
                ensure!(
                    &state == a && &logits == b && &projected == c,
                    "draft context batch mismatch"
                );
            } else {
                reference = Some((state.clone(), logits.clone(), projected.clone()));
            }
            let graph = unsafe { e.stream.capture(|| draft.block(&e))? };
            for _ in 0..2 {
                unsafe {
                    graph.launch(&e.stream)?;
                }
                e.sync()?;
                ensure!(
                    hash(&draft.logits)? == logits && hash(&draft.projected)? == projected,
                    "draft graph changed output"
                );
            }
            ensure!(
                draft.context().pos.read(0, 8)? == (draft.end() as i64).to_le_bytes(),
                "draft proposal advanced committed cursor"
            );
            cases.push(serde_json::json!({"base":base,"end":draft.end(),"context_batch":block,"logits_sha256":logits,"selector_sha256":projected,"graph_exact":true,"context_state_sha256":state}));
        }
    }
    // Each slot keeps only its own committed K/V and cursors. Deliberately
    // overwrite every shared scratch buffer and replay a different graph in
    // between visits; this catches captures bound to the wrong owner's pointers.
    let policy = crate::sampling::Sampling::new(731, 1., 20, 0.95)?;
    let mut retained = Vec::new();
    for (slot, base) in [0usize, 1984, 4032, 1048320].into_iter().enumerate() {
        draft.bind(&e, slot)?;
        draft.start_context(&e, base)?;
        for row in [0usize, 64] {
            let bytes = (0..64 * 20480)
                .flat_map(|i| {
                    half::bf16::from_f32(
                        (((base + row + i / 20480) * 13 + i % 20480 * 7 + slot) % 127) as f32 / 64.
                            - 1.,
                    )
                    .to_bits()
                    .to_le_bytes()
                })
                .collect::<Vec<_>>();
            e.sync()?;
            draft.taps.write(0, &bytes)?;
            draft.add_taps(&e, 64)?;
        }
        let proposed = draft.propose(&e, 198 + slot as i32, policy)?;
        let state = draft
            .context()
            .kc
            .iter()
            .chain(&draft.context().vc)
            .chain([&draft.context().pos, &draft.context().lo])
            .map(hash)
            .collect::<Result<Vec<_>>>()?;
        retained.push((draft.end(), state, proposed, hash(&draft.projected)?));
    }
    for slot in [2usize, 0, 3, 1, 3, 0, 1, 2] {
        draft.bind(&e, slot)?;
        let proposed = draft.propose(&e, 198 + slot as i32, policy)?;
        let state = draft
            .context()
            .kc
            .iter()
            .chain(&draft.context().vc)
            .chain([&draft.context().pos, &draft.context().lo])
            .map(hash)
            .collect::<Result<Vec<_>>>()?;
        let (end, expected, tokens, projection) = &retained[slot];
        ensure!(
            draft.end() == *end
                && state == *expected
                && proposed == *tokens
                && hash(&draft.projected)? == *projection,
            "retained drafter slot drift"
        );
    }
    let snapshots = snapshot::probe(&e, &mut draft)?;
    let backlogs = backlog::probe(&e, &mut draft)?;
    Ok(
        serde_json::json!({"rank":rank,"cases":cases,"retained_slots":4,
        "retained_graph_replays":8,"exact_retained_state_and_proposals":true,
        "snapshots":snapshots,
        "accepted_feature_backlogs":backlogs,
        "scope":"Five-layer DFlash2 forward with synthetic committed taps; 1-vs-64 context updates, ring/graph equality and four retained K/V owners sharing weights/scratch. Not serving."}),
    )
}
