//! Bounded logical prefill with shared attention scratch and full-chunk experts.
use super::{
    Attention, Engine,
    prefill_split_pipe::{
        self, FFN_SLOTS, Field, Pending, Pipe as SplitPipe, ShareToken, plan::Plan,
    },
};
use crate::{
    cuda::{Buffer, Device, Kernel, Pinned},
    dsa::words,
    fast_experts::FastExperts,
    front::{load_kernel, load_native},
    grouping::Grouping,
    prefill::{FastKda, Gemms, Matrix},
};
use anyhow::{Result, ensure};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    path::Path,
    rc::Rc,
};
const MAX: usize = 512;
pub(super) struct Inputs<'a> {
    pub layer: usize,
    pub position: usize,
    pub norm: u64,
    pub qr: u64,
    pub qa: u64,
    pub attended: u64,
    pub rows: usize,
    pub dsa: &'a crate::dsa::Dsa,
}
#[derive(Clone, Copy)]
pub(super) struct NextPre {
    pub layer: usize,
    pub branch: &'static str,
    pub norm: &'static str,
}
const TAP_LAYERS: [usize; 5] = [5, 14, 24, 33, 42];
/// An expert layer's row-split finish carried into the next layer's attention:
/// each piece's exchange, HC post and next-layer share run two pieces ahead of
/// the attention piece that reads them.
struct Deferred {
    rows: usize,
    next: NextPre,
    tap: Option<usize>,
    pending: [Option<Pending>; FFN_SLOTS],
    shares: [Option<ShareToken>; FFN_SLOTS],
    issued: usize,
    posted: usize,
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(super) struct RowStats {
    pub(super) batches: u64,
    pub(super) fallback_batches: u64,
    pub(super) embedded_rows: u64,
    pub(super) post_rows: u64,
    pub(super) partial_exchanges: u64,
    pub(super) shared_pieces: u64,
    pub(super) captured_rows: u64,
}
pub(super) struct Batch {
    pub(super) max_rows: usize,
    pub(super) attention_rows: usize,
    pub(super) hc_slabs: bool,
    pub(super) overlap: bool,
    pub(super) hc_pipeline: bool,
    direct: crate::prefill_partial::Policy,
    pub(super) partial_stats: Cell<crate::prefill_partial::Stats>,
    pipe: RefCell<Option<super::prefill_pipe::Pipe>>,
    split_pipe: RefCell<Option<SplitPipe>>,
    row_policy: prefill_split_pipe::plan::Policy,
    rank: usize,
    pub(super) row_stats: Cell<RowStats>,
    k: BTreeMap<&'static str, Kernel>,
    matrices: Vec<BTreeMap<&'static str, Matrix>>,
    shared_x: Vec<bool>,
    kda: FastKda,
    experts: FastExperts,
    group: Grouping,
    cast: Kernel,
    ids: Buffer,
    ids_host: RefCell<Pinned>,
    /// (row, vision table row) pairs of this slab's image rows.
    vision_map: Buffer,
    x: Buffer,
    norm: Buffer,
    xs: Buffer,
    hcpart: Buffer,
    post: Buffer,
    comb: Buffer,
    proj: Buffer,
    ga: Buffer,
    gg: Buffer,
    out: Buffer,
    send: Buffer,
    gathered: Buffer,
    gu: Buffer,
    activation: Buffer,
    sums: Buffer,
    router_part: Buffer,
    router_logits: Buffer,
    picks: Buffer,
    weights: Buffer,
    shared: Buffer,
    qr: Buffer,
    latent: Buffer,
    q: Buffer,
    qa: Buffer,
    attended: Buffer,
    expanded: Buffer,
    po: Buffer,
    pm: Buffer,
    pl: Buffer,
}
impl Batch {
    pub(super) fn new(d: &Rc<Device>, e: &Engine<'_>, root: &Path, native: &Path) -> Result<Self> {
        Self::new_with_rows(d, e, root, native, MAX)
    }
    pub(super) fn new_with_rows(
        d: &Rc<Device>,
        e: &Engine<'_>,
        root: &Path,
        native: &Path,
        max_rows: usize,
    ) -> Result<Self> {
        ensure!(
            [512, 1024, 2048, 4096, 8192].contains(&max_rows),
            "prefill scratch row bound"
        );
        let attention_rows = max_rows.min(1024);
        let direct = e.direct_prefill;
        let mut k = BTreeMap::new();
        for name in [
            "embed",
            "prefill_hc",
            "hc_finish",
            "prefill_hc_post",
            "swiglu",
            "shared_swiglu",
            "prefill_router",
            "router_sum",
            "topk",
            "combine",
            "rms1536",
            "rms512",
            "prefill_absorb",
            "prefill_expand",
            "latent_write",
            "prefill_latent_chunks32",
            "latent_merge1",
        ] {
            let alias = if name == "combine" && direct.enabled() {
                "combine_bf16"
            } else {
                name
            };
            k.insert(name, load_kernel(d, root, alias)?);
        }
        let gemms = Gemms::new(d, root, native)?;
        let mut matrices = Vec::new();
        let mut shared_x = Vec::new();
        for layer in 0..45 {
            let mut specs = if layer % 4 == 3 {
                vec![("dsa.proj", false), ("dsa.q_b", false), ("dsa.o", true)]
            } else {
                vec![
                    ("kda.proj", false),
                    ("kda.fb", false),
                    ("kda.gb", false),
                    ("kda.o", true),
                ]
            };
            specs.extend(if layer < 3 {
                vec![("mlp.gu", false), ("mlp.down", true)]
            } else {
                vec![("moe.shared.gu", false), ("moe.shared.down", true)]
            });
            let mut map = BTreeMap::new();
            for (name, f32out) in specs {
                map.insert(
                    name,
                    gemms.prepare_f8(
                        d,
                        &e.stream,
                        e.p,
                        &e.m,
                        &format!("weights.layers[{layer}].{name}"),
                        direct.f32_output(name, f32out),
                    )?,
                );
            }
            matrices.push(map);
            shared_x.push(
                layer < 3
                    || e.p
                        .read_tensor(&format!("weights.layers[{layer}].moe.experts.suh_g"))?
                        == e.p
                            .read_tensor(&format!("weights.layers[{layer}].moe.experts.suh_u"))?,
            );
        }
        let a = |width: usize, bytes: usize| d.alloc(max_rows * width * bytes);
        let b = |width: usize, bytes: usize| d.alloc(attention_rows * width * bytes);
        let this = Self {
            max_rows,
            attention_rows,
            hc_slabs: false,
            overlap: false,
            hc_pipeline: false,
            direct,
            partial_stats: Cell::new(crate::prefill_partial::Stats::default()),
            pipe: RefCell::new(None),
            split_pipe: RefCell::new(None),
            row_policy: e.row_prefill,
            rank: e.rank,
            row_stats: Cell::new(RowStats::default()),
            k,
            matrices,
            shared_x,
            kda: FastKda::new(d, root, native, attention_rows)?,
            experts: FastExperts::new(d, native, max_rows)?,
            group: Grouping::new(d, native, max_rows)?,
            cast: load_native(d, &native.join("prefill.cubin"), "cast_bf16", 256)?,
            ids: a(1, 4)?,
            ids_host: RefCell::new(Pinned::new(d, max_rows * 4)?),
            vision_map: a(2, 4)?,
            x: a(4 * 4096, 2)?,
            norm: a(4096, 2)?,
            xs: a(64, 4)?,
            hcpart: a(16 * 32, 4)?,
            post: a(4, 4)?,
            comb: a(16, 4)?,
            proj: a(6416, 2)?,
            ga: b(2048, 2)?,
            gg: b(2048, 2)?,
            out: a(4096, direct.element_bytes())?,
            send: b(4096, 2)?,
            gathered: b(4 * 4096, 2)?,
            gu: a(6144, 2)?,
            activation: a(3072, 2)?,
            sums: a(64, 4)?,
            router_part: a(8 * 288, 4)?,
            router_logits: a(288, 4)?,
            picks: a(9, 4)?,
            weights: a(9, 4)?,
            shared: a(4096, 4)?,
            qr: b(1536, 2)?,
            latent: b(512, 2)?,
            q: b(4096, 2)?,
            qa: b(16 * 512, 2)?,
            attended: b(16 * 512, 4)?,
            expanded: b(4096, 2)?,
            po: b(16 * 512, 4)?,
            pm: b(16, 4)?,
            pl: b(16, 4)?,
        };
        e.sync()?;
        Ok(this)
    }
    fn pre(&self, e: &Engine<'_>, layer: usize, branch: &str, nw: &str, rows: usize) -> Result<()> {
        let slab = if self.hc_slabs { 384 } else { rows };
        for offset in (0..rows).step_by(slab) {
            self.pre_range(e, layer, branch, nw, offset, (rows - offset).min(slab))?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn pre_range(
        &self,
        e: &Engine<'_>,
        layer: usize,
        branch: &str,
        nw: &str,
        offset: usize,
        rows: usize,
    ) -> Result<()> {
        let w = |n: &str| e.m.pointer(e.p, &format!("weights.layers[{layer}].{n}"));
        let s = &e.stream;
        words(
            &self.k["prefill_hc"],
            s,
            [rows.div_ceil(64) as u32, 16, 1],
            &[
                self.x.ptr + (offset * 32768) as u64,
                w(&format!("{branch}.fn"))?,
                self.hcpart.ptr,
                rows as u64,
            ],
        )?;
        words(
            &self.k["hc_finish"],
            s,
            [rows as u32, 1, 1],
            &[
                self.x.ptr + (offset * 32768) as u64,
                self.hcpart.ptr,
                w(&format!("{branch}.base"))?,
                w(&format!("{branch}.scale"))?,
                w(nw)?,
                self.norm.ptr + (offset * 8192) as u64,
                self.xs.ptr + (offset * 256) as u64,
                self.post.ptr + (offset * 16) as u64,
                self.comb.ptr + (offset * 64) as u64,
                1e-5f32.to_bits() as u64,
                1e-6f32.to_bits() as u64,
            ],
        )
    }
    pub(super) fn prepare_pipe(&self, d: &Rc<Device>) -> Result<()> {
        ensure!(
            self.pipe.borrow().is_none(),
            "prefill pipe already prepared"
        );
        *self.pipe.borrow_mut() = Some(super::prefill_pipe::Pipe::new(d, self.attention_rows)?);
        if self.row_policy.enabled() {
            ensure!(
                self.split_pipe.borrow().is_none(),
                "row pipe already prepared"
            );
            *self.split_pipe.borrow_mut() =
                Some(SplitPipe::new(d, self.rank, self.attention_rows)?);
        }
        Ok(())
    }
    pub(super) fn cast_rows(
        &self,
        e: &Engine<'_>,
        offset: usize,
        rows: usize,
        send: &Buffer,
    ) -> Result<()> {
        self.cast_region(e, offset, rows, send, 0)
    }
    fn cast_region(
        &self,
        e: &Engine<'_>,
        offset: usize,
        rows: usize,
        send: &Buffer,
        destination: usize,
    ) -> Result<()> {
        ensure!(
            destination <= send.bytes && rows * 8192 <= send.bytes - destination,
            "partial destination range"
        );
        if self.direct.enabled() {
            // FFN and non-overlapped attention retain bounded BF16 scratch.
            // The overlapped attention producer instead writes the pipe slot.
            unsafe {
                send.copy_from(
                    &e.stream,
                    destination,
                    &self.out,
                    offset * 4096 * 2,
                    rows * 4096 * 2,
                )?;
            }
            let mut stats = self.partial_stats.get();
            stats.bf16_copy_calls += 1;
            stats.bf16_copy_elements += (rows * 4096) as u64;
            self.partial_stats.set(stats);
            return Ok(());
        }
        words(
            &self.cast,
            &e.stream,
            [(rows * 4096).div_ceil(256) as u32, 1, 1],
            &[
                self.out.ptr + (offset * 4096 * 4) as u64,
                send.ptr + destination as u64,
                (rows * 4096) as u64,
            ],
        )?;
        let mut stats = self.partial_stats.get();
        stats.fp32_cast_calls += 1;
        stats.fp32_cast_elements += (rows * 4096) as u64;
        self.partial_stats.set(stats);
        Ok(())
    }
    pub(super) fn post_rows(
        &self,
        e: &Engine<'_>,
        offset: usize,
        rows: usize,
        gathered: &Buffer,
        next: Option<NextPre>,
    ) -> Result<()> {
        self.post_range(e, offset, rows, gathered, 0, rows * 4096, next)
    }
    #[allow(clippy::too_many_arguments)]
    fn post_range(
        &self,
        e: &Engine<'_>,
        offset: usize,
        rows: usize,
        gathered: &Buffer,
        first: usize,
        stride: usize,
        next: Option<NextPre>,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let slab = if next.is_some() { 384 } else { rows };
        for sub in (0..rows).step_by(slab) {
            let count = (rows - sub).min(slab);
            let base = offset + sub;
            let x = self.x.ptr + (base * 4 * 4096 * 2) as u64;
            words(
                &self.k["prefill_hc_post"],
                &e.stream,
                [count as u32, 4, 1],
                &[
                    x,
                    x,
                    gathered.ptr + ((first + sub) * 4096 * 2) as u64,
                    self.post.ptr + (base * 4 * 4) as u64,
                    self.comb.ptr + (base * 16 * 4) as u64,
                    stride as u64,
                ],
            )?;
            if let Some(n) = next {
                self.pre_range(e, n.layer, n.branch, n.norm, base, count)?;
            }
        }
        Ok(())
    }
    fn share_norm(
        &self,
        e: &Engine<'_>,
        pipe: &mut SplitPipe,
        offset: usize,
        rows: usize,
    ) -> Result<ShareToken> {
        let token = unsafe {
            pipe.share(
                &e.comm,
                &e.stream,
                rows,
                &[
                    Field {
                        buffer: &self.norm,
                        base: offset * 8192,
                        row_bytes: 8192,
                    },
                    Field {
                        buffer: &self.xs,
                        base: offset * 256,
                        row_bytes: 256,
                    },
                ],
            )?
        };
        let mut stats = self.row_stats.get();
        stats.shared_pieces += 1;
        self.row_stats.set(stats);
        Ok(token)
    }
    fn finish_owned(
        &self,
        e: &Engine<'_>,
        pipe: &mut SplitPipe,
        pending: Pending,
        offset: usize,
        rows: usize,
        next: Option<NextPre>,
    ) -> Result<Option<ShareToken>> {
        let view = pipe.wait(&e.stream, pending)?;
        let owned = view.owned;
        self.post_range(
            e,
            offset + owned.first,
            owned.count,
            view.partial,
            owned.first,
            view.rank_stride,
            next,
        )?;
        let mut stats = self.row_stats.get();
        stats.post_rows += owned.count as u64;
        self.row_stats.set(stats);
        next.map(|_| self.share_norm(e, pipe, offset, rows))
            .transpose()
    }
    fn finish_split(
        &self,
        e: &Engine<'_>,
        rows: usize,
        next: Option<NextPre>,
        pipe: &mut SplitPipe,
    ) -> Result<()> {
        pipe.begin()?;
        let mut pending = None;
        for offset in (0..rows).step_by(self.attention_rows) {
            let count = (rows - offset).min(self.attention_rows);
            let (buffer, destination) = pipe.producer_region(count)?;
            self.cast_region(e, offset, count, buffer, destination)?;
            let issued = unsafe { pipe.issue(&e.comm, &e.stream, count)? };
            let mut stats = self.row_stats.get();
            stats.partial_exchanges += 1;
            self.row_stats.set(stats);
            if let Some((previous, first, n)) = pending.take() {
                self.finish_owned(e, pipe, previous, first, n, next)?;
            }
            pending = Some((issued, offset, count));
        }
        if let Some((previous, first, n)) = pending {
            self.finish_owned(e, pipe, previous, first, n, next)?;
        }
        pipe.wait_shares(&e.stream)
    }
    /// Stage every expert piece in its own slot and issue the first two
    /// exchanges; the rest runs inside the next layer's attention.
    fn defer_finish(
        &self,
        e: &Engine<'_>,
        rows: usize,
        next: NextPre,
        tap: Option<usize>,
        pipe: &mut SplitPipe,
    ) -> Result<Deferred> {
        pipe.begin()?;
        for offset in (0..rows).step_by(self.attention_rows) {
            let count = (rows - offset).min(self.attention_rows);
            let (buffer, destination) =
                pipe.ffn_producer_region(offset / self.attention_rows, count)?;
            self.cast_region(e, offset, count, buffer, destination)?;
        }
        let mut deferred = Deferred {
            rows,
            next,
            tap,
            pending: Default::default(),
            shares: [None; FFN_SLOTS],
            issued: 0,
            posted: 0,
        };
        self.issue_deferred(e, &mut deferred, pipe)?;
        self.issue_deferred(e, &mut deferred, pipe)?;
        Ok(deferred)
    }
    fn issue_deferred(&self, e: &Engine<'_>, d: &mut Deferred, pipe: &mut SplitPipe) -> Result<()> {
        let offset = d.issued * self.attention_rows;
        if offset >= d.rows {
            return Ok(());
        }
        let count = (d.rows - offset).min(self.attention_rows);
        d.pending[d.issued] = Some(unsafe { pipe.issue_ffn(&e.comm, &e.stream, d.issued, count)? });
        d.issued += 1;
        let mut stats = self.row_stats.get();
        stats.partial_exchanges += 1;
        self.row_stats.set(stats);
        Ok(())
    }
    fn post_deferred(&self, e: &Engine<'_>, d: &mut Deferred, pipe: &mut SplitPipe) -> Result<()> {
        let offset = d.posted * self.attention_rows;
        if offset >= d.rows {
            return Ok(());
        }
        let count = (d.rows - offset).min(self.attention_rows);
        let pending = d.pending[d.posted]
            .take()
            .ok_or_else(|| anyhow::anyhow!("expert piece posted before its exchange"))?;
        d.shares[d.posted] = self.finish_owned(e, pipe, pending, offset, count, Some(d.next))?;
        if let Some(tap) = d.tap {
            self.capture_piece(e, d.rows, offset, count, tap)?;
        }
        d.posted += 1;
        Ok(())
    }
    /// The MoE router for rows [first, first + count). Each row's logits, picks
    /// and weights are the same alone or in a window.
    fn route(&self, e: &Engine<'_>, layer: usize, first: usize, count: usize) -> Result<()> {
        let w = |n: &str| e.m.pointer(e.p, &format!("weights.layers[{layer}].{n}"));
        let s = &e.stream;
        let logits = self.router_logits.ptr + (first * 288 * 4) as u64;
        words(
            &self.k["prefill_router"],
            s,
            [count.div_ceil(128) as u32, 9, 8],
            &[
                self.norm.ptr + (first * 4096 * 2) as u64,
                w("moe.router")?,
                self.router_part.ptr,
                count as u64,
                4096,
            ],
        )?;
        words(
            &self.k["router_sum"],
            s,
            [(count * 288).div_ceil(1024) as u32, 1, 1],
            &[self.router_part.ptr, logits, (count * 288) as u64],
        )?;
        words(
            &self.k["topk"],
            s,
            [count as u32, 1, 1],
            &[
                logits,
                w("moe.bias")?,
                self.picks.ptr + (first * 9 * 4) as u64,
                self.weights.ptr + (first * 9 * 4) as u64,
                2.5f32.to_bits() as u64,
            ],
        )
    }
    /// Retained feature rows of this rank's owned rows in one piece.
    fn capture_piece(
        &self,
        e: &Engine<'_>,
        rows: usize,
        offset: usize,
        count: usize,
        tap: usize,
    ) -> Result<()> {
        let Some(taps) = &e.taps else {
            return Ok(());
        };
        let owned = Plan::new(count, self.rank)?.own();
        let end = offset + owned.first + owned.count;
        let first = (offset + owned.first)
            .max(rows.saturating_sub(taps.cap))
            .min(end);
        taps.capture_range(&e.stream, &self.x, first, end - first, tap)?;
        let mut stats = self.row_stats.get();
        stats.captured_rows += (end - first) as u64;
        self.row_stats.set(stats);
        Ok(())
    }
    fn finish(&self, e: &Engine<'_>, rows: usize, next: Option<NextPre>) -> Result<()> {
        let mut owner = self.pipe.borrow_mut();
        if self.overlap {
            let pipe = owner
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("unprepared prefill pipe"))?;
            pipe.begin()?;
            for offset in (0..rows).step_by(self.attention_rows) {
                pipe.issue(
                    self,
                    e,
                    offset,
                    (rows - offset).min(self.attention_rows),
                    next,
                    false,
                )?;
            }
            return pipe.drain(self, e);
        }
        for offset in (0..rows).step_by(self.attention_rows) {
            let count = (rows - offset).min(self.attention_rows);
            self.cast_rows(e, offset, count, &self.send)?;
            unsafe {
                e.gather_prefix(&self.send, &self.gathered, count * 4096 * 2)?;
            }
            self.post_rows(e, offset, count, &self.gathered, next)?;
        }
        Ok(())
    }
    fn dense(&self, s: &crate::cuda::Stream, input: Inputs<'_>) -> Result<()> {
        let rows = input.rows;
        let a = input.dsa;
        words(
            &self.k["prefill_latent_chunks32"],
            s,
            [rows.div_ceil(2) as u32, 1, 1],
            &[
                input.qa,
                a.cache.ptr,
                a.cache.ptr,
                a.position.ptr,
                self.po.ptr,
                self.pm.ptr,
                self.pl.ptr,
                rows as u64,
                a.page_table.ptr,
            ],
        )?;
        words(
            &self.k["latent_merge1"],
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
        Ok(())
    }
    pub(super) fn run(&self, e: &Engine<'_>, position: usize, tokens: &[i32]) -> Result<i32> {
        self.run_with(e, position, tokens, 512, |input| {
            self.dense(&e.stream, input)
        })
    }
    pub(super) fn run_with(
        &self,
        e: &Engine<'_>,
        position: usize,
        tokens: &[i32],
        capacity: usize,
        do_attention: impl Fn(Inputs<'_>) -> Result<()>,
    ) -> Result<i32> {
        self.run_profiled(e, position, tokens, capacity, do_attention, None)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_profiled(
        &self,
        e: &Engine<'_>,
        position: usize,
        tokens: &[i32],
        capacity: usize,
        do_attention: impl Fn(Inputs<'_>) -> Result<()>,
        trace: Option<&mut crate::phase_profile::Trace>,
    ) -> Result<i32> {
        self.run_profiled_output(e, position, tokens, capacity, do_attention, trace, true)?
            .ok_or_else(|| anyhow::anyhow!("requested prefill head missing"))
    }
    /// Intermediate service slabs need recurrence/cache state and target taps,
    /// but no vocabulary projection or selected token. The diagnostic entry
    /// points above retain their complete head contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_profiled_output(
        &self,
        e: &Engine<'_>,
        position: usize,
        tokens: &[i32],
        capacity: usize,
        do_attention: impl Fn(Inputs<'_>) -> Result<()>,
        trace: Option<&mut crate::phase_profile::Trace>,
        emit_head: bool,
    ) -> Result<Option<i32>> {
        let result = self.run_profiled_output_inner(
            e,
            position,
            tokens,
            capacity,
            do_attention,
            trace,
            emit_head,
        );
        let joined = e.prefetch_join();
        result.and_then(|value| joined.map(|()| value))
    }
    #[allow(clippy::too_many_arguments)]
    fn run_profiled_output_inner(
        &self,
        e: &Engine<'_>,
        position: usize,
        tokens: &[i32],
        capacity: usize,
        do_attention: impl Fn(Inputs<'_>) -> Result<()>,
        mut trace: Option<&mut crate::phase_profile::Trace>,
        emit_head: bool,
    ) -> Result<Option<i32>> {
        let mut mark = |label, layer| -> Result<()> {
            if let Some(t) = trace.as_deref_mut() {
                t.mark(&e.stream, label, layer)?;
            }
            Ok(())
        };
        let rows = tokens.len();
        let row_split =
            self.row_policy
                .applies(rows, self.attention_rows, self.overlap, self.hc_pipeline)?;
        let hint_rows = if self.overlap { 0 } else { rows };
        ensure!(
            !self.hc_pipeline || (self.overlap && self.hc_slabs),
            "HC pipeline requires overlap and slabs"
        );
        ensure!(
            rows > 0
                && rows <= self.max_rows
                && position.is_multiple_of(64)
                && position + rows <= capacity,
            "bounded aligned prefill required"
        );
        ensure!(
            tokens.iter().all(|&t| crate::vision_input::prompt_token(t)),
            "token outside vocabulary"
        );
        let s = &e.stream;
        let mut split_owner = self.split_pipe.borrow_mut();
        let mut split = if row_split {
            Some(
                split_owner
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("unprepared split pipe"))?,
            )
        } else {
            None
        };
        let mut row_stats = self.row_stats.get();
        if row_split {
            row_stats.batches += 1;
        } else {
            row_stats.fallback_batches += 1;
        }
        self.row_stats.set(row_stats);
        // A batch owns this pinned staging area until the owner stream completes.
        // Enqueue the upload on that same stream before embedding reads the IDs.
        e.sync()?;
        let (ids, map) = e.vision_ids(tokens)?;
        let images = map.len() / 2;
        if images > 0 {
            self.vision_map.write(
                0,
                &map.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
        }
        {
            let mut host = self.ids_host.borrow_mut();
            for (i, t) in ids.iter().enumerate() {
                host.bytes_mut()[i * 4..i * 4 + 4].copy_from_slice(&t.to_le_bytes());
            }
            unsafe {
                host.upload_to(s, &self.ids, 0, 0, rows * 4)?;
            }
        }
        mark("embedding", 45)?;
        for attn in &e.attention {
            if let Attention::Dsa(a) = attn {
                a.set_position(s, position)?;
            }
        }
        if let Some(pipe) = split.as_deref_mut() {
            pipe.begin()?;
            for offset in (0..rows).step_by(self.attention_rows) {
                let count = (rows - offset).min(self.attention_rows);
                let owned = Plan::new(count, self.rank)?.own();
                let first = offset + owned.first;
                if owned.count > 0 {
                    words(
                        &self.k["embed"],
                        s,
                        [owned.count as u32, 64, 1],
                        &[
                            self.ids.ptr + (first * 4) as u64,
                            e.m.pointer(e.p, "weights.embed")?,
                            self.x.ptr + (first * 32768) as u64,
                        ],
                    )?;
                    e.vision_inject(
                        self.x.ptr,
                        &self.vision_map,
                        images,
                        first,
                        first + owned.count,
                    )?;
                    self.pre_range(e, 0, "attn_hc", "in_norm", first, owned.count)?;
                }
                let mut stats = self.row_stats.get();
                stats.embedded_rows += owned.count as u64;
                self.row_stats.set(stats);
                self.share_norm(e, pipe, offset, count)?;
            }
            pipe.wait_shares(s)?;
        } else {
            words(
                &self.k["embed"],
                s,
                [rows as u32, 64, 1],
                &[self.ids.ptr, e.m.pointer(e.p, "weights.embed")?, self.x.ptr],
            )?;
            e.vision_inject(self.x.ptr, &self.vision_map, images, 0, rows)?;
        }
        let mut deferred: Option<Deferred> = None;
        for (layer, attn) in e.attention.iter().enumerate() {
            let w = |n: &str| e.m.pointer(e.p, &format!("weights.layers[{layer}].{n}"));
            let mm = &self.matrices[layer];
            mark("attention_hc_pre", layer)?;
            if !row_split && (!self.hc_pipeline || layer == 0) {
                self.pre(e, layer, "attn_hc", "in_norm", rows)?;
            }
            mark(if layer % 4 == 3 { "dsa" } else { "kda" }, layer)?;
            let mut pipe_owner = self.pipe.borrow_mut();
            let mut pipe = if self.overlap && !row_split {
                Some(
                    pipe_owner
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("unprepared prefill pipe"))?,
                )
            } else {
                None
            };
            if let Some(p) = pipe.as_deref_mut() {
                p.begin()?;
            }
            let (projection, width) = match attn {
                Attention::Kda(_) => ("kda.proj", 6416),
                Attention::Dsa(_) => ("dsa.proj", 2048),
            };
            if let (Some(p), Some(d)) = (split.as_deref_mut(), deferred.as_mut()) {
                p.begin_attention()?;
                for _ in 0..2 {
                    self.post_deferred(e, d, p)?;
                    self.issue_deferred(e, d, p)?;
                }
            } else {
                if let Some(p) = split.as_deref_mut() {
                    p.begin()?;
                }
                // One input projection for the whole chunk: each output row is the
                // same at any batch height (retained kernels and cuBLAS alike).
                mm[projection].run(s, self.norm.ptr, 4096, self.proj.ptr, rows)?;
            }
            let mut owned_pending = None;
            let mut last_share = None;
            let mut routed_rows = 0;
            for offset in (0..rows).step_by(self.attention_rows) {
                let rows = (rows - offset).min(self.attention_rows);
                let position = position + offset;
                let norm = self.norm.ptr + (offset * 4096 * 2) as u64;
                if let (Some(p), Some(d)) = (split.as_deref_mut(), deferred.as_ref()) {
                    let share = d.shares[offset / self.attention_rows]
                        .ok_or_else(|| anyhow::anyhow!("attention piece before its rows"))?;
                    p.wait_share(s, share)?;
                    mm[projection].run(
                        s,
                        norm,
                        4096,
                        self.proj.ptr + (offset * width * 2) as u64,
                        rows,
                    )?;
                }
                let out = if self.direct.enabled() {
                    if let Some(p) = split.as_deref() {
                        p.producer_pointer(rows)?
                    } else if let Some(p) = pipe.as_deref() {
                        p.producer_pointer(rows)?
                    } else {
                        self.out.ptr + (offset * 4096 * 2) as u64
                    }
                } else {
                    self.out.ptr + (offset * 4096 * 4) as u64
                };
                match attn {
                    Attention::Kda(a) => {
                        let proj = self.proj.ptr + (offset * 6416 * 2) as u64;
                        mm["kda.fb"].run(s, proj + 6144 * 2, 6416, self.ga.ptr, rows)?;
                        mm["kda.gb"].run(s, proj + 6272 * 2, 6416, self.gg.ptr, rows)?;
                        if offset == 0 {
                            e.prefetch_mark(
                                layer,
                                crate::l2_prefetch_plan::Site::AttentionCore,
                                hint_rows,
                            )?;
                        }
                        self.kda.run(
                            s,
                            e.p,
                            &e.m,
                            layer,
                            proj,
                            self.ga.ptr,
                            self.gg.ptr,
                            a.conv.ptr,
                            a.states[position % 2].ptr,
                            a.states[(position + rows) % 2].ptr,
                            position,
                            rows,
                        )?;
                        mm["kda.o"].run(s, self.kda.output.ptr, 2048, out, rows)?;
                    }
                    Attention::Dsa(a) => {
                        a.set_position(s, position)?;
                        let proj = self.proj.ptr + (offset * 2048 * 2) as u64;
                        for (alias, off, name, out, n) in [
                            ("rms1536", 0, "dsa.q_norm", self.qr.ptr, 1536),
                            ("rms512", 3072, "dsa.kv_norm", self.latent.ptr, 512),
                        ] {
                            words(
                                &self.k[alias],
                                s,
                                [rows as u32, 1, 1],
                                &[
                                    proj + off,
                                    2048,
                                    w(name)?,
                                    out,
                                    n,
                                    self.sums.ptr,
                                    1e-5f32.to_bits() as u64,
                                ],
                            )?;
                        }
                        mm["dsa.q_b"].run(s, self.qr.ptr, 1536, self.q.ptr, rows)?;
                        let wk = w("dsa.kv_k.weight")?;
                        let wv = w("dsa.kv_v.weight")?;
                        words(
                            &self.k["prefill_absorb"],
                            s,
                            [128, rows.div_ceil(64) as u32, 1],
                            &[self.q.ptr, wk, wk, wk, self.qa.ptr, rows as u64],
                        )?;
                        if offset == 0 {
                            e.prefetch_mark(
                                layer,
                                crate::l2_prefetch_plan::Site::AttentionCore,
                                hint_rows,
                            )?;
                        }
                        words(
                            &self.k["latent_write"],
                            s,
                            [rows as u32, 1, 1],
                            &[
                                self.latent.ptr,
                                a.cache.ptr,
                                a.cache.ptr,
                                a.position.ptr,
                                a.page_table.ptr,
                            ],
                        )?;
                        do_attention(Inputs {
                            layer,
                            position,
                            norm,
                            qr: self.qr.ptr,
                            qa: self.qa.ptr,
                            attended: self.attended.ptr,
                            rows,
                            dsa: a,
                        })?;
                        words(
                            &self.k["prefill_expand"],
                            s,
                            [64, rows.div_ceil(64) as u32, 1],
                            &[
                                self.attended.ptr,
                                wv,
                                wv,
                                wv,
                                self.expanded.ptr,
                                rows as u64,
                            ],
                        )?;
                        mm["dsa.o"].run(s, self.expanded.ptr, 4096, out, rows)?;
                    }
                }
                if let Some(p) = split.as_deref_mut() {
                    if !self.direct.enabled() {
                        let (buffer, destination) = p.producer_region(rows)?;
                        self.cast_region(e, offset, rows, buffer, destination)?;
                    }
                    let current = unsafe { p.issue(&e.comm, s, rows)? };
                    let next = Some(NextPre {
                        layer,
                        branch: "ffn_hc",
                        norm: "post_norm",
                    });
                    if let Some((previous, first, count)) = owned_pending.take() {
                        last_share = self.finish_owned(e, p, previous, first, count, next)?;
                    }
                    owned_pending = Some((current, offset, rows));
                    let mut stats = self.row_stats.get();
                    stats.partial_exchanges += 1;
                    self.row_stats.set(stats);
                    if self.direct.enabled() {
                        let mut stats = self.partial_stats.get();
                        stats.direct_slot_calls += 1;
                        stats.direct_slot_elements += (rows * 4096) as u64;
                        self.partial_stats.set(stats);
                    }
                    if let Some(d) = deferred.as_mut() {
                        self.post_deferred(e, d, p)?;
                        self.issue_deferred(e, d, p)?;
                    }
                }
                if let Some(p) = pipe.as_deref_mut() {
                    p.issue(
                        self,
                        e,
                        offset,
                        rows,
                        self.hc_pipeline.then_some(NextPre {
                            layer,
                            branch: "ffn_hc",
                            norm: "post_norm",
                        }),
                        self.direct.enabled(),
                    )?;
                    if self.direct.enabled() {
                        let mut stats = self.partial_stats.get();
                        stats.direct_slot_calls += 1;
                        stats.direct_slot_elements += (rows * 4096) as u64;
                        self.partial_stats.set(stats);
                    }
                }
            }
            if let Attention::Dsa(a) = attn {
                a.set_position(s, position)?;
            }
            mark("attention_gather_post", layer)?;
            if let Some(p) = pipe {
                p.drain(self, e)?;
            }
            if let Some(p) = split.as_deref_mut() {
                if let Some((previous, first, count)) = owned_pending.take() {
                    // Route the rows that have arrived while the last piece's
                    // exchange and share finish.
                    if let (true, Some(share)) = (layer >= 3, last_share) {
                        p.wait_share(s, share)?;
                        self.route(e, layer, 0, first)?;
                        routed_rows = first;
                    }
                    self.finish_owned(
                        e,
                        p,
                        previous,
                        first,
                        count,
                        Some(NextPre {
                            layer,
                            branch: "ffn_hc",
                            norm: "post_norm",
                        }),
                    )?;
                }
                p.wait_shares(s)?;
            }
            if let Some(d) = deferred.take() {
                ensure!(
                    d.posted * self.attention_rows >= d.rows
                        && d.pending.iter().all(Option::is_none),
                    "deferred expert finish not drained"
                );
            }
            drop(pipe_owner);
            if !self.overlap {
                e.prefetch_mark(
                    layer,
                    crate::l2_prefetch_plan::Site::AttentionGather,
                    hint_rows,
                )?;
                self.finish(e, rows, None)?;
            }
            mark("ffn_hc_pre", layer)?;
            if !self.hc_pipeline {
                self.pre(e, layer, "ffn_hc", "post_norm", rows)?;
            }
            mark(if layer < 3 { "mlp" } else { "moe" }, layer)?;
            if layer < 3 {
                mm["mlp.gu"].run(s, self.norm.ptr, 4096, self.gu.ptr, rows)?;
                words(
                    &self.k["swiglu"],
                    s,
                    [rows as u32, 6, 1],
                    &[
                        self.gu.ptr,
                        self.activation.ptr,
                        self.sums.ptr,
                        10f32.to_bits() as u64,
                    ],
                )?;
                mm["mlp.down"].run(s, self.activation.ptr, 3072, self.out.ptr, rows)?;
            } else {
                self.route(e, layer, routed_rows, rows - routed_rows)?;
                self.group.run(s, self.picks.ptr, rows)?;
                let routed = || {
                    self.experts.run(
                        s,
                        e.p,
                        &e.m,
                        layer,
                        self.norm.ptr,
                        self.picks.ptr,
                        self.group.ids.ptr,
                        self.group.count.ptr,
                        self.group.members.ptr,
                        rows,
                        self.max_rows,
                        self.shared_x[layer],
                    )
                };
                e.prefetch_experts(layer, hint_rows, &self.group.ids, &self.group.count)?;
                if e.prefetch.is_none() {
                    routed()?;
                }
                mm["moe.shared.gu"].run(s, self.norm.ptr, 4096, self.gu.ptr, rows)?;
                words(
                    &self.k["shared_swiglu"],
                    s,
                    [rows as u32, 1, 1],
                    &[
                        self.gu.ptr,
                        self.activation.ptr,
                        self.sums.ptr,
                        10f32.to_bits() as u64,
                    ],
                )?;
                mm["moe.shared.down"].run(s, self.activation.ptr, 512, self.shared.ptr, rows)?;
                if e.prefetch.is_some() {
                    routed()?;
                }
                words(
                    &self.k["combine"],
                    s,
                    [rows as u32, 4, 1],
                    &[
                        self.experts.output.ptr,
                        self.shared.ptr,
                        self.weights.ptr,
                        self.out.ptr,
                    ],
                )?;
            }
            mark("ffn_gather_post", layer)?;
            e.prefetch_mark(
                layer,
                crate::l2_prefetch_plan::Site::FeedForwardGather,
                hint_rows,
            )?;
            let next = (self.hc_pipeline && layer < 44).then_some(NextPre {
                layer: layer + 1,
                branch: "attn_hc",
                norm: "in_norm",
            });
            let tap = e
                .taps
                .as_ref()
                .and_then(|_| TAP_LAYERS.iter().position(|&x| x == layer));
            match (split.as_deref_mut(), next) {
                (Some(p), Some(n)) => deferred = Some(self.defer_finish(e, rows, n, tap, p)?),
                (Some(p), None) => self.finish_split(e, rows, next, p)?,
                (None, _) => self.finish(e, rows, next)?,
            }
            if let Some(taps) = &e.taps {
                if row_split {
                    if let (Some(tap_layer), None) = (tap, &deferred) {
                        for offset in (0..rows).step_by(self.attention_rows) {
                            let count = (rows - offset).min(self.attention_rows);
                            self.capture_piece(e, rows, offset, count, tap_layer)?;
                        }
                    }
                } else {
                    taps.capture(s, &self.x, rows, layer)?;
                }
            }
        }
        ensure!(
            deferred.is_none(),
            "expert finish deferred past the last layer"
        );
        if let Some(p) = split {
            // Only retained feature rows are externally observable. Share each
            // original ownership interval, splitting the circular destination.
            if let Some(taps) = &e.taps {
                for offset in (0..rows).step_by(self.attention_rows) {
                    let count = (rows - offset).min(self.attention_rows);
                    for span in prefill_split_pipe::plan::retained_spans(
                        position, rows, offset, count, taps.cap,
                    )?
                    .into_iter()
                    .flatten()
                    {
                        unsafe {
                            p.share_clipped(
                                &e.comm,
                                s,
                                count,
                                span.first - offset,
                                span.count,
                                &[Field {
                                    buffer: &taps.rows,
                                    base: span.ring * crate::target_taps::WIDTH,
                                    row_bytes: crate::target_taps::WIDTH,
                                }],
                            )?;
                        }
                    }
                }
            }
            let last_piece = (rows - 1) / self.attention_rows * self.attention_rows;
            let count = rows - last_piece;
            unsafe {
                p.share_clipped(
                    &e.comm,
                    s,
                    count,
                    count - 1,
                    1,
                    &[Field {
                        buffer: &self.x,
                        base: (rows - 1) * 32768,
                        row_bytes: 32768,
                    }],
                )?;
            }
            p.wait_shares(s)?;
            p.begin()?;
        }
        mark("head", 45)?;
        unsafe {
            e.x.copy_from(s, 0, &self.x, (rows - 1) * 4 * 4096 * 2, 4 * 4096 * 2)?;
        }
        if emit_head {
            e.head()?;
        }
        e.prefetch_join()?;
        mark("end", 45)?;
        e.sync()?;
        if emit_head {
            e.selection.read().map(Some)
        } else {
            Ok(None)
        }
    }
}
