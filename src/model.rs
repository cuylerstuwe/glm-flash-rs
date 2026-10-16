//! TP4 native model forward with bounded diagnostic and long-context adapters.
#[path = "drafter.rs"]
mod drafter;
use crate::{
    comm::Nccl,
    cuda::{Buffer, Device, Graph, Kernel, Stream},
    dsa::{Dsa, words},
    front::{fmm, launch_ptrs, load_kernel, reduce},
    moe::Moe,
    prepared::{Loaded, Prepared},
};
use anyhow::{Result, ensure};
pub use drafter::probe as probe_drafter;
#[path = "batch.rs"]
mod batch;
#[path = "long_model.rs"]
mod long;
#[path = "prefill_pipe.rs"]
mod prefill_pipe;
#[path = "prefill_split_pipe.rs"]
mod prefill_split_pipe;
pub use long::run as run_long;
pub use long::run_cached_requests;
pub use long::run_disk_requests;
pub use long::run_owner;
pub use long::run_prefill_head;
pub use long::run_prefill_partial;
pub use long::run_prefill_row;
pub use long::run_priority;
pub use long::run_profile;
pub use long::run_profile_batches;
pub use long::run_profile_hc_pipeline;
pub use long::run_profile_hc_slabs;
pub use long::run_profile_large_batches;
pub use long::run_profile_overlap;
pub use long::run_requests;
pub use long::run_retained_speculation;
pub use long::run_reuse;
pub use long::run_sampled_requests;
pub use long::run_sampling;
pub use long::run_speculative;
pub use long::run_speculative_workloads;
pub use long::run_ssd;
pub use long::run_ssd_controller;
pub use long::run_ssd_cooperative;
pub use long::run_ssd_recovery;
pub use long::run_ssd_restore;
pub use long::run_store;
pub use long::run_termination;
pub use long::run_verifier;
pub use long::run_verifier_sampled;
pub use prefill_split_pipe::probe as probe_prefill_split;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell, collections::BTreeMap, net::SocketAddr, path::Path, rc::Rc, time::Instant,
};
struct Kda {
    conv: Rc<Buffer>,
    states: [Rc<Buffer>; 2],
    proj: Buffer,
}
enum Attention {
    Kda(Kda),
    Dsa(Box<Dsa>),
}

/// One CUDA owner thread. Graphs drop before streams, collectives, modules and
/// buffers. Rc-backed CUDA owners make accidental Send/Sync impossible.
pub struct Engine<'a> {
    taps: Option<crate::target_taps::Taps>,
    batch: Option<batch::Batch>,
    device: Rc<Device>,
    graphs: Option<[Graph; 2]>,
    prefetch: Option<RefCell<crate::l2_prefetch::Owner>>,
    prefetch_identity: [u8; 32],
    direct_prefill: crate::prefill_partial::Policy,
    row_prefill: prefill_split_pipe::plan::Policy,
    stream: Stream,
    roce: Option<crate::roce::Roce>,
    comm: Nccl,
    p: &'a Prepared,
    m: Loaded,
    k: BTreeMap<&'static str, Kernel>,
    chain: Kernel,
    attention: Vec<Attention>,
    moe: Moe,
    selection: crate::select::Greedy,
    rank: usize,
    audit: bool,
    id: Buffer,
    x: Buffer,
    norm: Buffer,
    xs: Buffer,
    post: Buffer,
    comb: Buffer,
    hcpart: Buffer,
    partial: Buffer,
    ga: Buffer,
    gg: Buffer,
    kout: Buffer,
    out: Buffer,
    gathered: Buffer,
    gu: Buffer,
    activation: Buffer,
    sums: Buffer,
    hidden: Buffer,
    logits: Buffer,
    all_logits: Buffer,
    /// The active prefill's image rows: sorted virtual IDs and their device rows.
    vision: RefCell<Option<(Rc<[i32]>, u64)>>,
    vision_rows: Kernel,
    vision_map: Buffer,
    /// The serial forward's token is an image row (`vision_map` holds its pair).
    vision_token: std::cell::Cell<bool>,
}
impl<'a> Engine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: &'a Prepared,
        rank: usize,
        root: &Path,
        native: &Path,
        nccl: &Path,
        master: SocketAddr,
        audit: bool,
        use_roce: bool,
    ) -> Result<Self> {
        let d = Device::new()?;
        let m = p.load(&d)?;
        let stream = d.stream()?;
        let mut k = BTreeMap::new();
        for name in [
            "embed",
            "hc_partial",
            "hc_finish",
            "hc_post",
            "kda_proj",
            "kda_gate",
            "kda_out",
            "reduce2_bf16",
            "reduce4_f32",
            "mlp_gu",
            "swiglu",
            "mlp_down",
            "stream_mean",
            "final_norm",
            "head",
            "conv_shift",
        ] {
            k.insert(name, load_kernel(&d, root, name)?);
        }
        let chain = crate::front::load_native(&d, &native.join("kda.cubin"), "chain_kernel", 1024)?;
        let alloc = |n| -> Result<Buffer> {
            let b = d.alloc(n)?;
            b.zero()?;
            Ok(b)
        };
        let mut attention = Vec::new();
        for layer in 0..45 {
            if layer % 4 == 3 {
                attention.push(Attention::Dsa(Box::new(Dsa::new(&d, root)?)));
            } else {
                attention.push(Attention::Kda(Kda {
                    conv: Rc::new(alloc(3 * 6144 * 2)?),
                    states: [
                        Rc::new(alloc(16 * 128 * 128 * 4)?),
                        Rc::new(alloc(16 * 128 * 128 * 4)?),
                    ],
                    proj: alloc(6416 * 2)?,
                }));
            }
        }
        let moe = Moe::new(&d, root, &native.join("exl3.cubin"))?;
        let id = alloc(4)?;
        let x = alloc(4 * 4096 * 2)?;
        let norm = alloc(4096 * 2)?;
        let xs = alloc(64 * 4)?;
        let post = alloc(4 * 4)?;
        let comb = alloc(16 * 4)?;
        let hcpart = alloc(16 * 32 * 4)?;
        let partial = alloc(8 * 6416 * 4)?;
        let ga = alloc(2048 * 2)?;
        let gg = alloc(2048 * 2)?;
        let kout = alloc(2048 * 2)?;
        let out = alloc(4096 * 4)?;
        let gathered = alloc(4 * 4096 * 4)?;
        let gu = alloc(6144 * 2)?;
        let activation = alloc(3072 * 2)?;
        let sums = alloc(64 * 4)?;
        let hidden = alloc(4096 * 2)?;
        let logits = alloc(38720 * 2)?;
        let all_logits = alloc(154880 * 2)?;
        let comm = Nccl::new(nccl, rank, 4, master)?;
        let roce = if use_roce {
            Some(crate::roce::Roce::new(
                &d,
                &stream,
                &comm,
                &native.join("libalaska_roce.so"),
                rank,
            )?)
        } else {
            None
        };
        let selection = crate::select::Greedy::new(&d, native)?;
        let vision_rows =
            crate::front::load_native(&d, &native.join("vision_rows.cubin"), "vision_rows", 256)?;
        let vision_map = d.alloc(2 * 4)?;

        let mut engine = Self {
            taps: None,
            batch: None,
            device: d,
            graphs: None,
            prefetch: None,
            prefetch_identity: [0; 32],
            direct_prefill: crate::prefill_partial::Policy::default(),
            row_prefill: prefill_split_pipe::plan::Policy::default(),
            stream,
            comm,
            roce,
            p,
            m,
            k,
            chain,
            attention,
            moe,
            selection,
            rank,
            audit,
            id,
            x,
            norm,
            xs,
            post,
            comb,
            hcpart,
            partial,
            ga,
            gg,
            kout,
            out,
            gathered,
            gu,
            activation,
            sums,
            hidden,
            logits,
            all_logits,
            vision: RefCell::new(None),
            vision_rows,
            vision_map,
            vision_token: std::cell::Cell::new(false),
        };
        engine.install_prefetch_env(native)?;
        engine.install_direct_prefill_env()?;
        Ok(engine)
    }
    fn install_direct_prefill_env(&mut self) -> Result<()> {
        let policy = crate::prefill_partial::Policy::environment();
        let rows = prefill_split_pipe::plan::Policy::environment();
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALPFCF02");
        record[8] = u8::from(policy.is_ok() && rows.is_ok());
        if let Ok(policy) = policy {
            record[9..25].copy_from_slice(&policy.identity());
        }
        if let Ok(rows) = rows {
            record[25..41].copy_from_slice(&rows.identity());
        }
        let all = self
            .comm
            .exchange_host(&self.device, &self.stream, &record)?;
        ensure!(
            all.iter().all(|other| other == &record) && record[8] == 1,
            "prefill output/ownership configuration differs/invalid on a rank"
        );
        self.direct_prefill = policy?;
        self.row_prefill = rows?;
        Ok(())
    }
    fn install_prefetch_env(&mut self, native: &Path) -> Result<()> {
        let config = crate::l2_prefetch::Config::from_env(native);
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALL2PF01");
        record[8] = u8::from(config.is_ok());
        if let Ok(config) = &config {
            record[9..41].copy_from_slice(&crate::l2_prefetch::Config::identity(config.as_ref()));
        }
        let all = self
            .comm
            .exchange_host(&self.device, &self.stream, &record)?;
        ensure!(
            all.iter().all(|r| r == &record) && record[8] == 1,
            "L2 candidate configuration differs/invalid on a rank"
        );
        self.prefetch_identity = record[9..41].try_into().unwrap();
        let config = config?;
        let candidate = config
            .as_ref()
            .map(|c| {
                crate::l2_prefetch::Owner::new(&self.device, self.p, &self.m, &c.library, &c.sha256)
            })
            .transpose();
        record[41] = u8::from(candidate.is_ok());
        let all = self
            .comm
            .exchange_host(&self.device, &self.stream, &record)?;
        ensure!(
            all.iter().all(|r| r == &record) && record[41] == 1,
            "L2 candidate initialization failed on a rank"
        );
        self.prefetch = candidate?.map(RefCell::new);
        Ok(())
    }
    fn prefetch_mark(
        &self,
        layer: usize,
        site: crate::l2_prefetch_plan::Site,
        rows: usize,
    ) -> Result<()> {
        if let Some(p) = &self.prefetch {
            p.borrow_mut().mark(&self.stream, layer, site, rows)?;
        }
        Ok(())
    }
    fn prefetch_experts(
        &self,
        layer: usize,
        rows: usize,
        ids: &Buffer,
        count: &Buffer,
    ) -> Result<()> {
        if let Some(p) = &self.prefetch {
            p.borrow_mut()
                .experts(&self.stream, layer, rows, ids, count)?;
        }
        Ok(())
    }
    fn prefetch_join(&self) -> Result<()> {
        if let Some(p) = &self.prefetch {
            p.borrow_mut().join(&self.stream)?;
        }
        Ok(())
    }
    fn prefetch_stats(&self) -> serde_json::Value {
        self.prefetch.as_ref().map_or(serde_json::Value::Null, |p| {
            serde_json::json!(p.borrow().stats())
        })
    }
    fn prefetch_receipt(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled":self.prefetch.is_some(),
            "identity":self.prefetch_identity.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "stats":self.prefetch_stats(),
            "counter_scope":"eager/capture hook calls; graph replays are not counted"
        })
    }
    /// Embedding IDs for prompt rows (image rows read row 0) and the
    /// (row, table row) pairs that image rows overwrite after embedding.
    pub(crate) fn vision_ids(&self, tokens: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
        let vision = self.vision.borrow();
        let mut map = Vec::new();
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if t < crate::vision_input::BASE {
                    return Ok(t);
                }
                let (keys, _) = vision
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("image row without a vision table"))?;
                let k = keys
                    .binary_search(&t)
                    .map_err(|_| anyhow::anyhow!("image row missing from the vision table"))?;
                map.extend([i as i32, k as i32]);
                Ok(0)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((ids, map))
    }
    /// Overwrite the embedded rows of `x` in [lo, hi) listed in `map`.
    pub(crate) fn vision_inject(
        &self,
        x: u64,
        map: &Buffer,
        pairs: usize,
        lo: usize,
        hi: usize,
    ) -> Result<()> {
        if pairs == 0 {
            return Ok(());
        }
        let rows = self
            .vision
            .borrow()
            .as_ref()
            .map(|v| v.1)
            .ok_or_else(|| anyhow::anyhow!("image rows without a vision table"))?;
        words(
            &self.vision_rows,
            &self.stream,
            [pairs as u32, 1, 1],
            &[x, map.ptr, pairs as u64, rows, lo as u64, hi as u64],
        )
    }
    pub(crate) fn set_vision(&self, table: Option<(Rc<[i32]>, u64)>) {
        *self.vision.borrow_mut() = table;
    }
    fn compute(&self, parity: usize) -> Result<()> {
        self.compute_with(parity, |layer, a, input| {
            a.run_hook(&self.stream, self.p, &self.m, layer, input, || {
                self.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, 1)
            })
        })
    }
    fn compute_with(
        &self,
        parity: usize,
        do_dsa: impl Fn(usize, &Dsa, u64) -> Result<()>,
    ) -> Result<()> {
        let result = self.compute_inner(parity, do_dsa);
        let joined = self.prefetch_join();
        result.and(joined)
    }
    fn compute_inner(
        &self,
        parity: usize,
        do_dsa: impl Fn(usize, &Dsa, u64) -> Result<()>,
    ) -> Result<()> {
        let Self {
            p,
            m,
            k,
            stream,
            chain,
            attention,
            moe,
            id,
            x,
            norm,
            xs,
            post,
            comb,
            hcpart,
            partial,
            ga,
            gg,
            kout,
            out,
            gathered,
            gu,
            activation,
            sums,
            ..
        } = self;
        let pre = |layer: usize, branch: &str, nw: &str| -> Result<()> {
            let ptr = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].{name}"));
            launch_ptrs(
                &k["hc_partial"],
                stream,
                [1, 16, 1],
                &[x.ptr, ptr(&format!("{branch}.fn"))?, hcpart.ptr],
            )?;
            words(
                &k["hc_finish"],
                stream,
                [1, 1, 1],
                &[
                    x.ptr,
                    hcpart.ptr,
                    ptr(&format!("{branch}.base"))?,
                    ptr(&format!("{branch}.scale"))?,
                    ptr(nw)?,
                    norm.ptr,
                    xs.ptr,
                    post.ptr,
                    comb.ptr,
                    1e-5f32.to_bits() as u64,
                    1e-6f32.to_bits() as u64,
                ],
            )
        };
        let finish = |local: &Buffer, layer, site| -> Result<()> {
            self.prefetch_mark(layer, site, 1)?;
            unsafe {
                self.gather(local, gathered)?;
            }
            words(
                &k["hc_post"],
                stream,
                [1, 4, 1],
                &[x.ptr, x.ptr, gathered.ptr, post.ptr, comb.ptr, 4096],
            )
        };
        launch_ptrs(
            &k["embed"],
            stream,
            [1, 64, 1],
            &[id.ptr, m.pointer(p, "weights.embed")?, x.ptr],
        )?;
        if self.vision_token.get() {
            self.vision_inject(x.ptr, &self.vision_map, 1, 0, 1)?;
        }
        for (layer, attn) in attention.iter().enumerate() {
            let ptr = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].{name}"));
            pre(layer, "attn_hc", "in_norm")?;
            match attn {
                Attention::Dsa(a) => {
                    do_dsa(layer, a, norm.ptr)?;
                    finish(
                        &a.output,
                        layer,
                        crate::l2_prefetch_plan::Site::AttentionGather,
                    )?;
                }
                Attention::Kda(a) => {
                    fmm(
                        &k["kda_proj"],
                        stream,
                        norm.ptr,
                        ptr("kda.proj.weight")?,
                        ptr("kda.proj.scale")?,
                        a.proj.ptr,
                        partial.ptr,
                        4096,
                        6416,
                        2,
                    )?;
                    reduce(&k["reduce2_bf16"], stream, partial.ptr, a.proj.ptr, 6416)?;
                    for (offset, name, dst) in [(6144, "fb", ga.ptr), (6272, "gb", gg.ptr)] {
                        fmm(
                            &k["kda_gate"],
                            stream,
                            a.proj.ptr + offset * 2,
                            ptr(&format!("kda.{name}.weight"))?,
                            ptr(&format!("kda.{name}.scale"))?,
                            dst,
                            partial.ptr,
                            6416,
                            2048,
                            1,
                        )?;
                    }
                    self.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, 1)?;
                    let mut args = vec![
                        16u64,
                        a.proj.ptr,
                        6416,
                        6400,
                        ga.ptr,
                        2048,
                        gg.ptr,
                        2048,
                        a.conv.ptr,
                        ptr("kda.conv")?,
                        a.states[parity].ptr,
                        ptr("kda.a_log")?,
                        ptr("kda.dt_bias")?,
                        ptr("kda.norm")?,
                        1e-5f32.to_bits() as u64,
                        (-5f32).to_bits() as u64,
                        1,
                        kout.ptr,
                        a.states[1 - parity].ptr,
                        0,
                        0,
                        0,
                        0,
                    ];
                    let mut params = args.iter_mut().map(crate::cuda::arg).collect::<Vec<_>>();
                    unsafe {
                        chain.launch(stream, [16, 1, 1], &mut params)?;
                    }
                    fmm(
                        &k["kda_out"],
                        stream,
                        kout.ptr,
                        ptr("kda.o.weight")?,
                        ptr("kda.o.scale")?,
                        out.ptr,
                        partial.ptr,
                        2048,
                        4096,
                        4,
                    )?;
                    reduce(&k["reduce4_f32"], stream, partial.ptr, out.ptr, 4096)?;
                    finish(out, layer, crate::l2_prefetch_plan::Site::AttentionGather)?;
                    words(
                        &k["conv_shift"],
                        stream,
                        [1, 6, 1],
                        &[a.conv.ptr, a.proj.ptr, 3 * 6144, 6416, 6416],
                    )?;
                }
            }
            pre(layer, "ffn_hc", "post_norm")?;
            if layer < 3 {
                fmm(
                    &k["mlp_gu"],
                    stream,
                    norm.ptr,
                    ptr("mlp.gu.weight")?,
                    ptr("mlp.gu.scale")?,
                    gu.ptr,
                    partial.ptr,
                    4096,
                    6144,
                    2,
                )?;
                reduce(&k["reduce2_bf16"], stream, partial.ptr, gu.ptr, 6144)?;
                words(
                    &k["swiglu"],
                    stream,
                    [1, 6, 1],
                    &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                )?;
                fmm(
                    &k["mlp_down"],
                    stream,
                    activation.ptr,
                    ptr("mlp.down.weight")?,
                    ptr("mlp.down.scale")?,
                    out.ptr,
                    partial.ptr,
                    3072,
                    4096,
                    4,
                )?;
                reduce(&k["reduce4_f32"], stream, partial.ptr, out.ptr, 4096)?;
                finish(out, layer, crate::l2_prefetch_plan::Site::FeedForwardGather)?;
            } else {
                moe.run_prefetched(stream, p, m, layer, norm.ptr, self.prefetch.as_ref())?;
                finish(
                    &moe.output,
                    layer,
                    crate::l2_prefetch_plan::Site::FeedForwardGather,
                )?;
            }
            if let Some(taps) = &self.taps {
                taps.capture(stream, x, 1, layer)?;
            }
        }
        self.head()
    }
    fn head(&self) -> Result<()> {
        let Self {
            k,
            stream,
            x,
            hidden,
            m,
            p,
            norm,
            xs,
            logits,
            partial,
            ..
        } = self;
        launch_ptrs(&k["stream_mean"], stream, [1, 4, 1], &[x.ptr, hidden.ptr])?;
        words(
            &k["final_norm"],
            stream,
            [1, 1, 1],
            &[
                hidden.ptr,
                4096,
                m.pointer(p, "weights.norm")?,
                norm.ptr,
                4096,
                xs.ptr,
                1e-5f32.to_bits() as u64,
            ],
        )?;
        words(
            &k["head"],
            stream,
            [1, 605, 1],
            &[
                norm.ptr,
                m.pointer(p, "weights.head.weight")?,
                logits.ptr,
                partial.ptr,
                4096,
            ],
        )?;
        self.publish_logits()
    }
    /// Publish the already-computed local logits without repeating normalization
    /// or the vocabulary projection. Also preserves the diagnostic full head.
    fn publish_logits(&self) -> Result<()> {
        let Self {
            stream,
            logits,
            audit,
            all_logits,
            selection,
            rank,
            ..
        } = self;
        // NCCL all-gather copies these BF16 bytes; no floating-point reduction is performed.
        if *audit {
            unsafe {
                self.gather(logits, all_logits)?;
            }
        }
        selection.local(stream, logits, *rank)?;
        unsafe {
            self.gather(&selection.local, &selection.gathered)?;
        }
        selection.merge(stream)?;
        Ok(())
    }
    pub fn enable_prefill(&mut self, root: &Path, native: &Path) -> Result<()> {
        self.batch = Some(batch::Batch::new(&self.device, self, root, native)?);
        Ok(())
    }
    pub fn prefill(&self, position: usize, tokens: &[i32]) -> Result<i32> {
        self.batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batch prefill not initialized"))?
            .run(self, position, tokens)
    }
    unsafe fn gather(&self, send: &Buffer, recv: &Buffer) -> Result<()> {
        unsafe { self.gather_prefix(send, recv, send.bytes) }
    }
    /// RoCE gather between mapped page-locked addresses; false (nothing issued)
    /// when RoCE does not carry this size, since NCCL needs device buffers.
    unsafe fn gather_mapped(&self, send: u64, recv: u64, bytes: usize) -> Result<bool> {
        match &self.roce {
            Some(roce) if bytes <= roce.max_bytes => {
                unsafe { roce.gather_raw(send, recv, bytes, &self.stream)? };
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    unsafe fn gather_prefix(&self, send: &Buffer, recv: &Buffer, bytes: usize) -> Result<()> {
        if let Some(roce) = &self.roce
            && bytes <= roce.max_bytes
        {
            return unsafe { roce.gather_prefix(send, recv, bytes, &self.stream) };
        }
        unsafe { self.comm.all_gather_prefix(send, recv, bytes, &self.stream) }
    }
    fn sync(&self) -> Result<()> {
        self.stream.sync()?;
        if let Some(roce) = &self.roce {
            roce.check()?;
        }
        Ok(())
    }
    pub fn reset(&self) -> Result<()> {
        self.sync()?;
        for a in &self.attention {
            match a {
                Attention::Kda(a) => {
                    unsafe {
                        a.conv.zero_prefix(&self.stream, a.conv.bytes)?;
                    };
                    for st in &a.states {
                        unsafe {
                            st.zero_prefix(&self.stream, st.bytes)?;
                        };
                    }
                }
                Attention::Dsa(a) => {
                    unsafe {
                        a.cache.zero_prefix(&self.stream, a.cache.bytes)?;
                    };
                    a.set_position(&self.stream, 0)?;
                }
            }
        }
        Ok(())
    }
    pub fn compile_graphs(&mut self, token: i32) -> Result<()> {
        ensure!((0..154880).contains(&token), "invalid warmup token");
        self.reset()?;
        unsafe {
            self.id.set_i32(&self.stream, token)?;
        };
        self.compute(0)?;
        self.sync()?;
        let eager_token = self.selection.read()?;
        let eager = if self.audit {
            Some(self.all_logits.read(0, self.all_logits.bytes)?)
        } else {
            None
        };
        self.reset()?;
        let graph0 = unsafe { self.stream.capture(|| self.compute(0))? };
        let graph1 = unsafe { self.stream.capture(|| self.compute(1))? };
        unsafe {
            graph0.launch(&self.stream)?;
        }
        self.sync()?;
        ensure!(
            eager_token == self.selection.read()?,
            "graph/eager selected-token mismatch"
        );
        if let Some(eager) = eager {
            ensure!(
                eager == self.all_logits.read(0, self.all_logits.bytes)?,
                "full-model graph/eager logits mismatch"
            );
        }

        self.graphs = Some([graph0, graph1]);
        self.reset()
    }
    pub fn step(&self, position: usize, token: i32) -> Result<i32> {
        ensure!(
            position < 512 && (0..154880).contains(&token),
            "invalid bounded model step"
        );
        let graphs = self
            .graphs
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("graphs not compiled"))?;
        unsafe {
            self.id.set_i32(&self.stream, token)?;
        };
        for attention in &self.attention {
            if let Attention::Dsa(dsa) = attention {
                dsa.set_position(&self.stream, position)?;
            }
        }
        unsafe {
            graphs[position % 2].launch(&self.stream)?;
        }
        self.sync()?;
        self.selection.read()
    }
}
impl Drop for Engine<'_> {
    fn drop(&mut self) {
        // Pinned prefill IDs and other asynchronous inputs must outlive GPU work,
        // including when an operation returns an error part way through enqueue.
        let _ = self.stream.sync();
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
    audit: bool,
    use_roce: bool,
) -> Result<serde_json::Value> {
    ensure!(
        !token_sets.is_empty() && new_tokens > 0,
        "at least one prompt and one output token are required"
    );
    for tokens in token_sets {
        ensure!(
            !tokens.is_empty() && tokens.len() + new_tokens <= 512,
            "bounded model diagnostic requires 1..512 total tokens"
        );
        ensure!(
            tokens.iter().all(|v| (0..154880).contains(v)),
            "token outside vocabulary"
        );
    }

    let mut engine = Engine::new(p, rank, root, native, nccl, master, audit, use_roce)?;
    engine.compile_graphs(token_sets[0][0])?;
    let Engine {
        id,
        attention,
        all_logits,
        ..
    } = &engine;
    let mut cases = Vec::new();
    for tokens in token_sets {
        engine.reset()?;
        let mut generated = Vec::new();
        let mut next = tokens[0];
        let mut hashes = Vec::new();
        let start = Instant::now();
        let mut prefill_seconds = 0.;
        for pos in 0..tokens.len() + new_tokens.saturating_sub(1) {
            if pos < tokens.len() {
                next = tokens[pos];
            }
            next = engine.step(pos, next)?;
            if audit {
                let bytes = all_logits.read(0, all_logits.bytes)?;
                ensure!(
                    crate::select::reference(&bytes)? as i32 == next,
                    "GPU/CPU greedy mismatch at {pos}"
                );
                hashes.push(format!("{:x}", Sha256::digest(&bytes)));
            }
            if pos + 1 == tokens.len() {
                prefill_seconds = start.elapsed().as_secs_f64();
            }
            if pos + 1 >= tokens.len() {
                generated.push(next);
                if [154820, 154827, 154828, 154829].contains(&next) {
                    break;
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        // Outside measured inference: compare complete final recurrent/cache bytes.
        let mut state_hash = Sha256::new();
        for a in attention {
            match a {
                Attention::Kda(a) => {
                    state_hash.update(a.conv.read(0, a.conv.bytes)?);
                    for state in &a.states {
                        state_hash.update(state.read(0, state.bytes)?);
                    }
                }
                Attention::Dsa(a) => {
                    state_hash.update(a.cache.read(0, a.cache.bytes)?);
                }
            }
        }
        let state_sha256 = format!("{:x}", state_hash.finalize());
        engine.reset()?;
        let mut forced = tokens.to_vec();
        forced.extend(&generated);
        for pos in 0..hashes.len() {
            unsafe {
                id.set_i32(&engine.stream, forced[pos])?;
            };
            for a in attention {
                if let Attention::Dsa(a) = a {
                    a.set_position(&engine.stream, pos)?;
                }
            }
            engine.compute(pos % 2)?;
            engine.sync()?;
            ensure!(
                format!(
                    "{:x}",
                    Sha256::digest(all_logits.read(0, all_logits.bytes)?)
                ) == hashes[pos],
                "graph/eager multi-token drift at position {pos}"
            );
        }
        cases.push(serde_json::json!({"prompt_tokens":tokens,"generated_tokens":generated,"per_step_logits_sha256":hashes,"elapsed_seconds":elapsed,"prefill_seconds":prefill_seconds,"decode_seconds":elapsed-prefill_seconds,"final_state_sha256":state_sha256,"every_step_graph_eager_equal":audit}));
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    ensure!(
        !maps.contains("libpython") && !maps.contains("libtorch"),
        "Python/Torch library mapped"
    );
    Ok(
        serde_json::json!({"milestone":"native_full_model_decode","rank":rank,"layers":45,"cases":cases,"graph_eager_equal":audit,"diagnostic_full_logits":audit,"transport":if use_roce {"roce"}else{"nccl"},"python_torch_mapped":false,"scope":"512-token diagnostic maximum with greedy decode; excludes serving, long-context, native-priority and speculative-decoding checks"}),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_batch(
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
        !token_sets.is_empty() && new_tokens > 0,
        "empty batch qualification"
    );
    for tokens in token_sets {
        ensure!(
            !tokens.is_empty() && tokens.len() + new_tokens <= 512,
            "bounded batch context"
        );
    }
    let mut e = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    e.compile_graphs(token_sets[0][0])?;
    e.enable_prefill(root, native)?;
    let logits_hash = || -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(e.all_logits.read(0, e.all_logits.bytes)?)
        ))
    };
    let state_hash = |position: usize| -> Result<String> {
        let mut h = Sha256::new();
        for a in &e.attention {
            match a {
                Attention::Kda(a) => {
                    h.update(a.conv.read(0, a.conv.bytes)?);
                    h.update(a.states[position % 2].read(0, a.states[position % 2].bytes)?);
                }
                Attention::Dsa(a) => h.update(a.cache.read(0, position * 528)?),
            }
        }
        Ok(format!("{:x}", h.finalize()))
    };
    let mut cases = Vec::new();
    for tokens in token_sets {
        eprintln!("full-model batch parity prompt_rows={}", tokens.len());
        e.reset()?;
        let begin = Instant::now();
        let mut next = e.prefill(0, tokens)?;
        let pf_seconds = begin.elapsed().as_secs_f64();
        let prefill_state = state_hash(tokens.len())?;
        let mut hashes = vec![logits_hash()?];
        let mut generated = vec![next];
        let decode = Instant::now();
        while generated.len() < new_tokens && ![154820, 154827, 154828, 154829].contains(&next) {
            next = e.step(tokens.len() + generated.len() - 1, next)?;
            generated.push(next);
            hashes.push(logits_hash()?);
        }
        let decode_seconds = decode.elapsed().as_secs_f64();
        let final_state = state_hash(tokens.len() + generated.len() - 1)?;
        e.reset()?;
        let mut offset = 0;
        let mut chunk_next = 0;
        while offset < tokens.len() {
            let end = (offset + 64).min(tokens.len());
            chunk_next = e.prefill(offset, &tokens[offset..end])?;
            offset = end;
        }
        ensure!(
            prefill_state == state_hash(tokens.len())?,
            "chunked prefill state mismatch"
        );
        ensure!(
            chunk_next == generated[0] && logits_hash()? == hashes[0],
            "chunked prefill logits mismatch"
        );
        for (step, &token) in generated.iter().enumerate().skip(1) {
            chunk_next = e.step(tokens.len() + step - 1, chunk_next)?;
            ensure!(
                chunk_next == token && logits_hash()? == hashes[step],
                "chunked prefill decode drift at {step}"
            );
        }
        ensure!(
            final_state == state_hash(tokens.len() + generated.len() - 1)?,
            "chunked final state mismatch"
        );
        cases.push(serde_json::json!({"prompt_tokens":tokens,"generated_tokens":generated,"per_step_logits_sha256":hashes,
            "prefill_seconds":pf_seconds,"decode_with_audit_seconds":decode_seconds,"prefill_state_sha256":prefill_state,
            "final_active_state_sha256":final_state,"whole_and_64_row_chunks_exact":true}));
    }
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"cases":cases,
        "scope":"Full-model bounded fast prefill and greedy generation. Includes diagnostic readback; does not measure end-to-end serving performance."}),
    )
}

pub use long::run_restored_requests;

pub use long::serve;
