//! Full-target verifier with batched projections, experts and recurrent updates.
//! Accepted-prefix commits and bounded undo protect each request's state.
use super::{Attention, Engine, Long, ROWS};
use crate::{
    candidates::{Candidates, LOCAL, payload_bytes},
    cuda::{Buffer, Kernel},
    dsa::words,
    experts::ExpertGpu,
    front::load_kernel,
    grouping::Grouping,
    kda_commit_plan::{Policy as CommitPolicy, Source as CommitSource},
    kda_window::{self, MAX_ROWS},
    prepared::Prepared,
    sampling::Sampling,
    verifier_dsa_window::{Policy as DsaWindowPolicy, Stats as DsaWindowStats},
    verifier_graph_cache::{Action as GraphAction, Cache as GraphCache, Policy as GraphPolicy},
    window_gemm::{Projection, SHAPES},
    window_ops::{HcWeights, Ops},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, net::SocketAddr, path::Path};
#[path = "speculative_workloads.rs"]
mod workloads;
pub use workloads::run as run_workloads;
/// Explicit diagnostic-only arm; serving uses its validated configuration flag.
/// Agree malformed/mixed settings before any policy-dependent native work.
pub(super) fn diagnostic_temperature(long: &Long<'_>) -> Result<f64> {
    let parsed = match std::env::var("ALASKA_SPECULATIVE_GREEDY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(false),
        Ok("1") => Ok(true),
        _ => Err(anyhow::anyhow!("ALASKA_SPECULATIVE_GREEDY must be 0/1")),
    };
    let mut record = [0u8; 16];
    record[..8].copy_from_slice(b"ALGRDY01");
    record[8] = u8::from(parsed.is_ok());
    record[9] = u8::from(*parsed.as_ref().unwrap_or(&false));
    let e = &long.engine;
    let all = e.comm.exchange_host(&e.device, &e.stream, &record)?;
    ensure!(
        record[8] == 1 && all.iter().all(|r| r == &record),
        "greedy diagnostic policy differs/invalid on a rank"
    );
    Ok(if parsed? { 0. } else { 1. })
}
#[path = "accepted_path_boundary.rs"]
pub(in crate::model) mod accepted_path;
struct KdaSave {
    scratch: kda_window::Scratch,
    proj: Buffer,
    ga: Buffer,
    gg: Buffer,
}
struct Undo {
    keys: Buffer,
    gates: Buffer,
}
/// Every index layer's ring and undo pointers for one batched copy launch.
fn undo_table(
    index: &[crate::indexer::IndexCache],
    undo: &[Undo],
) -> Result<(crate::batch_copy::IndexUndo, usize, usize)> {
    ensure!(
        index.len() == undo.len() && index.len() <= crate::batch_copy::INDEX_LAYERS,
        "index undo layer count"
    );
    let mut table = crate::batch_copy::IndexUndo::default();
    let mut ring_rows = None;
    for (l, (i, u)) in index.iter().zip(undo).enumerate() {
        ensure!(
            ring_rows.is_none_or(|r| r == i.ring_rows())
                && u.keys.bytes >= MAX_ROWS * crate::indexer::IndexCache::ROW_BYTES
                && u.gates.bytes >= MAX_ROWS * crate::indexer::IndexCache::ROW_BYTES,
            "index undo geometry"
        );
        ring_rows = Some(i.ring_rows());
        table.ring_keys[l] = i.keys.ptr;
        table.ring_gates[l] = i.gates.ptr;
        table.undo_keys[l] = u.keys.ptr;
        table.undo_gates[l] = u.gates.ptr;
    }
    Ok((table, index.len(), ring_rows.unwrap_or(1)))
}
pub(super) struct Outcome {
    graph_built: bool,
    hashes: Vec<String>,
    pub(super) keep: usize,
    pub(super) sampled: Vec<i32>,
}
#[derive(Clone, Copy)]
pub(super) enum Decision {
    Forced(usize),
    Sampled(Sampling),
}
/// The hook sees only serial-equivalent output, never predictions after a
/// rejected proposal or (when enabled) EOS. Errors remain values until the
/// caller has coordinated validity with every rank before publication.
fn clipped_keep(
    tokens: &[i32],
    sampled: &[i32],
    stop_eos: bool,
    clip: impl FnOnce(&[i32]) -> Result<usize>,
) -> Result<usize> {
    ensure!(
        !tokens.is_empty() && tokens.len() <= MAX_ROWS && sampled.len() == tokens.len(),
        "accepted-path proposal/sample bounds"
    );
    let mut accepted = (1..tokens.len())
        .find(|&i| tokens[i] != sampled[i - 1])
        .unwrap_or(tokens.len());
    if stop_eos
        && let Some(i) = sampled[..accepted]
            .iter()
            .position(|id| crate::output_decode::EOS.contains(&(*id as u32)))
    {
        accepted = i + 1;
    }
    let keep = clip(&sampled[..accepted])?;
    ensure!(
        (1..=accepted).contains(&keep),
        "accepted-path clipping bounds"
    );
    Ok(keep)
}
#[path = "multi_decode.rs"]
mod multi_decode;

type GraphKey = (usize, usize, usize, [(usize, usize); MAX_ROWS], bool);
type MultiGraphKey = (usize, Vec<usize>, Vec<usize>, Vec<(usize, usize)>, bool);
pub(super) struct Verifier {
    graphs: GraphCache<GraphKey, crate::cuda::Graph>,
    multi_graphs: BTreeMap<[usize; 4], crate::cuda::Graph>,
    multi_spec_graphs: GraphCache<MultiGraphKey, crate::cuda::Graph>,
    multi_packed: Option<crate::packed_candidates::Packed>,
    multi_save: Option<multi_decode::Saved>,
    multi_native: std::path::PathBuf,
    shared_branch: Option<crate::shared_expert::Branch>,
    trace: Option<crate::phase_profile::Trace>,
    use_graphs: bool,
    greedy_graph: bool,
    stop_eos: bool,
    reuse_head: bool,
    control: super::store::Control,
    positions: Buffer,
    audit: bool,
    batch_dsa: bool,
    dsa_window: DsaWindowPolicy,
    dsa_window_stats: DsaWindowStats,
    dsa_proj: Buffer,
    dsa_qr: Buffer,
    dsa_latent: Buffer,
    dsa_q: Buffer,
    dsa_attended: Buffer,
    taps: Option<crate::target_taps::Taps>,
    samplers: Vec<Candidates>,
    packed: Option<crate::packed_candidates::Packed>,
    greedy_rows: crate::select::GreedyRows,
    k: BTreeMap<&'static str, Kernel>,
    projections: BTreeMap<&'static str, Projection>,
    ops: Ops,
    experts: ExpertGpu,
    group: Grouping,
    kda_batch: Option<crate::kda_commit::Batch>,
    kda_commit_stats: crate::kda_commit::Stats,
    kda: Vec<KdaSave>,
    undo: Vec<Undo>,
    copies: crate::batch_copy::BatchCopy,
    ids: Buffer,
    x: Buffer,
    partial: Buffer,
    out: Buffer,
    gathered: Buffer,
    gu: Buffer,
    activation: Buffer,
    sums: Buffer,
    shared: Buffer,
    all_logits: Buffer,
}
impl Drop for Verifier {
    fn drop(&mut self) {
        if self.dsa_window.enabled()
            && let Ok(stats) = serde_json::to_string(&self.dsa_window_stats)
        {
            eprintln!("verifier_dsa_window={stats}");
        }
        if self.kda_batch.is_some()
            && let Ok(stats) = serde_json::to_string(&self.kda_commit_stats)
        {
            eprintln!("verifier_kda_commit={stats}");
        }
        if self.graphs.bounded() {
            // No tokens, model data or credentials. Qualification collects this
            // final ledger alongside externally measured RSS/device headroom.
            if let Ok(stats) = serde_json::to_string(&self.graphs.stats()) {
                eprintln!("verifier_graph_cache={stats}");
            }
        }
    }
}
impl Verifier {
    pub(super) fn new_retained(
        long: &Long<'_>,
        root: &Path,
        native: &Path,
        policy: GraphPolicy,
        commit_policy: CommitPolicy,
        dsa_window: DsaWindowPolicy,
    ) -> Result<Self> {
        let mut verifier =
            Self::new_with_policies(long, root, native, policy, commit_policy, dsa_window)?;
        verifier.batch_dsa = true;
        verifier.use_graphs = true;
        verifier.stop_eos = true;
        verifier.audit = false;
        verifier.taps = Some(crate::target_taps::Taps::new(
            &long.engine.device,
            native,
            MAX_ROWS,
        )?);
        Ok(verifier)
    }
    pub(super) fn accepted_taps(&self) -> Result<&crate::target_taps::Taps> {
        self.taps
            .as_ref()
            .context("retained verifier capture unavailable")
    }
    fn new(long: &Long<'_>, root: &Path, native: &Path) -> Result<Self> {
        let (graph, commit, window) = Self::diagnostic_policies(long)?;
        Self::new_with_policies(long, root, native, graph, commit, window)
    }
    pub(super) fn diagnostic_policies(
        long: &Long<'_>,
    ) -> Result<(GraphPolicy, CommitPolicy, DsaWindowPolicy)> {
        // Diagnostics also agree optional native-work policies before entering
        // the model's collectives; malformed rank-local input cannot strand peers.
        let graph = GraphPolicy::environment();
        let commit = CommitPolicy::environment();
        let window = DsaWindowPolicy::environment();
        let mut record = [0u8; 64];
        record[..7].copy_from_slice(b"ALVFY02");
        record[7] = u8::from(graph.is_ok() && commit.is_ok() && window.is_ok());
        let mut hash = Sha256::new();
        hash.update(graph.as_ref().map_or([0; 40], |p| p.identity()));
        hash.update(commit.as_ref().map_or([0; 16], |p| p.identity()));
        hash.update(window.as_ref().map_or([0; 16], |p| p.identity()));
        record[8..40].copy_from_slice(&hash.finalize());
        let e = &long.engine;
        let all = e.comm.exchange_host(&e.device, &e.stream, &record)?;
        ensure!(
            record[7] == 1 && all.iter().all(|other| other == &record),
            "verifier native policies differ/invalid on a rank"
        );
        Ok((graph?, commit?, window?))
    }
    fn new_with_policies(
        long: &Long<'_>,
        root: &Path,
        native: &Path,
        policy: GraphPolicy,
        commit_policy: CommitPolicy,
        dsa_window: DsaWindowPolicy,
    ) -> Result<Self> {
        let greedy_graph = match std::env::var("ALASKA_VERIFY_GREEDY_GRAPH").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            _ => Err(anyhow::anyhow!("ALASKA_VERIFY_GREEDY_GRAPH must be 0/1")),
        };
        // Agree even malformed input before any policy-dependent capture or
        // collective. This option changes scheduling, never the greedy kernels.
        let mut record = [0u8; 16];
        record[..8].copy_from_slice(b"ALGTAL01");
        record[8] = u8::from(greedy_graph.is_ok());
        record[9] = u8::from(*greedy_graph.as_ref().unwrap_or(&false));
        let e = &long.engine;
        let all = e.comm.exchange_host(&e.device, &e.stream, &record)?;
        ensure!(
            record[8] == 1 && all.iter().all(|r| r == &record),
            "greedy graph policy differs/invalid on a rank"
        );
        let greedy_graph = greedy_graph?;
        let d = &long.engine.device;
        let packed = crate::packed_candidates::Packed::configured(
            d,
            &long.engine.comm,
            &long.engine.stream,
            native,
        )?;
        let a = |n| d.alloc(MAX_ROWS * n);
        let mut k = BTreeMap::new();
        for name in [
            "embed",
            "swiglu",
            "shared_swiglu",
            "combine",
            "rms1536",
            "rms512",
            "latent_write",
        ] {
            k.insert(name, load_kernel(d, root, name)?);
        }
        let projections = SHAPES
            .into_iter()
            .map(|shape| Ok((shape.alias, Projection::new(d, root, shape)?)))
            .collect::<Result<_>>()?;
        let mut kda = Vec::new();
        for _ in 0..34 {
            kda.push(KdaSave {
                scratch: kda_window::Scratch::new(d, &native.join("kda.cubin"))?,
                proj: a(6416 * 2)?,
                ga: a(2048 * 2)?,
                gg: a(2048 * 2)?,
            });
        }
        let kda_batch = if commit_policy.enabled() {
            let tables = long
                .sessions
                .iter()
                .flat_map(|session| {
                    (0..2).map(|parity| {
                        session
                            .recurrent
                            .iter()
                            .zip(&kda)
                            .map(|(state, save)| {
                                // Built before any handle swap: index = physical buffer.
                                save.scratch.replay_pointers_into(
                                    &state.states[parity],
                                    &state.states[1 - parity],
                                )
                            })
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            Some(crate::kda_commit::Batch::new(d, native, &tables)?)
        } else {
            None
        };
        let undo = (0..11)
            .map(|_| {
                Ok(Undo {
                    keys: a(128 * 2)?,
                    gates: a(128 * 2)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            graphs: GraphCache::new(policy),
            multi_graphs: BTreeMap::new(),
            multi_spec_graphs: GraphCache::new(crate::verifier_graph_cache::Policy::multi_request()),
            multi_packed: None,
            multi_save: None,
            multi_native: native.to_path_buf(),
            shared_branch: crate::shared_expert::Branch::configured(
                d,
                &long.engine.comm,
                &long.engine.stream,
            )?,
            trace: None,
            use_graphs: false,
            greedy_graph,
            stop_eos: false,
            reuse_head: match std::env::var("ALASKA_VERIFY_REUSE_HEAD").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("0") => false,
                Ok("1") => true,
                _ => anyhow::bail!("ALASKA_VERIFY_REUSE_HEAD must be 0/1"),
            },
            control: super::store::Control::new(d)?,
            positions: a(4)?,
            audit: true,
            batch_dsa: false,
            dsa_window,
            dsa_window_stats: Default::default(),
            dsa_proj: a(2048 * 2)?,
            dsa_qr: a(1536 * 2)?,
            dsa_latent: a(512 * 2)?,
            dsa_q: a(4096 * 2)?,
            dsa_attended: a(16 * 512 * 4)?,
            taps: None,
            packed,
            samplers: (0..MAX_ROWS)
                .map(|_| Candidates::new(d, native))
                .collect::<Result<_>>()?,
            greedy_rows: crate::select::GreedyRows::new(d, native)?,
            k,
            projections,
            ops: Ops::new(d, root)?,
            experts: ExpertGpu::new_rows(d, &native.join("exl3.cubin"), MAX_ROWS)?,
            group: Grouping::new(d, native, MAX_ROWS)?,
            kda_batch,
            kda_commit_stats: Default::default(),
            kda,
            undo,
            copies: crate::batch_copy::BatchCopy::new(d, native)?,
            ids: a(4)?,
            x: a(4 * 4096 * 2)?,
            partial: a(8 * 6416 * 4)?,
            out: a(4096 * 4)?,
            gathered: a(4 * 4096 * 4)?,
            gu: a(6144 * 2)?,
            activation: a(3072 * 2)?,
            sums: a(64 * 4)?,
            shared: a(4096 * 4)?,
            all_logits: a(154880 * 2)?,
        })
    }
    /// Keep is supplied by the common diagnostic input, not an acceptance policy.
    /// Diagnostic only: a forced keep bypasses target acceptance checks.
    fn forced(&mut self, long: &mut Long<'_>, tokens: &[i32], keep: usize) -> Result<Vec<String>> {
        Ok(self.execute(long, tokens, Decision::Forced(keep))?.hashes)
    }
    fn execute(
        &mut self,
        long: &mut Long<'_>,
        tokens: &[i32],
        decision: Decision,
    ) -> Result<Outcome> {
        self.execute_with_clip(long, tokens, decision, |accepted| Ok(accepted.len()))
    }
    /// Stage request-local output decisions before GPU state publication. The
    /// hook must return a nonempty prefix and must not publish network output or
    /// perform rank-dependent collectives. Forced diagnostics never call it.
    pub(super) fn execute_with_clip(
        &mut self,
        long: &mut Long<'_>,
        tokens: &[i32],
        decision: Decision,
        clip: impl FnOnce(&[i32]) -> Result<usize>,
    ) -> Result<Outcome> {
        self.execute_with_eos(long, tokens, decision, self.stop_eos, clip)
    }
    /// A retained service owner supplies its immutable request EOS policy.
    /// Diagnostic callers above retain their original explicit stop mode.
    pub(super) fn execute_with_eos(
        &mut self,
        long: &mut Long<'_>,
        tokens: &[i32],
        decision: Decision,
        stop_eos: bool,
        clip: impl FnOnce(&[i32]) -> Result<usize>,
    ) -> Result<Outcome> {
        crate::round_timing::enter(crate::round_timing::Phase::VerifyLaunch);
        long.ready_active()?;
        let n = tokens.len();
        let pos = long.sessions[long.active].tokens.len();
        ensure!(
            (1..=MAX_ROWS).contains(&n)
                && match decision {
                    Decision::Forced(keep) =>
                        keep <= n && pos + n <= long.sessions[long.active].prompt_end,
                    Decision::Sampled(_) =>
                        pos >= long.sessions[long.active].prompt_end
                            && pos + n < long.sessions[long.active].total_end
                            && long.sessions[long.active].pending == Some(tokens[0]),
                }
                && tokens.iter().all(|t| (0..154880).contains(t)),
            "forced verifier bounds"
        );
        long.healthy = false;
        long.memory.map(&long.engine.stream, long.active, pos + n)?;
        let e = &long.engine;
        let s = &e.stream;
        e.sync()?;
        self.ids.write(
            0,
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let (undo, layers, ring_rows) = undo_table(&long.sessions[long.active].index, &self.undo)?;
        unsafe {
            self.copies.index_undo(
                s,
                &undo,
                layers,
                pos,
                0,
                n,
                ring_rows,
                crate::indexer::IndexCache::ROW_BYTES,
                false,
            )?;
        }
        self.positions.write(
            0,
            &(0..n)
                .flat_map(|r| ((pos + r) as i32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut graph_built = false;
        // A greedy tail belongs to the graph identity: a sampled or forced
        // prompt window must never reuse a graph with another selection path.
        let graph_greedy = self.greedy_graph
            && self.use_graphs
            && self.trace.is_none()
            && matches!(decision, Decision::Sampled(policy) if policy.greedy());
        if let Some(mut trace) = self.trace.take() {
            self.forward(long, n, pos, Some(&mut trace))?;
            self.trace = Some(trace);
        } else if self.use_graphs {
            let key = self.graph_key(long, pos, n, graph_greedy)?;
            match self.graphs.action(&key) {
                GraphAction::Replay => unsafe {
                    self.graphs.get(&key).launch(s)?;
                },
                GraphAction::Eager => self.forward_selected(long, n, pos, graph_greedy)?,
                GraphAction::Capture => {
                    graph_built = true;
                    let graph =
                        unsafe { s.capture(|| self.forward_selected(long, n, pos, graph_greedy))? };
                    let nodes = graph.node_count();
                    match self.graphs.insert(key, nodes, graph) {
                        Ok(()) => unsafe {
                            self.graphs.get(&key).launch(s)?;
                        },
                        // Node-count admission can vary by transport/rank. The
                        // same captured computation still runs exactly once;
                        // only retention differs, never the collective order.
                        Err(graph) => unsafe {
                            graph.launch(s)?;
                        },
                    }
                }
            }
        } else {
            self.forward(long, n, pos, None)?;
        }
        crate::round_timing::enter(crate::round_timing::Phase::VerifyWait);
        if self.dsa_window.enabled() {
            // Count once per submitted forward, including graph replays; never
            // count only host-side graph capture as though it were model work.
            if self.batch_dsa && self.dsa_window.window(pos, n)? {
                self.dsa_window_stats.window_rounds += 1;
                self.dsa_window_stats.window_rows += n as u64;
            } else {
                self.dsa_window_stats.fallback_rounds += 1;
                self.dsa_window_stats.fallback_rows += n as u64;
            }
        }
        let Self {
            audit,
            control,
            trace,
            taps,
            samplers,
            packed,
            greedy_rows,
            ops,
            kda_batch,
            kda_commit_stats,
            kda,
            x,
            all_logits,
            ..
        } = self;
        let flip = kda_batch.is_some();
        let (keep, sampled) = match decision {
            Decision::Forced(keep) => (keep, Vec::new()),
            Decision::Sampled(policy) => {
                let mut packed_error = None;
                let sampled = if policy.greedy() {
                    // Candidates' bit-ordered top-1 is not the serial numeric
                    // argmax contract for signed zeros. The row batch runs the
                    // exact serial greedy kernels' per-row code, one row a block.
                    if !graph_greedy {
                        greedy_rows.local(s, &ops.logits, n, e.rank)?;
                        unsafe {
                            e.gather_prefix(&greedy_rows.local, &greedy_rows.gathered, n * 16)?;
                        }
                        greedy_rows.merge(s, n)?;
                    }
                    e.sync()?;
                    greedy_rows.read(n)?
                } else if let Some(packed) = packed {
                    let layout = crate::packed_candidates_layout::Layout::for_policy(
                        n,
                        policy,
                        (pos + 1) as u64,
                    )?;
                    packed.local(s, &ops.logits, e.rank, layout)?;
                    unsafe {
                        e.gather_prefix(&packed.local, &packed.gathered, layout.rank_bytes)?;
                        if policy.gpu_fits(4 * layout.k) {
                            packed.gpu(s, layout, policy, (pos + 1) as u64)?;
                        }
                    }
                    e.sync()?;
                    match packed.read(layout, policy, (pos + 1) as u64) {
                        Ok(rows) => rows[..n].iter().map(|r| r.0 as i32).collect(),
                        Err(error) => {
                            packed_error = Some(error);
                            Vec::new()
                        }
                    }
                } else {
                    let gpu = policy.gpu_fits(4 * policy.local_count());
                    for (row, sampler) in samplers.iter_mut().take(n).enumerate() {
                        sampler.local_offset(
                            s,
                            &ops.logits,
                            row * LOCAL * 2,
                            e.rank,
                            policy.local_count(),
                        )?;
                        unsafe {
                            e.gather_prefix(
                                &sampler.local,
                                &sampler.gathered,
                                payload_bytes(policy.local_count())?,
                            )?;
                            if gpu {
                                sampler.gpu(s, policy, (pos + row + 1) as u64)?;
                            }
                        }
                    }
                    e.sync()?;
                    samplers
                        .iter()
                        .take(n)
                        .enumerate()
                        .map(|(row, sampler)| {
                            Ok(if gpu {
                                sampler.read_gpu(policy)?
                            } else {
                                sampler.read_host(policy, (pos + row + 1) as u64)?
                            }
                            .0 as i32)
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                // A rank-local malformed packed result must still reach the
                // existing common failure fence; it never authorizes discard.
                let clipped = match packed_error {
                    Some(error) => Err(error),
                    None => clipped_keep(tokens, &sampled, stop_eos, clip),
                };
                // This exchange is the same pre-publication collective as
                // before. Its typed result alone can authorize request discard.
                let mut h = Sha256::new();
                h.update(policy.encode());
                h.update([u8::from(stop_eos)]);
                h.update((long.active as u64).to_le_bytes());
                h.update((pos as u64).to_le_bytes());
                h.update((n as u64).to_le_bytes());
                for &t in &sampled {
                    h.update(t.to_le_bytes());
                }
                let record = accepted_path::record(h.finalize().into(), &clipped);
                let all = control.exchange(long, &record)?;
                let keep = match accepted_path::decide(&all, &record, long.active, pos)? {
                    accepted_path::Decision::Keep(keep) => keep,
                    accepted_path::Decision::Discard(proof) => return Err(proof.into()),
                };
                (keep, sampled)
            }
        };
        // Publish the same accepted prefix in every recurrent layer. The common
        // owner must choose keep before this becomes a serving operation.
        if let Some(trace) = trace.as_mut() {
            trace.mark(s, "kda_commit", 45)?;
        }
        let source = CommitSource::new(n, keep)?;
        if let Some(batch) = kda_batch.as_mut()
            && source == CommitSource::Replay
        {
            // The previous invocation drained at the verifier boundary; this
            // pinned upload cannot overwrite counters still read by the GPU.
            unsafe {
                batch.replay(s, long.active, long.kda_phys(long.active, pos), n, keep)?;
            }
        }
        let kda_layers = || {
            e.attention.iter().filter_map(|a| {
                if let Attention::Kda(a) = a {
                    Some(a)
                } else {
                    None
                }
            })
        };
        // Accepting no rows leaves both parities and the convolution tail
        // untouched. Otherwise one launch rebuilds every layer's tail (same
        // ordered row copies through the committed-tail scratch) and copies the
        // chosen state; no in-place restrict-pointer replay is used.
        if kda_batch.is_some() && source != CommitSource::Unchanged {
            let mut table = crate::batch_copy::KdaCommit::default();
            let mut layers = 0;
            for (l, (a, save)) in kda_layers().zip(kda.iter()).enumerate() {
                ensure!(l < crate::batch_copy::KDA_LAYERS, "KDA layer count");
                table.conv[l] = a.conv.ptr;
                table.proj[l] = save.proj.ptr;
                table.tail[l] = save.scratch.committed_conv.ptr;
                layers += 1;
            }
            ensure!(layers == crate::batch_copy::KDA_LAYERS, "KDA layer count");
            // The window (full accept) or the replay (partial) already wrote the
            // committed state into the other physical buffer; only the tail moves.
            unsafe {
                self.copies.kda_commit(
                    s,
                    &table,
                    keep,
                    kda_window::CONV_CHANNELS * 2,
                    kda_window::PROJ_STRIDE * 2,
                    None,
                )?;
            }
        }
        for (a, save) in kda_layers().zip(kda.iter_mut()) {
            let (state, conv) = if kda_batch.is_some() {
                continue;
            } else {
                let committed = unsafe {
                    save.scratch.commit_captured(
                        s,
                        kda_window::Inputs {
                            state: &a.states[pos % 2],
                            conv: &a.conv,
                            projection: &save.proj,
                            gate_a: &save.ga,
                            gate_g: &save.gg,
                        },
                        n,
                        keep,
                    )?
                };
                (committed.state, committed.conv)
            };
            unsafe {
                a.states[(pos + keep) % 2].copy_from(s, 0, state, 0, kda_window::STATE_BYTES)?;
                a.conv.copy_from(s, 0, conv, 0, kda_window::CONV_BYTES)?;
            }
        }
        if kda_batch.is_some() {
            let layer_bytes = (kda.len() * kda_window::STATE_BYTES) as u64;
            kda_commit_stats.rounds += 1;
            match source {
                CommitSource::Unchanged => {
                    kda_commit_stats.unchanged += 1;
                    kda_commit_stats.avoided_state_copy_bytes += 2 * layer_bytes;
                }
                CommitSource::Proposal => {
                    kda_commit_stats.full += 1;
                    kda_commit_stats.state_copy_bytes += layer_bytes;
                    kda_commit_stats.avoided_state_copy_bytes += layer_bytes;
                }
                CommitSource::Replay => {
                    kda_commit_stats.partial += 1;
                    kda_commit_stats.replay_launches += 1;
                    kda_commit_stats.state_copy_bytes += layer_bytes;
                }
            }
        }
        if let Some(trace) = trace.as_mut() {
            trace.mark(s, "commit_suffix_head", 45)?;
        }
        let (undo, layers, ring_rows) = undo_table(&long.sessions[long.active].index, &self.undo)?;
        unsafe {
            self.copies.index_undo(
                s,
                &undo,
                layers,
                pos,
                keep,
                n,
                ring_rows,
                crate::indexer::IndexCache::ROW_BYTES,
                true,
            )?;
        }
        unsafe {
            long.sessions[long.active]
                .position
                .set_i32(s, (pos + keep) as i32)?;
        }
        if keep > 0 {
            unsafe {
                e.x.copy_from(s, 0, x, (keep - 1) * 4 * 4096 * 2, 4 * 4096 * 2)?;
            }
            if self.reuse_head {
                unsafe {
                    e.logits
                        .copy_from(s, 0, &ops.logits, (keep - 1) * LOCAL * 2, LOCAL * 2)?;
                }
                // A sampled decision publishes its own pending token below; only
                // forced commits and audits read the greedy selection.
                if sampled.is_empty() || *audit {
                    e.publish_logits()?;
                }
            } else {
                e.head()?;
            }
        }
        if let (Some(committed), Some(proposed)) = (&e.taps, taps.as_ref()) {
            committed.append(s, pos, proposed, keep)?;
        }
        if let Some(trace) = trace.as_mut() {
            trace.mark(s, "commit_drain", 45)?;
        }
        e.sync()?;
        let hashes = if *audit {
            let bytes = all_logits.read(0, n * 154880 * 2)?;
            (0..n)
                .map(|row| {
                    let mut h = Sha256::new();
                    for rank in 0..4 {
                        let off = (rank * n + row) * 38720 * 2;
                        h.update(&bytes[off..off + 38720 * 2]);
                    }
                    format!("{:x}", h.finalize())
                })
                .collect()
        } else {
            Vec::new()
        };
        long.sessions[long.active]
            .tokens
            .extend_from_slice(&tokens[..keep]);
        if keep > 0 {
            long.sessions[long.active].mixed = true;
            if sampled.is_empty() {
                long.sessions[long.active].pending = (pos + keep
                    >= long.sessions[long.active].prompt_end)
                    .then_some(e.selection.read()?);
            }
            long.latest_logits = Some(long.active);
        }
        long.healthy = true;
        if !sampled.is_empty() {
            long.sessions[long.active].pending = Some(sampled[keep - 1]);
        }
        // The committed state is in the other physical buffer; an even keep
        // leaves it under the old parity handle, so exchange the handles.
        if flip && keep > 0 && keep.is_multiple_of(2) {
            long.swap_kda_states(long.active)?;
        }
        Ok(Outcome {
            graph_built,
            hashes,
            keep,
            sampled,
        })
    }
    fn graph_key(&self, long: &Long<'_>, pos: usize, n: usize, greedy: bool) -> Result<GraphKey> {
        let row_geometry = |at: usize| {
            (
                if at < 2051 { (at + 1).div_ceil(512) } else { 0 },
                if at + 1 > 2051 {
                    ((at + 1) / 4).next_power_of_two().clamp(1024, 1048576 / 4)
                } else {
                    0
                },
            )
        };
        // One-call windows depend only on their last row's chunk count or
        // selection bucket; per-row windows on every row's geometry.
        let whole = self.batch_dsa && (pos + n <= 2051 || self.dsa_window.window(pos, n)?);
        let geometry = std::array::from_fn(|row| match row {
            _ if row >= n => (0, 0),
            0 if whole => row_geometry(pos + n - 1),
            _ if whole => (0, 0),
            _ => row_geometry(pos + row),
        });
        Ok((
            long.active,
            long.kda_phys(long.active, pos),
            n,
            geometry,
            greedy,
        ))
    }
    /// Graphs bake in their slot's buffers, so without this every slot's first
    /// requests recapture keys the other slots already hold. Captures each
    /// slot's dense windows and first two selection buckets, both parities.
    pub(super) fn prewarm(&mut self, long: &mut Long<'_>) -> Result<usize> {
        if !self.use_graphs || !self.graphs.bounded() || self.trace.is_some() {
            return Ok(0);
        }
        let mut windows = Vec::new();
        for n in 1..=MAX_ROWS {
            for chunk in 0..4 {
                windows.extend([(chunk * 512 + 2, n), (chunk * 512 + 3, n)]);
            }
            windows.extend([(2050 - n, n), (2051 - n, n)]);
            windows.extend([(2051, n), (2052, n), (4100, n), (4101, n)]);
        }
        let mut captured = 0;
        for slot in 0..long.sessions.len() {
            long.bind(slot)?;
            for &(pos, n) in &windows {
                // Keep the existing four-slot sampled prewarm. Only the main
                // slot additionally prewarms greedy tails, within the unchanged
                // graph/node budgets; other slots use ordinary lazy capture.
                for greedy in [false, true] {
                    if greedy && (!self.greedy_graph || slot != 0) {
                        continue;
                    }
                    let key = self.graph_key(long, pos, n, greedy)?;
                    if self.graphs.contains(&key) {
                        continue;
                    }
                    let s = &long.engine.stream;
                    // Capture only: on GB10 a graph explicitly uploaded with
                    // cuGraphUpload replays more slowly than one whose first
                    // launch performs the upload (measured on GB10).
                    let graph =
                        unsafe { s.capture(|| self.forward_selected(long, n, pos, greedy))? };
                    let nodes = graph.node_count();
                    captured += usize::from(self.graphs.insert(key, nodes, graph).is_ok());
                }
            }
        }
        long.bind(0)?;
        Ok(captured)
    }
    fn forward_selected(
        &mut self,
        long: &Long<'_>,
        n: usize,
        pos: usize,
        greedy: bool,
    ) -> Result<()> {
        self.forward(long, n, pos, None)?;
        if greedy {
            let e = &long.engine;
            self.greedy_rows
                .local(&e.stream, &self.ops.logits, n, e.rank)?;
            unsafe {
                e.gather_prefix(&self.greedy_rows.local, &self.greedy_rows.gathered, n * 16)?;
            }
            self.greedy_rows.merge(&e.stream, n)?;
        }
        Ok(())
    }
    fn forward(
        &mut self,
        long: &Long<'_>,
        n: usize,
        pos: usize,
        trace: Option<&mut crate::phase_profile::Trace>,
    ) -> Result<()> {
        let result = self.forward_inner(long, n, pos, trace);
        let joined = long.engine.prefetch_join();
        result.and(joined)
    }
    fn forward_inner(
        &mut self,
        long: &Long<'_>,
        n: usize,
        pos: usize,
        mut trace: Option<&mut crate::phase_profile::Trace>,
    ) -> Result<()> {
        let e = &long.engine;
        let s = &e.stream;
        let mut mark = |label, layer| -> Result<()> {
            if let Some(t) = trace.as_deref_mut() {
                t.mark(s, label, layer)?;
            }
            Ok(())
        };
        mark("embed", 0)?;
        // The batched commit takes the window's final KDA state in the slot's
        // other physical buffer (rec[1 - cur]); no state copy later.
        let flip = self.kda_batch.is_some();
        let Self {
            shared_branch,
            audit,
            batch_dsa,
            dsa_window,
            dsa_proj,
            dsa_qr,
            dsa_latent,
            dsa_q,
            dsa_attended,
            taps,
            k,
            projections,
            ops,
            experts,
            group,
            kda,
            positions,
            ids,
            x,
            partial,
            out,
            gathered,
            gu,
            activation,
            sums,
            shared,
            all_logits,
            ..
        } = self;
        let mat_on = |stream: &crate::cuda::Stream,
                      alias: &str,
                      stem: &str,
                      input: &Buffer,
                      offset,
                      stride,
                      output: &Buffer|
         -> Result<()> {
            unsafe {
                projections[alias].run_offset(
                    stream,
                    input,
                    offset,
                    stride,
                    e.m.pointer(e.p, &format!("{stem}.weight"))?,
                    e.m.pointer(e.p, &format!("{stem}.scale"))?,
                    output,
                    partial,
                    n,
                )
            }
        };
        let mat = |alias: &str, stem: &str, input: &Buffer, offset, stride, output: &Buffer| {
            mat_on(s, alias, stem, input, offset, stride, output)
        };
        let finish = || -> Result<()> {
            unsafe {
                e.gather_prefix(out, gathered, n * 4096 * 4)?;
                ops.hc_post(s, x, gathered, x, n)
            }
        };
        words(
            &k["embed"],
            s,
            [n as u32, 64, 1],
            &[ids.ptr, e.m.pointer(e.p, "weights.embed")?, x.ptr],
        )?;
        let mut saves = kda.iter_mut();
        for (layer, attn) in e.attention.iter().enumerate() {
            let stem = |name: &str| format!("weights.layers[{layer}].{name}");
            let w = |name: &str| e.m.pointer(e.p, &stem(name));
            let pre = |branch: &str, norm: &str| -> Result<()> {
                unsafe {
                    ops.hc_pre(
                        s,
                        x,
                        HcWeights {
                            function: w(&format!("{branch}.fn"))?,
                            base: w(&format!("{branch}.base"))?,
                            scale: w(&format!("{branch}.scale"))?,
                            norm: w(norm)?,
                        },
                        n,
                    )
                }
            };
            mark("hc_attention", layer)?;
            pre("attn_hc", "in_norm")?;
            mark(
                if matches!(attn, Attention::Kda(_)) {
                    "kda"
                } else {
                    "dsa_projection"
                },
                layer,
            )?;
            match attn {
                Attention::Kda(a) => {
                    let save = saves.next().context("KDA verifier geometry")?;
                    mat(
                        "kda_proj",
                        &stem("kda.proj"),
                        &ops.norm,
                        0,
                        4096,
                        &save.proj,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.fb"),
                        &save.proj,
                        6144 * 2,
                        6416,
                        &save.ga,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.gb"),
                        &save.proj,
                        6272 * 2,
                        6416,
                        &save.gg,
                    )?;
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    let inputs = kda_window::Inputs {
                        state: &a.states[pos % 2],
                        conv: &a.conv,
                        projection: &save.proj,
                        gate_a: &save.ga,
                        gate_g: &save.gg,
                    };
                    let weights = kda_window::Weights {
                        conv: w("kda.conv")?,
                        a_log: w("kda.a_log")?,
                        dt_bias: w("kda.dt_bias")?,
                        norm: w("kda.norm")?,
                    };
                    let window = unsafe {
                        if flip {
                            save.scratch.evaluate_into(
                                s,
                                inputs,
                                weights,
                                n,
                                &a.states[(pos + 1) % 2],
                            )?
                        } else {
                            save.scratch.evaluate(s, inputs, weights, n)?
                        }
                    };
                    mat("kda_out", &stem("kda.o"), window.output(), 0, 2048, out)?;
                }
                Attention::Dsa(a) if *batch_dsa => {
                    mat("dsa_proj", &stem("dsa.proj"), &ops.norm, 0, 4096, dsa_proj)?;
                    for (alias, off, name, dst, width) in [
                        ("rms1536", 0, "dsa.q_norm", dsa_qr.ptr, 1536),
                        ("rms512", 3072, "dsa.kv_norm", dsa_latent.ptr, 512),
                    ] {
                        words(
                            &k[alias],
                            s,
                            [n as u32, 1, 1],
                            &[
                                dsa_proj.ptr + off,
                                2048,
                                w(name)?,
                                dst,
                                width,
                                sums.ptr,
                                1e-5f32.to_bits() as u64,
                            ],
                        )?;
                    }
                    mat("dsa_qb", &stem("dsa.q_b"), dsa_qr, 0, 1536, dsa_q)?;
                    unsafe {
                        ops.absorb(s, dsa_q, w("dsa.kv_k.weight")?, n)?;
                    }
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    unsafe {
                        a.position.copy_from(s, 0, positions, 0, 4)?;
                    }
                    words(
                        &k["latent_write"],
                        s,
                        [n as u32, 1, 1],
                        &[
                            dsa_latent.ptr,
                            a.cache.ptr,
                            a.cache.ptr,
                            a.position.ptr,
                            a.page_table.ptr,
                        ],
                    )?;
                    long.projection.run(
                        s,
                        e.p,
                        &e.m,
                        layer,
                        ops.norm.ptr,
                        dsa_qr.ptr,
                        n,
                        pos,
                        false,
                    )?;
                    mark("dsa_index_attention", layer)?;
                    // Fully sparse windows, and fully dense ones (exact R-row BM16
                    // chunks), attend in one call; mixed windows stay per row.
                    if dsa_window.window(pos, n)? || pos + n <= 2051 {
                        // All new ring rows/pools are staged together. The
                        // existing score/selection kernels retain each row's
                        // own visible-pool bound; future pools stay invisible.
                        // fast=false keeps the qualified sparse chunk/merge
                        // arithmetic, and this path never enters dense R=1 ABI.
                        long.attend_projected(
                            super::super::batch::Inputs {
                                layer,
                                position: pos,
                                norm: ops.norm.ptr,
                                qr: dsa_qr.ptr,
                                qa: ops.qa.ptr,
                                attended: dsa_attended.ptr,
                                rows: n,
                                dsa: a,
                            },
                            pos,
                            false,
                            long.projection.raw.ptr,
                            long.projection.gate.ptr,
                            long.projection.query.ptr,
                        )?;
                    } else {
                        // Dense and mixed windows keep the exact one-row ABI.
                        for row in 0..n {
                            unsafe {
                                a.position.copy_from(s, 0, positions, row * 4, 4)?;
                            }
                            long.attend_projected(
                                super::super::batch::Inputs {
                                    layer,
                                    position: pos + row,
                                    norm: ops.norm.ptr + (row * 4096 * 2) as u64,
                                    qr: dsa_qr.ptr + (row * 1536 * 2) as u64,
                                    qa: ops.qa.ptr + (row * 16 * 512 * 2) as u64,
                                    attended: dsa_attended.ptr + (row * 16 * 512 * 4) as u64,
                                    rows: 1,
                                    dsa: a,
                                },
                                pos + row,
                                false,
                                long.projection.raw.ptr + (row * 160 * 2) as u64,
                                long.projection.gate.ptr + (row * 128 * 4) as u64,
                                long.projection.query.ptr + (row * 4096 * 2) as u64,
                            )?;
                        }
                    }
                    unsafe {
                        mark("dsa_expand_output", layer)?;
                        ops.expand(s, dsa_attended, w("dsa.kv_v.weight")?, n)?;
                    }
                    mat("dsa_out", &stem("dsa.o"), &ops.expanded, 0, 4096, out)?;
                }
                Attention::Dsa(a) => {
                    for row in 0..n {
                        unsafe {
                            a.position.copy_from(s, 0, positions, row * 4, 4)?;
                        }
                        let norm = ops.norm.ptr + (row * 4096 * 2) as u64;
                        a.run_with(
                            s,
                            e.p,
                            &e.m,
                            layer,
                            norm,
                            || {
                                if row == 0 {
                                    e.prefetch_mark(
                                        layer,
                                        crate::l2_prefetch_plan::Site::AttentionCore,
                                        n,
                                    )?;
                                }
                                Ok(())
                            },
                            |a| {
                                long.attend(
                                    super::super::batch::Inputs {
                                        layer,
                                        position: pos + row,
                                        norm,
                                        qr: a.qr.ptr,
                                        qa: a.qa.ptr,
                                        attended: a.attended.ptr,
                                        rows: 1,
                                        dsa: a,
                                    },
                                    pos + row,
                                    false,
                                )
                            },
                        )?;
                        unsafe {
                            out.copy_from(s, row * 4096 * 4, &a.output, 0, 4096 * 4)?;
                        }
                    }
                }
            }
            mark("attention_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionGather, n)?;
            finish()?;
            mark("hc_ffn", layer)?;
            pre("ffn_hc", "post_norm")?;
            mark(if layer < 3 { "dense_ffn" } else { "experts" }, layer)?;
            if layer < 3 {
                mat("mlp_gu", &stem("mlp.gu"), &ops.norm, 0, 4096, gu)?;
                words(
                    &k["swiglu"],
                    s,
                    [n as u32, 6, 1],
                    &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                )?;
                mat("mlp_down", &stem("mlp.down"), activation, 0, 3072, out)?;
            } else {
                unsafe {
                    ops.route(s, &ops.norm, w("moe.router")?, w("moe.bias")?, n)?;
                }
                group.run(s, ops.picks.ptr, n)?;
                let routed = || {
                    experts.run_rows(
                        s,
                        e.p,
                        &e.m,
                        layer,
                        ops.norm.ptr,
                        ops.picks.ptr,
                        group.ids.ptr,
                        group.count.ptr,
                        group.members.ptr,
                        n,
                        MAX_ROWS,
                    )
                };
                e.prefetch_experts(layer, n, &group.ids, &group.count)?;
                let shared_work = |stream: &crate::cuda::Stream| -> Result<()> {
                    mat_on(
                        stream,
                        "shared_gu",
                        &stem("moe.shared.gu"),
                        &ops.norm,
                        0,
                        4096,
                        gu,
                    )?;
                    words(
                        &k["shared_swiglu"],
                        stream,
                        [n as u32, 1, 1],
                        &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                    )?;
                    mat_on(
                        stream,
                        "shared_down",
                        &stem("moe.shared.down"),
                        activation,
                        0,
                        512,
                        shared,
                    )
                };
                if let Some(branch) = shared_branch.as_mut() {
                    // Routed ExpertGpu owns separate xg/xu/xd/z/y scratch.
                    // Both branches only read ops.norm; combine stays on the
                    // original stream after their join, with unchanged sums.
                    branch.run(s, shared_work, routed)?;
                } else {
                    if e.prefetch.is_none() {
                        routed()?;
                    }
                    shared_work(s)?;
                    if e.prefetch.is_some() {
                        routed()?;
                    }
                }
                words(
                    &k["combine"],
                    s,
                    [n as u32, 4, 1],
                    &[experts.y.ptr, shared.ptr, ops.weights.ptr, out.ptr],
                )?;
            }
            mark("ffn_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::FeedForwardGather, n)?;
            finish()?;
            if let Some(taps) = taps.as_ref() {
                taps.capture(s, x, n, layer)?;
            }
        }
        mark("head", 45)?;
        unsafe {
            ops.head(
                s,
                x,
                e.m.pointer(e.p, "weights.norm")?,
                e.m.pointer(e.p, "weights.head.weight")?,
                n,
            )?;
            if *audit {
                e.gather_prefix(&ops.logits, all_logits, n * 38720 * 2)?;
            }
        }
        mark("sampling_commit", 45)?;
        Ok(())
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
    _output: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let mut verifier = Verifier::new(&long, root, native)?;
    verifier.batch_dsa = true;
    verifier.use_graphs = true;
    let source = token_sets.last().context("verifier prompt fixture")?;
    ensure!(source.len() >= 8192, "verifier fixture too short");
    let proposal: [i32; MAX_ROWS] =
        std::array::from_fn(|i| [19, 11, 23, 25, 27, 31, 42, 99][i % 8]);
    let mut cases = Vec::new();
    for pos in [128usize, 2047, 2048, 2050, 8190] {
        let prefix = &source[..pos];
        // References include every accepted length, with serial decode arithmetic.
        long.start(pos + MAX_ROWS, 1)?;
        for (i, chunk) in prefix.chunks(ROWS).enumerate() {
            long.prefill(i * ROWS, chunk)?;
        }
        let initial = long.state(pos)?;
        let mut states = vec![initial];
        let mut logits = Vec::new();
        for token in proposal {
            long.exact_prompt_token(token)?;
            states.push(long.state(long.sessions[0].tokens.len())?);
            logits.push(long.logits()?);
        }
        for keep in 0..=MAX_ROWS {
            long.start(pos + MAX_ROWS, 1)?;
            for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                long.prefill(i * ROWS, chunk)?;
            }
            let actual = verifier.forced(&mut long, &proposal, keep)?;
            ensure!(
                actual == logits,
                "verifier logits differ at position={pos} keep={keep}"
            );
            ensure!(
                long.state(pos + keep)? == states[keep],
                "verifier committed state differs at position={pos} keep={keep}"
            );
            if keep > 0 {
                ensure!(
                    long.logits()? == logits[keep - 1],
                    "verifier published head differs"
                );
            }
            eprintln!("verifier exact position={pos} rows={MAX_ROWS} keep={keep}");
            cases.push(serde_json::json!({"position":pos,"rows":MAX_ROWS,"keep":keep,"exact_logits":true,"exact_state":true}));
        }
    }
    let sparse_window_cases = if verifier.dsa_window.enabled() {
        check_sparse_windows(&mut long, &mut verifier, source, &proposal)?
    } else {
        Vec::new()
    };
    Ok(
        serde_json::json!({"rank":rank,"l2_prefetch":long.engine.prefetch_receipt(),"cases":cases,
        "sparse_window_cases":sparse_window_cases,
        "scope":"forced full-target verifier, graph execution and rejected-ring rollback; optional wholly sparse DSA window cases use independent serial state/logit references; no drafter, sampling acceptance or serving"}),
    )
}

fn check_sparse_windows(
    long: &mut Long<'_>,
    verifier: &mut Verifier,
    source: &[i32],
    proposal: &[i32; MAX_ROWS],
) -> Result<Vec<serde_json::Value>> {
    let mut cases = Vec::new();
    for pos in [2051usize, 4093, 8191] {
        let prefix = &source[..pos];
        long.start(pos + MAX_ROWS, 1)?;
        for (i, chunk) in prefix.chunks(ROWS).enumerate() {
            long.prefill(i * ROWS, chunk)?;
        }
        let mut states = vec![long.state(pos)?];
        let mut logits = Vec::new();
        for &token in proposal {
            long.exact_prompt_token(token)?;
            states.push(long.state(long.sessions[long.active].tokens.len())?);
            logits.push(long.logits()?);
        }
        for rows in [1, 2, 7, MAX_ROWS] {
            let mut keeps = vec![0, 1, rows / 2, rows];
            keeps.sort_unstable();
            keeps.dedup();
            for keep in keeps {
                long.start(pos + MAX_ROWS, 1)?;
                for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                    long.prefill(i * ROWS, chunk)?;
                }
                let actual = verifier.forced(long, &proposal[..rows], keep)?;
                ensure!(
                    actual == logits[..rows] && long.state(pos + keep)? == states[keep],
                    "sparse window state/logits position={pos} rows={rows} keep={keep}"
                );
                if keep > 0 {
                    ensure!(
                        long.logits()? == logits[keep - 1],
                        "sparse window accepted head"
                    );
                }
                // Rejected complete pools remain unpublished suffix bytes.
                // A real serial continuation must recompute them correctly from
                // accepted ring rows, rather than consuming stale future pools.
                for &token in &proposal[keep..] {
                    long.exact_prompt_token(token)?;
                }
                ensure!(
                    long.state(pos + MAX_ROWS)? == states[MAX_ROWS]
                        && long.logits()? == logits[MAX_ROWS - 1],
                    "sparse window continuation after rollback"
                );
                cases.push(serde_json::json!({"position":pos,"rows":rows,"keep":keep,
                    "exact_logits":true,"exact_state":true,"exact_followup_state":true,
                    "exact_followup_logits":true}));
            }
        }
    }
    Ok(cases)
}

#[allow(clippy::too_many_arguments)]
pub fn run_sampled(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    _output: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let mut verifier = Verifier::new(&long, root, native)?;
    verifier.batch_dsa = true;
    verifier.use_graphs = true;
    let mut serial = Candidates::new(&long.engine.device, native)?;
    let temperature = diagnostic_temperature(&long)?;
    let source = token_sets.last().context("sampled verifier fixture")?;
    ensure!(source.len() >= 8192, "sampled verifier fixture too short");
    if let Ok(fault) = std::env::var("ALASKA_CLIP_FAULT") {
        ensure!(
            matches!(fault.as_str(), "invalid" | "error" | "different"),
            "unknown clipping fault probe"
        );
        let policy = Sampling::new(731, temperature, 20, 0.95)?;
        let pos = 128;
        long.start(pos, 32)?;
        long.prefill(0, &source[..pos])?;
        let mut next = long.sample(&mut serial, policy)?.0 as i32;
        let mut tokens = vec![next];
        for row in 0..MAX_ROWS {
            long.step(pos + row, next)?;
            next = long.sample(&mut serial, policy)?.0 as i32;
            tokens.push(next);
        }
        long.start(pos, 32)?;
        long.prefill(0, &source[..pos])?;
        ensure!(
            long.sample(&mut serial, policy)?.0 as i32 == tokens[0],
            "fault-probe serial prefix changed"
        );
        let outcome = verifier.execute_with_clip(
            &mut long,
            &tokens[..MAX_ROWS],
            Decision::Sampled(policy),
            |accepted| {
                ensure!(
                    accepted == &tokens[1..=MAX_ROWS],
                    "fault probe acceptance changed"
                );
                if rank == 2 {
                    match fault.as_str() {
                        "invalid" => Ok(0),
                        "error" => anyhow::bail!("injected rank-local output parser error"),
                        "different" => Ok(1),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(2)
                }
            },
        );
        ensure!(
            outcome.is_err()
                && !long.healthy
                && long.sessions[long.active].tokens.len() == pos
                && long.sessions[long.active].pending == Some(tokens[0]),
            "clipping fault published state or left owner healthy"
        );
        // No further work may use this poisoned model. Returning drops the
        // entire isolated group after every rank has reached the failed fence.
        return Ok(
            serde_json::json!({"rank":rank,"temperature":temperature,"fault":fault,"fault_rank":2,
            "collective_rejection":true,"owner_poisoned":true,"host_cursor_and_pending_unpublished":true,
            "scope":"Injected accepted-path error before commit; evaluated GPU scratch is intentionally discarded with the failed owner."}),
        );
    }
    let mut cases = Vec::new();
    let mut clipped_cases = Vec::new();
    let clip_only = matches!(std::env::var("ALASKA_CLIP_ONLY").as_deref(), Ok("1"));
    for pos in [128usize, 2050, 8190] {
        for (seed, top_k) in [(731u64, 20usize), (917, 128)] {
            if clip_only && top_k != 20 {
                continue;
            }
            let policy = Sampling::new(seed, temperature, top_k, 0.95)?;
            let prefix = &source[..pos];
            long.start(pos, 32)?;
            for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                long.prefill(i * ROWS, chunk)?;
            }
            let mut next = long.sample(&mut serial, policy)?.0 as i32;
            let mut tokens = vec![next];
            let mut states = Vec::new();
            for row in 0..MAX_ROWS {
                long.step(pos + row, next)?;
                next = long.sample(&mut serial, policy)?.0 as i32;
                tokens.push(next);
                states.push(long.state(pos + row + 1)?);
            }
            for reject_at in (1..=MAX_ROWS).filter(|_| !clip_only) {
                long.start(pos, 32)?;
                for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                    long.prefill(i * ROWS, chunk)?;
                }
                ensure!(
                    long.sample(&mut serial, policy)?.0 as i32 == tokens[0],
                    "sampled cold prefix drift"
                );
                let mut proposed = tokens[..MAX_ROWS].to_vec();
                if reject_at < MAX_ROWS {
                    proposed[reject_at] = (proposed[reject_at] + 1) % 154880;
                }
                let result = verifier.execute(&mut long, &proposed, Decision::Sampled(policy))?;
                ensure!(
                    result.keep == reject_at
                        && result.sampled[..result.keep] == tokens[1..=result.keep],
                    "sampled acceptance drift pos={pos} reject={reject_at}"
                );
                ensure!(
                    long.state(pos + result.keep)? == states[result.keep - 1]
                        && long.sessions[long.active].pending == Some(tokens[result.keep]),
                    "sampled committed cursor/state drift"
                );
                eprintln!(
                    "sampled verifier exact position={pos} seed={seed} top_k={top_k} keep={reject_at}"
                );
                cases.push(serde_json::json!({"position":pos,"seed":seed,"temperature":temperature,"top_k":top_k,"keep":reject_at,"exact_sampled_prefix":true,"exact_state":true}));
            }
            if top_k == 20 {
                for clip_at in [1, 4, MAX_ROWS] {
                    long.start(pos, 32)?;
                    for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                        long.prefill(i * ROWS, chunk)?;
                    }
                    ensure!(
                        long.sample(&mut serial, policy)?.0 as i32 == tokens[0],
                        "clipped sampled cold prefix drift"
                    );
                    let mut called = false;
                    let result = verifier.execute_with_clip(
                        &mut long,
                        &tokens[..MAX_ROWS],
                        Decision::Sampled(policy),
                        |accepted| {
                            ensure!(
                                accepted == &tokens[1..=MAX_ROWS],
                                "clipping hook did not receive the fully accepted path"
                            );
                            called = true;
                            Ok(clip_at)
                        },
                    )?;
                    ensure!(
                        called
                            && result.keep == clip_at
                            && result.sampled[..clip_at] == tokens[1..=clip_at]
                            && long.state(pos + clip_at)? == states[clip_at - 1]
                            && long.sessions[long.active].pending == Some(tokens[clip_at]),
                        "clipped sampled state/pending drift pos={pos} clip={clip_at}"
                    );
                    eprintln!("clipped sampled verifier exact position={pos} keep={clip_at}");
                    clipped_cases.push(serde_json::json!({
                        "position":pos,"seed":seed,"temperature":temperature,"top_k":top_k,
                        "target_accepted":MAX_ROWS,"keep":clip_at,
                        "exact_sampled_prefix":true,"exact_state":true,"exact_pending":true,
                    }));
                }
            }
        }
    }
    Ok(
        serde_json::json!({"rank":rank,"temperature":temperature,"l2_prefetch":long.engine.prefetch_receipt(),"cases":cases,"clipped_cases":clipped_cases,"scope":"target acceptance with deliberately controlled proposals; exact serial greedy or GPU/host sampled policies and accepted-path clipping; no drafter or serving"}),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_draft(
    p: &Prepared,
    draft_manifest: &Path,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    token_sets: &[Vec<i32>],
    use_graphs: bool,
) -> Result<serde_json::Value> {
    use crate::target_taps::{Taps, WIDTH};
    use std::time::Instant;
    let mut engine = Engine::new(p, rank, root, native, nccl, master, false, true)?;
    let tap = Taps::new(&engine.device, native, 2048)?;
    // Independent tap arithmetic oracle: the pinned stream_mean kernel on
    // identical input. Exercise tail truncation and ring wrap, all five columns.
    let rows = 2051usize;
    let input = engine.device.alloc(rows * 4 * 4096 * 2)?;
    let means = engine.device.alloc(rows * 4096 * 2)?;
    let bytes = (0..rows * 4 * 4096)
        .flat_map(|i| {
            half::bf16::from_f32(((i * 17 % 4093) as f32 - 2046.) / 97.)
                .to_bits()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    input.write(0, &bytes)?;
    tap.set_position(&engine.stream, 2046)?;
    words(
        &engine.k["stream_mean"],
        &engine.stream,
        [rows as u32, 4, 1],
        &[input.ptr, means.ptr],
    )?;
    for layer in [5, 14, 24, 33, 42] {
        tap.capture(&engine.stream, &input, rows, layer)?;
    }
    engine.sync()?;
    let reference = means.read(0, means.bytes)?;
    let captured = tap.rows.read(0, tap.rows.bytes)?;
    for row in 3..rows {
        for layer in 0..5 {
            let off = ((2046 + row) % 2048) * WIDTH + layer * 8192;
            ensure!(
                captured[off..off + 8192] == reference[row * 8192..(row + 1) * 8192],
                "tap mean/striding mismatch"
            );
        }
    }
    drop((input, means, bytes, reference, captured));
    engine.taps = Some(tap);
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let temperature = diagnostic_temperature(&long)?;
    let mut verifier = Verifier::new(&long, root, native)?;
    verifier.batch_dsa = true;
    verifier.use_graphs = use_graphs;
    verifier.audit = false;
    verifier.taps = Some(Taps::new(&long.engine.device, native, 8)?);
    let mut draft =
        super::super::drafter::Drafter::new(&long.engine, draft_manifest, root, native)?;
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let source = token_sets.last().context("speculative fixture")?;
    ensure!(source.len() >= 8192, "speculative fixture length");
    let mut cases = Vec::new();
    for pos in [128usize, 2050, 8190] {
        for (seed, top_k) in [(731u64, 20usize), (917, 128)] {
            let policy = Sampling::new(seed, temperature, top_k, 0.95)?;
            let prefix = &source[..pos];
            let mut expected = Vec::new();
            let mut states = Vec::new();
            let mut taps = Vec::new();
            long.start(pos, 48)?;
            for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                long.prefill(i * ROWS, chunk)?;
            }
            let first = long.sample(&mut sampler, policy)?.0 as i32;
            let mut next = first;
            let mut serial_seconds = 0.;
            for row in 0..32 {
                let start = Instant::now();
                long.step(pos + row, next)?;
                next = long.sample(&mut sampler, policy)?.0 as i32;
                serial_seconds += start.elapsed().as_secs_f64();
                expected.push(next);
                states.push(long.state(pos + row + 1)?);
                let off = ((pos + row) % 2048) * WIDTH;
                taps.push(format!(
                    "{:x}",
                    Sha256::digest(long.engine.taps.as_ref().unwrap().rows.read(off, WIDTH)?)
                ));
            }
            long.start(pos, 48)?;
            for (i, chunk) in prefix.chunks(ROWS).enumerate() {
                long.prefill(i * ROWS, chunk)?;
            }
            ensure!(
                long.sample(&mut sampler, policy)?.0 as i32 == first,
                "draft cold prefix drift"
            );
            draft.start_context(&long.engine, pos.saturating_sub(2047))?;
            draft.append_history(&long.engine, long.engine.taps.as_ref().unwrap(), pos)?;
            long.engine.sync()?;
            let mut committed = 0;
            let mut rounds = Vec::new();
            while committed < 32 {
                let pending = long.sessions[long.active].pending.unwrap();
                let start = Instant::now();
                let proposals = draft.propose(&long.engine, pending, policy)?;
                let draft_seconds = start.elapsed().as_secs_f64();
                let n = (32 - committed).min(8);
                let mut input = vec![pending];
                input.extend_from_slice(&proposals[..n - 1]);
                let start = Instant::now();
                let out = verifier.execute(&mut long, &input, Decision::Sampled(policy))?;
                let verify_seconds = start.elapsed().as_secs_f64();
                ensure!(
                    out.sampled[..out.keep] == expected[committed..committed + out.keep],
                    "speculative output drift"
                );
                for row in 0..out.keep {
                    let off = ((pos + committed + row) % 2048) * WIDTH;
                    ensure!(
                        format!(
                            "{:x}",
                            Sha256::digest(
                                long.engine.taps.as_ref().unwrap().rows.read(off, WIDTH)?
                            )
                        ) == taps[committed + row],
                        "accepted target tap drift"
                    );
                }
                committed += out.keep;
                ensure!(
                    long.state(pos + committed)? == states[committed - 1],
                    "speculative state drift"
                );
                let start = Instant::now();
                draft.append_history(
                    &long.engine,
                    long.engine.taps.as_ref().unwrap(),
                    pos + committed,
                )?;
                long.engine.sync()?;
                let context_seconds = start.elapsed().as_secs_f64();
                rounds.push(serde_json::json!({"rows":n,"keep":out.keep,"verifier_graph_built":out.graph_built,"draft_seconds":draft_seconds,"verify_seconds":verify_seconds,"context_seconds":context_seconds}));
            }
            eprintln!(
                "DFlash sampled exact pos={pos} seed={seed} top_k={top_k} rounds={}",
                rounds.len()
            );
            cases.push(serde_json::json!({"position":pos,"seed":seed,"temperature":temperature,"top_k":top_k,"serial_seconds":serial_seconds,"rounds":rounds,"exact_target_tokens":true,"exact_committed_state":true,"exact_accepted_taps":true}));
        }
    }
    Ok(
        serde_json::json!({"rank":rank,"temperature":temperature,"l2_prefetch":long.engine.prefetch_receipt(),"verifier_graphs":use_graphs,"tap_mean_exact":true,"cases":cases,"scope":"Real DFlash proposals plus greedy or sampled full-target verification with batched exact DSA projections. Diagnostic one owner with exact serial output/state/taps. Timings exclude state hashes but are not production speed."}),
    )
}

#[cfg(test)]
mod clipping_tests {
    use super::clipped_keep;

    #[test]
    fn hook_cannot_observe_predictions_after_the_first_rejection() -> anyhow::Result<()> {
        let keep = clipped_keep(&[10, 20, 21, 99], &[20, 21, 22, 154827], true, |accepted| {
            assert_eq!(accepted, [20, 21, 22]);
            Ok(2)
        })?;
        assert_eq!(keep, 2);
        Ok(())
    }

    #[test]
    fn eos_is_clamped_before_the_hook_but_fixed_length_control_is_unchanged() -> anyhow::Result<()>
    {
        for stop_eos in [false, true] {
            let keep = clipped_keep(
                &[10, 20, 154827, 99],
                &[20, 154827, 99, 100],
                stop_eos,
                |accepted| {
                    assert_eq!(
                        accepted,
                        if stop_eos {
                            &[20, 154827][..]
                        } else {
                            &[20, 154827, 99, 100][..]
                        }
                    );
                    Ok(accepted.len())
                },
            )?;
            assert_eq!(keep, if stop_eos { 2 } else { 4 });
        }
        Ok(())
    }

    #[test]
    fn invalid_clip_or_hook_failure_is_a_value_for_collective_validation() {
        let tokens = [10, 20, 21, 22];
        let sampled = [20, 21, 22, 23];
        for keep in [0, 5, usize::MAX] {
            assert!(clipped_keep(&tokens, &sampled, false, |_| Ok(keep)).is_err());
        }
        assert!(
            clipped_keep(&tokens, &sampled, false, |_| anyhow::bail!(
                "output parser failed"
            ))
            .is_err()
        );
        let called = std::cell::Cell::new(false);
        assert!(
            clipped_keep(&tokens, &sampled[..3], false, |_| {
                called.set(true);
                Ok(1)
            })
            .is_err()
        );
        assert!(!called.get());
        assert!(clipped_keep(&[], &[], false, |_| Ok(0)).is_err());
    }
}
