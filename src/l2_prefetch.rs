//! Default-off retained L2 prefetch owner. Setup-only allocations; read hints only.
//! Must be owned after captured graphs and before model weights are destroyed.
use crate::{
    cuda::{Buffer, Device, Fence, Stream},
    l2_prefetch_plan::{self as plan, ExpertPlan, Item, Site, Span, Tensor},
    prepared::{Loaded, Prepared},
};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};
#[path = "l2_prefetch_probe.rs"]
mod probe;
pub(crate) use probe::run as probe;
pub(crate) struct Config {
    pub(crate) library: PathBuf,
    pub(crate) sha256: String,
}
impl Config {
    pub(crate) fn from_env(native: &Path) -> Result<Option<Self>> {
        let enabled = match std::env::var("ALASKA_L2_PREFETCH") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => String::new(),
            Err(error) => return Err(error).context("L2 enablement must be valid UTF-8"),
        };
        let policy = plan::Policy::parse(&enabled)?;
        if policy == plan::Policy::Off {
            return Ok(None);
        }
        let library = std::env::var_os("ALASKA_L2_PREFETCH_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| native.join("libalaska_l2pf.so"));
        ensure!(
            library.is_absolute(),
            "L2 native library must have an absolute path"
        );
        let sha256 = std::env::var("ALASKA_L2_PREFETCH_SHA256")
            .context("L2 candidate needs an explicit native pin")?;
        ensure!(
            sha256.len() == 64
                && sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "L2 native pin encoding"
        );
        Ok(Some(Self { library, sha256 }))
    }
    pub(crate) fn identity(config: Option<&Self>) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"retained-l2-prefetch-bulk-afoe-8m-64k-32k-64rows-v1");
        hash.update([u8::from(config.is_some())]);
        if let Some(c) = config {
            hash.update(c.sha256.as_bytes());
        }
        hash.finalize().into()
    }
}
type Properties = unsafe extern "C" fn(*mut i32, *mut i32) -> i32;
type Capturing = unsafe extern "C" fn(*mut c_void, *mut i32) -> i32;
type Segments = unsafe extern "C" fn(u64, i32, i32, *mut c_void) -> i32;
type Experts = unsafe extern "C" fn(u64, u64, i32, u64, i64, i64, i64, i32, *mut c_void) -> i32;
struct Native {
    properties: Properties,
    capturing: Capturing,
    segments: Segments,
    experts: Experts,
    _library: Library,
    _owner_thread: PhantomData<Rc<()>>,
}
fn check(code: i32) -> Result<()> {
    ensure!(code == 0, "native L2 prefetch CUDA error {code}");
    Ok(())
}
impl Native {
    fn open(path: &Path, pin: &str) -> Result<Self> {
        ensure!(
            pin.len() == 64
                && pin
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "L2 native SHA encoding"
        );
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(path)?)) == pin,
            "L2 native library identity drift"
        );
        unsafe {
            let library = Library::new(path)?;
            let abi: unsafe extern "C" fn() -> u32 = *library.get(b"alaska_l2pf_abi\0")?;
            ensure!(abi() == 1, "L2 native ABI mismatch");
            Ok(Self {
                properties: *library.get(b"alaska_l2pf_properties\0")?,
                capturing: *library.get(b"alaska_l2pf_capturing\0")?,
                segments: *library.get(b"alaska_l2pf_segments\0")?,
                experts: *library.get(b"alaska_l2pf_experts\0")?,
                _library: library,
                _owner_thread: PhantomData,
            })
        }
    }
    fn is_capturing(&self, stream: &Stream) -> Result<bool> {
        let mut active = 0;
        check(unsafe { (self.capturing)(stream.raw(), &mut active) })?;
        ensure!((0..=1).contains(&active), "L2 capture status");
        Ok(active == 1)
    }
}
struct Table {
    buffer: Buffer,
    count: i32,
    grid: i32,
    bytes: u64,
}
struct ExpertTable {
    buffer: Buffer,
    plan: ExpertPlan,
}
#[derive(Default, Clone, Debug, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) launches: u64,
    pub(crate) expert_launches: u64,
    pub(crate) skipped_rows: u64,
    pub(crate) bytes_planned: u64,
    pub(crate) by_site: [u64; 3],
}
pub(crate) struct Owner {
    tables: BTreeMap<(usize, Site), Table>,
    experts: BTreeMap<usize, ExpertTable>,
    fork: Fence,
    joined: Fence,
    side: Stream,
    native: Native,
    sms: u32,
    pending: bool,
    captured: bool,
    stats: Stats,
}
fn tensor(p: &Prepared, m: &Loaded, path: &str) -> Result<Tensor> {
    let t = p.tensor(path)?;
    let mut stride = 1u64;
    let mut contiguous = true;
    for (&n, &s) in t.shape.iter().zip(&t.stride).rev() {
        if n > 1 && s != stride {
            contiguous = false;
        }
        stride = stride.checked_mul(n).context("L2 tensor shape overflow")?;
    }
    Ok(Tensor {
        address: m.pointer(p, path)?,
        bytes: stride
            .checked_mul(t.element_bytes()?)
            .context("L2 tensor byte overflow")?,
        contiguous,
    })
}
fn item(p: &Prepared, m: &Loaded, path: &str) -> Result<Item> {
    if p.tensors.contains_key(path) {
        return Ok(Item::Tensor(tensor(p, m, path)?));
    }
    let weight = tensor(p, m, &format!("{path}.weight"))?;
    let scale = format!("{path}.scale");
    if p.tensors.contains_key(&scale) {
        return Ok(Item::Scaled {
            scale: tensor(p, m, &scale)?,
            weight,
        });
    }
    // Canonical non-expert weights are FP8/BF16. Do not infer a Q4 schedule
    // from an unexpected prepared cache and silently hint a different layout.
    ensure!(
        !p.tensors.contains_key(&format!("{path}.scales")),
        "noncanonical L2 matrix layout: {path}"
    );
    Ok(Item::Tensor(weight))
}
fn items(p: &Prepared, m: &Loaded, paths: Vec<String>) -> Result<Vec<Item>> {
    paths.into_iter().map(|path| item(p, m, &path)).collect()
}
fn table(device: &Rc<Device>, spans: &[Span], sms: u32) -> Result<Table> {
    ensure!(
        !spans.is_empty() && spans.len() <= i32::MAX as usize,
        "L2 span table bounds"
    );
    let mut bytes = Vec::with_capacity(spans.len() * 16);
    let mut total = 0;
    for span in spans {
        bytes.extend_from_slice(&span.address.to_ne_bytes());
        bytes.extend_from_slice(&span.bytes.to_ne_bytes());
        total += span.bytes;
    }
    let buffer = device.alloc(bytes.len())?;
    buffer.write(0, &bytes)?;
    Ok(Table {
        buffer,
        count: spans.len() as i32,
        grid: plan::grid(total, sms)? as i32,
        bytes: total,
    })
}
impl Owner {
    /// Explicit opt-in only. The model must retain all Prepared/Loaded weights
    /// and this owner until every referencing graph has been destroyed.
    pub(crate) fn new(
        device: &Rc<Device>,
        p: &Prepared,
        m: &Loaded,
        library: &Path,
        pin: &str,
    ) -> Result<Self> {
        ensure!(
            device.arch == 121,
            "L2 retained candidate requires GB10 sm121"
        );
        device.memory()?; // Set this driver's primary context before CUDA runtime queries.
        let native = Native::open(library, pin)?;
        let mut sms = 0;
        let mut l2 = 0;
        check(unsafe { (native.properties)(&mut sms, &mut l2) })?;
        ensure!(sms > 0 && l2 >= 0, "L2 native device properties");
        let budget = plan::budget(l2 as u64);
        let mut owner = Self {
            tables: BTreeMap::new(),
            experts: BTreeMap::new(),
            fork: device.fence()?,
            joined: device.fence()?,
            side: device.stream()?,
            native,
            sms: sms as u32,
            pending: false,
            captured: false,
            stats: Stats::default(),
        };
        // The target has 45 layers. This path does not execute MTP;
        // DFlash uses separate weights and scheduling.
        for layer in 0..45 {
            let prefix = format!("weights.layers[{layer}]");
            let attn = if layer % 4 == 3 { "dsa" } else { "kda" };
            let mut after_attention =
                vec![format!("{prefix}.ffn_hc.fn"), format!("{prefix}.post_norm")];
            if p.tensors.contains_key(&format!("{prefix}.moe.router")) {
                after_attention.extend([
                    format!("{prefix}.moe.router"),
                    format!("{prefix}.moe.bias"),
                    format!("{prefix}.moe.shared.gu"),
                ]);
                let gate_path = format!("{prefix}.moe.experts.gt");
                let up_path = format!("{prefix}.moe.experts.ut");
                let gate = p.tensor(&gate_path)?;
                ensure!(
                    gate.shape == p.tensor(&up_path)?.shape,
                    "L2 routed expert shapes differ"
                );
                let plan = plan::experts(
                    tensor(p, m, &gate_path)?,
                    tensor(p, m, &up_path)?,
                    *gate.shape.first().context("L2 expert rank")?,
                )?
                .context("canonical expert layout cannot be prefetched")?;
                let bases = plan
                    .bases
                    .into_iter()
                    .flat_map(u64::to_ne_bytes)
                    .collect::<Vec<_>>();
                let buffer = device.alloc(bases.len())?;
                buffer.write(0, &bases)?;
                owner.experts.insert(layer, ExpertTable { buffer, plan });
            } else {
                after_attention.push(format!("{prefix}.mlp.gu"));
            }
            let after_ffn = if layer < 44 {
                let next = layer + 1;
                let next_attn = if next % 4 == 3 { "dsa" } else { "kda" };
                vec![
                    format!("weights.layers[{next}].attn_hc.fn"),
                    format!("weights.layers[{next}].in_norm"),
                    format!("weights.layers[{next}].{next_attn}.proj"),
                ]
            } else {
                vec!["weights.norm".into(), "weights.head".into()]
            };
            let mut after_core = Vec::new();
            if attn == "dsa" {
                after_core.push(format!("{prefix}.dsa.kv_v"));
            }
            after_core.push(format!("{prefix}.{attn}.o"));
            for (index, (site, paths)) in [
                (Site::AttentionGather, after_attention),
                (Site::FeedForwardGather, after_ffn),
                (Site::AttentionCore, after_core),
            ]
            .into_iter()
            .enumerate()
            {
                let spans = plan::take(&items(p, m, paths)?, budget)?;
                if !spans.is_empty() {
                    let table = table(device, &spans, sms as u32)?;
                    owner.stats.bytes_planned += table.bytes;
                    owner.stats.by_site[index] += table.bytes;
                    owner.tables.insert((layer, site), table);
                }
            }
        }
        Ok(owner)
    }
    fn fork(&mut self, main: &Stream) -> Result<()> {
        let capture = self.native.is_capturing(main)?;
        if self.pending && capture != self.captured {
            self.pending = false;
        }
        self.fork.record(main)?;
        self.fork.wait_on(&self.side)?;
        self.captured = capture;
        self.pending = true;
        Ok(())
    }
    pub(crate) fn mark(
        &mut self,
        main: &Stream,
        layer: usize,
        site: Site,
        rows: usize,
    ) -> Result<()> {
        if !plan::rows_enabled(rows) {
            self.stats.skipped_rows += 1;
            return Ok(());
        }
        let Some(table) = self.tables.get(&(layer, site)) else {
            return Ok(());
        };
        let (ptr, count, grid) = (table.buffer.ptr, table.count, table.grid);
        self.fork(main)?;
        check(unsafe { (self.native.segments)(ptr, count, grid, self.side.raw()) })?;
        self.stats.launches += 1;
        Ok(())
    }
    /// Call immediately after grouped routing, before shared expert work. IDs
    /// and count storage must remain alive until the forward's final join.
    /// Later layers reuse it; the kernel bounds-checks every
    /// ID/count, so a late hint may warm an unneeded expert but never a model value.
    pub(crate) fn experts(
        &mut self,
        main: &Stream,
        layer: usize,
        rows: usize,
        ids: &Buffer,
        count: &Buffer,
    ) -> Result<()> {
        if !plan::rows_enabled(rows) {
            return Ok(());
        }
        let Some(table) = self.experts.get(&layer) else {
            return Ok(());
        };
        ensure!(
            ids.bytes > 0
                && ids.bytes.is_multiple_of(4)
                && ids.bytes / 4 <= i32::MAX as usize
                && count.bytes >= 4,
            "L2 routing buffer bounds"
        );
        let (bases, plan) = (table.buffer.ptr, table.plan);
        let (maxu, grid) = plan::expert_grid(rows, plan.experts, ids.bytes / 4, self.sms)?;
        self.fork(main)?;
        check(unsafe {
            (self.native.experts)(
                ids.ptr,
                count.ptr,
                maxu,
                bases,
                plan.stride as i64,
                plan.bytes as i64,
                plan.experts as i64,
                grid,
                self.side.raw(),
            )
        })?;
        self.stats.expert_launches += 1;
        Ok(())
    }
    /// Every eager/captured forward must close its fork, including paths that
    /// deliberately omit the final head. A graph may never return an open fork.
    pub(crate) fn join(&mut self, main: &Stream) -> Result<()> {
        if self.pending {
            if self.native.is_capturing(main)? == self.captured {
                self.joined.record(&self.side)?;
                self.joined.wait_on(main)?;
            }
            self.pending = false;
        }
        Ok(())
    }
    pub(crate) fn stats(&self) -> &Stats {
        &self.stats
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        // No table/library/weight reference may disappear during an eager read.
        // Graph ownership remains the caller's documented Engine lifetime rule.
        let _ = self.side.sync();
    }
}
