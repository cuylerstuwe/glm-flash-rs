//! Exact serial-arithmetic row operations for a bounded target verifier.
//! Scratch is allocated once. Launches allocate no host/device storage and do
//! not synchronize; the caller retains all inputs until the owning stream drains.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream, arg},
    front::load_kernel,
    kda_window::MAX_ROWS,
};
use anyhow::{Result, ensure};
use std::{collections::BTreeMap, path::Path, rc::Rc};

pub struct Ops {
    k: BTreeMap<&'static str, Kernel>,
    hc_part: Buffer,
    pub norm: Buffer,
    pub sums: Buffer,
    pub post: Buffer,
    pub comb: Buffer,
    router_part: Buffer,
    pub router_logits: Buffer,
    pub picks: Buffer,
    pub weights: Buffer,
    pub qa: Buffer,
    pub expanded: Buffer,
    pub hidden: Buffer,
    pub logits: Buffer,
}
#[derive(Clone, Copy)]
pub struct HcWeights {
    pub function: u64,
    pub base: u64,
    pub scale: u64,
    pub norm: u64,
}

/// All pointers must be valid in the stream's context for the pinned kernel ABI.
unsafe fn launch<const N: usize>(
    k: &Kernel,
    s: &Stream,
    grid: [u32; 3],
    values: [u64; N],
) -> Result<()> {
    ensure!(
        cfg!(target_endian = "little"),
        "kernel scalar ABI requires little endian"
    );
    // Triton appends global/profile scratch pointers even when both sizes are
    // zero. Keep the same two null ABI slots as the serial launch helpers.
    ensure!(N + 2 <= 32, "verifier kernel argument bound");
    let mut storage = [0u64; 32];
    storage[..N].copy_from_slice(&values);
    let mut params = storage.each_mut().map(arg);
    unsafe { k.launch(s, grid, &mut params[..N + 2]) }
}
fn rows(n: usize) -> Result<()> {
    ensure!((1..=MAX_ROWS).contains(&n), "verifier row bound");
    Ok(())
}
fn fits(b: &Buffer, n: usize, row_bytes: usize) -> Result<()> {
    rows(n)?;
    ensure!(b.bytes >= n * row_bytes, "verifier buffer bound");
    Ok(())
}
impl Ops {
    pub fn new(d: &Rc<Device>, root: &Path) -> Result<Self> {
        let mut k = BTreeMap::new();
        for name in [
            "hc_partial",
            "hc_finish",
            "hc_post",
            "verify_head",
            "verify_router_part",
            "router_sum",
            "topk",
            "verify_absorb",
            "verify_expand",
            "stream_mean",
            "final_norm",
        ] {
            k.insert(name, load_kernel(d, root, name)?);
        }
        let a = |row_bytes| d.alloc(MAX_ROWS * row_bytes);
        Ok(Self {
            k,
            hc_part: a(16 * 32 * 4)?,
            norm: a(4096 * 2)?,
            sums: a(64 * 4)?,
            post: a(4 * 4)?,
            comb: a(16 * 4)?,
            router_part: a(8 * 288 * 4)?,
            router_logits: a(288 * 4)?,
            picks: a(9 * 4)?,
            weights: a(9 * 4)?,
            qa: a(16 * 512 * 2)?,
            expanded: a(4096 * 2)?,
            hidden: a(4096 * 2)?,
            logits: a(38720 * 2)?,
        })
    }
    /// Input [R,4,4096] BF16; weight pointers are the canonical HC branch.
    pub unsafe fn hc_pre(&self, s: &Stream, x: &Buffer, w: HcWeights, n: usize) -> Result<()> {
        fits(x, n, 4 * 4096 * 2)?;
        unsafe {
            launch(
                &self.k["hc_partial"],
                s,
                [n as u32, 16, 1],
                [x.ptr, w.function, self.hc_part.ptr],
            )?;
            launch(
                &self.k["hc_finish"],
                s,
                [n as u32, 1, 1],
                [
                    x.ptr,
                    self.hc_part.ptr,
                    w.base,
                    w.scale,
                    w.norm,
                    self.norm.ptr,
                    self.sums.ptr,
                    self.post.ptr,
                    self.comb.ptr,
                    1e-5f32.to_bits() as u64,
                    1e-6f32.to_bits() as u64,
                ],
            )
        }
    }
    /// G is [4,R,4096] FP32 in rank order. X and output may be the same buffer.
    pub unsafe fn hc_post(
        &self,
        s: &Stream,
        x: &Buffer,
        g: &Buffer,
        out: &Buffer,
        n: usize,
    ) -> Result<()> {
        fits(x, n, 4 * 4096 * 2)?;
        fits(out, n, 4 * 4096 * 2)?;
        fits(g, n, 4 * 4096 * 4)?;
        unsafe {
            launch(
                &self.k["hc_post"],
                s,
                [n as u32, 4, 1],
                [
                    x.ptr,
                    out.ptr,
                    g.ptr,
                    self.post.ptr,
                    self.comb.ptr,
                    (n * 4096) as u64,
                ],
            )
        }
    }
    /// Input [R,4096] BF16, canonical BF16 router and FP32 bias.
    pub unsafe fn route(
        &self,
        s: &Stream,
        input: &Buffer,
        weight: u64,
        bias: u64,
        n: usize,
    ) -> Result<()> {
        fits(input, n, 4096 * 2)?;
        unsafe {
            launch(
                &self.k["verify_router_part"],
                s,
                [1, 9, 8],
                [input.ptr, weight, self.router_part.ptr, n as u64, 4096],
            )?;
            launch(
                &self.k["router_sum"],
                s,
                [(n * 288).div_ceil(1024) as u32, 1, 1],
                [
                    self.router_part.ptr,
                    self.router_logits.ptr,
                    (n * 288) as u64,
                ],
            )?;
            launch(
                &self.k["topk"],
                s,
                [n as u32, 1, 1],
                [
                    self.router_logits.ptr,
                    bias,
                    self.picks.ptr,
                    self.weights.ptr,
                    2.5f32.to_bits() as u64,
                ],
            )
        }
    }
    /// Input [R,16,256] BF16; canonical BF16 DSA key projection.
    pub unsafe fn absorb(&self, s: &Stream, q: &Buffer, weight: u64, n: usize) -> Result<()> {
        fits(q, n, 4096 * 2)?;
        unsafe {
            launch(
                &self.k["verify_absorb"],
                s,
                [128, 1, 1],
                [q.ptr, weight, weight, weight, self.qa.ptr, n as u64],
            )
        }
    }
    /// Input [R,16,512] FP32; canonical BF16 DSA value projection.
    pub unsafe fn expand(&self, s: &Stream, u: &Buffer, weight: u64, n: usize) -> Result<()> {
        fits(u, n, 16 * 512 * 4)?;
        unsafe {
            launch(
                &self.k["verify_expand"],
                s,
                [64, 1, 1],
                [u.ptr, weight, weight, weight, self.expanded.ptr, n as u64],
            )
        }
    }
    /// Input residual streams [R,4,4096] BF16. Produce all per-rank logits.
    pub unsafe fn head(
        &self,
        s: &Stream,
        x: &Buffer,
        norm: u64,
        weight: u64,
        n: usize,
    ) -> Result<()> {
        fits(x, n, 4 * 4096 * 2)?;
        unsafe {
            launch(
                &self.k["stream_mean"],
                s,
                [n as u32, 4, 1],
                [x.ptr, self.hidden.ptr],
            )?;
            launch(
                &self.k["final_norm"],
                s,
                [n as u32, 1, 1],
                [
                    self.hidden.ptr,
                    4096,
                    norm,
                    self.norm.ptr,
                    4096,
                    self.sums.ptr,
                    1e-5f32.to_bits() as u64,
                ],
            )?;
            // SK=1: PART is unused in this pinned BF16 head variant.
            launch(
                &self.k["verify_head"],
                s,
                [1, 605, 1],
                [
                    self.norm.ptr,
                    weight,
                    self.logits.ptr,
                    self.logits.ptr,
                    n as u64,
                    4096,
                ],
            )
        }
    }
}
