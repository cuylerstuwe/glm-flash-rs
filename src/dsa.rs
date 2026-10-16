//! Exact decode arithmetic for the dense latent-attention path (bounded to 512 tokens).
use crate::{
    cuda::{Buffer, Device, Kernel, Stream, arg},
    front::{
        bf, bf_values, compare, f32_values, fmm, fp8, launch_ptrs, load_kernel, reduce,
        reference_mm_at,
    },
    prepared::{Loaded, Prepared},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, rc::Rc, time::Instant};

pub(crate) fn words(k: &Kernel, s: &Stream, grid: [u32; 3], values: &[u64]) -> Result<()> {
    ensure!(
        cfg!(target_endian = "little"),
        "kernel scalar ABI requires little endian"
    );
    let mut v = values.to_vec();
    v.extend([0, 0]);
    let mut args = v.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { k.launch(s, grid, &mut args) }
}
pub struct Dsa {
    pub(crate) kernels: BTreeMap<&'static str, Kernel>,
    pub projection: Buffer,
    pub qr: Buffer,
    pub latent: Buffer,
    pub q: Buffer,
    pub qa: Buffer,
    pub cache: Rc<Buffer>,
    pub position: Rc<Buffer>,
    pub page_table: Rc<Buffer>,
    pub capacity: usize,
    pub attended: Buffer,
    pub expanded: Buffer,
    pub output: Buffer,
    partial: Buffer,
    sums: Buffer,
    po: Buffer,
    pm: Buffer,
    pl: Buffer,
}
impl Dsa {
    pub fn new(d: &Rc<Device>, root: &Path) -> Result<Self> {
        let mut kernels = BTreeMap::new();
        for name in [
            "dsa_proj",
            "dsa_qb",
            "dsa_out",
            "rms1536",
            "rms512",
            "absorb",
            "expand",
            "latent_write",
            "latent_chunks",
            "latent_merge1",
            "reduce8_bf16",
            "reduce2_bf16",
            "reduce4_f32",
        ] {
            kernels.insert(name, load_kernel(d, root, name)?);
        }
        let alloc = |n| -> Result<Buffer> {
            let b = d.alloc(n)?;
            b.zero()?;
            Ok(b)
        };
        let page_table = alloc(8)?;
        page_table.write(
            0,
            &[0i32, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            kernels,
            projection: alloc(2048 * 2)?,
            qr: alloc(1536 * 2)?,
            latent: alloc(512 * 2)?,
            q: alloc(4096 * 2)?,
            qa: alloc(16 * 512 * 2)?,
            cache: Rc::new(alloc(512 * 528)?),
            position: Rc::new(alloc(4)?),
            page_table: Rc::new(page_table),
            capacity: 512,
            attended: alloc(16 * 512 * 4)?,
            expanded: alloc(4096 * 2)?,
            output: alloc(4096 * 4)?,
            partial: alloc(8 * 4096 * 4)?,
            sums: alloc(64 * 4)?,
            po: alloc(16 * 512 * 4)?,
            pm: alloc(16 * 4)?,
            pl: alloc(16 * 4)?,
        })
    }
    pub fn set_position(&self, stream: &Stream, position: usize) -> Result<()> {
        ensure!(
            position < self.capacity,
            "dense DSA component capacity is 512; use the long-context adapter for sparse selection"
        );
        unsafe { self.position.set_i32(stream, position as i32) }
    }
    pub fn run(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
    ) -> Result<()> {
        self.run_hook(stream, p, m, layer, input, || Ok(()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_hook(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
        before_core: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.run_with(stream, p, m, layer, input, before_core, |a| a.dense(stream))
    }
    fn dense(&self, stream: &Stream) -> Result<()> {
        let k = &self.kernels;
        launch_ptrs(
            &k["latent_chunks"],
            stream,
            [1, 1, 1],
            &[
                self.qa.ptr,
                self.cache.ptr,
                self.cache.ptr,
                self.position.ptr,
                self.po.ptr,
                self.pm.ptr,
                self.pl.ptr,
                self.page_table.ptr,
            ],
        )?;
        words(
            &k["latent_merge1"],
            stream,
            [16, 1, 1],
            &[self.po.ptr, self.pm.ptr, self.pl.ptr, self.attended.ptr, 16],
        )?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run_with(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
        before_core: impl FnOnce() -> Result<()>,
        attend: impl FnOnce(&Self) -> Result<()>,
    ) -> Result<()> {
        let w = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].dsa.{name}"));
        let k = &self.kernels;
        fmm(
            &k["dsa_proj"],
            stream,
            input,
            w("proj.weight")?,
            w("proj.scale")?,
            self.projection.ptr,
            self.partial.ptr,
            4096,
            2048,
            8,
        )?;
        reduce(
            &k["reduce8_bf16"],
            stream,
            self.partial.ptr,
            self.projection.ptr,
            2048,
        )?;
        for (alias, off, name, out, n) in [
            ("rms1536", 0, "q_norm", self.qr.ptr, 1536),
            ("rms512", 1536 * 2, "kv_norm", self.latent.ptr, 512),
        ] {
            words(
                &k[alias],
                stream,
                [1, 1, 1],
                &[
                    self.projection.ptr + off,
                    2048,
                    w(name)?,
                    out,
                    n,
                    self.sums.ptr,
                    1e-5f32.to_bits() as u64,
                ],
            )?;
        }
        fmm(
            &k["dsa_qb"],
            stream,
            self.qr.ptr,
            w("q_b.weight")?,
            w("q_b.scale")?,
            self.q.ptr,
            self.partial.ptr,
            1536,
            4096,
            2,
        )?;
        reduce(
            &k["reduce2_bf16"],
            stream,
            self.partial.ptr,
            self.q.ptr,
            4096,
        )?;
        let wk = w("kv_k.weight")?;
        launch_ptrs(
            &k["absorb"],
            stream,
            [128, 1, 1],
            &[self.q.ptr, wk, wk, wk, self.qa.ptr],
        )?;
        before_core()?;
        launch_ptrs(
            &k["latent_write"],
            stream,
            [1, 1, 1],
            &[
                self.latent.ptr,
                self.cache.ptr,
                self.cache.ptr,
                self.position.ptr,
                self.page_table.ptr,
            ],
        )?;
        attend(self)?;
        let wv = w("kv_v.weight")?;
        launch_ptrs(
            &k["expand"],
            stream,
            [64, 1, 1],
            &[self.attended.ptr, wv, wv, wv, self.expanded.ptr],
        )?;
        fmm(
            &k["dsa_out"],
            stream,
            self.expanded.ptr,
            w("o.weight")?,
            w("o.scale")?,
            self.output.ptr,
            self.partial.ptr,
            4096,
            4096,
            4,
        )?;
        reduce(
            &k["reduce4_f32"],
            stream,
            self.partial.ptr,
            self.output.ptr,
            4096,
        )
    }
}
fn quantize(x: &[f32]) -> (Vec<u8>, f32) {
    let amax = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let bits = amax.to_bits();
    let e = if amax == 0. {
        0
    } else {
        (((bits >> 23) & 255) as i32 - 135 + i32::from((bits & 0x7fffff) > 0x600000))
            .clamp(-126, 126)
    };
    let scale = 2f32.powi(e);
    let q = x
        .iter()
        .map(|v| {
            let y = v.abs() / scale;
            let mut best = 0u8;
            let mut err = f32::INFINITY;
            for code in 0..=126u8 {
                let distance = (fp8(code) - y).abs();
                if distance < err || (distance == err && code & 1 == 0) {
                    best = code;
                    err = distance;
                }
            }
            best | if v.is_sign_negative() { 128 } else { 0 }
        })
        .collect();
    (q, scale)
}
fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let r = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5)
        .sqrt()
        .recip();
    x.iter().zip(w).map(|(x, w)| bf(bf(x * r) * w)).collect()
}
fn reference_absorb(q: &[f32], w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; 16 * 512];
    for h in 0..16 {
        for j in 0..512 {
            let mut acc = 0f32;
            for i in 0..256 {
                acc = q[h * 256 + i].mul_add(w[(h * 256 + i) * 512 + j], acc);
            }
            out[h * 512 + j] = bf(acc);
        }
    }
    out
}
fn reference_expand(u: &[f32], w: &[f32]) -> Vec<f32> {
    (0..4096)
        .map(|i| {
            let mut a = 0f32;
            for j in 0..512 {
                a = u[(i / 256) * 512 + j].mul_add(w[i * 512 + j], a);
            }
            bf(a)
        })
        .collect()
}
fn attention(q: &[f32], rows: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>, usize) {
    let mut out = vec![0f32; 16 * 512];
    let mut rounding_bound = vec![0f32; 16 * 512];
    let mut boundary_probabilities = 0;
    for h in 0..16 {
        let mut max = f32::NEG_INFINITY;
        let mut denominator = 0f32;
        let mut numerator = vec![0f32; 512];
        let mut uncertainty = vec![0f32; 512];
        for tile in rows.chunks(32) {
            let scores = tile
                .iter()
                .map(|r| {
                    q[h * 512..(h + 1) * 512]
                        .iter()
                        .zip(r)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * 0.0625
                })
                .collect::<Vec<_>>();
            let next = scores.iter().copied().fold(max, f32::max);
            let alpha = (max - next).exp();
            denominator *= alpha;
            for v in &mut numerator {
                *v *= alpha;
            }
            for v in &mut uncertainty {
                *v *= alpha;
            }
            for (row, score) in tile.iter().zip(scores) {
                let prob = (score - next).exp();
                // CPU sums/exp and tensor-core sums/exp can straddle a BF16 midpoint.
                // Propagate only the adjacent rounding bins reachable within 20 ppm;
                // do not loosen the comparison for other coefficients or change GPU math.
                let rounded = bf(prob);
                let delta = (rounded - bf(prob * (1. - 2e-5)))
                    .abs()
                    .max((bf(prob * (1. + 2e-5)) - rounded).abs());
                boundary_probabilities += usize::from(delta > 0.);
                denominator += prob;
                for (a, v) in numerator.iter_mut().zip(row) {
                    *a += bf(prob) * v;
                }
                for (a, v) in uncertainty.iter_mut().zip(row) {
                    *a += delta * v.abs();
                }
            }
            max = next;
        }
        for j in 0..512 {
            out[h * 512 + j] = numerator[j] / denominator;
            rounding_bound[h * 512 + j] = uncertainty[j] / denominator;
        }
    }
    (out, rounding_bound, boundary_probabilities)
}
fn compare_attention(
    actual: &[f32],
    expected: &[f32],
    bound: &[f32],
    boundaries: usize,
) -> Result<serde_json::Value> {
    let mut max = 0f32;
    let mut error = 0f64;
    let mut signal = 0f64;
    let mut covered = 0;
    for ((&a, &b), &extra) in actual.iter().zip(expected).zip(bound) {
        ensure!(
            a.is_finite() && b.is_finite() && extra.is_finite(),
            "nonfinite attention reference"
        );
        let err = (a - b).abs();
        let ordinary = 2e-5 + 0.001 * b.abs();
        ensure!(
            err <= ordinary + extra,
            "attention discrepancy {err} exceeds ordinary tolerance {ordinary} plus boundary allowance {extra}"
        );
        covered += usize::from(err > ordinary);
        max = max.max(err);
        error += (err as f64).powi(2);
        signal += (b as f64).powi(2);
    }
    Ok(
        serde_json::json!({"elements":actual.len(),"max_absolute_error":max,"relative_rms":(error/signal.max(1e-30)).sqrt(),"bf16_boundary_probabilities":boundaries,"elements_requiring_boundary_allowance":covered,"maximum_boundary_allowance":bound.iter().copied().fold(0f32,f32::max),"probability_relative_uncertainty":2e-5,"atol":2e-5,"rtol":0.001}),
    )
}
pub fn probe(p: &Prepared, root: &Path) -> Result<serde_json::Value> {
    for (n, t, s) in [
        ("proj.weight", "float8_e4m3fn", vec![2048, 4096]),
        ("q_b.weight", "float8_e4m3fn", vec![4096, 1536]),
        ("kv_k.weight", "bfloat16", vec![4096, 512]),
        ("kv_v.weight", "bfloat16", vec![4096, 512]),
        ("o.weight", "float8_e4m3fn", vec![4096, 4096]),
    ] {
        p.expect(&format!("weights.layers[3].dsa.{n}"), t, &s)?;
    }
    let d = Device::new()?;
    let m = p.load_selected(&d, &["weights.layers[3].dsa."])?;
    let s = d.stream()?;
    let a = Dsa::new(&d, root)?;
    let x = d.alloc(8192)?;
    let tensor = |name: &str| -> Result<Vec<f32>> {
        Ok(bf_values(
            &p.read_tensor(&format!("weights.layers[3].dsa.{name}"))?,
        ))
    };
    let wk = tensor("kv_k.weight")?;
    let wv = tensor("kv_v.weight")?;
    let mut rows = Vec::new();
    let mut checks = Vec::new();
    // Cross both a 32-row attention tile and the 256-row physical-page boundary.
    for pos in 0..258usize {
        let input = (0..4096)
            .map(|i| bf(((i * 73 + pos * 137) % 997) as f32 / 512. - 0.97))
            .collect::<Vec<_>>();
        x.write(
            0,
            &input
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        a.set_position(&s, pos)?;
        a.run(&s, p, &m, 3, x.ptr)?;
        s.sync()?;
        let lat = bf_values(&a.latent.read(0, a.latent.bytes)?);
        let (q, scale) = quantize(&lat);
        let row = a.cache.read(pos * 528, 528)?;
        ensure!(
            row[..512] == q && row[512..516] == scale.to_le_bytes(),
            "FP8 cache mismatch at position {pos}"
        );
        rows.push(q.iter().map(|v| fp8(*v) * scale).collect::<Vec<_>>());
        if [0, 1, 31, 32, 255, 256, 257].contains(&pos) {
            let proj = bf_values(&a.projection.read(0, a.projection.bytes)?);
            let qr = bf_values(&a.qr.read(0, a.qr.bytes)?);
            let query = bf_values(&a.q.read(0, a.q.bytes)?);
            let qa = bf_values(&a.qa.read(0, a.qa.bytes)?);
            let u = f32_values(&a.attended.read(0, a.attended.bytes)?);
            let expanded = bf_values(&a.expanded.read(0, a.expanded.bytes)?);
            let (attention_reference, bound, boundaries) = attention(&qa, &rows);
            let attention_check=compare_attention(&u,&attention_reference,&bound,boundaries).map_err(|e|{
                if let Ok(path)=std::env::var("ALASKA_RUST_DSA_DIAGNOSTICS") {
                    let data=serde_json::json!({"position":pos,"queries":qa,"rows":rows,"actual":u,"expected":attention_reference});
                    let _=std::fs::write(path,serde_json::to_vec(&data).unwrap());
                }
                anyhow::anyhow!("DSA attention position {pos}: {e}")
            })?;
            checks.push(serde_json::json!({"position":pos,
                "projection":compare(&proj,&reference_mm_at(p,"weights.layers[3].dsa.proj",&input,8,true)?,2e-5,0.008)?,
                "query_norm":compare(&qr,&rms(&proj[..1536],&tensor("q_norm")?),2e-5,0.008)?,
                "latent_norm":compare(&lat,&rms(&proj[1536..],&tensor("kv_norm")?),2e-5,0.008)?,
                "query_projection":compare(&query,&reference_mm_at(p,"weights.layers[3].dsa.q_b",&qr,2,true)?,2e-5,0.008)?,
                "absorbed_query":compare(&qa,&reference_absorb(&query,&wk),1e-6,0.0001)?,
                "attention":attention_check,
                "expand":compare(&expanded,&reference_expand(&u,&wv),1e-6,0.0001)?,
                "output":compare(&f32_values(&a.output.read(0,a.output.bytes)?),&reference_mm_at(p,"weights.layers[3].dsa.o",&expanded,4,false)?,2e-5,0.0003)?}));
        }
    }
    let eager = a.output.read(0, a.output.bytes)?;
    let graph = unsafe { s.capture(|| a.run(&s, p, &m, 3, x.ptr))? };
    unsafe {
        graph.launch(&s)?;
    }
    s.sync()?;
    ensure!(
        eager == a.output.read(0, a.output.bytes)?,
        "DSA graph/eager mismatch"
    );
    let start = Instant::now();
    for _ in 0..100 {
        unsafe {
            graph.launch(&s)?;
        }
    }
    s.sync()?;
    ensure!(a.set_position(&s, 512).is_err(), "capacity guard missing");
    Ok(
        serde_json::json!({"component":"dense_latent_attention","positions_checked":258,"cache_row_bytes_exact":true,"checks":checks,"graph_eager_equal":true,"graph_microseconds":start.elapsed().as_secs_f64()*1e4,"output_sha256":format!("{:x}",Sha256::digest(eager)),"scope":"one TP-local DSA branch, at most 512 context tokens; not full model inference"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_bounds_do_not_accept_unexplained_error() {
        assert!(compare_attention(&[0.01], &[0.], &[0.], 0).is_err());
        assert!(compare_attention(&[0.0008], &[0.], &[0.001], 1).is_ok());
    }
    #[test]
    fn e4m3_ties_even_and_sign() {
        let (q, s) = quantize(&[0., -0., 1.0625, 1.1875, 448.]);
        assert_eq!(s, 1.);
        assert_eq!(q, vec![0, 128, 56, 58, 126]);
    }
    #[test]
    fn capacity_quantizer_zero() {
        let (q, s) = quantize(&[0.; 512]);
        assert_eq!(s, 1.);
        assert_eq!(q, vec![0; 512]);
    }
}
