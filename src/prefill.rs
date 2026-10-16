//! Qualified fast-prefill arithmetic, with explicit Rust-owned weights/scratch.
//! This component does not lift the model's short-context limit.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{bf, bf_values, fp8, load_kernel, load_native},
    prepared::{Loaded, Prepared},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, rc::Rc, time::Instant};

pub struct Gemms {
    kernels: BTreeMap<(usize, usize, bool), Rc<Kernel>>,
    copy: Kernel,
    blas: Option<Rc<crate::draft_blas::Blas>>,
}
/// Fewest rows at which a shape runs on cuBLAS. Below 129 rows cuBLAS switches
/// algorithms and its sums stop matching the retained kernels bit for bit; the
/// 160-wide indexer shape never matches. Larger floors are where cuBLAS is not
/// faster at 1,024-row attention blocks.
fn blas_rows(n: usize, k: usize, f32_out: bool) -> Option<usize> {
    match (n, k, f32_out) {
        (160, 4096, _) | (2048, 128, _) => None,
        (6416, 4096, false) | (4096, 2048, true) | (4096, 512, true) => Some(4096),
        _ => Some(256),
    }
}
impl Gemms {
    pub fn new(d: &Rc<Device>, root: &Path, native: &Path) -> Result<Self> {
        let specs: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(include_str!("../native/kernel-manifest.json"))?;
        let mut kernels = BTreeMap::new();
        for name in specs.keys().filter(|n| n.starts_with("prefill_b16_")) {
            let shape: Vec<_> = name[12..].split('_').collect();
            ensure!(shape.len() == 3, "prefill GEMM alias shape");
            kernels.insert(
                (shape[0].parse()?, shape[1].parse()?, shape[2] == "f32"),
                Rc::new(load_kernel(d, root, name)?),
            );
        }
        let blas = match std::env::var("ALASKA_PREFILL_BLAS").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("1") => {
                Some(Rc::new(crate::draft_blas::Blas::new(d)?))
            }
            Ok("0") => None,
            _ => anyhow::bail!("ALASKA_PREFILL_BLAS must be 0/1"),
        };
        Ok(Self {
            kernels,
            copy: load_native(d, &native.join("prefill.cubin"), "f8_b16_copy", 256)?,
            blas,
        })
    }
    pub fn prepare_f8(
        &self,
        d: &Rc<Device>,
        stream: &Stream,
        p: &Prepared,
        m: &Loaded,
        stem: &str,
        f32_out: bool,
    ) -> Result<Matrix> {
        let w = p
            .tensors
            .get(&format!("{stem}.weight"))
            .context("FP8 matrix missing")?;
        ensure!(w.shape.len() == 2, "matrix rank");
        let n = w.shape[0] as usize;
        let k = w.shape[1] as usize;
        p.expect(
            &format!("{stem}.weight"),
            "float8_e4m3fn",
            &[n as u64, k as u64],
        )?;
        p.expect(&format!("{stem}.scale"), "bfloat16", &[n as u64])?;
        ensure!(w.stride == [k as u64, 1], "noncontiguous FP8 weight");
        let kernel = self
            .kernels
            .get(&(n, k, f32_out))
            .context("missing prefill GEMM shape")?
            .clone();
        let weight = d.alloc(
            n.checked_mul(k)
                .and_then(|x| x.checked_mul(2))
                .context("weight size overflow")?,
        )?;
        words(
            &self.copy,
            stream,
            [(n * k).div_ceil(256) as u32, 1, 1],
            &[
                m.pointer(p, &format!("{stem}.weight"))?,
                m.pointer(p, &format!("{stem}.scale"))?,
                weight.ptr,
                (n * k) as u64,
                k as u64,
            ],
        )?;
        let blas = self
            .blas
            .clone()
            .and_then(|b| blas_rows(n, k, f32_out).map(|rows| (b, rows)));
        Ok(Matrix {
            kernel,
            weight,
            n,
            k,
            f32_out,
            blas,
        })
    }
}
pub struct Matrix {
    kernel: Rc<Kernel>,
    weight: Buffer,
    n: usize,
    k: usize,
    f32_out: bool,
    blas: Option<(Rc<crate::draft_blas::Blas>, usize)>,
}
impl Matrix {
    pub fn run(&self, s: &Stream, input: u64, stride: usize, out: u64, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= 8192 && stride >= self.k && stride.is_multiple_of(16),
            "invalid prefill GEMM rows/stride"
        );
        if let Some((blas, least)) = &self.blas
            && rows >= *least
        {
            return unsafe {
                blas.gemm(
                    s,
                    self.weight.ptr,
                    self.n,
                    self.k,
                    input,
                    stride,
                    out,
                    rows,
                    self.f32_out,
                )
            };
        }
        // All selected retained kernels have BM=BN=128, SK=1, BK=64 and
        // dynamic (not divisibility-specialized) M. The input row stride is 16-aligned.
        words(
            &self.kernel,
            s,
            [(rows.div_ceil(128) * self.n.div_ceil(128)) as u32, 1, 1],
            &[input, self.weight.ptr, out, rows as u64, stride as u64],
        )
    }
    fn row_bytes(&self) -> usize {
        self.n * if self.f32_out { 4 } else { 2 }
    }
}

const HEADS: usize = 16;
const WORK_FLOATS: usize = 45248; // frozen fast_kda.WORK per (64-row chunk, head)
pub struct FastKda {
    prep: Kernel,
    state: Kernel,
    norm: Kernel,
    shift: Kernel,
    work: Buffer,
    pub output: Buffer,
    max_rows: usize,
}
impl FastKda {
    pub fn new(d: &Rc<Device>, root: &Path, native: &Path, max_rows: usize) -> Result<Self> {
        ensure!(
            max_rows > 0 && max_rows <= 8192,
            "prefill KDA scratch bound"
        );
        Ok(Self {
            prep: load_kernel(d, root, "prefill_kda_prep")?,
            state: load_kernel(d, root, "prefill_kda_state")?,
            norm: load_kernel(d, root, "prefill_kda_norm")?,
            shift: load_native(d, &native.join("prefill.cubin"), "shift_conv", 256)?,
            work: d.alloc(max_rows.div_ceil(64) * HEADS * WORK_FLOATS * 4)?,
            output: d.alloc(max_rows * 2048 * 2)?,
            max_rows,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        projection: u64,
        a: u64,
        g: u64,
        conv: u64,
        state_in: u64,
        state_out: u64,
        position: usize,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            position.is_multiple_of(64) && rows > 0 && rows <= self.max_rows,
            "fast KDA requires an absolute 64-row boundary and bounded rows"
        );
        let ptr = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].kda.{name}"));
        let w = self.work.ptr;
        let chunks = rows.div_ceil(64);
        words(
            &self.prep,
            stream,
            [chunks as u32, HEADS as u32, 1],
            &[
                projection,
                6416,
                6400,
                a,
                2048,
                conv,
                ptr("conv")?,
                ptr("a_log")?,
                ptr("dt_bias")?,
                w,
                w,
                w,
                w,
                w,
                w,
                w,
                w,
                w,
                (-5f32).to_bits() as u64,
                rows as u64,
                0,
            ],
        )?;
        words(
            &self.state,
            stream,
            [HEADS as u32, 2, 1],
            &[
                w,
                w,
                w,
                w,
                w,
                state_in,
                state_out,
                self.output.ptr,
                rows as u64,
                0,
                chunks as u64,
            ],
        )?;
        words(
            &self.norm,
            stream,
            [rows.div_ceil(32) as u32, HEADS as u32, 1],
            &[
                self.output.ptr,
                g,
                2048,
                ptr("norm")?,
                1e-5f32.to_bits() as u64,
                rows as u64,
            ],
        )?;
        words(
            &self.shift,
            stream,
            [24, 1, 1],
            &[projection, conv, rows as u64, 6416, 6144],
        )
    }
}

/// Diagnostic: row counts at which cuBLAS output differs bitwise from each
/// retained prefill GEMM (all multiples of 64 up to 8192 plus unaligned sizes).
pub fn blas_exact_probe(root: &Path, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let gemms = Gemms::new(&d, root, native)?;
    let blas = crate::draft_blas::Blas::new(&d)?;
    let mut seed = 0x2545_F491u32;
    let mut values = |count: usize, scale: f32| {
        bf_bytes((0..count).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2. - 1.) * scale
        }))
    };
    let mut rows: Vec<usize> = (1..=128).map(|i| i * 64).collect();
    rows.extend([
        1, 2, 17, 63, 65, 100, 127, 129, 255, 257, 511, 513, 1000, 1023, 1025, 4097, 8191,
    ]);
    let mut shapes = Vec::new();
    for (&(n, k, f32_out), kernel) in &gemms.kernels {
        let weight = d.alloc(n * k * 2)?;
        weight.write(0, &values(n * k, 0.03))?;
        let matrix = Matrix {
            kernel: kernel.clone(),
            weight,
            n,
            k,
            f32_out,
            blas: None,
        };
        let x = d.alloc(8192 * k * 2)?;
        x.write(0, &values(8192 * k, 1.))?;
        let width = if f32_out { 4 } else { 2 };
        let a = d.alloc(8192 * n * width)?;
        let b = d.alloc(8192 * n * width)?;
        let mut differing = Vec::new();
        for &m in &rows {
            matrix.run(&s, x.ptr, k, a.ptr, m)?;
            unsafe { blas.gemm(&s, matrix.weight.ptr, n, k, x.ptr, k, b.ptr, m, f32_out)? };
            s.sync()?;
            if a.read(0, m * n * width)? != b.read(0, m * n * width)? {
                differing.push(m);
            }
        }
        shapes.push(serde_json::json!({"n":n,"k":k,"f32_out":f32_out,"rows_tested":rows.len(),"differing_rows":differing}));
    }
    Ok(serde_json::json!({"shapes":shapes,
        "scope":"Diagnostic bitwise comparison on synthetic operands; one GPU."}))
}

/// Diagnostic: each retained BF16 prefill GEMM shape against cuBLAS (BF16 inputs,
/// FP32 accumulation) on identical synthetic operands; GPU event timings and the
/// largest output difference. Single GPU, no model weights.
pub fn blas_probe(root: &Path, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let gemms = Gemms::new(&d, root, native)?;
    let blas = crate::draft_blas::Blas::new(&d)?;
    let mut seed = 0x9E37_79B9u32;
    let mut values = |count: usize, scale: f32| {
        bf_bytes((0..count).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2. - 1.) * scale
        }))
    };
    let decode = |bytes: &[u8], f32_out: bool| -> Vec<f32> {
        if f32_out {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        } else {
            bf_values(bytes)
        }
    };
    let mut cases = Vec::new();
    for (&(n, k, f32_out), kernel) in &gemms.kernels {
        let weight = d.alloc(n * k * 2)?;
        weight.write(0, &values(n * k, 0.03))?;
        let matrix = Matrix {
            kernel: kernel.clone(),
            weight,
            n,
            k,
            f32_out,
            blas: None,
        };
        let x = d.alloc(8192 * k * 2)?;
        x.write(0, &values(8192 * k, 1.))?;
        let width = if f32_out { 4 } else { 2 };
        let a = d.alloc(8192 * n * width)?;
        let b = d.alloc(8192 * n * width)?;
        for rows in [1024usize, 8192] {
            let run = |which: usize| -> Result<()> {
                if which == 0 {
                    matrix.run(&s, x.ptr, k, a.ptr, rows)
                } else {
                    unsafe {
                        blas.gemm(&s, matrix.weight.ptr, n, k, x.ptr, k, b.ptr, rows, f32_out)
                    }
                }
            };
            for which in [0, 1, 0, 1] {
                run(which)?;
            }
            s.sync()?;
            let mut ms = [0f32; 2];
            for round in 0..6 {
                let which = round % 2;
                let mut start = d.event()?;
                let mut end = d.event()?;
                start.record(&s)?;
                for _ in 0..10 {
                    run(which)?;
                }
                end.record(&s)?;
                ms[which] += end.elapsed_after(&start)?;
            }
            let calls = 30.;
            let (fb16, cublas) = (ms[0] / calls, ms[1] / calls);
            let ya = decode(&a.read(0, rows * n * width)?, f32_out);
            let yb = decode(&b.read(0, rows * n * width)?, f32_out);
            let mut max_diff = 0f32;
            let mut max_value = 0f32;
            let mut equal = 0usize;
            for (p, q) in ya.iter().zip(&yb) {
                max_diff = max_diff.max((p - q).abs());
                max_value = max_value.max(p.abs());
                equal += usize::from(p.to_bits() == q.to_bits());
            }
            let flops = 2. * rows as f64 * n as f64 * k as f64;
            cases.push(serde_json::json!({"n":n,"k":k,"f32_out":f32_out,"rows":rows,
                "fb16_ms":fb16,"cublas_ms":cublas,"speedup":fb16 / cublas,
                "fb16_tflops":flops / (fb16 as f64 * 1e9),"cublas_tflops":flops / (cublas as f64 * 1e9),
                "max_abs_diff":max_diff,"max_abs_value":max_value,
                "bit_equal_fraction":equal as f64 / ya.len() as f64}));
        }
    }
    Ok(serde_json::json!({"cases":cases,
        "scope":"Diagnostic synthetic-operand timing on one GPU; not model output or serving qualification."}))
}

fn bf_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|v| ((bf(v).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn hash(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}

pub fn probe(p: &Prepared, root: &Path, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let m = p.load(&d)?;
    let s = d.stream()?;
    let gemms = Gemms::new(&d, root, native)?;
    let stem = "weights.layers[0].kda";
    let proj = gemms.prepare_f8(&d, &s, p, &m, &format!("{stem}.proj"), false)?;
    let fb = gemms.prepare_f8(&d, &s, p, &m, &format!("{stem}.fb"), false)?;
    let gb = gemms.prepare_f8(&d, &s, p, &m, &format!("{stem}.gb"), false)?;
    let out = gemms.prepare_f8(&d, &s, p, &m, &format!("{stem}.o"), true)?;
    s.sync()?;
    let mut copied = 0;
    for (name, mat) in [("proj", &proj), ("fb", &fb), ("gb", &gb), ("o", &out)] {
        let weights = p.read_tensor(&format!("{stem}.{name}.weight"))?;
        let scales = bf_values(&p.read_tensor(&format!("{stem}.{name}.scale"))?);
        let expected = bf_bytes(
            weights
                .iter()
                .enumerate()
                .map(|(i, &w)| fp8(w) * scales[i / mat.k]),
        );
        ensure!(
            mat.weight.read(0, expected.len())? == expected,
            "BF16 weight copy differs: {name}"
        );
        copied += expected.len();
    }
    let rows = 257;
    let x = d.alloc(rows * 4096 * 2)?;
    x.write(
        0,
        &bf_bytes((0..rows * 4096).map(|i| {
            if i < 4096 {
                if i == 17 { 1.0 } else { 0.0 }
            } else {
                ((i.wrapping_mul(173).wrapping_add(19) % 257) as f32 - 128.) / 128.
            }
        })),
    )?;
    let projection = d.alloc(rows * proj.row_bytes() + 32)?;
    projection.zero()?;
    let a = d.alloc(rows * fb.row_bytes())?;
    let g = d.alloc(rows * gb.row_bytes())?;
    proj.run(&s, x.ptr, 4096, projection.ptr, rows)?;
    fb.run(&s, projection.ptr + 6144 * 2, 6416, a.ptr, rows)?;
    gb.run(&s, projection.ptr + 6272 * 2, 6416, g.ptr, rows)?;
    s.sync()?;
    let whole = projection.read(0, rows * proj.row_bytes())?;
    let first = bf_values(&whole[..proj.row_bytes()]);
    let stored = bf_values(&proj.weight.read(0, proj.weight.bytes)?);
    ensure!(
        first
            .iter()
            .enumerate()
            .all(|(n, &v)| v == stored[n * 4096 + 17]),
        "unit-vector GEMM mismatch"
    );
    ensure!(
        projection.read(rows * proj.row_bytes(), 32)? == [0u8; 32],
        "GEMM wrote past rows"
    );
    let split = d.alloc(rows * proj.row_bytes())?;
    let mut begin = 0;
    for len in [64, 128, 65] {
        proj.run(
            &s,
            x.ptr + (begin * 4096 * 2) as u64,
            4096,
            split.ptr + (begin * proj.row_bytes()) as u64,
            len,
        )?;
        begin += len;
    }
    s.sync()?;
    ensure!(
        whole == split.read(0, split.bytes)?,
        "GEMM rows changed with partition"
    );

    let kda = FastKda::new(&d, root, native, rows)?;
    let conv = d.alloc(3 * 6144 * 2)?;
    let state = d.alloc(HEADS * 128 * 128 * 4)?;
    conv.zero()?;
    state.zero()?;
    kda.run(
        &s,
        p,
        &m,
        0,
        projection.ptr,
        a.ptr,
        g.ptr,
        conv.ptr,
        state.ptr,
        state.ptr,
        0,
        rows,
    )?;
    s.sync()?;
    let whole_kda = kda.output.read(0, kda.output.bytes)?;
    let whole_state = state.read(0, state.bytes)?;
    let whole_conv = conv.read(0, conv.bytes)?;
    ensure!(
        bf_values(&whole_kda).iter().all(|v| v.is_finite()),
        "nonfinite KDA output"
    );
    let partitions: &[&[usize]] = &[&[64, 128, 65], &[256, 1], &[128, 64, 64, 1]];
    for partition in partitions {
        conv.zero()?;
        state.zero()?;
        let mut begin = 0;
        let mut split_output = Vec::new();
        for &len in *partition {
            kda.run(
                &s,
                p,
                &m,
                0,
                projection.ptr + (begin * 6416 * 2) as u64,
                a.ptr + (begin * 2048 * 2) as u64,
                g.ptr + (begin * 2048 * 2) as u64,
                conv.ptr,
                state.ptr,
                state.ptr,
                begin,
                len,
            )?;
            s.sync()?;
            split_output.extend(kda.output.read(0, len * 2048 * 2)?);
            begin += len;
        }
        ensure!(
            whole_kda == split_output,
            "KDA outputs changed with aligned chunks"
        );
        ensure!(
            whole_state == state.read(0, state.bytes)?,
            "KDA state changed with aligned chunks"
        );
        ensure!(
            whole_conv == conv.read(0, conv.bytes)?,
            "convolution window changed with chunks"
        );
    }
    let prefixes = [1, 63, 64, 65, 128];
    for &len in &prefixes {
        conv.zero()?;
        state.zero()?;
        kda.run(
            &s,
            p,
            &m,
            0,
            projection.ptr,
            a.ptr,
            g.ptr,
            conv.ptr,
            state.ptr,
            state.ptr,
            0,
            len,
        )?;
        s.sync()?;
        ensure!(
            whole_kda[..len * 2048 * 2] == kda.output.read(0, len * 2048 * 2)?,
            "KDA prefix output changed at {len} rows"
        );
    }
    ensure!(
        kda.run(
            &s,
            p,
            &m,
            0,
            projection.ptr,
            a.ptr,
            g.ptr,
            conv.ptr,
            state.ptr,
            state.ptr,
            1,
            64
        )
        .is_err(),
        "unaligned prefill admitted"
    );

    // Same kernel, arithmetic, weights, rows and allocated outputs: isolate
    // matrix batching from the complete model and from decoder precision policy.
    let mut timings = Vec::new();
    for round in 0..5 {
        let mut times = [0.; 2];
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            s.sync()?;
            let start = Instant::now();
            if mode == 0 {
                proj.run(&s, x.ptr, 4096, split.ptr, rows)?;
            } else {
                for i in 0..rows {
                    proj.run(
                        &s,
                        x.ptr + (i * 4096 * 2) as u64,
                        4096,
                        split.ptr + (i * proj.row_bytes()) as u64,
                        1,
                    )?;
                }
            }
            s.sync()?;
            times[mode] = start.elapsed().as_secs_f64();
        }
        ensure!(
            whole == split.read(0, split.bytes)?,
            "timed row-wise GEMM differs"
        );
        timings.push(serde_json::json!({"batch_seconds":times[0],"serial_rows_seconds":times[1]}));
    }
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"rows":rows,"bf16_copy_bytes_exact":copied,
        "kda_partitions":partitions,"prefix_lengths":prefixes,"gemm_partition_exact":true,"unit_vector_exact":true,"kda_partition_outputs_exact":true,
        "kda_partition_state_exact":true,"conv_partition_exact":true,"unaligned_rejected":true,
        "kda_output_sha256":hash(&whole_kda),"kda_state_sha256":hash(&whole_state),"gemm_timings":timings,
        "scope":"Component prefill arithmetic only; does not measure full-model or end-to-end serving performance."}),
    )
}
