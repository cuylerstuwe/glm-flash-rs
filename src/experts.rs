//! EXL3 routed-expert computation, using the pinned device algorithm through a native ABI.
use crate::{
    cuda::{Device, Kernel, Stream, arg},
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use half::f16;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
const STEM: &str = "weights.layers[3].moe.experts";
const HAD: f32 = 0.088_388_346;
fn half_values(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|v| f16::from_bits(u16::from_le_bytes(*v)).to_f32())
        .collect()
}
fn bf(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff0000)
}
fn hf(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}
fn had(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for block in y.as_chunks_mut::<128>().0 {
        let mut stride = 1;
        while stride < 128 {
            for base in (0..128).step_by(stride * 2) {
                for j in 0..stride {
                    let a = block[base + j];
                    let b = block[base + j + stride];
                    block[base + j] = a + b;
                    block[base + j + stride] = a - b;
                }
            }
            stride *= 2;
        }
    }
    y
}
fn codebook(s: u32) -> f32 {
    let x = (s.wrapping_mul(0xcbac1fed) & 0x8fff8fff) ^ 0x3b603b60;
    hf(f16::from_bits(x as u16).to_f32() + f16::from_bits((x >> 16) as u16).to_f32())
}
fn unpack(words: &[u8], k: usize, n: usize) -> Vec<f32> {
    let mut w = vec![0f32; k * n];
    for kt in 0..k / 16 {
        for nt in 0..n / 16 {
            let start = (kt * (n / 16) + nt) * 128;
            let tile: Vec<u32> = words[start..start + 128]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect();
            for p in 0..256 {
                let first = p * 4 + 4 + 1024 - 16;
                let last = first + 16;
                let i0 = (first / 32) % 32;
                let i1 = ((last - 1) / 32) % 32;
                let shift = ((last - 1) / 32 + 1) * 32 - last;
                let state = ((((tile[i0] as u64) << 32) | (tile[i1] as u64)) >> shift) & 0xffff;
                let lane = p / 8;
                let j = p % 8;
                let row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
                let col = lane / 4 + 8 * (j >> 2);
                w[(kt * 16 + row) * n + nt * 16 + col] = codebook(state as u32);
            }
        }
    }
    w
}
// Fixed split/warp order; scalar K accumulation is an independent approximate oracle for tensor cores.
fn matmul(x: &[f32], w: &[f32], n: usize, sk: usize) -> Vec<f32> {
    let k = x.len();
    let mut y = vec![0f32; n];
    for col in 0..n {
        let mut sum = 0f32;
        for split in 0..sk {
            let mut warps = [0f32; 4];
            for (warp, acc) in warps.iter_mut().enumerate() {
                let begin = (split * 4 + warp) * k / (sk * 4);
                let end = begin + k / (sk * 4);
                for j in begin..end {
                    *acc += x[j] * w[j * n + col];
                }
            }
            let partial = ((warps[0] + warps[1]) + warps[2]) + warps[3];
            sum += partial;
        }
        y[col] = sum;
    }
    y
}
fn tensor_head(p: &Prepared, name: &str, bytes: u64) -> Result<Vec<u8>> {
    p.read_tensor_segment(&format!("{STEM}.{name}"), 0, bytes)
}
fn reference_first(p: &Prepared, x: &[f32]) -> Result<Vec<f32>> {
    let mut branches = Vec::new();
    for (trellis, suh, svh) in [("gt", "suh_g", "svh_g"), ("ut", "suh_u", "svh_u")] {
        let in_scale = half_values(&tensor_head(p, suh, 8192)?);
        let out_scale = half_values(&tensor_head(p, svh, 1024)?);
        let scaled: Vec<_> = x.iter().zip(in_scale).map(|(x, s)| x * s).collect();
        let rotated: Vec<_> = had(&scaled).into_iter().map(|v| hf(v * HAD)).collect();
        let w = unpack(&tensor_head(p, trellis, 1048576)?, 4096, 512);
        let y = had(&matmul(&rotated, &w, 512, 4));
        branches.push(
            y.into_iter()
                .zip(out_scale)
                .map(|(y, s)| bf(y * HAD * s))
                .collect::<Vec<_>>(),
        );
    }
    let input_scale = half_values(&tensor_head(p, "suh_d", 1024)?);
    let activation: Vec<_> = branches[0]
        .iter()
        .zip(&branches[1])
        .zip(input_scale)
        .map(|((&g, &u), s)| {
            let g = g.min(10.);
            let u = u.clamp(-10., 10.);
            bf(bf(g / (1. + (-g).exp())) * u) * s
        })
        .collect();
    let rotated: Vec<_> = had(&activation).into_iter().map(|v| hf(v * HAD)).collect();
    let w = unpack(&tensor_head(p, "dt", 1048576)?, 512, 4096);
    let out_scale = half_values(&tensor_head(p, "svh_d", 8192)?);
    Ok(had(&matmul(&rotated, &w, 4096, 1))
        .into_iter()
        .zip(out_scale)
        .map(|(y, s)| y * HAD * s)
        .collect())
}
fn launch(kernel: &Kernel, stream: &Stream, grid: [u32; 3], words: &[u64]) -> Result<()> {
    ensure!(
        cfg!(target_endian = "little"),
        "CUDA launch scalar ABI expects little-endian"
    );
    let mut w = words.to_vec();
    let mut args = w.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { kernel.launch(stream, grid, &mut args) }
}
pub fn probe(p: &Prepared, native: &Path) -> Result<serde_json::Value> {
    for (name, kind, shape) in [
        ("gt", "int32", vec![288, 256, 32, 32]),
        ("ut", "int32", vec![288, 256, 32, 32]),
        ("dt", "int32", vec![288, 32, 256, 32]),
        ("suh_g", "float16", vec![288, 4096]),
        ("suh_u", "float16", vec![288, 4096]),
        ("suh_d", "float16", vec![288, 512]),
        ("svh_g", "float16", vec![288, 512]),
        ("svh_u", "float16", vec![288, 512]),
        ("svh_d", "float16", vec![288, 4096]),
    ] {
        p.expect(&format!("{STEM}.{name}"), kind, &shape)?;
    }
    let d = Device::new()?;
    let m = p.load_selected(&d, &[STEM])?;
    let stream = d.stream()?;
    let rot = crate::front::load_native(&d, native, "rot_in_kernel", 32)?;
    let grouped = crate::front::load_native(&d, native, "exl3_grouped_8_4", 128)?;
    let gateup = crate::front::load_native(&d, native, "gateup_epilogue_kernel", 32)?;
    let down = crate::front::load_native(&d, native, "down_epilogue_kernel", 32)?;
    let alloc = |bytes| -> Result<_> {
        let b = d.alloc(bytes)?;
        b.zero()?;
        Ok(b)
    };
    let values: Vec<f32> = (0..4096)
        .map(|i| bf(((i * 73 % 997) as f32 - 498.) / 512.))
        .collect();
    let x = alloc(8192)?;
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    x.write(0, &bytes)?;
    let picks = alloc(9 * 4)?;
    let ids = alloc(9 * 4)?;
    let count = alloc(4)?;
    let members = alloc(9 * 4)?;
    let picked: [i32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 288];
    let raw: Vec<_> = picked.iter().flat_map(|v| v.to_le_bytes()).collect();
    picks.write(0, &raw)?;
    ids.write(0, &raw)?;
    count.write(0, &9i32.to_le_bytes())?;
    let raw: Vec<_> = (0..9i32).flat_map(i32::to_le_bytes).collect();
    members.write(0, &raw)?;
    let xg = alloc(9 * 4096 * 2)?;
    let xu = alloc(9 * 4096 * 2)?;
    let xd = alloc(9 * 512 * 2)?;
    let z = alloc(2 * 4 * 9 * 4096 * 4)?;
    let y = alloc(9 * 4096 * 4)?;
    let w = |name: &str| m.pointer(p, &format!("{STEM}.{name}"));
    let compute = || -> Result<()> {
        launch(
            &rot,
            &stream,
            [9, 32, 2],
            &[
                x.ptr,
                4096,
                picks.ptr,
                w("suh_g")?,
                w("suh_u")?,
                xg.ptr,
                xu.ptr,
                4096,
                9,
            ],
        )?;
        launch(
            &grouped,
            &stream,
            [9, 4, 8],
            &[
                xg.ptr,
                xu.ptr,
                w("gt")?,
                w("ut")?,
                ids.ptr,
                count.ptr,
                members.ptr,
                z.ptr,
                4096,
                512,
                9,
                4,
                1,
                9,
                288,
            ],
        )?;
        launch(
            &gateup,
            &stream,
            [9, 4, 1],
            &[
                z.ptr,
                picks.ptr,
                w("svh_g")?,
                w("svh_u")?,
                w("suh_d")?,
                xd.ptr,
                9,
                512,
                4,
                9,
                10f32.to_bits() as u64,
                0,
            ],
        )?;
        launch(
            &grouped,
            &stream,
            [9, 32, 1],
            &[
                xd.ptr,
                xd.ptr,
                w("dt")?,
                w("dt")?,
                ids.ptr,
                count.ptr,
                members.ptr,
                z.ptr,
                512,
                4096,
                9,
                1,
                1,
                9,
                288,
            ],
        )?;
        launch(
            &down,
            &stream,
            [9, 32, 1],
            &[z.ptr, picks.ptr, w("svh_d")?, y.ptr, 9, 4096, 1, 9, 0],
        )
    };
    compute()?;
    stream.sync()?;
    let result = y.read(0, y.bytes)?;
    let expected = reference_first(p, &values)?;
    let actual: Vec<_> = result[..4096 * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let mut err = 0f64;
    let mut signal = 0f64;
    let mut max = 0f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite expert output");
        let e = a - b;
        err += (e as f64).powi(2);
        signal += (b as f64).powi(2);
        max = max.max(e.abs());
    }
    let relative_rms = (err / signal.max(1e-30)).sqrt();
    ensure!(
        relative_rms < 0.001,
        "EXL3 reference mismatch, relative RMS {relative_rms}, max {max}"
    );
    let graph = unsafe { stream.capture(compute)? };
    for _ in 0..5 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    ensure!(y.read(0, y.bytes)? == result, "EXL3 graph/eager mismatch");
    let start = Instant::now();
    for _ in 0..50 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let us = start.elapsed().as_secs_f64() * 20000.;
    Ok(
        serde_json::json!({"milestone":"native_exl3_routed_experts","full_inference":false,"fixed_expert_ids":picked,"reference_expert":0,"relative_rms":relative_rms,"max_absolute_error":max,"graph_byte_equal":true,"graph_us":us,"output_sha256":format!("{:x}",Sha256::digest(&result))}),
    )
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LoadPolicy {
    pub optimized: bool,
    pub programmatic: bool,
    pub discard: bool,
}
impl LoadPolicy {
    pub const ORIGINAL: Self = Self {
        optimized: false,
        programmatic: false,
        discard: false,
    };
    fn process() -> Result<Self> {
        // Select the load/prefetch policy explicitly. The absent-value default
        // uses direct loads; all policies preserve target arithmetic and precision.
        let mut policy = match std::env::var("ALASKA_EXPERT_LOADS").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("original") => Self::ORIGINAL,
            Ok("plain") => Self {
                optimized: true,
                ..Self::ORIGINAL
            },
            Ok("pdl") => Self {
                optimized: true,
                programmatic: true,
                ..Self::ORIGINAL
            },
            _ => anyhow::bail!("ALASKA_EXPERT_LOADS must be original/plain/pdl"),
        };
        policy.discard = match std::env::var("ALASKA_EXPERT_DISCARD").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => false,
            Ok("1") => true,
            _ => anyhow::bail!("ALASKA_EXPERT_DISCARD must be 0/1"),
        };
        Ok(policy)
    }
}
pub struct ExpertGpu {
    rot: Kernel,
    grouped: Kernel,
    gateup: Kernel,
    down: Kernel,
    xg: crate::cuda::Buffer,
    xu: crate::cuda::Buffer,
    xd: crate::cuda::Buffer,
    z: crate::cuda::Buffer,
    pub y: crate::cuda::Buffer,
    max_rows: usize,
    discard: bool,
}
impl ExpertGpu {
    pub fn new(d: &std::rc::Rc<Device>, native: &Path) -> Result<Self> {
        Self::new_rows(d, native, 1)
    }
    pub fn new_rows(d: &std::rc::Rc<Device>, native: &Path, max_rows: usize) -> Result<Self> {
        Self::with_policy(d, native, max_rows, LoadPolicy::process()?)
    }
    pub fn with_policy(
        d: &std::rc::Rc<Device>,
        native: &Path,
        max_rows: usize,
        policy: LoadPolicy,
    ) -> Result<Self> {
        ensure!(
            (1..=crate::kda_window::MAX_ROWS).contains(&max_rows),
            "expert verifier scratch bound"
        );
        let alloc = |n| -> Result<_> {
            let b = d.alloc(n)?;
            b.zero()?;
            Ok(b)
        };
        ensure!(
            !policy.programmatic || policy.optimized,
            "PDL needs the dependency-aware kernel"
        );
        let mut grouped = if policy.optimized {
            crate::front::load_native(
                d,
                &native.with_file_name("expert_loads.cubin"),
                "exl3_load_nc_8_1",
                128,
            )?
        } else {
            crate::front::load_native(d, native, "exl3_grouped_8_4", 128)?
        };
        if policy.programmatic {
            grouped = grouped.with_programmatic_dependency()?;
        }
        Ok(Self {
            rot: crate::front::load_native(d, native, "rot_in_kernel", 32)?,
            grouped,
            gateup: crate::front::load_native(d, native, "gateup_epilogue_kernel", 32)?,
            down: crate::front::load_native(d, native, "down_epilogue_kernel", 32)?,
            xg: alloc(max_rows * 9 * 4096 * 2)?,
            xu: alloc(max_rows * 9 * 4096 * 2)?,
            xd: alloc(max_rows * 9 * 512 * 2)?,
            z: alloc(max_rows * 2 * 4 * 9 * 4096 * 4)?,
            y: alloc(max_rows * 9 * 4096 * 4)?,
            max_rows,
            discard: policy.discard,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &crate::prepared::Loaded,
        layer: usize,
        input: u64,
        picks: u64,
        ids: u64,
        count: u64,
        members: u64,
    ) -> Result<()> {
        self.run_rows(stream, p, m, layer, input, picks, ids, count, members, 1, 1)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        stream: &Stream,
        p: &Prepared,
        m: &crate::prepared::Loaded,
        layer: usize,
        input: u64,
        picks: u64,
        ids: u64,
        count: u64,
        members: u64,
        rows: usize,
        maxm: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.max_rows && maxm >= rows && maxm <= self.max_rows,
            "expert verifier row/member bounds"
        );
        let pairs = (rows * 9) as u64;
        let distinct = (rows * 8 + 1) as u32;
        let w = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].moe.experts.{name}"));
        launch(
            &self.rot,
            stream,
            [pairs as u32, 32, 2],
            &[
                input,
                4096,
                picks,
                w("suh_g")?,
                w("suh_u")?,
                self.xg.ptr,
                self.xu.ptr,
                4096,
                9,
            ],
        )?;
        launch(
            &self.grouped,
            stream,
            [distinct, 4, 8],
            &[
                self.xg.ptr,
                self.xu.ptr,
                w("gt")?,
                w("ut")?,
                ids,
                count,
                members,
                self.z.ptr,
                4096,
                512,
                pairs,
                4,
                maxm as u64,
                9,
                288,
            ],
        )?;
        launch(
            &self.gateup,
            stream,
            [pairs as u32, 4, 1],
            &[
                self.z.ptr,
                picks,
                w("svh_g")?,
                w("svh_u")?,
                w("suh_d")?,
                self.xd.ptr,
                pairs,
                512,
                4,
                9,
                10f32.to_bits() as u64,
                self.discard as u64,
            ],
        )?;
        launch(
            &self.grouped,
            stream,
            [distinct, 32, 1],
            &[
                self.xd.ptr,
                self.xd.ptr,
                w("dt")?,
                w("dt")?,
                ids,
                count,
                members,
                self.z.ptr,
                512,
                4096,
                pairs,
                1,
                maxm as u64,
                9,
                288,
            ],
        )?;
        launch(
            &self.down,
            stream,
            [pairs as u32, 32, 1],
            &[
                self.z.ptr,
                picks,
                w("svh_d")?,
                self.y.ptr,
                pairs,
                4096,
                1,
                9,
                self.discard as u64,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hadamard_involution() {
        let x: Vec<_> = (0..128).map(|i| i as f32 - 64.).collect();
        let z = had(&had(&x));
        for (a, b) in x.iter().zip(z) {
            assert_eq!(*a * 128., b);
        }
    }
    #[test]
    fn codebook_finite() {
        for s in 0..65536 {
            assert!(codebook(s).is_finite());
        }
    }
}
