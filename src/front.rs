//! First TP-local attention branch. Deliberately no claim of a complete model forward.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream, arg},
    prepared::{Loaded, Prepared},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, rc::Rc, time::Instant};
#[derive(Deserialize)]
struct Spec {
    key: String,
    name: String,
    cubin_sha256: String,
    metadata_sha256: String,
    ptx_sha256: String,
}
pub(crate) fn load_native(
    device: &Rc<Device>,
    path: &Path,
    name: &str,
    threads: u32,
) -> Result<Kernel> {
    load_native_shared(device, path, name, threads, 0)
}
pub(crate) fn load_native_shared(
    device: &Rc<Device>,
    path: &Path,
    name: &str,
    threads: u32,
    shared: u32,
) -> Result<Kernel> {
    let expected = match path.file_name().and_then(|n| n.to_str()) {
        Some("kda.cubin") => "341c5c2329c89b611f8ae710e55e668f80fd967a63d8d800cc93f79c09b449e7",
        Some("exl3.cubin") => "3e8a6aca7055e0fab162880d7c2caadea3fb545e34654eec6b24090d4f7bb2c9",
        Some("expert_loads.cubin") => {
            "2b890bda39a577693d10d01dbf39e135e37bda86f0a75d19529462e5af9166b6"
        }
        Some("select.cubin") => "aab9bb8e62f12d84ba139c2d549f3726c4ccaf40253d9cd7f66d2ec08d6857ac",
        Some("sampling.cubin") => {
            "8e9716171b77218d7d8993546105ea4e0fca1b5f8290b501dc57c30b729fcf0e"
        }
        Some("candidates.cubin") => {
            "24a3cd6a326010efc2535e4aae8c4d17c79334c907b67d9e74af1e089a37a03d"
        }
        Some("packed_candidates.cubin") => {
            "b20e5d4965b08cae090e4a71c85cfaa5093ef2d6c7652801f3dfd6ea920a2154"
        }
        Some("masked_candidates.cubin") => {
            "472dde6c4ae963dc3fa4458b7b94a1a722dcf2d130398a349d5014be745fb803"
        }
        Some("prefill.cubin") => "2e287ea49206fe88f18400aeb7c3b4d64612b7fb9c9f6a03293f36ffe4f57c2d",
        Some("exl3_prefill.cubin") => {
            "5197ad1a179e574653a0406d59d437546c35c9d76b7c321ec44ef58564506443"
        }
        Some("grouping.cubin") => {
            "9ea7bb9a37395ead349fb114b4148e09485f31ebaba64d79405a735fe9ef8af7"
        }
        Some("target_taps.cubin") => {
            "f6e147883b04aeb591c7a47f75562f6015c736ecaa4efda89f58da667e4b68d0"
        }
        Some("draft_support.cubin") => {
            "4ddc4e632377c6cb36ca42ed5e2c5cfdf78e820bf27775ef292e72cb276e2df7"
        }
        Some("batch_copy.cubin") => {
            "3d44fee3b99db336168c84c71b2325859178fe717a3585b1fe2649cb445ecc99"
        }
        Some("vision_rows.cubin") => {
            "ee1d546afee9ada8ffc632ea9557eaf06a939cdd04e62882fadd7b0d7f534bc3"
        }
        _ => anyhow::bail!("unrecognized native device artifact"),
    };
    ensure!(device.arch == 121, "native kernels require sm121");
    ensure!(
        format!("{:x}", Sha256::digest(std::fs::read(path)?)) == expected,
        "native cubin hash drift"
    );
    Kernel::load(device, path, name, threads, shared)
}
pub fn load_kernel(device: &Rc<Device>, root: &Path, alias: &str) -> Result<Kernel> {
    let specs: BTreeMap<String, Spec> =
        serde_json::from_str(include_str!("../native/kernel-manifest.json"))?;
    let s = specs.get(alias).context("unknown kernel alias")?;
    let prefix = root.join(&s.key).join(&s.name);
    for (ext, sha) in [
        ("cubin", &s.cubin_sha256),
        ("json", &s.metadata_sha256),
        ("ptx", &s.ptx_sha256),
    ] {
        ensure!(
            format!(
                "{:x}",
                Sha256::digest(std::fs::read(prefix.with_extension(ext))?)
            ) == *sha,
            "kernel {alias} {ext} drift"
        );
    }
    crate::cached(device, root, &s.key, &s.name, &s.cubin_sha256)
}
fn pointer(m: &Loaded, p: &Prepared, s: &str) -> Result<u64> {
    m.pointer(p, &format!("weights.layers[0].{s}"))
}
pub(crate) fn launch_ptrs(
    kernel: &Kernel,
    stream: &Stream,
    grid: [u32; 3],
    ptrs: &[u64],
) -> Result<()> {
    let mut words = ptrs.to_vec();
    words.extend([0, 0]);
    let mut args = words.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { kernel.launch(stream, grid, &mut args) }
}
#[allow(clippy::too_many_arguments)] // Mirrors the pinned kernel's positional launch ABI.
pub(crate) fn fmm(
    kernel: &Kernel,
    stream: &Stream,
    input: u64,
    w: u64,
    scale: u64,
    out: u64,
    part: u64,
    k: i32,
    n: i32,
    sk: u32,
) -> Result<()> {
    let mut ptrs = [input, w, scale, out, part];
    let mut stride = k;
    let mut out_stride = n;
    let mut nulls = [0u64; 2];
    let mut args = ptrs.iter_mut().map(arg).collect::<Vec<_>>();
    args.extend([arg(&mut stride), arg(&mut out_stride)]);
    args.extend(nulls.iter_mut().map(arg));
    unsafe { kernel.launch(stream, [1, (n as u32).div_ceil(64), sk], &mut args) }
}
pub(crate) fn reduce(kernel: &Kernel, stream: &Stream, part: u64, out: u64, n: i32) -> Result<()> {
    let mut a = part;
    let mut b = out;
    let mut n = n;
    let mut z = 0u64;
    let mut z2 = 0u64;
    unsafe {
        kernel.launch(
            stream,
            [(n as u32).div_ceil(1024), 1, 1],
            &mut [
                arg(&mut a),
                arg(&mut b),
                arg(&mut n),
                arg(&mut z),
                arg(&mut z2),
            ],
        )
    }
}
pub(crate) fn bf(v: f32) -> f32 {
    f32::from_bits((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) & 0xffff0000)
}
pub(crate) fn bf_values(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|v| f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16))
        .collect()
}
pub(crate) fn f32_values(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
pub(crate) fn fp8(v: u8) -> f32 {
    let e = (v >> 3) & 15;
    let m = v & 7;
    let sign = if v & 128 != 0 { -1.0 } else { 1.0 };
    if e == 0 {
        sign * (m as f32) * 2f32.powi(-9)
    } else {
        sign * (1. + m as f32 / 8.) * 2f32.powi(e as i32 - 7)
    }
}
// Independent scalar Rust oracle: same fixed split-K and per-slice FP32 scale/rounding.
fn reference_mm(p: &Prepared, name: &str, x: &[f32], sk: usize, round: bool) -> Result<Vec<f32>> {
    let stem = format!("weights.layers[0].{name}");
    reference_mm_at(p, &stem, x, sk, round)
}
pub(crate) fn reference_mm_at(
    p: &Prepared,
    stem: &str,
    x: &[f32],
    sk: usize,
    round: bool,
) -> Result<Vec<f32>> {
    let w = p.read_tensor(&format!("{stem}.weight"))?;
    let s = bf_values(&p.read_tensor(&format!("{stem}.scale"))?);
    let k = x.len();
    ensure!(w.len() == s.len() * k, "matrix oracle dimensions");
    let mut y = vec![0f32; s.len()];
    for n in 0..s.len() {
        let mut sum = 0f32;
        for part in 0..sk {
            let mut acc = 0f32;
            for j in part * k / sk..(part + 1) * k / sk {
                acc += x[j] * fp8(w[n * k + j]);
            }
            let partial = acc * s[n];
            sum = if part == 0 { partial } else { sum + partial };
        }
        y[n] = if round { bf(sum) } else { sum };
    }
    Ok(y)
}
pub(crate) fn compare(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<serde_json::Value> {
    ensure!(actual.len() == expected.len(), "reference length mismatch");
    let mut max = 0f32;
    let mut rms = 0f64;
    let mut signal = 0f64;
    let mut outside = 0usize;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "non-finite model activation"
        );
        let err = (a - b).abs();
        max = max.max(err);
        rms += (err as f64).powi(2);
        signal += (b as f64).powi(2);
        if err > atol + rtol * b.abs() {
            outside += 1;
        }
    }
    ensure!(
        outside == 0,
        "{outside}/{} scalar-reference values exceeded atol={atol}, rtol={rtol}, max_abs={max}, relative_rms={}",
        actual.len(),
        (rms / signal.max(1e-30)).sqrt()
    );
    Ok(
        serde_json::json!({"elements":actual.len(),"max_absolute_error":max,"relative_rms":(rms/signal.max(1e-30)).sqrt(),"atol":atol,"rtol":rtol}),
    )
}
fn warp_reduce(x: &[f32]) -> f32 {
    assert_eq!(x.len(), 128);
    let mut lanes = [0f32; 32];
    for l in 0..32 {
        for j in 0..4 {
            lanes[l] += x[l * 4 + j];
        }
    }
    for shift in [16, 8, 4, 2, 1] {
        let old = lanes;
        for l in 0..32 {
            lanes[l] = old[l] + old[l ^ shift];
        }
    }
    lanes[0]
}
fn reference_kda_first(p: &Prepared, proj: &[f32], gate: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let cw = bf_values(&p.read_tensor("weights.layers[0].kda.conv")?);
    let nw = bf_values(&p.read_tensor("weights.layers[0].kda.norm")?);
    let mut y = vec![0f32; 2048];
    let mut state = vec![0f32; 16 * 128 * 128];
    for h in 0..16 {
        let mut qkv = [[0f32; 128]; 3];
        for (c, vector) in qkv.iter_mut().enumerate() {
            for (j, entry) in vector.iter_mut().enumerate() {
                let i = c * 2048 + h * 128 + j;
                let v = proj[i] * cw[i * 4 + 3];
                *entry = bf(v / (1. + (-v).exp()));
            }
        }
        for (c, vector) in qkv.iter_mut().enumerate().take(2) {
            let sq = vector.map(|x| x * x);
            let inv = 1. / (warp_reduce(&sq) + 1e-6).sqrt();
            for v in vector {
                *v *= inv;
                if c == 0 {
                    *v *= 1. / 128f32.sqrt();
                }
            }
        }
        let beta = bf(1. / (1. + (-proj[6400 + h]).exp()));
        let mut yh = [0f32; 128];
        for j in 0..128 {
            let delta = qkv[2][j] * beta;
            let mut terms = [0f32; 128];
            for i in 0..128 {
                let s = qkv[1][i] * delta;
                state[(h * 128 + j) * 128 + i] = s;
                terms[i] = s * qkv[0][i];
            }
            yh[j] = bf(warp_reduce(&terms));
        }
        let rinv = 1. / (warp_reduce(&yh.map(|v| v * v)) / 128. + 1e-5).sqrt();
        for j in 0..128 {
            y[h * 128 + j] = bf((nw[j] * (yh[j] * rinv)) / (1. + (-gate[h * 128 + j]).exp()));
        }
    }
    Ok((y, state))
}
pub fn run(
    p: &Prepared,
    root: &Path,
    native: &Path,
    tp: Option<(usize, &Path, std::net::SocketAddr)>,
) -> Result<serde_json::Value> {
    p.expect("weights.embed", "bfloat16", &[154880, 4096])?;
    for (name, kind, shape) in [
        ("attn_hc.fn", "bfloat16", vec![24, 16384]),
        ("attn_hc.base", "float32", vec![24]),
        ("attn_hc.scale", "float32", vec![3]),
        ("in_norm", "bfloat16", vec![4096]),
        ("kda.proj.weight", "float8_e4m3fn", vec![6416, 4096]),
        ("kda.proj.scale", "bfloat16", vec![6416]),
        ("kda.fb.weight", "float8_e4m3fn", vec![2048, 128]),
        ("kda.fb.scale", "bfloat16", vec![2048]),
        ("kda.gb.weight", "float8_e4m3fn", vec![2048, 128]),
        ("kda.gb.scale", "bfloat16", vec![2048]),
        ("kda.o.weight", "float8_e4m3fn", vec![4096, 2048]),
        ("kda.o.scale", "bfloat16", vec![4096]),
        ("kda.conv", "bfloat16", vec![6144, 4]),
        ("kda.a_log", "float32", vec![16]),
        ("kda.dt_bias", "float32", vec![2048]),
        ("kda.norm", "bfloat16", vec![128]),
    ] {
        p.expect(&format!("weights.layers[0].{name}"), kind, &shape)?;
    }
    if tp.is_some() {
        for (name, kind, shape) in [
            ("ffn_hc.fn", "bfloat16", vec![24, 16384]),
            ("ffn_hc.base", "float32", vec![24]),
            ("ffn_hc.scale", "float32", vec![3]),
            ("post_norm", "bfloat16", vec![4096]),
            ("mlp.gu.weight", "float8_e4m3fn", vec![6144, 4096]),
            ("mlp.gu.scale", "bfloat16", vec![6144]),
            ("mlp.down.weight", "float8_e4m3fn", vec![4096, 3072]),
            ("mlp.down.scale", "bfloat16", vec![4096]),
        ] {
            p.expect(&format!("weights.layers[0].{name}"), kind, &shape)?;
        }
    }
    let d = Device::new()?;
    let m = p.load_selected(&d, &["weights.embed", "weights.layers[0]."])?;
    let stream = d.stream()?;
    let mut kernels = BTreeMap::new();
    for name in [
        "embed",
        "hc_partial",
        "hc_finish",
        "kda_proj",
        "kda_gate",
        "kda_out",
        "reduce2_bf16",
        "reduce4_f32",
        "hc_post",
        "mlp_gu",
        "swiglu",
        "mlp_down",
    ] {
        kernels.insert(name, load_kernel(&d, root, name)?);
    }
    let chain = load_native(&d, native, "chain_kernel", 1024)?;
    let alloc = |n| -> Result<Buffer> {
        let b = d.alloc(n)?;
        b.zero()?;
        Ok(b)
    };
    let id = alloc(4)?;
    id.write(0, &42i32.to_le_bytes())?;
    let x = alloc(4 * 4096 * 2)?;
    let normed = alloc(4096 * 2)?;
    let xs = alloc(64 * 4)?;
    let post = alloc(4 * 4)?;
    let comb = alloc(16 * 4)?;
    let hcpart = alloc(16 * 32 * 4)?;
    let proj = alloc(6416 * 2)?;
    let partial = alloc(4 * 6416 * 4)?;
    let gate_a = alloc(2048 * 2)?;
    let gate_g = alloc(2048 * 2)?;
    let conv = alloc(3 * 6144 * 2)?;
    let state = alloc(16 * 128 * 128 * 4)?;
    let state_out = alloc(16 * 128 * 128 * 4)?;
    let kout = alloc(2048 * 2)?;
    let out = alloc(4096 * 4)?;
    let ptrw = |s: &str| pointer(&m, p, s);
    let compute = || -> Result<()> {
        launch_ptrs(
            &kernels["embed"],
            &stream,
            [1, 64, 1],
            &[id.ptr, m.pointer(p, "weights.embed")?, x.ptr],
        )?;
        launch_ptrs(
            &kernels["hc_partial"],
            &stream,
            [1, 16, 1],
            &[x.ptr, ptrw("attn_hc.fn")?, hcpart.ptr],
        )?;
        let mut ptrs = [
            x.ptr,
            hcpart.ptr,
            ptrw("attn_hc.base")?,
            ptrw("attn_hc.scale")?,
            ptrw("in_norm")?,
            normed.ptr,
            xs.ptr,
            post.ptr,
            comb.ptr,
        ];
        let mut eps = 1e-5f32;
        let mut hc_eps = 1e-6f32;
        let mut nulls = [0u64; 2];
        let mut args = ptrs.iter_mut().map(arg).collect::<Vec<_>>();
        args.extend([arg(&mut eps), arg(&mut hc_eps)]);
        args.extend(nulls.iter_mut().map(arg));
        unsafe {
            kernels["hc_finish"].launch(&stream, [1, 1, 1], &mut args)?;
        }
        fmm(
            &kernels["kda_proj"],
            &stream,
            normed.ptr,
            ptrw("kda.proj.weight")?,
            ptrw("kda.proj.scale")?,
            proj.ptr,
            partial.ptr,
            4096,
            6416,
            2,
        )?;
        reduce(
            &kernels["reduce2_bf16"],
            &stream,
            partial.ptr,
            proj.ptr,
            6416,
        )?;
        for (offset, name, dst) in [(6144, "fb", gate_a.ptr), (6272, "gb", gate_g.ptr)] {
            fmm(
                &kernels["kda_gate"],
                &stream,
                proj.ptr + offset * 2,
                ptrw(&format!("kda.{name}.weight"))?,
                ptrw(&format!("kda.{name}.scale"))?,
                dst,
                partial.ptr,
                6416,
                2048,
                1,
            )?;
        }
        let mut h = 16i32;
        let mut pptr = proj.ptr;
        let mut ps = 6416i32;
        let mut boff = 6400i32;
        let mut aptr = gate_a.ptr;
        let mut astride = 2048i32;
        let mut gptr = gate_g.ptr;
        let mut gstride = 2048i32;
        let mut cp = conv.ptr;
        let mut cw = ptrw("kda.conv")?;
        let mut state_ptr = state.ptr;
        let mut alog = ptrw("kda.a_log")?;
        let mut dt = ptrw("kda.dt_bias")?;
        let mut nw = ptrw("kda.norm")?;
        let mut lower = -5f32;
        let mut rows = 1i32;
        let mut op = kout.ptr;
        let mut sop = state_out.ptr;
        let mut save = [0u64; 4];
        let mut args = vec![
            arg(&mut h),
            arg(&mut pptr),
            arg(&mut ps),
            arg(&mut boff),
            arg(&mut aptr),
            arg(&mut astride),
            arg(&mut gptr),
            arg(&mut gstride),
            arg(&mut cp),
            arg(&mut cw),
            arg(&mut state_ptr),
            arg(&mut alog),
            arg(&mut dt),
            arg(&mut nw),
            arg(&mut eps),
            arg(&mut lower),
            arg(&mut rows),
            arg(&mut op),
            arg(&mut sop),
        ];
        args.extend(save.iter_mut().map(arg));
        unsafe {
            chain.launch(&stream, [16, 1, 1], &mut args)?;
        }
        fmm(
            &kernels["kda_out"],
            &stream,
            kout.ptr,
            ptrw("kda.o.weight")?,
            ptrw("kda.o.scale")?,
            out.ptr,
            partial.ptr,
            2048,
            4096,
            4,
        )?;
        reduce(&kernels["reduce4_f32"], &stream, partial.ptr, out.ptr, 4096)?;
        Ok(())
    };
    compute()?;
    stream.sync()?;
    let norm = bf_values(&normed.read(0, normed.bytes)?);
    let projection = bf_values(&proj.read(0, proj.bytes)?);
    let projection_check = compare(
        &projection,
        &reference_mm(p, "kda.proj", &norm, 2, true)?,
        2e-5,
        0.008,
    )?;
    let ga = bf_values(&gate_a.read(0, gate_a.bytes)?);
    let gg = bf_values(&gate_g.read(0, gate_g.bytes)?);
    let ga_check = compare(
        &ga,
        &reference_mm(p, "kda.fb", &projection[6144..6272], 1, true)?,
        2e-5,
        0.008,
    )?;
    let gg_check = compare(
        &gg,
        &reference_mm(p, "kda.gb", &projection[6272..6400], 1, true)?,
        2e-5,
        0.008,
    )?;
    let ko = bf_values(&kout.read(0, kout.bytes)?);
    let (kref, sref) = reference_kda_first(p, &projection, &gg)?;
    let kda_check = compare(&ko, &kref, 2e-5, 0.008)?;
    let state_check = compare(
        &f32_values(&state_out.read(0, state_out.bytes)?),
        &sref,
        2e-7,
        0.0002,
    )?;
    let expected = reference_mm(p, "kda.o", &ko, 4, false)?;
    let output = out.read(0, out.bytes)?;
    let out_check = compare(&f32_values(&output), &expected, 2e-5, 0.0002)?;
    let nccl = tp
        .map(|(rank, path, master)| crate::comm::Nccl::new(path, rank, 4, master))
        .transpose()?;
    let gathered = alloc(4 * 4096 * 4)?;
    let gu = alloc(6144 * 2)?;
    let activation = alloc(3072 * 2)?;
    let actsum = alloc(48 * 4)?;
    let hc_post = || -> Result<()> {
        let mut ptrs = [x.ptr, x.ptr, gathered.ptr, post.ptr, comb.ptr];
        let mut stride = 4096i32;
        let mut nulls = [0u64; 2];
        let mut args = ptrs.iter_mut().map(arg).collect::<Vec<_>>();
        args.push(arg(&mut stride));
        args.extend(nulls.iter_mut().map(arg));
        unsafe { kernels["hc_post"].launch(&stream, [1, 4, 1], &mut args) }
    };
    let full_compute = || -> Result<()> {
        compute()?;
        if let Some(comm) = &nccl {
            unsafe {
                comm.all_gather_f32(&out, &gathered, &stream)?;
            }
            hc_post()?;
            launch_ptrs(
                &kernels["hc_partial"],
                &stream,
                [1, 16, 1],
                &[x.ptr, ptrw("ffn_hc.fn")?, hcpart.ptr],
            )?;
            let mut ptrs = [
                x.ptr,
                hcpart.ptr,
                ptrw("ffn_hc.base")?,
                ptrw("ffn_hc.scale")?,
                ptrw("post_norm")?,
                normed.ptr,
                xs.ptr,
                post.ptr,
                comb.ptr,
            ];
            let mut eps = 1e-5f32;
            let mut hc_eps = 1e-6f32;
            let mut nulls = [0u64; 2];
            let mut args = ptrs.iter_mut().map(arg).collect::<Vec<_>>();
            args.extend([arg(&mut eps), arg(&mut hc_eps)]);
            args.extend(nulls.iter_mut().map(arg));
            unsafe {
                kernels["hc_finish"].launch(&stream, [1, 1, 1], &mut args)?;
            }
            fmm(
                &kernels["mlp_gu"],
                &stream,
                normed.ptr,
                ptrw("mlp.gu.weight")?,
                ptrw("mlp.gu.scale")?,
                gu.ptr,
                partial.ptr,
                4096,
                6144,
                2,
            )?;
            reduce(&kernels["reduce2_bf16"], &stream, partial.ptr, gu.ptr, 6144)?;
            let mut gp = gu.ptr;
            let mut ap = activation.ptr;
            let mut asp = actsum.ptr;
            let mut limit = 10f32;
            let mut z = 0u64;
            let mut z2 = 0u64;
            unsafe {
                kernels["swiglu"].launch(
                    &stream,
                    [1, 6, 1],
                    &mut [
                        arg(&mut gp),
                        arg(&mut ap),
                        arg(&mut asp),
                        arg(&mut limit),
                        arg(&mut z),
                        arg(&mut z2),
                    ],
                )?;
            }
            fmm(
                &kernels["mlp_down"],
                &stream,
                activation.ptr,
                ptrw("mlp.down.weight")?,
                ptrw("mlp.down.scale")?,
                out.ptr,
                partial.ptr,
                3072,
                4096,
                4,
            )?;
            reduce(&kernels["reduce4_f32"], &stream, partial.ptr, out.ptr, 4096)?;
            unsafe {
                comm.all_gather_f32(&out, &gathered, &stream)?;
            }
            hc_post()?;
        }
        Ok(())
    };
    full_compute()?;
    stream.sync()?;
    let result = if nccl.is_some() { &x } else { &out };
    let output = result.read(0, result.bytes)?;
    let mlp_checks = if nccl.is_some() {
        let n = bf_values(&normed.read(0, normed.bytes)?);
        let gu_actual = bf_values(&gu.read(0, gu.bytes)?);
        let gu_check = compare(
            &gu_actual,
            &reference_mm(p, "mlp.gu", &n, 2, true)?,
            2e-5,
            0.008,
        )?;
        let act_actual = bf_values(&activation.read(0, activation.bytes)?);
        let act_ref: Vec<_> = (0..3072)
            .map(|j| {
                let g = gu_actual[j].min(10.);
                let u = gu_actual[j + 3072].clamp(-10., 10.);
                bf(bf(g / (1. + (-g).exp())) * u)
            })
            .collect();
        let act_check = compare(&act_actual, &act_ref, 2e-5, 0.008)?;
        let down_check = compare(
            &f32_values(&out.read(0, out.bytes)?),
            &reference_mm(p, "mlp.down", &act_actual, 4, false)?,
            2e-5,
            0.0002,
        )?;
        serde_json::json!({"gate_up":gu_check,"activation":act_check,"down":down_check})
    } else {
        serde_json::Value::Null
    };
    let graph = unsafe { stream.capture(full_compute)? };
    for _ in 0..10 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    ensure!(
        result.read(0, result.bytes)? == output,
        "graph/eager bytes differ"
    );
    let start = Instant::now();
    for _ in 0..100 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let us = start.elapsed().as_secs_f64() * 10000.;
    drop(graph);
    stream.sync()?;
    Ok(
        serde_json::json!({"milestone":if nccl.is_some(){"first_complete_tp4_layer"}else{"first_tp_local_kda_attention_branch"},"full_inference":false,"token":42,"loaded_gpu_bytes":m.bytes,"load_seconds":m.seconds,"graph_branch_us":us,"projection_reference":projection_check,"forget_gate_reference":ga_check,"output_gate_reference":gg_check,"kda_reference":kda_check,"kda_state_reference":state_check,"output_projection_reference":out_check,"mlp_reference":mlp_checks,"graph_byte_equal":true,"output_sha256":format!("{:x}",Sha256::digest(&output))}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp8_values() {
        assert_eq!(fp8(0x38), 1.);
        assert_eq!(fp8(0xb8), -1.);
        assert_eq!(fp8(0x7e), 448.);
        assert_eq!(fp8(1), 1. / 512.);
    }
    #[test]
    fn bf16_ties_even() {
        assert_eq!(bf(f32::from_bits(0x3f808000)), 1.);
        assert_eq!(bf(f32::from_bits(0x3f818000)), f32::from_bits(0x3f820000));
    }
}
