//! Candidate sampler qualification, independent of model execution and transport.
use crate::{
    cuda::{Device, Kernel, Stream, arg},
    front::load_native,
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;

fn launch(k: &Kernel, s: &Stream, grid: usize, values: &mut [u64]) -> Result<()> {
    let mut args = values.iter_mut().map(arg).collect::<Vec<_>>();
    unsafe { k.launch(s, [grid as u32, 1, 1], &mut args) }
}
fn doubles(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn digest(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}
pub(crate) fn probe(native: &Path) -> Result<serde_json::Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let libm = std::fs::read("/lib/aarch64-linux-gnu/libm.so.6")?;
    ensure!(
        digest(&libm) == "5e0391562de36de323fb9cb0f2c1b6bd6d034c840b1dac018499f5fe4ded1b4a",
        "libm oracle provenance drift"
    );
    let math = load_native(
        &device,
        &native.join("sampling.cubin"),
        "sampling_math",
        256,
    )?;
    let draw = load_native(
        &device,
        &native.join("sampling.cubin"),
        "sampling_choose",
        256,
    )?;
    let mut x = vec![
        0.0,
        -0.0,
        1.0,
        f64::MIN_POSITIVE,
        2.0f64.powi(-54),
        37.5,
        512.0,
        700.0,
        745.0,
        1024.0,
        f64::MAX,
    ];
    for bits in [
        0x3fee000000000000u64,
        0x3ff108ffffffffff,
        0x3ff0000000000000,
    ] {
        for delta in 0..1024 {
            x.push(f64::from_bits(bits + delta));
            x.push(f64::from_bits(bits - delta));
        }
    }
    for i in 0..32768 {
        let a = sampling::mix(i);
        x.push(sampling::uniform(a, i, (i % 154880) as u32));
        x.push(-sampling::uniform(a, i, (i % 154880) as u32).ln());
        x.push((a >> 11) as f64 * 2.0f64.powi(-53) * 800.0);
    }
    let src = device.alloc(x.len() * 8)?;
    src.write(0, &doubles(&x))?;
    let dst = device.alloc(x.len() * 24)?;
    let mut math_hash = Sha256::new();
    for seed in [0, u64::MAX, 450] {
        launch(
            &math,
            &stream,
            x.len().div_ceil(256),
            &mut [src.ptr, dst.ptr, x.len() as u64, seed],
        )?;
        stream.sync()?;
        let got = dst.read(0, dst.bytes)?;
        for (i, value) in x.iter().enumerate() {
            let expect = [
                value.ln(),
                (-value).exp(),
                sampling::uniform(seed, i as u64, (i % 154880) as u32),
            ];
            for (j, want) in expect.into_iter().enumerate() {
                let bits = u64::from_le_bytes(got[24 * i + 8 * j..24 * i + 8 * j + 8].try_into()?);
                ensure!(
                    bits == want.to_bits(),
                    "math mismatch input[{i}]={value:?}, op={j}: got={bits:016x}, expected={:016x}",
                    want.to_bits()
                );
            }
        }
        math_hash.update(got);
    }
    const ROWS: usize = 128;
    let values = device.alloc(ROWS * 256 * 4)?;
    let ids = device.alloc(ROWS * 256 * 4)?;
    let ints = device.alloc(ROWS * 4 * 8)?;
    let params = device.alloc(ROWS * 2 * 8)?;
    let out = device.alloc(ROWS * 2 * 4)?;
    let kernel_args = [values.ptr, ids.ptr, ints.ptr, params.ptr, out.ptr];
    let graph =
        unsafe { stream.capture(|| launch(&draw, &stream, ROWS, &mut kernel_args.clone()))? };
    let mut draw_hash = Sha256::new();
    let mut first = None;
    for batch in 0..33 {
        // Repeat the first 128 rows after 4096 unrelated rows, reusing one graph.
        let batch = if batch == 32 { 0 } else { batch };
        let mut vv = vec![0u8; values.bytes];
        let mut ii = vec![0u8; ids.bytes];
        let mut policy = Vec::new();
        let mut fp = Vec::new();
        let mut expected = Vec::new();
        for row in 0..ROWS {
            let case = batch * ROWS + row;
            let n = [4, 8, 16, 32, 64, 112, 128, 224, 256][case % 9];
            let temp = [-1.0, 0.0, 1e-12, 0.01, 0.7, 1.0, 2.0, 100.0][(case / 9) % 8];
            let top = [
                0.0,
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                1.0,
                1.1,
                -1.0,
                f64::from_bits(0.5f64.to_bits() - 1),
            ][(case / 72) % 10];
            let k = [0, 1, 7, 8, 9, 20, 63, 64, 65, 127, 128][(case / 13) % 11];
            let k = if k == 0 && n > 128 { 128 } else { k };
            let s = Sampling::new(sampling::mix(case as u64), temp, k, top)?;
            ensure!(s.gpu_fits(n), "probe GPU shape");
            let s = Sampling::decode(&s.encode())?;
            let pos = [0, 1, 16384, 1048575, u64::MAX][case % 5];
            let mut input = Vec::new();
            for j in 0..n {
                let id = ((case * 257 + (n - 1 - j) * 601) % 154880) as u32;
                let bits = sampling::mix((case * n + j) as u64);
                let v = match case % 7 {
                    0 => {
                        if j % 2 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    1 => (j % 3) as f32,
                    2 => (bits as i32) as f32 * 1e-6,
                    3 => {
                        if j == 0 {
                            f32::MAX
                        } else {
                            -f32::MAX
                        }
                    }
                    _ => (bits as i16) as f32 / 2048.0,
                };
                let at = (row * 256 + j) * 4;
                vv[at..at + 4].copy_from_slice(&v.to_le_bytes());
                ii[at..at + 4].copy_from_slice(&id.to_le_bytes());
                input.push(Candidate { id, value: v });
            }
            let (tok, keep) = sampling::choose(&input, pos, s)?;
            expected.extend(tok.to_le_bytes());
            expected.extend((keep as u32).to_le_bytes());
            policy.extend(
                [n as u64, s.kept(n) as u64, s.seed, pos]
                    .into_iter()
                    .flat_map(u64::to_le_bytes),
            );
            fp.extend([s.temperature, s.top_p]);
        }
        values.write(0, &vv)?;
        ids.write(0, &ii)?;
        ints.write(0, &policy)?;
        params.write(0, &doubles(&fp))?;
        // Buffer::write fences the default stream before the captured owner stream.
        unsafe {
            graph.launch(&stream)?;
        }
        stream.sync()?;
        let got = out.read(0, out.bytes)?;
        if got != expected {
            let at = got
                .as_chunks::<8>()
                .0
                .iter()
                .zip(expected.as_chunks::<8>().0.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            anyhow::bail!(
                "draw mismatch batch {batch} row {at}: got {:?}, expected {:?}",
                &got[at * 8..at * 8 + 8],
                &expected[at * 8..at * 8 + 8]
            );
        }
        if batch == 0 {
            if let Some(first) = &first {
                ensure!(&got == first, "replay changed");
            } else {
                first = Some(got.clone());
            }
        }
        draw_hash.update(got);
    }
    // Wider host fallback: no GPU launch outside the proven bounded draw shape.
    let wide: Vec<_> = (0..154880).map(|id| Candidate { id, value: 0.0 }).collect();
    let s = Sampling::new(123, 1.0, 154880, 0.5)?;
    let wide_result = sampling::choose(&wide, 1048575, s)?;
    ensure!(wide_result.1 == 77441, "full-vocabulary top-p cut");
    ensure!(s.local_count() == 38720, "full local width");
    let seed_checks = [
        sampling::seed_for(&[], 0),
        sampling::seed_for(&[154820, 123, 42], -9),
    ];
    let random = sampling::random_seed()?;
    ensure!(random < 1 << 63, "random seed range");
    Ok(
        json!({"python_executed":false,"model_executed":false,"libm_sha256":digest(&libm),"math_input_count":x.len(),"math_comparisons":x.len()*9,"math_bits_sha256":format!("{:x}",math_hash.finalize()),"draw_rows":4096,"replay_rows":128,"draws_sha256":format!("{:x}",draw_hash.finalize()),"graph_replay_exact":true,"wide_host_candidates":154880,"wide_host_result":wide_result,"seed_checks":seed_checks,"production_ready":false}),
    )
}
