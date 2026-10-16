use crate::{
    cuda::Device,
    draft_q4::{DRAFT_HEAD, Projection},
    drafter_weights,
    dsa::words,
    front::{bf_values, f32_values, load_kernel},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;
pub fn probe(manifest: &Path, root: &Path, rank: u64) -> Result<serde_json::Value> {
    let p = drafter_weights::open(manifest, rank)?;
    let d = Device::new()?;
    let s = d.stream()?;
    // The draft head is a target tensor; this probe covers drafter weights only.
    let shapes = &crate::draft_q4::SHAPES[..DRAFT_HEAD];
    let stems: Vec<_> = shapes.iter().map(|q| q.stem).collect();
    let m = p.load_selected(&d, &stems)?;
    let mut cases = Vec::new();
    for &q in shapes {
        let projection = Projection::new(&d, root, q)?;
        let weights = [
            m.pointer(&p, &format!("{}.weight", q.stem))?,
            m.pointer(&p, &format!("{}.scales", q.stem))?,
            m.pointer(&p, &format!("{}.biases", q.stem))?,
        ];
        let stride = q.k + 16;
        let row_bytes = q.n * if q.f32 { 4 } else { 2 };
        let input = d.alloc(64 * stride * 2)?;
        let sums = d.alloc(64 * q.k / 64 * 4)?;
        let output = d.alloc(64 * row_bytes)?;
        let part = d.alloc(64 * q.n * q.sk * 4)?;
        let single = d.alloc(row_bytes)?;
        let single_part = d.alloc(q.n * q.sk * 4)?;
        // Binary fractions make every group sum exact independently of sum ordering.
        let mut x = vec![0f32; 64 * stride];
        let mut xs = vec![0f32; 64 * q.k / 64];
        for row in 0..64 {
            for g in 0..q.k / 64 {
                for j in 0..64 {
                    let v = ((row * 17 + g * 11 + j * 7) % 31) as f32 / 16. - 1.;
                    x[row * stride + g * 64 + j] = v;
                    xs[row * (q.k / 64) + g] += v;
                }
            }
        }
        input.write(
            0,
            &x.iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        sums.write(
            0,
            &xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        if q.k != 3072 {
            let kernel = load_kernel(&d, root, &format!("draft_sums_{}", q.k))?;
            words(
                &kernel,
                &s,
                [64, (q.k / 64).div_ceil(16) as u32, 1],
                &[input.ptr, sums.ptr, stride as u64],
            )?;
            s.sync()?;
            ensure!(
                f32_values(&sums.read(0, sums.bytes)?) == xs,
                "draft group sums"
            );
        }
        let mut expected = Vec::new();
        for row in 0..64 {
            unsafe {
                projection.run(
                    &s,
                    input.ptr + (row * stride * 2) as u64,
                    stride,
                    sums.ptr + (row * q.k / 64 * 4) as u64,
                    weights,
                    &single,
                    &single_part,
                    1,
                )?;
            }
            s.sync()?;
            expected.extend(single.read(0, row_bytes)?);
        }
        for rows in [1, 2, 7, 8, 16, 17, 63, 64] {
            unsafe {
                projection.run(
                    &s, input.ptr, stride, sums.ptr, weights, &output, &part, rows,
                )?;
            }
            s.sync()?;
            ensure!(
                output.read(0, rows * row_bytes)? == expected[..rows * row_bytes],
                "draft {} rows={rows} changed arithmetic",
                q.stem
            );
        }
        let graph = unsafe {
            s.capture(|| {
                projection.run(&s, input.ptr, stride, sums.ptr, weights, &output, &part, 8)
            })?
        };
        for _ in 0..2 {
            unsafe {
                graph.launch(&s)?;
            }
            s.sync()?;
            ensure!(
                output.read(0, 8 * row_bytes)? == expected[..8 * row_bytes],
                "draft projection graph mismatch"
            );
        }
        // Independent scalar tiled-weight reference for distributed columns, all K.
        // Match per-group and fixed split-K accumulation; bound only FP32/FMA rounding.
        let w = p.read_tensor(&format!("{}.weight", q.stem))?;
        let scale = bf_values(&p.read_tensor(&format!("{}.scales", q.stem))?);
        let bias = bf_values(&p.read_tensor(&format!("{}.biases", q.stem))?);
        let values = if q.f32 {
            f32_values(&expected)
        } else {
            bf_values(&expected)
        };
        let mut max_relative = 0f32;
        for row in [0, 7, 63] {
            for col in [0, 1, 63, 64, q.n / 2, q.n - 1] {
                let mut total = 0f32;
                for slice in 0..q.sk {
                    let mut acc = 0f32;
                    for g in slice * (q.k / 64 / q.sk)..(slice + 1) * (q.k / 64 / q.sk) {
                        let mut dot = 0f32;
                        for j in 0..64 {
                            let wi = ((col / 64 * (q.k / 64) + g) * 64 + col % 64) * 8 + j / 8;
                            let word =
                                u32::from_le_bytes(w[wi * 4..wi * 4 + 4].try_into().unwrap());
                            let value = ((word >> (4 * (j % 8))) & 15) as f32;
                            dot += x[row * stride + g * 64 + j] * value;
                        }
                        let index = g * q.n + col;
                        acc = dot.mul_add(scale[index], acc);
                        acc = xs[row * (q.k / 64) + g].mul_add(bias[index], acc);
                    }
                    total = if slice == 0 { acc } else { total + acc };
                }
                let reference = if q.f32 {
                    total
                } else {
                    half::bf16::from_f32(total).to_f32()
                };
                let actual = values[row * q.n + col];
                let error = (actual - reference).abs();
                max_relative = max_relative.max(error / (1. + reference.abs()));
                ensure!(
                    actual.is_finite() && error <= 0.0001 + 0.008 * reference.abs(),
                    "draft scalar oracle {} row={row} col={col} actual={actual} reference={reference}",
                    q.stem
                );
            }
        }
        cases.push(serde_json::json!({"stem":q.stem,"n":q.n,"k":q.k,"split_k":q.sk,"rows":[1,2,7,8,16,17,63,64],"exact_serial_rows":true,"graph_exact":true,"scalar_max_relative":max_relative,"sha256":format!("{:x}",Sha256::digest(expected))}));
    }
    Ok(
        serde_json::json!({"rank":rank,"cases":cases,"scope":"DFlash2 prepared Q4 projections only; no full drafter or speculative serving"}),
    )
}
