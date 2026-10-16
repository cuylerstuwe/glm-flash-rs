use crate::kda_window::MAX_ROWS;
use crate::{
    cuda::Device,
    front::{fmm, load_kernel, reduce},
    prepared::Prepared,
    window_gemm::{Projection, SHAPES},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn probe(p: &Prepared, root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let stream = d.stream()?;
    let stems: Vec<_> = SHAPES.iter().map(|s| s.stem).collect();
    let m = p.load_selected(&d, &stems)?;
    let mut cases = Vec::new();
    for shape in SHAPES {
        p.expect(
            &format!("{}.weight", shape.stem),
            "float8_e4m3fn",
            &[shape.n as u64, shape.k as u64],
        )?;
        p.expect(
            &format!("{}.scale", shape.stem),
            "bfloat16",
            &[shape.n as u64],
        )?;
        let w = m.pointer(p, &format!("{}.weight", shape.stem))?;
        let scale = m.pointer(p, &format!("{}.scale", shape.stem))?;
        let batch = Projection::new(&d, root, shape)?;
        let serial = load_kernel(&d, root, shape.alias)?;
        let reduction = if shape.sk > 1 {
            Some(load_kernel(
                &d,
                root,
                &format!(
                    "reduce{}_{}",
                    shape.sk,
                    if shape.f32 { "f32" } else { "bf16" }
                ),
            )?)
        } else {
            None
        };
        let row_bytes = shape.n * if shape.f32 { 4 } else { 2 };
        let stride = shape.k + 16; // Padding catches accidental contiguous-row assumptions.
        let input = d.alloc(MAX_ROWS * stride * 2)?;
        let output = d.alloc(MAX_ROWS * row_bytes)?;
        let part = d.alloc(MAX_ROWS * shape.n * shape.sk as usize * 4)?;
        let single = d.alloc(row_bytes)?;
        let one_part = d.alloc(shape.n * shape.sk as usize * 4)?;
        let mut seed = 0x23941924u32;
        let bytes: Vec<u8> = (0..MAX_ROWS * stride)
            .flat_map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                half::bf16::from_f32((seed % 8193) as f32 / 4096. - 1.)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        input.write(0, &bytes)?;
        let mut expected = Vec::new();
        for row in 0..MAX_ROWS {
            fmm(
                &serial,
                &stream,
                input.ptr + (row * stride * 2) as u64,
                w,
                scale,
                single.ptr,
                one_part.ptr,
                stride as i32,
                shape.n as i32,
                shape.sk,
            )?;
            if let Some(r) = &reduction {
                reduce(r, &stream, one_part.ptr, single.ptr, shape.n as i32)?;
            }
            stream.sync()?;
            expected.extend(single.read(0, row_bytes)?);
        }
        for rows in [1, 2, 3, 7, 8, 9, 15, 16] {
            unsafe {
                batch.run(&stream, &input, stride, w, scale, &output, &part, rows)?;
            }
            stream.sync()?;
            ensure!(
                output.read(0, rows * row_bytes)? == expected[..rows * row_bytes],
                "{} verifier differs at rows={rows}",
                shape.alias
            );
        }
        let graph = unsafe {
            stream.capture(|| {
                batch.run(&stream, &input, stride, w, scale, &output, &part, MAX_ROWS)
            })?
        };
        for _ in 0..2 {
            unsafe {
                graph.launch(&stream)?;
            }
            stream.sync()?;
            ensure!(
                output.read(0, MAX_ROWS * row_bytes)? == expected,
                "{} graph differs",
                shape.alias
            );
        }
        // Paired graph timings isolate sixteen serial projections versus one
        // sixteen-row projection. Verification readbacks are outside this interval.
        // These component samples are not a model throughput comparison.
        let serial_graph = unsafe {
            stream.capture(|| {
                for row in 0..MAX_ROWS {
                    fmm(
                        &serial,
                        &stream,
                        input.ptr + (row * stride * 2) as u64,
                        w,
                        scale,
                        output.ptr + (row * row_bytes) as u64,
                        one_part.ptr,
                        stride as i32,
                        shape.n as i32,
                        shape.sk,
                    )?;
                    if let Some(r) = &reduction {
                        reduce(
                            r,
                            &stream,
                            one_part.ptr,
                            output.ptr + (row * row_bytes) as u64,
                            shape.n as i32,
                        )?;
                    }
                }
                Ok(())
            })?
        };
        unsafe {
            serial_graph.launch(&stream)?;
        }
        stream.sync()?;
        ensure!(
            output.read(0, MAX_ROWS * row_bytes)? == expected,
            "serial benchmark graph differs"
        );
        let mut start = d.event()?;
        let mut end = d.event()?;
        ensure!(
            end.elapsed_after(&start).is_err(),
            "unrecorded timing accepted"
        );
        let mut timings = Vec::new();
        for round in 0..4 {
            let mut serial_us = 0.;
            let mut batch_us = 0.;
            for batched in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let selected = if batched { &graph } else { &serial_graph };
                unsafe {
                    selected.launch(&stream)?;
                }
                start.record(&stream)?;
                for _ in 0..100 {
                    unsafe {
                        selected.launch(&stream)?;
                    }
                }
                end.record(&stream)?;
                let us = end.elapsed_after(&start)? * 1000. / 100.;
                if batched {
                    batch_us = us;
                } else {
                    serial_us = us;
                }
            }
            timings.push(serde_json::json!({"round":round,"serial_sixteen_rows_us":serial_us,"batch_sixteen_rows_us":batch_us}));
        }
        cases.push(serde_json::json!({"alias":shape.alias,"n":shape.n,"k":shape.k,"split_k":shape.sk,
            "rows":[1,2,3,7,8,9,15,16],"exact":true,"graph_replays":2,"serial_sha256":format!("{:x}",Sha256::digest(&expected)),
            "timing_scope":"CUDA-event mean of 100 graph replays; warm paired alternating order; no host readback or target-model claim","timings":timings}));
    }
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"device":d.name,"cases":cases,
        "scope":"exact multirow FP8 projection component, no whole verifier or speed claim"}),
    )
}
