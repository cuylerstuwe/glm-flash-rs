//! Rejection-point coverage against serial chain calls and convolution-tail shifts.
use crate::{
    cuda::{Buffer, Device},
    front::load_kernel,
    kda_window::*,
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn probe(p: &Prepared, root: &Path, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let stream = d.stream()?;
    let m = p.load_selected(
        &d,
        &[
            "weights.layers[0].kda.conv",
            "weights.layers[0].kda.a_log",
            "weights.layers[0].kda.dt_bias",
            "weights.layers[0].kda.norm",
        ],
    )?;
    let wp = |s: &str| m.pointer(p, &format!("weights.layers[0].kda.{s}"));
    let weights = Weights {
        conv: wp("conv")?,
        a_log: wp("a_log")?,
        dt_bias: wp("dt_bias")?,
        norm: wp("norm")?,
    };
    let native = native.join("kda.cubin");
    let mut batch = Scratch::new(&d, &native)?;
    let mut serial = Scratch::new(&d, &native)?;
    let conv_shift = load_kernel(&d, root, "conv_shift")?;
    let mut seed = 0x76291423u32;
    let mut floats = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed % 4097) as f32 / 4096. - 0.5
            })
            .collect()
    };
    let bfbytes = |xs: Vec<f32>| -> Vec<u8> {
        xs.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect()
    };
    let upload = |bytes: &[u8]| -> Result<Buffer> {
        let b = d.alloc(bytes.len())?;
        b.write(0, bytes)?;
        Ok(b)
    };
    let initial_state: Vec<_> = floats(STATE_BYTES / 4)
        .iter()
        .flat_map(|x| (x * 0.03125).to_le_bytes())
        .collect();
    let initial_conv = bfbytes(floats(CONV_BYTES / 2));
    let state = upload(&initial_state)?;
    let conv = upload(&initial_conv)?;
    let projections = bfbytes(floats(MAX_ROWS * PROJ_STRIDE));
    let gate_a = bfbytes(floats(MAX_ROWS * CHANNELS));
    let gate_g = bfbytes(floats(MAX_ROWS * CHANNELS));
    let projection = upload(&projections)?;
    let ga = upload(&gate_a)?;
    let gg = upload(&gate_g)?;
    let inputs = || Inputs {
        state: &state,
        conv: &conv,
        projection: &projection,
        gate_a: &ga,
        gate_g: &gg,
    };
    let ss = upload(&initial_state)?;
    let sc = upload(&initial_conv)?;
    let sp = d.alloc(PROJ_STRIDE * 2)?;
    let sa = d.alloc(CHANNELS * 2)?;
    let sg = d.alloc(CHANNELS * 2)?;
    let mut expected_states = vec![initial_state.clone()];
    let mut expected_outputs = Vec::new();
    let mut expected_convs = vec![initial_conv.clone()];
    for row in 0..MAX_ROWS {
        sp.write(
            0,
            &projections[row * PROJ_STRIDE * 2..(row + 1) * PROJ_STRIDE * 2],
        )?;
        sa.write(0, &gate_a[row * CHANNELS * 2..(row + 1) * CHANNELS * 2])?;
        sg.write(0, &gate_g[row * CHANNELS * 2..(row + 1) * CHANNELS * 2])?;
        unsafe {
            // Use the serial chain result directly; do not call the batched commit
            // implementation as its own reference.
            let _proposal = serial.evaluate(
                &stream,
                Inputs {
                    state: &ss,
                    conv: &sc,
                    projection: &sp,
                    gate_a: &sa,
                    gate_g: &sg,
                },
                weights,
                1,
            )?;
        }
        unsafe {
            ss.copy_from(&stream, 0, &serial.proposal_state, 0, STATE_BYTES)?;
        }
        crate::dsa::words(
            &conv_shift,
            &stream,
            [1, 6, 1],
            &[
                sc.ptr,
                sp.ptr,
                3 * CONV_CHANNELS as u64,
                PROJ_STRIDE as u64,
                PROJ_STRIDE as u64,
            ],
        )?;
        stream.sync()?;
        expected_states.push(ss.read(0, STATE_BYTES)?);
        expected_outputs.extend(serial.output.read(0, CHANNELS * 2)?);
        let mut cpu_tail = expected_convs.last().unwrap()[CONV_CHANNELS * 2..].to_vec();
        cpu_tail.extend_from_slice(
            &projections[row * PROJ_STRIDE * 2..row * PROJ_STRIDE * 2 + CONV_CHANNELS * 2],
        );
        ensure!(
            sc.read(0, CONV_BYTES)? == cpu_tail,
            "serial convolution tail differs from independent byte reference"
        );
        expected_convs.push(cpu_tail);
    }
    let mut cases = Vec::new();
    for rows in 1..=MAX_ROWS {
        for keep in 0..=rows {
            unsafe {
                let committed = batch
                    .evaluate(&stream, inputs(), weights, rows)?
                    .commit(&stream, keep)?;
                stream.sync()?;
                ensure!(committed.rows == keep, "committed row count");
                ensure!(
                    committed.state.read(0, STATE_BYTES)? == expected_states[keep],
                    "KDA state rows={rows} keep={keep}"
                );
                ensure!(
                    committed.conv.read(0, CONV_BYTES)? == expected_convs[keep],
                    "KDA convolution rows={rows} keep={keep}"
                );
            }
            ensure!(
                batch.output.read(0, rows * CHANNELS * 2)?
                    == expected_outputs[..rows * CHANNELS * 2],
                "KDA window output differs from serial rows={rows}"
            );
            ensure!(
                batch.proposal_state.read(0, STATE_BYTES)? == expected_states[rows],
                "KDA full proposal state"
            );
            cases.push(serde_json::json!({"rows":rows,"keep":keep,"exact":true}));
        }
    }
    for keep in [0, 1, 4, MAX_ROWS] {
        let graph = unsafe {
            stream.capture(|| {
                batch
                    .evaluate(&stream, inputs(), weights, MAX_ROWS)?
                    .commit(&stream, keep)?;
                Ok(())
            })?
        };
        for _ in 0..2 {
            unsafe {
                graph.launch(&stream)?;
            }
            stream.sync()?;
            ensure!(
                batch.committed_state.read(0, STATE_BYTES)? == expected_states[keep],
                "graph recurrent rollback"
            );
            ensure!(
                batch.committed_conv.read(0, CONV_BYTES)? == expected_convs[keep],
                "graph convolution rollback"
            );
        }
    }
    unsafe {
        ensure!(
            batch.evaluate(&stream, inputs(), weights, 0).is_err(),
            "zero proposal accepted"
        );
        ensure!(
            batch
                .evaluate(&stream, inputs(), weights, MAX_ROWS + 1)
                .is_err(),
            "oversized proposal accepted"
        );
        let old = batch.committed_state.read(0, STATE_BYTES)?;
        ensure!(
            batch
                .evaluate(&stream, inputs(), weights, 2)?
                .commit(&stream, 3)
                .is_err(),
            "excess acceptance allowed"
        );
        stream.sync()?;
        ensure!(
            batch.committed_state.read(0, STATE_BYTES)? == old,
            "invalid commit changed prior result"
        );
        // Dropping an uncommitted proposal leaves the caller's live state intact.
        let _discarded = batch.evaluate(&stream, inputs(), weights, MAX_ROWS)?;
    }
    stream.sync()?;
    ensure!(
        state.read(0, STATE_BYTES)? == initial_state && conv.read(0, CONV_BYTES)? == initial_conv,
        "proposal or commit modified caller's live state"
    );
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"device":d.name,"cases":cases,
        "graph_keeps":[0,1,4,MAX_ROWS],"graph_replays":2,"caller_state_immutable":true,
        "invalid_commit_preserves_result":true,"serial_output_sha256":format!("{:x}",Sha256::digest(&expected_outputs)),
        "serial_state_sha256":expected_states.iter().map(|v|format!("{:x}",Sha256::digest(v))).collect::<Vec<_>>(),
        "scope":"bounded KDA verifier recurrence/convolution component; no target forward, drafter, serving or speed claim"}),
    )
}
