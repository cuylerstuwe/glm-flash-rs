//! Exact sixteen-row EXL3 verification, retaining serial split-K and epilogues.
use crate::kda_window::MAX_ROWS;
use crate::{
    cuda::Device,
    experts::{ExpertGpu, LoadPolicy},
    front::{launch_ptrs, load_kernel},
    grouping::Grouping,
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn probe(p: &Prepared, root: &Path, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let stream = d.stream()?;
    let m = p.load_selected(&d, &["weights.layers[3].moe.experts."])?;
    let path = native.join("exl3.cubin");
    let serial = ExpertGpu::with_policy(&d, &path, 1, LoadPolicy::ORIGINAL)?;
    let policies = [
        LoadPolicy::ORIGINAL,
        LoadPolicy {
            discard: true,
            ..LoadPolicy::ORIGINAL
        },
        LoadPolicy {
            optimized: true,
            ..LoadPolicy::ORIGINAL
        },
        LoadPolicy {
            optimized: true,
            programmatic: true,
            discard: false,
        },
        LoadPolicy {
            optimized: true,
            programmatic: true,
            discard: true,
        },
    ];
    let batches = policies
        .into_iter()
        .map(|policy| Ok((policy, ExpertGpu::with_policy(&d, &path, MAX_ROWS, policy)?)))
        .collect::<Result<Vec<_>>>()?;
    let grouped = Grouping::new(&d, native, MAX_ROWS)?;
    let group_one = load_kernel(&d, root, "group")?;
    let picks = d.alloc(MAX_ROWS * 9 * 4)?;
    let input = d.alloc(MAX_ROWS * 4096 * 2)?;
    let one_picks = d.alloc(9 * 4)?;
    let one_ids = d.alloc(9 * 4)?;
    let one_count = d.alloc(4)?;
    let one_members = d.alloc(9 * 4)?;
    let mut seed = 0x76252835u32;
    let data: Vec<u8> = (0..MAX_ROWS * 4096)
        .flat_map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            half::bf16::from_f32((seed % 8193) as f32 / 4096. - 1.)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    input.write(0, &data)?;
    let mut cases = Vec::new();
    for pattern in 0..3 {
        // Identical routed experts, partial overlap and disjoint groups. Every
        // row retains eight distinct expert IDs plus the shared slot.
        let ids: Vec<i32> = (0..MAX_ROWS)
            .flat_map(|r| {
                let r = r as i32;
                (0..9).map(move |s| {
                    if s == 8 {
                        288
                    } else {
                        (s * 31
                            + if pattern == 0 {
                                0
                            } else if pattern == 1 {
                                r * 31
                            } else {
                                r
                            })
                            % 288
                    }
                })
            })
            .collect();
        let pick_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        picks.write(0, &pick_bytes)?;
        let mut expected = Vec::new();
        for row in 0..MAX_ROWS {
            one_picks.write(0, &pick_bytes[row * 9 * 4..(row + 1) * 9 * 4])?;
            launch_ptrs(
                &group_one,
                &stream,
                [1, 1, 1],
                &[one_picks.ptr, one_ids.ptr, one_count.ptr, one_members.ptr],
            )?;
            serial.run(
                &stream,
                p,
                &m,
                3,
                input.ptr + (row * 4096 * 2) as u64,
                one_picks.ptr,
                one_ids.ptr,
                one_count.ptr,
                one_members.ptr,
            )?;
            stream.sync()?;
            expected.extend(serial.y.read(0, 9 * 4096 * 4)?);
        }
        for (policy, batch) in &batches {
            for rows in [1, 2, 3, 7, 8, 9, 15, 16] {
                grouped.run(&stream, picks.ptr, rows)?;
                batch.run_rows(
                    &stream,
                    p,
                    &m,
                    3,
                    input.ptr,
                    picks.ptr,
                    grouped.ids.ptr,
                    grouped.count.ptr,
                    grouped.members.ptr,
                    rows,
                    MAX_ROWS,
                )?;
                stream.sync()?;
                ensure!(
                    batch.y.read(0, rows * 9 * 4096 * 4)? == expected[..rows * 9 * 4096 * 4],
                    "EXL3 verifier differs pattern={pattern} rows={rows}"
                );
            }
            let graph = unsafe {
                stream.capture(|| {
                    grouped.run(&stream, picks.ptr, MAX_ROWS)?;
                    batch.run_rows(
                        &stream,
                        p,
                        &m,
                        3,
                        input.ptr,
                        picks.ptr,
                        grouped.ids.ptr,
                        grouped.count.ptr,
                        grouped.members.ptr,
                        MAX_ROWS,
                        MAX_ROWS,
                    )
                })?
            };
            for _ in 0..2 {
                unsafe {
                    graph.launch(&stream)?;
                }
                stream.sync()?;
                ensure!(
                    batch.y.read(0, MAX_ROWS * 9 * 4096 * 4)? == expected,
                    "EXL3 verifier graph differs"
                );
            }
            let mut start = d.event()?;
            let mut end = d.event()?;
            start.record(&stream)?;
            for _ in 0..100 {
                unsafe {
                    graph.launch(&stream)?;
                }
            }
            end.record(&stream)?;
            let graph_ms = end.elapsed_after(&start)? / 100.;
            ensure!(
                batch.y.read(0, MAX_ROWS * 9 * 4096 * 4)? == expected,
                "EXL3 replay/discard drift"
            );
            cases.push(
            serde_json::json!({"pattern":pattern,"policy":policy,"graph_ms":graph_ms,"rows":[1,2,3,7,8,9,15,16],"exact":true,"graph_replays":2,
            "output_sha256":format!("{:x}",Sha256::digest(&expected))}),
        );
        }
    }
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"cases":cases,
        "scope":"canonical split-K EXL3 routed expert windows, optimized loads/PDL/dead-scratch discard against original serial bytes; component graph timing, no full-model speed claim"}),
    )
}
