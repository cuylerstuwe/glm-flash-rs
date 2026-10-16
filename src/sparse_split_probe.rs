//! CUDA pool/expansion parity with poisoned packed padding; no model or transport claim.
use super::*;
use split_plan::{Plan, RECV_BYTES, ROW_BYTES, SEND_BYTES};
pub(crate) fn run(root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let select = Select::new(&d, root, 64, 262144)?;
    let scores = d.alloc(64 * 262144 * 4)?;
    let pos = d.alloc(4)?;
    let send = d.alloc(SEND_BYTES)?;
    let recv = d.alloc(RECV_BYTES)?;
    let output = d.alloc(1024 * WIDTH * 4)?;
    let counts = d.alloc(1024 * 4)?;
    let poison = vec![0xda; RECV_BYTES];
    let mut cases = Vec::new();
    for (rows, np) in [
        (256, 4096),
        (257, 4096),
        (319, 8192),
        (511, 8192),
        (512, 8192),
        (513, 8192),
        (1023, 8192),
        (1024, 8192),
        (257, 262144),
    ] {
        let position = np * 4 - rows;
        let plan = Plan::new(true, true, position, rows)?.context("split probe plan")?;
        let setup = |offset: usize, n: usize| -> Result<()> {
            let mut data = Vec::with_capacity(n * np * 4);
            for row in offset..offset + n {
                let visible = (position + row + 1) / 4;
                for pool in 0..np {
                    let value = if pool >= visible {
                        f32::NAN
                    } else {
                        match (pool + row) % 1031 {
                            0 => f32::INFINITY,
                            1 => f32::NEG_INFINITY,
                            2 => f32::from_bits(0xffc12345),
                            3 => -0.,
                            4 => 0.,
                            _ => (((pool * 37 + row * 19) % 65536) as i32 - 32768) as f32 / 256.,
                        }
                    };
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            scores.write(0, &data)?;
            pos.write(0, &(position as i32 + offset as i32).to_le_bytes())
        };
        let mut expected_pools = Vec::new();
        let mut expected_tokens = Vec::new();
        let mut expected_counts = Vec::new();
        for offset in (0..rows).step_by(64) {
            let n = (rows - offset).min(64);
            setup(offset, n)?;
            select.run(&s, &scores, &pos, n, np)?;
            s.sync()?;
            expected_pools.extend(select.pools.read(0, n * ROW_BYTES)?);
            expected_tokens.extend(select.tokens.read(0, n * WIDTH * 4)?);
            expected_counts.extend(select.counts.read(0, n * 4)?);
        }
        recv.write(0, &poison)?;
        for rank in 0..4 {
            send.write(0, &poison[..SEND_BYTES])?;
            let own = plan.own(rank)?;
            for offset in own.clone().step_by(64) {
                let n = (own.end - offset).min(64);
                setup(offset, n)?;
                select.pools_into(
                    &s,
                    &scores,
                    &pos,
                    n,
                    np,
                    &send,
                    (offset - own.start) * ROW_BYTES,
                )?;
                s.sync()?;
            }
            // Local component stand-in for the rank-ordered collective. Actual
            // RoCE/NCCL ordering is a separate all-rank model qualification gate.
            unsafe {
                recv.copy_from(&s, rank * plan.stride, &send, 0, plan.stride)?;
            }
            s.sync()?;
        }
        let packed = recv.read(0, RECV_BYTES)?;
        for rank in 0..4 {
            let own = plan.own(rank)?;
            let begin = rank * plan.stride;
            let live = own.len() * ROW_BYTES;
            ensure!(
                packed[begin..begin + live]
                    == expected_pools[own.start * ROW_BYTES..own.end * ROW_BYTES],
                "split selected pool IDs"
            );
            ensure!(
                packed[begin + live..begin + plan.stride]
                    .iter()
                    .all(|b| *b == 0xda),
                "split overwrote padding"
            );
        }
        let expand = || -> Result<()> {
            for offset in (0..rows).step_by(64) {
                let n = (rows - offset).min(64);
                let (at, _) = plan.region(offset, n)?;
                unsafe {
                    pos.set_i32(&s, (position + offset) as i32)?;
                }
                select.expand_from(&s, &recv, at, &pos, n)?;
                unsafe {
                    output.copy_from(&s, offset * WIDTH * 4, &select.tokens, 0, n * WIDTH * 4)?;
                    counts.copy_from(&s, offset * 4, &select.counts, 0, n * 4)?;
                }
            }
            Ok(())
        };
        let check = || -> Result<()> {
            ensure!(
                output.read(0, rows * WIDTH * 4)? == expected_tokens,
                "split expanded token IDs"
            );
            ensure!(
                counts.read(0, rows * 4)? == expected_counts,
                "split visible token counts"
            );
            Ok(())
        };
        expand()?;
        s.sync()?;
        check()?;
        let graph = unsafe { s.capture(expand)? };
        for _ in 0..2 {
            unsafe { graph.launch(&s)? };
            s.sync()?;
            check()?;
        }
        cases.push(serde_json::json!({"rows":rows,"pools":np,"position":position,"stride_bytes":plan.stride,"exact_selected_ids":true,"padding_preserved":true,"exact_expanded_ids":true,"graph_replays":2}));
    }
    Ok(
        serde_json::json!({"all":true,"cases":cases,"scope":"Supplied-score pool selection, packed ownership and expansion only; real scoring, model, transport and speed require separate qualification"}),
    )
}
