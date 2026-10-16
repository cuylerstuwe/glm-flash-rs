//! Native TP4 byte/ordering probe; no weights or inference workload.
use super::*;
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, net::SocketAddr, path::Path};

const GUARD: usize = 256;
const POISON: u8 = 0xa5;
const ROUNDS: usize = 3;

fn word(rank: usize, round: usize, row: usize, column: usize) -> [u8; 2] {
    ((rank * 997 + round * 271 + row * 73 + column * 17) as u16).to_le_bytes()
}
fn input(rank: usize, round: usize, rows: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| (0..WIDTH).flat_map(move |column| word(rank, round, row, column)))
        .collect()
}
fn expected(rank: usize, round: usize, rows: usize) -> Result<Vec<u8>> {
    let own = Plan::new(rows, rank)?.own();
    let mut bytes = vec![POISON; WORLD * MAX_ROWS * WIDTH * 2 + GUARD * 2];
    for peer in 0..WORLD {
        let selected = if peer == rank {
            0..rows
        } else {
            own.first..own.first + own.count
        };
        for row in selected {
            for col in 0..WIDTH {
                let at = GUARD + ((peer * MAX_ROWS + row) * WIDTH + col) * 2;
                bytes[at..at + 2].copy_from_slice(&word(peer, round, row, col));
            }
        }
    }
    Ok(bytes)
}

// Two outstanding exchanges, a third reuse of slot zero, and consumers queued
// on the owner stream before the next producer. The same schedule is capturable.
unsafe fn partial_rounds(
    pipe: &mut Pipe,
    comm: &Nccl,
    owner: &Stream,
    sources: &[Buffer],
    outputs: &[Buffer],
    rows: usize,
) -> Result<()> {
    pipe.begin()?;
    let owned = Plan::new(rows, pipe.rank)?.own();
    let mut pending = VecDeque::new();
    for (round, source) in sources.iter().enumerate() {
        if round == 2 {
            let first = pending.pop_front().unwrap();
            let view = pipe.wait(owner, first)?;
            unsafe {
                outputs[0].copy_from(owner, GUARD, view.partial, 0, view.partial.bytes)?;
            }
        }
        let index = (pipe.turn % 2) as usize;
        unsafe {
            let partial = &pipe.slots[index].partial;
            partial.fill_prefix(owner, partial.bytes, POISON)?;
        }
        let (buffer, offset) = pipe.producer_region(rows)?;
        unsafe {
            buffer.copy_from(owner, offset, source, 0, rows * WIDTH * 2)?;
        }
        pending.push_back(unsafe { pipe.issue(comm, owner, rows)? });
        if round == 1 {
            ensure!(
                pipe.begin().is_err(),
                "outstanding exchanges reported drained"
            );
            ensure!(
                pipe.producer_region(rows).is_err(),
                "outstanding producer reused"
            );
        }
    }
    for (i, token) in pending.into_iter().enumerate() {
        let view = pipe.wait(owner, token)?;
        ensure!(view.rank_stride == MAX_ROWS * WIDTH, "partial rank stride");
        ensure!(view.owned == owned, "partial ownership");
        unsafe {
            outputs[i + 1].copy_from(owner, GUARD, view.partial, 0, view.partial.bytes)?;
        }
    }
    pipe.begin()?;
    Ok(())
}

fn assert_buffers(outputs: &[Buffer], expected: &[Vec<u8>]) -> Result<Vec<String>> {
    outputs
        .iter()
        .zip(expected)
        .map(|(buffer, reference)| {
            let actual = buffer.read(0, buffer.bytes)?;
            ensure!(
                &actual == reference,
                "rank-owned partial/guard/poison mismatch"
            );
            Ok(format!("{:x}", Sha256::digest(&actual)))
        })
        .collect()
}

pub fn run(rank: usize, library: &Path, master: SocketAddr) -> Result<serde_json::Value> {
    ensure!(rank < WORLD, "TP4 rank");
    let device = Device::new()?;
    let owner = device.stream()?;
    let comm = Nccl::new(library, rank, WORLD, master)?;
    let mut pipe = Pipe::new(&device, rank, MAX_ROWS)?;
    let mut cases = Vec::new();
    for rows in [1, 63, 64, 65, 127, 128, 129, 257, 511, 512, 513, 1023, 1024] {
        let sources = (0..ROUNDS)
            .map(|round| {
                let b = device.alloc(rows * WIDTH * 2)?;
                b.write(0, &input(rank, round, rows))?;
                Ok(b)
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = (0..ROUNDS)
            .map(|_| device.alloc(WORLD * MAX_ROWS * WIDTH * 2 + GUARD * 2))
            .collect::<Result<Vec<_>>>()?;
        let references = (0..ROUNDS)
            .map(|round| expected(rank, round, rows))
            .collect::<Result<Vec<_>>>()?;
        let poison = || -> Result<()> {
            for b in &outputs {
                unsafe {
                    b.fill_prefix(&owner, b.bytes, POISON)?;
                }
            }
            Ok(())
        };
        poison()?;
        unsafe {
            partial_rounds(&mut pipe, &comm, &owner, &sources, &outputs, rows)?;
        }
        owner.sync()?;
        let hashes = assert_buffers(&outputs, &references)?;
        let graph = unsafe {
            owner.capture(|| {
                poison()?;
                partial_rounds(&mut pipe, &comm, &owner, &sources, &outputs, rows)
            })?
        };
        for _ in 0..2 {
            unsafe {
                graph.launch(&owner)?;
            }
            owner.sync()?;
            ensure!(
                assert_buffers(&outputs, &references)? == hashes,
                "captured partial mismatch"
            );
        }
        cases.push(serde_json::json!({"rows":rows,"owned":Plan::new(rows,rank)?.own().count,"rounds":ROUNDS,"graph_replays":2,"guard_and_inactive_exact":true,"hashes":hashes}));
        drop(graph);
    }

    // Owned in-place fields have guard bytes and poisoned foreign rows. Exercise
    // single/paired fields and clipped tails without introducing a reduction.
    let mut shares = Vec::new();
    for (rows, first, count) in [
        (1, 0, 1),
        (65, 0, 65),
        (129, 63, 65),
        (257, 256, 1),
        (1024, 0, 1024),
    ] {
        for paired in [false, true] {
            let widths: &[usize] = if paired { &[WIDTH * 2, 256] } else { &[5120] };
            let mut fields = Vec::new();
            let mut originals = Vec::new();
            let mut references = Vec::new();
            let own = Plan::new(rows, rank)?.own();
            for &width in widths {
                let mut initial = vec![POISON; GUARD * 2 + count * width];
                let mut reference = initial.clone();
                for row in first..first + count {
                    let peer = (0..WORLD)
                        .find(|&p| {
                            let v = Plan::new(rows, p).unwrap().own();
                            (v.first..v.first + v.count).contains(&row)
                        })
                        .unwrap();
                    for col in 0..width {
                        let at = GUARD + (row - first) * width + col;
                        let byte = (peer * 43 + row * 29 + col * 11) as u8;
                        reference[at] = byte;
                        if (own.first..own.first + own.count).contains(&row) {
                            initial[at] = byte;
                        }
                    }
                }
                let b = device.alloc(initial.len())?;
                b.write(0, &initial)?;
                let original = device.alloc(initial.len())?;
                original.write(0, &initial)?;
                fields.push(b);
                originals.push(original);
                references.push(reference);
            }
            let descriptors = fields
                .iter()
                .zip(widths)
                .map(|(buffer, &row_bytes)| Field {
                    buffer,
                    base: GUARD,
                    row_bytes,
                })
                .collect::<Vec<_>>();
            unsafe {
                pipe.share_clipped(&comm, &owner, rows, first, count, &descriptors)?;
            }
            ensure!(pipe.begin().is_err(), "pending shares reported drained");
            pipe.wait_shares(&owner)?;
            owner.sync()?;
            let hashes = assert_buffers(&fields, &references)?;
            pipe.begin()?;
            let graph = unsafe {
                owner.capture(|| {
                    for (field, original) in fields.iter().zip(&originals) {
                        field.copy_from(&owner, 0, original, 0, original.bytes)?;
                    }
                    pipe.share_clipped(&comm, &owner, rows, first, count, &descriptors)?;
                    pipe.wait_shares(&owner)
                })?
            };
            for _ in 0..2 {
                unsafe {
                    graph.launch(&owner)?;
                }
                owner.sync()?;
                ensure!(
                    assert_buffers(&fields, &references)? == hashes,
                    "captured share mismatch"
                );
            }
            drop(graph);
            shares.push(serde_json::json!({"rows":rows,"first":first,"count":count,"fields":widths.len(),"graph_replays":2,"exact":true,"hashes":hashes}));
        }
    }
    // All ranks reject these locally before opening any NCCL group. A following
    // common exchange demonstrates the communicator remains usable.
    let tiny = device.alloc(64)?;
    for (peer, offset) in [
        (WORLD, 0),
        ((rank + 1) % WORLD, 65),
        ((rank + 1) % WORLD, 0),
    ] {
        let bad = [Exchange {
            peer,
            send: Region {
                buffer: &tiny,
                offset,
                bytes: 4,
            },
            receive: Region {
                buffer: &tiny,
                offset: 0,
                bytes: 4,
            },
        }];
        ensure!(
            unsafe { comm.exchange_regions(&bad, &owner) }.is_err(),
            "invalid descriptor accepted"
        );
    }
    let all = comm.exchange_host(&device, &owner, b"split-comm-finished")?;
    ensure!(
        all.iter().all(|x| x == b"split-comm-finished"),
        "final rank agreement"
    );
    Ok(
        serde_json::json!({"rank":rank,"partial_cases":cases,"share_cases":shares,"invalid_descriptors_rejected":3,"passed":true,"scope":"Byte/order/guard/component proof only; no model, cache, native-hold or performance acceptance."}),
    )
}
