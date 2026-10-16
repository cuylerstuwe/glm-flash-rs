//! Exact saved native-ATen sets, then TP4 transport and draw. No model execution.
use crate::{
    candidates::{self, Candidates, LOCAL, payload_bytes},
    comm::Nccl,
    cuda::Device,
    roce::Roce,
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
    net::SocketAddr,
    path::Path,
};

pub(crate) struct Fixture {
    pub(crate) k: usize,
    pub(crate) logits: Vec<u8>,
    pub(crate) ids: Vec<u32>,
}
pub(crate) fn fixtures(path: &Path) -> Result<Vec<Fixture>> {
    ensure!(std::fs::metadata(path)?.len() == 7_000_132, "oracle length");
    let bytes = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "47cbcdc8d093c5d637f2b3fc3932b75061e1dd424553d80283ee48450d62ce37",
        "oracle provenance drift"
    );
    let mut input = Cursor::new(bytes);
    let mut magic = [0; 8];
    input.read_exact(&mut magic)?;
    ensure!(&magic == b"ALTOPK01", "oracle header");
    fn word(input: &mut Cursor<Vec<u8>>) -> Result<u32> {
        let mut b = [0; 4];
        input.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    ensure!(word(&mut input)? == 70, "oracle count");
    let mut out = Vec::new();
    for case in 0..10 {
        for k in [1, 8, 28, 64, 136, 512, LOCAL] {
            ensure!(
                word(&mut input)? == case && word(&mut input)? as usize == k,
                "oracle order"
            );
            let mut logits = vec![0; LOCAL * 2];
            input.read_exact(&mut logits)?;
            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                ids.push(word(&mut input)?);
            }
            ensure!(
                ids.iter().all(|&id| (id as usize) < LOCAL) && ids.windows(2).all(|w| w[0] < w[1]),
                "oracle IDs"
            );
            out.push(Fixture { k, logits, ids });
        }
    }
    ensure!(
        input.position() == input.get_ref().len() as u64,
        "oracle trailing data"
    );
    Ok(out)
}
pub(crate) fn expected(f: &Fixture, rank: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((f.k + 1) * 8);
    for &id in &f.ids {
        let i = id as usize * 2;
        out.extend(
            ((u16::from_le_bytes(f.logits[i..i + 2].try_into().unwrap()) as u32) << 16)
                .to_le_bytes(),
        );
        out.extend((id + (rank * LOCAL) as u32).to_le_bytes());
    }
    out.extend(0u32.to_le_bytes());
    out.extend((f.k as u32).to_le_bytes());
    out
}
fn policies(k: usize) -> Result<Vec<Sampling>> {
    let top = if k == LOCAL { sampling::VOCAB } else { k - 8 };
    [
        (0, 1.0, 0.95),
        (u64::MAX, 0.01, 0.0),
        (450, 2.0, 0.5),
        (29, 1e-12, 1.0),
    ]
    .into_iter()
    .map(|(seed, t, p)| Sampling::new(seed, t, top, p))
    .collect()
}
fn reference(rows: &[u8], k: usize, position: u64, s: Sampling) -> Result<(u32, usize)> {
    let input: Vec<Candidate> = candidates::decode(rows, k)?;
    sampling::choose(&input, position, s)
}
pub(crate) fn probe(native: &Path, oracle: &Path) -> Result<serde_json::Value> {
    let f = fixtures(oracle)?;
    let device = Device::new()?;
    let stream = device.stream()?;
    let logits = device.alloc(LOCAL * 2)?;
    let mut select = Candidates::new(&device, native)?;
    let mut digest = Sha256::new();
    let mut graphs = BTreeMap::new();
    for rank in 0..4 {
        for (i, fixture) in f.iter().enumerate() {
            logits.write(0, &fixture.logits)?;
            let graph = if let std::collections::btree_map::Entry::Vacant(e) =
                graphs.entry((rank, fixture.k))
            {
                e.insert(unsafe {
                    stream.capture(|| select.local(&stream, &logits, rank, fixture.k))?
                })
            } else {
                graphs.get_mut(&(rank, fixture.k)).unwrap()
            };
            unsafe {
                graph.launch(&stream)?;
            }
            stream.sync()?;
            let actual = select.local.read(0, payload_bytes(fixture.k)?)?;
            ensure!(
                actual == expected(fixture, rank),
                "ATen set/value mismatch rank {rank} fixture {i}"
            );
            digest.update(actual);
        }
    }
    // Every nonfinite class is rejected, including one that is below the cutoff.
    for (i, bad) in [0x7f80u16, 0xff80, 0x7fc1, 0xffc1].into_iter().enumerate() {
        let mut bytes = f[2].logits.clone();
        bytes[i * 2..i * 2 + 2].copy_from_slice(&bad.to_le_bytes());
        logits.write(0, &bytes)?;
        select.local(&stream, &logits, 0, 28)?;
        stream.sync()?;
        let got = select.local.read(0, payload_bytes(28)?)?;
        ensure!(
            u32::from_le_bytes(got[224..228].try_into()?) & 1 != 0,
            "nonfinite accepted"
        );
    }
    let mut base = Vec::new();
    for rank in 0..4 {
        base.extend(expected(&f[2 + rank * 7], rank));
    }
    let s = Sampling::new(450, 1.0, 20, 0.95)?;
    let want = reference(&base, 28, 16384, s)?;
    let mut faults = 0;
    for rank in 0..4 {
        for (offset, value) in [
            (0, f32::NAN.to_bits()),
            (0, 1),
            (4, 154880),
            (12, (rank * LOCAL) as u32),
            (224, 1),
            (228, 29),
        ] {
            let mut bad = base.clone();
            let at = rank * payload_bytes(28)? + offset;
            bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
            select.gathered.write(0, &bad)?;
            unsafe {
                select.gpu(&stream, s, 16384)?;
            }
            stream.sync()?;
            ensure!(
                select.read_gpu(s).is_err() && select.read_host(s, 16384).is_err(),
                "malformed gather accepted"
            );
            // A fresh good draw must overwrite both the failure status and token.
            select.gathered.write(0, &base)?;
            unsafe {
                select.gpu(&stream, s, 16384)?;
            }
            stream.sync()?;
            ensure!(
                select.read_gpu(s)? == want,
                "draw did not recover after clobber"
            );
            faults += 1;
        }
    }
    Ok(
        json!({"oracle_fixtures":70,"rank_offsets":4,"exact_set_value_checks":280,"reused_graph_shapes":graphs.len(),"selected_bytes_sha256":format!("{:x}",digest.finalize()),"nonfinite_classes_rejected":4,"payload_faults_rejected_and_recovered":faults,"python_executed":false,"model_executed":false}),
    )
}
pub(crate) fn collective(
    rank: usize,
    native: &Path,
    library: &Path,
    master: SocketAddr,
    oracle: &Path,
    use_roce: bool,
) -> Result<serde_json::Value> {
    let f = fixtures(oracle)?;
    let device = Device::new()?;
    let stream = device.stream()?;
    let nccl = Nccl::new(library, rank, 4, master)?;
    let roce = if use_roce {
        Some(Roce::new(
            &device,
            &stream,
            &nccl,
            &native.join("libalaska_roce.so"),
            rank,
        )?)
    } else {
        None
    };
    let logits = device.alloc(LOCAL * 2)?;
    let mut select = Candidates::new(&device, native)?;
    let sync = || -> Result<()> {
        stream.sync()?;
        if let Some(r) = &roce {
            r.check()?;
        }
        Ok(())
    };
    let mut graphs = BTreeMap::new();
    let mut records = Vec::new();
    let mut digest = Sha256::new();
    let mut gpu_draws = 0;
    let mut host_draws = 0;
    // Replay the first mixed set after every other input, including a full-vocab gather.
    for case in (0..10).chain(std::iter::once(0)) {
        for ki in 1..7 {
            let fixture = &f[((case + rank) % 10) * 7 + ki];
            let k = fixture.k;
            logits.write(0, &fixture.logits)?;
            if let std::collections::btree_map::Entry::Vacant(e) = graphs.entry(k) {
                e.insert(unsafe {
                    stream.capture(|| {
                        select.local(&stream, &logits, rank, k)?;
                        if let Some(r) = &roce {
                            r.gather_prefix(
                                &select.local,
                                &select.gathered,
                                payload_bytes(k)?,
                                &stream,
                            )
                        } else {
                            nccl.all_gather_prefix(
                                &select.local,
                                &select.gathered,
                                payload_bytes(k)?,
                                &stream,
                            )
                        }
                    })?
                });
            }
            unsafe {
                graphs[&k].launch(&stream)?;
            }
            sync()?;
            let mut expect = Vec::new();
            for peer in 0..4 {
                expect.extend(expected(&f[((case + peer) % 10) * 7 + ki], peer));
            }
            let actual = select.gathered.read(0, payload_bytes(k)? * 4)?;
            ensure!(actual == expect, "gather differs case {case} k {k}");
            digest.update(&actual);
            let mut draws = Vec::new();
            for (j, s) in policies(k)?.into_iter().enumerate() {
                let position = [0, 16384, 1048575, u64::MAX][j];
                let want = reference(&expect, k, position, s)?;
                ensure!(select.read_host(s, position)? == want, "wide host mismatch");
                if s.gpu_fits(4 * k) {
                    unsafe {
                        select.gpu(&stream, s, position)?;
                    }
                    sync()?;
                    ensure!(
                        select.read_gpu(s)? == want,
                        "GPU draw mismatch case {case} k {k} policy {j}"
                    );
                    gpu_draws += 1;
                } else {
                    host_draws += 1;
                }
                draws.push(want);
            }
            records.push(json!({"case":case,"local_k":k,"draws":draws}));
        }
    }
    // One bad shard must poison the sample on all four ranks, followed by clean reuse.
    let k = 28;
    let s = policies(k)?[0];
    let fixture = &f[rank * 7 + 2];
    let mut bad = fixture.logits.clone();
    if rank == 2 {
        bad[..2].copy_from_slice(&0xff80u16.to_le_bytes());
    }
    logits.write(0, &bad)?;
    unsafe {
        graphs[&k].launch(&stream)?;
        select.gpu(&stream, s, 16384)?;
    }
    sync()?;
    ensure!(
        select.read_gpu(s).is_err() && select.read_host(s, 16384).is_err(),
        "bad shard consumed"
    );
    logits.write(0, &fixture.logits)?;
    unsafe {
        graphs[&k].launch(&stream)?;
        select.gpu(&stream, s, 16384)?;
    }
    sync()?;
    let recovery = select.read_gpu(s)?;
    ensure!(
        recovery == select.read_host(s, 16384)?,
        "fault recovery mismatch"
    );
    Ok(
        json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"gathers":66,"payload_bytes_per_rank_min":72,"payload_bytes_per_rank_max":payload_bytes(LOCAL)?,"reused_graph_shapes":graphs.len(),"gathered_bytes_sha256":format!("{:x}",digest.finalize()),"gpu_draws":gpu_draws,"wide_host_draws":host_draws,"records":records,"one_rank_failure_rejected":true,"recovery":recovery,"python_executed":false,"model_executed":false}),
    )
}
