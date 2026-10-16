//! Local device/oracle controls, run independently on every GB10 before serving.
use crate::{
    candidates::{Candidates, LOCAL, payload_bytes},
    cuda::Device,
    masked_candidates::{MASK_WORDS, MaskedCandidates},
    sampling::{self, Candidate, Sampling},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

fn mask(case: usize) -> Vec<i32> {
    let mut words = vec![0i32; MASK_WORDS];
    for id in 0..sampling::VOCAB {
        let allowed = match case {
            0 => true,
            1 => id == 2 * LOCAL + 7,
            2 => id.is_multiple_of(997),
            3 => (LOCAL..LOCAL + 3).contains(&id),
            4 => id % LOCAL >= LOCAL - 65,
            5 => id % 64 == 31 || id % 64 == 32,
            6 => id % LOCAL < 257,
            7 => (LOCAL..LOCAL + 7).contains(&id),
            8 => (LOCAL..LOCAL + 8).contains(&id),
            9 => (LOCAL..LOCAL + 9).contains(&id),
            10 => (LOCAL..LOCAL + 19).contains(&id),
            _ => false,
        };
        if allowed {
            words[id / 32] |= 1i32 << (id % 32);
        }
    }
    words
}
fn logits(case: usize, rank: usize) -> Vec<u8> {
    (0..LOCAL)
        .flat_map(|i| {
            let b = match case {
                0 => half::bf16::from_f32(((i * 37 + rank * 111) % 1009) as f32 / 32.0 - 15.0)
                    .to_bits(),
                1 => {
                    if i % 3 == 0 {
                        0x8000
                    } else {
                        0
                    }
                }
                2 => 0xbf80,
                _ => {
                    if i % 17 == 0 {
                        0x7f7f
                    } else {
                        0xff7f
                    }
                }
            };
            b.to_le_bytes()
        })
        .collect()
}
fn reference(bytes: &[u8], words: &[i32], rank: usize, s: Sampling) -> Vec<Candidate> {
    let mut values: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            let id = rank * LOCAL + i;
            if (words[id / 32] as u32 >> (id % 32)) & 1 == 0 {
                return None;
            }
            let bits = u16::from_le_bytes(*b);
            let normalized = if s.greedy() && bits & 0x7fff == 0 {
                0
            } else {
                bits
            };
            let key = if normalized & 0x8000 != 0 {
                normalized ^ 0xffff
            } else {
                normalized ^ 0x8000
            };
            Some((
                key,
                Candidate {
                    id: id as u32,
                    value: f32::from_bits((bits as u32) << 16),
                },
            ))
        })
        .collect();
    values.sort_unstable_by_key(|(key, c)| (std::cmp::Reverse(*key), c.id));
    values.truncate(s.local_count());
    values.sort_unstable_by_key(|(_, c)| c.id);
    values.into_iter().map(|(_, c)| c).collect()
}
fn packed(v: &[Candidate], k: usize) -> Vec<u8> {
    let mut bytes = vec![0; payload_bytes(k).unwrap()];
    for (i, c) in v.iter().enumerate() {
        bytes[i * 8..i * 8 + 4].copy_from_slice(&c.value.to_le_bytes());
        bytes[i * 8 + 4..i * 8 + 8].copy_from_slice(&c.id.to_le_bytes());
    }
    bytes[k * 8 + 4..k * 8 + 8].copy_from_slice(&(v.len() as u32).to_le_bytes());
    bytes
}
pub(crate) fn run(native: &Path) -> Result<Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let input = device.alloc(LOCAL * 2)?;
    let mut selected = MaskedCandidates::new(&device, native)?;
    let baseline = Candidates::new(&device, native)?;
    let mut digest = Sha256::new();
    let (mut shards, mut draws, mut unmasked) = (0, 0, 0);
    let policies = [
        (0.0, 20, 0.95),
        (1.0, 20, 0.95),
        (0.01, 0, 0.5),
        (1.0, 56, 1.0),
        (2.0, 504, 0.95),
        (1.0, sampling::VOCAB, 0.0),
        (1.0, 20, f64::from_bits(1.0f64.to_bits() - 1)),
    ];
    for data in 0..4 {
        for case in 0..11 {
            let words = mask(case);
            for (temperature, top_k, top_p) in policies {
                let s = Sampling::new(450, temperature, top_k, top_p)?;
                let k = s.local_count();
                let mut expected = Vec::new();
                let mut gathered = Vec::new();
                unsafe {
                    selected.upload_mask(&stream, &words)?;
                }
                stream.sync()?;
                for rank in 0..4 {
                    let bytes = logits(data, rank);
                    input.write(0, &bytes)?;
                    let want = reference(&bytes, &words, rank, s);
                    // Capture uses stable owned pointers; replay observes the
                    // latest uploaded mask/logits without a host policy branch.
                    let graph =
                        unsafe { stream.capture(|| selected.local(&stream, &input, rank, s))? };
                    unsafe {
                        graph.launch(&stream)?;
                    }
                    stream.sync()?;
                    let got = selected.local.read(0, payload_bytes(k)?)?;
                    ensure!(
                        got == packed(&want, k),
                        "masked set/value mismatch data={data} mask={case} rank={rank} k={k}"
                    );
                    if case == 0 && !s.greedy() {
                        baseline.local(&stream, &input, rank, k)?;
                        stream.sync()?;
                        ensure!(
                            baseline.local.read(0, payload_bytes(k)?)? == got,
                            "all-allowed candidate drift"
                        );
                        unmasked += 1;
                    }
                    digest.update(&got);
                    gathered.extend(got);
                    expected.extend(want);
                    shards += 1;
                }
                selected.gathered.write(0, &gathered)?;
                for position in [0, 16384, 1048575] {
                    let want = sampling::choose_masked(&expected, position, s, 4 * k)?;
                    ensure!(
                        selected.read_host(s, position)? == want,
                        "masked host sample drift"
                    );
                    if s.gpu_fits(4 * k) {
                        unsafe {
                            selected.gpu(&stream, s, position)?;
                        }
                        stream.sync()?;
                        ensure!(selected.read_gpu(s)? == want, "masked GPU sample drift");
                    }
                    draws += 1;
                }
            }
        }
    }
    let s = Sampling::new(450, 1., 20, 0.95)?;
    let k = s.local_count();
    let graph = unsafe { stream.capture(|| selected.local(&stream, &input, 1, s))? };
    for (case, data) in [(0, 0), (3, 1), (1, 2), (4, 3), (0, 0)] {
        let words = mask(case);
        let bytes = logits(data, 1);
        input.write(0, &bytes)?;
        unsafe {
            selected.upload_mask(&stream, &words)?;
            graph.launch(&stream)?;
        }
        stream.sync()?;
        ensure!(
            selected.local.read(0, payload_bytes(k)?)?
                == packed(&reference(&bytes, &words, 1, s), k),
            "retained graph ignored changed mask or logits"
        );
    }
    let zero = vec![0; payload_bytes(k)? * 4];
    selected.gathered.write(0, &zero)?;
    unsafe {
        selected.gpu(&stream, s, 0)?;
    }
    stream.sync()?;
    ensure!(
        selected.read_gpu(s).is_err() && selected.read_host(s, 0).is_err(),
        "empty grammar accepted"
    );
    let words = mask(1);
    unsafe {
        selected.upload_mask(&stream, &words)?;
    }
    stream.sync()?;
    let mut bytes = logits(0, 0);
    for nonfinite in [0x7f80u16, 0xff80, 0x7fc1, 0xffc1] {
        bytes[..2].copy_from_slice(&nonfinite.to_le_bytes());
        input.write(0, &bytes)?;
        selected.local(&stream, &input, 0, s)?;
        stream.sync()?;
        let bad = selected.local.read(0, payload_bytes(k)?)?;
        ensure!(
            u32::from_le_bytes(bad[k * 8..k * 8 + 4].try_into()?) & 1 != 0,
            "masked nonfinite model logit accepted"
        );
    }
    let mut good = Vec::new();
    let words = mask(0);
    for rank in 0..4 {
        good.extend(packed(&reference(&logits(0, rank), &words, rank, s), k));
    }
    let mut faults = 0;
    for rank in 0..4 {
        for (offset, value) in [
            (0, f32::NAN.to_bits()),
            (0, 1),
            (4, 154880),
            (k * 8, 1),
            (k * 8 + 4, (k + 1) as u32),
        ] {
            let mut bad = good.clone();
            let at = rank * payload_bytes(k)? + offset;
            bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
            selected.gathered.write(0, &bad)?;
            unsafe {
                selected.gpu(&stream, s, 8)?;
            }
            stream.sync()?;
            ensure!(
                selected.read_gpu(s).is_err() && selected.read_host(s, 8).is_err(),
                "malformed masked payload accepted"
            );
            faults += 1;
        }
    }
    selected.gathered.write(0, &good)?;
    unsafe {
        selected.gpu(&stream, s, 8)?;
    }
    stream.sync()?;
    ensure!(
        selected.read_gpu(s)? == selected.read_host(s, 8)?,
        "masked clean reuse after faults"
    );
    Ok(
        json!({"shard_cases":shards,"sample_cases":draws,"all_allowed_matches":unmasked,
        "payload_faults":faults,"changed_mask_graph_replays":5,"empty_grammar_rejected":true,"masked_nonfinite_rejected":4,
        "digest":format!("{:x}",digest.finalize()),"scope":"Local GPU mask-before-topk, variable shard count and sampled draw controls; not collective/model/serving grammar acceptance."}),
    )
}
