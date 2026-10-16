#[path = "../../src/cache_pages.rs"]
#[allow(dead_code)]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
#[allow(dead_code)]
mod chat_json;
#[path = "../../src/grammar_bound.rs"]
mod grammar_bound;
#[path = "../../src/grammar_native.rs"]
#[allow(dead_code)]
mod grammar_native;
#[path = "../../src/grammar_prepare.rs"]
mod grammar_prepare;
#[path = "../../src/grammar_request.rs"]
#[allow(dead_code)]
mod grammar_request;
#[path = "../../src/grammar_runtime.rs"]
#[allow(dead_code)]
mod grammar_runtime;
#[path = "../../src/grammar_serial.rs"]
#[allow(dead_code)]
mod grammar_serial;
#[path = "../../src/grammar_vocab.rs"]
mod grammar_vocab;
#[path = "../../src/output_decode.rs"]
#[allow(dead_code)]
mod output_decode;
#[path = "../../src/sampling.rs"]
#[allow(dead_code)]
mod sampling;
#[path = "../../src/tokenizer.rs"]
#[allow(dead_code)]
mod tokenizer;
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc, time::Duration};
fn main() -> Result<()> {
    let path = std::env::var("ALASKA_TOKENIZER_JSON")?;
    let bytes = std::fs::read(&path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer identity"
    );
    let tok = tokenizer::load(Path::new(&path))?;
    let views = Arc::new(grammar_vocab::Views::new(
        &tok,
        &serde_json::from_slice(&bytes)?,
    )?);
    let library = std::env::var("ALASKA_XGRAMMAR_LIBRARY")?;
    let sha = std::env::var("ALASKA_XGRAMMAR_SHA256")?;
    let native = grammar_native::Native::open(Path::new(&library), &sha)?;
    let mut prepare = grammar_prepare::Preparer::new(native, views)?;
    let invalid =
        grammar_bound::Bound::from_request(&json!({"guided_grammar":"root ::= ["}), &[])?.unwrap();
    let mut job = prepare
        .start(invalid)
        .map_err(|_| anyhow::anyhow!("prepare queue rejected"))?;
    ensure!(
        job.poll(Duration::from_secs(30))?
            .ok_or_else(|| anyhow::anyhow!("prepare timeout"))?
            .is_err(),
        "invalid grammar compiled"
    );
    let bound=grammar_bound::Bound::from_request(&json!({"guided_json":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}),&[])?.unwrap();
    let bound = grammar_bound::Bound::decode(&bound.encode())?;
    let mut cancellations = Vec::new();
    for _ in 0..4 {
        let job = prepare
            .start(bound.clone())
            .map_err(|_| anyhow::anyhow!("cancel fixture capacity"))?;
        job.cancel();
        cancellations.push(job);
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    for mut job in cancellations {
        while !job.retired()? {
            ensure!(
                std::time::Instant::now() < deadline,
                "cancelled compiler did not retire"
            );
            std::thread::yield_now();
        }
    }
    ensure!(
        prepare.active_owners() == 0
            && prepare.active_bytes() == 0
            && prepare.cached_bytes() <= 256 << 20,
        "cancelled compiler ownership leaked"
    );
    let mut jobs = Vec::new();
    for _ in 0..4 {
        jobs.push(
            prepare
                .start(bound.clone())
                .map_err(|_| anyhow::anyhow!("prepare queue capacity"))?,
        );
    }
    let mut owners = Vec::new();
    for (i, mut job) in jobs.into_iter().enumerate() {
        let ready = job
            .poll(Duration::from_secs(30))?
            .ok_or_else(|| anyhow::anyhow!("prepare timeout"))??;
        ensure!(
            ready.bound_digest() == bound.digest() && ready.bytes() > 0,
            "prepare receipt identity"
        );
        owners.push(ready.bind(i as u64 + 1, 131072)?);
    }
    ensure!(
        prepare.active_owners() == 4 && prepare.active_bytes() > 0,
        "native handle accounting absent"
    );
    let mut masks = grammar_runtime::Pool::<grammar_prepare::LeasedMatcher>::new(4)?;
    let output = tok
        .encode(r#"{"x":42}"#, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut expected_digest = Vec::new();
    for (position, &token) in output.get_ids().iter().enumerate() {
        let mut pending = Vec::new();
        for mut owner in owners {
            let before = owner.digest();
            let win = owner.cut(&[0])?;
            pending.push((
                masks
                    .start(owner, win)
                    .map_err(|_| anyhow::anyhow!("mask queue rejected"))?,
                before,
            ));
        }
        owners = Vec::new();
        let mut reference = None;
        for (mut job, before) in pending {
            let done = job
                .poll(Duration::from_secs(30))?
                .ok_or_else(|| anyhow::anyhow!("mask timeout"))?;
            let bits = done.masks?;
            let row = bits.row(0).unwrap();
            ensure!(
                row[token as usize / 32] as u32 & (1 << (token % 32)) != 0,
                "canonical grammar denied encoded token"
            );
            if let Some(reference) = &reference {
                ensure!(reference == row, "interleaved matcher mask differs");
            } else {
                reference = Some(row.to_vec());
            }
            let mut owner = done.constraint;
            ensure!(owner.digest() == before, "mask worker changed progress");
            owner.advance(&[token as i32])?;
            ensure!(owner.emitted() == position + 1, "accepted cursor differs");
            owners.push(owner);
        }
    }
    for mut owner in owners {
        owner.advance(&[output_decode::EOS[0] as i32])?;
        ensure!(owner.finished()?, "EOS not terminal");
        expected_digest.push(owner.digest());
    }
    ensure!(
        expected_digest.windows(2).all(|pair| pair[0] != pair[1]),
        "distinct request owner identity lost"
    );
    let mut serial = grammar_serial::Serial::new(4)?;
    for id in 101..=104 {
        let mut job = prepare
            .start(bound.clone())
            .map_err(|_| anyhow::anyhow!("prepare queue capacity"))?;
        let ready = job
            .poll(Duration::from_secs(30))?
            .ok_or_else(|| anyhow::anyhow!("prepare timeout"))??;
        serial.insert(id, ready.bind(id, 131072)?)?;
    }
    for (position, token) in output
        .get_ids()
        .iter()
        .copied()
        .chain([output_decode::EOS[0]])
        .enumerate()
    {
        let progress = serial.digest();
        for id in 101..=104 {
            ensure!(serial.begin(id)?, "serial mask queue unexpectedly full");
        }
        ensure!(
            serial.digest() == progress,
            "worker start changed logical progress"
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        for id in 101..=104 {
            loop {
                match serial.poll(id)?.stage {
                    grammar_serial::Stage::Pending => {
                        ensure!(
                            std::time::Instant::now() < deadline,
                            "serial native mask timeout"
                        );
                        std::thread::yield_now();
                    }
                    grammar_serial::Stage::Ready => break,
                    _ => anyhow::bail!("serial native mask failed"),
                }
            }
            let row = serial
                .mask(id)?
                .ok_or_else(|| anyhow::anyhow!("missing serial native mask"))?;
            ensure!(
                row[token as usize / 32] as u32 & (1 << (token % 32)) != 0,
                "serial native mask denied canonical token"
            );
            serial.commit(id, token as i32, false)?;
            ensure!(
                serial.status(id)?.emitted == position + 1,
                "serial native cursor changed"
            );
        }
    }
    ensure!(serial.count() == 4, "serial native owner count changed");
    for id in 101..=104 {
        ensure!(
            serial.status(id)?.stage == grammar_serial::Stage::Terminal,
            "serial native EOS not terminal"
        );
        ensure!(serial.remove(id), "serial native owner missing");
    }
    serial.close();
    masks.close();
    prepare.close();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !masks.retired()? || !serial.retired()? || !prepare.retired()? {
        ensure!(
            std::time::Instant::now() < deadline,
            "worker retirement timeout"
        );
        std::thread::yield_now();
    }
    ensure!(
        prepare.active_owners() == 0
            && prepare.active_bytes() == 0
            && prepare.in_flight() == 0
            && serial.in_flight() == 0,
        "native handle accounting leaked"
    );
    ensure!(
        prepare.start(bound).is_err(),
        "closed compiler accepted request"
    );
    println!(
        "{{\"all\":true,\"canonical_vocab\":true,\"four_interleaved_matchers\":true,\"four_serial_manager_owners\":true,\"tokens_per_owner\":{},\"invalid_compile_recovery\":true,\"owner_identity\":true,\"native_handles_released\":true,\"cpu_workers_retired\":true,\"scope\":\"actual tokenizer, request wire, compiler actor, mask pool and retained serial manager; no model/CUDA/HTTP\"}}",
        output.len() + 1
    );
    Ok(())
}
