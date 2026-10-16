//! Explicit CPU-only real-file qualification. Never uploads to CUDA or serves.
#[path = "../../src/vision_weights.rs"]
mod vision_weights;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, time::Instant};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 1,
        "expected canonical vision.safetensors path"
    );
    let path = Path::new(&args[0]);
    let start = Instant::now();
    let mut file = vision_weights::Weights::open(path)?;
    let mut roles = BTreeSet::new();
    let mut digest = Sha256::new();
    let mut chunks = 0;
    let mut current = None;
    let mut expected_offset = 0;
    let verified = file.load(1024 * 1024, |spec, offset, bytes| {
        if current.as_deref() != Some(spec.name.as_str()) {
            ensure!(roles.insert(format!("{:?}", spec.role)), "duplicate role");
            ensure!(offset == 0, "new tensor starts after zero");
            current = Some(spec.name.clone());
            expected_offset = 0;
        }
        ensure!(offset == expected_offset, "nonsequential tensor chunks");
        expected_offset += bytes.len() as u64;
        ensure!(expected_offset <= spec.bytes(), "chunk exceeds shape");
        digest.update(bytes);
        chunks += 1;
        Ok(())
    })?;
    ensure!(
        roles.len() == 347 && verified.tensors == 347,
        "incomplete tower"
    );
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
    ensure!(host.len() <= 256, "hostname bound");
    let executable = std::env::current_exe().context("probe executable")?;
    let executable_sha256 = format!("{:x}", Sha256::digest(std::fs::read(executable)?));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed":true,"production_ready":false,"host":host.trim(),"path":path,
            "manifest_sha256":verified.identity,"binary_sha256":executable_sha256,
            "tensor_count":verified.tensors,"payload_bytes":verified.payload_bytes,
            "staging_bytes":verified.staging_bytes,"chunks":chunks,
            "payload_file_order_sha256":format!("{:x}",digest.finalize()),
            "elapsed_seconds":start.elapsed().as_secs_f64(),
            "scope":"Actual standalone Rust loader reads every pinned BF16 byte using bounded staging. No GPU allocation, tower inference, image preparation or serving."
        }))?
    );
    Ok(())
}
