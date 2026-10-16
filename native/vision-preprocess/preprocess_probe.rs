//! Actual Rust preprocessing, exported for byte-for-byte native ATen comparison.
//! Input decoding is intentionally external; this probe accepts pinned RGB bytes.
#![allow(dead_code)] // Shared modules expose service APIs outside this CPU probe.
#[path = "../../src/cache_pages.rs"]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
mod chat_json;
#[allow(clippy::duplicate_mod)] // Metadata fixture is also imported by shared module tests.
#[path = "../../src/vision_fixtures.rs"]
mod vision_fixtures;
#[path = "../../src/vision_input.rs"]
mod vision_input;
#[path = "../../src/vision_preprocess.rs"]
mod vision_preprocess;
#[path = "../../src/vision_resize.rs"]
mod vision_resize;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "expected RGB-file height width max-tokens exclusive-output-directory"
    );
    let height: usize = args[1].parse()?;
    let width: usize = args[2].parse()?;
    let plan = vision_preprocess::Plan::new(height, width, args[3].parse()?)?;
    ensure!(
        fs::metadata(&args[0])?.len() == (height * width * 3) as u64,
        "RGB file size"
    );
    let rgb = fs::read(&args[0])?;
    let rgb_sha = format!("{:x}", Sha256::digest(&rgb));
    let settings = vision_preprocess::Settings::canonical(
        &serde_json::from_str(vision_fixtures::CONFIG)?,
        &serde_json::from_str(vision_fixtures::PROCESSOR)?,
    )?;
    let prepared = vision_preprocess::Prepared::from_rgb(plan, rgb, &settings)?;
    let output = PathBuf::from(&args[4]);
    fs::create_dir(&output).context("exclusive output directory")?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("patches-f32.bin"))?;
    let mut file = BufWriter::new(file);
    let grid = plan.grid();
    let count = grid[1] * grid[2];
    let mut patch = vec![0.0f32; 1024 * vision_preprocess::PATCH_WIDTH];
    let mut hash = Sha256::new();
    let mut bytes = Vec::with_capacity(patch.len() * 4);
    for start in (0..count).step_by(1024) {
        let rows = 1024.min(count - start);
        let values = &mut patch[..rows * vision_preprocess::PATCH_WIDTH];
        prepared.write_patch_rows(&settings, start, rows, values)?;
        bytes.clear();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)?;
        hash.update(&bytes);
    }
    file.flush()?;
    let value = serde_json::json!({
        "passed":true,"production_ready":false,"scope":"actual Rust preprocessing only; no decode/tower",
        "source_size":[height,width],"rgb_sha256":rgb_sha,"canvas_size":plan.canvas_size(),
        "content_size":plan.content_size(),"grid":grid,"image_tokens":plan.tokens(),
        "retained_bytes":prepared.retained_bytes(),"patch_bytes":count * vision_preprocess::PATCH_WIDTH * 4,
        "patch_sha256":format!("{:x}",hash.finalize()),
        "image_key":prepared.key().0.iter().map(|v|format!("{v:02x}")).collect::<String>(),
        "settings_key_sha256":format!("{:x}",Sha256::digest(settings.key())),
        "executable_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?)),
    });
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&value)?,
    )?;
    println!("{value}");
    Ok(())
}
