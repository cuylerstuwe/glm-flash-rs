//! Isolated actual-source PNG decode and preparation probe; no Python/runtime API.
#![allow(dead_code)] // Shared source APIs also serve the later image owner.
#[path = "../../src/cache_pages.rs"]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
mod chat_json;
#[path = "../../src/vision_codecs.rs"]
mod vision_codecs;
#[path = "../../src/vision_decode.rs"]
mod vision_decode;
#[path = "../../src/vision_fetch.rs"]
mod vision_fetch;
#[path = "../../src/vision_fetch_policy.rs"]
mod vision_fetch_policy;
#[allow(clippy::duplicate_mod)]
#[path = "../../src/vision_fixtures.rs"]
mod vision_fixtures;
#[path = "../../src/vision_gif.rs"]
mod vision_gif;
#[path = "../../src/vision_input.rs"]
mod vision_input;
#[path = "../../src/vision_markup.rs"]
mod vision_markup;
#[path = "../../src/vision_prepare.rs"]
mod vision_prepare;
#[path = "../../src/vision_preprocess.rs"]
mod vision_preprocess;
#[path = "../../src/vision_request.rs"]
mod vision_request;
#[path = "../../src/vision_resize.rs"]
mod vision_resize;
#[path = "../../src/vision_source.rs"]
mod vision_source;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "expected image-file max-tokens exclusive-output-directory [private-native-codec-manifest]"
    );
    let mut file = fs::File::open(&args[0])?;
    ensure!(
        file.metadata()?.len() <= vision_decode::MAX_ENCODED_BYTES as u64,
        "encoded file bound"
    );
    let mut encoded = Vec::new();
    Read::take(&mut file, vision_decode::MAX_ENCODED_BYTES as u64 + 1).read_to_end(&mut encoded)?;
    let budget = vision_decode::DEFAULT_WORKSPACE_BYTES;
    let decoded = if encoded.starts_with(b"\x89PNG") {
        vision_decode::png(&encoded, budget, || Ok(()))?
    } else if encoded.starts_with(b"GIF") {
        vision_gif::decode(&encoded, budget, || Ok(()))?
    } else {
        let manifest = serde_json::from_slice(&fs::read(
            args.get(3)
                .context("native image codec manifest required")?,
        )?)?;
        let native = vision_codecs::Codecs::load(&manifest)?;
        if encoded.starts_with(&[0xff, 0xd8]) {
            native.jpeg(&encoded, budget, || Ok(()))?
        } else {
            native.webp(&encoded, budget, || Ok(()))?
        }
    };
    let (width, height, workspace_charge) =
        (decoded.width, decoded.height, decoded.workspace_charge);
    let output = PathBuf::from(&args[2]);
    fs::create_dir(&output).context("exclusive output directory")?;
    let rgb_sha = format!("{:x}", Sha256::digest(&decoded.rgb));
    fs::write(output.join("decoded.rgb"), &decoded.rgb)?;
    let settings = vision_preprocess::Settings::canonical(
        &serde_json::from_str(vision_fixtures::CONFIG)?,
        &serde_json::from_str(vision_fixtures::PROCESSOR)?,
    )?;
    let plan = vision_preprocess::Plan::new(height, width, args[1].parse()?)?;
    let prepared = vision_preprocess::Prepared::from_rgb(plan, decoded.rgb, &settings)?;
    let report = serde_json::json!({
        "passed":true,"production_ready":false,
        "scope":"actual Rust image decode and preprocessing; frozen-library differential/GPU/serving unqualified",
        "size":[height,width],"encoded_sha256":format!("{:x}",Sha256::digest(encoded)),
        "rgb_sha256":rgb_sha,"workspace_charge":workspace_charge,
        "retained_bytes":prepared.retained_bytes(),"canvas_size":plan.canvas_size(),
        "grid":plan.grid(),"image_tokens":plan.tokens(),
        "image_key":prepared.key().0.iter().map(|v|format!("{v:02x}")).collect::<String>(),
        "executable_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?)),
    });
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
