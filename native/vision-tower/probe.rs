//! Actual Rust-owned tower over isolated native primitives. Never a service.
#![allow(dead_code)] // Shared modules expose service APIs outside this probe.
#[path = "../../src/cache_pages.rs"]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
mod chat_json;
#[allow(clippy::duplicate_mod)] // Metadata fixture also used by module tests.
#[path = "../../src/vision_fixtures.rs"]
mod vision_fixtures;
#[path = "../../src/vision_input.rs"]
mod vision_input;
#[path = "../../src/vision_native.rs"]
mod vision_native;
#[path = "../../src/vision_preprocess.rs"]
mod vision_preprocess;
#[path = "../../src/vision_resize.rs"]
mod vision_resize;
#[path = "../../src/vision_tower.rs"]
mod vision_tower;
#[path = "../../src/vision_weights.rs"]
mod vision_weights;
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 9,
        "expected native-library library-sha256 weights RGB-file height width max-tokens exclusive-output-directory image-png-sha256"
    );
    let plan = vision_preprocess::Plan::new(args[4].parse()?, args[5].parse()?, args[6].parse()?)?;
    let height: usize = args[4].parse()?;
    let width: usize = args[5].parse()?;
    ensure!(
        fs::metadata(&args[3])?.len() == (height * width * 3) as u64,
        "RGB size"
    );
    ensure!(
        args[8].len() == 64
            && args[8]
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "source PNG digest required"
    );
    let settings = vision_preprocess::Settings::canonical(
        &serde_json::from_str(vision_fixtures::CONFIG)?,
        &serde_json::from_str(vision_fixtures::PROCESSOR)?,
    )?;
    let rgb = fs::read(&args[3])?;
    let rgb_sha = format!("{:x}", Sha256::digest(&rgb));
    let prepared = vision_preprocess::Prepared::from_rgb(plan, rgb, &settings)?;
    let output = Path::new(&args[7]);
    fs::create_dir(output)?;
    let native = vision_native::Native::open(Path::new(&args[0]), &args[1])?;
    let began = Instant::now();
    let tower = vision_tower::Tower::load(native.clone(), Path::new(&args[2]))?;
    let load_seconds = began.elapsed().as_secs_f64();
    let mut stages = Vec::new();
    let mut observe = |stage: &str, t: &vision_native::Tensor| -> Result<()> {
        let (kind, shape, length) = t.info()?;
        let bytes = t.bytes()?;
        ensure!(bytes.len() == length, "dump length");
        let dtype = match kind {
            vision_native::Kind::Bf16 => "BFloat16",
            vision_native::Kind::F32 => "Float",
            vision_native::Kind::I64 => "Long",
        };
        let record = serde_json::json!({"stage":stage,"dtype":dtype,"shape":shape,"bytes":length,"sha256":format!("{:x}",Sha256::digest(&bytes))});
        ensure!(
            !output.join(format!("{stage}.bin")).exists(),
            "duplicate stage"
        );
        fs::write(output.join(format!("{stage}.bin")), bytes)?;
        println!("{record}");
        stages.push(record);
        Ok(())
    };
    let began = Instant::now();
    let result = tower.encode(&prepared, &settings, &mut observe)?;
    native.sync()?;
    ensure!(
        result.info()?.1 == [plan.tokens() as i64, 4096],
        "tower output grid"
    );
    let report = serde_json::json!({"passed":true,"production_ready":false,"scope":"Rust-owned tower with diagnostic readbacks; unqualified until native oracle comparison",
        "png_sha256":args[8],"rgb_sha256":rgb_sha,"library_sha256":args[1],"grid":plan.grid(),"load_seconds":load_seconds,
        "diagnostic_encode_seconds":began.elapsed().as_secs_f64(),"stages":stages,"binary_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?))});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
