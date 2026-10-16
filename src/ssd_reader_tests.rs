use super::*;
use crate::ssd_format::Identity;
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn pattern(part: Source, offset: u64, buf: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let seed = Sha256::digest(format!("{part:?}").as_bytes())[0];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = ((offset + i as u64) % 251) as u8 ^ seed;
    }
}
fn select(job: &mut ReadJob) -> Result<Selected> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(s) = job.selection()? {
            return s.context("reader fixture missing");
        }
        ensure!(Instant::now() < deadline, "reader selection deadline");
        thread::yield_now();
    }
}
pub fn probe(root: &Path, rank: u32) -> Result<serde_json::Value> {
    probe_options(
        root,
        rank,
        std::env::var("ALASKA_SSD_READ_BATCH").as_deref() == Ok("1"),
    )
}
fn probe_options(root: &Path, rank: u32, batched: bool) -> Result<serde_json::Value> {
    ensure!(!root.exists(), "reader probe requires fresh directory");
    let id = Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [45; 32], rank, 1024)?;
    let mut store = Store::open(root, id.clone(), 128 << 20)?;
    store.save(&[7; 320], false, |part, offset, buf| {
        pattern(part, offset, buf);
        Ok(())
    })?;
    let request: Arc<[i32]> = vec![7; 321].into();
    let mut job = ReadJob::start_options(store, request.clone(), None, batched)?;
    let selected = select(&mut job)?;
    ensure!(selected.tokens == 320, "reader selection");
    for _ in 0..10_000 {
        ensure!(
            job.poll()?.is_none() && job.try_finish()?.is_none(),
            "reader bypassed admission"
        );
    }
    ensure!(
        Store::open(root, id.clone(), 128 << 20).is_err(),
        "reader lost exclusive catalog"
    );
    job.begin()?;
    let mut verified = 0;
    let mut slices = 0;
    let mut batches = 0;
    let mut expected = vec![0; CHUNK];
    let deadline = Instant::now() + Duration::from_secs(20);
    let store = loop {
        if let Some(c) = job.poll()? {
            if slices == 0 {
                for _ in 0..10_000 {
                    ensure!(
                        job.try_finish()?.is_none(),
                        "reader bypassed chunk acknowledgement"
                    );
                }
            }
            for (source, offset, bytes) in c.parts() {
                pattern(source, offset, &mut expected[..bytes.len()]);
                ensure!(
                    bytes == &expected[..bytes.len()],
                    "reader changed checked bytes"
                );
                slices += 1;
            }
            verified += c.bytes().len();
            batches += 1;
            job.acknowledge(c, Ok(()))?;
        }
        if let Some((store, result)) = job.try_finish()? {
            ensure!(result? == Some(selected), "reader completion identity");
            break store;
        }
        ensure!(Instant::now() < deadline, "reader completion deadline");
        thread::yield_now();
    };
    ensure!(
        if batched {
            batches < slices
        } else {
            batches == slices
        },
        "reader batching policy"
    );
    let mut cancel = ReadJob::start_options(store, request.clone(), None, batched)?;
    select(&mut cancel)?;
    cancel.begin()?;
    let deadline = Instant::now() + Duration::from_secs(10);
    let held = loop {
        if let Some(c) = cancel.poll()? {
            break c;
        }
        ensure!(
            Instant::now() < deadline,
            "reader cancellation chunk deadline"
        );
        thread::yield_now();
    };
    cancel.cancel();
    drop(held);
    let mut store = loop {
        if let Some((store, result)) = cancel.try_finish()? {
            ensure!(result.is_err(), "cancelled read succeeded");
            break store;
        }
        ensure!(Instant::now() < deadline, "reader cancellation deadline");
        thread::yield_now();
    };
    ensure!(
        store.lookup(&request)? == Some(selected.key),
        "cancel removed immutable prefix"
    );
    // Corrupt the last private chunk, after several earlier chunks are delivered.
    let path = root
        .join("entries")
        .join(format!("{}.entry", crate::ssd_format::hex(&selected.key)));
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    file.seek(SeekFrom::End(-1))?;
    let mut byte = [0];
    file.read_exact(&mut byte)?;
    byte[0] ^= 1;
    file.seek(SeekFrom::End(-1))?;
    file.write_all(&byte)?;
    file.sync_all()?;
    drop(file);
    let mut bad = ReadJob::start_options(store, request.clone(), None, batched)?;
    select(&mut bad)?;
    bad.begin()?;
    let mut delivered = 0;
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(c) = bad.poll()? {
            delivered += c.bytes().len();
            bad.acknowledge(c, Ok(()))?;
        }
        if let Some((mut store, result)) = bad.try_finish()? {
            ensure!(
                result.is_err() && delivered > CHUNK && store.lookup(&request)?.is_none(),
                "late corruption was not invalidated"
            );
            break;
        }
        ensure!(Instant::now() < deadline, "reader corruption deadline");
        thread::yield_now();
    }
    Ok(
        serde_json::json!({"rank":rank,"batched":batched,"byte_exact":true,"checked_payload_bytes":verified,"source_slices":slices,"handoffs":batches,"metadata_parts_limit":MAX_PARTS,"exclusive_catalog":true,"waits_for_admission":true,"waits_for_acknowledgement":true,"cancel_preserves_immutable_prefix":true,"late_checksum_error_forgets_entry":true,"delivered_before_checksum_error":delivered,"exchange_bytes":CHUNK,"codec_stage_bytes":CHUNK,"channel_capacity":1}),
    )
}
#[test]
fn bounded_reader_checks_bytes_admission_cancel_and_late_error() {
    check(false);
}
#[test]
fn batched_reader_checks_bytes_admission_cancel_and_late_error() {
    check(true);
}
#[cfg(test)]
fn check(batched: bool) {
    let root = std::env::temp_dir().join(format!(
        "alaska-reader-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    probe_options(&root, 0, batched).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
