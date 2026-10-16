//! Bounded GPU qualification for cooperative speculative-cache destinations.
//! Invoked only by the isolated retained-model probe, never by serving.
use super::*;
use crate::model::long::priority::Boundary;
use crate::priority::Snapshot as Hold;
use serde_json::{Value, json};

const CUT: usize = 2112;
const END: usize = 2176;
const OUTPUT: usize = 4;
const DESTINATION: usize = 1;
const FOREGROUND: usize = 2;

struct Fence {
    control: Control,
    tick: u64,
}
impl Fence {
    fn new(long: &Long<'_>) -> Result<Self> {
        Ok(Self {
            control: Control::new(&long.engine.device)?,
            tick: 0,
        })
    }
    fn work<T>(
        &mut self,
        long: &mut Long<'_>,
        op: impl FnOnce(&mut Long<'_>) -> Result<T>,
    ) -> Result<T> {
        let result = op(long);
        let mut hash = Sha256::new();
        hash.update(b"cooperative-companion-probe-v1");
        hash.update(self.tick.to_le_bytes());
        self.tick += 1;
        long.agree_speculation_work(&self.control, 851, hash.finalize().into(), result)
    }
    fn common(&mut self, long: &mut Long<'_>, local: bool) -> Result<bool> {
        let mut hash = Sha256::new();
        hash.update(b"cooperative-companion-wait-v1");
        hash.update(self.tick.to_le_bytes());
        self.tick += 1;
        self.control.agree(long, 852, hash.finalize().into(), local)
    }
}

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
struct State {
    target: String,
    draft: String,
    pending: Option<i32>,
}
fn state(long: &mut Long<'_>, cursor: usize) -> Result<State> {
    let saved = long
        .capture_speculation()?
        .context("companion probe requires enabled drafter")?;
    let mut h = Sha256::new();
    h.update(saved.metadata());
    for buffer in saved.buffers() {
        h.update(buffer.read(0, buffer.bytes)?);
    }
    Ok(State {
        target: long.state(cursor)?,
        draft: format!("{:x}", h.finalize()),
        pending: long.sessions[long.active].pending,
    })
}
fn prefill(fence: &mut Fence, long: &mut Long<'_>, tokens: &[i32], end: usize) -> Result<()> {
    while long.sessions[long.active].tokens.len() < end {
        let start = long.sessions[long.active].tokens.len();
        let next = (start + long.prefill_rows).min(end);
        fence.work(long, |long| long.prefill(start, &tokens[start..next]))?;
    }
    Ok(())
}
fn matches_state(
    fence: &mut Fence,
    long: &mut Long<'_>,
    cursor: usize,
    reference: &State,
) -> Result<()> {
    fence.work(long, |long| {
        ensure!(
            state(long, cursor)? == *reference,
            "companion restore/cold state drift"
        );
        Ok(())
    })
}
fn begin(
    reader: &mut Reader,
    fence: &mut Fence,
    long: &mut Long<'_>,
    tokens: &[i32],
) -> Result<()> {
    fence.work(long, |long| long.bind(DESTINATION))?;
    let started = reader.start(long, tokens.to_vec().into(), OUTPUT, true)?;
    fence.work(long, |long| {
        ensure!(started, "companion read admission skipped");
        ensure!(
            long.bind(DESTINATION).is_err()
                && long.start(64, 1).is_err()
                && long.prefill(0, &[3; 64]).is_err()
                && long.cancel().is_err(),
            "partial companion destination became runnable"
        );
        Ok(())
    })?;
    fence.work(long, |long| long.bind(0))
}
fn partial_companion(
    reader: &mut Reader,
    fence: &mut Fence,
    long: &mut Long<'_>,
) -> Result<(usize, usize)> {
    let private = reader
        .disk
        .id
        .segments(CUT)?
        .iter()
        .filter(|s| {
            !matches!(
                s.role,
                Role::Tokens | Role::DraftMetadata | Role::DraftKv(_)
            )
        })
        .map(|s| s.bytes as usize)
        .sum::<usize>();
    let target_bytes = (CUT - 1) / PAGE * format::PAGE_BYTES as usize + private;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let local = reader.active.as_ref().is_some_and(|active| {
            active.draft.is_some() && active.transfer.is_some_and(|s| s.bytes > target_bytes)
        });
        // Stop each ready rank at its first companion payload, so a faster
        // worker cannot finish while the slowest rank is still catching up.
        ensure!(
            reader.step(long, false, !local, false)?.is_none(),
            "partial fixture finished early"
        );
        if fence.common(long, local)? {
            let active = reader.active.as_ref().unwrap();
            let transfer = active.transfer.context("partial transfer")?;
            fence.work(long, |_| {
                ensure!(
                    transfer.bytes < target_bytes + CUT.min(2047) * 5120,
                    "entire companion copied before hold"
                );
                Ok(())
            })?;
            return Ok((transfer.copies, transfer.bytes));
        }
        let timely = Instant::now() < deadline;
        fence.work(long, |_| {
            ensure!(timely, "partial companion deadline");
            Ok(())
        })?;
        std::thread::yield_now();
    }
}
fn complete(reader: &mut Reader, fence: &mut Fence, long: &mut Long<'_>) -> Result<Report> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(report) = reader.step(long, false, true, false)? {
            return Ok(report);
        }
        let timely = Instant::now() < deadline;
        fence.work(long, |_| {
            ensure!(timely, "companion read completion deadline");
            Ok(())
        })?;
        std::thread::yield_now();
    }
}
fn assert_discarded(
    fence: &mut Fence,
    long: &mut Long<'_>,
    reader: &Reader,
    report: &Report,
) -> Result<()> {
    fence.work(long, |long| {
        ensure!(
            report.outcome == "discarded"
                && report.restored_tokens.is_none()
                && reader.active.is_none()
                && reader.disk.store.is_none()
                && long.memory.book().slot(DESTINATION)?.phase() == Phase::Idle
                && long.memory.book().slot(DESTINATION)?.pages().is_empty()
                && long.sessions[DESTINATION].tokens.is_empty()
                && long.sessions[DESTINATION].pending.is_none(),
            "partial target/companion not discarded after worker exit"
        );
        Ok(())
    })
}
fn cold_fallback(
    fence: &mut Fence,
    long: &mut Long<'_>,
    tokens: &[i32],
    prefix: &State,
    final_state: &State,
) -> Result<()> {
    fence.work(long, |long| long.bind(DESTINATION))?;
    fence.work(long, |long| long.start_mode(END, OUTPUT, true))?;
    prefill(fence, long, tokens, CUT)?;
    matches_state(fence, long, CUT, prefix)?;
    prefill(fence, long, tokens, END)?;
    matches_state(fence, long, END, final_state)?;
    fence.work(long, |long| long.finish())
}

/// Deliberately alter semantic metadata while preserving every file checksum.
/// The normal format reader must accept this file; only model metadata rejects it.
fn corrupt_metadata(path: &Path, id: &Identity, field: usize) -> Result<Value> {
    ensure!(matches!(field, 0 | 40), "companion corruption field");
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "fixture is not regular"
    );
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let mut header = format::Header::read(&mut file, id)?;
    let segment = id
        .segments(header.n)?
        .into_iter()
        .find(|s| s.role == Role::DraftMetadata)
        .context("fixture companion metadata segment")?;
    let body_offset = segment.offset as usize + field;
    let chunk = body_offset / CHUNK;
    let start = chunk * CHUNK;
    let count = (header.body_bytes as usize - start).min(CHUNK);
    let offset = header.header_bytes() as u64 + start as u64;
    let mut bytes = vec![0; count];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    ensure!(
        Sha256::digest(&bytes)[..] == header.sums[chunk],
        "fixture original checksum"
    );
    let before = bytes[body_offset - start];
    bytes[body_offset - start] ^= 1;
    header.sums[chunk] = Sha256::digest(&bytes).into();
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&bytes)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.encode(id))?;
    file.sync_all()?;
    let checked = format::Header::read(&mut file, id)?;
    let mut stage = vec![0; CHUNK];
    let mut visited = 0;
    format::read_body(&mut file, &checked, &mut stage, |_, bytes| {
        visited += bytes.len();
        Ok(())
    })?;
    ensure!(
        visited == header.body_bytes as usize,
        "fixture full checked body"
    );
    Ok(
        json!({"field_offset":field,"before_byte":before,"after_byte":before^1,
        "chunk":chunk,"new_chunk_sha256":format::hex(&header.sums[chunk]),
        "header_and_all_body_checksums_valid":true}),
    )
}

/// Reuses the caller's enabled Long after its retained-slot tests have finished.
pub(in crate::model) fn run(long: &mut Long<'_>, source: &[i32], path: &Path) -> Result<Value> {
    let mut fence = Fence::new(long)?;
    fence.work(long, |long| {
        ensure!(
            source.len() >= END && long.speculation_identity().is_some(),
            "companion probe inputs"
        );
        ensure!(
            (0..crate::cache_pages::SLOTS).all(|s| long
                .memory
                .book()
                .slot(s)
                .is_ok_and(|s| s.phase() == Phase::Idle)),
            "companion probe requires idle slots"
        );
        Ok(())
    })?;
    let tokens = &source[..END];
    let mut reader = Reader::new(Disk::new(long, path, 256 << 20)?);
    let mut boundary = Boundary::new(long)?;
    fence.work(long, |long| long.bind(0))?;
    fence.work(long, |long| long.start_mode(END, OUTPUT, true))?;
    prefill(&mut fence, long, tokens, CUT)?;
    let prefix = fence.work(long, |long| state(long, CUT))?;
    let saved = reader.idle()?.save(long)?;
    fence.work(long, |_| {
        ensure!(saved == Outcome::Saved, "companion snapshot skipped");
        Ok(())
    })?;
    prefill(&mut fence, long, tokens, END)?;
    let final_state = fence.work(long, |long| state(long, END))?;
    fence.work(long, |long| long.finish())?;

    begin(&mut reader, &mut fence, long, tokens)?;
    let held_progress = partial_companion(&mut reader, &mut fence, long)?;
    boundary.apply(
        long,
        (long.engine.rank == 0).then_some(Hold {
            revision: 1,
            hold: true,
        }),
    )?;
    fence.work(long, |long| long.bind(FOREGROUND))?;
    fence.work(long, |long| long.start(64, OUTPUT))?;
    fence.work(long, |long| long.prefill(0, &[43; 64]))?;
    let foreground = fence.work(long, |long| state(long, 64))?;
    for _ in 0..3 {
        ensure!(
            reader.step(long, true, true, false)?.is_none(),
            "held reader retired"
        );
        let active = reader.active.as_ref().context("held reader lost owner")?;
        let transfer = active.transfer.context("held reader lost transfer")?;
        let unchanged = (transfer.copies, transfer.bytes) == held_progress
            && active
                .draft
                .as_ref()
                .is_some_and(|s| s.gpu_bytes() == CUT.min(2047) * 5120);
        fence.work(long, |_| {
            ensure!(unchanged, "held companion copied/freed");
            Ok(())
        })?;
    }
    matches_state(&mut fence, long, 64, &foreground)?;
    // Finish foreground work before releasing the native hold. Then reuse that
    // other slot as active background work while the reader publishes, proving
    // that publication targets DESTINATION rather than Long::active.
    fence.work(long, |long| long.finish())?;
    boundary.apply(
        long,
        (long.engine.rank == 0).then_some(Hold {
            revision: 2,
            hold: false,
        }),
    )?;
    fence.work(long, |long| long.start_mode(64, OUTPUT, true))?;
    fence.work(long, |long| long.prefill(0, &[43; 64]))?;
    let restored = complete(&mut reader, &mut fence, long)?;
    fence.work(long, |long| {
        ensure!(
            restored.restored_tokens == Some(CUT) && long.active == FOREGROUND,
            "companion restore destination/bind drift"
        );
        Ok(())
    })?;
    matches_state(&mut fence, long, 64, &foreground)?;
    fence.work(long, |long| long.finish())?;
    fence.work(long, |long| long.bind(DESTINATION))?;
    matches_state(&mut fence, long, CUT, &prefix)?;
    prefill(&mut fence, long, tokens, END)?;
    matches_state(&mut fence, long, END, &final_state)?;
    fence.work(long, |long| long.finish())?;

    begin(&mut reader, &mut fence, long, tokens)?;
    let cancel_progress = partial_companion(&mut reader, &mut fence, long)?;
    boundary.apply(
        long,
        (long.engine.rank == 0).then_some(Hold {
            revision: 3,
            hold: true,
        }),
    )?;
    ensure!(
        reader
            .step(long, true, false, long.engine.rank == 0)?
            .is_none(),
        "cancel retired under hold"
    );
    let cancel_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        ensure!(
            reader.step(long, true, true, false)?.is_none(),
            "cancelled companion retired under native hold"
        );
        let active = reader
            .active
            .as_ref()
            .context("cancelled read owner lost")?;
        let transfer = active.transfer.context("cancelled transfer lost")?;
        let retained =
            active.draft.is_some() && (transfer.copies, transfer.bytes) == cancel_progress;
        let exited = active.job.is_none();
        fence.work(long, |_| {
            ensure!(
                retained,
                "cancelled held companion freed/copied before retirement"
            );
            Ok(())
        })?;
        if fence.common(long, exited)? {
            break;
        }
        let timely = Instant::now() < cancel_deadline;
        fence.work(long, |_| {
            ensure!(timely, "held cancellation worker exit deadline");
            Ok(())
        })?;
        std::thread::yield_now();
    }
    boundary.apply(
        long,
        (long.engine.rank == 0).then_some(Hold {
            revision: 4,
            hold: false,
        }),
    )?;
    let cancelled = complete(&mut reader, &mut fence, long)?;
    assert_discarded(&mut fence, long, &reader, &cancelled)?;
    cold_fallback(&mut fence, long, tokens, &prefix, &final_state)?;
    reader.idle()?.reopen(long)?;
    fence.work(long, |_| {
        ensure!(
            reader.disk.store.as_ref().is_some_and(|s| s.len() == 1),
            "cancellation deleted immutable snapshot"
        );
        Ok(())
    })?;

    let (key, _) = format::keys(&reader.disk.id.namespace, &tokens[..CUT])?;
    let entry = path
        .join("entries")
        .join(format!("{}.entry", format::hex(&key)));
    let mut failures = Vec::new();
    for field in [0usize, 40] {
        let corruption = if long.engine.rank == 2 {
            corrupt_metadata(&entry, &reader.disk.id, field)
        } else {
            Ok(json!(null))
        };
        let corruption = fence.work(long, |_| corruption)?;
        begin(&mut reader, &mut fence, long, tokens)?;
        let failed = complete(&mut reader, &mut fence, long)?;
        assert_discarded(&mut fence, long, &reader, &failed)?;
        fence.work(long, |long| {
            ensure!(
                failed.copy_bytes > 0,
                "metadata fault did not follow target copies"
            );
            if long.engine.rank == 2 {
                ensure!(
                    failed
                        .file_error
                        .as_deref()
                        .is_some_and(|e| e.contains("speculation snapshot")),
                    "semantic metadata error absent"
                );
            }
            Ok(())
        })?;
        cold_fallback(&mut fence, long, tokens, &prefix, &final_state)?;
        failures.push(json!({"field_offset":field,"corruption":corruption,"report":failed,"cold_target_and_draft_exact":true}));
        // Reconcile only after every worker has exited. The rank that rejected
        // semantic metadata must have quarantined it before returning the
        // catalog, so the intersection removes this entry from every rank.
        reader.idle()?.reopen(long)?;
        fence.work(long, |_| {
            ensure!(
                reader.disk.store.as_ref().is_some_and(|s| s.len() == 0),
                "malformed companion survived catalog reconciliation"
            );
            Ok(())
        })?;
        fence.work(long, |long| long.start_mode(END, OUTPUT, true))?;
        prefill(&mut fence, long, tokens, CUT)?;
        let saved = reader.idle()?.save(long)?;
        fence.work(long, |_| {
            ensure!(saved == Outcome::Saved, "fixture repair skipped");
            Ok(())
        })?;
        fence.work(long, |long| long.finish())?;
    }
    long.memory.book().check()?;
    Ok(
        json!({"prefix_tokens":CUT,"prompt_tokens":END,"cache_root":path,
        "prefix_reference":prefix,"prompt_reference":final_state,"foreground_reference":foreground,
        "held_partial_progress":{"copies":held_progress.0,"bytes":held_progress.1},
        "cancel_partial_progress":{"copies":cancel_progress.0,"bytes":cancel_progress.1},
        "restored":restored,"cancelled":cancelled,"metadata_faults":failures,
        "partial_companion_held":true,"explicit_destination_preserves_other_active_slot":true,
        "worker_exit_before_discard":true,"cold_target_and_draft_exact":true,
        "scope":"Isolated cooperative reader with enabled retained drafter; native hold, partial companion cancellation, two one-rank checksum-valid semantic metadata faults. No serving or speed claim."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_corruption_keeps_complete_file_checksums_valid() -> Result<()> {
        let id = Identity::new([[3; 32]; 4], [4; 32], 2, 1024)?.with_speculation([7; 32]);
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "alaska-companion-corruption-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let path = root.join("fixture.entry");
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        let pages = (0..(CUT - 1) / PAGE)
            .map(|_| format::PageRef {
                key: [11; 32],
                checksum: [13; 32],
            })
            .collect();
        let mut header = format::Header::new(&id, [17; 32], CUT, 1, pages)?;
        let metadata = id
            .segments(CUT)?
            .into_iter()
            .find(|s| s.role == Role::DraftMetadata)
            .unwrap();
        let mut stage = vec![0; CHUNK];
        format::write_body(&mut file, &mut header, &mut stage, |_, bytes| {
            bytes.fill(0);
            Ok(())
        })?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.encode(&id))?;
        file.sync_all()?;
        let chunk = metadata.offset as usize / CHUNK;
        for field in [0usize, 40] {
            let before = format::Header::read(&mut file, &id)?;
            let proof = corrupt_metadata(&path, &id, field)?;
            assert_eq!(proof["header_and_all_body_checksums_valid"], true);
            let after = format::Header::read(&mut file, &id)?;
            assert_eq!(before.file_bytes(), after.file_bytes());
            assert_eq!(before.key, after.key);
            assert_eq!(before.n, after.n);
            for (index, (a, b)) in before.sums.iter().zip(&after.sums).enumerate() {
                assert_eq!(a == b, index != chunk);
            }
            let mut bytes = [0; 64];
            file.seek(SeekFrom::Start(
                after.header_bytes() as u64 + metadata.offset,
            ))?;
            file.read_exact(&mut bytes)?;
            for (index, &value) in bytes.iter().enumerate() {
                assert_eq!(value, u8::from(index == 0 || (field == 40 && index == 40)));
            }
        }
        drop(file);
        fs::remove_file(path)?;
        fs::remove_dir(root)?;
        Ok(())
    }
}
