//! Separate-process qualification harness. The producer and consumer are built
//! into the same executable because executable identity is part of the format.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    tokens: usize,
    key: Hash,
    state: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    rank: usize,
    namespace: Hash,
    request: Hash,
    producer_pid: u32,
    producer_started_ns: String,
    snapshots: Vec<Snapshot>,
    reference: serde_json::Value,
}
fn request_hash(tokens: &[i32]) -> Hash {
    let mut h = Sha256::new();
    h.update(b"ssd-recovery-fixture-v1");
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}
pub(super) fn flip_last(path: &Path) -> Result<()> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.seek(SeekFrom::End(-1))?;
    let mut b = [0];
    f.read_exact(&mut b)?;
    b[0] ^= 1;
    f.seek(SeekFrom::End(-1))?;
    f.write_all(&b)?;
    f.sync_all()?;
    Ok(())
}
pub(super) fn discarded(long: &Long<'_>) -> Result<bool> {
    let slot = long.memory.book().slot(long.active)?;
    Ok(long.healthy
        && long.sessions[long.active].tokens.is_empty()
        && long.sessions[long.active].pending.is_none()
        && slot.phase() == Phase::Idle
        && slot.pages().is_empty())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    sets: &[Vec<i32>],
    _new_tokens: usize,
    use_roce: bool,
    bundle: &Path,
    produce: bool,
) -> Result<serde_json::Value> {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_string();
    let tokens = sets.last().context("SSD recovery fixture")?;
    ensure!(
        (4096..=16384).contains(&tokens.len()),
        "SSD recovery fixture length"
    );
    let request = request_hash(tokens);
    let page_cut = if tokens.len() >= 16384 { 8192 } else { 2048 };
    let cut = page_cut + 64;
    let budget = 256 << 20;
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let control = Control::new(&long.engine.device)?;
    let prepared = (|| -> Result<Option<Record>> {
        if produce {
            fs::create_dir(bundle).context("producer requires a fresh bundle")?;
            return Ok(None);
        }
        ensure!(
            fs::symlink_metadata(bundle)?.file_type().is_dir(),
            "recovery bundle type"
        );
        let path = bundle.join("reference.json");
        ensure!(
            fs::symlink_metadata(&path)?.file_type().is_file(),
            "recovery record type"
        );
        let f = File::open(path)?;
        ensure!(f.metadata()?.len() <= 1 << 20, "recovery record bound");
        let r: Record = serde_json::from_reader(f.take(1 << 20))?;
        ensure!(
            r.version == 1
                && r.rank == rank
                && r.request == request
                && r.producer_pid != std::process::id()
                && r.producer_started_ns != started
                && r.snapshots.iter().map(|s| s.tokens).eq([page_cut, cut]),
            "recovery fixture/process mismatch"
        );
        Ok(Some(r))
    })();
    ensure!(
        control.agree(&long, 200, request, prepared.is_ok())?,
        "recovery fixture preflight"
    );
    let expected = prepared?;
    let mut disk = Disk::new(&mut long, &bundle.join("cache"), budget)?;
    ensure!(disk.store.is_some(), "recovery storage disabled");
    if produce {
        ensure!(
            disk.store.as_ref().unwrap().len() == 0,
            "producer cache not empty"
        );
        long.start(tokens.len(), 4)?;
        let mut snapshots = Vec::new();
        for n in [page_cut, cut] {
            prefill_to(&mut long, &tokens[..n])?;
            let state = long.state(n)?;
            ensure!(
                disk.save(&mut long)? == Outcome::Saved && long.state(n)? == state,
                "recovery producer save changed state"
            );
            snapshots.push(Snapshot {
                tokens: n,
                key: format::keys(&disk.id.namespace, &tokens[..n])?.0,
                state,
            });
        }
        let r = Record {
            version: 1,
            rank,
            namespace: disk.id.namespace,
            request,
            producer_pid: std::process::id(),
            producer_started_ns: started,
            snapshots,
            reference: finish_reference(&mut long, tokens)?,
        };
        let saved = (|| -> Result<()> {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(bundle.join("reference.json"))?;
            serde_json::to_writer_pretty(&mut f, &r)?;
            f.write_all(b"\n")?;
            f.sync_all()?;
            File::open(bundle)?.sync_all()?;
            Ok(())
        })();
        disk.require(&mut long, 201, request, saved.is_ok())?;
        saved?;
        return Ok(serde_json::json!({"mode":"produce","rank":rank,
            "transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),
            "bundle":bundle,"record":r,"durable_producer_complete":true,
            "charged_bytes":disk.store.as_ref().unwrap().bytes(),"budget_bytes":budget}));
    }
    let expected = expected.unwrap();
    disk.require(
        &mut long,
        202,
        request,
        expected.namespace == disk.id.namespace && disk.store.as_ref().unwrap().len() == 2,
    )?;
    ensure!(
        long.sessions.iter().all(|s| s.tokens.is_empty())
            && (0..4).all(|i| long
                .memory
                .book()
                .slot(i)
                .is_ok_and(|s| s.pages().is_empty())),
        "recovery must precede all prompt prefill"
    );
    // This first restore occurs in an entirely new engine with no prompt state.
    for s in &expected.snapshots {
        ensure!(
            disk.restore(&mut long, tokens, 4, false, Some(s.key))? == Some(s.tokens)
                && long.state(s.tokens)? == s.state,
            "new-process snapshot bytes"
        );
        ensure!(
            finish_reference(&mut long, tokens)? == expected.reference,
            "new-process model reference"
        );
    }
    long.bind(1)?;
    ensure!(
        disk.restore(&mut long, tokens, 4, false, None)? == Some(cut),
        "restart longest prefix"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == expected.reference,
        "restart longest model"
    );
    let independent = long.state(tokens.len() + 3)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    long.bind(0)?;
    long.start(tokens.len(), 4)?;
    // Match the producer's slab boundaries, independent of stored payloads.
    prefill_to(&mut long, &tokens[..page_cut])?;
    prefill_to(&mut long, &tokens[..cut])?;
    ensure!(
        finish_reference(&mut long, tokens)? == expected.reference,
        "fresh cold reference"
    );

    // Corrupt a shared page after several coalesced groups have reached the GPU.
    // Flip it back only after coordinated rejection; these are isolated test files.
    let page = format::keys(&disk.id.namespace, &tokens[..cut])?.1[6];
    let page_path = disk
        .root
        .join("pages")
        .join(format!("{}.page", format::hex(&page)));
    if rank == 2 {
        flip_last(&page_path)?;
    }
    disk.require(&mut long, 203, request, true)?;
    ensure!(
        disk.restore(&mut long, tokens, 4, false, None)?.is_none() && discarded(&long)?,
        "page-body corruption must discard partial state"
    );
    if rank == 2 {
        flip_last(&page_path)?;
    }
    disk.require(&mut long, 204, request, true)?;
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..page_cut])?;
    prefill_to(&mut long, &tokens[..cut])?;
    ensure!(
        disk.save(&mut long)? == Outcome::Saved,
        "repair isolated page fixture"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == expected.reference,
        "page-fault cold model"
    );

    // Real filesystem open failure on just rank 1, after new pages were written.
    // Other ranks may commit; reopen must retain only the common entries.
    let input = vec![37; 2177];
    long.bind(3)?;
    long.start(input.len(), 4)?;
    prefill_to(&mut long, &input[..2176])?;
    let source_state = long.state(2176)?;
    let key = format::keys(&disk.id.namespace, &input[..2176])?.0;
    let obstruction = disk
        .root
        .join("entries")
        .join(format!("{}.entry.tmp", format::hex(&key)));
    if rank == 1 {
        fs::create_dir(&obstruction)?;
    }
    disk.require(&mut long, 205, key, true)?;
    ensure!(
        disk.save(&mut long)? == Outcome::Skipped
            && disk.store.is_none()
            && long.healthy
            && long.state(2176)? == source_state,
        "write failure must disable disk collectively without changing source"
    );
    if rank == 1 {
        fs::remove_dir(&obstruction)?;
    }
    let write_reference = finish_reference(&mut long, &input)?;
    long.start(input.len(), 4)?;
    prefill_to(&mut long, &input[..2176])?;
    ensure!(
        finish_reference(&mut long, &input)? == write_reference,
        "write-fault cold reference"
    );
    disk.reopen(&mut long)?;
    ensure!(
        disk.store
            .as_ref()
            .is_some_and(|s| s.len() == 2 && !s.order().contains(&key)),
        "failed-write restart intersection"
    );
    ensure!(
        disk.restore(&mut long, tokens, 4, false, None)? == Some(cut)
            && long.state(cut)? == expected.snapshots[1].state,
        "reopen retained prefix"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == expected.reference,
        "reopen retained model"
    );
    long.bind(1)?;
    ensure!(
        long.state(tokens.len() + 3)? == independent,
        "fault changed independent owner"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "fault changed paused owner");
    long.cancel()?;
    Ok(serde_json::json!({"mode":"recover","rank":rank,
        "transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),
        "bundle":bundle,"namespace":format::hex(&disk.id.namespace),
        "producer_pid":expected.producer_pid,"consumer_pid":std::process::id(),
        "producer_started_ns":expected.producer_started_ns,"consumer_started_ns":started,
        "restored_tokens":[page_cut,cut],"reference":expected.reference,
        "new_process_before_prefill_exact":true,"longest_prefix_after_restart_exact":true,
        "fresh_cold_reference_exact":true,"one_rank_page_corruption_cold_fallback_exact":true,
        "one_rank_write_failure_disables_tier_preserves_source":true,
        "write_failure_reopen_intersection_exact":true,"paused_and_independent_owners_unchanged":true,
        "charged_bytes":disk.store.as_ref().unwrap().bytes(),"budget_bytes":budget,
        "copy_calls":disk.copy_calls,"copied_bytes":disk.copy_bytes,
        "scope":"Separate-process pure-prefix recovery and filesystem faults; no power-loss or cooperative serving qualification."}))
}
