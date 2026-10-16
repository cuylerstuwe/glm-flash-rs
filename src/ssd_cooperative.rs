//! Cooperative RAM-snapshot writer. One file job/rank, one bounded source slice
//! per boundary, identical control order despite independent worker readiness.
use super::super::store::{Export, Store as Ram};
use super::*;
use crate::ssd_worker::WriteJob;
use std::time::{Duration, Instant};
#[path = "ssd_independent.rs"]
mod independent;

struct Active {
    source: Export,
    job: Option<WriteJob>,
    independent: Option<independent::CopyJob>,
    done: Option<(Store, Result<Outcome>)>,
    operation: Hash,
    tick: u64,
    abort: bool,
    slices: usize,
}
pub(super) struct Writer {
    // Kept private: no catalog operation or staging reuse while active.
    disk: Disk,
    active: Option<Active>,
    serial: u64,
    background: bool,
}
#[derive(serde::Serialize)]
pub(in super::super) struct Report {
    outcome: Option<&'static str>,
    file_error: Option<String>,
    slices: usize,
    copy_calls: usize,
    copy_bytes: usize,
    boundaries: u64,
}
fn outcome(value: Outcome) -> u8 {
    match value {
        Outcome::Saved => 1,
        Outcome::Duplicate => 2,
        Outcome::Skipped => 3,
    }
}
impl Writer {
    pub(super) fn progress(&self) -> (usize, usize) {
        self.active.as_ref().map_or((0, 0), |a| {
            a.independent
                .as_ref()
                .map_or((a.source.bytes, a.slices), |j| j.gate.progress())
        })
    }
    pub(super) fn background(&self) -> bool {
        self.background
    }
    /// Lending the read-only lane gives the new foreground restore priority
    /// over any optional save, including one produced by an older foreground
    /// request. Close and drain DMA before parking; the next writer step resumes
    /// using its original classification. File-owned bytes may still finish I/O.
    pub(super) fn pause_for_read(&mut self, long: &mut Long<'_>) -> Result<()> {
        let mut healthy = true;
        if let Some(job) = self.active.as_ref().and_then(|a| a.independent.as_ref()) {
            job.gate.hold(true);
            healthy = !job.gate.fatal.load(std::sync::atomic::Ordering::Acquire);
        }
        self.disk
            .require(long, 612, self.disk.id.namespace, healthy)
    }
    pub(super) fn at(disk: Disk, serial: u64) -> Self {
        let mut owner = Self::new(disk);
        owner.serial = serial;
        owner
    }
    pub(super) fn into_disk(self) -> Result<Disk> {
        ensure!(self.active.is_none(), "SSD job still owns catalog");
        Ok(self.disk)
    }
    fn new(disk: Disk) -> Self {
        Self {
            disk,
            active: None,
            serial: 0,
            background: true,
        }
    }
    pub(super) fn start(&mut self, long: &mut Long<'_>, ram: &mut Ram, id: u64) -> Result<bool> {
        self.start_mode(long, ram, id, true)
    }
    pub(super) fn start_mode(
        &mut self,
        long: &mut Long<'_>,
        ram: &mut Ram,
        id: u64,
        background: bool,
    ) -> Result<bool> {
        self.disk.ready(long)?;
        self.background = background;
        let mut h = Sha256::new();
        h.update(b"cooperative-ssd-save-v1");
        h.update(self.disk.id.namespace);
        h.update(self.serial.to_le_bytes());
        h.update(id.to_le_bytes());
        h.update([
            u8::from(background),
            u8::from(self.disk.copy_stream.is_some()),
        ]);
        let operation = h.finalize().into();
        self.disk.require(
            long,
            300,
            operation,
            self.active.is_none() && self.serial < u64::MAX,
        )?;
        if !self
            .disk
            .gate(long, 301, operation, self.disk.store.is_some())?
        {
            self.disk.store = None;
            return Ok(false);
        }
        self.disk.require(long, 302, self.disk.digest(), true)?;
        // The disk already owns 8 MiB pinned + 8 MiB codec staging. Reserve
        // another exchange, bounded page-map/Arc metadata, and retain 2 GiB free.
        let free = long.engine.device.allocatable_memory();
        self.disk.require(long, 303, operation, free.is_ok())?;
        if !self
            .disk
            .gate(long, 304, operation, free?.0 >= RESERVE + CHUNK + (1 << 20))?
        {
            return Ok(false);
        }
        let Some(source) = ram.export(long, id, self.disk.id.ring_rows)? else {
            return Ok(false);
        };
        let independent = self.disk.copy_stream.is_some();
        let mut copy_job = None;
        let job = if independent {
            let gate = crate::copy_gate::Gate::new();
            let plan = ram.copy_plan(long, &source, self.disk.id.ring_rows, &gate);
            self.disk.require(long, 309, operation, plan.is_ok())?;
            let started = independent::CopyJob::start(
                self.disk.store.take().unwrap(),
                source.tokens.clone(),
                plan?,
                self.disk.copy_stream.take().unwrap(),
                gate,
            );
            match started {
                Ok(job) => {
                    copy_job = Some(job);
                    Ok(None)
                }
                Err(e) => Err(e),
            }
        } else {
            WriteJob::start(self.disk.store.take().unwrap(), source.tokens.clone()).map(Some)
        };
        // A spawn failure is a file-tier failure. Other workers are cancelled
        // and retained until they exit; no replacement can bypass their locks.
        let started = self.disk.gate(long, 305, operation, job.is_ok())?;
        if let Err(e) = &job {
            eprintln!("SSD worker start failed: {e:#}");
        }
        let mut job = job.ok().flatten();
        if !started && let Some(j) = &mut job {
            j.cancel();
        }
        if !started && let Some(j) = &copy_job {
            j.cancel();
        }
        self.active = Some(Active {
            source,
            job,
            independent: copy_job,
            done: None,
            operation,
            tick: 0,
            abort: !started,
            slices: 0,
        });
        self.serial += 1;
        Ok(true)
    }
    /// Call at the same scheduler boundary on every rank. `foreground` is a
    /// common scheduler decision; local readiness/throttling and cancel may differ.
    /// No file data I/O, blocking receive or unfinished join occurs here.
    /// Retiring a failed catalog can close its already-open lock descriptor.
    pub(super) fn step(
        &mut self,
        long: &mut Long<'_>,
        ram: &mut Ram,
        foreground: bool,
        allow_copy: bool,
        cancel: bool,
    ) -> Result<Option<Report>> {
        self.disk.ready(long)?;
        let active = self.active.as_mut().context("SSD writer not active")?;
        active.abort |= cancel;
        let mut fatal = false;
        if let Some(job) = &mut active.independent {
            use std::sync::atomic::Ordering;
            if active.abort {
                job.cancel();
            }
            job.gate.hold(foreground || !allow_copy || active.abort);
            fatal |= job.gate.fatal.load(Ordering::Acquire);
            active.source.bytes = job.gate.bytes.load(Ordering::Relaxed);
            active.source.copies = job.gate.calls.load(Ordering::Relaxed);
            active.slices = job.gate.slices.load(Ordering::Relaxed);
            match job.try_finish() {
                Ok(Some((store, result, stream))) => {
                    active.abort |= result.is_err();
                    active.source.bytes = job.gate.bytes.load(Ordering::Relaxed);
                    active.source.copies = job.gate.calls.load(Ordering::Relaxed);
                    active.slices = job.gate.slices.load(Ordering::Relaxed);
                    fatal |= job.gate.fatal.load(Ordering::Acquire);
                    self.disk.copy_stream = Some(stream);
                    active.done = Some((store, result));
                    active.independent = None;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("SSD independent completion failed: {e:#}");
                    fatal = true;
                    job.cancel();
                }
            }
        }
        if let Some(job) = &mut active.job {
            if active.abort {
                job.cancel();
            }
            if !active.abort && !foreground && allow_copy {
                let copied = (|| -> Result<()> {
                    if let Some(mut request) = job.poll()? {
                        let copied = request.fill(|part, offset, bytes| {
                            ram.copy_export(
                                long,
                                &mut active.source,
                                &mut self.disk.stage,
                                part,
                                offset,
                                bytes,
                            )
                        });
                        let success = copied.is_ok();
                        job.respond(request, copied)?;
                        ensure!(success, "SSD source copy failed");
                        active.slices += 1;
                    }
                    Ok(())
                })();
                if let Err(e) = copied {
                    eprintln!("SSD source/worker protocol failed: {e:#}");
                    fatal = true;
                    job.cancel();
                }
            }
            match job.try_finish() {
                Ok(Some(done)) => {
                    active.abort |= done.1.is_err();
                    active.done = Some(done);
                    active.job = None;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("SSD worker completion failed: {e:#}");
                    fatal = true;
                    job.cancel();
                }
            }
        }
        let mut record = [0; 64];
        record[..32].copy_from_slice(&active.operation);
        record[32..40].copy_from_slice(&active.tick.to_le_bytes());
        record[40..44].copy_from_slice(&306u32.to_le_bytes());
        record[44] = u8::from(foreground);
        record[45] = u8::from(fatal);
        record[46] = u8::from(active.abort);
        record[47] = u8::from(active.job.is_none() && active.independent.is_none());
        record[48] = active
            .done
            .as_ref()
            .and_then(|d| d.1.as_ref().ok())
            .map(|v| outcome(*v))
            .unwrap_or(0);
        // Always the same collective, including on ranks without a ready slice.
        // Its stream drain also fences a cancelled source's final DMA.
        let all = self.disk.exchange(long, &record)?;
        if all.iter().any(|r| r[..45] != record[..45] || r[45] != 0) {
            eprintln!(
                "SSD control mismatch rank={} local={record:?} gathered={all:?}",
                long.engine.rank
            );
            if let Some(j) = &mut active.job {
                j.cancel();
            }
            if let Some(j) = &active.independent {
                j.cancel();
            }
            self.disk.healthy = false;
            long.healthy = false;
            anyhow::bail!("SSD cooperative control/source failure; owner poisoned");
        }
        active.abort |= all.iter().any(|r| r[46] != 0);
        if active.abort
            && let Some(j) = &mut active.job
        {
            j.cancel();
        }
        if active.abort
            && let Some(j) = &active.independent
        {
            j.cancel();
        }
        active.tick = active
            .tick
            .checked_add(1)
            .context("SSD progress counter exhausted")?;
        // Retain the immutable pin and catalog through a foreground hold even
        // if all file threads have finished. Retirement can free GPU allocations.
        if foreground || all.iter().any(|r| r[47] == 0) {
            return Ok(None);
        }
        if !active.abort {
            self.disk.require(
                long,
                307,
                active.operation,
                all.iter().all(|r| r[48] == record[48] && r[48] != 0),
            )?;
            let digest = active
                .done
                .as_ref()
                .context("SSD worker missing catalog")?
                .0
                .digest();
            self.disk.require(long, 308, digest, true)?;
        }
        let active = self.active.take().unwrap();
        ram.release_export(long, &active.source)?;
        let file_error = active
            .done
            .as_ref()
            .and_then(|done| done.1.as_ref().err())
            .map(|e| format!("{e:#}"));
        let value = if active.abort {
            None
        } else {
            let (store, result) = active.done.context("SSD completed catalog")?;
            self.disk.store = Some(store);
            Some(match result? {
                Outcome::Saved => "saved",
                Outcome::Duplicate => "duplicate",
                Outcome::Skipped => "skipped",
            })
        };
        self.disk.copy_calls += active.source.copies;
        self.disk.copy_bytes += active.source.bytes;
        Ok(Some(Report {
            outcome: value,
            file_error,
            slices: active.slices,
            copy_calls: active.source.copies,
            copy_bytes: active.source.bytes,
            boundaries: active.tick,
        }))
    }
    fn idle(&mut self) -> Result<&mut Disk> {
        ensure!(
            self.active.is_none(),
            "SSD writer still owns catalog/staging"
        );
        Ok(&mut self.disk)
    }
}

fn finish(writer: &mut Writer, long: &mut Long<'_>, ram: &mut Ram) -> Result<Report> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let tick = writer.active.as_ref().context("SSD harness active")?.tick;
        // Deterministic local staggering deliberately changes slice readiness,
        // while every rank executes every control boundary in the same order.
        let copy = !tick.is_multiple_of(long.engine.rank as u64 + 2);
        if let Some(report) = writer.step(long, ram, false, copy, false)? {
            return Ok(report);
        }
        ensure!(
            Instant::now() < deadline,
            "SSD cooperative harness deadline"
        );
        std::thread::yield_now();
    }
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
) -> Result<serde_json::Value> {
    let tokens = sets.last().context("cooperative SSD fixture")?;
    ensure!(tokens.len() == 16384, "cooperative SSD fixture length");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let control = Control::new(&long.engine.device)?;
    let mut record = [0; 64];
    record[..16].copy_from_slice(
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_le_bytes(),
    );
    let all = control.exchange(&long, &record)?;
    let run = u128::from_le_bytes(all[0][..16].try_into().unwrap());
    let path = PathBuf::from(format!("evidence/ssd-cooperative-{run}-rank{rank}"));
    ensure!(!path.exists(), "cooperative fixture cache collision");
    let mut writer = Writer::new(Disk::new(&mut long, &path, 256 << 20)?);
    let mut ram = Ram::new(&long, 256 << 20)?;
    let cut = 8256;
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..8192])?;
    prefill_to(&mut long, &tokens[..cut])?;
    let state = long.state(cut)?;
    let source_id = ram.save(&mut long)?.id.context("source admission")?;
    let reference = finish_reference(&mut long, tokens)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    ensure!(
        writer.start(&mut long, &mut ram, source_id)?,
        "worker admission"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    // Wait collectively until every rank has actually copied source chunks.
    loop {
        ensure!(
            writer
                .step(&mut long, &mut ram, false, true, false)?
                .is_none(),
            "worker finished before suspension"
        );
        let copied = writer.active.as_ref().unwrap().slices >= 30;
        if control.agree(&long, 320, [0; 32], copied)? {
            break;
        }
        ensure!(Instant::now() < deadline, "initial worker progress");
    }
    // An independent stream may progress between owner polls. Establish the
    // closed/drained boundary before recording the counter that must stay frozen.
    ensure!(
        writer
            .step(&mut long, &mut ram, true, true, false)?
            .is_none(),
        "suspension retired worker"
    );
    let before = writer.active.as_ref().unwrap().source.copies;
    // Original live source is overwritten. Five other RAM snapshots force
    // eviction decisions while the exported immutable source stays charged.
    let mut evictions = 0;
    for fill in 0..5 {
        ensure!(
            writer
                .step(&mut long, &mut ram, true, true, false)?
                .is_none(),
            "suspended worker unexpectedly complete"
        );
        long.bind(0)?;
        let input = vec![45 + fill; 2177];
        long.start(input.len(), 4)?;
        prefill_to(&mut long, &input[..2176])?;
        let saved = ram.save(&mut long)?;
        ensure!(
            !saved.evicted.contains(&source_id),
            "writer source was evicted"
        );
        evictions += saved.evicted.len();
        finish_reference(&mut long, &input)?;
    }
    ensure!(
        evictions > 0 && writer.active.as_ref().unwrap().source.copies == before,
        "foreground did not suspend writer DMA / eviction fixture"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "writer changed paused model");
    let saved = finish(&mut writer, &mut long, &mut ram)?;
    ensure!(saved.outcome == Some("saved"), "cooperative save failed");
    long.bind(1)?;
    ensure!(
        writer.idle()?.restore(&mut long, tokens, 4, false, None)? == Some(cut)
            && long.state(cut)? == state
            && finish_reference(&mut long, tokens)? == reference,
        "cooperative restore/model mismatch"
    );
    // Duplicate completes without source DMA and still retires its counted pin.
    ensure!(
        writer.start(&mut long, &mut ram, source_id)?,
        "duplicate admission"
    );
    let duplicate = finish(&mut writer, &mut long, &mut ram)?;
    ensure!(
        duplicate.outcome == Some("duplicate") && duplicate.copy_calls == 0,
        "duplicate worker DMA/outcome"
    );
    // Cancel only rank 0 while each worker awaits its first missing-page slice.
    long.bind(0)?;
    let other = vec![22; 2177];
    long.start(other.len(), 4)?;
    prefill_to(&mut long, &other[..2176])?;
    let other_id = ram.save(&mut long)?.id.context("cancel source")?;
    let other_state = long.state(2176)?;
    long.finish()?;
    ensure!(
        writer.start(&mut long, &mut ram, other_id)?,
        "cancel admission"
    );
    ensure!(
        writer
            .step(&mut long, &mut ram, true, false, rank == 0)?
            .is_none(),
        "unexpected immediate cancel completion"
    );
    let cancelled = finish(&mut writer, &mut long, &mut ram)?;
    ensure!(
        cancelled.outcome.is_none()
            && writer.idle()?.store.is_none()
            && long.state(2176)? == other_state,
        "cancel propagation/source state"
    );
    writer.idle()?.reopen(&mut long)?;
    ensure!(
        writer
            .idle()?
            .store
            .as_ref()
            .context("cancel reopen")?
            .len()
            == 1,
        "cancel left partial entry"
    );
    // Real rank-local file failure after source copies: entry commit .tmp is a
    // directory. All workers must retire before reopening the common catalog.
    let (key, _) = format::keys(&writer.disk.id.namespace, &other[..2176])?;
    let obstruction = path
        .join("entries")
        .join(format!("{}.entry.tmp", format::hex(&key)));
    let injected = if rank == 1 {
        fs::create_dir(&obstruction)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 321, [0; 32], injected.is_ok())?,
        "file fault setup"
    );
    injected?;
    ensure!(
        writer.start(&mut long, &mut ram, other_id)?,
        "fault admission"
    );
    let failed = finish(&mut writer, &mut long, &mut ram)?;
    ensure!(
        failed.outcome.is_none()
            && writer.idle()?.store.is_none()
            && long.state(2176)? == other_state,
        "file failure changed GPU/model state"
    );
    let removed = if rank == 1 {
        fs::remove_dir(&obstruction)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 322, [0; 32], removed.is_ok())?,
        "file fault cleanup"
    );
    removed?;
    writer.idle()?.reopen(&mut long)?;
    ensure!(
        writer
            .idle()?
            .store
            .as_ref()
            .context("failure reopen")?
            .len()
            == 1,
        "fault reconciliation intersection"
    );
    long.bind(3)?;
    ensure!(
        writer.idle()?.restore(&mut long, tokens, 4, false, None)? == Some(cut)
            && long.state(cut)? == state
            && finish_reference(&mut long, tokens)? == reference,
        "fault reopen lost valid original"
    );
    // Corrupt a later page: earlier uploads must be discarded on all ranks.
    let (_, pages) = format::keys(&writer.disk.id.namespace, &tokens[..cut])?;
    let bad = path
        .join("pages")
        .join(format!("{}.page", format::hex(&pages[6])));
    let corrupted = if rank == 2 {
        recovery::flip_last(&bad)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 323, [0; 32], corrupted.is_ok())?,
        "late checksum fixture"
    );
    corrupted?;
    long.bind(1)?;
    ensure!(
        writer
            .idle()?
            .restore(&mut long, tokens, 4, false, None)?
            .is_none()
            && recovery::discarded(&long)?,
        "worker cache partial load not discarded"
    );
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..8192])?;
    prefill_to(&mut long, &tokens[..cut])?;
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "worker corruption cold fallback"
    );
    long.bind(2)?;
    ensure!(
        long.state(64)? == paused,
        "file faults changed paused owner"
    );
    long.resume()?;
    long.finish()?;
    // A disagreement in foreground scheduling is fatal on every rank. No
    // further source copy is allowed, and the pinned source remains charged
    // until owner teardown. This is the final check in this process.
    ensure!(
        writer.start(&mut long, &mut ram, other_id)?,
        "disagreement fixture admission"
    );
    let before = writer.active.as_ref().unwrap().source.copies;
    ensure!(
        writer
            .step(&mut long, &mut ram, rank == 3, false, false)
            .is_err()
            && !long.healthy
            && !writer.disk.healthy
            && writer.active.as_ref().unwrap().source.copies == before
            && long.start(64, 4).is_err(),
        "cooperative disagreement did not fail closed"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce {"roce"} else {"nccl"},"prompt_tokens":tokens.len(),"snapshot_tokens":cut,"reference":reference,"snapshot_state":state,"namespace":format::hex(&writer.disk.id.namespace),"cache_root":path,"suspended_dma_unchanged":true,"foreground_model_work_while_suspended":true,"original_live_source_overwritten":true,"pinned_source_survives_ram_evictions":evictions,"asymmetric_worker_progress_exact":true,"saved":saved,"duplicate":duplicate,"cancelled":cancelled,"failed":failed,"rank_local_cancel_collective":true,"rank_local_file_failure_collective":true,"reopened_common_prefix_exact":true,"late_page_checksum_discards_partial_destination":true,"cold_fallback_exact":true,"paused_model_unchanged":true,"one_worker_per_rank":true,"rank_disagreement_poisons_before_new_copy":true,"pinned_stage_bytes":CHUNK,"codec_stage_bytes":CHUNK,"exchange_bytes":CHUNK,"native_lease_latency_qualified":false}),
    )
}
