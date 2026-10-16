//! Cooperative checked file reads into an explicitly unavailable destination.
use super::*;
use crate::ssd_reader::{ReadJob, Selected};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[path = "retained_companion_read_probe.rs"]
mod companion_probe;
pub(in crate::model) use companion_probe::run as run_companion_probe;

struct ActiveRead {
    request: Arc<[i32]>,
    output: usize,
    background: bool,
    slot: usize,
    operation: Hash,
    tick: u64,
    abort: bool,
    decided: bool,
    job: Option<ReadJob>,
    done: Option<(Store, Result<Option<Selected>>)>,
    choice: Option<Option<Selected>>,
    selected: Option<Selected>,
    transfer: Option<TransferState>,
    draft: Option<DraftSaved>,
    slices: usize,
    batches: usize,
}
pub(super) struct Reader {
    disk: Disk,
    active: Option<ActiveRead>,
    serial: u64,
    pub(super) file_wake: Option<crate::request_broker::FileWake>,
}
#[derive(serde::Serialize)]
pub(in super::super) struct Report {
    outcome: &'static str,
    pub(in super::super) restored_tokens: Option<usize>,
    file_error: Option<String>,
    slices: usize,
    batches: usize,
    copy_calls: usize,
    copy_bytes: usize,
    boundaries: u64,
}
impl Reader {
    pub(super) fn background(&self) -> bool {
        self.active.as_ref().is_none_or(|a| a.background)
    }
    pub(super) fn progress(&self) -> (usize, usize) {
        self.active
            .as_ref()
            .map_or((0, 0), |a| (a.transfer.map_or(0, |t| t.bytes), a.slices))
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
            file_wake: None,
        }
    }
    pub(super) fn start(
        &mut self,
        long: &mut Long<'_>,
        request: Arc<[i32]>,
        output: usize,
        background: bool,
    ) -> Result<bool> {
        self.disk.ready(long)?;
        let mut h = Sha256::new();
        h.update(b"cooperative-ssd-read-v1");
        h.update(self.serial.to_le_bytes());
        h.update(self.disk.id.namespace);
        h.update(request_key(long, &request, output, background));
        let operation = h.finalize().into();
        self.disk.require(
            long,
            500,
            operation,
            self.active.is_none()
                && self.serial < u64::MAX
                && output > 0
                && output <= MAX_OUTPUT
                && request.len() <= CONTEXT - output
                && request
                    .iter()
                    .all(|&t| crate::vision_input::prompt_token(t)),
        )?;
        if !self
            .disk
            .gate(long, 501, operation, self.disk.store.is_some())?
        {
            self.disk.store = None;
            return Ok(false);
        }
        let slot = long.active;
        let plan = long
            .memory
            .book()
            .plan(slot, request.len(), output, 0, background);
        self.disk.require(
            long,
            502,
            plan.as_ref().map(|p| p.digest()).unwrap_or([0; 32]),
            plan.is_ok(),
        )?;
        let free = long.engine.device.allocatable_memory();
        let draft_bytes = if long.speculation_identity().is_some() {
            DRAFT_RESTORE_BYTES
        } else {
            0
        };
        self.disk.require(long, 503, operation, free.is_ok())?;
        if !self.disk.gate(
            long,
            504,
            operation,
            free?.0 >= RESERVE + CHUNK + (8 << 20) + draft_bytes,
        )? {
            return Ok(false);
        }
        long.healthy = false;
        let reserved = (|| -> Result<()> {
            long.commit_admission(&plan?, 0)?;
            long.memory.begin_restore(slot)?;
            let s = &mut long.sessions[slot];
            s.tokens.clear();
            s.pending = None;
            s.mixed = false;
            s.prompt_end = 0;
            s.total_end = 0;
            s.background = background;
            unsafe {
                s.position.set_i32(&long.engine.stream, 0)?;
            }
            long.engine.sync()?;
            if long.latest_logits == Some(slot) {
                long.latest_logits = None;
            }
            Ok(())
        })();
        self.disk.require(long, 505, operation, reserved.is_ok())?;
        reserved?;
        long.healthy = true;
        let job = ReadJob::start_with_wake(
            self.disk.store.take().unwrap(),
            request.clone(),
            self.file_wake.clone(),
        );
        let started = self.disk.gate(long, 506, operation, job.is_ok())?;
        if let Err(e) = &job {
            eprintln!("SSD read worker start failed: {e:#}");
        }
        let mut job = job.ok();
        if !started && let Some(j) = &mut job {
            j.cancel();
        }
        self.active = Some(ActiveRead {
            request,
            output,
            background,
            slot,
            operation,
            tick: 0,
            abort: !started,
            decided: false,
            job,
            done: None,
            choice: None,
            selected: None,
            transfer: None,
            draft: None,
            slices: 0,
            batches: 0,
        });
        self.serial += 1;
        Ok(true)
    }
    pub(super) fn step(
        &mut self,
        long: &mut Long<'_>,
        foreground: bool,
        allow_copy: bool,
        cancel: bool,
    ) -> Result<Option<Report>> {
        self.disk.ready(long)?;
        let active = self.active.as_mut().context("SSD reader not active")?;
        active.abort |= cancel;
        let mut fatal = false;
        if let Some(job) = &mut active.job {
            // A semantic file rejection has already been acknowledged as Err.
            // Let that worker quarantine the entry before its normal exit;
            // setting its cancellation flag here would race Store::forget.
            if active.abort && !active.transfer.is_some_and(|t| t.metadata_invalid) {
                job.cancel();
            }
            active.choice = job.selection()?;
            if !active.abort && active.selected.is_some() && !foreground && allow_copy {
                let copied = (|| -> Result<()> {
                    if let Some(chunk) = job.poll()? {
                        ensure!(
                            long.memory.book().slot(active.slot)?.phase() == Phase::Restoring,
                            "SSD read destination became available"
                        );
                        let mut transfer = Transfer {
                            long,
                            stage: &mut self.disk.stage,
                            state: active.transfer.context("SSD read transfer missing")?,
                            draft: &mut active.draft,
                        };
                        let copied = chunk.parts().try_for_each(|(source, offset, bytes)| {
                            transfer.sink(source, offset, bytes)
                        });
                        active.transfer = Some(transfer.state);
                        let success = copied.is_ok();
                        let invalid_metadata = transfer.state.metadata_invalid;
                        let slices = chunk.parts().count();
                        job.acknowledge(chunk, copied)?;
                        if invalid_metadata {
                            // A checksummed file can still have invalid typed
                            // metadata. Wait for this worker to retire, then
                            // discard the reserved target/companion together.
                            // Acknowledge/channel errors above remain fatal.
                            active.abort = true;
                        } else {
                            ensure!(success, "SSD read GPU copy failed");
                        }
                        active.slices += slices;
                        active.batches += 1;
                    }
                    Ok(())
                })();
                if let Err(e) = copied {
                    eprintln!("SSD read copy/protocol failed: {e:#}");
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
                    eprintln!("SSD reader lost completion: {e:#}");
                    fatal = true;
                    job.cancel();
                }
            }
        }
        let mut r = [0; 64];
        r[..32].copy_from_slice(&active.operation);
        r[32..40].copy_from_slice(&active.tick.to_le_bytes());
        r[40..44].copy_from_slice(&507u32.to_le_bytes());
        r[44] = u8::from(foreground);
        r[45] = u8::from(fatal);
        r[46] = u8::from(active.abort);
        r[47] = u8::from(active.job.is_none());
        r[48] = u8::from(active.decided);
        r[49] = u8::from(active.choice.is_some() || active.done.is_some());
        let all = self.disk.exchange(long, &r)?;
        if all
            .iter()
            .any(|a| a[..45] != r[..45] || a[45] != 0 || a[48] != r[48])
        {
            if let Some(j) = &mut active.job {
                j.cancel();
            }
            self.disk.healthy = false;
            long.healthy = false;
            anyhow::bail!("SSD cooperative read control/source failure; owner poisoned");
        }
        active.abort |= all.iter().any(|a| a[46] != 0);
        active.tick = active
            .tick
            .checked_add(1)
            .context("SSD read boundary counter exhausted")?;
        if active.abort
            && !active.transfer.is_some_and(|t| t.metadata_invalid)
            && let Some(job) = &mut active.job
        {
            job.cancel();
        }
        // Preparing a destination table and committing/discarding it can also
        // enqueue CUDA work. Defer those transitions during foreground holds.
        if foreground {
            return Ok(None);
        }
        if !active.abort && !active.decided && all.iter().all(|a| a[49] != 0) {
            let mut choice = [0; 64];
            if let Some(Some(s)) = active.choice {
                choice[..32].copy_from_slice(&s.key);
                choice[32..40].copy_from_slice(&(s.tokens as u64).to_le_bytes());
                choice[40] = 1;
            }
            let choices = self.disk.exchange(long, &choice)?;
            active.decided = true;
            if !choices.iter().all(|c| *c == choice) {
                active.abort = true;
            } else if let Some(Some(s)) = active.choice {
                self.disk.require(
                    long,
                    508,
                    active.operation,
                    s.tokens > 0 && s.tokens < active.request.len() && s.tokens.is_multiple_of(64),
                )?;
                long.healthy = false;
                let prepared = (|| -> Result<()> {
                    long.memory
                        .map_restore(&long.engine.stream, active.slot, s.tokens)?;
                    validate_layout_at(long, active.slot, s.tokens, &self.disk.id)?;
                    long.engine.sync()
                })();
                self.disk
                    .require(long, 509, active.operation, prepared.is_ok())?;
                prepared?;
                long.healthy = true;
                active.selected = Some(s);
                active.transfer = Some(TransferState::new(active.slot, s.tokens));
                let begun = active
                    .job
                    .as_mut()
                    .context("selected reader already ended")?
                    .begin();
                self.disk
                    .require(long, 510, active.operation, begun.is_ok())?;
                begun?;
            }
        }
        if active.abort
            && !active.transfer.is_some_and(|t| t.metadata_invalid)
            && let Some(job) = &mut active.job
        {
            job.cancel();
        }
        if all.iter().any(|a| a[47] == 0) {
            return Ok(None);
        }
        if !active.abort {
            let valid = active
                .done
                .as_ref()
                .is_some_and(|(_, r)| r.as_ref().is_ok_and(|s| *s == active.selected));
            self.disk
                .require(long, 511, active.operation, valid && active.decided)?;
            let mut digest = [0; 64];
            digest[..32].copy_from_slice(&active.done.as_ref().unwrap().0.digest());
            // A lookup can remove a damaged candidate on only one rank. Keep
            // inference healthy but defer catalog reconciliation off the owner.
            let catalogs = self.disk.exchange(long, &digest)?;
            active.abort |= !catalogs.iter().all(|d| *d == digest);
        }
        let mut active = self.active.take().unwrap();
        let restored = if active.abort {
            None
        } else {
            active.selected.map(|s| s.tokens)
        };
        let mut transfer = active.transfer;
        long.healthy = false;
        let committed = (|| -> Result<()> {
            ensure!(
                long.memory.book().slot(active.slot)?.phase() == Phase::Restoring,
                "SSD restore reservation changed"
            );
            if let Some(n) = restored {
                let mut t = Transfer {
                    long,
                    stage: &mut self.disk.stage,
                    state: transfer.context("SSD restore transfer absent")?,
                    draft: &mut active.draft,
                };
                t.flush()?;
                transfer = Some(t.state);
                long.restore_speculation_at(active.slot, active.draft.as_ref(), n)?;
                let s = &mut long.sessions[active.slot];
                s.tokens.extend_from_slice(&active.request[..n]);
                s.pending = None;
                s.mixed = false;
                s.prompt_end = active.request.len();
                s.total_end = active.request.len() + active.output;
                s.background = active.background;
                unsafe {
                    s.position.set_i32(&long.engine.stream, n as i32)?;
                }
                long.engine.sync()?;
            } else {
                long.memory.release(&long.engine.stream, active.slot, 0)?;
                let s = &mut long.sessions[active.slot];
                s.tokens.clear();
                s.pending = None;
                s.mixed = false;
                s.prompt_end = 0;
                s.total_end = 0;
                unsafe {
                    s.position.set_i32(&long.engine.stream, 0)?;
                }
                long.engine.sync()?;
            }
            if long.latest_logits == Some(active.slot) {
                long.latest_logits = None;
            }
            Ok(())
        })();
        self.disk
            .require(long, 512, active.operation, committed.is_ok())?;
        committed?;
        // The destination stays Restoring until every rank has checked/drained.
        let activated = (|| -> Result<()> {
            if restored.is_some() {
                long.memory
                    .finish_restore(&long.engine.stream, active.slot)?;
            }
            // Restoring intentionally preserves the retained feature hold.
            // Normalize only after the destination has its final phase, then
            // agree success before making any owner runnable.
            long.sync_speculation_holds()
        })();
        self.disk
            .require(long, 514, active.operation, activated.is_ok())?;
        activated?;
        long.healthy = true;
        let file_error = active
            .done
            .as_ref()
            .and_then(|(_, r)| r.as_ref().err())
            .map(|e| format!("{e:#}"));
        if !active.abort {
            self.disk.store = Some(active.done.context("SSD returned catalog absent")?.0);
        }
        let state = transfer.unwrap_or(TransferState::new(active.slot, 0));
        self.disk.copy_calls += state.copies;
        self.disk.copy_bytes += state.bytes;
        Ok(Some(Report {
            outcome: if active.abort {
                "discarded"
            } else if restored.is_some() {
                "restored"
            } else {
                "miss"
            },
            restored_tokens: restored,
            file_error,
            slices: active.slices,
            batches: active.batches,
            copy_calls: state.copies,
            copy_bytes: state.bytes,
            boundaries: active.tick,
        }))
    }
    fn idle(&mut self) -> Result<&mut Disk> {
        ensure!(self.active.is_none(), "SSD reader owns catalog/staging");
        Ok(&mut self.disk)
    }
}

fn finish(reader: &mut Reader, long: &mut Long<'_>) -> Result<Report> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let tick = reader
            .active
            .as_ref()
            .context("reader fixture not active")?
            .tick;
        if let Some(r) = reader.step(
            long,
            false,
            !tick.is_multiple_of(long.engine.rank as u64 + 2),
            false,
        )? {
            return Ok(r);
        }
        ensure!(Instant::now() < deadline, "reader fixture deadline");
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
    let tokens = sets.last().context("read fixture")?;
    ensure!(tokens.len() == 16384, "read fixture size");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let control = Control::new(&long.engine.device)?;
    let mut stamp = [0; 64];
    stamp[..16].copy_from_slice(
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_le_bytes(),
    );
    let all = control.exchange(&long, &stamp)?;
    let run = u128::from_le_bytes(all[0][..16].try_into().unwrap());
    let path = PathBuf::from(format!("evidence/ssd-read-{run}-rank{rank}"));
    let mut reader = Reader::new(Disk::new(&mut long, &path, 256 << 20)?);
    long.start(tokens.len(), 4)?;
    let mut states = Vec::new();
    for n in [8192, 8256] {
        prefill_to(&mut long, &tokens[..n])?;
        states.push(long.state(n)?);
        ensure!(
            reader.idle()?.save(&mut long)? == Outcome::Saved,
            "read fixture save"
        );
    }
    let reference = finish_reference(&mut long, tokens)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    long.bind(1)?;
    ensure!(
        reader.start(&mut long, tokens.clone().into(), 4, true)?,
        "read admission"
    );
    ensure!(
        long.start(64, 4).is_err()
            && long.prefill(0, &[7; 64]).is_err()
            && long.cancel().is_err()
            && long.pause().is_err()
            && long.resume().is_err()
            && long.finish().is_err(),
        "partial destination runnable"
    );
    ensure!(
        long.healthy,
        "rejected partial-destination operation poisoned the owner before GPU work"
    );
    long.bind(0)?;
    ensure!(long.bind(1).is_err(), "partial destination bind allowed");
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        ensure!(
            reader.step(&mut long, false, true, false)?.is_none(),
            "restore finished before interleaving"
        );
        let enough = reader
            .active
            .as_ref()
            .unwrap()
            .transfer
            .is_some_and(|t| t.bytes >= 5 * 11 * (LATENT + POOLED));
        if control.agree(&long, 530, [0; 32], enough)? {
            break;
        }
        ensure!(Instant::now() < deadline, "read initial copy deadline");
    }
    let before = reader.active.as_ref().unwrap().transfer.unwrap().copies;
    let foreground_tokens = vec![45; 2177];
    long.start(foreground_tokens.len(), 4)?;
    prefill_to(&mut long, &foreground_tokens[..2176])?;
    ensure!(
        reader.step(&mut long, true, true, false)?.is_none(),
        "foreground completed suspended read"
    );
    let foreground = finish_reference(&mut long, &foreground_tokens)?;
    ensure!(
        reader.active.as_ref().unwrap().transfer.unwrap().copies == before,
        "foreground did not freeze H2D"
    );
    let restored = finish(&mut reader, &mut long)?;
    ensure!(
        restored.restored_tokens == Some(8256),
        "cooperative restore length"
    );
    long.bind(1)?;
    ensure!(
        long.state(8256)? == states[1] && finish_reference(&mut long, tokens)? == reference,
        "cooperative restore exact model"
    );
    long.bind(0)?;
    long.start(foreground_tokens.len(), 4)?;
    prefill_to(&mut long, &foreground_tokens[..2176])?;
    ensure!(
        finish_reference(&mut long, &foreground_tokens)? == foreground,
        "foreground changed during concurrent read"
    );
    // Longest strict prefix for a request ending at the later snapshot is the
    // full private-page-tail entry at 8192, not the equal-length entry at 8256.
    long.bind(3)?;
    ensure!(
        reader.start(&mut long, tokens[..8256].to_vec().into(), 4, false)?,
        "full tail admission"
    );
    long.bind(0)?;
    let full_tail = finish(&mut reader, &mut long)?;
    ensure!(full_tail.restored_tokens == Some(8192), "full tail lookup");
    long.bind(3)?;
    ensure!(long.state(8192)? == states[0], "full tail restore bytes");
    long.finish()?;
    // Cancel after earlier verified chunks have already reached the destination.
    long.bind(1)?;
    ensure!(
        reader.start(&mut long, tokens.clone().into(), 4, false)?,
        "cancel read admission"
    );
    long.bind(0)?;
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        ensure!(
            reader.step(&mut long, false, true, false)?.is_none(),
            "cancel fixture completed"
        );
        if control.agree(
            &long,
            531,
            [0; 32],
            reader
                .active
                .as_ref()
                .unwrap()
                .transfer
                .is_some_and(|t| t.bytes > 0),
        )? {
            break;
        }
        ensure!(Instant::now() < deadline, "cancel read copy deadline");
    }
    let first = reader.step(&mut long, true, false, rank == 0)?;
    let cancelled = if let Some(r) = first {
        r
    } else {
        finish(&mut reader, &mut long)?
    };
    ensure!(
        cancelled.outcome == "discarded"
            && reader.idle()?.store.is_none()
            && long.memory.book().slot(1)?.phase() == Phase::Idle
            && long.memory.book().slot(1)?.pages().is_empty()
            && long.sessions[1].tokens.is_empty(),
        "partial cancel not discarded"
    );
    reader.idle()?.reopen(&mut long)?;
    ensure!(
        reader
            .idle()?
            .store
            .as_ref()
            .context("read cancel reopen")?
            .len()
            == 2,
        "cancel removed immutable file"
    );
    // Batching may retain the next five-page group in the CPU exchange until
    // it is full. Damage a later page so at least two complete exchanges were
    // acknowledged and the first H2D group really drained before the fault.
    let (_, keys) = format::keys(&reader.disk.id.namespace, &tokens[..8256])?;
    let bad = path
        .join("pages")
        .join(format!("{}.page", format::hex(&keys[12])));
    let corrupt = if rank == 2 {
        recovery::flip_last(&bad)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 532, [0; 32], corrupt.is_ok())?,
        "reader corruption fixture"
    );
    corrupt?;
    long.bind(1)?;
    ensure!(
        reader.start(&mut long, tokens.clone().into(), 4, false)?,
        "bad read admission"
    );
    long.bind(0)?;
    let failed = finish(&mut reader, &mut long)?;
    ensure!(
        failed.outcome == "discarded"
            && reader.idle()?.store.is_none()
            && long.sessions[1].tokens.is_empty()
            && long.memory.book().slot(1)?.pages().is_empty(),
        "late read error exposed partial state"
    );
    ensure!(
        control.agree(
            &long,
            534,
            [0; 32],
            rank != 2 || failed.copy_bytes >= 5 * 11 * (LATENT + POOLED)
        )?,
        "late checksum did not follow H2D: {}",
        serde_json::to_string(&failed)?
    );
    long.bind(1)?;
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..8192])?;
    prefill_to(&mut long, &tokens[..8256])?;
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "read failure cold replay"
    );
    // Repair this isolated shared page only after rejection, then reconcile.
    let repair = if rank == 2 {
        recovery::flip_last(&bad)
    } else {
        Ok(())
    };
    ensure!(
        control.agree(&long, 533, [0; 32], repair.is_ok())?,
        "reader corruption cleanup"
    );
    repair?;
    reader.idle()?.reopen(&mut long)?;
    ensure!(
        reader
            .idle()?
            .store
            .as_ref()
            .context("reader reopen")?
            .len()
            == 1,
        "reader invalidation intersection"
    );
    long.bind(3)?;
    ensure!(
        reader.start(&mut long, tokens[..8256].to_vec().into(), 4, false)?,
        "retained read admission"
    );
    long.bind(0)?;
    let retained = finish(&mut reader, &mut long)?;
    ensure!(
        retained.restored_tokens == Some(8192),
        "retained prefix lost"
    );
    long.bind(3)?;
    ensure!(long.state(8192)? == states[0], "retained prefix changed");
    long.finish()?;
    long.bind(1)?;
    ensure!(
        reader.start(&mut long, vec![1; 400].into(), 4, false)?,
        "miss admission"
    );
    long.bind(0)?;
    let miss = finish(&mut reader, &mut long)?;
    ensure!(
        miss.outcome == "miss"
            && miss.copy_calls == 0
            && reader.idle()?.store.is_some()
            && long.memory.book().slot(1)?.phase() == Phase::Idle,
        "ordinary miss did not release reservation"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "reader changed paused model");
    long.resume()?;
    long.finish()?;
    long.memory.book().check()?;
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"snapshot_tokens":[8192,8256],"snapshot_states":states,"reference":reference,"foreground_reference":foreground,"cache_root":path,"namespace":format::hex(&reader.disk.id.namespace),"reserved_destination_rejects_execution_and_bind":true,"foreground_runs_while_h2d_suspended":true,"foreground_and_paused_state_exact":true,"partial_restore_cancelled_collectively":true,"late_checksum_discards_partial_restore":true,"failed_prefix_invalidated_on_reopen":true,"cold_fallback_exact":true,"full_and_partial_tails_exact":true,"ordinary_miss_releases_reservation":true,"restored":restored,"full_tail":full_tail,"cancelled":cancelled,"failed":failed,"retained":retained,"miss":miss,"pinned_stage_bytes":CHUNK,"codec_stage_bytes":CHUNK,"exchange_bytes":CHUNK,"catalog_lifecycle_still_synchronous":true,"native_priority_qualified":false}),
    )
}
