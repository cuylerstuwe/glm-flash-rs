//! Cooperative SSD policy on common request-owner boundaries.
use super::super::disk::{Completion as DiskCompletion, Controller};
use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
struct ReadOwner {
    id: u64,
    background: bool,
    cancelled: bool,
}
pub(super) struct RequestDisk {
    pub(super) controller: Controller,
    // Only the active writer pins RAM. Pending optional saves may coalesce/evict.
    pending: Option<(u64, bool)>,
    foreground_writes: bool,
    reading: Option<ReadOwner>,
    parked_reading: Option<ReadOwner>,
    parked_watch: Option<crate::cache_watch::Watch>,
    pub(super) stopping: bool,
    started: u64,
    dropped: u64,
    failures: u32,
    retry_after: Duration,
    records: VecDeque<serde_json::Value>,
    watch: Option<crate::cache_watch::Watch>,
    next_watch: Duration,
    abort_job: bool,
    stalled_jobs: u64,
}
impl RequestDisk {
    pub(super) fn telemetry(&self) -> serde_json::Value {
        let progress = self.controller.progress();
        serde_json::json!({"enabled":self.controller.enabled(),"busy":self.controller.busy(),
            "pending":self.pending.is_some(),"reading":self.reading.map(|r|r.id),"stopping":self.stopping,
            "progress_bytes":progress.0,"progress_steps":progress.1,"foreground_job":self.controller.foreground(),
            "started":self.started,"coalesced":self.dropped,"consecutive_failures":self.failures,
            "stalled_jobs":self.stalled_jobs,"last_completion":self.records.back()})
    }
    pub(super) fn offer(&mut self, id: Option<u64>, background: bool) {
        if !self.stopping && id.is_some() {
            self.pending = id.map(|id| (id, background || !self.foreground_writes));
        }
    }
    pub(super) fn hint(&self, tokens: &[i32], background: bool) -> Result<Option<usize>> {
        if self.stopping {
            Ok(None)
        } else if !background && !self.abort_job && !self.reading.is_some_and(|r| r.cancelled) {
            self.controller.foreground_hint(tokens)
        } else {
            self.controller.hint(tokens)
        }
    }
    pub(super) fn cancel_restore(&mut self, id: u64) -> Result<()> {
        let r = if self.reading.is_some_and(|r| r.id == id) {
            self.reading.as_mut()
        } else {
            self.parked_reading.as_mut()
        }
        .context("missing SSD request owner")?;
        ensure!(r.id == id, "wrong SSD cancellation owner");
        r.cancelled = true;
        Ok(())
    }
    pub(super) fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"request-disk-read-write-v2");
        h.update(self.controller.digest());
        h.update(self.pending.map_or(0, |p| p.0).to_le_bytes());
        h.update([
            u8::from(self.pending.is_some_and(|p| p.1)),
            u8::from(self.foreground_writes),
        ]);
        h.update([u8::from(self.stopping)]);
        h.update(self.started.to_le_bytes());
        h.update(self.dropped.to_le_bytes());
        h.update(self.failures.to_le_bytes());
        h.update(self.next_watch.as_nanos().to_le_bytes());
        h.update([u8::from(self.abort_job)]);
        h.update(self.stalled_jobs.to_le_bytes());
        h.update(self.retry_after.as_nanos().to_le_bytes());
        h.update(self.reading.map_or(0, |r| r.id).to_le_bytes());
        h.update(self.parked_reading.map_or(0, |r| r.id).to_le_bytes());
        h.update([u8::from(self.parked_reading.is_some_and(|r| r.cancelled))]);
        h.update([
            u8::from(self.reading.is_some_and(|r| r.background)),
            u8::from(self.reading.is_some_and(|r| r.cancelled)),
        ]);
        h.finalize().into()
    }
    pub(super) fn wait(&self, now: Duration, held: bool) -> Duration {
        let allowed = !held || self.controller.foreground() || self.pending.is_some_and(|p| !p.1);
        if allowed
            && (self.controller.busy() || (self.controller.enabled() && self.pending.is_some()))
        {
            Duration::from_millis(2)
        } else if !held && !self.stopping && !self.controller.enabled() {
            self.retry_after
                .saturating_sub(now)
                .min(Duration::from_secs(1))
        } else {
            Duration::from_secs(1)
        }
    }
}
impl Scheduler {
    pub(super) fn install_disk(
        &mut self,
        long: &mut Long<'_>,
        root: &Path,
        budget: u64,
    ) -> Result<()> {
        ensure!(
            self.queue.count() == 0 && self.disk.is_none() && self.cache.is_some(),
            "disk installation requires an unused RAM-backed owner"
        );
        self.disk = Some(RequestDisk {
            controller: Controller::new(long, root, budget)?,
            pending: None,
            foreground_writes: std::env::var("ALASKA_SSD_COPY_WORKER").as_deref() == Ok("1"),
            reading: None,
            parked_reading: None,
            parked_watch: None,
            stopping: false,
            started: 0,
            dropped: 0,
            failures: 0,
            retry_after: Duration::ZERO,
            records: VecDeque::new(),
            watch: None,
            next_watch: Duration::ZERO,
            abort_job: false,
            stalled_jobs: 0,
        });
        Ok(())
    }
    pub(super) fn start_restore(
        &mut self,
        long: &mut Long<'_>,
        id: u64,
        slot: usize,
    ) -> Result<bool> {
        let r = self.queue.request(id)?;
        let disk = self.disk.as_mut().context("missing restore controller")?;
        ensure!(
            (disk.reading.is_none()
                || (!r.background && disk.reading.is_some_and(|r| r.background)))
                && !disk.stopping,
            "SSD reader already owned"
        );
        let parked_progress = disk.controller.progress();
        let started = disk
            .controller
            .read(long, r.tokens.clone(), r.output, r.background)?;
        if started {
            if disk.controller.parked() {
                ensure!(
                    disk.parked_reading.is_none() && disk.parked_watch.is_none(),
                    "nested SSD foreground lane"
                );
                // An optional save from an older foreground request was
                // allowed before parking. Exclude the entire lane loan from
                // its progress/cancellation clock, as for a held background job.
                if let Some(watch) = &mut disk.watch {
                    watch.observe(self.stamp, false, parked_progress)?;
                }
                disk.parked_reading = disk.reading.take();
                disk.parked_watch = disk.watch.take();
                disk.next_watch = Duration::ZERO;
            }
            disk.reading = Some(ReadOwner {
                id,
                background: r.background,
                cancelled: false,
            });
            self.queue.reserve_restore(id, slot, self.stamp)?;
        }
        Ok(started)
    }
    pub(super) fn disk_step(
        &mut self,
        long: &mut Long<'_>,
        stop: bool,
        report: &mut Report,
    ) -> Result<bool> {
        let Some(disk) = &mut self.disk else {
            return Ok(true);
        };
        if stop {
            disk.stopping = true;
            disk.pending = None;
        }
        let ram = self.cache.as_mut().context("disk lost RAM owner")?;
        let mut restored = None;
        if disk.controller.busy() {
            let allowed = !long.priority_hold || disk.controller.foreground();
            let watch = disk.watch.get_or_insert_with(Default::default);
            watch.observe(self.stamp, allowed, disk.controller.progress())?;
            if disk.stopping || disk.abort_job || disk.reading.is_some_and(|r| r.cancelled) {
                watch.cancel();
            }
            // Progress is rank-local; timeout and failure decisions must be
            // common. One bounded check per second, not per copied slice.
            if allowed && self.stamp >= disk.next_watch {
                let stalled = watch.stalled(Duration::from_secs(120));
                let expired = watch.expired(Duration::from_secs(30));
                let mut record = [0; 64];
                record[..8].copy_from_slice(b"ALSTAL01");
                record[8..40].copy_from_slice(&disk.digest());
                record[40] = u8::from(stalled);
                record[41] = u8::from(expired);
                let all = self.wire.control.exchange(long, &record)?;
                ensure!(
                    all.iter()
                        .all(|r| r[..40] == record[..40] && r[40] <= 1 && r[41] <= 1),
                    "SSD watchdog ownership disagreed"
                );
                ensure!(
                    all.iter().all(|r| r[41] == 0),
                    "SSD worker did not retire after cancellation; supervisor recovery required"
                );
                if all.iter().any(|r| r[40] != 0) && !disk.abort_job {
                    disk.abort_job = true;
                    disk.stalled_jobs += 1;
                    disk.watch.as_mut().unwrap().cancel();
                }
                disk.next_watch = self.stamp + Duration::from_secs(1);
            }
            let cancel =
                disk.stopping || disk.abort_job || disk.reading.is_some_and(|r| r.cancelled);
            let was_parked = disk.controller.parked();
            if let Some(completion) = disk.controller.step(long, ram, false, true, cancel)? {
                disk.watch = None;
                disk.next_watch = Duration::ZERO;
                disk.abort_job = false;
                if let DiskCompletion::Read(r) = &completion {
                    restored = Some((
                        disk.reading
                            .take()
                            .context("unowned SSD read completion")?
                            .id,
                        r.restored_tokens,
                    ));
                }
                if was_parked {
                    disk.reading = disk.parked_reading.take();
                    disk.watch = disk.parked_watch.take();
                }
                if disk.records.len() == 16 {
                    disk.records.pop_front();
                }
                disk.records.push_back(serde_json::to_value(completion)?);
                if was_parked {
                    // The mutable background catalog remains owned by its
                    // original worker; returning to it is not an open failure.
                } else if disk.controller.enabled() {
                    disk.failures = 0;
                } else if !disk.stopping {
                    disk.retry_after =
                        self.stamp + Duration::from_secs(1u64 << disk.failures.min(6));
                    disk.failures = disk.failures.saturating_add(1);
                }
            }
        } else if !disk.stopping {
            if !long.priority_hold && !disk.controller.enabled() && self.stamp >= disk.retry_after {
                disk.controller.open(long)?;
            } else if disk.controller.enabled()
                && (!long.priority_hold || disk.pending.is_some_and(|p| !p.1))
                && !(self.queue.free_slot().is_some() && self.queue.queued(false).is_some())
                && let Some((id, background)) = disk.pending.take()
            {
                // Source availability and admission are common even if a local
                // snapshot has disappeared. Do not enter unmatched collectives.
                if self
                    .wire
                    .control
                    .agree(long, 831, disk.digest(), ram.contains(id))?
                {
                    disk.started +=
                        u64::from(disk.controller.write_mode(long, ram, id, background)?);
                } else {
                    disk.dropped += 1;
                }
            }
        }
        let idle = !disk.controller.busy();
        if let Some((id, cached)) = restored {
            let r = self.queue.request(id)?;
            let slot = r.slot.context("restored request lost destination")?;
            if let Some(reason) = r.cancellation() {
                ensure!(cached.is_none(), "cancelled SSD load committed state");
                self.queue.finish_restore(id, 0)?;
                self.queue.finish(id, reason, self.stamp)?;
                self.output_streams.remove(&id);
                if let Some(speculation) = &mut self.speculation {
                    speculation.remove(id);
                }
                report.completed = Some(id);
            } else {
                if cached.is_none() {
                    long.bind(slot)?;
                    let cold = long.start_mode(r.tokens.len(), r.output, r.background);
                    long.agree_speculation_work(
                        &self.wire.control,
                        832,
                        self.queue.digest(),
                        cold,
                    )?;
                }
                self.queue.finish_restore(id, cached.unwrap_or(0))?;
                report.admitted = Some(id);
            }
            ensure!(
                self.wire
                    .control
                    .agree(long, 833, self.queue.digest(), true)?,
                "SSD request commit disagreed"
            );
        }
        Ok(idle)
    }
}
#[path = "request_disk_tests.rs"]
mod tests;
pub use tests::{run, run_restore};
