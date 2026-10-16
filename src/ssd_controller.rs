//! One exclusive catalog/staging owner across lifecycle, writer and reader jobs.
use super::super::store::Store as Ram;
use super::*;
use crate::ssd_lifecycle::{CatalogJob, MAX_KEYS};
use std::sync::Arc;

#[derive(serde::Serialize)]
pub(in super::super) struct OpenReport {
    enabled: bool,
    entries: usize,
    file_error: Option<String>,
    boundaries: u64,
}
struct Opening {
    disk: Disk,
    job: Option<CatalogJob>,
    done: Option<Result<Store>>,
    order: Option<Vec<Hash>>,
    local: BTreeSet<Hash>,
    keep: Vec<Hash>,
    operation: Hash,
    tick: u64,
    phase: u8,
    cursor: usize,
    count: usize,
    abort: bool,
}
impl Opening {
    fn start(mut disk: Disk, long: &mut Long<'_>, serial: u64) -> Result<Self> {
        disk.ready(long)?;
        let mut hash = Sha256::new();
        hash.update(b"ssd-catalog-open-v1");
        hash.update(disk.id.namespace);
        hash.update(serial.to_le_bytes());
        hash.update(disk.budget.to_le_bytes());
        let operation = hash.finalize().into();
        let free = long.engine.device.allocatable_memory();
        disk.require(long, 600, operation, free.is_ok())?;
        // New codec staging, catalog's existing 64 MiB metadata cap and bounded inventories.
        let fits = disk.gate(
            long,
            601,
            operation,
            free?.0 >= RESERVE + CHUNK + (65 << 20),
        )?;
        let job = if fits {
            Some(CatalogJob::open(
                disk.root.clone(),
                disk.id.clone(),
                disk.budget,
                disk.store.take(),
            ))
        } else {
            None
        };
        let started = disk.gate(
            long,
            602,
            operation,
            job.as_ref().is_some_and(Result::is_ok),
        )?;
        if let Some(Err(e)) = &job {
            eprintln!("SSD lifecycle start failed: {e:#}");
        }
        let mut job = job.and_then(Result::ok);
        if !started && let Some(job) = &mut job {
            job.cancel();
        }
        Ok(Self {
            disk,
            job,
            done: None,
            order: None,
            local: BTreeSet::new(),
            keep: Vec::new(),
            operation,
            tick: 0,
            phase: 0,
            cursor: 0,
            count: 0,
            abort: !started,
        })
    }
    fn step(
        &mut self,
        long: &mut Long<'_>,
        foreground: bool,
        cancel: bool,
    ) -> Result<Option<OpenReport>> {
        self.disk.ready(long)?;
        self.abort |= cancel;
        let mut fatal = false;
        if let Some(job) = &mut self.job {
            if self.abort {
                job.cancel();
            }
            if self.order.is_none() {
                match job.inventory() {
                    Ok(Some(order)) => self.order = Some(order),
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("SSD inventory protocol: {e:#}");
                        fatal = true;
                        job.cancel();
                    }
                }
            }
            match job.try_finish() {
                Ok(Some(done)) => {
                    self.abort |= done.is_err();
                    self.done = Some(done);
                    self.job = None;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("SSD lifecycle completion: {e:#}");
                    fatal = true;
                    job.cancel();
                }
            }
        }
        let mut record = [0; 64];
        record[..32].copy_from_slice(&self.operation);
        record[32..40].copy_from_slice(&self.tick.to_le_bytes());
        record[40..44].copy_from_slice(&603u32.to_le_bytes());
        record[44] = u8::from(foreground);
        record[45] = self.phase;
        record[46] = u8::from(self.abort);
        record[47] = u8::from(self.job.is_none());
        record[48] = u8::from(self.order.is_some());
        record[49] = u8::from(fatal);
        record[50..58]
            .copy_from_slice(&(self.order.as_ref().map_or(0, Vec::len) as u64).to_le_bytes());
        let all = self.disk.exchange(long, &record)?;
        if all.iter().any(|r| r[..46] != record[..46] || r[49] != 0) {
            if let Some(job) = &mut self.job {
                job.cancel();
            }
            self.disk.healthy = false;
            long.healthy = false;
            anyhow::bail!("SSD lifecycle control disagreement; owner poisoned");
        }
        self.tick = self
            .tick
            .checked_add(1)
            .context("SSD lifecycle counter exhausted")?;
        self.abort |= all.iter().any(|r| r[46] != 0);
        if self.abort
            && let Some(job) = &mut self.job
        {
            job.cancel();
        }
        // Polling cancellation/readiness is bounded; defer extra metadata collectives,
        // catalog publication/retirement and mutation dispatch during foreground holds.
        if foreground {
            return Ok(None);
        }
        if !self.abort {
            if self.phase == 0 && all.iter().all(|r| r[48] != 0) {
                self.count = u64::from_le_bytes(all[0][50..58].try_into().unwrap()) as usize;
                self.disk
                    .require(long, 604, self.operation, self.count <= MAX_KEYS)?;
                self.local = self.order.as_ref().unwrap().iter().copied().collect();
                self.phase = 1;
            } else if self.phase == 1 {
                if self.cursor < self.count {
                    let mut key = [0; 64];
                    if long.engine.rank == 0 {
                        key[..32].copy_from_slice(&self.order.as_ref().unwrap()[self.cursor]);
                    }
                    let keys = self.disk.exchange(long, &key)?;
                    let key: Hash = keys[0][..32].try_into().unwrap();
                    if self.disk.gate(long, 605, key, self.local.contains(&key))? {
                        self.keep.push(key);
                    }
                    self.cursor += 1;
                } else {
                    let mut hash = Sha256::new();
                    for key in &self.keep {
                        hash.update(key);
                    }
                    self.disk.require(long, 606, hash.finalize().into(), true)?;
                    let sent = self
                        .job
                        .as_mut()
                        .context("catalog ended before apply")?
                        .apply(std::mem::take(&mut self.keep));
                    self.disk.require(long, 607, self.operation, sent.is_ok())?;
                    sent?;
                    self.phase = 2;
                }
            }
        }
        if all.iter().any(|r| r[47] == 0) {
            return Ok(None);
        }
        if !self.abort {
            let valid = self.phase == 2 && self.done.as_ref().is_some_and(Result::is_ok);
            self.disk.require(long, 608, self.operation, valid)?;
            let digest = self.done.as_ref().unwrap().as_ref().unwrap().digest();
            self.disk.require(long, 609, digest, true)?;
        }
        let error = self
            .done
            .as_ref()
            .and_then(|r| r.as_ref().err())
            .map(|e| format!("{e:#}"));
        if self.abort {
            self.disk.store = None;
            self.done = None;
        } else {
            self.disk.store = Some(self.done.take().context("catalog completion missing")??);
        }
        Ok(Some(OpenReport {
            enabled: !self.abort,
            entries: self.disk.store.as_ref().map_or(0, Store::len),
            file_error: error,
            boundaries: self.tick,
        }))
    }
}

enum State {
    Idle(Box<Disk>),
    Opening(Box<Opening>),
    Writing(Box<cooperative::Writer>),
    Reading(Box<restore::Reader>),
    Transition,
}
pub(in super::super) struct Controller {
    state: State,
    serial: u64,
    file_wake: Option<crate::request_broker::FileWake>,
    // One bounded, read-only foreground lane. The original operation retains
    // its worker, partial destination, snapshot pins and exact copy position.
    spare: Option<Box<Disk>>,
    parked: Option<State>,
}
#[derive(serde::Serialize)]
#[serde(tag = "kind", content = "result")]
pub(in super::super) enum Completion {
    Open(OpenReport),
    Write(cooperative::Report),
    Read(restore::Report),
}
impl Controller {
    pub(in super::super) fn set_file_wake(&mut self, wake: crate::request_broker::FileWake) {
        self.file_wake = Some(wake);
    }
    pub(in super::super) fn new(long: &mut Long<'_>, root: &Path, budget: u64) -> Result<Self> {
        // Identity/executable verification and CUDA allocation are startup-only.
        // No catalog filesystem access occurs in prepare or this constructor.
        let disk = Disk::prepare(long, root, budget)?;
        let spare = if std::env::var("ALASKA_SSD_FOREGROUND_READ").as_deref() == Ok("1") {
            Some(Box::new(disk.read_lane(long)?))
        } else {
            None
        };
        let mut controller = Self {
            state: State::Idle(Box::new(disk)),
            serial: 0,
            file_wake: None,
            spare,
            parked: None,
        };
        controller.open(long)?;
        Ok(controller)
    }
    fn idle(&self) -> Result<&Disk> {
        match &self.state {
            State::Idle(d) => Ok(d),
            _ => anyhow::bail!("SSD controller already owns an operation"),
        }
    }
    fn take(&mut self) -> Result<Disk> {
        self.idle()?;
        ensure!(self.serial < u64::MAX, "SSD controller serial exhausted");
        self.serial += 1;
        match std::mem::replace(&mut self.state, State::Transition) {
            State::Idle(d) => Ok(*d),
            _ => unreachable!(),
        }
    }
    pub(in super::super) fn open(&mut self, long: &mut Long<'_>) -> Result<()> {
        long.ready()?;
        ensure!(!long.priority_hold, "new SSD job held by native priority");
        self.idle()?;
        if let Some(spare) = &mut self.spare {
            spare.store = None;
        }
        let disk = self.take()?;
        self.state = State::Opening(Box::new(Opening::start(disk, long, self.serial)?));
        Ok(())
    }
    pub(in super::super) fn write(
        &mut self,
        long: &mut Long<'_>,
        ram: &mut Ram,
        id: u64,
    ) -> Result<bool> {
        self.write_mode(long, ram, id, true)
    }
    pub(in super::super) fn write_mode(
        &mut self,
        long: &mut Long<'_>,
        ram: &mut Ram,
        id: u64,
        background: bool,
    ) -> Result<bool> {
        long.ready()?;
        ensure!(
            !long.priority_hold || !background,
            "new SSD job held by native priority"
        );
        // The previous request's optional foreground save must not hide older
        // completed SSD prefixes from the next foreground request.
        self.refresh_view(long)?;
        let disk = self.take()?;
        let mut writer = cooperative::Writer::at(disk, self.serial);
        let result = writer.start_mode(long, ram, id, background);
        self.state = if matches!(result, Ok(false)) {
            State::Idle(Box::new(writer.into_disk()?))
        } else {
            State::Writing(Box::new(writer))
        };
        result
    }
    pub(in super::super) fn read(
        &mut self,
        long: &mut Long<'_>,
        tokens: Arc<[i32]>,
        output: usize,
        background: bool,
    ) -> Result<bool> {
        long.ready()?;
        ensure!(
            !long.priority_hold || !background,
            "new SSD job held by native priority"
        );
        if background {
            self.refresh_view(long)?;
        }
        let disk = if !background && self.busy() {
            ensure!(
                self.can_foreground_read(),
                "foreground SSD lane unavailable"
            );
            ensure!(
                long.priority_hold,
                "native priority must be held before lending the SSD lane"
            );
            ensure!(self.serial < u64::MAX, "SSD controller serial exhausted");
            // A foreground-classified writer may have resumed after the native
            // boundary. Explicitly quiesce it before retaining its exact state.
            if let State::Writing(writer) = &mut self.state {
                writer.pause_for_read(long)?;
            }
            self.serial += 1;
            self.parked = Some(std::mem::replace(&mut self.state, State::Transition));
            *self.spare.take().unwrap()
        } else {
            self.take()?
        };
        let mut reader = restore::Reader::at(disk, self.serial);
        reader.file_wake = self.file_wake.clone();
        let result = reader.start(long, tokens, output, background);
        self.state = if matches!(result, Ok(false)) {
            let disk = Box::new(reader.into_disk()?);
            if let Some(parked) = self.parked.take() {
                self.spare = Some(disk);
                parked
            } else {
                State::Idle(disk)
            }
        } else {
            State::Reading(Box::new(reader))
        };
        result
    }
    pub(in super::super) fn step(
        &mut self,
        long: &mut Long<'_>,
        ram: &mut Ram,
        foreground: bool,
        allow_copy: bool,
        cancel: bool,
    ) -> Result<Option<Completion>> {
        // Foreground prefix restoration belongs to foreground work. Only an
        // explicitly foreground reader may copy through the native hold.
        let background = !self.foreground();
        let foreground = foreground || (long.priority_hold && background);
        let completion = match &mut self.state {
            State::Opening(job) => job.step(long, foreground, cancel)?.map(Completion::Open),
            State::Writing(job) => job
                .step(long, ram, foreground, allow_copy, cancel)?
                .map(Completion::Write),
            State::Reading(job) => job
                .step(long, foreground, allow_copy, cancel)?
                .map(Completion::Read),
            State::Idle(_) => return Ok(None),
            State::Transition => anyhow::bail!("SSD controller poisoned during transition"),
        };
        if completion.is_some() {
            let disk = match std::mem::replace(&mut self.state, State::Transition) {
                State::Opening(job) => job.disk,
                State::Writing(job) => job.into_disk()?,
                State::Reading(job) => job.into_disk()?,
                _ => unreachable!(),
            };
            self.state = if let Some(parked) = self.parked.take() {
                self.spare = Some(Box::new(disk));
                parked
            } else {
                if let Some(spare) = &mut self.spare {
                    spare.store = None;
                }
                State::Idle(Box::new(disk))
            };
        }
        Ok(completion)
    }
    pub(in super::super) fn busy(&self) -> bool {
        !matches!(self.state, State::Idle(_))
    }
    pub(in super::super) fn foreground(&self) -> bool {
        matches!(&self.state, State::Reading(r) if !r.background())
            || matches!(&self.state, State::Writing(w) if !w.background())
    }
    pub(in super::super) fn enabled(&self) -> bool {
        matches!(&self.state, State::Idle(d) if d.store.is_some())
    }
    pub(in super::super) fn hint(&self, tokens: &[i32]) -> Result<Option<usize>> {
        match &self.state {
            State::Idle(d) => d
                .store
                .as_ref()
                .map(|s| s.hint(tokens))
                .transpose()
                .map(Option::flatten),
            _ => Ok(None),
        }
    }
    pub(in super::super) fn foreground_hint(&self, tokens: &[i32]) -> Result<Option<usize>> {
        if self.can_foreground_read() {
            self.spare
                .as_ref()
                .unwrap()
                .store
                .as_ref()
                .unwrap()
                .hint(tokens)
        } else {
            self.hint(tokens)
        }
    }
    pub(in super::super) fn parked(&self) -> bool {
        self.parked.is_some()
    }
    fn can_foreground_read(&self) -> bool {
        self.parked.is_none()
            && (matches!(self.state, State::Writing(_))
                || matches!(&self.state, State::Reading(r) if r.background()))
            && self.spare.as_ref().is_some_and(|d| d.store.is_some())
    }
    fn refresh_view(&mut self, long: &mut Long<'_>) -> Result<()> {
        let Some(spare) = &mut self.spare else {
            return Ok(());
        };
        let State::Idle(disk) = &mut self.state else {
            anyhow::bail!("read view requires idle catalog");
        };
        // At most the original catalog's 64 MiB metadata plus one codec buffer.
        // A stale hint is never a hit: the worker checks exact tokens and all
        // immutable file checksums. Concurrent eviction becomes a safe miss.
        spare.store = None;
        let free = long.engine.device.allocatable_memory();
        if !disk.gate(
            long,
            610,
            disk.id.namespace,
            free.as_ref()
                .is_ok_and(|(n, _)| *n >= RESERVE + CHUNK + (65 << 20)),
        )? {
            return Ok(());
        }
        let view = disk.store.as_ref().map(Store::reader_view).transpose();
        let valid = disk.gate(long, 611, disk.id.namespace, view.is_ok())?;
        if valid {
            spare.store = view?;
        }
        Ok(())
    }
    pub(in super::super) fn namespace(&self) -> Result<Hash> {
        Ok(self.idle()?.id.namespace)
    }
    /// Common scheduling state only: local readiness and byte counts may differ.
    pub(in super::super) fn digest(&self) -> Hash {
        let mut h = Sha256::new();
        h.update(b"ssd-controller-owner-v1");
        h.update(self.serial.to_le_bytes());
        h.update([u8::from(self.parked.is_some())]);
        if let Some(spare) = &self.spare {
            h.update(spare.digest());
        }
        h.update([u8::from(self.foreground())]);
        h.update([match self.state {
            State::Idle(_) => 0,
            State::Opening(_) => 1,
            State::Writing(_) => 2,
            State::Reading(_) => 3,
            State::Transition => 4,
        }]);
        if let State::Idle(d) = &self.state {
            h.update(d.digest());
            h.update([u8::from(d.store.is_some())]);
        }
        h.finalize().into()
    }
    pub(in super::super) fn progress(&self) -> (usize, usize) {
        match &self.state {
            State::Writing(w) => w.progress(),
            State::Reading(r) => r.progress(),
            _ => (0, 0),
        }
    }
}

#[path = "ssd_controller_tests.rs"]
mod tests;
pub use tests::run;
