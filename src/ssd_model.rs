//! Synchronous pure-prefix SSD adapter. A bad file causes an all-rank cold miss;
//! device/transport or control disagreement poisons the model. No server leases.
use super::speculation::Saved as DraftSaved;
use super::{
    Engine, Long, prefix,
    store::{Control, finish_reference, prefill_to},
};
use crate::{
    cache_pages::{CONTEXT, MAX_OUTPUT, PAGE, Phase, spans},
    cuda::{Buffer, Device, Pinned},
    prepared::Prepared,
    ssd_format::{self as format, CHUNK, Hash, Identity, Role},
    ssd_store::{Outcome, Source, Store},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

const LATENT: usize = PAGE * 528;
const POOLED: usize = PAGE / 4 * 256;
const GROUP: usize = CHUNK / (LATENT + POOLED) / 11; // five full page sets per pinned buffer
const RESERVE: usize = 2 << 30;
// One compact retained drafter companion while a source/destination is pinned.
const DRAFT_RESTORE_BYTES: usize = 2047 * 5120;
#[path = "ssd_recovery.rs"]
mod recovery;
pub use recovery::run as run_recovery;
#[path = "ssd_cooperative.rs"]
mod cooperative;
pub use cooperative::run as run_cooperative;
#[path = "ssd_restore.rs"]
mod restore;
pub use restore::run as run_restore;
pub(in crate::model) use restore::run_companion_probe as run_companion_read_probe;
#[path = "ssd_controller.rs"]
mod controller;
pub use controller::run as run_controller;
pub(super) use controller::{Completion, Controller};
fn policy() -> Result<Hash> {
    // Conservatively invalidate on any executable rebuild. The binary contains
    // host arithmetic/scheduling, native cubin pins and the verified manifest.
    let mut f = File::open(std::env::current_exe()?)?;
    let mut h = Sha256::new();
    let mut chunk = [0; 65536];
    loop {
        let n = f.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        h.update(&chunk[..n]);
    }
    h.update(include_bytes!("../native/kernel-manifest.json"));
    Ok(h.finalize().into())
}
fn request_key(long: &Long<'_>, request: &[i32], output: usize, background: bool) -> Hash {
    let mut h = Sha256::new();
    h.update(b"ssd-request-v1");
    for v in [long.active, request.len(), output, usize::from(background)] {
        h.update((v as u64).to_le_bytes());
    }
    for t in request {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}
fn layer_buffer_at<'a>(
    long: &'a Long<'_>,
    slot: usize,
    n: usize,
    role: Role,
    draft: Option<&'a DraftSaved>,
) -> Result<(&'a Buffer, usize, usize)> {
    let session = &long.sessions[slot];
    let result = match role {
        Role::Conv(l) | Role::Recurrent(l) => {
            ensure!(l < 45 && l % 4 != 3, "SSD KDA layer role");
            let a = &session.recurrent[l - l / 4];
            let b = if matches!(role, Role::Conv(_)) {
                a.conv.as_ref()
            } else {
                a.states[n % 2].as_ref()
            };
            (b, 0, b.bytes)
        }
        Role::IndexKeys(l) | Role::IndexGates(l) => {
            ensure!(l < 45 && l % 4 == 3, "SSD index layer role");
            let a = &session.index[l / 4];
            let b = if matches!(role, Role::IndexKeys(_)) {
                &a.keys
            } else {
                &a.gates
            };
            (b, 0, b.bytes)
        }
        Role::TailLatent(f) | Role::TailPool(f) => {
            ensure!(f < 11 && n > 0, "SSD tail role");
            let pooled = matches!(role, Role::TailPool(_));
            let page = (n - 1) / PAGE;
            let tail = n - page * PAGE;
            let physical = *long
                .memory
                .book()
                .slot(slot)?
                .pages()
                .get(page)
                .context("SSD tail mapping")? as usize;
            let family = &long.memory.families[f];
            if pooled {
                (family.pooled.as_ref(), physical * POOLED, tail / 4 * 256)
            } else {
                (family.latent.as_ref(), physical * LATENT, tail * 528)
            }
        }
        Role::DraftKv(index) => {
            let buffer = draft
                .context("SSD missing drafter companion")?
                .buffers()
                .get(index)
                .context("SSD drafter tensor")?;
            (buffer, 0, buffer.bytes)
        }
        Role::DraftMetadata | Role::Tokens => anyhow::bail!("host segment has no live GPU buffer"),
    };
    Ok(result)
}
fn validate_layout(long: &Long<'_>, n: usize, id: &Identity) -> Result<()> {
    validate_layout_at(long, long.active, n, id)
}
fn validate_layout_at(long: &Long<'_>, slot: usize, n: usize, id: &Identity) -> Result<()> {
    ensure!(
        prefix::ARITHMETIC == format::ARITHMETIC,
        "SSD arithmetic identity"
    );
    ensure!(
        long.memory.book().slot(slot)?.pages().len() >= n.div_ceil(PAGE),
        "SSD live mapping incomplete"
    );
    ensure!(
        long.speculation_identity().is_some() == id.speculation.is_some(),
        "SSD drafter layout mode"
    );
    for s in id.segments(n)? {
        if !matches!(
            s.role,
            Role::Tokens | Role::DraftMetadata | Role::DraftKv(_)
        ) {
            ensure!(
                layer_buffer_at(long, slot, n, s.role, None)?.2 as u64 == s.bytes,
                "SSD snapshot role size"
            );
        }
    }
    Ok(())
}
/// A group is family-major in pinned memory so contiguous GPU pages need one
/// DMA per family/span, instead of one DMA per individual page. Disk remains
/// page-major. Only ordinary host copies transpose this bounded staging group.
struct Transfer<'a, 'b> {
    long: &'a Long<'b>,
    stage: &'a mut Pinned,
    state: TransferState,
    draft: &'a mut Option<DraftSaved>,
}
#[derive(Clone, Copy)]
struct TransferState {
    slot: usize,
    n: usize,
    group: Option<(usize, usize)>,
    pending: bool,
    cursor: usize,
    copies: usize,
    bytes: usize,
    gpu_error: bool,
    metadata_invalid: bool,
}
fn group_offset(count: usize, page: usize, family: usize, pooled: bool) -> usize {
    family * count * (LATENT + POOLED)
        + if pooled {
            count * LATENT + page * POOLED
        } else {
            page * LATENT
        }
}
impl TransferState {
    fn new(slot: usize, n: usize) -> Self {
        Self {
            slot,
            n,
            group: None,
            pending: false,
            cursor: 0,
            copies: 0,
            bytes: 0,
            gpu_error: false,
            metadata_invalid: false,
        }
    }
}
impl Transfer<'_, '_> {
    fn new<'a, 'b>(
        long: &'a Long<'b>,
        stage: &'a mut Pinned,
        n: usize,
        draft: &'a mut Option<DraftSaved>,
    ) -> Transfer<'a, 'b> {
        Transfer {
            long,
            stage,
            state: TransferState::new(long.active, n),
            draft,
        }
    }
    fn pages(&mut self, start: usize, count: usize, upload: bool) -> Result<()> {
        let result = (|| -> Result<()> {
            let mapping = self.long.memory.book().slot(self.state.slot)?.pages();
            ensure!(
                start + count <= mapping.len() && count <= GROUP,
                "SSD staging group bounds"
            );
            let s = &self.long.engine.stream;
            for (logical, physical, len) in spans(&mapping[start..start + count]) {
                for (family, buffers) in self.long.memory.families.iter().enumerate() {
                    for (pooled, buffer, stride) in [
                        (false, buffers.latent.as_ref(), LATENT),
                        (true, buffers.pooled.as_ref(), POOLED),
                    ] {
                        let host = group_offset(count, logical, family, pooled);
                        let gpu = physical as usize * stride;
                        unsafe {
                            if upload {
                                self.stage.upload_to(s, buffer, gpu, host, len * stride)?;
                            } else {
                                self.stage
                                    .download_from(s, host, buffer, gpu, len * stride)?;
                            }
                        }
                        self.state.copies += 1;
                        self.state.bytes += len * stride;
                    }
                }
            }
            s.sync()?;
            Ok(())
        })();
        if result.is_err() {
            self.state.gpu_error = true;
        }
        result
    }
    fn source(&mut self, part: Source, offset: u64, buf: &mut [u8]) -> Result<()> {
        let offset = usize::try_from(offset)?;
        match part {
            Source::Page {
                logical,
                family,
                pooled,
            } => {
                let shared = (self.state.n - 1) / PAGE;
                ensure!(logical < shared && family < 11, "SSD page source");
                if !self
                    .state
                    .group
                    .is_some_and(|(a, n)| logical >= a && logical < a + n)
                {
                    let count = GROUP.min(shared - logical);
                    self.pages(logical, count, false)?;
                    self.state.group = Some((logical, count));
                }
                let (start, count) = self.state.group.unwrap();
                let stride = if pooled { POOLED } else { LATENT };
                ensure!(
                    offset <= stride && buf.len() <= stride - offset,
                    "SSD page slice"
                );
                let at = group_offset(count, logical - start, family, pooled) + offset;
                buf.copy_from_slice(&self.stage.bytes_mut()[at..at + buf.len()]);
            }
            Source::Layer(role) => {
                self.state.group = None;
                if role == Role::DraftMetadata {
                    let metadata = self
                        .draft
                        .as_ref()
                        .context("SSD missing source drafter")?
                        .metadata();
                    ensure!(
                        offset <= metadata.len() && buf.len() <= metadata.len() - offset,
                        "SSD drafter metadata source slice"
                    );
                    buf.copy_from_slice(&metadata[offset..offset + buf.len()]);
                    return Ok(());
                }
                let result = (|| -> Result<()> {
                    let (buffer, start, len) = layer_buffer_at(
                        self.long,
                        self.state.slot,
                        self.state.n,
                        role,
                        self.draft.as_ref(),
                    )?;
                    ensure!(
                        offset <= len && buf.len() <= len - offset && buf.len() <= CHUNK,
                        "SSD private slice"
                    );
                    unsafe {
                        self.stage.download_from(
                            &self.long.engine.stream,
                            0,
                            buffer,
                            start + offset,
                            buf.len(),
                        )?;
                    }
                    self.long.engine.stream.sync()?;
                    buf.copy_from_slice(&self.stage.bytes_mut()[..buf.len()]);
                    self.state.copies += 1;
                    self.state.bytes += buf.len();
                    Ok(())
                })();
                if result.is_err() {
                    self.state.gpu_error = true;
                }
                result?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        if self.state.pending {
            let (start, count) = self.state.group.context("SSD missing staging group")?;
            ensure!(
                self.state.cursor == count * (LATENT + POOLED) * 11,
                "SSD incomplete staged page group"
            );
            self.pages(start, count, true)?;
            self.state.pending = false;
        }
        self.state.group = None;
        self.state.cursor = 0;
        Ok(())
    }
    fn sink(&mut self, part: Source, offset: u64, buf: &[u8]) -> Result<()> {
        let offset = usize::try_from(offset)?;
        match part {
            Source::Page {
                logical,
                family,
                pooled,
            } => {
                let shared = (self.state.n - 1) / PAGE;
                ensure!(logical < shared && family < 11, "SSD page target");
                if !self
                    .state
                    .group
                    .is_some_and(|(a, n)| logical >= a && logical < a + n)
                {
                    self.flush()?;
                    self.state.group = Some((logical, GROUP.min(shared - logical)));
                    self.state.pending = true;
                }
                let (start, count) = self.state.group.unwrap();
                let stride = if pooled { POOLED } else { LATENT };
                ensure!(
                    offset <= stride && buf.len() <= stride - offset,
                    "SSD target slice"
                );
                let linear = (logical - start) * (LATENT + POOLED) * 11
                    + family * (LATENT + POOLED)
                    + if pooled { LATENT + offset } else { offset };
                ensure!(linear == self.state.cursor, "SSD page payload out of order");
                let at = group_offset(count, logical - start, family, pooled) + offset;
                self.stage.bytes_mut()[at..at + buf.len()].copy_from_slice(buf);
                self.state.cursor += buf.len();
            }
            Source::Layer(role) => {
                self.flush()?;
                if role == Role::DraftMetadata {
                    let checked = (|| -> Result<&[u8; 64]> {
                        ensure!(
                            offset == 0 && buf.len() == 64 && self.draft.is_none(),
                            "SSD drafter metadata order/shape"
                        );
                        let metadata: &[u8; 64] = buf.try_into()?;
                        self.long
                            .validate_speculation_metadata(metadata, self.state.n)?;
                        Ok(metadata)
                    })();
                    if checked.is_err() {
                        self.state.metadata_invalid = true;
                    }
                    let metadata = checked?;
                    let allocated = self
                        .long
                        .allocate_speculation_restore(metadata, self.state.n);
                    if allocated.is_err() {
                        self.state.gpu_error = true;
                    }
                    *self.draft = Some(allocated?);
                    return Ok(());
                }
                let result = (|| -> Result<()> {
                    let (buffer, start, len) = layer_buffer_at(
                        self.long,
                        self.state.slot,
                        self.state.n,
                        role,
                        self.draft.as_ref(),
                    )?;
                    ensure!(
                        offset <= len && buf.len() <= len - offset && buf.len() <= CHUNK,
                        "SSD target private slice"
                    );
                    self.stage.bytes_mut()[..buf.len()].copy_from_slice(buf);
                    unsafe {
                        self.stage.upload_to(
                            &self.long.engine.stream,
                            buffer,
                            start + offset,
                            0,
                            buf.len(),
                        )?;
                    }
                    self.long.engine.stream.sync()?;
                    self.state.copies += 1;
                    self.state.bytes += buf.len();
                    Ok(())
                })();
                if result.is_err() {
                    self.state.gpu_error = true;
                }
                result?;
            }
        }
        Ok(())
    }
}

pub(super) struct Disk {
    device: Rc<Device>,
    stream: usize,
    model: String,
    speculation: Option<Hash>,
    control: Control,
    id: Identity,
    root: PathBuf,
    budget: u64,
    store: Option<Store>,
    stage: Pinned,
    copy_stream: Option<crate::cuda::CopyStream>,
    healthy: bool,
    copy_calls: usize,
    copy_bytes: usize,
}
impl Drop for Disk {
    fn drop(&mut self) {
        let _ = self.device.sync();
    }
}
impl Disk {
    fn exchange(&mut self, long: &mut Long<'_>, record: &[u8; 64]) -> Result<Vec<[u8; 64]>> {
        let result = self.control.exchange(long, record);
        if result.is_err() {
            self.healthy = false;
            long.healthy = false;
        }
        result
    }
    fn gate(&mut self, long: &mut Long<'_>, tag: u32, digest: Hash, ready: bool) -> Result<bool> {
        let result = self.control.agree(long, tag, digest, ready);
        if result.is_err() {
            self.healthy = false;
            long.healthy = false;
        }
        result
    }
    fn require(&mut self, long: &mut Long<'_>, tag: u32, digest: Hash, ready: bool) -> Result<()> {
        if !self.gate(long, tag, digest, ready)? {
            self.healthy = false;
            long.healthy = false;
            anyhow::bail!("SSD collective device/validation failure");
        }
        Ok(())
    }
    fn ready(&self, long: &Long<'_>) -> Result<()> {
        long.ready()?;
        ensure!(
            self.healthy
                && Rc::ptr_eq(&self.device, &long.engine.device)
                && self.stream == long.engine.stream.raw() as usize
                && self.model == long.model_identity
                && self.speculation == long.speculation_identity(),
            "SSD owner/health mismatch"
        );
        Ok(())
    }
    pub(super) fn new(long: &mut Long<'_>, root: &Path, budget: u64) -> Result<Self> {
        let mut disk = Self::prepare(long, root, budget)?;
        disk.reopen(long)?;
        Ok(disk)
    }
    fn prepare(long: &mut Long<'_>, root: &Path, budget: u64) -> Result<Self> {
        long.ready()?;
        long.healthy = false;
        let control = Control::new(&long.engine.device)?;
        let init = (|| -> Result<(Hash, [u8; 64])> {
            let policy = policy()?;
            let mut record = [0; 64];
            record[..32].copy_from_slice(
                &format::unhex(&long.model_identity).context("SSD model identity")?,
            );
            if let Some(draft) = long.speculation_identity() {
                let mut h = Sha256::new();
                h.update(b"target-and-drafter-v1");
                h.update(&record[..32]);
                h.update(draft);
                record[..32].copy_from_slice(&h.finalize());
            }
            record[32..40].copy_from_slice(&(long.engine.rank as u64).to_le_bytes());
            record[40..48].copy_from_slice(
                &(long.sessions[long.active].index[0].keys.bytes as u64 / 256).to_le_bytes(),
            );
            record[48..56].copy_from_slice(&budget.to_le_bytes());
            record[56..64].copy_from_slice(
                &(1 + u64::from(long.speculation_identity().is_some())).to_le_bytes(),
            );
            Ok((policy, record))
        })();
        if !control.agree(
            long,
            100,
            init.as_ref().map(|r| r.0).unwrap_or([0; 32]),
            init.is_ok(),
        )? {
            long.healthy = false;
            anyhow::bail!("SSD policy preflight");
        }
        let (policy, record) = init?;
        let all = control.exchange(long, &record)?;
        ensure!(
            all.iter().enumerate().all(
                |(rank, r)| r[32..40] == (rank as u64).to_le_bytes() && r[40..] == record[40..]
            ),
            "SSD shard/config agreement"
        );
        let shards: Vec<Hash> = all.iter().map(|r| r[..32].try_into().unwrap()).collect();
        let mut id = Identity::new(
            shards.try_into().unwrap(),
            policy,
            long.engine.rank as u32,
            long.sessions[long.active].index[0].keys.bytes / 256,
        )?;
        if long.speculation_identity().is_some() {
            // id.namespace already includes every rank's target/drafter pair.
            id = id.clone().with_speculation(id.namespace);
        }
        let free = long.engine.device.allocatable_memory();
        let draft_bytes = if long.speculation_identity().is_some() {
            DRAFT_RESTORE_BYTES
        } else {
            0
        };
        let independent_copy = std::env::var("ALASKA_SSD_COPY_WORKER").as_deref() == Ok("1");
        let mut copy_policy = Sha256::new();
        copy_policy.update(id.namespace);
        copy_policy.update([u8::from(independent_copy)]);
        copy_policy.update([u8::from(
            std::env::var("ALASKA_SSD_READ_BATCH").as_deref() == Ok("1"),
        )]);
        copy_policy.update([u8::from(
            std::env::var("ALASKA_SSD_FOREGROUND_READ").as_deref() == Ok("1"),
        )]);
        ensure!(
            control.agree(long, 103, copy_policy.finalize().into(), true)?,
            "SSD copy policy disagreement"
        );
        let fits = free.as_ref().is_ok_and(|(free, _)| {
            *free >= RESERVE + (2 + usize::from(independent_copy)) * CHUNK + draft_bytes
        });
        ensure!(
            control.agree(long, 101, id.namespace, fits)?,
            "SSD staging reserve"
        );
        let stage = Pinned::new(&long.engine.device, CHUNK);
        ensure!(
            control.agree(long, 102, id.namespace, stage.is_ok())?,
            "SSD pinned allocation failed collectively"
        );
        let copy_stream = independent_copy
            .then(|| crate::cuda::CopyStream::new(&long.engine.device, CHUNK))
            .transpose();
        ensure!(
            control.agree(long, 104, id.namespace, copy_stream.is_ok())?,
            "SSD copy stream allocation failed collectively"
        );
        let s = Self {
            device: long.engine.device.clone(),
            stream: long.engine.stream.raw() as usize,
            model: long.model_identity.clone(),
            speculation: long.speculation_identity(),
            control,
            id,
            root: root.to_owned(),
            budget,
            store: None,
            stage: stage?,
            copy_stream: copy_stream?,
            healthy: true,
            copy_calls: 0,
            copy_bytes: 0,
        };
        long.healthy = true;
        Ok(s)
    }
    fn digest(&self) -> Hash {
        self.store.as_ref().map(Store::digest).unwrap_or([0; 32])
    }
    fn read_lane(&self, long: &mut Long<'_>) -> Result<Self> {
        self.ready(long)?;
        let control = Control::new(&long.engine.device)?;
        let free = long.engine.device.allocatable_memory();
        ensure!(
            control.agree(
                long,
                105,
                self.id.namespace,
                free.as_ref().is_ok_and(
                    |(n, _)| *n >= RESERVE + 3 * CHUNK + (65 << 20) + DRAFT_RESTORE_BYTES
                )
            )?,
            "foreground SSD staging reserve"
        );
        let stage = Pinned::new(&long.engine.device, CHUNK);
        ensure!(
            control.agree(long, 106, self.id.namespace, stage.is_ok())?,
            "foreground SSD pinned allocation failed collectively"
        );
        Ok(Self {
            device: self.device.clone(),
            stream: self.stream,
            model: self.model.clone(),
            speculation: self.speculation,
            control,
            id: self.id.clone(),
            root: self.root.clone(),
            budget: self.budget,
            store: None,
            stage: stage?,
            copy_stream: None,
            healthy: true,
            copy_calls: 0,
            copy_bytes: 0,
        })
    }
    fn reconcile(&mut self, long: &mut Long<'_>) -> Result<()> {
        if self.store.is_none() {
            return Ok(());
        }
        let order = self.store.as_ref().unwrap().order();
        let local: BTreeSet<_> = order.iter().copied().collect();
        let mut record = [0; 64];
        record[..8].copy_from_slice(&(order.len() as u64).to_le_bytes());
        let all = self.exchange(long, &record)?;
        let count = u64::from_le_bytes(all[0][..8].try_into().unwrap()) as usize;
        ensure!(count <= 4096, "SSD leader inventory bound");
        let mut keep = Vec::new();
        let mut leader_keys = order.iter();
        for _ in 0..count {
            let mut r = [0; 64];
            if long.engine.rank == 0 {
                r[..32].copy_from_slice(leader_keys.next().context("SSD leader key count")?);
            }
            let all = self.exchange(long, &r)?;
            let key: Hash = all[0][..32].try_into().unwrap();
            if self.gate(long, 104, key, local.contains(&key))? {
                keep.push(key);
            }
        }
        let result = self.store.as_mut().unwrap().retain_order(&keep);
        if !self.gate(long, 105, self.id.namespace, result.is_ok())? {
            self.store = None;
            return Ok(());
        }
        result?;
        self.require(long, 106, self.digest(), true)
    }
    pub(super) fn save(&mut self, long: &mut Long<'_>) -> Result<Outcome> {
        self.ready(long)?;
        if self.store.is_none() {
            return Ok(Outcome::Skipped);
        }
        let n = long.sessions[long.active].tokens.len();
        let valid = (|| -> Result<()> {
            prefix::Key::new(
                &self.model,
                &long.sessions[long.active].tokens,
                long.sessions[long.active].mixed,
            )?;
            validate_layout(long, n, &self.id)?;
            Ok(())
        })();
        let digest = request_key(long, &long.sessions[long.active].tokens, 0, false);
        self.require(long, 110, digest, valid.is_ok())?;
        self.require(long, 111, self.digest(), true)?;
        long.healthy = false;
        let captured = long.capture_speculation();
        self.require(long, 116, digest, captured.is_ok())?;
        let mut draft = captured?;
        long.healthy = true;
        let mut transfer = Transfer::new(long, &mut self.stage, n, &mut draft);
        let result = self.store.as_mut().unwrap().save(
            &long.sessions[long.active].tokens,
            false,
            |part, offset, buf| transfer.source(part, offset, buf),
        );
        let gpu_error = transfer.state.gpu_error;
        self.copy_calls += transfer.state.copies;
        self.copy_bytes += transfer.state.bytes;
        if let Err(e) = &result {
            eprintln!("SSD save did not complete: {e:#}");
        }
        self.require(long, 112, digest, !gpu_error)?;
        if !self.gate(long, 113, digest, result.is_ok())? {
            self.store = None;
            return Ok(Outcome::Skipped);
        }
        let outcome = result?;
        let mut r = [0; 64];
        r[0] = match outcome {
            Outcome::Saved => 1,
            Outcome::Duplicate => 2,
            Outcome::Skipped => 3,
        };
        let all = self.exchange(long, &r)?;
        self.require(long, 115, digest, all.iter().all(|p| p[0] == r[0]))?;
        self.require(long, 114, self.digest(), true)?;
        Ok(outcome)
    }
    pub(super) fn restore(
        &mut self,
        long: &mut Long<'_>,
        request: &[i32],
        output: usize,
        background: bool,
        preferred: Option<Hash>,
    ) -> Result<Option<usize>> {
        self.ready(long)?;
        if self.store.is_none() {
            return Ok(None);
        }
        let digest = request_key(long, request, output, background);
        let valid = long.memory.book().slot(long.active)?.phase() == Phase::Idle
            && output > 0
            && output <= MAX_OUTPUT
            && request.len() <= CONTEXT - output
            && request
                .iter()
                .all(|&t| crate::vision_input::prompt_token(t));
        self.require(long, 120, digest, valid)?;
        let chosen = if preferred.is_some() {
            Ok(preferred)
        } else {
            self.store.as_mut().unwrap().lookup(request)
        };
        if !self.gate(long, 121, digest, chosen.is_ok())? {
            self.store = None;
            return Ok(None);
        }
        let chosen = chosen?;
        let mut r = [0; 64];
        if let Some(key) = chosen {
            r[..32].copy_from_slice(&key);
            r[32] = 1;
        }
        let all = self.exchange(long, &r)?;
        if !all.iter().all(|a| a[..33] == r[..33]) {
            self.reconcile(long)?;
            return Ok(None);
        }
        let Some(key) = chosen else {
            self.reconcile(long)?;
            return Ok(None);
        };
        let n = self.store.as_mut().unwrap().prefix_len(&key, request);
        if !self.gate(long, 122, key, n.is_ok())? {
            self.drop_bad(long, &key)?;
            return Ok(None);
        }
        let n = n?;
        let keep = n.min(long.memory.book().slot(long.active)?.pages().len() * PAGE);
        let plan = long.admission(request.len(), output, keep, background);
        self.require(
            long,
            123,
            plan.as_ref().map(|p| p.digest()).unwrap_or([0; 32]),
            plan.is_ok(),
        )?;
        long.healthy = false;
        let prepared = (|| -> Result<()> {
            long.commit_admission(&plan?, keep)?;
            long.memory.map(&long.engine.stream, long.active, n)?;
            validate_layout(long, n, &self.id)
        })();
        self.require(long, 124, key, prepared.is_ok())?;
        prepared?;
        let mut draft = None;
        let mut transfer = Transfer::new(long, &mut self.stage, n, &mut draft);
        let result = self
            .store
            .as_mut()
            .unwrap()
            .load(&key, request, |part, offset, buf| {
                transfer.sink(part, offset, buf)
            })
            .and_then(|_| transfer.flush());
        let gpu_error = transfer.state.gpu_error;
        self.copy_calls += transfer.state.copies;
        self.copy_bytes += transfer.state.bytes;
        if let Err(e) = &result {
            eprintln!("SSD restore discarded: {e:#}");
        }
        self.require(long, 125, key, !gpu_error)?;
        if !self.gate(long, 126, key, result.is_ok())? {
            // No inference can observe partially copied state. Free only this
            // destination; start_mode will zero recurrent/ring state for cold replay.
            let reset = (|| -> Result<()> {
                long.engine.sync()?;
                long.memory.release(&long.engine.stream, long.active, 0)?;
                let s = &mut long.sessions[long.active];
                s.tokens.clear();
                s.pending = None;
                s.mixed = false;
                s.prompt_end = 0;
                s.total_end = 0;
                unsafe {
                    s.position.set_i32(&long.engine.stream, 0)?;
                }
                long.engine.sync()?;
                long.latest_logits = None;
                long.sync_speculation_holds()
            })();
            self.require(long, 127, key, reset.is_ok())?;
            reset?;
            long.healthy = true;
            self.drop_bad(long, &key)?;
            return Ok(None);
        }
        result?;
        let positioned = long
            .restore_speculation_at(long.active, draft.as_ref(), n)
            .and_then(|()| unsafe {
                long.sessions[long.active]
                    .position
                    .set_i32(&long.engine.stream, n as i32)
            })
            .and_then(|_| long.engine.sync());
        self.require(long, 128, key, positioned.is_ok())?;
        positioned?;
        let s = &mut long.sessions[long.active];
        s.tokens.clear();
        s.tokens.extend_from_slice(&request[..n]);
        s.pending = None;
        s.mixed = false;
        s.prompt_end = request.len();
        s.total_end = request.len() + output;
        s.background = background;
        long.latest_logits = None;
        let activated = long.sync_speculation_holds();
        self.require(long, 129, self.digest(), activated.is_ok())?;
        activated?;
        long.healthy = true;
        Ok(Some(n))
    }
    fn drop_bad(&mut self, long: &mut Long<'_>, key: &Hash) -> Result<()> {
        let result = self.store.as_mut().unwrap().forget(key);
        if !self.gate(long, 130, *key, result.is_ok())? {
            self.store = None;
            return Ok(());
        }
        result?;
        self.reconcile(long)
    }
    fn reopen(&mut self, long: &mut Long<'_>) -> Result<()> {
        self.ready(long)?;
        self.store = None;
        let opened = Store::open(&self.root, self.id.clone(), self.budget);
        if self.gate(long, 131, self.id.namespace, opened.is_ok())? {
            self.store = Some(opened?);
            self.reconcile(long)?;
        }
        Ok(())
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
    let tokens = sets.last().context("SSD model fixture")?;
    ensure!((4096..=16384).contains(&tokens.len()), "SSD fixture length");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    let path = PathBuf::from(format!(
        "evidence/model-ssd-{}-{}-rank{}",
        if use_roce { "roce" } else { "nccl" },
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        rank
    ));
    let cut = if tokens.len() >= 16384 { 8256 } else { 2112 };
    let page_cut = cut - 64;
    let branch_cut = cut + 64;
    let budget = if cut > 4096 { 256 << 20 } else { 192 << 20 };
    let mut disk = Disk::new(&mut long, &path, budget)?;
    ensure!(disk.store.is_some(), "SSD qualification storage disabled");
    long.start(tokens.len(), 4)?;
    prefill_to(&mut long, &tokens[..page_cut])?;
    let page_state = long.state(page_cut)?;
    let before = disk.copy_calls;
    ensure!(
        disk.save(&mut long)? == Outcome::Saved,
        "SSD full-page save"
    );
    let initial_copies = disk.copy_calls - before;
    ensure!(
        initial_copies < (page_cut - 1) / PAGE * 22 + 112,
        "SSD coalesced copies"
    );
    ensure!(
        long.state(page_cut)? == page_state,
        "SSD staging changed source"
    );
    let page_key = format::keys(&disk.id.namespace, &tokens[..page_cut])?.0;
    prefill_to(&mut long, &tokens[..cut])?;
    let prefix_state = long.state(cut)?;
    ensure!(disk.save(&mut long)? == Outcome::Saved, "SSD partial save");
    ensure!(disk.save(&mut long)? == Outcome::Duplicate, "SSD duplicate");
    let first_key = format::keys(&disk.id.namespace, &tokens[..cut])?.0;
    let reference = finish_reference(&mut long, tokens)?;
    long.bind(2)?;
    long.start(64, 4)?;
    long.prefill(0, &[9; 64])?;
    let paused = long.state(64)?;
    long.pause()?;
    long.bind(1)?;
    ensure!(
        disk.restore(&mut long, tokens, 4, false, Some(page_key))? == Some(page_cut)
            && long.state(page_cut)? == page_state,
        "SSD full tail state"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "SSD full tail model"
    );
    let mut branches = Vec::new();
    let mut reports = Vec::new();
    for branch in 0..3 {
        long.bind(1)?;
        long.start(64, 4)?;
        long.prefill(0, &[8; 64])?;
        long.finish()?;
        let mut input = tokens[..branch_cut + 1].to_vec();
        input[cut] = 10 + branch;
        ensure!(
            disk.restore(&mut long, &input, 4, false, Some(first_key))? == Some(cut)
                && long.state(cut)? == prefix_state,
            "SSD shared prefix state"
        );
        prefill_to(&mut long, &input[..branch_cut])?;
        let before = disk.copy_bytes;
        ensure!(disk.save(&mut long)? == Outcome::Saved, "SSD branch save");
        let state = long.state(branch_cut)?;
        let key = format::keys(&disk.id.namespace, &input[..branch_cut])?.0;
        let bytes = disk.copy_bytes - before;
        let private = format::body_bytes(branch_cut, &disk.id)? - branch_cut as u64 * 4;
        ensure!(
            bytes as u64 == private,
            "SSD shared pages were copied again"
        );
        reports.push(serde_json::json!({"copied_bytes":bytes,"charged_bytes":disk.store.as_ref().unwrap().bytes()}));
        let expected = finish_reference(&mut long, &input)?;
        branches.push((key, input, state, expected));
    }
    disk.reopen(&mut long)?;
    for (key, input, state, expected) in &branches {
        long.bind(1)?;
        ensure!(
            disk.restore(&mut long, input, 4, false, Some(*key))? == Some(branch_cut)
                && long.state(branch_cut)? == *state,
            "SSD restart branch state"
        );
        ensure!(
            finish_reference(&mut long, input)? == *expected,
            "SSD restart branch model"
        );
    }
    long.bind(0)?;
    ensure!(
        disk.restore(&mut long, tokens, 4, false, Some(first_key))? == Some(cut),
        "SSD original prefix retained"
    );
    ensure!(
        finish_reference(&mut long, tokens)? == reference,
        "SSD original model"
    );
    let independent = long.state(tokens.len() + 3)?;
    let mut retained = None;
    for fill in 0..5 {
        long.bind(1)?;
        let input = vec![30 + fill; 2177];
        long.start(input.len(), 4)?;
        prefill_to(&mut long, &input[..2176])?;
        ensure!(disk.save(&mut long)? == Outcome::Saved, "SSD fill save");
        let key = format::keys(&disk.id.namespace, &input[..2176])?.0;
        let expected = finish_reference(&mut long, &input)?;
        retained = Some((key, input, expected));
    }
    ensure!(
        !disk.store.as_ref().unwrap().order().contains(&first_key),
        "SSD eviction fixture"
    );
    let (key, input, expected) = retained.unwrap();
    disk.reopen(&mut long)?;
    long.bind(3)?;
    ensure!(
        disk.restore(&mut long, &input, 4, false, None)? == Some(2176),
        "SSD restart longest lookup"
    );
    ensure!(
        finish_reference(&mut long, &input)? == expected,
        "SSD restart after eviction model"
    );
    // Corrupt only rank 3, in a late private chunk after pages and earlier
    // recurrent fields have already been delivered to the target slot.
    if rank == 3 {
        let mut f = OpenOptions::new().read(true).write(true).open(
            disk.root
                .join("entries")
                .join(format!("{}.entry", format::hex(&key))),
        )?;
        f.seek(SeekFrom::End(-1))?;
        let mut b = [0];
        f.read_exact(&mut b)?;
        b[0] ^= 1;
        f.seek(SeekFrom::End(-1))?;
        f.write_all(&b)?;
        f.sync_all()?;
    }
    disk.require(&mut long, 190, key, true)?;
    long.bind(1)?;
    ensure!(
        disk.restore(&mut long, &input, 4, false, Some(key))?
            .is_none(),
        "one-rank corruption fallback"
    );
    ensure!(
        long.healthy
            && long.sessions[1].tokens.is_empty()
            && long.memory.book().slot(1)?.phase() == Phase::Idle
            && long.memory.book().slot(1)?.pages().is_empty(),
        "partial restore not discarded"
    );
    long.start(input.len(), 4)?;
    ensure!(
        finish_reference(&mut long, &input)? == expected,
        "cold fallback model"
    );
    long.start(input.len(), 4)?;
    prefill_to(&mut long, &input[..2176])?;
    disk.save(&mut long)?;
    long.finish()?;
    let page = format::keys(&disk.id.namespace, &input[..2176])?.1[0];
    if rank == 1 {
        fs::remove_file(
            disk.root
                .join("pages")
                .join(format!("{}.page", format::hex(&page))),
        )?;
    }
    disk.require(&mut long, 191, key, true)?;
    disk.reopen(&mut long)?;
    ensure!(
        !disk.store.as_ref().unwrap().order().contains(&key),
        "four-rank restart intersection"
    );
    long.bind(0)?;
    ensure!(
        long.state(tokens.len() + 3)? == independent,
        "SSD eviction/fallback changed copied live state"
    );
    long.bind(2)?;
    ensure!(long.state(64)? == paused, "SSD changed paused owner");
    long.cancel()?;
    long.bind(1)?;
    let mappings = long.memory.book().slot(1)?.pages().to_vec();
    let cursor = long.sessions[1].tokens.clone();
    let digest = disk.digest();
    let used = disk.store.as_ref().unwrap().bytes();
    let mut divergent = input.clone();
    if rank == 3 {
        divergent[0] = 77;
    }
    ensure!(
        disk.restore(&mut long, &divergent, 4, false, None).is_err()
            && !long.healthy
            && !disk.healthy
            && disk.digest() == digest
            && long.memory.book().slot(1)?.pages() == mappings
            && long.sessions[1].tokens == cursor,
        "SSD request disagreement before mutation"
    );
    Ok(
        serde_json::json!({"rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"reused_tokens":cut,"page_boundary_tokens":page_cut,"branch_snapshot_tokens":branch_cut,
        "cache_directory":path,"namespace":format::hex(&disk.id.namespace),"budget_bytes":budget,"charged_bytes":used,"pinned_stage_bytes":CHUNK,"file_stage_bytes":CHUNK,"initial_copy_calls":initial_copies,
        "copy_calls":disk.copy_calls,"copied_bytes":disk.copy_bytes,"branch_saves":reports,"reference":reference,
        "full_and_partial_tails_exact":true,"shared_page_bytes_not_recopied":true,"restart_branches_exact":true,"eviction_and_reload_exact":true,"one_rank_late_corruption_cold_fallback_exact":true,
        "missing_page_restart_intersection":true,"paused_and_independent_live_owners_unchanged":true,"request_disagreement_before_mutation":true,
        "scope":"Synchronous pure-prefix live GPU/SSD adapter; no cooperative server scheduling or in-flight request persistence."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staging_layout_has_no_holes_or_overlap() {
        assert_eq!(GROUP, 5);
        for count in 1..=GROUP {
            let mut intervals = Vec::new();
            for page in 0..count {
                for family in 0..11 {
                    for pooled in [false, true] {
                        let start = group_offset(count, page, family, pooled);
                        intervals.push((start, start + if pooled { POOLED } else { LATENT }));
                    }
                }
            }
            intervals.sort();
            assert_eq!(intervals[0].0, 0);
            assert!(intervals.windows(2).all(|p| p[0].1 == p[1].0));
            assert_eq!(intervals.last().unwrap().1, count * (LATENT + POOLED) * 11);
            assert!(intervals.last().unwrap().1 <= CHUNK);
        }
    }
}
