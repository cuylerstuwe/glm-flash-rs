//! Immutable, accounted RAM snapshot export. No CUDA allocation leaves its owner.
use super::*;
use crate::{
    cuda::Pinned,
    ssd_format::{self as format, CHUNK, Role},
    ssd_store::Source,
};
use anyhow::Context;
use std::sync::Arc;

/// Only validated addresses cross threads; the counted lease remains on the
/// model owner. The copy gate closes/drains before either RAM or Long teardown.
pub(in super::super) struct CopyPlan {
    pages: Vec<usize>,
    slabs: std::collections::BTreeMap<usize, Vec<(u64, u64)>>,
    layers: Vec<(Role, u64, usize)>,
    metadata: Vec<u8>,
    group: Option<(usize, usize)>,
}
impl CopyPlan {
    pub(in super::super) fn fill(
        &mut self,
        stream: &mut crate::cuda::CopyStream,
        gate: &crate::copy_gate::Gate,
        part: Source,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        use std::sync::atomic::Ordering;
        let _permit = gate.enter()?;
        let offset = usize::try_from(offset)?;
        let mut copies = Vec::with_capacity(22 * GROUP);
        let host = match part {
            Source::Page {
                logical,
                family,
                pooled,
            } => {
                ensure!(
                    logical < self.pages.len() && family < 11,
                    "async export page bounds"
                );
                if !self
                    .group
                    .is_some_and(|(start, count)| (start..start + count).contains(&logical))
                {
                    let count = GROUP.min(self.pages.len() - logical);
                    let mut cursor = 0;
                    while cursor < count {
                        let physical = self.pages[logical + cursor];
                        let mut run = 1;
                        while cursor + run < count
                            && self.pages[logical + cursor + run] == physical + run
                            && (physical + run) / EXTENT == physical / EXTENT
                        {
                            run += 1;
                        }
                        let slab = self
                            .slabs
                            .get(&(physical / EXTENT))
                            .context("async export slab")?;
                        for (f, &(latent, pool)) in slab.iter().enumerate() {
                            for (pooled, ptr, stride) in
                                [(false, latent, LATENT), (true, pool, POOLED)]
                            {
                                copies.push((
                                    at(count, cursor, f, pooled),
                                    ptr + (physical % EXTENT * stride) as u64,
                                    run * stride,
                                ));
                            }
                        }
                        cursor += run;
                    }
                    self.group = Some((logical, count));
                }
                let stride = if pooled { POOLED } else { LATENT };
                ensure!(
                    offset <= stride && buf.len() <= stride - offset,
                    "async export page slice"
                );
                let (start, count) = self.group.unwrap();
                at(count, logical - start, family, pooled) + offset
            }
            Source::Layer(role) => {
                self.group = None;
                if role == Role::DraftMetadata {
                    ensure!(
                        offset <= self.metadata.len() && buf.len() <= self.metadata.len() - offset,
                        "async export metadata slice"
                    );
                    buf.copy_from_slice(&self.metadata[offset..offset + buf.len()]);
                    return Ok(());
                }
                let &(_, ptr, len) = self
                    .layers
                    .iter()
                    .find(|(r, _, _)| *r == role)
                    .context("async export role")?;
                ensure!(
                    buf.len() <= CHUNK && offset <= len && buf.len() <= len - offset,
                    "async export private slice"
                );
                copies.push((0, ptr + offset as u64, buf.len()));
                0
            }
        };
        if !copies.is_empty() {
            // SAFETY: the plan was constructed from validated pinned buffers;
            // gate cancellation/drain precedes source destruction on every path.
            if let Err(e) = unsafe { stream.batch(&copies) } {
                gate.fatal.store(true, Ordering::Release);
                return Err(e);
            }
            gate.calls.fetch_add(copies.len(), Ordering::Relaxed);
            gate.bytes
                .fetch_add(copies.iter().map(|c| c.2).sum::<usize>(), Ordering::Relaxed);
        }
        buf.copy_from_slice(&stream.bytes()[host..host + buf.len()]);
        gate.slices.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

const GROUP: usize = CHUNK / (LATENT + POOLED) / 11;
pub(in super::super) struct Export {
    lease: Lease,
    pub(in super::super) tokens: Arc<[i32]>,
    pages: Vec<usize>,
    group: Option<(usize, usize)>,
    pub(in super::super) copies: usize,
    pub(in super::super) bytes: usize,
}
fn layer(snapshot: &Snapshot, role: Role) -> Result<&Buffer> {
    match role {
        Role::Conv(l) | Role::Recurrent(l) => {
            let Some(SavedLayer::Kda { conv, active }) = snapshot.layers.get(l) else {
                anyhow::bail!("RAM export KDA role");
            };
            Ok(if matches!(role, Role::Conv(_)) {
                conv
            } else {
                active
            })
        }
        Role::IndexKeys(l) | Role::IndexGates(l) => {
            let Some(SavedLayer::Dsa { keys, gates }) = snapshot.layers.get(l) else {
                anyhow::bail!("RAM export index role");
            };
            Ok(if matches!(role, Role::IndexKeys(_)) {
                keys
            } else {
                gates
            })
        }
        Role::TailLatent(f) | Role::TailPool(f) => {
            let family = snapshot.tail.get(f).context("RAM export tail role")?;
            Ok(if matches!(role, Role::TailLatent(_)) {
                &family.latent
            } else {
                &family.pooled
            })
        }
        Role::DraftKv(index) => snapshot
            .draft
            .as_ref()
            .context("RAM export missing drafter")?
            .buffers()
            .get(index)
            .context("RAM export drafter tensor"),
        Role::DraftMetadata | Role::Tokens => {
            anyhow::bail!("RAM export host role has no GPU buffer")
        }
    }
}
fn at(count: usize, page: usize, family: usize, pooled: bool) -> usize {
    family * count * (LATENT + POOLED)
        + if pooled {
            count * LATENT + page * POOLED
        } else {
            page * LATENT
        }
}
impl Store {
    pub(in super::super) fn copy_plan(
        &mut self,
        long: &mut Long<'_>,
        source: &Export,
        ring_rows: usize,
        gate: &Arc<crate::copy_gate::Gate>,
    ) -> Result<CopyPlan> {
        self.ready(long)?;
        self.index.validate_lease(&source.lease)?;
        let snapshot = self
            .snapshots
            .get(&source.lease.entry())
            .context("async export source")?;
        let mut layers = Vec::new();
        for segment in format::segments_with_speculation(
            source.tokens.len(),
            ring_rows,
            self.speculation.is_some(),
        )? {
            if segment.role != Role::Tokens && segment.role != Role::DraftMetadata {
                let buffer = layer(snapshot, segment.role)?;
                ensure!(
                    buffer.bytes as u64 == segment.bytes,
                    "async export source size"
                );
                layers.push((segment.role, buffer.ptr, buffer.bytes));
            }
        }
        let mut slabs = std::collections::BTreeMap::new();
        for &page in &source.pages {
            let id = page / EXTENT;
            if let std::collections::btree_map::Entry::Vacant(e) = slabs.entry(id) {
                e.insert(
                    self.slabs
                        .get(&id)
                        .context("async export source slab")?
                        .iter()
                        .map(|f| (f.latent.ptr, f.pooled.ptr))
                        .collect(),
                );
            }
        }
        // Snapshot-producing work and all map validation precede worker admission.
        long.engine.sync()?;
        ensure!(
            self.copy_gate
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .is_none(),
            "RAM copy worker already registered"
        );
        self.copy_gate = Some(Arc::downgrade(gate));
        long.copy_gate = Some(Arc::downgrade(gate));
        long.engine.device.register_copy_gate(gate)?;
        Ok(CopyPlan {
            pages: source.pages.clone(),
            slabs,
            layers,
            metadata: snapshot
                .draft
                .as_ref()
                .map_or_else(Vec::new, |d| d.metadata().to_vec()),
            group: None,
        })
    }
    pub(in super::super) fn export(
        &mut self,
        long: &mut Long<'_>,
        id: u64,
        ring_rows: usize,
    ) -> Result<Option<Export>> {
        self.ready(long)?;
        let prepared = (|| -> Result<_> {
            let (tokens, pages) = self.index.entry(id)?;
            let snapshot = self.snapshots.get(&id).context("RAM export source")?;
            ensure!(
                snapshot.key.tokens == tokens && pages.len() == (tokens.len() - 1) / PAGE,
                "RAM export mapping"
            );
            ensure!(
                snapshot.draft.is_some() == self.speculation.is_some(),
                "RAM export drafter mode"
            );
            for segment in format::segments_with_speculation(
                tokens.len(),
                ring_rows,
                self.speculation.is_some(),
            )? {
                if segment.role != Role::Tokens && segment.role != Role::DraftMetadata {
                    ensure!(
                        layer(snapshot, segment.role)?.bytes as u64 == segment.bytes,
                        "RAM export role size"
                    );
                }
            }
            for &page in &pages {
                ensure!(
                    self.slabs.contains_key(&(page / EXTENT)),
                    "RAM export slab missing"
                );
            }
            Ok((tokens, pages))
        })();
        let mut h = Sha256::new();
        h.update(self.index.digest());
        h.update(id.to_le_bytes());
        h.update((ring_rows as u64).to_le_bytes());
        self.healthy = false;
        long.healthy = false;
        ensure!(
            self.control
                .agree(long, 20, h.finalize().into(), prepared.is_ok())?,
            "RAM export validation failed collectively"
        );
        let (tokens, pages) = prepared?;
        self.healthy = true;
        long.healthy = true;
        Ok(self.pin(long, id)?.map(|lease| Export {
            lease,
            tokens,
            pages,
            group: None,
            copies: 0,
            bytes: 0,
        }))
    }
    pub(in super::super) fn release_export(
        &mut self,
        long: &mut Long<'_>,
        source: &Export,
    ) -> Result<()> {
        self.release(long, &source.lease)
    }
    pub(in super::super) fn copy_export(
        &mut self,
        long: &mut Long<'_>,
        source: &mut Export,
        stage: &mut Pinned,
        part: Source,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        self.ready(long)?;
        let result = (|| -> Result<()> {
            self.index.validate_lease(&source.lease)?;
            let offset = usize::try_from(offset)?;
            let stream = &long.engine.stream;
            match part {
                Source::Page {
                    logical,
                    family,
                    pooled,
                } => {
                    ensure!(
                        logical < source.pages.len() && family < 11,
                        "RAM export page bounds"
                    );
                    if !source
                        .group
                        .is_some_and(|(start, count)| (start..start + count).contains(&logical))
                    {
                        let count = GROUP.min(source.pages.len() - logical);
                        let mut cursor = 0;
                        while cursor < count {
                            let physical = source.pages[logical + cursor];
                            let mut run = 1;
                            while cursor + run < count
                                && source.pages[logical + cursor + run] == physical + run
                                && (physical + run) / EXTENT == physical / EXTENT
                            {
                                run += 1;
                            }
                            let slab = self
                                .slabs
                                .get(&(physical / EXTENT))
                                .context("RAM export pinned slab")?;
                            for (f, buffers) in slab.iter().enumerate() {
                                for (pool, buffer, stride) in [
                                    (false, buffers.latent.as_ref(), LATENT),
                                    (true, buffers.pooled.as_ref(), POOLED),
                                ] {
                                    unsafe {
                                        stage.download_from(
                                            stream,
                                            at(count, cursor, f, pool),
                                            buffer,
                                            physical % EXTENT * stride,
                                            run * stride,
                                        )?;
                                    }
                                    source.copies += 1;
                                    source.bytes += run * stride;
                                }
                            }
                            cursor += run;
                        }
                        stream.sync()?;
                        source.group = Some((logical, count));
                    }
                    let stride = if pooled { POOLED } else { LATENT };
                    ensure!(
                        offset <= stride && buf.len() <= stride - offset,
                        "RAM export page slice"
                    );
                    let (start, count) = source.group.unwrap();
                    let host = at(count, logical - start, family, pooled) + offset;
                    buf.copy_from_slice(&stage.bytes_mut()[host..host + buf.len()]);
                }
                Source::Layer(role) => {
                    source.group = None;
                    let snapshot = self
                        .snapshots
                        .get(&source.lease.entry())
                        .context("RAM export pinned source")?;
                    if role == Role::DraftMetadata {
                        let metadata = snapshot
                            .draft
                            .as_ref()
                            .context("RAM export missing drafter metadata")?
                            .metadata();
                        ensure!(
                            offset <= metadata.len() && buf.len() <= metadata.len() - offset,
                            "RAM export drafter metadata slice"
                        );
                        buf.copy_from_slice(&metadata[offset..offset + buf.len()]);
                        return Ok(());
                    }
                    let buffer = layer(snapshot, role)?;
                    ensure!(
                        buf.len() <= CHUNK
                            && offset <= buffer.bytes
                            && buf.len() <= buffer.bytes - offset,
                        "RAM export private slice"
                    );
                    unsafe {
                        stage.download_from(stream, 0, buffer, offset, buf.len())?;
                    }
                    stream.sync()?;
                    source.copies += 1;
                    source.bytes += buf.len();
                    buf.copy_from_slice(&stage.bytes_mut()[..buf.len()]);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            // Fail closed, keeping the pin registered until owner teardown.
            self.healthy = false;
            long.healthy = false;
        }
        result
    }
}
