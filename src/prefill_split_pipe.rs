//! Bounded rank-indexed partial exchange, behind the default-off row policy.
//! All consumers/producers use one owner stream; NCCL uses a separate stream.
use crate::{
    comm::{Exchange, Nccl, Region},
    cuda::{Buffer, Device, Fence, Stream},
};
use anyhow::{Result, ensure};
use std::rc::Rc;

#[path = "prefill_row_plan.rs"]
pub(super) mod plan;
use plan::{MAX_ROWS, Plan, Rows, WIDTH, WORLD};
#[path = "prefill_split_probe.rs"]
mod probe;
pub use probe::run as probe;

/// One dedicated slot per expert-layer piece, so a layer's partial exchanges can
/// stay outstanding while the next layer's attention runs.
pub(super) const FFN_SLOTS: usize = 8;
const SHARE_FENCES: usize = 32;
struct Slot {
    partial: Buffer,
    ready: Fence,
    done: Fence,
    pending: Option<u64>,
}
struct Sharing {
    ready: Fence,
    done: Fence,
    pending: bool,
}
/// Completion of one queued share. The side stream runs in issue order, so
/// waiting for it also orders every earlier exchange and share.
#[derive(Clone, Copy)]
pub(super) struct ShareToken {
    index: usize,
    serial: u64,
}
/// Not cloneable: one caller must consume each outstanding partial exchange.
#[must_use]
pub(super) struct Pending {
    slot: usize,
    serial: u64,
    plan: Plan,
    storage: u64,
}
pub(super) struct View<'a> {
    pub(super) partial: &'a Buffer,
    pub(super) owned: Rows,
    pub(super) rank_stride: usize,
}
pub(super) struct Field<'a> {
    pub(super) buffer: &'a Buffer,
    pub(super) base: usize,
    pub(super) row_bytes: usize,
}
pub(super) struct Pipe {
    device: Rc<Device>,
    stream: Stream,
    /// Two alternating attention slots, then `FFN_SLOTS` expert-piece slots.
    slots: Vec<Slot>,
    sharing: Sharing,
    share_done: Vec<Fence>,
    shares: u64,
    rank: usize,
    capacity: usize,
    next: u64,
    turn: u64,
    healthy: bool,
}
impl Pipe {
    pub(super) fn new(d: &Rc<Device>, rank: usize, capacity: usize) -> Result<Self> {
        ensure!(
            rank < WORLD && (1..=MAX_ROWS).contains(&capacity),
            "split pipe rank/capacity"
        );
        let slot = || -> Result<Slot> {
            Ok(Slot {
                partial: d.alloc(WORLD * capacity * WIDTH * 2)?,
                ready: d.fence()?,
                done: d.fence()?,
                pending: None,
            })
        };
        Ok(Self {
            device: d.clone(),
            stream: d.stream()?,
            slots: (0..2 + FFN_SLOTS).map(|_| slot()).collect::<Result<_>>()?,
            sharing: Sharing {
                ready: d.fence()?,
                done: d.fence()?,
                pending: false,
            },
            share_done: (0..SHARE_FENCES)
                .map(|_| d.fence())
                .collect::<Result<_>>()?,
            shares: 0,
            rank,
            capacity,
            next: 0,
            turn: 0,
            healthy: true,
        })
    }
    pub(super) fn begin(&self) -> Result<()> {
        ensure!(
            self.healthy && self.slots.iter().all(|s| s.pending.is_none()) && !self.sharing.pending,
            "split pipe not healthy/drained"
        );
        Ok(())
    }
    /// A layer's attention may start while the previous expert layer's pieces
    /// and shares are still outstanding; only the attention slots must be free.
    pub(super) fn begin_attention(&self) -> Result<()> {
        ensure!(
            self.healthy && self.slots[..2].iter().all(|s| s.pending.is_none()),
            "split pipe attention slots not drained"
        );
        Ok(())
    }
    /// The producer writes only its rank slice, on the owner stream, before issue.
    pub(super) fn producer_pointer(&self, rows: usize) -> Result<u64> {
        let (buffer, offset) = self.producer_region(rows)?;
        Ok(buffer.ptr + offset as u64)
    }
    pub(super) fn producer_region(&self, rows: usize) -> Result<(&Buffer, usize)> {
        self.slot_region((self.turn % 2) as usize, rows)
    }
    pub(super) fn ffn_producer_region(
        &self,
        piece: usize,
        rows: usize,
    ) -> Result<(&Buffer, usize)> {
        ensure!(piece < FFN_SLOTS, "split pipe expert piece");
        self.slot_region(2 + piece, rows)
    }
    fn slot_region(&self, index: usize, rows: usize) -> Result<(&Buffer, usize)> {
        self.plan(rows)?;
        let slot = &self.slots[index];
        ensure!(
            slot.pending.is_none(),
            "split pipe producer overwrites outstanding exchange"
        );
        Ok((&slot.partial, self.rank * self.capacity * WIDTH * 2))
    }
    fn plan(&self, rows: usize) -> Result<Plan> {
        ensure!(
            self.healthy && rows <= self.capacity && self.next < u64::MAX,
            "split pipe health/capacity/sequence"
        );
        Plan::new(rows, self.rank)
    }
    /// Caller has commonly agreed this schedule and retains the NCCL owner.
    /// Buffer/device and stream identities must match; no raw pointer escapes
    /// beyond this pipe and its surrounding model/graph lifetime.
    pub(super) unsafe fn issue(
        &mut self,
        comm: &Nccl,
        owner: &Stream,
        rows: usize,
    ) -> Result<Pending> {
        let index = (self.turn % 2) as usize;
        let pending = unsafe { self.issue_slot(comm, owner, index, rows)? };
        self.turn += 1;
        Ok(pending)
    }
    /// `issue` for expert piece `piece`, written through `ffn_producer_region`.
    pub(super) unsafe fn issue_ffn(
        &mut self,
        comm: &Nccl,
        owner: &Stream,
        piece: usize,
        rows: usize,
    ) -> Result<Pending> {
        ensure!(piece < FFN_SLOTS, "split pipe expert piece");
        unsafe { self.issue_slot(comm, owner, 2 + piece, rows) }
    }
    unsafe fn issue_slot(
        &mut self,
        comm: &Nccl,
        owner: &Stream,
        index: usize,
        rows: usize,
    ) -> Result<Pending> {
        let plan = self.plan(rows)?;
        let offsets = plan.partials(self.capacity)?;
        let slot = &mut self.slots[index];
        ensure!(
            slot.pending.is_none(),
            "split pipe issue overwrites outstanding exchange"
        );
        self.healthy = false;
        slot.ready.record(owner)?;
        slot.ready.wait_on(&self.stream)?;
        let exchanges = offsets.map(|p| Exchange {
            peer: p.peer,
            send: Region {
                buffer: &slot.partial,
                offset: p.send_offset,
                bytes: p.send_bytes,
            },
            receive: Region {
                buffer: &slot.partial,
                offset: p.receive_offset,
                bytes: p.receive_bytes,
            },
        });
        unsafe {
            comm.exchange_regions(&exchanges, &self.stream)?;
        }
        slot.done.record(&self.stream)?;
        let serial = self.next;
        self.next += 1;
        slot.pending = Some(serial);
        self.healthy = true;
        Ok(Pending {
            slot: index,
            serial,
            plan,
            storage: slot.partial.ptr,
        })
    }
    /// Queue the consumer dependency on the owner stream. Reuse is safe only
    /// after the caller enqueues its HC consumer there, before another producer.
    pub(super) fn wait(&mut self, owner: &Stream, pending: Pending) -> Result<View<'_>> {
        ensure!(
            self.healthy
                && self.slots[pending.slot].pending == Some(pending.serial)
                && self.slots[pending.slot].partial.ptr == pending.storage,
            "split pipe stale exchange handle"
        );
        self.healthy = false;
        let slot = &mut self.slots[pending.slot];
        slot.done.wait_on(owner)?;
        slot.pending = None;
        self.healthy = true;
        Ok(View {
            partial: &slot.partial,
            owned: pending.plan.own(),
            rank_stride: self.capacity * WIDTH,
        })
    }
    /// In-place owned-row broadcasts. The caller retains all fields and must
    /// not modify a sent row until wait_shares orders completion on the owner.
    /// The same reusable fence pair covers all queued shares on the side stream.
    pub(super) unsafe fn share(
        &mut self,
        comm: &Nccl,
        owner: &Stream,
        rows: usize,
        fields: &[Field<'_>],
    ) -> Result<ShareToken> {
        unsafe { self.share_clipped(comm, owner, rows, 0, rows, fields) }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn share_clipped(
        &mut self,
        comm: &Nccl,
        owner: &Stream,
        rows: usize,
        first: usize,
        count: usize,
        fields: &[Field<'_>],
    ) -> Result<ShareToken> {
        let plan = self.plan(rows)?;
        ensure!(
            !fields.is_empty() && fields.len() <= 2,
            "split pipe share field bound"
        );
        // Validate every field before recording events or opening a group.
        let mut offsets = [[plan::Transfer {
            peer: 0,
            send_offset: 0,
            send_bytes: 0,
            receive_offset: 0,
            receive_bytes: 0,
        }; 3]; 2];
        for (f, out) in fields.iter().zip(&mut offsets) {
            *out = if first == 0 && count == rows {
                plan.share(f.row_bytes, f.base, f.buffer.bytes)?
            } else {
                plan.share_clipped(f.row_bytes, f.base, f.buffer.bytes, first, count)?
            };
        }
        self.healthy = false;
        self.sharing.ready.record(owner)?;
        self.sharing.ready.wait_on(&self.stream)?;
        let field = |i: usize| {
            let f = &fields[i / 3];
            let p = offsets[i / 3][i % 3];
            Exchange {
                peer: p.peer,
                send: Region {
                    buffer: f.buffer,
                    offset: p.send_offset,
                    bytes: p.send_bytes,
                },
                receive: Region {
                    buffer: f.buffer,
                    offset: p.receive_offset,
                    bytes: p.receive_bytes,
                },
            }
        };
        // Six fixed descriptors, with no heap work inside the native group.
        if fields.len() == 1 {
            let entries = std::array::from_fn::<_, 3, _>(field);
            unsafe {
                comm.exchange_regions(&entries, &self.stream)?;
            }
        } else {
            let entries = std::array::from_fn::<_, 6, _>(field);
            unsafe {
                comm.exchange_regions(&entries, &self.stream)?;
            }
        }
        self.sharing.done.record(&self.stream)?;
        let token = ShareToken {
            index: (self.shares % SHARE_FENCES as u64) as usize,
            serial: self.shares,
        };
        self.share_done[token.index].record(&self.stream)?;
        self.shares += 1;
        self.sharing.pending = true;
        self.healthy = true;
        Ok(token)
    }
    /// Order one share's completion on the owner stream; `wait_shares` still
    /// owns the drained state.
    pub(super) fn wait_share(&mut self, owner: &Stream, token: ShareToken) -> Result<()> {
        ensure!(
            self.healthy
                && token.serial < self.shares
                && self.shares - token.serial <= SHARE_FENCES as u64,
            "split pipe stale share token"
        );
        self.share_done[token.index].wait_on(owner)
    }
    pub(super) fn wait_shares(&mut self, owner: &Stream) -> Result<()> {
        ensure!(self.healthy, "split pipe unhealthy");
        if self.sharing.pending {
            self.healthy = false;
            self.sharing.done.wait_on(owner)?;
            self.sharing.pending = false;
            self.healthy = true;
        }
        Ok(())
    }
}
impl Drop for Pipe {
    fn drop(&mut self) {
        // Teardown only. Both the communication work and already queued owner
        // consumers must finish before partial/event storage is destroyed.
        let _ = self.device.sync();
    }
}
