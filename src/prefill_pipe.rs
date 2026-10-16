//! Two bounded rank-gather slots on a separate NCCL stream. Main-stream ready
//! events protect send reuse, done events protect gather readers. One owner.
use super::{
    Engine,
    batch::{Batch, NextPre},
};
use crate::cuda::{Buffer, Device, Fence, Stream};
use anyhow::{Result, ensure};
use std::rc::Rc;
struct Slot {
    send: Buffer,
    gathered: Buffer,
    ready: Fence,
    done: Fence,
}
struct Pending {
    next: Option<NextPre>,
    slot: usize,
    offset: usize,
    rows: usize,
}
pub(super) struct Pipe {
    stream: Stream,
    slots: [Slot; 2],
    pending: Option<Pending>,
    turn: usize,
    capacity: usize,
}
impl Pipe {
    pub(super) fn new(d: &Rc<Device>, capacity: usize) -> Result<Self> {
        ensure!(capacity <= 1024 && capacity > 0, "prefill pipe capacity");
        let slot = || -> Result<Slot> {
            Ok(Slot {
                send: d.alloc(capacity * 4096 * 2)?,
                gathered: d.alloc(capacity * 4 * 4096 * 2)?,
                ready: d.fence()?,
                done: d.fence()?,
            })
        };
        Ok(Self {
            stream: d.stream()?,
            slots: [slot()?, slot()?],
            pending: None,
            turn: 0,
            capacity,
        })
    }
    pub(super) fn begin(&mut self) -> Result<()> {
        ensure!(self.pending.is_none(), "prefill pipe not drained");
        self.turn = 0;
        Ok(())
    }
    pub(super) fn issue(
        &mut self,
        batch: &Batch,
        e: &Engine<'_>,
        offset: usize,
        rows: usize,
        next: Option<NextPre>,
        source_in_slot: bool,
    ) -> Result<()> {
        ensure!(rows > 0 && rows <= self.capacity, "prefill pipe rows");
        let idx = self.turn % 2;
        self.turn += 1;
        let slot = &mut self.slots[idx];
        if !source_in_slot {
            batch.cast_rows(e, offset, rows, &slot.send)?;
        }
        slot.ready.record(&e.stream)?;
        slot.ready.wait_on(&self.stream)?;
        unsafe {
            e.comm
                .all_gather_prefix(&slot.send, &slot.gathered, rows * 4096 * 2, &self.stream)?;
        }
        slot.done.record(&self.stream)?;
        if let Some(previous) = self.pending.take() {
            self.finish(batch, e, previous)?;
        }
        self.pending = Some(Pending {
            next,
            slot: idx,
            offset,
            rows,
        });
        Ok(())
    }
    /// The caller launches a producer on the owner stream before issue(). A
    /// slot is reused only after its previous gather's done event was waited
    /// on by that stream. No host synchronization or per-piece allocation.
    pub(super) fn producer_pointer(&self, rows: usize) -> Result<u64> {
        ensure!(rows > 0 && rows <= self.capacity, "prefill producer rows");
        Ok(self.slots[self.turn % 2].send.ptr)
    }
    fn finish(&self, batch: &Batch, e: &Engine<'_>, p: Pending) -> Result<()> {
        let slot = &self.slots[p.slot];
        slot.done.wait_on(&e.stream)?;
        batch.post_rows(e, p.offset, p.rows, &slot.gathered, p.next)
    }
    pub(super) fn drain(&mut self, batch: &Batch, e: &Engine<'_>) -> Result<()> {
        if let Some(p) = self.pending.take() {
            self.finish(batch, e, p)?;
        }
        Ok(())
    }
}
impl Drop for Pipe {
    fn drop(&mut self) {
        let _ = self.stream.sync();
    }
}
