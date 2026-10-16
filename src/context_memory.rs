//! Shared physical latent/pool-key families, per-slot mappings and coalesced copies.
//! Admission, pause acknowledgements, recurrent state and SSD format are separate layers.
use crate::{
    cache_pages::{Admission, CONTEXT, LOOKAHEAD, PAGE, POOL_PAGES, PageBook, SLOTS, spans},
    cuda::{Buffer, Device, Pinned, Stream},
};
use anyhow::{Result, ensure};
use std::rc::Rc;
const LATENT_PAGE: usize = PAGE * 528;
const POOL_PAGE: usize = PAGE / 4 * 128 * 2;
const TABLE_PAGES: usize = (CONTEXT + LOOKAHEAD).div_ceil(PAGE);
pub struct Family {
    pub latent: Rc<Buffer>,
    pub pooled: Rc<Buffer>,
}
pub struct Packed {
    device: Rc<Device>,
    pub families: Vec<Family>,
    pages: usize,
}
pub struct ContextMemory {
    device: Rc<Device>,
    book: PageBook,
    tables: Vec<Rc<Buffer>>,
    host_tables: Vec<Pinned>,
    pub families: Vec<Family>,
    poisoned: bool,
    owner_stream: usize,
}
impl ContextMemory {
    pub fn new(d: &Rc<Device>, s: &Stream, layers: usize) -> Result<Self> {
        ensure!((1..=12).contains(&layers), "DSA family count");
        let mut families = Vec::new();
        let alloc = |n| -> Result<Rc<Buffer>> {
            let b = d.alloc(n)?;
            unsafe {
                b.zero_prefix(s, n)?;
            }
            Ok(Rc::new(b))
        };
        for _ in 0..layers {
            families.push(Family {
                latent: alloc((POOL_PAGES + 1) * LATENT_PAGE)?,
                pooled: alloc((POOL_PAGES + 1) * POOL_PAGE)?,
            });
        }
        let tables = (0..SLOTS)
            .map(|_| alloc(TABLE_PAGES * 4))
            .collect::<Result<Vec<_>>>()?;
        let host_tables = (0..SLOTS)
            .map(|_| Pinned::new(d, TABLE_PAGES * 4))
            .collect::<Result<Vec<_>>>()?;
        let mut result = Self {
            device: d.clone(),
            book: PageBook::new(),
            tables,
            host_tables,
            families,
            poisoned: false,
            owner_stream: s.raw() as usize,
        };
        for slot in 0..SLOTS {
            result.upload(s, slot)?;
        }
        s.sync()?;
        Ok(result)
    }
    pub fn book(&self) -> &PageBook {
        &self.book
    }
    pub fn table(&self, slot: usize) -> Result<Rc<Buffer>> {
        self.tables
            .get(slot)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("slot out of range"))
    }
    fn ready(&self) -> Result<()> {
        ensure!(!self.poisoned, "cache owner is poisoned after CUDA failure");
        Ok(())
    }
    fn check_stream(&self, s: &Stream) -> Result<()> {
        self.ready()?;
        ensure!(
            s.raw() as usize == self.owner_stream,
            "cache requires its ordered owner stream"
        );
        Ok(())
    }
    fn upload(&mut self, s: &Stream, slot: usize) -> Result<()> {
        let pages = self.book.slot(slot)?.pages();
        let null = self.book.null_page();
        let bytes = self.host_tables[slot].bytes_mut();
        for (i, chunk) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            chunk.copy_from_slice(&pages.get(i).copied().unwrap_or(null).to_le_bytes());
        }
        // Caller drained the prior table copy before mutating pinned storage.
        let result = unsafe {
            self.host_tables[slot].upload_to(s, &self.tables[slot], 0, 0, TABLE_PAGES * 4)
        };
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn commit(&mut self, s: &Stream, plan: &Admission) -> Result<()> {
        self.check_stream(s)?;
        s.sync()?;
        self.book.commit(plan)?;
        for &slot in plan.spills() {
            self.upload(s, slot)?;
        }
        self.upload(s, plan.slot())
    }
    pub fn map(&mut self, s: &Stream, slot: usize, end: usize) -> Result<()> {
        self.check_stream(s)?;
        let old = self.book.slot(slot)?.pages().len();
        if end.div_ceil(PAGE) <= old {
            return self.book.map(slot, end);
        }
        s.sync()?;
        self.book.map(slot, end)?;
        self.upload(s, slot)
    }
    pub fn pause(&mut self, s: &Stream, slot: usize) -> Result<()> {
        self.check_stream(s)?;
        s.sync()?;
        self.book.pause(slot)
    }
    pub fn begin_restore(&mut self, slot: usize) -> Result<()> {
        self.ready()?;
        self.book.begin_restore(slot)
    }
    pub fn map_restore(&mut self, s: &Stream, slot: usize, end: usize) -> Result<()> {
        self.check_stream(s)?;
        s.sync()?;
        self.book.map_restore(slot, end)?;
        self.upload(s, slot)
    }
    pub fn finish_restore(&mut self, s: &Stream, slot: usize) -> Result<()> {
        self.check_stream(s)?;
        s.sync()?;
        self.book.finish_restore(slot)
    }
    pub fn resume(&mut self, slot: usize) -> Result<()> {
        self.ready()?;
        self.book.resume(slot)
    }
    pub fn release(&mut self, s: &Stream, slot: usize, keep: usize) -> Result<()> {
        self.check_stream(s)?;
        s.sync()?;
        self.book.release(slot, keep)?;
        self.upload(s, slot)
    }
    /// Data copies are ordered on the owner stream. The caller retains Packed
    /// until those copies finish. Complete snapshots must also include recurrence/rings.
    pub fn pack(&self, s: &Stream, slot: usize, tokens: usize) -> Result<Packed> {
        self.check_stream(s)?;
        let count = tokens.div_ceil(PAGE);
        let pages = self.book.slot(slot)?.pages();
        ensure!(
            count > 0 && count <= pages.len(),
            "snapshot prefix not mapped"
        );
        let mut families = Vec::new();
        for family in &self.families {
            let latent = Rc::new(self.device.alloc(count * LATENT_PAGE)?);
            let pooled = Rc::new(self.device.alloc(count * POOL_PAGE)?);
            for (logical, physical, len) in spans(&pages[..count]) {
                for (src, dst, size) in [
                    (&family.latent, &latent, LATENT_PAGE),
                    (&family.pooled, &pooled, POOL_PAGE),
                ] {
                    unsafe {
                        dst.copy_from(
                            s,
                            logical * size,
                            src,
                            physical as usize * size,
                            len * size,
                        )?;
                    }
                }
            }
            families.push(Family { latent, pooled });
        }
        Ok(Packed {
            device: self.device.clone(),
            families,
            pages: count,
        })
    }
    pub fn restore(&self, s: &Stream, slot: usize, snapshot: &Packed) -> Result<()> {
        self.check_stream(s)?;
        let state = self.book.slot(slot)?;
        ensure!(
            state.phase() == crate::cache_pages::Phase::Active,
            "restore requires active destination"
        );
        ensure!(
            snapshot.pages <= state.pages().len() && snapshot.families.len() == self.families.len(),
            "snapshot layout mismatch"
        );
        for (dst, src) in self.families.iter().zip(&snapshot.families) {
            for (logical, physical, len) in spans(&state.pages()[..snapshot.pages]) {
                for (dst, src, size) in [
                    (&dst.latent, &src.latent, LATENT_PAGE),
                    (&dst.pooled, &src.pooled, POOL_PAGE),
                ] {
                    unsafe {
                        dst.copy_from(
                            s,
                            physical as usize * size,
                            src,
                            logical * size,
                            len * size,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
// Device-wide destruction fences protect buffers referenced by outstanding work.
// They run on snapshot eviction/owner destruction, not on generation steps.
impl Drop for ContextMemory {
    fn drop(&mut self) {
        let _ = self.device.sync();
    }
}
impl Drop for Packed {
    fn drop(&mut self) {
        let _ = self.device.sync();
    }
}
pub fn probe() -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let mut memory = ContextMemory::new(&d, &s, 11)?;
    for slot in 0..2 {
        let p = memory.book().plan(slot, 1024, 1, 0, false)?;
        memory.commit(&s, &p)?;
        memory.map(&s, slot, 1024)?;
    }
    memory.pause(&s, 1)?;
    let protected = memory.book().slot(1)?.pages().to_vec();
    memory.release(&s, 0, 256)?;
    let p = memory.book().plan(2, 512, 1, 0, false)?;
    memory.commit(&s, &p)?;
    memory.map(&s, 2, 512)?;
    let p = memory.book().plan(0, 1024, 1, 256, false)?;
    memory.commit(&s, &p)?;
    memory.map(&s, 0, 1024)?;
    s.sync()?;
    let original = memory.book().slot(0)?.pages().to_vec();
    ensure!(original == [0, 3, 8, 9], "fragmentation fixture");
    let mut expected = Vec::new();
    for (layer, family) in memory.families.iter().enumerate() {
        let mut parts = Vec::new();
        for (buffer, size, kind) in [
            (&family.latent, LATENT_PAGE, 0),
            (&family.pooled, POOL_PAGE, 1),
        ] {
            let bytes: Vec<u8> = (0..4 * size)
                .map(|i| ((i * 73 + layer * 41 + kind * 19) % 251) as u8)
                .collect();
            for (logical, &physical) in original.iter().enumerate() {
                buffer.write(
                    physical as usize * size,
                    &bytes[logical * size..(logical + 1) * size],
                )?;
            }
            parts.push(bytes);
        }
        expected.push(parts);
    }
    d.sync()?;
    let snapshot = memory.pack(&s, 0, 1024)?;
    for (family, parts) in snapshot.families.iter().zip(&expected) {
        ensure!(
            family.latent.read(0, family.latent.bytes)? == parts[0]
                && family.pooled.read(0, family.pooled.bytes)? == parts[1],
            "packed logical bytes"
        );
    }
    memory.pause(&s, 0)?;
    let p = memory.book().plan(3, CONTEXT - 131072, 131072, 0, false)?;
    memory.commit(&s, &p)?;
    memory.map(&s, 3, CONTEXT + LOOKAHEAD)?;
    memory.restore(&s, 3, &snapshot)?;
    let migrated = memory.pack(&s, 3, 1024)?;
    for (family, parts) in migrated.families.iter().zip(&expected) {
        ensure!(
            family.latent.read(0, family.latent.bytes)? == parts[0]
                && family.pooled.read(0, family.pooled.bytes)? == parts[1],
            "restored logical bytes"
        );
    }
    ensure!(
        memory.book().slot(0)?.pages() == original && memory.book().slot(1)?.pages() == protected,
        "paused mappings changed"
    );
    memory.resume(0)?;
    for slot in 0..SLOTS {
        let table = memory.table(slot)?.read(0, TABLE_PAGES * 4)?;
        for (i, word) in table.as_chunks::<4>().0.iter().enumerate() {
            ensure!(
                u32::from_le_bytes(*word)
                    == memory
                        .book()
                        .slot(slot)?
                        .pages()
                        .get(i)
                        .copied()
                        .unwrap_or(memory.book().null_page()),
                "device mapping"
            );
        }
    }
    for family in &memory.families {
        for (buffer, size) in [(&family.latent, LATENT_PAGE), (&family.pooled, POOL_PAGE)] {
            ensure!(
                buffer
                    .read(POOL_PAGES * size, size)?
                    .iter()
                    .all(|b| *b == 0),
                "null page not zero"
            );
        }
    }
    let wrong_stream = d.stream()?;
    ensure!(
        memory.pack(&wrong_stream, 0, 1024).is_err(),
        "foreign stream accepted"
    );
    memory.book().check()?;
    s.sync()?;
    Ok(
        serde_json::json!({"pool_tokens":POOL_PAGES*PAGE,"context_limit":CONTEXT,"lookahead_tokens":LOOKAHEAD,"layers":11,"slots":SLOTS,"physical_family_bytes":memory.families.iter().map(|f|f.latent.bytes+f.pooled.bytes).sum::<usize>(),"fragmented_pages":original,"coalesced_spans":spans(&original),"packed_and_restored_bytes_exact":true,"paused_owners_retained":true,"all_device_mappings_exact":true,"null_pages_zero":true,"scope":"Page ownership and latent/pool-key copies only; complete recurrent snapshots and scheduler integration require separate tests."}),
    )
}
