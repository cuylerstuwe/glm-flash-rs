//! Deterministic page ownership and admission accounting. No GPU or I/O side effects.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{cmp::Reverse, collections::BinaryHeap};
pub const PAGE: usize = 256;
pub const CONTEXT: usize = 1_048_576;
pub const MAX_OUTPUT: usize = 131_072;
pub const LOOKAHEAD: usize = 64;
pub const POOL_PAGES: usize = 2_097_664 / PAGE;
pub const SLOTS: usize = 4;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Active,
    Paused,
    Restoring,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pages: Vec<u32>,
    phase: Phase,
    background: bool,
    quota: usize,
    touched: u64,
}
impl Slot {
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    fn outstanding(&self) -> usize {
        self.quota.saturating_sub(self.pages.len())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    epoch: u64,
    basis: [Slot; SLOTS],
    slot: usize,
    quota: usize,
    keep: usize,
    spills: Vec<usize>,
    background: bool,
}
impl Admission {
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"alaska-admission-v1");
        for n in [
            self.epoch,
            self.slot as u64,
            self.quota as u64,
            self.keep as u64,
            u64::from(self.background),
            self.spills.len() as u64,
        ] {
            h.update(n.to_le_bytes());
        }
        for n in &self.spills {
            h.update((*n as u64).to_le_bytes());
        }
        for slot in &self.basis {
            let phase = match slot.phase {
                Phase::Idle => 0u64,
                Phase::Active => 1,
                Phase::Paused => 2,
                Phase::Restoring => 3,
            };
            for n in [
                phase,
                u64::from(slot.background),
                slot.quota as u64,
                slot.touched,
                slot.pages.len() as u64,
            ] {
                h.update(n.to_le_bytes());
            }
            for page in &slot.pages {
                h.update(page.to_le_bytes());
            }
        }
        h.finalize().into()
    }
    pub fn slot(&self) -> usize {
        self.slot
    }
    pub fn spills(&self) -> &[usize] {
        &self.spills
    }
}
#[derive(Clone, Debug)]
pub struct PageBook {
    free: BinaryHeap<Reverse<u32>>,
    slots: [Slot; SLOTS],
    epoch: u64,
}
impl Default for PageBook {
    fn default() -> Self {
        Self::new()
    }
}
impl PageBook {
    pub fn new() -> Self {
        Self {
            free: (0..POOL_PAGES as u32).map(Reverse).collect(),
            slots: std::array::from_fn(|_| Slot {
                pages: Vec::new(),
                phase: Phase::Idle,
                background: false,
                quota: 0,
                touched: 0,
            }),
            epoch: 0,
        }
    }
    pub fn slot(&self, slot: usize) -> Result<&Slot> {
        self.slots
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("slot out of range"))
    }
    pub fn available(&self) -> usize {
        self.free.len() - self.slots.iter().map(Slot::outstanding).sum::<usize>()
    }
    pub fn null_page(&self) -> u32 {
        POOL_PAGES as u32
    }
    /// The caller has already selected the final live/RAM/SSD restore destination.
    /// Plan without mutation: spill I/O can finish before applying an agreed plan.
    pub fn plan(
        &self,
        slot: usize,
        prompt: usize,
        output: usize,
        keep: usize,
        background: bool,
    ) -> Result<Admission> {
        ensure!(
            slot < SLOTS && self.slots[slot].phase == Phase::Idle,
            "admission requires idle destination"
        );
        ensure!(
            prompt > 0 && output > 0 && output <= MAX_OUTPUT && prompt <= CONTEXT - output,
            "request context/output limits"
        );
        ensure!(
            keep <= prompt && keep.div_ceil(PAGE) <= self.slots[slot].pages.len(),
            "retained prefix exceeds live mapping"
        );
        let quota = (prompt + output + LOOKAHEAD).div_ceil(PAGE);
        if background {
            let live: Vec<_> = self
                .slots
                .iter()
                .filter(|s| s.phase != Phase::Idle && s.background)
                .collect();
            ensure!(live.len() < SLOTS - 1, "last slot reserved for foreground");
            ensure!(
                live.iter().map(|s| s.quota).sum::<usize>() + quota <= POOL_PAGES / 2,
                "background page reservation exceeded"
            );
        }
        // Existing destination pages are already paid for. Credit once before
        // choosing OTHER idle victims; active and paused owners are never victims.
        let mut available = self.available() + self.slots[slot].pages.len();
        let mut candidates: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != slot && s.phase == Phase::Idle && !s.pages.is_empty())
            .map(|(i, s)| (s.touched, i))
            .collect();
        candidates.sort_unstable();
        let mut spills = Vec::new();
        for (_, i) in candidates {
            if available >= quota {
                break;
            }
            available += self.slots[i].pages.len();
            spills.push(i);
        }
        ensure!(
            available >= quota,
            "insufficient capacity while active/paused state is retained"
        );
        Ok(Admission {
            epoch: self.epoch,
            basis: self.slots.clone(),
            slot,
            quota,
            keep,
            spills,
            background,
        })
    }
    pub fn commit(&mut self, plan: &Admission) -> Result<()> {
        ensure!(
            plan.epoch == self.epoch && plan.basis == self.slots,
            "stale or foreign admission plan"
        );
        // Fields are private where policy-relevant; all fallible validation was
        // performed by plan(), and mutation changes epoch to prevent replay.
        ensure!(
            self.slots[plan.slot].phase == Phase::Idle,
            "destination became busy"
        );
        for &i in &plan.spills {
            self.truncate(i, 0);
        }
        self.truncate(plan.slot, plan.keep);
        let slot = &mut self.slots[plan.slot];
        slot.quota = plan.quota;
        slot.background = plan.background;
        slot.phase = Phase::Active;
        self.epoch += 1;
        slot.touched = self.epoch;
        Ok(())
    }
    pub fn map(&mut self, slot: usize, end: usize) -> Result<()> {
        ensure!(
            slot < SLOTS && self.slots[slot].phase == Phase::Active,
            "only active owner may map pages"
        );
        self.map_owned(slot, end)
    }
    pub fn map_restore(&mut self, slot: usize, end: usize) -> Result<()> {
        ensure!(
            self.slot(slot)?.phase == Phase::Restoring,
            "restore mapping requires reserved destination"
        );
        self.map_owned(slot, end)
    }
    fn map_owned(&mut self, slot: usize, end: usize) -> Result<()> {
        ensure!(end <= CONTEXT + LOOKAHEAD, "cache capacity exceeded");
        let need = end.div_ceil(PAGE);
        let have = self.slots[slot].pages.len();
        if need <= have {
            return Ok(());
        }
        let other_promises = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != slot)
            .map(|(_, s)| s.outstanding())
            .sum::<usize>();
        ensure!(
            need - have <= self.free.len() - other_promises,
            "mapping would steal another owner's reservation"
        );
        for _ in have..need {
            self.slots[slot].pages.push(self.free.pop().unwrap().0);
        }
        self.epoch += 1;
        Ok(())
    }
    pub fn pause(&mut self, slot: usize) -> Result<()> {
        ensure!(
            slot < SLOTS && self.slots[slot].phase == Phase::Active,
            "pause requires active owner"
        );
        self.slots[slot].phase = Phase::Paused;
        self.epoch += 1;
        Ok(())
    }
    pub fn begin_restore(&mut self, slot: usize) -> Result<()> {
        ensure!(
            self.slot(slot)?.phase == Phase::Active,
            "restore reservation requires admitted owner"
        );
        self.slots[slot].phase = Phase::Restoring;
        self.epoch += 1;
        Ok(())
    }
    pub fn finish_restore(&mut self, slot: usize) -> Result<()> {
        ensure!(
            self.slot(slot)?.phase == Phase::Restoring,
            "restore commit requires reserved destination"
        );
        self.slots[slot].phase = Phase::Active;
        self.epoch += 1;
        Ok(())
    }
    pub fn resume(&mut self, slot: usize) -> Result<()> {
        ensure!(
            slot < SLOTS && self.slots[slot].phase == Phase::Paused,
            "resume requires paused owner"
        );
        self.slots[slot].phase = Phase::Active;
        self.epoch += 1;
        Ok(())
    }
    pub fn release(&mut self, slot: usize, keep: usize) -> Result<()> {
        ensure!(
            slot < SLOTS && self.slots[slot].phase != Phase::Idle,
            "release requires live owner"
        );
        ensure!(
            self.slots[slot].phase != Phase::Restoring || keep == 0,
            "cannot retain a partial restore"
        );
        ensure!(
            keep.div_ceil(PAGE) <= self.slots[slot].pages.len(),
            "release prefix not mapped"
        );
        self.truncate(slot, keep);
        self.slots[slot].quota = 0;
        self.slots[slot].phase = Phase::Idle;
        self.epoch += 1;
        Ok(())
    }
    fn truncate(&mut self, slot: usize, keep: usize) {
        for page in self.slots[slot].pages.drain(keep.div_ceil(PAGE)..) {
            self.free.push(Reverse(page));
        }
    }
    /// Expensive audit for tests and diagnostic boundaries, never a decode-loop scan.
    pub fn check(&self) -> Result<()> {
        let mut seen = vec![false; POOL_PAGES];
        let mut count = 0;
        for page in self
            .free
            .iter()
            .map(|p| p.0)
            .chain(self.slots.iter().flat_map(|s| s.pages.iter().copied()))
        {
            ensure!(
                (page as usize) < POOL_PAGES && !seen[page as usize],
                "duplicate/out-of-range page ownership"
            );
            seen[page as usize] = true;
            count += 1;
        }
        ensure!(count == POOL_PAGES, "lost page ownership");
        ensure!(
            self.slots.iter().map(Slot::outstanding).sum::<usize>() <= self.free.len(),
            "overcommitted reservations"
        );
        Ok(())
    }
}
/// Logical offset, physical offset, count. Adjacent pages use one DMA span.
pub fn spans(pages: &[u32]) -> Vec<(usize, u32, usize)> {
    let mut out: Vec<(usize, u32, usize)> = Vec::new();
    for (i, &page) in pages.iter().enumerate() {
        if let Some((_, start, len)) = out.last_mut()
            && *start as usize + *len == page as usize
        {
            *len += 1;
        } else {
            out.push((i, page, 1));
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    fn admit(b: &mut PageBook, s: usize, p: usize, o: usize, k: usize, bg: bool) {
        let plan = b.plan(s, p, o, k, bg).unwrap();
        b.commit(&plan).unwrap();
    }
    #[test]
    fn restoring_destination_is_reserved_unrunnable_and_cannot_retain_partial_pages() {
        let mut b = PageBook::new();
        admit(&mut b, 0, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, true);
        let stale = b.plan(1, 64, 4, 0, false).unwrap();
        b.begin_restore(0).unwrap();
        assert!(b.commit(&stale).is_err());
        b.map_restore(0, 8256).unwrap();
        let pages = b.slots[0].pages.clone();
        let available = b.available();
        assert!(b.map(0, 8256).is_err());
        assert!(b.pause(0).is_err());
        assert!(b.resume(0).is_err());
        assert!(b.plan(0, 64, 4, 0, false).is_err());
        assert!(b.release(0, 8256).is_err());
        admit(&mut b, 1, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, false);
        assert_eq!(b.available(), 0);
        assert_eq!(b.slots[0].pages, pages);
        assert!(b.plan(2, 64, 4, 0, false).is_err());
        b.release(0, 0).unwrap();
        assert_eq!(b.available(), available);
        assert!(b.slots[0].pages.is_empty());
        assert_eq!(b.slots[1].phase, Phase::Active);
        b.check().unwrap();
        admit(&mut b, 0, 64, 4, 0, true);
        b.begin_restore(0).unwrap();
        b.map_restore(0, 64).unwrap();
        b.finish_restore(0).unwrap();
        assert_eq!(b.slots[0].phase, Phase::Active);
        assert!(b.finish_restore(0).is_err());
        assert!(b.map_restore(0, 64).is_err());
    }
    #[test]
    fn full_context_background_and_foreground_keep_their_promises() {
        let mut b = PageBook::new();
        admit(&mut b, 0, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, true);
        b.map(0, CONTEXT + LOOKAHEAD).unwrap();
        b.pause(0).unwrap();
        let pages = b.slots[0].pages.clone();
        admit(&mut b, 1, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, false);
        b.map(1, CONTEXT + LOOKAHEAD).unwrap();
        assert_eq!(b.available(), 0);
        assert_eq!(b.slots[0].pages, pages);
        assert!(b.plan(2, 1, 1, 0, false).is_err());
        b.check().unwrap();
    }
    #[test]
    fn continuation_credits_own_pages_once_without_spilling_them() {
        let mut b = PageBook::new();
        admit(&mut b, 0, 300_000, MAX_OUTPUT, 0, false);
        b.map(0, 300_000).unwrap();
        b.release(0, 300_000).unwrap();
        let retained = b.slots[0].pages.clone();
        admit(&mut b, 1, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, true);
        b.map(1, CONTEXT + LOOKAHEAD).unwrap();
        b.pause(1).unwrap();
        let p = b.plan(0, 304_783, MAX_OUTPUT, 300_000, false).unwrap();
        assert!(p.spills.is_empty());
        b.commit(&p).unwrap();
        assert_eq!(b.slots[0].pages, retained);
        b.map(0, 304_783).unwrap();
        b.check().unwrap();
    }
    #[test]
    fn idle_victims_only_and_stale_plans_are_atomic() {
        let mut b = PageBook::new();
        for s in 0..3 {
            admit(&mut b, s, 500_000, 1, 0, false);
            b.map(s, 500_000).unwrap();
            b.release(s, 500_000).unwrap();
        }
        let p = b
            .plan(3, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, false)
            .unwrap();
        assert_eq!(p.spills, vec![0]);
        b.commit(&p).unwrap();
        assert!(b.slots[0].pages.is_empty());
        let before = format!("{b:?}");
        assert!(b.commit(&p).is_err());
        assert_eq!(before, format!("{b:?}"));
    }
    #[test]
    fn paused_owner_cannot_be_mapped_replaced_or_replayed() {
        let mut b = PageBook::new();
        admit(&mut b, 0, 4096, 128, 0, true);
        b.map(0, 4096).unwrap();
        b.pause(0).unwrap();
        let pages = b.slots[0].pages.clone();
        assert!(b.map(0, 4097).is_err());
        assert!(b.plan(0, 5000, 128, 4096, false).is_err());
        b.resume(0).unwrap();
        b.map(0, 4097).unwrap();
        assert_eq!(pages, &b.slots[0].pages[..pages.len()]);
    }
    #[test]
    fn over_quota_growth_cannot_steal_unmapped_promises() {
        let mut b = PageBook::new();
        admit(&mut b, 0, 1, 1, 0, false);
        admit(&mut b, 1, CONTEXT - MAX_OUTPUT, MAX_OUTPUT, 0, false);
        admit(&mut b, 2, CONTEXT - 512, 1, 0, false);
        let before = b.free.len();
        assert!(b.map(0, 1024).is_err());
        assert_eq!(before, b.free.len());
        b.map(1, CONTEXT + LOOKAHEAD).unwrap();
    }
    #[test]
    fn context_output_background_and_last_slot_guards() {
        let mut b = PageBook::new();
        assert!(b.plan(0, CONTEXT, 1, 0, false).is_err());
        assert!(b.plan(0, 1, MAX_OUTPUT + 1, 0, false).is_err());
        for s in 0..3 {
            admit(&mut b, s, 1, 1, 0, true);
        }
        assert!(b.plan(3, 1, 1, 0, true).is_err());
        admit(&mut b, 3, 1, 1, 0, false);
    }
    #[test]
    fn contiguous_and_fragmented_spans_preserve_logical_order() {
        assert_eq!(
            spans(&[8, 9, 10, 3, 4, 17]),
            vec![(0, 8, 3), (3, 3, 2), (5, 17, 1)]
        );
        assert!(spans(&[]).is_empty());
    }
    #[test]
    fn foreign_plan_cannot_replace_another_owners_book() {
        let mut a = PageBook::new();
        let mut b = PageBook::new();
        admit(&mut a, 0, 4096, 128, 0, false);
        admit(&mut b, 1, 4096, 128, 0, false);
        let plan = a.plan(2, 4096, 128, 0, false).unwrap();
        let before = format!("{b:?}");
        assert!(b.commit(&plan).is_err());
        assert_eq!(before, format!("{b:?}"));
    }
    #[test]
    fn deterministic_repeated_ownership_rounds() {
        let mut a = PageBook::new();
        let mut b = PageBook::new();
        let mut rng = 0x18238491u64;
        for step in 0..4000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let slot = (rng as usize) % SLOTS;
            let prompt = 1 + ((rng >> 8) as usize) % 200_000;
            for book in [&mut a, &mut b] {
                match book.slots[slot].phase {
                    Phase::Idle => {
                        if let Ok(p) = book.plan(slot, prompt, 128, 0, false) {
                            book.commit(&p).unwrap();
                            book.map(slot, prompt).unwrap();
                        }
                    }
                    Phase::Active => {
                        if step % 5 == 0 {
                            book.begin_restore(slot).unwrap();
                        } else if step % 3 == 0 {
                            book.pause(slot).unwrap();
                        } else {
                            book.release(slot, prompt.min(book.slots[slot].pages.len() * PAGE))
                                .unwrap();
                        }
                    }
                    Phase::Paused => book.resume(slot).unwrap(),
                    Phase::Restoring => {
                        if step % 2 == 0 {
                            book.finish_restore(slot).unwrap();
                        } else {
                            book.release(slot, 0).unwrap();
                        }
                    }
                };
                book.check().unwrap();
            }
            assert_eq!(a.slots, b.slots);
            assert_eq!(a.available(), b.available());
        }
    }
}
