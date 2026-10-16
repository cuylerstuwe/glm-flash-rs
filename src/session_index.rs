//! Deterministic, bounded metadata for pure 64-grid prefix snapshots.
//! Payload allocation/copies and collective agreement are deliberately separate.
use crate::cache_pages::{CONTEXT, PAGE};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

type Hash = [u8; 32];
pub const EXTENT: usize = 16;
pub fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
const LATENT_ROW: u64 = 528 * 11;
const POOLED_ROW: u64 = 128 * 2 * 11;
const PAGE_BYTES: u64 = PAGE as u64 * LATENT_ROW + (PAGE / 4) as u64 * POOLED_ROW;
// Conservative accounting allowances, not assertions about allocator internals.
const ENTRY_META: u64 = 1024;
const PAGE_META: u64 = 256;
const ENTRY_PAGE_META: u64 = 64;
const MAX_ENTRIES: usize = 4096;
const MAX_LEASES: usize = 64;
const LEASE_META: u64 = 128;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct Entry {
    ids: Arc<[i32]>,
    key: Hash,
    pages: Vec<Hash>,
    private: u64,
    used: u64,
}
#[derive(Clone)]
struct PageRef {
    physical: usize,
    refs: usize,
}
#[derive(Clone, Default)]
struct State {
    entries: BTreeMap<u64, Entry>,
    by_key: BTreeMap<Hash, u64>,
    pages: BTreeMap<Hash, PageRef>,
    extents: BTreeMap<usize, usize>,
    free: BTreeSet<usize>,
    private: u64,
    clock: u64,
    next_id: u64,
    leases: BTreeMap<u64, u64>,
    next_lease: u64,
}
/// An explicit counted source pin. It never owns GPU buffers itself. Dropping
/// an unreleased lease leaves the entry pinned (fail closed); release only after
/// the CUDA owner has drained its final copy. Owner identity is process-local.
#[derive(Debug)]
pub struct Lease {
    owner: u64,
    serial: u64,
    entry: u64,
}
impl Lease {
    pub fn entry(&self) -> u64 {
        self.entry
    }
    pub fn serial(&self) -> u64 {
        self.serial
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Saved,
    Duplicate,
    Skipped,
}
pub struct Plan {
    owner: u64,
    epoch: u64,
    next: Option<State>,
    pub status: Status,
    pub entry: Option<u64>,
    pub evicted: Vec<u64>,
    /// (logical page, physical slab page); only these need payload copies.
    pub new_pages: Vec<(usize, usize)>,
}
pub struct Index {
    owner: u64,
    epoch: u64,
    namespace: Hash,
    budget: u64,
    state: State,
    /// Every state change bumps `epoch`; scheduler agreements read the digest
    /// several times a round, and hashing every page of a large store is not
    /// free.
    memo: std::cell::Cell<Option<([u64; 4], Hash)>>,
}

fn hash_ids(namespace: &Hash, ids: &[i32]) -> Hash {
    let mut h = Sha256::new();
    h.update(b"alaska-rust-pure-prefix-v1");
    h.update(namespace);
    h.update((ids.len() as u64).to_le_bytes());
    for token in ids {
        h.update(token.to_le_bytes());
    }
    h.finalize().into()
}
fn page_keys(namespace: &Hash, ids: &[i32]) -> Vec<Hash> {
    let mut chain: Hash =
        Sha256::digest([b"alaska-rust-pages-v1".as_slice(), namespace].concat()).into();
    let mut keys = Vec::with_capacity((ids.len() - 1) / PAGE);
    // As in the retained engine, the key includes the next token even without
    // MTP. A full last page lacking that token remains private. This namespace
    // is pure KDA64/BF16 only; mixed history or future MTP need distinct formats.
    for page in 0..(ids.len() - 1) / PAGE {
        let end = (page + 1) * PAGE;
        let mut h = Sha256::new();
        h.update(chain);
        for token in &ids[page * PAGE..end] {
            h.update(token.to_le_bytes());
        }
        chain = h.finalize().into();
        let mut h = Sha256::new();
        h.update(b"page");
        h.update(chain);
        h.update(ids[end].to_le_bytes());
        keys.push(h.finalize().into());
    }
    keys
}
impl State {
    fn gpu_bytes(&self) -> u64 {
        self.extents.len() as u64 * EXTENT as u64 * PAGE_BYTES
            + self
                .entries
                .values()
                .map(|e| {
                    e.private
                        - e.ids.len() as u64 * 4
                        - ENTRY_META
                        - e.pages.len() as u64 * ENTRY_PAGE_META
                })
                .sum::<u64>()
    }
    fn used(&self) -> u64 {
        self.private
            + self.extents.len() as u64 * EXTENT as u64 * PAGE_BYTES
            + self.pages.len() as u64 * PAGE_META
            + self.leases.len() as u64 * LEASE_META
    }
    fn tick(&mut self) -> Result<u64> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session clock overflow"))?;
        Ok(self.clock)
    }
    fn unref(&mut self, key: &Hash) {
        let page = self.pages.get_mut(key).expect("owned page");
        page.refs -= 1;
        if page.refs != 0 {
            return;
        }
        let physical = page.physical;
        self.pages.remove(key);
        let extent = physical / EXTENT;
        let count = self.extents.get_mut(&extent).expect("owned extent");
        *count -= 1;
        self.free.insert(physical);
        if *count == 0 {
            self.extents.remove(&extent);
            for p in extent * EXTENT..(extent + 1) * EXTENT {
                self.free.remove(&p);
            }
        }
    }
    fn remove(&mut self, id: u64) {
        let entry = self.entries.remove(&id).expect("owned entry");
        self.by_key.remove(&entry.key);
        self.private -= entry.private;
        for key in &entry.pages {
            self.unref(key);
        }
    }
    fn allocate(&mut self) -> usize {
        if self.free.is_empty() {
            let extent = (0..).find(|e| !self.extents.contains_key(e)).unwrap();
            self.extents.insert(extent, 0);
            self.free.extend(extent * EXTENT..(extent + 1) * EXTENT);
        }
        let physical = self.free.pop_first().unwrap();
        *self.extents.get_mut(&(physical / EXTENT)).unwrap() += 1;
        physical
    }
    fn extra(&self, new_pages: usize, private: u64) -> u64 {
        let extents = new_pages.saturating_sub(self.free.len()).div_ceil(EXTENT);
        private + new_pages as u64 * PAGE_META + extents as u64 * EXTENT as u64 * PAGE_BYTES
    }
}
impl Index {
    pub fn can_pin(&self, id: u64) -> Result<bool> {
        ensure!(
            self.state.entries.contains_key(&id),
            "missing session pin source"
        );
        ensure!(
            self.epoch < u64::MAX && self.state.next_lease < u64::MAX,
            "session pin counter exhausted"
        );
        Ok(self.state.leases.len() < MAX_LEASES
            && self
                .used()
                .checked_add(LEASE_META)
                .is_some_and(|n| n <= self.budget))
    }
    pub fn pin(&mut self, id: u64) -> Result<Lease> {
        ensure!(self.can_pin(id)?, "session pin capacity");
        let serial = self.state.next_lease;
        self.state.next_lease += 1;
        self.state.leases.insert(serial, id);
        self.epoch += 1;
        Ok(Lease {
            owner: self.owner,
            serial,
            entry: id,
        })
    }
    pub fn validate_lease(&self, lease: &Lease) -> Result<()> {
        ensure!(
            lease.owner == self.owner
                && self.state.leases.get(&lease.serial) == Some(&lease.entry)
                && self.state.entries.contains_key(&lease.entry),
            "stale or foreign session lease"
        );
        Ok(())
    }
    pub fn release(&mut self, lease: &Lease) -> Result<()> {
        self.validate_lease(lease)?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session epoch overflow"))?;
        self.state.leases.remove(&lease.serial);
        self.epoch = epoch;
        Ok(())
    }
    pub fn gpu_bytes(&self) -> u64 {
        self.state.gpu_bytes()
    }
    pub fn planned_gpu_bytes(&self, plan: &Plan) -> Result<u64> {
        Ok(self.planned_state(plan)?.gpu_bytes())
    }
    pub fn planned_extents(&self, plan: &Plan) -> Result<Vec<usize>> {
        Ok(self.planned_state(plan)?.extents.keys().copied().collect())
    }
    fn planned_state<'a>(&'a self, plan: &'a Plan) -> Result<&'a State> {
        ensure!(
            plan.owner == self.owner && plan.epoch == self.epoch,
            "stale or foreign session plan"
        );
        Ok(plan.next.as_ref().unwrap_or(&self.state))
    }
    pub fn entry(&self, id: u64) -> Result<(Arc<[i32]>, Vec<usize>)> {
        let entry = self
            .state
            .entries
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("missing session entry"))?;
        Ok((
            entry.ids.clone(),
            entry
                .pages
                .iter()
                .map(|k| self.state.pages[k].physical)
                .collect(),
        ))
    }
    pub fn new(namespace: Hash, budget: u64) -> Self {
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            namespace,
            budget,
            state: State::default(),
            memo: std::cell::Cell::new(None),
        }
    }
    pub fn used(&self) -> u64 {
        self.state.used()
    }
    /// IDs are immutable packed storage. state_bytes must be the actual private
    /// recurrent/conv/ring payload size; page tails and IDs are charged here.
    pub fn plan(&self, ids: Arc<[i32]>, state_bytes: u64) -> Result<Plan> {
        ensure!(
            !ids.is_empty()
                && ids.len() < CONTEXT
                && ids.len().is_multiple_of(64)
                && ids.iter().all(|&t| crate::vision_input::prompt_token(t)),
            "pure-prefix token bounds/grid"
        );
        let mut plan = Plan {
            owner: self.owner,
            epoch: self.epoch,
            next: None,
            status: Status::Skipped,
            entry: None,
            evicted: Vec::new(),
            new_pages: Vec::new(),
        };
        let key = hash_ids(&self.namespace, &ids);
        if let Some(id) = self.state.by_key.get(&key) {
            ensure!(self.state.entries[id].ids == ids, "session key collision");
            let mut next = self.state.clone();
            let used = next.tick()?;
            next.entries.get_mut(id).unwrap().used = used;
            plan.next = Some(next);
            plan.status = Status::Duplicate;
            plan.entry = Some(*id);
            return Ok(plan);
        }
        let keys = page_keys(&self.namespace, &ids);
        let tail = ids.len() - keys.len() * PAGE;
        let private = state_bytes
            .checked_add(
                tail as u64 * LATENT_ROW
                    + (tail / 4) as u64 * POOLED_ROW
                    + ids.len() as u64 * 4
                    + ENTRY_META
                    + keys.len() as u64 * ENTRY_PAGE_META,
            )
            .ok_or_else(|| anyhow::anyhow!("session payload size overflow"))?;
        // Reject an oversized standalone snapshot BEFORE pinning or evicting.
        let floor = private
            .checked_add(
                keys.len().div_ceil(EXTENT) as u64 * EXTENT as u64 * PAGE_BYTES
                    + keys.len() as u64 * PAGE_META,
            )
            .ok_or_else(|| anyhow::anyhow!("session footprint overflow"))?;
        if floor > self.budget {
            return Ok(plan);
        }
        let mut next = self.state.clone();
        let mut new = 0;
        for key in &keys {
            if let Some(page) = next.pages.get_mut(key) {
                page.refs += 1;
            } else {
                new += 1;
            }
        }
        let mut evicted = Vec::new();
        while next
            .used()
            .checked_add(next.extra(new, private))
            .is_none_or(|n| n > self.budget)
            || next.entries.len() >= MAX_ENTRIES
        {
            let victim = next
                .entries
                .iter()
                .filter(|(id, _)| !next.leases.values().any(|p| p == *id))
                .min_by_key(|(id, e)| (e.used, **id))
                .map(|(id, _)| *id);
            let Some(victim) = victim else {
                // Fragmented pinned extents can exceed a standalone floor.
                // Abandon the simulation and retain the ORIGINAL cache intact.
                return Ok(plan);
            };
            next.remove(victim);
            evicted.push(victim);
        }
        for (logical, key) in keys.iter().enumerate() {
            if !next.pages.contains_key(key) {
                let physical = next.allocate();
                next.pages.insert(*key, PageRef { physical, refs: 1 });
                plan.new_pages.push((logical, physical));
            }
        }
        let id = next.next_id;
        next.next_id = id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session ID overflow"))?;
        let used = next.tick()?;
        next.private += private;
        next.entries.insert(
            id,
            Entry {
                ids,
                key,
                pages: keys,
                private,
                used,
            },
        );
        next.by_key.insert(key, id);
        plan.status = Status::Saved;
        plan.entry = Some(id);
        plan.evicted = evicted;
        plan.next = Some(next);
        Ok(plan)
    }
    pub fn commit(&mut self, plan: Plan) -> Result<()> {
        ensure!(
            plan.owner == self.owner && plan.epoch == self.epoch,
            "stale or foreign session plan"
        );
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session epoch overflow"))?;
        if let Some(next) = plan.next {
            self.state = next;
        }
        self.epoch = epoch;
        Ok(())
    }
    pub fn find(&self, request: &[i32]) -> Option<u64> {
        // Compare packed slices, not per-token heap objects or a hash alone.
        self.state
            .entries
            .iter()
            .filter(|(_, e)| e.ids.len() < request.len() && request.starts_with(&e.ids))
            .max_by_key(|(id, e)| (e.ids.len(), e.used, **id))
            .map(|(id, _)| *id)
    }
    /// The longest prefix `request` shares with any entry's tokens.
    pub fn common_prefix(&self, request: &[i32]) -> usize {
        self.state
            .entries
            .values()
            .map(|e| common_prefix(&e.ids, request))
            .max()
            .unwrap_or(0)
    }
    /// Whether an entry holds exactly `request[..len]`.
    pub fn holds(&self, request: &[i32], len: usize) -> bool {
        self.state
            .entries
            .values()
            .any(|e| e.ids.len() == len && request.starts_with(&e.ids))
    }
    pub fn touch(&mut self, id: u64) -> Result<()> {
        ensure!(
            self.state.entries.contains_key(&id),
            "missing session entry"
        );
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session epoch overflow"))?;
        let used = self.state.tick()?;
        self.state.entries.get_mut(&id).unwrap().used = used;
        self.epoch = epoch;
        Ok(())
    }
    pub fn digest(&self) -> Hash {
        let key = [
            self.epoch,
            self.state.clock,
            self.state.entries.len() as u64,
            self.state.pages.len() as u64,
        ];
        if let Some((at, hash)) = self.memo.get()
            && at == key
        {
            return hash;
        }
        let hash = self.digest_state(&self.state, self.epoch);
        self.memo.set(Some((key, hash)));
        hash
    }
    pub fn planned_digest(&self, plan: &Plan) -> Result<Hash> {
        ensure!(
            plan.owner == self.owner && plan.epoch == self.epoch,
            "stale or foreign session plan"
        );
        Ok(self.digest_state(
            plan.next.as_ref().unwrap_or(&self.state),
            self.epoch
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("session epoch overflow"))?,
        ))
    }
    fn digest_state(&self, state: &State, epoch: u64) -> Hash {
        let mut h = Sha256::new();
        h.update(b"alaska-rust-session-index-v2-pins");
        h.update(self.namespace);
        for n in [
            self.budget,
            epoch,
            state.private,
            state.clock,
            state.next_id,
            state.entries.len() as u64,
            state.pages.len() as u64,
            state.extents.len() as u64,
            state.free.len() as u64,
            state.leases.len() as u64,
            state.next_lease,
        ] {
            h.update(n.to_le_bytes());
        }
        for (id, e) in &state.entries {
            for n in [
                *id,
                e.ids.len() as u64,
                e.private,
                e.used,
                e.pages.len() as u64,
            ] {
                h.update(n.to_le_bytes());
            }
            h.update(e.key);
            for k in &e.pages {
                h.update(k);
            }
        }
        for (key, p) in &state.pages {
            h.update(key);
            h.update((p.physical as u64).to_le_bytes());
            h.update((p.refs as u64).to_le_bytes());
        }
        for (id, n) in &state.extents {
            h.update((*id as u64).to_le_bytes());
            h.update((*n as u64).to_le_bytes());
        }
        for n in &state.free {
            h.update((*n as u64).to_le_bytes());
        }
        for (serial, entry) in &state.leases {
            h.update(serial.to_le_bytes());
            h.update(entry.to_le_bytes());
        }
        h.finalize().into()
    }
    pub fn check(&self) -> Result<()> {
        let s = &self.state;
        ensure!(
            s.used() <= self.budget && s.entries.len() <= MAX_ENTRIES,
            "session budget/count"
        );
        ensure!(
            s.leases.len() <= MAX_LEASES
                && s.leases
                    .iter()
                    .all(|(serial, id)| *serial < s.next_lease && s.entries.contains_key(id)),
            "session pin ownership"
        );
        ensure!(s.entries.len() == s.by_key.len(), "session key index size");
        let mut refs = BTreeMap::<Hash, usize>::new();
        let mut private = 0;
        for (id, e) in &s.entries {
            ensure!(
                s.by_key.get(&e.key) == Some(id)
                    && e.key == hash_ids(&self.namespace, &e.ids)
                    && e.pages == page_keys(&self.namespace, &e.ids)
                    && e.used <= s.clock
                    && *id < s.next_id,
                "session entry identity"
            );
            private += e.private;
            for key in &e.pages {
                *refs.entry(*key).or_default() += 1;
            }
        }
        ensure!(
            private == s.private && refs.len() == s.pages.len(),
            "session accounting"
        );
        let mut physical = BTreeSet::new();
        let mut extents = BTreeMap::<usize, usize>::new();
        for (key, p) in &s.pages {
            ensure!(
                refs.get(key) == Some(&p.refs)
                    && physical.insert(p.physical)
                    && !s.free.contains(&p.physical),
                "session page ownership"
            );
            *extents.entry(p.physical / EXTENT).or_default() += 1;
        }
        ensure!(extents == s.extents, "session extent ownership");
        let free: BTreeSet<_> = extents
            .keys()
            .flat_map(|e| e * EXTENT..(e + 1) * EXTENT)
            .filter(|p| !physical.contains(p))
            .collect();
        ensure!(free == s.free, "session free page ownership");
        Ok(())
    }
}

pub fn probe() -> Result<serde_json::Value> {
    let mut ranks: Vec<_> = (0..4)
        .map(|_| Index::new([19; 32], 160 * 1024 * 1024))
        .collect();
    let mut saved = 0;
    let mut skipped = 0;
    let mut duplicates = 0;
    let mut evicted = 0;
    let mut peak = 0;
    let mut seed = 729u64;
    for round in 0..1000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let length = if round % 37 == 0 {
            65536
        } else {
            64 * (1 + ((seed >> 16) as usize % 160))
        };
        let mut ids = vec![7; length];
        if round % 5 != 0 {
            ids[length.saturating_sub(512)] = 10 + round % 13;
        }
        let ids: Arc<[i32]> = ids.into();
        let mut plans = Vec::new();
        for rank in &ranks {
            plans.push(rank.plan(ids.clone(), 2 * 1024 * 1024)?);
        }
        let expected = ranks[0].planned_digest(&plans[0])?;
        for (rank, plan) in ranks.iter().zip(&plans) {
            ensure!(
                rank.planned_digest(plan)? == expected
                    && plan.status == plans[0].status
                    && plan.entry == plans[0].entry
                    && plan.evicted == plans[0].evicted
                    && plan.new_pages == plans[0].new_pages,
                "session decisions differ"
            );
        }
        match plans[0].status {
            Status::Saved => saved += 1,
            Status::Duplicate => duplicates += 1,
            Status::Skipped => skipped += 1,
        }
        evicted += plans[0].evicted.len();
        for (rank, plan) in ranks.iter_mut().zip(plans) {
            rank.commit(plan)?;
            rank.check()?;
            ensure!(
                rank.digest() == expected,
                "committed session digest differs"
            );
            let mut request = ids.to_vec();
            request.push(9);
            if let Some(id) = rank.find(&request) {
                rank.touch(id)?;
            }
            peak = peak.max(rank.used());
        }
        ensure!(
            ranks.iter().all(|r| r.digest() == ranks[0].digest()),
            "lookup/touch digest differs"
        );
    }
    Ok(
        serde_json::json!({"rounds":1000,"simulated_ranks":4,"saved":saved,"duplicates":duplicates,"skipped":skipped,"evicted":evicted,"peak_charged_bytes":peak,"budget_bytes":ranks[0].budget,"digest":format!("{:x}",sha2::digest::generic_array::GenericArray::from(ranks[0].digest())),"scope":"CPU metadata and payload accounting only; no GPU/SSD allocation, collective transport, or persistence."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn save(index: &mut Index, ids: &[i32], bytes: u64) -> u64 {
        let plan = index.plan(ids.into(), bytes).unwrap();
        assert_eq!(plan.status, Status::Saved);
        let id = plan.entry.unwrap();
        index.commit(plan).unwrap();
        index.check().unwrap();
        id
    }
    #[test]
    fn counted_pins_keep_sources_charged_and_reject_stale_plans() {
        let private = 64 * LATENT_ROW + 16 * POOLED_ROW + 64 * 4 + ENTRY_META;
        let mut x = Index::new([0; 32], 3 * private + 1024);
        let a = save(&mut x, &[7; 64], 0);
        let b = save(&mut x, &[8; 64], 0);
        let c = save(&mut x, &[9; 64], 0);
        let before = x.used();
        let stale = x.plan(vec![10; 64].into(), 0).unwrap();
        let one = x.pin(a).unwrap();
        let two = x.pin(a).unwrap();
        assert_eq!(x.used(), before + 2 * LEASE_META);
        let pinned = x.digest();
        assert!(x.commit(stale).is_err());
        assert_eq!(x.digest(), pinned);
        let p = x.plan(vec![10; 64].into(), 0).unwrap();
        assert_eq!(p.evicted, [b]);
        let d = p.entry.unwrap();
        x.commit(p).unwrap();
        x.release(&one).unwrap();
        x.validate_lease(&two).unwrap();
        let pc = x.pin(c).unwrap();
        let pd = x.pin(d).unwrap();
        let before = x.digest();
        let denied = x.plan(vec![11; 64].into(), 0).unwrap();
        assert_eq!(denied.status, Status::Skipped);
        assert!(denied.evicted.is_empty());
        assert_eq!(x.digest(), before);
        let stale = x.plan(vec![12; 64].into(), 0).unwrap();
        x.release(&two).unwrap();
        assert!(x.commit(stale).is_err());
        assert!(x.release(&one).is_err());
        let p = x.plan(vec![13; 64].into(), 0).unwrap();
        assert_eq!(p.evicted, [a]);
        x.commit(p).unwrap();
        x.release(&pc).unwrap();
        x.release(&pd).unwrap();
        x.check().unwrap();
    }
    #[test]
    fn pin_bound_owner_identity_and_rank_digest() {
        let mut a = Index::new([5; 32], 128 << 20);
        let mut b = Index::new([5; 32], 128 << 20);
        let x = save(&mut a, &[7; 320], 4096);
        let y = save(&mut b, &[7; 320], 4096);
        let mut leases = Vec::new();
        for _ in 0..MAX_LEASES {
            let p = a.pin(x).unwrap();
            let q = b.pin(y).unwrap();
            assert_eq!(a.digest(), b.digest());
            let before = b.digest();
            assert!(b.release(&p).is_err());
            assert_eq!(b.digest(), before);
            leases.push((p, q));
        }
        let before = a.digest();
        assert!(!a.can_pin(x).unwrap());
        assert!(a.pin(x).is_err());
        assert_eq!(a.digest(), before);
        for (p, q) in leases {
            a.release(&p).unwrap();
            b.release(&q).unwrap();
        }
        a.check().unwrap();
        b.check().unwrap();
        assert_eq!(a.digest(), b.digest());
    }
    #[test]
    fn pinned_shared_pages_survive_sibling_eviction() {
        let mut x = Index::new([0; 32], 32 << 20);
        let a = save(&mut x, &[7; 320], 4096);
        let lease = x.pin(a).unwrap();
        let mapping = x.entry(a).unwrap();
        for i in 0..20 {
            let mut branch = vec![7; 576];
            branch[575] = 10 + i;
            let p = x.plan(branch.into(), 4096).unwrap();
            assert!(!p.evicted.contains(&a));
            x.commit(p).unwrap();
            x.check().unwrap();
            assert_eq!(x.entry(a).unwrap(), mapping);
        }
        x.release(&lease).unwrap();
        x.check().unwrap();
    }
    #[test]
    fn key_grid_tail_and_dependency() {
        assert_eq!(page_keys(&[0; 32], &[7; 256]).len(), 0);
        assert_eq!(page_keys(&[0; 32], &[7; 320]).len(), 1);
        let a = vec![7; 576];
        let mut b = a.clone();
        b[256] = 8;
        assert_ne!(page_keys(&[0; 32], &a)[0], page_keys(&[0; 32], &b)[0]);
        b = a.clone();
        b[512] = 8;
        assert_eq!(page_keys(&[0; 32], &a)[0], page_keys(&[0; 32], &b)[0]);
        assert_ne!(page_keys(&[0; 32], &a)[1], page_keys(&[0; 32], &b)[1]);
        assert_ne!(page_keys(&[0; 32], &a), page_keys(&[1; 32], &a));
        let x = Index::new([0; 32], 1 << 30);
        for ids in [
            vec![],
            vec![7; 63],
            vec![-1; 64],
            vec![154880; 64],
            vec![7; CONTEXT],
        ] {
            assert!(x.plan(ids.into(), 0).is_err());
        }
    }
    #[test]
    fn shared_pages_pin_before_lru_eviction_and_tail_is_private() {
        let state = 1024;
        let ids = vec![7; 320];
        let first_private =
            state + 64 * LATENT_ROW + 16 * POOLED_ROW + 320 * 4 + ENTRY_META + ENTRY_PAGE_META;
        let budget = EXTENT as u64 * PAGE_BYTES + PAGE_META + first_private + 1024;
        let mut x = Index::new([0; 32], budget);
        let a = save(&mut x, &ids, state);
        let mut next = ids.clone();
        next[319] = 8;
        let p = x.plan(next.into(), state).unwrap();
        assert_eq!(p.status, Status::Saved);
        assert_eq!(p.evicted, vec![a]);
        assert!(p.new_pages.is_empty());
        x.commit(p).unwrap();
        x.check().unwrap();
        assert_eq!(x.state.pages.len(), 1);
        assert_eq!(x.used(), budget - 1024);
    }
    #[test]
    fn oversize_does_not_evict_and_bad_commit_is_atomic() {
        let mut x = Index::new([0; 32], 32 << 20);
        save(&mut x, &[7; 320], 1024);
        let before = x.digest();
        let p = x.plan(vec![7; 32768].into(), 1024).unwrap();
        assert_eq!(p.status, Status::Skipped);
        assert!(p.evicted.is_empty());
        assert_eq!(x.digest(), before);
        assert!(p.next.is_none());
        let foreign = Index::new([0; 32], 32 << 20)
            .plan(vec![7; 64].into(), 0)
            .unwrap();
        assert!(x.commit(foreign).is_err());
        assert_eq!(x.digest(), before);
        let stale = x.plan(vec![7; 64].into(), 0).unwrap();
        x.touch(0).unwrap();
        let touched = x.digest();
        assert!(x.commit(stale).is_err());
        assert_eq!(x.digest(), touched);
        assert!(x.plan(vec![7; 64].into(), u64::MAX).is_err());
        assert_eq!(x.digest(), touched);
    }
    #[test]
    fn exact_prefix_lookup_duplicate_touch_and_truncation() {
        let mut x = Index::new([0; 32], 128 << 20);
        let short = save(&mut x, &[7; 64], 0);
        let long = save(&mut x, &[7; 320], 0);
        assert_eq!(x.find(&[7; 321]), Some(long));
        assert_eq!(x.find(&[7; 320]), Some(short));
        assert_eq!(x.find(&[7; 64]), None);
        assert_eq!(x.find(&[8; 321]), None);
        let before = x.used();
        let p = x.plan(vec![7; 320].into(), 0).unwrap();
        assert_eq!(p.status, Status::Duplicate);
        assert_eq!(p.entry, Some(long));
        assert!(p.new_pages.is_empty());
        x.commit(p).unwrap();
        assert_eq!(x.used(), before);
        x.check().unwrap();
        let mut branch = vec![7; 320];
        branch[63] = 8;
        save(&mut x, &branch, 0);
        branch.push(9);
        assert_ne!(x.find(&branch), Some(long));
    }
    #[test]
    fn fragmented_pins_cannot_destroy_cache_on_failed_admission() {
        let extent = EXTENT as u64 * PAGE_BYTES;
        let mut x = Index::new([0; 32], 3 * extent);
        save(&mut x, &[7; 320], 0); // prefix page at physical 0
        save(&mut x, &vec![8; 15 * PAGE + 64], 0); // fill physical 1..15
        save(&mut x, &[7; 576], 0); // shares 0, new prefix page at physical 16
        assert_eq!(x.state.extents.len(), 2);
        let before = x.digest();
        let mut next = vec![7; 576];
        next[575] = 9;
        let plan = x.plan(next.into(), extent + 1).unwrap();
        // Standalone: one extent + private fits. In place: the two pinned
        // extents + private cannot fit even after every old entry is removed.
        assert_eq!(plan.status, Status::Skipped);
        assert!(plan.evicted.is_empty() && plan.new_pages.is_empty() && plan.next.is_none());
        assert_eq!(x.digest(), before);
        x.check().unwrap();
    }
    #[test]
    fn restore_touch_changes_lru_victim() {
        let private = 64 * LATENT_ROW + 16 * POOLED_ROW + 64 * 4 + ENTRY_META;
        let mut x = Index::new([0; 32], 3 * private);
        let first = save(&mut x, &[7; 64], 0);
        let second = save(&mut x, &[8; 64], 0);
        save(&mut x, &[9; 64], 0);
        x.touch(first).unwrap();
        let plan = x.plan(vec![10; 64].into(), 0).unwrap();
        assert_eq!(plan.evicted, vec![second]);
        x.commit(plan).unwrap();
        x.check().unwrap();
        assert!(x.find(&[7; 65]).is_some());
        assert!(x.find(&[8; 65]).is_none());
    }
    #[test]
    fn deterministic_four_rank_accounting_under_eviction() {
        probe().unwrap();
    }
}
