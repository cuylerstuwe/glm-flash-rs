//! Automatic live/RAM selection. No active or paused owner may be a target.
use super::*;
use crate::cache_pages::{PAGE, Phase, SLOTS};
#[derive(Clone, Copy)]
pub(super) enum Source {
    Cold,
    Live,
    Ram(u64),
    Disk,
}
pub(super) struct Choice {
    pub slot: usize,
    pub cached: usize,
    pub source: Source,
}
impl Choice {
    pub fn keep(&self, long: &Long<'_>) -> Result<usize> {
        if matches!(self.source, Source::Disk) {
            return Ok(0);
        }
        Ok(self
            .cached
            .min(long.memory.book().slot(self.slot)?.pages().len() * PAGE))
    }
    pub fn digest(&self, scheduler: &Scheduler) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"request-cache-v1");
        h.update(scheduler.queue.digest());
        h.update(scheduler.cache.as_ref().map_or([0; 32], |s| s.digest()));
        for n in [
            self.slot as u64,
            self.cached as u64,
            match self.source {
                Source::Cold => 0,
                Source::Live => 1,
                Source::Ram(_) => 2,
                Source::Disk => 3,
            },
            match self.source {
                Source::Ram(id) => id,
                _ => 0,
            },
        ] {
            h.update(n.to_le_bytes());
        }
        h.finalize().into()
    }
}
// The compatible service's pure-prefill snapshot policy. A 64-row
// grid cut changes no bits: fast KDA blocks and attention rows are 64-aligned.
const GRID: usize = 64;
const TAIL: usize = 256;
const EVERY: usize = 16384;
const FORK_MIN: usize = 512;
const SHARE_MIN: usize = 2048;
const SHARE_MAX: usize = 131072;
/// Snapshot marks for a pure prompt resumed at `resume`: the fork point
/// (`shared`: the longest prefix common with a stored entry or an in-flight
/// prompt), the end of the system prompt (the first role token at or after
/// 2,048 tokens and within 131,072) and every 16,384 tokens, all on the grid,
/// at least 512 past the resume point and before the prompt's last grid point.
pub(super) fn marks(prompt: &[i32], resume: usize, shared: usize, roles: &[i32]) -> Vec<usize> {
    let limit = prompt.len() / GRID * GRID;
    let mut want = std::collections::BTreeSet::new();
    let mut add = |q: usize| {
        if q > 0 && q < limit && q >= resume + FORK_MIN {
            want.insert(q);
        }
    };
    add(shared.min(limit) / GRID * GRID);
    let end = prompt.len().min(SHARE_MAX + 1);
    if let Some(at) = prompt
        .get(SHARE_MIN..end)
        .and_then(|tail| tail.iter().position(|t| roles.contains(t)))
    {
        add((SHARE_MIN + at).min(limit) / GRID * GRID);
    }
    let mut m = (resume / EVERY + 1) * EVERY;
    while m < limit {
        add(m);
        m += EVERY;
    }
    want.into_iter().collect()
}
/// Where a pure prefill of [begin, n) in `step`-row slabs ends a slab and saves:
/// each grid mark, and the prompt's own snapshot at its last grid point strictly
/// before n, cut there unless the last slab already starts within 256 rows.
pub(super) fn snapshot_points(begin: usize, n: usize, step: usize, marks: &[usize]) -> Vec<usize> {
    let mut points: std::collections::BTreeSet<usize> = marks
        .iter()
        .copied()
        .filter(|&q| begin < q && q < n && q.is_multiple_of(GRID))
        .collect();
    let (mut start, mut last) = (begin, begin);
    for q in points.iter().copied().chain([n]) {
        while start < q {
            last = start;
            start = (start + step).min(q);
        }
    }
    let own = (n - 1) / GRID * GRID;
    let at = if last >= own || own - last <= TAIL {
        last
    } else {
        own
    };
    if at > begin {
        points.insert(at);
    }
    points.into_iter().collect()
}
impl Scheduler {
    /// Pure prefills follow the production snapshot plan; mixed (reply-resumed)
    /// prefixes and services without a store take no snapshots.
    pub(super) fn snapshot_marks(
        &self,
        long: &Long<'_>,
        id: u64,
        slot: usize,
    ) -> Result<Vec<usize>> {
        let Some(store) = &self.cache else {
            return Ok(Vec::new());
        };
        if long.sessions[slot].mixed {
            return Ok(Vec::new());
        }
        let r = self.queue.request(id)?;
        let shared = self
            .queue
            .admitted(id)
            .map(|p| crate::session_index::common_prefix(&p.tokens, &r.tokens))
            .chain([store.common_prefix(&r.tokens)])
            .max()
            .unwrap_or(0);
        let wanted: Vec<usize> = marks(&r.tokens, r.done, shared, &self.share_roles)
            .into_iter()
            .filter(|&m| !store.holds(&r.tokens, m))
            .collect();
        Ok(snapshot_points(
            r.done,
            r.tokens.len(),
            long.prefill_rows,
            &wanted,
        ))
    }
    /// The production service's prefix-share wait: a queued request waits while
    /// an unpaused partner still prefilling will save a mark inside their shared
    /// prefix at least 512 tokens past what this request can resume now. A
    /// foreground request never waits on background work.
    pub(super) fn share_wait(&self, long: &Long<'_>, id: u64) -> Result<bool> {
        if self.cache.is_none() || self.share_roles.is_empty() {
            return Ok(false);
        }
        let r = self.queue.request(id)?;
        if !r.drafts() {
            return Ok(false);
        }
        let mut have = None;
        for p in self.queue.admitted(id) {
            if p.paused || p.cancellation().is_some() || p.done >= p.tokens.len() {
                continue;
            }
            if !r.background && p.background {
                continue;
            }
            let shared = crate::session_index::common_prefix(&p.tokens, &r.tokens);
            let Some(ahead) = p
                .marks
                .iter()
                .flatten()
                .copied()
                .filter(|&q| p.done < q && q < r.tokens.len() && q <= shared)
                .max()
            else {
                continue;
            };
            let have = match have {
                Some(h) => h,
                None => *have.insert(self.cache_choice(long, id)?.cached),
            };
            if ahead >= have + FORK_MIN {
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Every rank receives rank 0's rows for `keys` (sorted, unique image row
    /// IDs of the remaining prompt), as the production service ships them.
    pub(super) fn vision_exchange(
        &mut self,
        long: &Long<'_>,
        id: u64,
        keys: Vec<i32>,
    ) -> Result<Option<(std::rc::Rc<[i32]>, crate::cuda::Buffer)>> {
        use crate::vision_service::ROW_BYTES;
        let r = self.queue.request(id)?;
        let rows = if long.engine.rank == 0 {
            self.broker
                .as_ref()
                .and_then(|b| b.vision(id, &r.tokens))
                .context("prompt image rows unavailable")
                .and_then(|table| table.select(&keys))
        } else {
            Ok(Vec::new())
        };
        let mut h = Sha256::new();
        h.update(b"request-vision-v1");
        h.update(id.to_le_bytes());
        for key in &keys {
            h.update(key.to_le_bytes());
        }
        if !self
            .wire
            .control
            .agree(long, 870, h.finalize().into(), rows.is_ok())?
        {
            return Ok(None);
        }
        let rows = rows?;
        let bytes = keys.len() * ROW_BYTES;
        let device = long.engine.device.alloc(bytes)?;
        const CHUNK: usize = 128 * ROW_BYTES;
        let send = long.engine.device.alloc(CHUNK)?;
        let recv = long.engine.device.alloc(4 * CHUNK)?;
        for start in (0..bytes).step_by(CHUNK) {
            let count = (bytes - start).min(CHUNK);
            if long.engine.rank == 0 {
                send.write(0, &rows[start..start + count])?;
            }
            unsafe {
                long.engine.gather_prefix(&send, &recv, count)?;
                device.copy_from(&long.engine.stream, start, &recv, 0, count)?;
            }
            long.engine.sync()?;
        }
        Ok(Some((keys.into(), device)))
    }
    pub(super) fn install_cache(&mut self, long: &Long<'_>, budget: u64) -> Result<()> {
        ensure!(
            self.queue.count() == 0 && self.cache.is_none(),
            "cache installation after admission"
        );
        self.cache = Some(super::super::store::Store::new(long, budget)?);
        Ok(())
    }
    pub(super) fn cache_choice(&self, long: &Long<'_>, id: u64) -> Result<Choice> {
        let request = self.queue.request(id)?;
        let first = self.queue.free_slot().context("no cache destination")?;
        let mut choice = Choice {
            slot: first,
            cached: 0,
            source: Source::Cold,
        };
        let Some(store) = &self.cache else {
            return Ok(choice);
        };
        // Prefer the least retained idle destination for cold/copy-in admission.
        // Live continuation overrides it only after validating actual mappings.
        let mut retained = usize::MAX;
        for slot in 0..SLOTS {
            if !self.queue.slot_free(slot) {
                continue;
            }
            let live = long.memory.book().slot(slot)?;
            ensure!(
                live.phase() == Phase::Idle,
                "unowned non-idle cache destination"
            );
            if live.pages().len() < retained {
                retained = live.pages().len();
                choice.slot = slot;
            }
        }
        // A request containing an idle slot's whole last prompt supersedes that
        // slot's reply (a regenerate or an edited reply). Resume in the slot
        // whose kept state covers the longest prefix.
        // Other live sessions stay retained, and the slot's graphs are warm.
        let mut superseded = 0;
        for slot in 0..SLOTS {
            if !self.queue.slot_free(slot) {
                continue;
            }
            let session = &long.sessions[slot];
            let shared = crate::session_index::common_prefix(&session.tokens, &request.tokens);
            if session.prompt_end > 0 && shared >= session.prompt_end && shared > superseded {
                superseded = shared;
                choice.slot = slot;
            }
        }
        let destination = choice.slot;
        for slot in 0..SLOTS {
            if !self.queue.slot_free(slot) {
                continue;
            }
            let session = &long.sessions[slot];
            if super::super::reuse::eligible(
                &session.tokens,
                session.mixed,
                long.memory.book().slot(slot)?.pages().len(),
                &request.tokens,
                request.output,
            )
            .is_ok()
                && session.tokens.len() > choice.cached
            {
                choice = Choice {
                    slot,
                    cached: session.tokens.len(),
                    source: Source::Live,
                };
            }
        }
        if let Some((entry, cached)) = store.find(long, &request.tokens)?
            && cached > choice.cached
        {
            choice = Choice {
                slot: destination,
                cached,
                source: Source::Ram(entry),
            };
        }
        if let Some(cached) = self
            .disk
            .as_ref()
            .map(|d| d.hint(&request.tokens, request.background))
            .transpose()?
            .flatten()
            && cached > choice.cached
        {
            choice = Choice {
                slot: destination,
                cached,
                source: Source::Disk,
            };
        }
        Ok(choice)
    }
}
#[path = "request_cache_tests.rs"]
mod tests;
pub use tests::run;
#[cfg(test)]
mod cpu {
    use super::*;
    #[test]
    fn snapshot_marks_and_cuts() {
        let roles = [154827, 154828, 154829];
        // Fresh prompts: the own snapshot is the last grid point before n, cut
        // unless the final slab starts within 256 rows of it.
        assert_eq!(snapshot_points(0, 14880, 8192, &[]), vec![14848]);
        assert_eq!(snapshot_points(0, 8192, 8192, &[]), vec![8128]);
        assert_eq!(snapshot_points(0, 8400, 8192, &[]), vec![8192]);
        assert!(snapshot_points(0, 200, 8192, &[]).is_empty());
        assert!(snapshot_points(0, 64, 8192, &[]).is_empty());
        // A resumed prompt never re-saves its resume point.
        assert!(snapshot_points(14848, 14880, 8192, &[]).is_empty());
        assert_eq!(snapshot_points(14848, 30000, 8192, &[]), vec![29952]);
        // Every 16,384 tokens past the resume point (at least 512 beyond it).
        let prompt = vec![7; 40000];
        assert_eq!(marks(&prompt, 0, 0, &roles), vec![16384, 32768]);
        assert_eq!(marks(&prompt, 16128, 0, &roles), vec![32768]);
        assert_eq!(marks(&prompt, 16000, 0, &roles), vec![32768]);
        assert_eq!(marks(&prompt, 15808, 0, &roles), vec![16384, 32768]);
        assert_eq!(
            snapshot_points(0, 40000, 8192, &[16384, 32768]),
            vec![16384, 32768, 39936]
        );
        // Fork point: the shared prefix on the grid, before the last grid point.
        assert_eq!(marks(&prompt, 0, 5000, &roles), vec![4992, 16384, 32768]);
        assert_eq!(marks(&prompt, 4800, 5000, &roles), vec![16384, 32768]);
        assert_eq!(marks(&prompt, 0, 40000, &roles), vec![16384, 32768]);
        // System prompt end: the first role token at or after 2,048 tokens.
        let mut chat = vec![7; 20000];
        chat[1000] = 154827;
        chat[9001] = 154828;
        assert_eq!(marks(&chat, 0, 0, &roles), vec![8960, 16384]);
        chat[3000] = 154829;
        assert_eq!(marks(&chat, 0, 0, &roles), vec![2944, 16384]);
        // Marks cut slabs; each cut restarts the slab grid.
        assert_eq!(
            snapshot_points(0, 20000, 8192, &[2944, 16384]),
            vec![2944, 16384, 19968]
        );
    }
}
