//! Bounded graph retention policy. CUDA owners remain outside this CPU-only module.
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    bounded: bool,
    entries: usize,
    nodes: usize,
    sightings: usize,
    after: u8,
}
impl Policy {
    pub(crate) fn multi_request() -> Self {
        Self {
            bounded: true,
            entries: 96,
            nodes: 1_048_576,
            sightings: 192,
            after: 1,
        }
    }
    pub(crate) fn parse(value: Option<&str>) -> Result<Self> {
        ensure!(
            matches!(value, None | Some("0") | Some("1")),
            "ALASKA_VERIFY_GRAPH_CACHE must be 0/1"
        );
        Ok(if value == Some("1") {
            // A lone stream repeats each (slot, parity, rows, geometry) key many
            // times; an eager verifier round is launch-bound and costs more than
            // capturing, so capture on first sight.
            // Startup prewarming retains 896 graphs across four slots; each
            // costs about 6 MiB of device memory regardless of node count.
            Self {
                bounded: true,
                entries: 1408,
                nodes: 4_194_304,
                sightings: 4096,
                after: 1,
            }
        } else {
            Self {
                bounded: false,
                entries: 32,
                nodes: usize::MAX,
                sightings: 0,
                after: 1,
            }
        })
    }
    pub(crate) fn environment() -> Result<Self> {
        match std::env::var("ALASKA_VERIFY_GRAPH_CACHE") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn identity(self) -> [u8; 40] {
        let mut bytes = [0; 40];
        for (chunk, value) in bytes.as_chunks_mut::<8>().0.iter_mut().zip([
            u64::from(self.bounded),
            self.entries as u64,
            self.nodes as u64,
            self.sightings as u64,
            u64::from(self.after),
        ]) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Replay,
    Eager,
    Capture,
}
struct Entry<V> {
    value: V,
    nodes: usize,
    used: u64,
}
struct Seen {
    count: u8,
    used: u64,
}
#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) hits: u64,
    pub(crate) eager: u64,
    pub(crate) captures: u64,
    pub(crate) evictions: u64,
    pub(crate) oversized: u64,
    pub(crate) retained_graphs: usize,
    pub(crate) retained_nodes: usize,
    pub(crate) peak_graphs: usize,
    pub(crate) peak_nodes: usize,
    pub(crate) largest_graph_nodes: usize,
    pub(crate) tracked_sightings: usize,
}
pub(crate) struct Cache<K, V> {
    policy: Policy,
    entries: BTreeMap<K, Entry<V>>,
    seen: BTreeMap<K, Seen>,
    clock: u64,
    stats: Stats,
}
impl<K: Ord + Clone, V> Cache<K, V> {
    pub(crate) fn new(policy: Policy) -> Self {
        Self {
            policy,
            entries: BTreeMap::new(),
            seen: BTreeMap::new(),
            clock: 0,
            stats: Stats::default(),
        }
    }
    pub(crate) fn stats(&self) -> Stats {
        Stats {
            retained_graphs: self.entries.len(),
            retained_nodes: self.stats.retained_nodes,
            tracked_sightings: self.seen.len(),
            ..self.stats
        }
    }
    pub(crate) fn bounded(&self) -> bool {
        self.policy.bounded
    }
    fn tick(&mut self) -> u64 {
        // Saturation is safe: equal-age entries have deterministic key order.
        self.clock = self.clock.saturating_add(1);
        self.clock
    }
    pub(crate) fn action(&mut self, key: &K) -> Action {
        let now = self.tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.used = now;
            self.stats.hits += 1;
            return Action::Replay;
        }
        if !self.policy.bounded {
            if self.entries.len() >= self.policy.entries {
                self.stats.evictions += self.entries.len() as u64;
                self.entries.clear();
                self.stats.retained_nodes = 0;
            }
            return Action::Capture;
        }
        if !self.seen.contains_key(key) && self.seen.len() == self.policy.sightings {
            let oldest = self
                .seen
                .iter()
                .min_by_key(|(_, seen)| seen.used)
                .unwrap()
                .0
                .clone();
            self.seen.remove(&oldest);
        }
        let seen = self.seen.entry(key.clone()).or_insert(Seen {
            count: 0,
            used: now,
        });
        seen.count = seen.count.saturating_add(1);
        seen.used = now;
        if seen.count >= self.policy.after {
            Action::Capture
        } else {
            self.stats.eager += 1;
            Action::Eager
        }
    }
    pub(crate) fn get(&self, key: &K) -> &V {
        &self.entries[key].value
    }
    pub(crate) fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }
    /// Retained resources obey both caps. At most one just-captured candidate is
    /// temporarily owned by the caller. An individually oversized graph may be
    /// launched once but is returned rather than published into the cache.
    pub(crate) fn insert(&mut self, key: K, nodes: usize, value: V) -> Result<(), V> {
        self.stats.captures += 1;
        self.stats.largest_graph_nodes = self.stats.largest_graph_nodes.max(nodes);
        self.seen.remove(&key);
        if nodes > self.policy.nodes {
            self.stats.oversized += 1;
            return Err(value);
        }
        // A same-key replacement is not a normal lookup path, but ownership and
        // accounting remain correct if a diagnostic explicitly replaces one.
        if let Some(old) = self.entries.remove(&key) {
            self.stats.retained_nodes -= old.nodes;
        }
        while self.entries.len() >= self.policy.entries
            || self.stats.retained_nodes > self.policy.nodes - nodes
        {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .unwrap()
                .0
                .clone();
            let old = self.entries.remove(&oldest).unwrap();
            self.stats.retained_nodes -= old.nodes;
            self.stats.evictions += 1;
        }
        let used = self.tick();
        self.entries.insert(key, Entry { value, nodes, used });
        self.stats.retained_nodes += nodes;
        self.stats.peak_graphs = self.stats.peak_graphs.max(self.entries.len());
        self.stats.peak_nodes = self.stats.peak_nodes.max(self.stats.retained_nodes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn small(entries: usize, nodes: usize) -> Policy {
        Policy {
            bounded: true,
            entries,
            nodes,
            sightings: 4,
            after: 2,
        }
    }
    fn capture(cache: &mut Cache<usize, usize>, key: usize, nodes: usize) {
        assert_eq!(cache.action(&key), Action::Eager);
        assert_eq!(cache.action(&key), Action::Capture);
        cache.insert(key, nodes, key).unwrap();
    }
    #[test]
    fn default_keeps_original_clear_all_behavior_and_candidate_captures_first() -> Result<()> {
        let mut old = Cache::new(Policy::parse(None)?);
        for key in 0..32 {
            assert_eq!(old.action(&key), Action::Capture);
            old.insert(key, 3, key).unwrap();
        }
        assert_eq!(old.action(&31), Action::Replay);
        assert_eq!(old.action(&32), Action::Capture);
        assert_eq!(old.stats().retained_graphs, 0);
        let mut new = Cache::<usize, usize>::new(Policy::parse(Some("1"))?);
        assert_eq!(new.action(&1), Action::Capture);
        new.insert(1, 3, 1).unwrap();
        assert_eq!(new.action(&1), Action::Replay);
        assert_eq!(new.stats().eager, 0);
        assert_ne!(
            Policy::parse(None)?.identity(),
            Policy::parse(Some("1"))?.identity()
        );
        assert!(Policy::parse(Some("true")).is_err());
        Ok(())
    }
    #[test]
    fn cold_shape_scan_does_not_destroy_hot_graphs_or_grow_sightings() {
        let mut cache = Cache::new(small(2, 12));
        capture(&mut cache, 1, 4);
        capture(&mut cache, 2, 4);
        for key in 3..100 {
            assert_eq!(cache.action(&1), Action::Replay);
            assert_eq!(cache.action(&key), Action::Eager);
        }
        assert_eq!(cache.stats().retained_graphs, 2);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().tracked_sightings, 4);
        capture(&mut cache, 101, 4);
        assert_eq!(cache.action(&1), Action::Replay);
        assert_eq!(cache.action(&2), Action::Eager);
        assert_eq!(cache.stats().evictions, 1);
    }
    #[test]
    fn node_budget_evicts_only_enough_and_oversized_graph_is_returned() {
        let mut cache = Cache::new(small(5, 10));
        capture(&mut cache, 1, 4);
        capture(&mut cache, 2, 4);
        assert_eq!(cache.action(&1), Action::Replay);
        capture(&mut cache, 3, 5);
        assert_eq!(cache.stats().retained_nodes, 9);
        assert_eq!(cache.stats().retained_graphs, 2);
        assert_eq!(*cache.get(&1), 1);
        assert_eq!(cache.insert(4, 11, 99), Err(99));
        assert_eq!(cache.stats().retained_nodes, 9);
        assert_eq!(cache.stats().oversized, 1);
        assert!(cache.stats().peak_nodes <= 10);
    }
    #[test]
    fn repeated_four_slot_shapes_fit_without_warm_recapture() -> Result<()> {
        let mut cache = Cache::new(Policy::parse(Some("1"))?);
        for _ in 0..9 {
            for slot in 0..4 {
                for parity in 0..2 {
                    for rows in 1..=16 {
                        let key = (slot, parity, rows);
                        if cache.action(&key) == Action::Capture {
                            cache.insert(key, 2048, key).unwrap();
                        }
                    }
                }
            }
        }
        assert_eq!(cache.stats().captures, 128);
        assert_eq!(cache.stats().hits, 128 * 8);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().retained_nodes, 262144);
        Ok(())
    }
}
