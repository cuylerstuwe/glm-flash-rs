//! Conservative GB10 page-cache credit for unified-memory admission.
//! CUDA's immediate-free figure excludes reclaimable Linux file pages on UMA.
//! No swap credit, cache dropping, allocator cache, or change to cache budgets.
use std::{collections::BTreeMap, io::Read};

const KEEP: u64 = 2 << 30;
const KEYS: [&str; 6] = [
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Dirty",
    "Writeback",
    "Mapped",
];

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct View {
    pub integrated: bool,
    pub cuda_free: u64,
    pub cuda_total: u64,
    pub page_cache_credit: u64,
    pub usable: u64,
    pub host_available: Option<u64>,
}

fn parse(text: &str) -> Option<BTreeMap<&str, u64>> {
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let Some((key, tail)) = line.split_once(':') else {
            continue;
        };
        if !KEYS.contains(&key) {
            continue;
        }
        let mut parts = tail.split_whitespace();
        let number = parts.next()?.parse::<u64>().ok()?.checked_mul(1024)?;
        if parts.next()? != "kB" || parts.next().is_some() || values.insert(key, number).is_some() {
            return None;
        }
    }
    if KEYS.iter().any(|key| !values.contains_key(key))
        || values["MemTotal"] == 0
        || values.values().any(|n| *n > values["MemTotal"])
    {
        return None;
    }
    Some(values)
}

pub(crate) fn view(free: usize, total: usize, integrated: bool, text: &str) -> View {
    let mi = parse(text);
    let credit = if integrated {
        mi.as_ref().map_or(0, |m| {
            m["MemAvailable"]
                .saturating_sub(m["MemFree"])
                .saturating_sub(m["Dirty"])
                .saturating_sub(m["Writeback"])
                .saturating_sub(m["Mapped"].max(KEEP))
        })
    } else {
        0
    };
    let credit = credit.min((total as u64).saturating_sub(free as u64));
    View {
        integrated,
        cuda_free: free as u64,
        cuda_total: total as u64,
        page_cache_credit: credit,
        usable: (free as u64).saturating_add(credit),
        host_available: mi.as_ref().map(|m| m["MemAvailable"]),
    }
}

pub(crate) fn read(free: usize, total: usize, integrated: bool) -> View {
    let mut text = String::new();
    if integrated {
        let result = std::fs::File::open("/proc/meminfo")
            .and_then(|f| f.take(65537).read_to_string(&mut text));
        if result.is_err() || text.len() > 65536 {
            text.clear();
        }
    }
    view(free, total, integrated, &text)
}

#[cfg(test)]
mod tests {
    use super::*;
    const G: usize = 1 << 30;
    fn fixture(
        free: usize,
        available: usize,
        dirty: usize,
        writeback: usize,
        mapped: usize,
    ) -> String {
        format!(
            "MemTotal: {} kB\nMemFree: {} kB\nMemAvailable: {} kB\nDirty: {} kB\nWriteback: {} kB\nMapped: {} kB\nSwapFree: 999999999 kB\n",
            128 * G / 1024,
            free / 1024,
            available / 1024,
            dirty / 1024,
            writeback / 1024,
            mapped / 1024
        )
    }
    #[test]
    fn page_cache_credit_preserves_margins() {
        let v = view(
            3 * G,
            128 * G,
            true,
            &fixture(3 * G, 60 * G, G, 2 * G, 4 * G),
        );
        assert_eq!(v.page_cache_credit, 50 * G as u64);
        assert_eq!(v.usable, 53 * G as u64);
        let v = view(3 * G, 128 * G, true, &fixture(3 * G, 60 * G, 0, 0, G));
        assert_eq!(v.usable, 58 * G as u64);
    }
    #[test]
    fn no_credit_for_discrete_missing_invalid_or_pressure() {
        let valid = fixture(3 * G, 60 * G, 0, 0, 0);
        assert_eq!(view(3 * G, 128 * G, false, &valid).usable, 3 * G as u64);
        for text in [
            String::new(),
            valid.replace("MemAvailable", "Absent"),
            valid.replace("kB", "MB"),
            valid.clone() + "Dirty: 0 kB\n",
            valid.replace("Dirty: 0", "Dirty: -1"),
            valid.replace("Dirty: 0", "Dirty: 18446744073709551615"),
            fixture(3 * G, 129 * G, 0, 0, 0),
            fixture(3 * G, 4 * G, G, G, G),
        ] {
            assert_eq!(view(3 * G, 128 * G, true, &text).usable, 3 * G as u64);
        }
    }
    #[test]
    fn credit_cannot_exceed_device_total_or_count_swap() {
        let v = view(3 * G, 4 * G, true, &fixture(3 * G, 60 * G, 0, 0, 0));
        assert_eq!(v.usable, 4 * G as u64);
        assert_eq!(
            view(3 * G, 128 * G, true, &fixture(3 * G, 3 * G, 0, 0, 0)).usable,
            3 * G as u64
        );
    }
}
