//! Native ownership of the retained five-pass radix selection for DSA pools.
#[path = "sparse_split_plan.rs"]
pub(crate) mod split_plan;
#[path = "sparse_split_probe.rs"]
mod split_probe;
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::load_kernel,
};
use anyhow::{Context, Result, ensure};
pub(crate) use split_probe::run as probe_split;
use std::{collections::BTreeMap, path::Path, rc::Rc, time::Instant};
const K: usize = 512;
const WIDTH: usize = 2051;
fn config(np: usize) -> (usize, usize) {
    let chunk = if np <= 16384 { 2048 } else { 4096 };
    (chunk, np.div_ceil(chunk).next_power_of_two())
}
pub struct Select {
    kernels: BTreeMap<(usize, usize), Vec<Kernel>>,
    expand: Kernel,
    totals: Buffer,
    hist: Buffer,
    greater: Buffer,
    pub pools: Buffer,
    pub tokens: Buffer,
    pub counts: Buffer,
    max_rows: usize,
    max_pools: usize,
}
impl Select {
    pub fn new(d: &Rc<Device>, root: &Path, max_rows: usize, max_pools: usize) -> Result<Self> {
        ensure!(
            max_rows > 0
                && max_rows <= 1024
                && (512..=262144).contains(&max_pools)
                && max_pools.is_multiple_of(16),
            "selector capacity"
        );
        let mut kernels = BTreeMap::new();
        for (chunk, cp) in [
            (2048, 1),
            (2048, 2),
            (2048, 4),
            (2048, 8),
            (4096, 8),
            (4096, 16),
            (4096, 32),
            (4096, 64),
        ] {
            let mut stages = Vec::new();
            for step in 0..5 {
                stages.push(load_kernel(
                    d,
                    root,
                    &format!("sparse_select_{chunk}_{cp}_{step}"),
                )?);
            }
            kernels.insert((chunk, cp), stages);
        }
        let cp = config(max_pools).1;
        Ok(Self {
            kernels,
            expand: load_kernel(d, root, "sparse_tokens")?,
            totals: d.alloc(max_rows * 1024 * 4)?,
            hist: d.alloc(max_rows * cp * 256 * 4)?,
            greater: d.alloc(max_rows * cp * 4)?,
            pools: d.alloc(max_rows * K * 8)?,
            tokens: d.alloc(max_rows * WIDTH * 4)?,
            counts: d.alloc(max_rows * 4)?,
            max_rows,
            max_pools,
        })
    }
    pub fn run(
        &self,
        s: &Stream,
        scores: &Buffer,
        position: &Buffer,
        rows: usize,
        np: usize,
    ) -> Result<()> {
        self.pools_into(s, scores, position, rows, np, &self.pools, 0)?;
        self.expand_from(s, &self.pools, 0, position, rows)
    }
    /// The retained five selection launches, writing only a checked pool region.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pools_into(
        &self,
        s: &Stream,
        scores: &Buffer,
        position: &Buffer,
        rows: usize,
        np: usize,
        pools: &Buffer,
        pool_offset: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && rows <= self.max_rows
                && (512..=self.max_pools).contains(&np)
                && np.is_multiple_of(16),
            "selector shape"
        );
        ensure!(
            scores.bytes >= rows * np * 4 && position.bytes >= 4,
            "selector storage"
        );
        let pool = pool_region(pools, pool_offset, rows)?;
        let (chunk, cp) = config(np);
        let kernels = self
            .kernels
            .get(&(chunk, cp))
            .context("selector configuration unavailable")?;
        unsafe {
            self.totals.zero_prefix(s, rows * 1024 * 4)?;
        }
        for kernel in kernels {
            words(
                kernel,
                s,
                [rows as u32, cp as u32, 1],
                &[
                    scores.ptr,
                    np as u64,
                    position.ptr,
                    self.totals.ptr,
                    self.hist.ptr,
                    self.greater.ptr,
                    pool,
                ],
            )?;
        }
        Ok(())
    }
    /// Expansion reads a previously selected or gathered region. It never reads
    /// stride padding, and retains the original token/count buffers and kernel.
    pub(crate) fn expand_from(
        &self,
        s: &Stream,
        pools: &Buffer,
        pool_offset: usize,
        position: &Buffer,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.max_rows && position.bytes >= 4,
            "selector expansion shape"
        );
        let pool = pool_region(pools, pool_offset, rows)?;
        words(
            &self.expand,
            s,
            [rows as u32, 1, 1],
            &[pool, self.tokens.ptr, self.counts.ptr, position.ptr],
        )
    }
}
fn pool_region(buffer: &Buffer, offset: usize, rows: usize) -> Result<u64> {
    let bytes = rows.checked_mul(K * 8).context("pool region overflow")?;
    ensure!(
        offset.is_multiple_of(8)
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= buffer.bytes),
        "pool region storage"
    );
    buffer
        .ptr
        .checked_add(offset as u64)
        .context("pool address overflow")
}
fn order(x: f32) -> u32 {
    let b = if x == 0. {
        0
    } else if x.is_nan() {
        0x7fc00000
    } else {
        x.to_bits()
    };
    if b & 0x80000000 != 0 {
        !b
    } else {
        b ^ 0x80000000
    }
}
fn i32s(x: &[u8]) -> Vec<i32> {
    x.as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_le_bytes(*v))
        .collect()
}
fn i64s(x: &[u8]) -> Vec<i64> {
    x.as_chunks::<8>()
        .0
        .iter()
        .map(|v| i64::from_le_bytes(*v))
        .collect()
}
pub fn probe(root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let rows = 4;
    let cap = 262144;
    let select = Select::new(&d, root, rows, cap)?;
    let scores = d.alloc(rows * cap * 4)?;
    let pos = d.alloc(4)?;
    let mut cases = Vec::new();
    let mut seed = 0xa634e017839fe51bu64;
    for np in [
        512usize, 1024, 2048, 2064, 4096, 8192, 16384, 16448, 32768, 65536, 131072, 262144,
    ] {
        for early in [false, true] {
            let position = if early {
                if np == 512 { 2047 } else { 2049 }
            } else {
                np * 4 - rows
            };
            for tied in [false, true] {
                let mut data = vec![0f32; rows * np];
                let mut expected = Vec::new();
                let mut tokens = Vec::new();
                let mut counts = Vec::new();
                for r in 0..rows {
                    let visible = np.min(((position + r + 1) / 4).max(K));
                    for (i, v) in data[r * np..(r + 1) * np].iter_mut().enumerate() {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        *v = if i >= visible {
                            f32::NAN
                        } else if tied {
                            0.
                        } else {
                            ((seed % 65536) as i32 - 32768) as f32 / 256.
                        };
                        if !tied && i < visible {
                            *v = match i % 1031 {
                                0 => f32::INFINITY,
                                1 => f32::NEG_INFINITY,
                                2 => f32::from_bits(0xffc12345),
                                3 => -0.,
                                4 => 0.,
                                _ => *v,
                            };
                        }
                    }
                    let mut ids: Vec<usize> = (0..visible).collect();
                    ids.sort_unstable_by(|&a, &b| {
                        order(data[r * np + b])
                            .cmp(&order(data[r * np + a]))
                            .then(a.cmp(&b))
                    });
                    ids.truncate(K);
                    ids.sort_unstable();
                    expected.extend(ids.iter().map(|i| *i as i64));
                    let complete = (position + r + 1) / 4;
                    let tail = (position + r + 1) % 4;
                    if complete > K {
                        tokens.extend(
                            ids.iter()
                                .flat_map(|p| (0..4).map(move |j| (p * 4 + j) as i32)),
                        );
                        tokens.extend((0..3).map(|j| {
                            if j < tail {
                                (complete * 4 + j) as i32
                            } else {
                                -1
                            }
                        }));
                        counts.push((2048 + tail) as i32);
                    } else {
                        tokens.extend(vec![-1; WIDTH]);
                        counts.push(0);
                    }
                }
                scores.write(
                    0,
                    &data
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                pos.write(0, &(position as i32).to_le_bytes())?;
                let run = || select.run(&s, &scores, &pos, rows, np);
                let check = || -> Result<()> {
                    ensure!(
                        i64s(&select.pools.read(0, rows * K * 8)?) == expected,
                        "pool set/order at np={np} early={early} tied={tied}"
                    );
                    ensure!(
                        i32s(&select.tokens.read(0, rows * WIDTH * 4)?) == tokens,
                        "expanded token order"
                    );
                    ensure!(
                        i32s(&select.counts.read(0, rows * 4)?) == counts,
                        "sparse visibility counts"
                    );
                    Ok(())
                };
                run()?;
                s.sync()?;
                check()?;
                let graph = unsafe { s.capture(run)? };
                let begin = Instant::now();
                for _ in 0..3 {
                    unsafe { graph.launch(&s)? };
                    s.sync()?;
                    check()?;
                }
                cases.push(serde_json::json!({"pools":np,"rows":rows,"position":position,"early":early,"all_tied":tied,
                    "exact":true,"graph_replays":3,"with_readback_seconds":begin.elapsed().as_secs_f64()}));
            }
        }
    }
    Ok(
        serde_json::json!({"cases":cases,"scope":"Selection only, supplied scores through 262144 pools (1M positions). Not a long-context model qualification."}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_score_order() {
        assert_eq!(order(-0.), order(0.));
        assert_eq!(order(f32::NAN), order(f32::from_bits(0xffc12345)));
        let x = [f32::NEG_INFINITY, -1., 0., 1., f32::INFINITY, f32::NAN];
        for a in x.windows(2) {
            assert!(order(a[0]) < order(a[1]));
        }
        for np in [512, 2048, 2064, 16384, 16448, 262144] {
            let (c, n) = config(np);
            assert!(c * n >= np && n.is_power_of_two());
        }
    }
}
