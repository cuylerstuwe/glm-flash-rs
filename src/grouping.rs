//! Stable GPU grouping for the canonical top-8 + shared router output.
//! No host readback, sort allocations, or floating-point atomics.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::load_native,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc, time::Instant};
const E: usize = 289;
const WORDS: usize = 256;
pub struct Grouping {
    clear: Kernel,
    mark: Kernel,
    prefix_kernel: Kernel,
    ids_kernel: Kernel,
    scatter: Kernel,
    small: Kernel,
    bits: Buffer,
    prefix: Buffer,
    counts: Buffer,
    place: Buffer,
    pub ids: Buffer,
    pub count: Buffer,
    pub members: Buffer,
    max_rows: usize,
}
impl Grouping {
    pub fn new(d: &Rc<Device>, native: &Path, max_rows: usize) -> Result<Self> {
        ensure!(max_rows > 0 && max_rows <= 8192, "grouping capacity");
        let path = native.join("grouping.cubin");
        Ok(Self {
            clear: load_native(d, &path, "group_clear", 256)?,
            mark: load_native(d, &path, "group_mark", 256)?,
            prefix_kernel: load_native(d, &path, "group_prefix", 256)?,
            ids_kernel: load_native(d, &path, "group_ids", 512)?,
            scatter: load_native(d, &path, "group_scatter", 256)?,
            small: load_native(d, &path, "group_small", 512)?,
            bits: d.alloc(E * WORDS * 4)?,
            prefix: d.alloc(E * WORDS * 4)?,
            counts: d.alloc(E * 4)?,
            place: d.alloc(E * 4)?,
            ids: d.alloc(E * 4)?,
            count: d.alloc(4)?,
            members: d.alloc(E * max_rows * 4)?,
            max_rows,
        })
    }
    /// `picks` must contain `rows` canonical top-8 distinct experts (0..288),
    /// followed by the shared expert 288. Only the retained router supplies it.
    pub fn run(&self, s: &Stream, picks: u64, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "grouping rows exceed scratch"
        );
        // Verifier-sized scratch only: the single block pads every member row.
        if rows <= 32 && self.max_rows <= 32 {
            return words(
                &self.small,
                s,
                [1, 1, 1],
                &[
                    picks,
                    rows as u64,
                    self.ids.ptr,
                    self.count.ptr,
                    self.members.ptr,
                    self.max_rows as u64,
                ],
            );
        }
        words(&self.clear, s, [E as u32, 1, 1], &[self.bits.ptr])?;
        words(
            &self.mark,
            s,
            [rows.div_ceil(256) as u32, 1, 1],
            &[picks, self.bits.ptr, rows as u64],
        )?;
        words(
            &self.prefix_kernel,
            s,
            [E as u32, 1, 1],
            &[self.bits.ptr, self.prefix.ptr, self.counts.ptr],
        )?;
        words(
            &self.ids_kernel,
            s,
            [1, 1, 1],
            &[
                self.counts.ptr,
                self.place.ptr,
                self.ids.ptr,
                self.count.ptr,
            ],
        )?;
        words(
            &self.scatter,
            s,
            [E as u32, 1, 1],
            &[
                picks,
                self.bits.ptr,
                self.prefix.ptr,
                self.counts.ptr,
                self.place.ptr,
                self.count.ptr,
                self.members.ptr,
                self.max_rows as u64,
            ],
        )
    }
}
fn ints(x: &[u8]) -> Vec<i32> {
    x.as_chunks::<4>()
        .0
        .iter()
        .map(|x| i32::from_le_bytes(*x))
        .collect()
}
fn bytes(x: &[i32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn probe(native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let picks = d.alloc(8192 * 9 * 4)?;
    let mut cases = Vec::new();
    let mut seed = 0x0123456789abcdefu64;
    let sizes: [(usize, &[usize]); 2] = [
        (
            8192,
            &[
                8192, 1, 2, 31, 32, 33, 63, 64, 65, 129, 1024, 4097, 8191, 8192,
            ],
        ),
        (16, &[1, 2, 3, 8, 15, 16]),
    ];
    for (capacity, row_counts) in sizes {
        let g = Grouping::new(&d, native, capacity)?;
        for &rows in row_counts {
            for concentrated in [true, false] {
                let mut p = Vec::with_capacity(rows * 9);
                let mut expected: Vec<Vec<i32>> = vec![Vec::new(); E];
                for r in 0..rows {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let start = if concentrated { 0 } else { seed as usize % 288 };
                    for slot in 0..9 {
                        let e = if slot == 8 {
                            288
                        } else {
                            (start + slot * 37) % 288
                        };
                        p.push(e as i32);
                        expected[e].push((r * 32 + slot) as i32);
                    }
                }
                picks.write(0, &bytes(&p))?;
                // Poison outputs and run twice to detect stale bitmap/count/tail state.
                g.ids.write(0, &bytes(&vec![-77; E]))?;
                g.members.write(0, &bytes(&vec![12345; E * capacity]))?;
                let run = || g.run(&s, picks.ptr, rows);
                run()?;
                s.sync()?;
                let ids: Vec<i32> = expected
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(e, _)| e as i32)
                    .collect();
                let mut want = vec![-1i32; E * capacity];
                for (u, &e) in ids.iter().enumerate() {
                    want[u * capacity..u * capacity + expected[e as usize].len()]
                        .copy_from_slice(&expected[e as usize]);
                }
                let check = || -> Result<()> {
                    ensure!(
                        ints(&g.count.read(0, 4)?)[0] as usize == ids.len(),
                        "group count"
                    );
                    ensure!(
                        ints(&g.ids.read(0, ids.len() * 4)?) == ids,
                        "stable expert order"
                    );
                    ensure!(
                        ints(&g.members.read(0, want.len() * 4)?) == want,
                        "stable member order or stale tail at {rows}"
                    );
                    Ok(())
                };
                check()?;
                let graph = unsafe { s.capture(run)? };
                for _ in 0..3 {
                    unsafe { graph.launch(&s)? };
                    s.sync()?;
                    check()?;
                }
                let start = Instant::now();
                for _ in 0..100 {
                    unsafe { graph.launch(&s)? };
                }
                s.sync()?;
                cases.push(
                    serde_json::json!({"capacity":capacity,"rows":rows,"concentrated":concentrated,
                "unique":ids.len(),"exact":true,"graph_replays_checked":3,
                "graph_mean_seconds":start.elapsed().as_secs_f64()/100.}),
                );
            }
        }
    }
    Ok(
        serde_json::json!({"cases":cases,"scope":"Exact canonical top-8 stable GPU grouping; CPU oracle and graph replay."}),
    )
}
