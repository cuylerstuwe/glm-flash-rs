//! Retained fast-prefill and decode sparse attention, with explicit scratch ownership.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{bf, fp8, load_kernel},
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};
const HEADS: usize = 16;
const LATENT: usize = 512;
const WIDTH: usize = 2051;
pub struct SparseAttention {
    one: Kernel,
    chunks: Kernel,
    merge: Kernel,
    po: Buffer,
    pm: Buffer,
    pl: Buffer,
    max_rows: usize,
}
impl SparseAttention {
    pub fn new(d: &Rc<Device>, root: &Path, max_rows: usize) -> Result<Self> {
        ensure!((1..=1024).contains(&max_rows), "sparse attention capacity");
        Ok(Self {
            one: load_kernel(d, root, "sparse_attention_one")?,
            chunks: load_kernel(d, root, "sparse_attention_chunks")?,
            merge: load_kernel(d, root, "sparse_attention_merge")?,
            po: d.alloc(5 * max_rows * HEADS * LATENT * 4)?,
            pm: d.alloc(5 * max_rows * HEADS * 4)?,
            pl: d.alloc(5 * max_rows * HEADS * 4)?,
            max_rows,
        })
    }
    /// All input storage and page mappings remain owned by the caller until completion.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        s: &Stream,
        qa: u64,
        cache: u64,
        pages: u64,
        tokens: u64,
        counts: u64,
        out: u64,
        rows: usize,
        fast: bool,
    ) -> Result<()> {
        ensure!(rows > 0 && rows <= self.max_rows, "sparse attention rows");
        if fast {
            words(
                &self.one,
                s,
                [rows as u32, 1, 1],
                &[qa, cache, cache, tokens, counts, out, pages],
            )
        } else {
            words(
                &self.chunks,
                s,
                [rows as u32, 1, 5],
                &[
                    qa,
                    cache,
                    cache,
                    tokens,
                    counts,
                    self.po.ptr,
                    self.pm.ptr,
                    self.pl.ptr,
                    rows as u64,
                    pages,
                ],
            )?;
            words(
                &self.merge,
                s,
                [rows as u32, HEADS as u32, 1],
                &[
                    self.po.ptr,
                    self.pm.ptr,
                    self.pl.ptr,
                    out,
                    counts,
                    rows as u64,
                ],
            )
        }
    }
}
fn bfbytes(x: f32) -> [u8; 2] {
    ((bf(x).to_bits() >> 16) as u16).to_le_bytes()
}
pub fn probe(root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let counts = [
        0i32, 1, 31, 32, 33, 511, 512, 513, 1024, 2047, 2048, 2049, 2050, 2051, 1, 2051, 2051,
    ];
    let rows = counts.len();
    let capacity = 1048576;
    let page_count = capacity / 256;
    let attention = SparseAttention::new(&d, root, rows)?;
    let cache = d.alloc(capacity * 528)?;
    let fragmented = d.alloc(capacity * 528)?;
    let pages = d.alloc(page_count * 4)?;
    let reversed = d.alloc(page_count * 4)?;
    pages.write(
        0,
        &(0..page_count)
            .flat_map(|i| (i as i32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    reversed.write(
        0,
        &(0..page_count)
            .rev()
            .flat_map(|i| (i as i32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let qa = d.alloc(rows * HEADS * LATENT * 2)?;
    let tokens = d.alloc(rows * WIDTH * 4)?;
    let cnt = d.alloc(rows * 4)?;
    let out = d.alloc(rows * HEADS * LATENT * 4)?;
    let other = d.alloc(out.bytes)?;
    let single_q = d.alloc(HEADS * LATENT * 2)?;
    let single_t = d.alloc(WIDTH * 4)?;
    let single_c = d.alloc(4)?;
    let single_o = d.alloc(HEADS * LATENT * 4)?;
    let token_ids: Vec<i32> = (0..rows)
        .flat_map(|r| {
            (0..WIDTH).map(move |i| {
                if i < counts[r] as usize {
                    ((capacity - 1 - i * 509 - r * 257) % capacity) as i32
                } else {
                    -1
                }
            })
        })
        .collect();
    tokens.write(
        0,
        &token_ids
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    cnt.write(
        0,
        &counts
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let initial: Vec<u8> = (0..rows * HEADS * LATENT)
        .flat_map(|_| 42f32.to_le_bytes())
        .collect();
    let query: Vec<u8> = (0..rows * HEADS * LATENT)
        .flat_map(|i| bfbytes(((i * 17 + 11) % 127) as f32 / 1024. - 0.0625))
        .collect();
    let codes = [0x30u8, 0x38, 0x3c, 0xb0, 0xb8, 0xbc];
    let mut cases = Vec::new();
    for kind in ["constant", "uniform-query", "nontrivial"] {
        let mut bytes = vec![0u8; capacity * 528];
        for t in 0..capacity {
            let row = &mut bytes[t * 528..(t + 1) * 528];
            for (i, b) in row[..512].iter_mut().enumerate() {
                *b = if kind == "constant" {
                    0x38
                } else {
                    codes[(t * 7 + i * 11) % codes.len()]
                };
            }
            row[512..516].copy_from_slice(&1f32.to_le_bytes());
        }
        cache.write(0, &bytes)?;
        // Reverse physical pages while preserving each logical page's contents.
        let page_bytes = 256 * 528;
        for a in 0..page_count / 2 {
            let b = page_count - 1 - a;
            let (left, right) = bytes.split_at_mut(b * page_bytes);
            left[a * page_bytes..(a + 1) * page_bytes].swap_with_slice(&mut right[..page_bytes]);
        }
        fragmented.write(0, &bytes)?;
        drop(bytes);
        if kind == "uniform-query" {
            qa.zero()?;
        } else {
            qa.write(0, &query)?;
        }
        // Setup operations are completed before nonblocking work begins.
        d.sync()?;
        for fast in [false, true] {
            out.write(0, &initial)?;
            other.write(0, &initial)?;
            attention.run(
                &s, qa.ptr, cache.ptr, pages.ptr, tokens.ptr, cnt.ptr, out.ptr, rows, fast,
            )?;
            attention.run(
                &s,
                qa.ptr,
                fragmented.ptr,
                reversed.ptr,
                tokens.ptr,
                cnt.ptr,
                other.ptr,
                rows,
                fast,
            )?;
            let expected = out.read(0, out.bytes)?;
            ensure!(
                other.read(0, other.bytes)? == expected,
                "physical page mapping mismatch {kind} fast={fast}"
            );
            let mut maximum_error = 0f32;
            for r in 0..rows {
                for j in 0..HEADS * LATENT {
                    let off = (r * HEADS * LATENT + j) * 4;
                    let value = f32::from_le_bytes(expected[off..off + 4].try_into().unwrap());
                    ensure!(value.is_finite(), "nonfinite sparse attention");
                    let reference = if counts[r] == 0 {
                        Some(42.)
                    } else if kind == "constant" {
                        Some(1.)
                    } else if kind == "uniform-query" {
                        Some(
                            (0..counts[r] as usize)
                                .map(|i| {
                                    fp8(codes[(token_ids[r * WIDTH + i] as usize * 7
                                        + (j % LATENT) * 11)
                                        % codes.len()])
                                })
                                .sum::<f32>()
                                / counts[r] as f32,
                        )
                    } else {
                        None
                    };
                    if let Some(reference) = reference {
                        let error = (value - reference).abs();
                        maximum_error = maximum_error.max(error);
                        ensure!(
                            error <= 1e-6,
                            "scalar attention {kind} fast={fast} r={r} j={j}: {value} != {reference}"
                        );
                    }
                }
                unsafe {
                    single_q.copy_from(&s, 0, &qa, r * HEADS * LATENT * 2, single_q.bytes)?;
                    single_t.copy_from(&s, 0, &tokens, r * WIDTH * 4, single_t.bytes)?;
                    single_c.copy_from(&s, 0, &cnt, r * 4, 4)?;
                }
                // Preserve the untouched dense row's sentinel via a stream copy.
                unsafe {
                    single_o.copy_from(&s, 0, &out, r * HEADS * LATENT * 4, single_o.bytes)?;
                }
                attention.run(
                    &s,
                    single_q.ptr,
                    cache.ptr,
                    pages.ptr,
                    single_t.ptr,
                    single_c.ptr,
                    single_o.ptr,
                    1,
                    fast,
                )?;
                ensure!(
                    single_o.read(0, single_o.bytes)?
                        == expected[r * single_o.bytes..(r + 1) * single_o.bytes],
                    "row independence {kind} fast={fast} row={r}"
                );
            }
            let graph = unsafe {
                s.capture(|| {
                    attention.run(
                        &s,
                        qa.ptr,
                        fragmented.ptr,
                        reversed.ptr,
                        tokens.ptr,
                        cnt.ptr,
                        other.ptr,
                        rows,
                        fast,
                    )
                })?
            };
            for _ in 0..3 {
                unsafe {
                    graph.launch(&s)?;
                }
                ensure!(
                    other.read(0, other.bytes)? == expected,
                    "attention graph replay"
                );
            }
            use sha2::{Digest, Sha256};
            cases.push(serde_json::json!({"kind":kind,"fast_prefill":fast,"page_permutation_exact":true,"row_independence_exact":true,"graph_replays":3,"scalar_max_abs_error":maximum_error,"output_sha256":format!("{:x}",Sha256::digest(&expected))}));
        }
    }
    Ok(
        serde_json::json!({"capacity":capacity,"rows":rows,"counts":counts,"cases":cases,"scope":"Sparse attention components only; prefill and decode use distinct arithmetic schedules."}),
    )
}
