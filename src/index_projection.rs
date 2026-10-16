//! Index key/head-weight, query and compression-gate projections on canonical BF16 weights.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{bf, bf_values, load_kernel},
    prepared::{Loaded, Prepared},
};
use anyhow::{Result, ensure};
use std::{collections::BTreeMap, path::Path, rc::Rc};
/// The first sparse query is at zero-based position 2051. A crossing window
/// keeps all query rows so its original row offsets and arithmetic are unchanged.
fn needs_query(position: usize, rows: usize) -> bool {
    position >= 2051 || rows > 2051 - position
}
pub struct Projection {
    k: BTreeMap<&'static str, Kernel>,
    part: Buffer,
    gate_part: Buffer,
    pub raw: Buffer,
    pub query: Buffer,
    pub gate: Buffer,
    max_rows: usize,
    skip_dense_query: bool,
}
impl Projection {
    pub fn new(d: &Rc<Device>, root: &Path, max_rows: usize) -> Result<Self> {
        ensure!((1..=1024).contains(&max_rows), "index projection capacity");
        let mut k = BTreeMap::new();
        for name in [
            "index_kw_decode",
            "index_qb_decode",
            "index_gate_decode",
            "index_gate_prefill",
            "reduce8_bf16",
            "reduce4_bf16",
            "router_sum",
            "prefill_b16_160_4096_bf16",
            "prefill_b16_4096_1536_bf16",
        ] {
            k.insert(name, load_kernel(d, root, name)?);
        }
        Ok(Self {
            k,
            part: d.alloc(max_rows * 4 * 4096 * 4)?,
            gate_part: d.alloc(max_rows * 8 * 128 * 4)?,
            raw: d.alloc(max_rows * 160 * 2)?,
            query: d.alloc(max_rows * 4096 * 2)?,
            gate: d.alloc(max_rows * 128 * 4)?,
            max_rows,
            skip_dense_query: match std::env::var("ALASKA_SKIP_DENSE_INDEX_QUERY").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("0") => false,
                Ok("1") => true,
                _ => anyhow::bail!("ALASKA_SKIP_DENSE_INDEX_QUERY must be 0/1"),
            },
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        s: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
        qr: u64,
        rows: usize,
        position: usize,
        fast: bool,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.max_rows && (fast || rows <= 16),
            "index projection rows"
        );
        let w = |n: &str| m.pointer(p, &format!("weights.layers[{layer}].dsa.index.{n}"));
        for (name, alias, n, cols, sk, x, out) in [
            (
                "kw.weight",
                "index_kw_decode",
                160usize,
                4096usize,
                8usize,
                input,
                self.raw.ptr,
            ),
            (
                "qb.weight",
                "index_qb_decode",
                4096,
                1536,
                4,
                qr,
                self.query.ptr,
            ),
        ] {
            // Keys and compression gates must still advance through dense
            // attention; later sparse queries consume those historical pools.
            // Query is scratch, and dense attention never reads it.
            if name == "qb.weight" && self.skip_dense_query && !needs_query(position, rows) {
                continue;
            }
            p.expect(
                &format!("weights.layers[{layer}].dsa.index.{name}"),
                "bfloat16",
                &[n as u64, cols as u64],
            )?;
            if fast {
                let alias = if n == 160 {
                    "prefill_b16_160_4096_bf16"
                } else {
                    "prefill_b16_4096_1536_bf16"
                };
                words(
                    &self.k[alias],
                    s,
                    [
                        (rows.div_ceil(if n == 160 { 64 } else { 128 }) * n.div_ceil(128)) as u32,
                        1,
                        1,
                    ],
                    &[x, w(name)?, out, rows as u64, cols as u64],
                )?;
            } else {
                words(
                    &self.k[alias],
                    s,
                    [rows.div_ceil(16) as u32, n.div_ceil(64) as u32, sk as u32],
                    &[x, w(name)?, out, self.part.ptr, rows as u64, cols as u64],
                )?;
                let reduce = if sk == 8 {
                    "reduce8_bf16"
                } else {
                    "reduce4_bf16"
                };
                words(
                    &self.k[reduce],
                    s,
                    [(rows * n).div_ceil(1024) as u32, 1, 1],
                    &[self.part.ptr, out, (rows * n) as u64],
                )?;
            }
        }
        p.expect(
            &format!("weights.layers[{layer}].dsa.index.gate"),
            "bfloat16",
            &[128, 4096],
        )?;
        let (alias, bm) = if fast {
            ("index_gate_prefill", 128)
        } else {
            ("index_gate_decode", 16)
        };
        words(
            &self.k[alias],
            s,
            [rows.div_ceil(bm) as u32, 4, 8],
            &[input, w("gate")?, self.gate_part.ptr, rows as u64, 4096],
        )?;
        words(
            &self.k["router_sum"],
            s,
            [(rows * 128).div_ceil(1024) as u32, 1, 1],
            &[self.gate_part.ptr, self.gate.ptr, (rows * 128) as u64],
        )
    }
}
fn bfbytes(x: f32) -> [u8; 2] {
    ((bf(x).to_bits() >> 16) as u16).to_le_bytes()
}
pub fn probe(p: &Prepared, root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let m = p.load_selected(&d, &["weights.layers[3].dsa.index"])?;
    let mut projection = Projection::new(&d, root, 129)?;
    let x = d.alloc(129 * 4096 * 2)?;
    let q = d.alloc(129 * 1536 * 2)?;
    x.zero()?;
    q.zero()?;
    x.write(37 * 2, &bfbytes(1.))?;
    q.write(67 * 2, &bfbytes(1.))?;
    d.sync()?;
    let kw = p.read_tensor("weights.layers[3].dsa.index.kw.weight")?;
    let qb = p.read_tensor("weights.layers[3].dsa.index.qb.weight")?;
    let gate = bf_values(&p.read_tensor("weights.layers[3].dsa.index.gate")?);
    for fast in [false, true] {
        projection.run(&s, p, &m, 3, x.ptr, q.ptr, 1, 2051, fast)?;
        let raw = projection.raw.read(0, 160 * 2)?;
        let query = projection.query.read(0, 4096 * 2)?;
        let actual_gate = projection.gate.read(0, 128 * 4)?;
        for r in 0..160 {
            ensure!(
                raw[r * 2..r * 2 + 2] == kw[(r * 4096 + 37) * 2..(r * 4096 + 37) * 2 + 2],
                "index key unit reference"
            );
        }
        for r in 0..4096 {
            ensure!(
                query[r * 2..r * 2 + 2] == qb[(r * 1536 + 67) * 2..(r * 1536 + 67) * 2 + 2],
                "index query unit reference"
            );
        }
        for r in 0..128 {
            ensure!(
                actual_gate[r * 4..r * 4 + 4] == gate[r * 4096 + 37].to_le_bytes(),
                "index gate unit reference"
            );
        }
    }
    let xb: Vec<u8> = (0..129 * 4096)
        .flat_map(|i| bfbytes(((i * 17 + 3) % 263) as f32 / 512. - 0.25))
        .collect();
    let qb: Vec<u8> = (0..129 * 1536)
        .flat_map(|i| bfbytes(((i * 59 + 7) % 269) as f32 / 512. - 0.25))
        .collect();
    x.write(0, &xb)?;
    q.write(0, &qb)?;
    d.sync()?;
    let mut cases = Vec::new();
    for (fast, rows) in [
        (false, 1),
        (false, 15),
        (false, 16),
        (true, 1),
        (true, 17),
        (true, 129),
    ] {
        projection.run(&s, p, &m, 3, x.ptr, q.ptr, rows, 2051, fast)?;
        let raw = projection.raw.read(0, rows * 160 * 2)?;
        let query = projection.query.read(0, rows * 4096 * 2)?;
        let gate = projection.gate.read(0, rows * 128 * 4)?;
        for r in 0..rows {
            projection.run(
                &s,
                p,
                &m,
                3,
                x.ptr + (r * 4096 * 2) as u64,
                q.ptr + (r * 1536 * 2) as u64,
                1,
                2051,
                fast,
            )?;
            for (buffer, expected, width) in [
                (&projection.raw, &raw, 160 * 2),
                (&projection.query, &query, 4096 * 2),
                (&projection.gate, &gate, 128 * 4),
            ] {
                ensure!(
                    buffer.read(0, width)? == expected[r * width..(r + 1) * width],
                    "index projection row independence fast={fast} rows={rows} row={r} bytes_per_row={width}"
                );
            }
        }
        let graph =
            unsafe { s.capture(|| projection.run(&s, p, &m, 3, x.ptr, q.ptr, rows, 2051, fast))? };
        for _ in 0..3 {
            unsafe {
                graph.launch(&s)?;
            }
            ensure!(
                projection.raw.read(0, raw.len())? == raw
                    && projection.query.read(0, query.len())? == query
                    && projection.gate.read(0, gate.len())? == gate,
                "projection graph"
            );
        }
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&raw);
        h.update(&query);
        h.update(&gate);
        cases.push(serde_json::json!({"rows":rows,"fast_prefill":fast,"rows_independent_exact":true,"graph_replays":3,"output_sha256":format!("{:x}",h.finalize())}));
    }
    let mut dense_cases = Vec::new();
    for fast in [false, true] {
        for (position, rows) in [
            (2050, 1),
            (2051, 1),
            (2035, 16),
            (2036, 16),
            (2050, 16),
            (2051, 16),
            (1923, 128),
            (1923, 129),
        ] {
            if !fast && rows > 16 {
                continue;
            }
            projection.skip_dense_query = false;
            projection.run(&s, p, &m, 3, x.ptr, q.ptr, rows, position, fast)?;
            let raw = projection.raw.read(0, rows * 160 * 2)?;
            let query = projection.query.read(0, rows * 4096 * 2)?;
            let gate = projection.gate.read(0, rows * 128 * 4)?;
            let raw_poison = vec![0xcd; raw.len()];
            let query_poison = vec![0xff; query.len()];
            let gate_poison = vec![0xcd; gate.len()];
            ensure!(
                raw != raw_poison && query != query_poison && gate != gate_poison,
                "index projection poison must differ from reference"
            );
            for enabled in [false, true] {
                projection.skip_dense_query = enabled;
                let skipped = enabled && !needs_query(position, rows);
                let graph = unsafe {
                    s.capture(|| projection.run(&s, p, &m, 3, x.ptr, q.ptr, rows, position, fast))?
                };
                for replay in 0..4 {
                    // Poison every output before eager execution and each replay:
                    // stale keys/gates cannot pass, and a skipped query must not
                    // be written even when its previous value is unusable.
                    projection.raw.write(0, &raw_poison)?;
                    projection.query.write(0, &query_poison)?;
                    projection.gate.write(0, &gate_poison)?;
                    if replay == 0 {
                        projection.run(&s, p, &m, 3, x.ptr, q.ptr, rows, position, fast)?;
                    } else {
                        unsafe { graph.launch(&s)? };
                    }
                    ensure!(
                        projection.raw.read(0, raw.len())? == raw
                            && projection.gate.read(0, gate.len())? == gate
                            && projection.query.read(0, query.len())?
                                == if skipped { &query_poison } else { &query }[..],
                        "dense query gate mismatch position={position} rows={rows} fast={fast} enabled={enabled} replay={replay}"
                    );
                }
                dense_cases.push(serde_json::json!({
                    "position":position,"rows":rows,"fast_prefill":fast,
                    "optimization_enabled":enabled,"query_skipped":skipped,
                    "keys_gates_exact":true,"query_or_poison_exact":true,"graph_replays":3,
                }));
            }
        }
    }
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"unit_vectors_stored_weight_exact":true,"cases":cases,"dense_query_cases":dense_cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::needs_query;

    #[test]
    fn dense_query_boundary_includes_every_row_in_the_window() {
        for (position, rows, expected) in [
            (0, 1, false),
            (0, 2051, false),
            (0, 2052, true),
            (2050, 1, false),
            (2051, 1, true),
            (2035, 16, false),
            (2036, 16, true),
            (2050, 16, true),
            (2051, 16, true),
            (usize::MAX, 16, true),
        ] {
            assert_eq!(needs_query(position, rows), expected);
        }
    }
}
