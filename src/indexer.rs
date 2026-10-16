//! Bounded index rings and paged pool keys; retained index arithmetic.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{bf_values, load_kernel},
    prepared::{Loaded, Prepared},
};
use anyhow::{Result, ensure};
use std::{collections::BTreeMap, path::Path, rc::Rc};
const DIM: usize = 128;
pub struct IndexCache {
    write: Kernel,
    pool: Kernel,
    scores: BTreeMap<usize, Kernel>,
    pub keys: Buffer,
    pub gates: Buffer,
    pub pools: Rc<Buffer>,
    pub pages: Rc<Buffer>,
    ring_rows: usize,
    capacity: usize,
    page_ids: Vec<i32>,
}
impl IndexCache {
    pub fn new(
        d: &Rc<Device>,
        root: &Path,
        capacity: usize,
        chunk: usize,
        fragmented: bool,
    ) -> Result<Self> {
        ensure!(
            (256..=1048576).contains(&capacity)
                && capacity.is_multiple_of(256)
                && chunk > 0
                && chunk <= 8192,
            "index capacity"
        );
        let ring_rows = (chunk + 64).next_power_of_two().max(256);
        let alloc = |n| -> Result<Buffer> {
            let b = d.alloc(n)?;
            b.zero()?;
            Ok(b)
        };
        let mut page_ids: Vec<i32> = (0..capacity / 256).map(|i| i as i32).collect();
        if fragmented {
            page_ids.reverse();
        }
        let pages = alloc(page_ids.len() * 4)?;
        pages.write(
            0,
            &page_ids
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut scores = BTreeMap::new();
        for g in [16, 32, 64, 128] {
            scores.insert(g, load_kernel(d, root, &format!("index_scores_{g}"))?);
        }
        Ok(Self {
            write: load_kernel(d, root, "index_write")?,
            pool: load_kernel(d, root, "index_pool")?,
            scores,
            keys: alloc(ring_rows * DIM * 2)?,
            gates: alloc(ring_rows * DIM * 2)?,
            pools: Rc::new(alloc(capacity / 4 * DIM * 2)?),
            pages: Rc::new(pages),
            ring_rows,
            capacity,
            page_ids,
        })
    }
    pub fn shared(
        d: &Rc<Device>,
        root: &Path,
        capacity: usize,
        chunk: usize,
        pools: Rc<Buffer>,
        pages: Rc<Buffer>,
    ) -> Result<Self> {
        ensure!(
            capacity == 1048576
                && chunk > 0
                && chunk <= 8192
                && pools.bytes >= capacity / 4 * DIM * 2
                && pages.bytes >= capacity / 256 * 4,
            "shared index storage"
        );
        let ring_rows = (chunk + 64).next_power_of_two().max(256);
        let mut scores = BTreeMap::new();
        for g in [16, 32, 64, 128] {
            scores.insert(g, load_kernel(d, root, &format!("index_scores_{g}"))?);
        }
        Ok(Self {
            write: load_kernel(d, root, "index_write")?,
            pool: load_kernel(d, root, "index_pool")?,
            scores,
            keys: d.alloc(ring_rows * DIM * 2)?,
            gates: d.alloc(ring_rows * DIM * 2)?,
            pools,
            pages,
            ring_rows,
            capacity,
            page_ids: Vec::new(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        s: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        raw: u64,
        gate: u64,
        rows: usize,
        position: usize,
        pos: &Buffer,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows + 3 <= self.ring_rows && position + rows <= self.capacity,
            "index write bounds"
        );
        let w = |n: &str| m.pointer(p, &format!("weights.layers[{layer}].dsa.index.{n}"));
        ensure!(pos.bytes >= 4, "index position storage");
        words(
            &self.write,
            s,
            [rows as u32, 1, 1],
            &[
                raw,
                160,
                gate,
                w("ln_w")?,
                w("ln_b")?,
                self.keys.ptr,
                self.gates.ptr,
                pos.ptr,
                self.ring_rows as u64,
                1e-6f32.to_bits() as u64,
            ],
        )?;
        words(
            &self.pool,
            s,
            [(rows / 4 + 2) as u32, 1, 1],
            &[
                self.keys.ptr,
                self.gates.ptr,
                w("ape")?,
                self.pools.ptr,
                pos.ptr,
                rows as u64,
                self.ring_rows as u64,
                self.pages.ptr,
            ],
        )
    }
    /// Save only the ring rows a bounded verifier can overwrite. Pool/latent
    /// suffixes stay unpublished behind the accepted token cursor.
    pub fn ring_rows(&self) -> usize {
        self.ring_rows
    }
    pub const ROW_BYTES: usize = DIM * 2;
    /// Position is supplied by the stream owner; pool keys through that position
    /// must already exist. Launches visit only visible pools, without CPU copies.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        s: &Stream,
        pos: &Buffer,
        query: u64,
        weights: u64,
        stride: usize,
        out: &Buffer,
        rows: usize,
        np: usize,
        groups: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && (512..=self.capacity / 4).contains(&np)
                && np.is_multiple_of(16)
                && stride >= 32
                && stride.is_multiple_of(16),
            "score shape"
        );
        ensure!(out.bytes >= rows * np * 4, "score scratch");
        let k = self
            .scores
            .get(&groups)
            .ok_or_else(|| anyhow::anyhow!("score launch groups"))?;
        words(
            k,
            s,
            [rows as u32, groups as u32, 1],
            &[
                query,
                weights,
                stride as u64,
                self.pools.ptr,
                out.ptr,
                pos.ptr,
                rows as u64,
                np as u64,
                (128f32.sqrt().recip()).to_bits() as u64,
                self.pages.ptr,
            ],
        )
    }
    fn logical_pools(&self, n: usize) -> Result<Vec<u8>> {
        let raw = self.pools.read(0, self.pools.bytes)?;
        let mut out = Vec::with_capacity(n * DIM * 2);
        for i in 0..n {
            let physical = self.page_ids[i / 64] as usize * 64 + i % 64;
            out.extend_from_slice(&raw[physical * DIM * 2..(physical + 1) * DIM * 2]);
        }
        Ok(out)
    }
}
fn bfbytes(x: f32) -> [u8; 2] {
    ((crate::front::bf(x).to_bits() >> 16) as u16).to_le_bytes()
}
pub fn probe(p: &Prepared, root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let m = p.load_selected(&d, &["weights.layers[3].dsa.index"])?;
    let whole = IndexCache::new(&d, root, 8192, 5001, false)?;
    let ring = IndexCache::new(&d, root, 8192, 256, true)?;
    let pos = d.alloc(4)?;
    pos.zero()?;
    let n = 5001;
    let raw = d.alloc(n * 160 * 2)?;
    let gates = d.alloc(n * 128 * 4)?;
    // A constant key makes LayerNorm exactly the stored bias. Gates cancel the
    // real positional bias, so each pool uses exactly one-quarter weights.
    let ape = bf_values(&p.read_tensor("weights.layers[3].dsa.index.ape")?);
    raw.write(
        0,
        &(0..n * 160).flat_map(|_| bfbytes(2.)).collect::<Vec<_>>(),
    )?;
    gates.write(
        0,
        &(0..n * 128)
            .flat_map(|i| (-ape[(i / 128 % 4) * 128 + i % 128]).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    whole.update(&s, p, &m, 3, raw.ptr, gates.ptr, 4, 0, &pos)?;
    s.sync()?;
    let bias = p.read_tensor("weights.layers[3].dsa.index.ln_b")?;
    ensure!(
        whole.keys.read(0, 128 * 2)? == bias,
        "constant LayerNorm bias"
    );
    ensure!(whole.logical_pools(1)? == bias, "uniform pool reference");
    // Nontrivial projected keys/gates, complete pools spanning many ring wraps.
    let key_bytes: Vec<u8> = (0..n * 160)
        .flat_map(|i| bfbytes(((i * 73 + 5) % 257) as f32 / 64. - 2.))
        .collect();
    let gate_bytes: Vec<u8> = (0..n * 128)
        .flat_map(|i| (((i * 107 + 31) % 509) as f32 / 128. - 2.).to_le_bytes())
        .collect();
    raw.write(0, &key_bytes)?;
    gates.write(0, &gate_bytes)?;
    whole.update(&s, p, &m, 3, raw.ptr, gates.ptr, n, 0, &pos)?;
    s.sync()?;
    let expected = whole.logical_pools(n / 4)?;
    let mut offset = 0;
    let sizes = [63, 1, 65, 127, 256, 3, 4, 129];
    let mut calls = 0;
    while offset < n {
        let r = sizes[calls % sizes.len()].min(n - offset);
        unsafe {
            pos.set_i32(&s, offset as i32)?;
        }
        ring.update(
            &s,
            p,
            &m,
            3,
            raw.ptr + (offset * 160 * 2) as u64,
            gates.ptr + (offset * 128 * 4) as u64,
            r,
            offset,
            &pos,
        )?;
        offset += r;
        calls += 1;
    }
    s.sync()?;
    ensure!(
        ring.logical_pools(n / 4)? == expected,
        "ring/chunk/page pool mismatch"
    );
    let kr = ring.keys.read(0, ring.keys.bytes)?;
    let gr = ring.gates.read(0, ring.gates.bytes)?;
    let kw = whole.keys.read(0, whole.keys.bytes)?;
    let gw = whole.gates.read(0, whole.gates.bytes)?;
    for row in n - ring.ring_rows..n {
        let a = (row % ring.ring_rows) * 128 * 2;
        let b = row * 128 * 2;
        ensure!(
            kr[a..a + 256] == kw[b..b + 256] && gr[a..a + 256] == gw[b..b + 256],
            "ring tail mismatch"
        );
    }
    // Compare the retained score schedules on the same physical pool table.
    let rows = 4;
    let np = 2048;
    let query = d.alloc(rows * 4096 * 2)?;
    let weights = d.alloc(rows * 160 * 2)?;
    let score = d.alloc(rows * np * 4)?;
    let qbytes: Vec<u8> = (0..rows * 4096)
        .flat_map(|i| bfbytes(((i * 59 + 23) % 251) as f32 / 128. - 1.))
        .collect();
    query.write(0, &qbytes)?;
    weights.write(
        0,
        &(0..rows * 160)
            .flat_map(|i| bfbytes(((i * 7 + 3) % 83) as f32 / 64. - 0.5))
            .collect::<Vec<_>>(),
    )?;
    unsafe {
        pos.set_i32(&s, (n - rows) as i32)?;
    }
    ring.score(&s, &pos, query.ptr, weights.ptr, 160, &score, rows, np, 16)?;
    s.sync()?;
    let scores = score.read(0, rows * np * 4)?;
    let mut schedules = Vec::new();
    for groups in [32, 64, 128] {
        ring.score(
            &s,
            &pos,
            query.ptr,
            weights.ptr,
            160,
            &score,
            rows,
            np,
            groups,
        )?;
        s.sync()?;
        // Only visible scores are written; compare each row through its bound.
        let actual = score.read(0, rows * np * 4)?;
        for r in 0..rows {
            let vis = (n - rows + r + 1) / 4;
            ensure!(
                actual[r * np * 4..(r * np + vis) * 4] == scores[r * np * 4..(r * np + vis) * 4],
                "score schedule mismatch"
            );
        }
        schedules.push(groups);
    }
    // Keep zeroing, scalar positions, cache writes, scoring and selection on
    // the same stream. A default-stream memset would race a nonblocking stream.
    let select = crate::sparse_select::Select::new(&d, root, rows, np)?;
    let run = || -> Result<()> {
        unsafe {
            pos.set_i32(&s, (n - 5) as i32)?;
        }
        ring.update(
            &s,
            p,
            &m,
            3,
            raw.ptr + ((n - 5) * 160 * 2) as u64,
            gates.ptr + ((n - 5) * 128 * 4) as u64,
            5,
            n - 5,
            &pos,
        )?;
        unsafe {
            pos.set_i32(&s, (n - rows) as i32)?;
            query.zero_prefix(&s, query.bytes)?;
        }
        ring.score(&s, &pos, query.ptr, weights.ptr, 160, &score, rows, np, 128)?;
        select.run(&s, &score, &pos, rows, np)
    };
    let graph = unsafe { s.capture(run)? };
    for _ in 0..3 {
        query.write(0, &qbytes)?;
        unsafe {
            graph.launch(&s)?;
        }
        s.sync()?;
        let zeros = score.read(0, rows * np * 4)?;
        for r in 0..rows {
            let vis = (n - rows + r + 1) / 4;
            for i in 0..vis {
                let j = (r * np + i) * 4;
                let v = f32::from_le_bytes(zeros[j..j + 4].try_into().unwrap());
                ensure!(v == 0., "zero-query score row={r} pool={i} value={v}");
            }
        }
        let ids = select.pools.read(0, rows * 512 * 8)?;
        for (i, v) in ids.as_chunks::<8>().0.iter().enumerate() {
            ensure!(
                i64::from_le_bytes(*v) == (i % 512) as i64,
                "zero-query tie order offset={i} got={}",
                i64::from_le_bytes(*v)
            );
        }
        ensure!(
            ring.logical_pools(n / 4)? == expected,
            "graph pool mismatch"
        );
        ensure!(
            ring.keys.read(0, ring.keys.bytes)? == kr
                && ring.gates.read(0, ring.gates.bytes)? == gr,
            "graph ring mismatch"
        );
    }
    use sha2::{Digest, Sha256};
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"rows":n,"ring_rows":ring.ring_rows,"chunks":calls,
        "uniform_pool_scalar_exact":true,"ring_keys_gates_exact":true,"pool_partitions_and_fragmented_pages_exact":true,
        "pool_sha256":format!("{:x}",Sha256::digest(&expected)),"score_schedules_exact":schedules,
        "zero_query_select_earliest_pools_exact":true,"stream_ordered_graph_replays":3,"scope":"Indexer cache/pooling/scoring components; projection and full sparse model integration require separate model tests."}),
    )
}
