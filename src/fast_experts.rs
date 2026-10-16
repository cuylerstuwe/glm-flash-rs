//! Fused EXL3 prefill: retained fat-family device arithmetic, Rust scratch/launch ownership.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{load_native, load_native_shared},
    prepared::{Loaded, Prepared},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{path::Path, rc::Rc, time::Instant};
const SLOTS: usize = 9;
const EXPERTS: usize = 288;
const MAXU: usize = EXPERTS + 1;

struct Persistent {
    kernel: Kernel,
    grid: u32,
}
impl Persistent {
    fn new(d: &Rc<Device>, native: &Path, name: &str, threads: u32, shared: u32) -> Result<Self> {
        let kernel =
            load_native_shared(d, &native.join("exl3_prefill.cubin"), name, threads, shared)?;
        let grid = kernel.resident_grid().with_context(|| name.to_owned())?;
        Ok(Self { kernel, grid })
    }
}
pub struct FastExperts {
    rot: Kernel,
    rot_two: Kernel,
    plan_kernel: Kernel,
    gu_shared: Persistent,
    gu_separate: Persistent,
    down: Persistent,
    down_large: Persistent,
    xg: Buffer,
    xu: Buffer,
    xd: Buffer,
    plan: Buffer,
    pub output: Buffer,
    max_rows: usize,
}
impl FastExperts {
    pub fn new(d: &Rc<Device>, native: &Path, max_rows: usize) -> Result<Self> {
        ensure!(
            max_rows > 0 && max_rows <= 8192,
            "expert prefill scratch bound"
        );
        let cubin = native.join("exl3_prefill.cubin");
        Ok(Self {
            rot: load_native(d, &cubin, "fast_rot_in", 256)?,
            rot_two: load_native(d, &native.join("exl3.cubin"), "rot_in_kernel", 32)?,
            plan_kernel: load_native(d, &cubin, "fast_expert_plan", 1024)?,
            gu_shared: Persistent::new(d, native, "fast_gu_shared", 256, 49152)?,
            gu_separate: Persistent::new(d, native, "fast_gu_separate", 256, 73728)?,
            down: Persistent::new(d, native, "fast_down", 128, 36864)?,
            down_large: Persistent::new(d, native, "fast_down_large", 128, 33792)?,
            xg: d.alloc(max_rows * SLOTS * 4096 * 2)?,
            xu: d.alloc(max_rows * SLOTS * 4096 * 2)?,
            xd: d.alloc(max_rows * SLOTS * 512 * 2)?,
            output: d.alloc(max_rows * SLOTS * 4096 * 4)?,
            plan: d.alloc((MAXU + 3) * 4)?,
            max_rows,
        })
    }
    fn plan(
        &self,
        s: &Stream,
        ids: u64,
        count: u64,
        members: u64,
        maxm: usize,
        bm: usize,
    ) -> Result<()> {
        words(
            &self.plan_kernel,
            s,
            [1, 1, 1],
            &[
                ids,
                count,
                members,
                MAXU as u64,
                maxm as u64,
                EXPERTS as u64,
                bm as u64,
                self.plan.ptr,
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        s: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
        picks: u64,
        ids: u64,
        count: u64,
        members: u64,
        rows: usize,
        maxm: usize,
        shared_x: bool,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.max_rows && maxm >= rows && maxm <= self.max_rows,
            "expert rows/member stride outside scratch"
        );
        let w = |name: &str| m.pointer(p, &format!("weights.layers[{layer}].moe.experts.{name}"));
        let pairs = rows * SLOTS;
        if shared_x {
            words(
                &self.rot,
                s,
                [(pairs * 32).div_ceil(8) as u32, 1, 1],
                &[
                    input,
                    4096,
                    picks,
                    w("suh_g")?,
                    self.xg.ptr,
                    4096,
                    SLOTS as u64,
                    (pairs * 32) as u64,
                ],
            )?;
        } else {
            words(
                &self.rot_two,
                s,
                [pairs as u32, 32, 2],
                &[
                    input,
                    4096,
                    picks,
                    w("suh_g")?,
                    w("suh_u")?,
                    self.xg.ptr,
                    self.xu.ptr,
                    4096,
                    SLOTS as u64,
                ],
            )?;
        }
        self.plan(s, ids, count, members, maxm, 64)?;
        let gu = if shared_x {
            &self.gu_shared
        } else {
            &self.gu_separate
        };
        let ticket = self.plan.ptr + ((MAXU + 2) * 4) as u64;
        words(
            &gu.kernel,
            s,
            [gu.grid, 1, 1],
            &[
                self.xg.ptr,
                if shared_x { self.xg.ptr } else { self.xu.ptr },
                w("gt")?,
                w("ut")?,
                ids,
                members,
                self.plan.ptr,
                ticket,
                4096,
                512,
                maxm as u64,
                SLOTS as u64,
                MAXU as u64,
                w("svh_g")?,
                w("svh_u")?,
                w("suh_d")?,
                self.xd.ptr,
                0,
                10f32.to_bits() as u64,
            ],
        )?;
        let large = maxm >= 4096;
        let dn = if large { &self.down_large } else { &self.down };
        self.plan(s, ids, count, members, maxm, if large { 128 } else { 64 })?;
        words(
            &dn.kernel,
            s,
            [dn.grid, 1, 1],
            &[
                self.xd.ptr,
                self.xd.ptr,
                w("dt")?,
                w("dt")?,
                ids,
                members,
                self.plan.ptr,
                ticket,
                512,
                4096,
                maxm as u64,
                SLOTS as u64,
                MAXU as u64,
                w("svh_d")?,
                w("svh_d")?,
                w("svh_d")?,
                0,
                self.output.ptr,
                0,
            ],
        )
    }
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn hash(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}
pub fn probe(p: &Prepared, native: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let m = p.load_selected(&d, &["weights.layers[3].moe.experts"])?;
    let s = d.stream()?;
    let max_rows = 4097;
    let engine = FastExperts::new(&d, native, max_rows)?;
    let w = |name: &str| m.pointer(p, &format!("weights.layers[3].moe.experts.{name}"));
    ensure!(
        p.read_tensor("weights.layers[3].moe.experts.suh_g")?
            == p.read_tensor("weights.layers[3].moe.experts.suh_u")?,
        "shared input sign vectors differ"
    );
    let x = d.alloc(max_rows * 4096 * 2)?;
    let input: Vec<u8> = (0..max_rows * 4096)
        .flat_map(|i| {
            let f = ((i * 73 + 23) % 257) as f32 / 128. - 1.;
            ((crate::front::bf(f).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    x.write(0, &input)?;
    let picks = d.alloc(max_rows * SLOTS * 4)?;
    let ids = d.alloc(MAXU * 4)?;
    let count = d.alloc(4)?;
    let members = d.alloc(MAXU * max_rows * 4)?;
    let reference_down = d.alloc(max_rows * SLOTS * 4096 * 4)?;
    let reference_xd = d.alloc(max_rows * SLOTS * 512 * 2)?;
    let cubin = native.join("exl3_prefill.cubin");
    let oracle_gu = load_native(&d, &cubin, "fast_gu_oracle", 128)?;
    let oracle_dn = load_native(&d, &cubin, "fast_down_oracle", 128)?;
    let mut cases = Vec::new();
    for (rows, maxm) in [
        (1usize, 129usize),
        (63, 129),
        (64, 129),
        (65, 129),
        (129, 129),
        (4097, 4097),
    ] {
        eprintln!("expert parity rows={rows} member_stride={maxm}");
        let picked: Vec<i32> = (0..rows)
            .flat_map(|r| {
                (0..SLOTS).map(move |slot| {
                    if slot == 8 {
                        288
                    } else {
                        ((r * 7 + slot) % 16) as i32
                    }
                })
            })
            .collect();
        let unique: Vec<i32> = (0..EXPERTS + 1)
            .filter(|e| picked.contains(&(*e as i32)))
            .map(|e| e as i32)
            .collect();
        let mut group = vec![-1i32; MAXU * maxm];
        for (u, &e) in unique.iter().enumerate() {
            let mut n = 0;
            for (i, &picked) in picked.iter().enumerate() {
                if picked == e {
                    group[u * maxm + n] = (i / SLOTS * 32 + i % SLOTS) as i32;
                    n += 1;
                }
            }
        }
        let mut uid = vec![-1i32; MAXU];
        uid[..unique.len()].copy_from_slice(&unique);
        picks.write(0, &i32_bytes(&picked))?;
        ids.write(0, &i32_bytes(&uid))?;
        count.write(0, &(unique.len() as i32).to_le_bytes())?;
        members.write(0, &i32_bytes(&group))?;
        engine.output.zero()?;
        engine.xd.zero()?;
        engine.run(
            &s,
            p,
            &m,
            3,
            x.ptr,
            picks.ptr,
            ids.ptr,
            count.ptr,
            members.ptr,
            rows,
            maxm,
            true,
        )?;
        s.sync()?;
        let y = engine.output.read(0, rows * SLOTS * 4096 * 4)?;
        let xd = engine.xd.read(0, rows * SLOTS * 512 * 2)?;
        ensure!(
            crate::front::f32_values(&y).iter().all(|v| v.is_finite()),
            "nonfinite expert result"
        );
        reference_down.zero()?;
        reference_xd.zero()?;
        words(
            &oracle_gu,
            &s,
            [4, MAXU as u32, 1],
            &[
                engine.xg.ptr,
                engine.xg.ptr,
                w("gt")?,
                w("ut")?,
                ids.ptr,
                count.ptr,
                members.ptr,
                w("svh_g")?,
                w("svh_u")?,
                w("suh_d")?,
                reference_xd.ptr,
                4096,
                512,
                maxm as u64,
                SLOTS as u64,
                EXPERTS as u64,
                10f32.to_bits() as u64,
            ],
        )?;
        words(
            &oracle_dn,
            &s,
            [32, MAXU as u32, 1],
            &[
                reference_xd.ptr,
                w("dt")?,
                ids.ptr,
                count.ptr,
                members.ptr,
                w("svh_d")?,
                reference_down.ptr,
                512,
                4096,
                maxm as u64,
                SLOTS as u64,
                EXPERTS as u64,
            ],
        )?;
        s.sync()?;
        ensure!(
            xd == reference_xd.read(0, xd.len())?,
            "fused gate/up differs from retained oracle at {rows}"
        );
        ensure!(
            y == reference_down.read(0, y.len())?,
            "fused down differs from retained oracle at {rows}"
        );
        engine.run(
            &s,
            p,
            &m,
            3,
            x.ptr,
            picks.ptr,
            ids.ptr,
            count.ptr,
            members.ptr,
            rows,
            maxm,
            false,
        )?;
        s.sync()?;
        ensure!(
            y == engine.output.read(0, y.len())?,
            "shared/separate rotation differs at {rows}"
        );
        // A captured graph replays its plan reset and ticket scheduling each time.
        let graph = unsafe {
            s.capture(|| {
                engine.run(
                    &s,
                    p,
                    &m,
                    3,
                    x.ptr,
                    picks.ptr,
                    ids.ptr,
                    count.ptr,
                    members.ptr,
                    rows,
                    maxm,
                    true,
                )
            })?
        };
        let start = Instant::now();
        for _ in 0..3 {
            unsafe {
                graph.launch(&s)?;
            }
            s.sync()?;
            ensure!(
                y == engine.output.read(0, y.len())?,
                "graph replay/ticket drift at {rows}"
            );
        }
        cases.push(serde_json::json!({"rows":rows,"member_stride":maxm,"bytes_exact":y.len(),"gateup_oracle_exact":true,
            "down_oracle_exact":true,"shared_separate_exact":true,"graph_replays_exact":3,
            "output_sha256":hash(&y),"graph_replay_with_readback_seconds":start.elapsed().as_secs_f64()}));
    }
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"cases":cases,
        "resident_grids":{"gu_shared":engine.gu_shared.grid,"gu_separate":engine.gu_separate.grid,
            "down":engine.down.grid,"down_large":engine.down_large.grid},
        "scope":"Fused prefill expert component. Full-model integration remains a separate gate."}),
    )
}
