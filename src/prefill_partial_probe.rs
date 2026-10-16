//! Isolated exact-byte qualification for direct BF16 rank partials. The oracle
//! is the retained FP32 kernel followed by the existing native RN-BF16 cast.
//! This does not run a model forward, collective or normal serving request.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{bf, load_kernel, load_native},
    prefill::Gemms,
    prepared::Prepared,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, rc::Rc};

const WIDTH: usize = 4096;
const SLOTS: usize = 9;
const GUARD: usize = 256;
const MAX_ROWS: usize = 1024;
const ROWS: [usize; 10] = [1, 63, 64, 65, 127, 128, 129, 257, 512, 1024];
const MATRICES: [(&str, usize); 3] = [
    ("weights.layers[0].kda.o", 2048),
    ("weights.layers[3].dsa.o", 4096),
    ("weights.layers[0].mlp.down", 3072),
];

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn range(rows: usize, element_bytes: usize) -> Result<std::ops::Range<usize>> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows) && matches!(element_bytes, 2 | 4),
        "partial probe output geometry"
    );
    Ok(GUARD..GUARD + rows * WIDTH * element_bytes)
}
fn guarded_payload(bytes: &[u8], rows: usize, element_bytes: usize, poison: u8) -> Result<&[u8]> {
    let live = range(rows, element_bytes)?;
    ensure!(
        bytes.len() == 2 * GUARD + MAX_ROWS * WIDTH * element_bytes,
        "partial probe guard allocation"
    );
    ensure!(
        bytes[..live.start].iter().all(|b| *b == poison)
            && bytes[live.end..].iter().all(|b| *b == poison),
        "partial probe prefix/inactive rows/suffix overwritten"
    );
    Ok(&bytes[live])
}
struct Guarded {
    buffer: Buffer,
    element_bytes: usize,
    poison: u8,
}
impl Guarded {
    fn new(d: &Rc<Device>, element_bytes: usize, poison: u8) -> Result<Self> {
        range(MAX_ROWS, element_bytes)?;
        Ok(Self {
            buffer: d.alloc(2 * GUARD + MAX_ROWS * WIDTH * element_bytes)?,
            element_bytes,
            poison,
        })
    }
    fn pointer(&self) -> u64 {
        self.buffer.ptr + GUARD as u64
    }
    fn reset(&self) -> Result<()> {
        self.buffer.write(0, &vec![self.poison; self.buffer.bytes])
    }
    fn read(&self, rows: usize) -> Result<Vec<u8>> {
        let bytes = self.buffer.read(0, self.buffer.bytes)?;
        Ok(guarded_payload(&bytes, rows, self.element_bytes, self.poison)?.to_vec())
    }
}
struct Outputs {
    f32: Guarded,
    cast: Guarded,
    direct: Guarded,
}
impl Outputs {
    fn new(d: &Rc<Device>) -> Result<Self> {
        Ok(Self {
            f32: Guarded::new(d, 4, 0xda)?,
            cast: Guarded::new(d, 2, 0xa5)?,
            direct: Guarded::new(d, 2, 0x6d)?,
        })
    }
    fn reset(&self) -> Result<()> {
        self.f32.reset()?;
        self.cast.reset()?;
        self.direct.reset()
    }
    fn check(&self, rows: usize) -> Result<Vec<u8>> {
        let reference = self.f32.read(rows)?;
        ensure!(
            reference
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| f32::from_le_bytes(*v).is_finite()),
            "finite fixture produced a nonfinite FP32 partial"
        );
        let cast = self.cast.read(rows)?;
        let direct = self.direct.read(rows)?;
        ensure!(cast == direct, "direct BF16 differs from FP32 plus cast");
        Ok(direct)
    }
}
fn cast(k: &Kernel, s: &Stream, outputs: &Outputs, rows: usize) -> Result<()> {
    range(rows, 2)?;
    words(
        k,
        s,
        [(rows * WIDTH).div_ceil(256) as u32, 1, 1],
        &[
            outputs.f32.pointer(),
            outputs.cast.pointer(),
            (rows * WIDTH) as u64,
        ],
    )
}

/// Includes signed zeros, small and large finite values and varying signs.
/// All entries are explicitly BF16 before either matrix sees them.
fn matrix_input(rows: usize, width: usize) -> Vec<u8> {
    (0..rows * width)
        .flat_map(|i| {
            let value = match i % 37 {
                0 => -0.,
                1 => 0.,
                2 => 2f32.powi(-40),
                3 => -2f32.powi(-40),
                4 => 128.,
                5 => -128.,
                _ => ((i.wrapping_mul(173).wrapping_add(19) % 257) as f32 - 128.) / 128.,
            };
            ((bf(value).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}
fn weight(row: usize, slot: usize) -> f32 {
    match row % 4 {
        0 => 1.,
        1 => {
            if slot.is_multiple_of(2) {
                0.5
            } else {
                -0.25
            }
        }
        2 => [1., 0.1, -0.3, 0.7, -0.9, 0.01, 0.001, -1., 0.5][slot],
        _ => {
            if slot.is_multiple_of(2) {
                2f32.powi(-20)
            } else {
                2f32.powi(20)
            }
        }
    }
}
fn value(row: usize, column: usize, slot: usize) -> f32 {
    // The last shared slot lives in a distinct allocation. The unused ninth
    // routed slot is separately poisoned, so a wrong stride cannot pass.
    match (column + row * 3) % 12 {
        0 => [16777216., 1., -16777216., 0., 0., 0., 0., 0., 0.][slot],
        1 => [
            1e20,
            -1e20,
            1e-20,
            -1e-20,
            0.,
            0.,
            0.,
            0.,
            f32::from_bits(0x3f808000),
        ][slot],
        2 => {
            if slot == 8 {
                f32::from_bits(0x3f808000)
            } else {
                0.
            }
        }
        3 => {
            if slot == 8 {
                f32::from_bits(0x3f808001)
            } else {
                0.
            }
        }
        4 => {
            if slot == 8 {
                f32::from_bits(0x3f807fff)
            } else {
                0.
            }
        }
        5 => [1., 2f32.powi(-8), 2f32.powi(-23), 0., 0., 0., 0., 0., 0.][slot],
        6 => -0.,
        7 => {
            if slot.is_multiple_of(2) {
                2f32.powi(-100)
            } else {
                -2f32.powi(-100)
            }
        }
        8 => {
            if slot.is_multiple_of(2) {
                2f32.powi(100)
            } else {
                -2f32.powi(100)
            }
        }
        9 => {
            if slot == 8 {
                f32::from_bits(0x80010000)
            } else {
                0.
            }
        }
        10 => {
            if slot == 8 {
                f32::from_bits(0x00010000)
            } else {
                0.
            }
        }
        _ => ((row * 101 + column * 17 + slot * 29) % 513) as f32 / 128. - 2.,
    }
}
fn combine_inputs(rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut experts = Vec::with_capacity(rows * SLOTS * WIDTH * 4);
    let mut shared = Vec::with_capacity(rows * WIDTH * 4);
    let mut weights = Vec::with_capacity(rows * SLOTS * 4);
    for row in 0..rows {
        for slot in 0..SLOTS {
            weights.extend_from_slice(&weight(row, slot).to_le_bytes());
            for column in 0..WIDTH {
                let x = if slot == 8 {
                    -65536.
                } else {
                    value(row, column, slot)
                };
                experts.extend_from_slice(&x.to_le_bytes());
            }
        }
        for column in 0..WIDTH {
            shared.extend_from_slice(&value(row, column, 8).to_le_bytes());
        }
    }
    (experts, shared, weights)
}

fn qualify(
    s: &Stream,
    outputs: &Outputs,
    rows: usize,
    launch: impl Fn() -> Result<()>,
) -> Result<String> {
    outputs.reset()?;
    launch()?;
    s.sync()?;
    let expected = outputs.check(rows)?;
    // Poison outputs before each replay. Two replays ensure capture did not
    // merely observe a preceding eager result or leave an inactive tail dirty.
    let graph = unsafe { s.capture(&launch)? };
    for _ in 0..2 {
        outputs.reset()?;
        unsafe { graph.launch(s)? };
        s.sync()?;
        ensure!(
            outputs.check(rows)? == expected,
            "partial graph replay changed"
        );
    }
    Ok(hash(&expected))
}

pub(crate) fn run(p: &Prepared, root: &Path, native: &Path) -> Result<Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    // Select only the three canonical matrix storages; no target forward or
    // unrelated expert weights are needed by this component.
    let loaded = p.load_selected(&d, &MATRICES.map(|(stem, _)| stem))?;
    let gemms = Gemms::new(&d, root, native)?;
    let cast_kernel = load_native(&d, &native.join("prefill.cubin"), "cast_bf16", 256)?;
    let outputs = Outputs::new(&d)?;
    let mut matrix_cases = Vec::new();
    for (stem, width) in MATRICES {
        p.expect(
            &format!("{stem}.weight"),
            "float8_e4m3fn",
            &[WIDTH as u64, width as u64],
        )?;
        let fp32 = gemms.prepare_f8(&d, &s, p, &loaded, stem, true)?;
        let bf16 = gemms.prepare_f8(&d, &s, p, &loaded, stem, false)?;
        s.sync()?;
        let input = d.alloc(MAX_ROWS * width * 2)?;
        for rows in ROWS {
            let bytes = matrix_input(rows, width);
            input.write(0, &bytes)?;
            let output_sha256 = qualify(&s, &outputs, rows, || {
                fp32.run(&s, input.ptr, width, outputs.f32.pointer(), rows)?;
                cast(&cast_kernel, &s, &outputs, rows)?;
                bf16.run(&s, input.ptr, width, outputs.direct.pointer(), rows)
            })?;
            matrix_cases.push(json!({"matrix":stem,"rows":rows,"input_width":width,
                "input_sha256":hash(&bytes),"output_sha256":output_sha256,
                "exact_bf16":true,"guards_and_inactive_rows":true,"graph_replays":2}));
        }
    }
    let combine = load_kernel(&d, root, "combine")?;
    let combine_bf16 = load_kernel(&d, root, "combine_bf16")?;
    let mut combine_cases = Vec::new();
    for rows in ROWS {
        let (experts, shared, weights) = combine_inputs(rows);
        let y = d.alloc(experts.len())?;
        let sy = d.alloc(shared.len())?;
        let w = d.alloc(weights.len())?;
        y.write(0, &experts)?;
        sy.write(0, &shared)?;
        w.write(0, &weights)?;
        let output_sha256 = qualify(&s, &outputs, rows, || {
            words(
                &combine,
                &s,
                [rows as u32, 4, 1],
                &[y.ptr, sy.ptr, w.ptr, outputs.f32.pointer()],
            )?;
            cast(&cast_kernel, &s, &outputs, rows)?;
            words(
                &combine_bf16,
                &s,
                [rows as u32, 4, 1],
                &[y.ptr, sy.ptr, w.ptr, outputs.direct.pointer()],
            )
        })?;
        combine_cases.push(json!({"rows":rows,"expert_input_sha256":hash(&experts),
            "shared_input_sha256":hash(&shared),"weights_sha256":hash(&weights),
            "output_sha256":output_sha256,"exact_bf16":true,
            "guards_and_inactive_rows":true,"graph_replays":2}));
    }
    let manifest: Value = serde_json::from_str(include_str!("../native/kernel-manifest.json"))?;
    let mut kernels = serde_json::Map::new();
    for alias in ["combine".to_owned(), "combine_bf16".to_owned()]
        .into_iter()
        .chain(MATRICES.into_iter().flat_map(|(_, width)| {
            ["f32", "bf16"].map(move |ty| format!("prefill_b16_{WIDTH}_{width}_{ty}"))
        }))
    {
        kernels.insert(alias.clone(), manifest[&alias].clone());
    }
    Ok(json!({"all":true,"version":1,"rank":p.manifest.key["rank"],
        "matrices":matrix_cases,"combine":combine_cases,"kernels":kernels,
        "cast_cubin_sha256":hash(&std::fs::read(native.join("prefill.cubin"))?),
        "guard_bytes":GUARD,"max_rows":MAX_ROWS,"fixture":"finite-cancellation-rounding-v1",
        "python_executed":false,"model_forward_executed":false,
        "scope":"Single-GPU exact final-partial GEMM/combine bytes and capture replay; pipe ownership, actual model state, collective ordering and speed require separate qualification"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guarded_layout_rejects_each_boundary_and_inactive_row_corruption() -> Result<()> {
        for size in [2, 4] {
            for rows in ROWS {
                let live = range(rows, size)?;
                let mut bytes = vec![0xa5; 2 * GUARD + MAX_ROWS * WIDTH * size];
                bytes[live.clone()].fill(0);
                assert_eq!(
                    guarded_payload(&bytes, rows, size, 0xa5)?.len(),
                    rows * WIDTH * size
                );
                for at in [0, live.start - 1, live.end, bytes.len() - 1] {
                    bytes[at] = 0;
                    assert!(guarded_payload(&bytes, rows, size, 0xa5).is_err());
                    bytes[at] = 0xa5;
                }
            }
        }
        for (rows, size) in [(0, 2), (MAX_ROWS + 1, 2), (1, 1), (1, 8)] {
            assert!(range(rows, size).is_err());
        }
        Ok(())
    }
    #[test]
    fn fixtures_cover_finite_cancellation_rounding_and_separate_shared_storage() {
        assert_eq!(value(0, 0, 0), -value(0, 0, 2));
        assert_eq!(value(0, 0, 1), 1.);
        assert_eq!(value(0, 2, 8).to_bits(), 0x3f808000);
        assert_eq!(value(0, 3, 8).to_bits(), 0x3f808001);
        assert_eq!(value(0, 4, 8).to_bits(), 0x3f807fff);
        assert_eq!(value(0, 6, 8).to_bits(), (-0f32).to_bits());
        let (experts, shared, weights) = combine_inputs(4);
        assert_eq!(experts.len(), 4 * SLOTS * WIDTH * 4);
        assert_eq!(shared.len(), 4 * WIDTH * 4);
        assert_eq!(weights.len(), 4 * SLOTS * 4);
        for bytes in [&experts, &shared, &weights] {
            assert!(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v).is_finite())
            );
        }
        for row in 0..4 {
            assert!(
                experts[(row * SLOTS + 8) * WIDTH * 4..(row * SLOTS + 9) * WIDTH * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v) == -65536.)
            );
        }
        for (_, width) in MATRICES {
            let input = matrix_input(65, width);
            assert_eq!(input.len(), 65 * width * 2);
            assert!(
                input.as_chunks::<2>().0.iter().all(|v| {
                    f32::from_bits(u32::from(u16::from_le_bytes(*v)) << 16).is_finite()
                })
            );
        }
    }
}
