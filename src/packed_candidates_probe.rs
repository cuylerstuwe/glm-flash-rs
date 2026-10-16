//! GPU qualification against the existing pinned ATen candidate oracle. Not
//! called during normal serving; does not load model weights or invoke Python.
use crate::{
    candidates,
    candidates_probe::{expected, fixtures},
    cuda::Device,
    packed_candidates::Packed,
    packed_candidates_layout::{LOCAL, Layout, ROWS},
    sampling::{self, Sampling},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;
pub(crate) fn run(native: &Path, oracle: &Path) -> Result<serde_json::Value> {
    let f = fixtures(oracle)?;
    let d = Device::new()?;
    let s = d.stream()?;
    let logits = d.alloc(ROWS * LOCAL * 2)?;
    let mut packed = Packed::new(&d, native)?;
    let mut hash = Sha256::new();
    let mut shapes = 0;
    let mut rows_checked = 0;
    for n in [1, 2, 7, 16] {
        for (ki, k) in [8, 28, 64, 136, 512, LOCAL].into_iter().enumerate() {
            let layout = Layout::new(n, k)?;
            let mut gathered = Vec::with_capacity(layout.gathered_bytes);
            let mut row_candidates = vec![Vec::new(); n];
            for rank in 0..4 {
                let mut data = vec![0; logits.bytes];
                for b in data.as_chunks_mut::<2>().0 {
                    b.copy_from_slice(&0x7fc1u16.to_le_bytes());
                }
                let mut want = Vec::new();
                for row in 0..n {
                    let fixture = &f[((rank + row) % 10) * 7 + ki + 1];
                    ensure!(fixture.k == k, "packed fixture shape");
                    data[row * LOCAL * 2..(row + 1) * LOCAL * 2].copy_from_slice(&fixture.logits);
                    want.extend(expected(fixture, rank));
                }
                logits.write(0, &data)?;
                // Capture and replay the exact row grid; unused logits are NaN.
                let graph = unsafe { s.capture(|| packed.local(&s, &logits, rank, layout))? };
                unsafe {
                    graph.launch(&s)?;
                }
                s.sync()?;
                let actual = packed.local.read(0, layout.rank_bytes)?;
                ensure!(
                    actual == want,
                    "packed selected set/value differs from ATen oracle"
                );
                packed.local.write(0, &vec![0xa5; layout.rank_bytes])?;
                unsafe {
                    graph.launch(&s)?;
                }
                s.sync()?;
                ensure!(
                    packed.local.read(0, layout.rank_bytes)? == actual,
                    "packed graph replay failed to overwrite poisoned destination"
                );
                hash.update(&actual);
                gathered.extend(actual);
            }
            for (row, out) in row_candidates.iter_mut().enumerate() {
                let mut scalar = Vec::new();
                for rank in 0..4 {
                    scalar.extend(expected(&f[((rank + row) % 10) * 7 + ki + 1], rank));
                }
                *out = candidates::decode(&scalar, k)?;
            }
            packed.gathered.write(0, &gathered)?;
            for (seed, temperature, top_p, position) in [
                (450, 1., 0.95, 0),
                (u64::MAX, 0.01, 0., u64::MAX - 15),
                (17, 1e-12, 1., 500001),
                (731, 2., 0.5, 1048559),
            ] {
                let top = if k == LOCAL { sampling::VOCAB } else { k - 8 };
                let policy = Sampling::new(seed, temperature, top, top_p)?;
                if policy.gpu_fits(4 * k) {
                    unsafe {
                        packed.gpu(&s, layout, policy, position)?;
                    }
                }
                s.sync()?;
                let actual = packed.read(layout, policy, position)?;
                for (row, expected) in row_candidates.iter().enumerate() {
                    ensure!(
                        actual[row] == sampling::choose(expected, position + row as u64, policy)?,
                        "packed positioned sample changed"
                    );
                    hash.update(actual[row].0.to_le_bytes());
                    hash.update((actual[row].1 as u64).to_le_bytes());
                    rows_checked += 1;
                }
            }
            shapes += 1;
        }
    }
    // Corrupt each rank's final row, then recover with exactly the same owner.
    let policy = Sampling::new(450, 1., 20, 0.95)?;
    let layout = Layout::new(16, 28)?;
    let mut good = Vec::new();
    for rank in 0..4 {
        for row in 0..16 {
            good.extend(expected(&f[((rank + row) % 10) * 7 + 2], rank));
        }
    }
    let want = layout.choose(&good, policy, 8191)?;
    let mut faults = 0;
    for rank in 0..4 {
        for (off, value) in [
            (0, 0x7f800000u32),
            (0, 0xff800000),
            (0, 0x7fc10000),
            (0, 1),
            (4, 154880),
            (12, (rank * LOCAL) as u32),
            (28 * 8, 1),
            (28 * 8 + 4, 27),
        ] {
            let mut bad = good.clone();
            let at = layout.range(rank, 15)?.start + off;
            bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
            packed.gathered.write(0, &bad)?;
            unsafe {
                packed.gpu(&s, layout, policy, 8191)?;
            }
            s.sync()?;
            ensure!(
                packed.read(layout, policy, 8191).is_err(),
                "packed malformed row accepted"
            );
            packed.gathered.write(0, &good)?;
            unsafe {
                packed.gpu(&s, layout, policy, 8191)?;
            }
            s.sync()?;
            ensure!(
                packed.read(layout, policy, 8191)? == want,
                "packed good owner failed to recover"
            );
            faults += 1;
        }
    }
    // A nonfinite logit below the selected cutoff must also invalidate its row.
    for bad in [0x7f80u16, 0xff80, 0x7fc1, 0xffc1] {
        let mut data = vec![0; 16 * LOCAL * 2];
        for row in 0..16 {
            data[row * LOCAL * 2..(row + 1) * LOCAL * 2].copy_from_slice(&f[2].logits);
        }
        data[(15 * LOCAL + LOCAL - 1) * 2..(16 * LOCAL) * 2].copy_from_slice(&bad.to_le_bytes());
        logits.write(0, &data)?;
        packed.local(&s, &logits, 0, layout)?;
        s.sync()?;
        let local = packed.local.read(0, layout.rank_bytes)?;
        let status = 15 * layout.row_bytes + 28 * 8;
        ensure!(
            u32::from_le_bytes(local[status..status + 4].try_into()?) != 0,
            "nonfinite inactive-cutoff logit accepted"
        );
    }
    Ok(
        serde_json::json!({"shape_cases":shapes,"sample_rows_checked":rows_checked,"selected_sets_match_pinned_aten":true,"positioned_draws_match_scalar":true,"payload_faults_rejected_and_recovered":faults,"nonfinite_classes_rejected":4,"inactive_rows_poisoned":true,"sha256":format!("{:x}",hash.finalize()),"python_executed":false,"model_executed":false,"scope":"single-GPU row collection and draw; actual transport and model acceptance require separate qualification"}),
    )
}
