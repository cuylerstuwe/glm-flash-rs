//! Exact greedy selection; stochastic production sampling is a separate gate.
#[path = "greedy_rows.rs"]
mod rows;
use crate::{
    cuda::{Buffer, Device, Kernel, Stream, arg},
    front::load_native,
};
use anyhow::{Result, ensure};
pub use rows::reference;
use std::{path::Path, rc::Rc};

pub struct Greedy {
    local_kernel: Kernel,
    merge_kernel: Kernel,
    pub local: Buffer,
    pub gathered: Buffer,
    result: Buffer,
}
impl Greedy {
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        Ok(Self {
            local_kernel: load_native(device, &native.join("select.cubin"), "greedy_local", 256)?,
            merge_kernel: load_native(device, &native.join("select.cubin"), "greedy_merge", 32)?,
            local: device.alloc(16)?,
            gathered: device.alloc(64)?,
            result: device.alloc(8)?,
        })
    }
    pub fn local(&self, stream: &Stream, logits: &Buffer, rank: usize) -> Result<()> {
        ensure!(
            rank < 4 && logits.bytes == 38720 * 2,
            "expected one TP4 vocabulary shard"
        );
        self.local_offset(stream, logits, 0, rank)
    }
    /// Use the exact serial greedy kernel on a checked BF16 row of a wider head.
    pub(crate) fn local_offset(
        &self,
        stream: &Stream,
        logits: &Buffer,
        offset: usize,
        rank: usize,
    ) -> Result<()> {
        let ptr = rows::address(logits.ptr, logits.bytes, offset, rank)?;
        let mut values = [
            ptr,
            (rank * rows::LOCAL) as u64,
            rows::LOCAL as u64,
            self.local.ptr,
        ];
        let mut args = values.iter_mut().map(arg).collect::<Vec<_>>();
        unsafe { self.local_kernel.launch(stream, [1, 1, 1], &mut args) }
    }
    pub fn merge(&self, stream: &Stream) -> Result<()> {
        let mut values = [self.gathered.ptr, self.result.ptr];
        let mut args = values.iter_mut().map(arg).collect::<Vec<_>>();
        unsafe { self.merge_kernel.launch(stream, [1, 1, 1], &mut args) }
    }
    pub fn read(&self) -> Result<i32> {
        let mut data = [0u8; 8];
        self.result.read_into(0, &mut data)?;
        ensure!(
            i32::from_le_bytes(data[4..].try_into()?) == 0,
            "nonfinite model logits"
        );
        let token = i32::from_le_bytes(data[..4].try_into()?);
        ensure!((0..154880).contains(&token), "invalid selected token");
        Ok(token)
    }
}

/// The same exact selection for up to sixteen verifier rows at once: one local
/// launch, one gather of `rows * 16` bytes per rank, one merge, one readback.
pub struct GreedyRows {
    local_kernel: Kernel,
    merge_kernel: Kernel,
    pub local: Buffer,
    pub gathered: Buffer,
    result: Buffer,
}
impl GreedyRows {
    pub const MAX: usize = 16;
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        let cubin = native.join("select.cubin");
        Ok(Self {
            local_kernel: load_native(device, &cubin, "greedy_local_rows", 256)?,
            merge_kernel: load_native(device, &cubin, "greedy_merge_rows", 32)?,
            local: device.alloc(Self::MAX * 16)?,
            gathered: device.alloc(4 * Self::MAX * 16)?,
            result: device.alloc(Self::MAX * 8)?,
        })
    }
    /// `logits` holds `rows` contiguous BF16 rows of this rank's vocabulary shard.
    pub fn local(&self, stream: &Stream, logits: &Buffer, rows: usize, rank: usize) -> Result<()> {
        ensure!(
            (1..=Self::MAX).contains(&rows) && rank < 4 && logits.bytes >= rows * rows::ROW_BYTES,
            "greedy row batch bounds"
        );
        let mut values = [
            logits.ptr,
            (rank * rows::LOCAL) as u64,
            rows::LOCAL as u64,
            rows::LOCAL as u64,
            self.local.ptr,
        ];
        let mut args = values.iter_mut().map(arg).collect::<Vec<_>>();
        unsafe {
            self.local_kernel
                .launch(stream, [rows as u32, 1, 1], &mut args)
        }
    }
    /// The rows' local records must already be gathered rank-major (prefix bytes).
    pub fn merge(&self, stream: &Stream, rows: usize) -> Result<()> {
        ensure!((1..=Self::MAX).contains(&rows), "greedy row batch bounds");
        let mut values = [self.gathered.ptr, rows as u64, self.result.ptr];
        let mut args = values.iter_mut().map(arg).collect::<Vec<_>>();
        unsafe { self.merge_kernel.launch(stream, [1, 1, 1], &mut args) }
    }
    pub fn read(&self, rows: usize) -> Result<Vec<i32>> {
        ensure!((1..=Self::MAX).contains(&rows), "greedy row batch bounds");
        let data = self.result.read(0, rows * 8)?;
        data.as_chunks::<8>()
            .0
            .iter()
            .map(|row| {
                ensure!(
                    i32::from_le_bytes(row[4..].try_into()?) == 0,
                    "nonfinite model logits"
                );
                let token = i32::from_le_bytes(row[..4].try_into()?);
                ensure!((0..154880).contains(&token), "invalid selected token");
                Ok(token)
            })
            .collect()
    }
}

pub fn probe(native: &Path) -> Result<serde_json::Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let selector = Greedy::new(&device, native)?;
    let logits = device.alloc(38720 * 2)?;
    let row_logits = device.alloc(16 * rows::ROW_BYTES)?;
    let poison = vec![0x7fc1u16; 16 * rows::LOCAL]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut seed = 0x19fa8432u32;
    let mut cases = Vec::new();
    let mut offset_cases = Vec::new();
    for case in 0..8 {
        let mut bytes = Vec::new();
        for _ in 0..154880 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let bits = match case {
                0 => (seed as u16 & 0x807f) | ((seed as u16 % 254) << 7),
                1 => 0x8000,
                2 => 0,
                3 => 0xbf80,
                _ => 0,
            };
            bytes.extend(bits.to_le_bytes());
        }
        if case == 2 {
            bytes[..2].copy_from_slice(&0x8000u16.to_le_bytes());
        }
        if case == 3 {
            for id in [38719, 38720, 77440, 154879] {
                bytes[id * 2..id * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            }
        }
        if case >= 4 {
            let bad = [0x7f80u16, 0xff80, 0x7fc1, 0xffc1][case - 4];
            bytes[(case - 4) * 38720 * 2..(case - 4) * 38720 * 2 + 2]
                .copy_from_slice(&bad.to_le_bytes());
        }
        let mut candidates = Vec::new();
        for rank in 0..4 {
            logits.write(0, &bytes[rank * 38720 * 2..(rank + 1) * 38720 * 2])?;
            selector.local(&stream, &logits, rank)?;
            stream.sync()?;
            candidates.extend(selector.local.read(0, 16)?);
        }
        selector.gathered.write(0, &candidates)?;
        selector.merge(&stream)?;
        stream.sync()?;
        if case < 4 {
            let expected = reference(&bytes)?;
            ensure!(
                selector.read()? as usize == expected,
                "greedy mismatch in case {case}"
            );
            cases.push(serde_json::json!({"case":case,"token":expected,"exact":true}));
        } else {
            ensure!(
                selector.read().is_err() && reference(&bytes).is_err(),
                "nonfinite value was accepted"
            );
            cases.push(serde_json::json!({"case":case,"nonfinite_rejected":true}));
        }
        for row in [0usize, 1, 7, 15] {
            let mut candidates = Vec::new();
            for rank in 0..4 {
                row_logits.write(0, &poison)?;
                row_logits.write(
                    row * rows::ROW_BYTES,
                    &bytes[rank * rows::ROW_BYTES..(rank + 1) * rows::ROW_BYTES],
                )?;
                selector.local_offset(&stream, &row_logits, row * rows::ROW_BYTES, rank)?;
                stream.sync()?;
                candidates.extend(selector.local.read(0, 16)?);
            }
            selector.gathered.write(0, &candidates)?;
            selector.merge(&stream)?;
            stream.sync()?;
            if case < 4 {
                ensure!(
                    selector.read()? as usize == reference(&bytes)?,
                    "offset greedy mismatch"
                );
            } else {
                ensure!(
                    selector.read().is_err(),
                    "offset greedy accepted nonfinite row"
                );
            }
            offset_cases.push(
                serde_json::json!({"case":case,"row":row,"exact":true,"other_rows_poisoned":true}),
            );
        }
    }
    for offset in [1, row_logits.bytes, usize::MAX] {
        ensure!(
            selector
                .local_offset(&stream, &row_logits, offset, 0)
                .is_err(),
            "invalid greedy row offset accepted"
        );
    }
    ensure!(
        selector.local_offset(&stream, &row_logits, 0, 4).is_err(),
        "invalid greedy row rank accepted"
    );
    Ok(
        serde_json::json!({"cases":cases,"offset_cases":offset_cases,"invalid_offsets_rejected":true,"scope":"GPU selection from all four shard positions and bounded verifier rows, signed zeros, ties, NaN and infinities; no network or model arithmetic"}),
    )
}
