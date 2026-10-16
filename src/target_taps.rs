//! Bounded target-feature capture. Proposal rows have a separate bounded
//! owner; only accepted rows may be copied into committed history.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::load_native,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};
pub const WIDTH: usize = 5 * 4096 * 2;
pub struct Taps {
    kernel: Kernel,
    pub rows: Buffer,
    pub position: Buffer,
    pub cap: usize,
}
impl Taps {
    pub fn new(d: &Rc<Device>, native: &Path, cap: usize) -> Result<Self> {
        ensure!(matches!(cap, 8 | 16 | 2048), "target tap capacity");
        let rows = d.alloc(cap * WIDTH)?;
        rows.zero()?;
        let position = d.alloc(4)?;
        position.zero()?;
        Ok(Self {
            kernel: load_native(d, &native.join("target_taps.cubin"), "target_taps", 256)?,
            rows,
            position,
            cap,
        })
    }
    pub fn set_position(&self, s: &Stream, pos: usize) -> Result<()> {
        ensure!(pos <= 1048576, "target tap position");
        unsafe { self.position.set_i32(s, pos as i32) }
    }
    pub fn capture(&self, s: &Stream, source: &Buffer, rows: usize, layer: usize) -> Result<()> {
        let Some(layer) = [5, 14, 24, 33, 42].iter().position(|&v| v == layer) else {
            return Ok(());
        };
        ensure!(
            rows > 0 && rows * 4 * 4096 * 2 <= source.bytes,
            "target tap input"
        );
        let n = rows.min(self.cap);
        self.capture_range(s, source, rows - n, n, layer)
    }
    /// Capture a checked subset into its original absolute ring positions.
    /// Layer is the already resolved tap index, not a transformer layer number.
    pub(crate) fn capture_range(
        &self,
        s: &Stream,
        source: &Buffer,
        first: usize,
        rows: usize,
        layer: usize,
    ) -> Result<()> {
        ensure!(
            layer < 5
                && rows <= self.cap
                && first <= source.bytes / 32768
                && rows <= source.bytes / 32768 - first,
            "target tap range"
        );
        if rows == 0 {
            return Ok(());
        }
        words(
            &self.kernel,
            s,
            [(rows * 4096).div_ceil(256) as u32, 1, 1],
            &[
                source.ptr,
                self.rows.ptr,
                self.position.ptr,
                first as u64,
                rows as u64,
                self.cap as u64,
                layer as u64,
            ],
        )
    }
    pub fn copy_to(
        &self,
        s: &Stream,
        absolute: usize,
        n: usize,
        destination: &Buffer,
    ) -> Result<()> {
        ensure!(
            n <= self.cap && n * WIDTH <= destination.bytes,
            "target tap copy bound"
        );
        let start = absolute % self.cap;
        let first = n.min(self.cap - start);
        unsafe {
            destination.copy_from(s, 0, &self.rows, start * WIDTH, first * WIDTH)?;
            if first < n {
                destination.copy_from(s, first * WIDTH, &self.rows, 0, (n - first) * WIDTH)?;
            }
        }
        Ok(())
    }
    pub fn append(&self, s: &Stream, position: usize, source: &Taps, keep: usize) -> Result<()> {
        ensure!(
            matches!(source.cap, 8 | 16) && keep <= source.cap && self.cap == 2048,
            "target tap commit geometry"
        );
        for row in 0..keep {
            unsafe {
                self.rows.copy_from(
                    s,
                    ((position + row) % self.cap) * WIDTH,
                    &source.rows,
                    row * WIDTH,
                    WIDTH,
                )?;
            }
        }
        Ok(())
    }
}
