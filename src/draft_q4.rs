//! DFlash2 affine Q4 projections. No conversion, JIT or row-dependent
//! reduction policy. Caller owns bounded scratch and all pointed-to weights.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    front::{load_kernel, reduce},
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};
#[derive(Clone, Copy)]
pub struct Shape {
    pub stem: &'static str,
    pub n: usize,
    pub k: usize,
    pub sk: usize,
    pub f32: bool,
}
pub const SHAPES: [Shape; 8] = [
    Shape {
        stem: "drafter.fc",
        n: 4096,
        k: 20480,
        sk: 4,
        f32: false,
    },
    Shape {
        stem: "drafter.layers[0].a_kp",
        n: 1024,
        k: 4096,
        sk: 8,
        f32: false,
    },
    Shape {
        stem: "drafter.layers[0].qkv",
        n: 1536,
        k: 4096,
        sk: 8,
        f32: false,
    },
    Shape {
        stem: "drafter.layers[0].kv",
        n: 512,
        k: 4096,
        sk: 8,
        f32: false,
    },
    Shape {
        stem: "drafter.layers[0].o",
        n: 4096,
        k: 1024,
        sk: 2,
        f32: true,
    },
    Shape {
        stem: "drafter.layers[0].gu",
        n: 6144,
        k: 4096,
        sk: 2,
        f32: false,
    },
    Shape {
        stem: "drafter.layers[0].down",
        n: 4096,
        k: 3072,
        sk: 4,
        f32: true,
    },
    // The drafter's 4-bit copy of this rank's BF16 vocabulary head
    // (prepared `weights.draft_head`); split-K 1 writes the output directly.
    Shape {
        stem: "weights.draft_head",
        n: 38720,
        k: 4096,
        sk: 1,
        f32: false,
    },
];
pub const DRAFT_HEAD: usize = 7;
pub struct Projection {
    shape: Shape,
    kernel: Kernel,
    reduction: Option<Kernel>,
}
impl Projection {
    pub fn new(d: &Rc<Device>, root: &Path, shape: Shape) -> Result<Self> {
        ensure!(
            SHAPES
                .iter()
                .any(|s| (s.n, s.k, s.sk, s.f32) == (shape.n, shape.k, shape.sk, shape.f32)),
            "unqualified draft Q4 shape"
        );
        let dtype = if shape.f32 { "f32" } else { "bf16" };
        Ok(Self {
            shape,
            kernel: load_kernel(
                d,
                root,
                &format!("draft_q4_{}_{}_{}", shape.n, shape.k, dtype),
            )?,
            reduction: if shape.sk > 1 {
                Some(load_kernel(
                    d,
                    root,
                    &format!("reduce{}_{}", shape.sk, dtype),
                )?)
            } else {
                None
            },
        })
    }
    /// # Safety
    /// Inputs, group sums and weights must have the indicated shapes/types, remain
    /// alive and unmodified through stream completion, and not alias output/scratch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        s: &Stream,
        input: u64,
        stride: usize,
        sums: u64,
        weights: [u64; 3],
        out: &Buffer,
        part: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let q = self.shape;
        ensure!(
            (1..=64).contains(&rows) && stride >= q.k && stride.is_multiple_of(16),
            "draft Q4 row/stride bound"
        );
        ensure!(
            out.bytes >= rows * q.n * if q.f32 { 4 } else { 2 }
                && (q.sk == 1 || part.bytes >= rows * q.n * q.sk * 4),
            "draft Q4 scratch bound"
        );
        let part = if q.sk == 1 { out.ptr } else { part.ptr };
        words(
            &self.kernel,
            s,
            [
                rows.div_ceil(16) as u32,
                q.n.div_ceil(64) as u32,
                q.sk as u32,
            ],
            &[
                input,
                sums,
                weights[0],
                weights[1],
                weights[2],
                out.ptr,
                part,
                rows as u64,
                stride as u64,
            ],
        )?;
        match &self.reduction {
            Some(reduction) => reduce(reduction, s, part, out.ptr, (rows * q.n) as i32),
            None => Ok(()),
        }
    }
}
