//! Small multirow FP8 projections with the serial decoder's arithmetic.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream, arg},
    front::{load_kernel, reduce},
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub alias: &'static str,
    pub stem: &'static str,
    pub n: usize,
    pub k: usize,
    pub sk: u32,
    pub f32: bool,
}
pub const SHAPES: [Shape; 10] = [
    Shape {
        alias: "kda_proj",
        stem: "weights.layers[0].kda.proj",
        n: 6416,
        k: 4096,
        sk: 2,
        f32: false,
    },
    Shape {
        alias: "kda_gate",
        stem: "weights.layers[0].kda.fb",
        n: 2048,
        k: 128,
        sk: 1,
        f32: false,
    },
    Shape {
        alias: "kda_out",
        stem: "weights.layers[0].kda.o",
        n: 4096,
        k: 2048,
        sk: 4,
        f32: true,
    },
    Shape {
        alias: "mlp_gu",
        stem: "weights.layers[0].mlp.gu",
        n: 6144,
        k: 4096,
        sk: 2,
        f32: false,
    },
    Shape {
        alias: "mlp_down",
        stem: "weights.layers[0].mlp.down",
        n: 4096,
        k: 3072,
        sk: 4,
        f32: true,
    },
    Shape {
        alias: "dsa_proj",
        stem: "weights.layers[3].dsa.proj",
        n: 2048,
        k: 4096,
        sk: 8,
        f32: false,
    },
    Shape {
        alias: "dsa_qb",
        stem: "weights.layers[3].dsa.q_b",
        n: 4096,
        k: 1536,
        sk: 2,
        f32: false,
    },
    Shape {
        alias: "dsa_out",
        stem: "weights.layers[3].dsa.o",
        n: 4096,
        k: 4096,
        sk: 4,
        f32: true,
    },
    Shape {
        alias: "shared_gu",
        stem: "weights.layers[3].moe.shared.gu",
        n: 1024,
        k: 4096,
        sk: 8,
        f32: false,
    },
    Shape {
        alias: "shared_down",
        stem: "weights.layers[3].moe.shared.down",
        n: 4096,
        k: 512,
        sk: 1,
        f32: true,
    },
];
pub struct Projection {
    pub shape: Shape,
    kernel: Kernel,
    reduction: Option<Kernel>,
}
impl Projection {
    pub fn new(d: &Rc<Device>, root: &Path, shape: Shape) -> Result<Self> {
        ensure!(
            SHAPES.contains(&shape),
            "unqualified verifier projection geometry"
        );
        let reduction = if shape.sk > 1 {
            Some(load_kernel(
                d,
                root,
                &format!(
                    "reduce{}_{}",
                    shape.sk,
                    if shape.f32 { "f32" } else { "bf16" }
                ),
            )?)
        } else {
            None
        };
        Ok(Self {
            shape,
            kernel: load_kernel(d, root, &format!("verify_{}", shape.alias))?,
            reduction,
        })
    }
    /// Buffers/weights must share the stream context and outlive completion or
    /// graph capture. No allocation or host synchronization in the launch path.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        stream: &Stream,
        input: &Buffer,
        input_stride: usize,
        weight: u64,
        scale: u64,
        output: &Buffer,
        partial: &Buffer,
        rows: usize,
    ) -> Result<()> {
        unsafe {
            self.run_offset(
                stream,
                input,
                0,
                input_stride,
                weight,
                scale,
                output,
                partial,
                rows,
            )
        }
    }
    /// As run, with a checked byte offset into the source allocation.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_offset(
        &self,
        stream: &Stream,
        input: &Buffer,
        offset: usize,
        input_stride: usize,
        weight: u64,
        scale: u64,
        output: &Buffer,
        partial: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let s = self.shape;
        ensure!(
            (1..=crate::kda_window::MAX_ROWS).contains(&rows),
            "verifier projection row bound"
        );
        ensure!(
            input_stride >= s.k
                && input_stride <= i32::MAX as usize
                && input_stride.is_multiple_of(16),
            "verifier input stride"
        );
        let needed = (rows - 1)
            .checked_mul(input_stride)
            .and_then(|n| n.checked_add(s.k))
            .and_then(|n| n.checked_mul(2));
        ensure!(
            offset.is_multiple_of(2)
                && needed
                    .and_then(|n| n.checked_add(offset))
                    .is_some_and(|n| n <= input.bytes),
            "verifier input bounds"
        );
        ensure!(
            output.bytes >= rows * s.n * if s.f32 { 4 } else { 2 }
                && partial.bytes >= rows * s.n * s.sk as usize * 4,
            "verifier output bounds"
        );
        let mut words = [
            input.ptr + offset as u64,
            weight,
            scale,
            output.ptr,
            partial.ptr,
            rows as u64,
            input_stride as u64,
            s.n as u64,
            0,
            0,
        ];
        unsafe {
            self.kernel.launch(
                stream,
                [1, s.n.div_ceil(64) as u32, s.sk],
                &mut words.each_mut().map(arg),
            )?;
        }
        if let Some(k) = &self.reduction {
            reduce(k, stream, partial.ptr, output.ptr, (rows * s.n) as i32)?;
        }
        Ok(())
    }
}
