//! Exact speculative KDA transaction using pinned chain/replay kernels.
//! A proposal never changes the caller's committed recurrent or convolution state.
//! The model verifier uses this transaction to commit an accepted prefix.
use crate::cuda::{Buffer, Device, Kernel, Stream, arg};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc};

pub const MAX_ROWS: usize = 16;
pub const HEADS: usize = 16;
pub const CHANNELS: usize = HEADS * 128;
pub const CONV_CHANNELS: usize = CHANNELS * 3;
pub const PROJ_STRIDE: usize = 6416;
pub const STATE_BYTES: usize = HEADS * 128 * 128 * 4;
pub const CONV_BYTES: usize = 3 * CONV_CHANNELS * 2;

#[derive(Clone, Copy)]
pub struct Weights {
    pub conv: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub norm: u64,
}

pub struct Inputs<'a> {
    pub state: &'a Buffer,
    pub conv: &'a Buffer,
    pub projection: &'a Buffer,
    pub gate_a: &'a Buffer,
    pub gate_g: &'a Buffer,
}

pub struct Scratch {
    chain: Kernel,
    replay: Kernel,
    pub output: Buffer,
    pub proposal_state: Buffer,
    pub committed_state: Buffer,
    pub committed_conv: Buffer,
    k: Buffer,
    v: Buffer,
    g: Buffer,
    b: Buffer,
}

/// Exclusive scratch borrow prevents another proposal from overwriting its saves.
#[must_use = "commit the accepted prefix, or discard the proposal"]
pub struct Window<'a> {
    scratch: &'a mut Scratch,
    input: Inputs<'a>,
    rows: usize,
}

pub struct Committed<'a> {
    pub state: &'a Buffer,
    pub conv: &'a Buffer,
    pub rows: usize,
}

impl Scratch {
    pub fn new(device: &Rc<Device>, native: &Path) -> Result<Self> {
        Ok(Self {
            chain: crate::front::load_native(device, native, "chain_kernel", 1024)?,
            replay: crate::front::load_native(device, native, "replay_kernel", 1024)?,
            output: device.alloc(MAX_ROWS * CHANNELS * 2)?,
            proposal_state: device.alloc(STATE_BYTES)?,
            committed_state: device.alloc(STATE_BYTES)?,
            committed_conv: device.alloc(CONV_BYTES)?,
            k: device.alloc(MAX_ROWS * CHANNELS * 4)?,
            v: device.alloc(MAX_ROWS * CHANNELS * 2)?,
            g: device.alloc(MAX_ROWS * CHANNELS * 4)?,
            b: device.alloc(MAX_ROWS * HEADS * 4)?,
        })
    }

    /// Inputs, weights and scratch must belong to this CUDA context and outlive
    /// stream completion and any captured graph. The owner must not modify inputs
    /// until commit completes. No allocation, synchronization or readback here.
    pub unsafe fn evaluate<'a>(
        &'a mut self,
        stream: &Stream,
        input: Inputs<'a>,
        weights: Weights,
        rows: usize,
    ) -> Result<Window<'a>> {
        let state_out = self.proposal_state.ptr;
        unsafe { self.evaluate_to(stream, input, weights, rows, state_out) }
    }
    /// As `evaluate`, but the window's final state goes to `state_out` (the
    /// slot's other recurrent buffer) instead of this scratch's proposal state.
    /// # Safety
    /// As `evaluate`; `state_out` must be a distinct `STATE_BYTES` buffer that
    /// no other reader needs until commit.
    pub unsafe fn evaluate_into<'a>(
        &'a mut self,
        stream: &Stream,
        input: Inputs<'a>,
        weights: Weights,
        rows: usize,
        state_out: &Buffer,
    ) -> Result<Window<'a>> {
        ensure!(
            state_out.bytes >= STATE_BYTES && state_out.ptr != input.state.ptr,
            "KDA window state destination"
        );
        unsafe { self.evaluate_to(stream, input, weights, rows, state_out.ptr) }
    }
    unsafe fn evaluate_to<'a>(
        &'a mut self,
        stream: &Stream,
        input: Inputs<'a>,
        weights: Weights,
        rows: usize,
        state_out: u64,
    ) -> Result<Window<'a>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "KDA window row limit");
        ensure!(
            input.state.bytes >= STATE_BYTES && input.conv.bytes >= CONV_BYTES,
            "KDA window state bounds"
        );
        ensure!(
            input.projection.bytes >= rows * PROJ_STRIDE * 2
                && input.gate_a.bytes >= rows * CHANNELS * 2
                && input.gate_g.bytes >= rows * CHANNELS * 2,
            "KDA window input bounds"
        );
        let mut words = [
            HEADS as u64,
            input.projection.ptr,
            PROJ_STRIDE as u64,
            6400,
            input.gate_a.ptr,
            CHANNELS as u64,
            input.gate_g.ptr,
            CHANNELS as u64,
            input.conv.ptr,
            weights.conv,
            input.state.ptr,
            weights.a_log,
            weights.dt_bias,
            weights.norm,
            1e-5f32.to_bits() as u64,
            (-5f32).to_bits() as u64,
            rows as u64,
            self.output.ptr,
            state_out,
            self.k.ptr,
            self.v.ptr,
            self.g.ptr,
            self.b.ptr,
        ];
        unsafe {
            self.chain
                .launch(stream, [HEADS as u32, 1, 1], &mut words.each_mut().map(arg))?;
        }
        Ok(Window {
            scratch: self,
            input,
            rows,
        })
    }
}

impl Scratch {
    /// Replay from `state` into `dest` (the slot's other recurrent buffer).
    pub(crate) fn replay_pointers_into(&self, state: &Buffer, dest: &Buffer) -> [u64; 6] {
        [
            state.ptr, dest.ptr, self.k.ptr, self.v.ptr, self.g.ptr, self.b.ptr,
        ]
    }

    /// Stage the exact accepted convolution tail without modifying its source.
    /// Same lifetime and stream ownership contract as `commit_captured`.
    pub(crate) unsafe fn commit_conv(
        &self,
        stream: &Stream,
        conv: &Buffer,
        projection: &Buffer,
        rows: usize,
        keep: usize,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && keep <= rows,
            "KDA convolution accepted row bounds"
        );
        // Last three rows of [previous tail; accepted projected q|k|v].
        // No overlapping shifts and no rejected rows enter the retained tail.
        for row in 0..3 {
            let at = keep + row;
            let (from, src) = if at < 3 {
                (conv, at * CONV_CHANNELS * 2)
            } else {
                (projection, (at - 3) * PROJ_STRIDE * 2)
            };
            unsafe {
                self.committed_conv.copy_from(
                    stream,
                    row * CONV_CHANNELS * 2,
                    from,
                    src,
                    CONV_CHANNELS * 2,
                )?;
            }
        }
        Ok(())
    }

    /// Commit saves produced by a captured evaluate call on this exact scratch.
    /// # Safety
    /// The owner must have launched that graph with these input storage objects,
    /// row count, original committed state and convolution tail, with no scratch
    /// reuse or input mutation in between. All operations use the same stream.
    pub unsafe fn commit_captured<'a>(
        &'a mut self,
        stream: &Stream,
        input: Inputs<'a>,
        rows: usize,
        keep: usize,
    ) -> Result<Committed<'a>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "captured KDA row count");
        unsafe {
            Window {
                scratch: self,
                input,
                rows,
            }
            .commit(stream, keep)
        }
    }
}

impl<'a> Window<'a> {
    pub fn output(&self) -> &Buffer {
        &self.scratch.output
    }

    /// Finish into separate scratch; the model owner publishes these buffers only
    /// after every layer/rank commits and the stream drains. Partial proposals
    /// replay from the original state; accepting everything reuses the full result.
    /// A zero-row discard is supported for cancellation before publication.
    /// The same stream/lifetime requirements as evaluate apply.
    pub unsafe fn commit(self, stream: &Stream, keep: usize) -> Result<Committed<'a>> {
        ensure!(keep <= self.rows, "KDA accepted prefix exceeds proposal");
        let s = self.scratch;
        unsafe {
            if keep == 0 || keep == self.rows {
                let from = if keep == 0 {
                    self.input.state
                } else {
                    &s.proposal_state
                };
                s.committed_state
                    .copy_from(stream, 0, from, 0, STATE_BYTES)?;
            } else {
                let mut words = [
                    HEADS as u64,
                    self.input.state.ptr,
                    s.k.ptr,
                    s.v.ptr,
                    s.g.ptr,
                    s.b.ptr,
                    keep as u64,
                    s.committed_state.ptr,
                ];
                s.replay
                    .launch(stream, [HEADS as u32, 1, 1], &mut words.each_mut().map(arg))?;
            }
            s.commit_conv(
                stream,
                self.input.conv,
                self.input.projection,
                self.rows,
                keep,
            )?;
        }
        Ok(Committed {
            state: &s.committed_state,
            conv: &s.committed_conv,
            rows: keep,
        })
    }
}
