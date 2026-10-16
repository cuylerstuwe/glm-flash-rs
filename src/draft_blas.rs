//! Native cuBLAS: the drafter's BF16 selector, and the target's BF16 prefill
//! GEMMs only at shapes and sizes whose results match the retained kernels bit
//! for bit (`prefill::blas_rows`). No Python/PyTorch runtime is loaded.
use crate::cuda::{Buffer, Device, Stream};
use anyhow::{Result, ensure};
use libloading::Library;
use std::{ffi::c_void, ptr, rc::Rc};
type Handle = *mut c_void;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> i32;
type SetWorkspace = unsafe extern "C" fn(Handle, *mut c_void, usize) -> i32;
type Gemm = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
) -> i32;
pub struct Blas {
    handle: Handle,
    destroy: Destroy,
    set_stream: SetStream,
    set_workspace: SetWorkspace,
    workspace: Buffer,
    gemm: Gemm,
    _lib: Library,
}
fn check(v: i32) -> Result<()> {
    ensure!(v == 0, "cuBLAS status {v}");
    Ok(())
}
impl Blas {
    pub fn new(device: &Rc<Device>) -> Result<Self> {
        let lib = unsafe { Library::new("/usr/local/cuda/lib64/libcublas.so.13")? };
        let create: unsafe extern "C" fn(*mut Handle) -> i32 =
            unsafe { *lib.get(b"cublasCreate_v2\0")? };
        let destroy = unsafe { *lib.get(b"cublasDestroy_v2\0")? };
        let set_stream = unsafe { *lib.get(b"cublasSetStream_v2\0")? };
        let set_workspace = unsafe { *lib.get(b"cublasSetWorkspace_v2\0")? };
        let gemm = unsafe { *lib.get(b"cublasGemmEx\0")? };
        let workspace = device.alloc(8 * 1024 * 1024)?;
        let mut handle = ptr::null_mut();
        check(unsafe { create(&mut handle) })?;
        Ok(Self {
            handle,
            destroy,
            set_stream,
            set_workspace,
            workspace,
            gemm,
            _lib: lib,
        })
    }
    /// # Safety
    /// x is contiguous [rows,4096] BF16, w [256,4096] BF16, and both remain
    /// live until stream completion. out is distinct and stores [rows,256] BF16.
    pub unsafe fn selector(
        &self,
        s: &Stream,
        x: u64,
        w: u64,
        out: &Buffer,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&rows) && out.bytes >= rows * 256 * 2,
            "selector shape"
        );
        check(unsafe { (self.set_stream)(self.handle, s.raw()) })?;
        check(unsafe {
            (self.set_workspace)(
                self.handle,
                self.workspace.ptr as *mut c_void,
                self.workspace.bytes,
            )
        })?;
        let alpha = 1f32;
        let beta = 0f32;
        // CUBLAS_OP_T=1, OP_N=0, CUDA_R_16BF=14, COMPUTE_32F=68.
        // Explicit BF16 inputs, FP32 accumulation, BF16 result, no reduced input mode.
        check(unsafe {
            (self.gemm)(
                self.handle,
                1,
                0,
                256,
                rows as i32,
                4096,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                14,
                4096,
                x as *const c_void,
                14,
                4096,
                (&beta as *const f32).cast(),
                out.ptr as *mut c_void,
                14,
                256,
                68,
                99,
            )
        })
    }
    /// Diagnostic: `out[rows,n] = x[rows,k] * w[n,k]^T` with BF16 inputs and
    /// FP32 accumulation, BF16 or FP32 result (the retained prefill GEMM contract).
    ///
    /// # Safety
    /// w is row-major [n,k] BF16, x has `stride >= k` BF16 elements per row, out
    /// is distinct and holds [rows,n]; all stay live until stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        s: &Stream,
        w: u64,
        n: usize,
        k: usize,
        x: u64,
        stride: usize,
        out: u64,
        rows: usize,
        f32_out: bool,
    ) -> Result<()> {
        ensure!(rows > 0 && stride >= k, "cuBLAS GEMM shape");
        check(unsafe { (self.set_stream)(self.handle, s.raw()) })?;
        check(unsafe {
            (self.set_workspace)(
                self.handle,
                self.workspace.ptr as *mut c_void,
                self.workspace.bytes,
            )
        })?;
        let alpha = 1f32;
        let beta = 0f32;
        // CUDA_R_32F=0 for an FP32 result, otherwise CUDA_R_16BF=14.
        check(unsafe {
            (self.gemm)(
                self.handle,
                1,
                0,
                n as i32,
                rows as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                14,
                k as i32,
                x as *const c_void,
                14,
                stride as i32,
                (&beta as *const f32).cast(),
                out as *mut c_void,
                if f32_out { 0 } else { 14 },
                n as i32,
                68,
                99,
            )
        })
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.handle);
        }
    }
}
