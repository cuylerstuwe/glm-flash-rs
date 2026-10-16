//! Pinned primitive ATen boundary; Rust owns all model sequencing and lifetimes.
//! Used by the vision encoder and differential probes; no Python/JIT.
use anyhow::{Context as _, Result, ensure};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, c_char, c_void},
    io::Read,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};
type Handle = *mut c_void;
type New = unsafe extern "C" fn(*mut Handle, *mut c_char, usize) -> i32;
type Free = unsafe extern "C" fn(Handle);
type Sync = unsafe extern "C" fn(Handle, *mut c_char, usize) -> i32;
type Empty = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    *const i64,
    usize,
    *mut Handle,
    *mut c_char,
    usize,
) -> i32;
type Upload =
    unsafe extern "C" fn(Handle, Handle, u64, *const u8, usize, *mut c_char, usize) -> i32;
type Download = unsafe extern "C" fn(Handle, Handle, *mut u8, usize, *mut c_char, usize) -> i32;
type Info = unsafe extern "C" fn(
    Handle,
    *mut i64,
    usize,
    *mut usize,
    *mut i32,
    *mut u64,
    *mut c_char,
    usize,
) -> i32;
type Op = unsafe extern "C" fn(
    Handle,
    i32,
    *const Handle,
    usize,
    *const i64,
    usize,
    *const f64,
    usize,
    *mut Handle,
    *mut c_char,
    usize,
) -> i32;
const ERROR: usize = 2048;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub(crate) enum Kind {
    Bf16 = 1,
    F32 = 2,
    I64 = 3,
}
#[derive(Clone, Copy)]
#[repr(i32)]
pub(crate) enum OpCode {
    Linear = 1,
    Reshape,
    Permute,
    Transpose,
    Slice,
    Select,
    Cat,
    Cast,
    Add,
    Mul,
    AddScalar,
    MulScalar,
    DivScalar,
    PowScalar,
    Rsqrt,
    Mean,
    Clamp,
    ClampMax,
    Silu,
    Cos,
    Sin,
    Sdpa,
    LayerNorm,
    Gelu,
    Arange,
    ScalarPow,
    Reciprocal,
    ToDevice,
    Unsqueeze,
    Expand,
    Neg,
}
struct Context {
    raw: NonNull<c_void>,
    free: Sync,
    tensor_free: Free,
    sync: Sync,
    empty: Empty,
    upload: Upload,
    download: Download,
    info: Info,
    op: Op,
    _library: Option<Library>,
}
impl Drop for Context {
    fn drop(&mut self) {
        let mut error = [0; ERROR];
        let status = unsafe { (self.free)(self.raw.as_ptr(), error.as_mut_ptr(), ERROR) };
        if status != 0 {
            // Pending GPU work must never outlive dlclose. On a CUDA drain
            // failure, preserve both the native context and its code until
            // process exit. This is a fatal ownership condition, not reusable
            // cache memory; normal work paths expose sync errors explicitly.
            if let Some(library) = self._library.take() {
                std::mem::forget(library);
            }
            eprintln!(
                "native vision context could not drain; native library retained until process exit"
            );
        }
    }
}
#[derive(Clone)]
pub(crate) struct Native(Rc<Context>);
pub(crate) struct Tensor {
    raw: NonNull<c_void>,
    owner: Native,
}
impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { (self.owner.0.tensor_free)(self.raw.as_ptr()) }
    }
}
fn check(status: i32, error: &[c_char; ERROR]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    anyhow::bail!("native vision error: {message}")
}
impl Native {
    /// The isolated shared object and its transitive native dependencies must
    /// have deployment pins; this gate verifies the bridge itself before dlopen.
    pub(crate) fn open(path: &Path, digest: &str) -> Result<Self> {
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase()),
            "vision library SHA256 required"
        );
        let mut file = std::fs::File::open(path)?;
        ensure!(file.metadata()?.len() <= 64 << 20, "vision bridge size");
        let mut hash = Sha256::new();
        let mut block = [0u8; 65536];
        loop {
            let count = file.read(&mut block)?;
            if count == 0 {
                break;
            }
            hash.update(&block[..count]);
        }
        ensure!(
            format!("{:x}", hash.finalize()) == digest,
            "vision bridge pin mismatch"
        );
        let library = unsafe { Library::new(path)? };
        let abi: unsafe extern "C" fn() -> *const c_char =
            unsafe { *library.get(b"alaska_vision_abi\0")? };
        let version = unsafe { CStr::from_ptr(abi()) };
        ensure!(
            version.to_bytes() == b"alaska-vision-primitives-v1",
            "vision ABI mismatch"
        );
        let new: New = unsafe { *library.get(b"alaska_vision_context_new\0")? };
        let free = unsafe { *library.get(b"alaska_vision_context_free\0")? };
        let tensor_free = unsafe { *library.get(b"alaska_vision_tensor_free\0")? };
        let sync = unsafe { *library.get(b"alaska_vision_sync\0")? };
        let empty = unsafe { *library.get(b"alaska_vision_empty\0")? };
        let upload = unsafe { *library.get(b"alaska_vision_upload\0")? };
        let download = unsafe { *library.get(b"alaska_vision_download\0")? };
        let info = unsafe { *library.get(b"alaska_vision_info\0")? };
        let op = unsafe { *library.get(b"alaska_vision_op\0")? };
        let mut raw = std::ptr::null_mut();
        let mut error = [0; ERROR];
        check(unsafe { new(&mut raw, error.as_mut_ptr(), ERROR) }, &error)?;
        Ok(Self(Rc::new(Context {
            raw: NonNull::new(raw).context("native vision context is null")?,
            free,
            tensor_free,
            sync,
            empty,
            upload,
            download,
            info,
            op,
            _library: Some(library),
        })))
    }
    fn tensor(&self, raw: Handle) -> Result<Tensor> {
        Ok(Tensor {
            raw: NonNull::new(raw).context("native vision tensor is null")?,
            owner: self.clone(),
        })
    }
    pub(crate) fn empty(&self, kind: Kind, device: i32, shape: &[i64]) -> Result<Tensor> {
        ensure!(
            (-1..=0).contains(&device) && !shape.is_empty() && shape.len() <= 8,
            "vision empty shape/device"
        );
        let elements =
            shape.iter().try_fold(
                1u64,
                |n, &d| if d > 0 { n.checked_mul(d as u64) } else { None },
            );
        ensure!(
            elements.is_some_and(|n| n <= 1 << 30),
            "vision tensor bound"
        );
        let mut raw = std::ptr::null_mut();
        let mut error = [0; ERROR];
        check(
            unsafe {
                (self.0.empty)(
                    self.0.raw.as_ptr(),
                    kind as i32,
                    device,
                    shape.as_ptr(),
                    shape.len(),
                    &mut raw,
                    error.as_mut_ptr(),
                    ERROR,
                )
            },
            &error,
        )?;
        self.tensor(raw)
    }
    pub(crate) fn op(
        &self,
        code: OpCode,
        inputs: &[&Tensor],
        dims: &[i64],
        scalars: &[f64],
    ) -> Result<Tensor> {
        ensure!(
            inputs.len() <= 16 && dims.len() <= 8 && scalars.len() <= 2,
            "vision operator bound"
        );
        ensure!(
            inputs.iter().all(|t| Rc::ptr_eq(&self.0, &t.owner.0)),
            "foreign vision tensor owner"
        );
        ensure!(
            scalars.iter().all(|v| v.is_finite()),
            "vision nonfinite scalar"
        );
        let mut pointers = [std::ptr::null_mut(); 16];
        for (p, t) in pointers.iter_mut().zip(inputs) {
            *p = t.raw.as_ptr();
        }
        let mut raw = std::ptr::null_mut();
        let mut error = [0; ERROR];
        check(
            unsafe {
                (self.0.op)(
                    self.0.raw.as_ptr(),
                    code as i32,
                    pointers.as_ptr(),
                    inputs.len(),
                    dims.as_ptr(),
                    dims.len(),
                    scalars.as_ptr(),
                    scalars.len(),
                    &mut raw,
                    error.as_mut_ptr(),
                    ERROR,
                )
            },
            &error,
        )?;
        self.tensor(raw)
    }
    pub(crate) fn sync(&self) -> Result<()> {
        let mut error = [0; ERROR];
        check(
            unsafe { (self.0.sync)(self.0.raw.as_ptr(), error.as_mut_ptr(), ERROR) },
            &error,
        )
    }
}
impl Tensor {
    pub(crate) fn upload(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= 8 << 20,
            "vision upload bound"
        );
        let mut error = [0; ERROR];
        check(
            unsafe {
                (self.owner.0.upload)(
                    self.owner.0.raw.as_ptr(),
                    self.raw.as_ptr(),
                    offset,
                    bytes.as_ptr(),
                    bytes.len(),
                    error.as_mut_ptr(),
                    ERROR,
                )
            },
            &error,
        )
    }
    pub(crate) fn info(&self) -> Result<(Kind, Vec<i64>, usize)> {
        let mut dims = [0; 8];
        let (mut rank, mut kind, mut bytes) = (0, 0, 0);
        let mut error = [0; ERROR];
        check(
            unsafe {
                (self.owner.0.info)(
                    self.raw.as_ptr(),
                    dims.as_mut_ptr(),
                    dims.len(),
                    &mut rank,
                    &mut kind,
                    &mut bytes,
                    error.as_mut_ptr(),
                    ERROR,
                )
            },
            &error,
        )?;
        ensure!(rank <= 8, "vision result rank");
        let kind = match kind {
            1 => Kind::Bf16,
            2 => Kind::F32,
            3 => Kind::I64,
            _ => anyhow::bail!("vision result dtype"),
        };
        Ok((kind, dims[..rank].to_vec(), usize::try_from(bytes)?))
    }
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        let (_, _, bytes) = self.info()?;
        ensure!(bytes <= 256 << 20, "diagnostic tensor download bound");
        let mut out = vec![0; bytes];
        let mut error = [0; ERROR];
        check(
            unsafe {
                (self.owner.0.download)(
                    self.owner.0.raw.as_ptr(),
                    self.raw.as_ptr(),
                    out.as_mut_ptr(),
                    out.len(),
                    error.as_mut_ptr(),
                    ERROR,
                )
            },
            &error,
        )?;
        Ok(out)
    }
}
