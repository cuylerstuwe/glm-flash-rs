//! Pinned XGrammar C++ boundary for grammar compilation and matcher ownership.
//! Rust owns the library and opaque handles; C++ owns tokenizer/compiler/matcher
//! internals. No Python, PyO3, TVM-FFI or interpreter initialization is involved.
use anyhow::{Context, Result, ensure};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_int, c_void},
    io::Read,
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    sync::Arc,
};
const REVISION: &str = "97787376faee5ed8466cfad57c99855e4ce2f6aa";
const ERROR_BYTES: usize = 2048;
type Handle = *mut c_void;
type Error = *mut c_char;
#[repr(C)]
#[derive(Clone, Copy)]
struct Bytes {
    data: *const u8,
    length: usize,
}
impl From<&[u8]> for Bytes {
    fn from(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            length: bytes.len(),
        }
    }
}
type CompilerNew = unsafe extern "C" fn(
    *const Bytes,
    usize,
    c_int,
    c_int,
    *const i32,
    usize,
    c_int,
    u64,
    *mut Handle,
    Error,
    usize,
) -> c_int;
type Free = unsafe extern "C" fn(Handle);
type Compile = unsafe extern "C" fn(Handle, c_int, Bytes, *mut Handle, Error, usize) -> c_int;
type Count = unsafe extern "C" fn(Handle, *mut u64, Error, usize) -> c_int;
type New = unsafe extern "C" fn(Handle, *mut Handle, Error, usize) -> c_int;
type Accept = unsafe extern "C" fn(Handle, i32, *mut c_int, Error, usize) -> c_int;
type Fill = unsafe extern "C" fn(Handle, *mut i32, usize, *mut c_int, Error, usize) -> c_int;
type Rollback = unsafe extern "C" fn(Handle, usize, Error, usize) -> c_int;
type Bool = unsafe extern "C" fn(Handle, *mut c_int, Error, usize) -> c_int;
struct Api {
    // Must stay loaded until every function pointer and native handle is gone.
    _library: Library,
    compiler_new: CompilerNew,
    compiler_free: Free,
    compile: Compile,
    grammar_free: Free,
    grammar_bytes: Count,
    compiler_cache_bytes: Count,
    matcher_new: New,
    matcher_free: Free,
    matcher_accept: Accept,
    matcher_fill: Fill,
    matcher_rollback: Rollback,
    matcher_terminated: Bool,
    #[allow(dead_code)] // Native ownership proof only; serving retains one matcher.
    matcher_fork: New,
}
fn checked(status: c_int, error: &[c_char; ERROR_BYTES]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // Native code always terminates within the fixed buffer, and zeroed initial
    // storage remains a terminator even if a malformed bridge returns early.
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    anyhow::bail!("native grammar error: {message}")
}
fn handle(pointer: Handle) -> Result<NonNull<c_void>> {
    NonNull::new(pointer).context("native grammar returned an empty handle")
}
fn boolean(value: c_int) -> Result<bool> {
    ensure!(
        (0..=1).contains(&value),
        "native grammar boolean out of range"
    );
    Ok(value != 0)
}
#[derive(Clone)]
pub(crate) struct Native(Arc<Api>);
impl Native {
    /// Loading native code requires an explicit content pin. The source/ABI
    /// checks complement this hash; a source revision string alone is no pin.
    pub(crate) fn open(path: &Path, expected_sha256: &str) -> Result<Self> {
        ensure!(
            expected_sha256.len() == 64
                && expected_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "native grammar SHA256 required"
        );
        let mut file = std::fs::File::open(path)?;
        ensure!(
            file.metadata()?.len() <= 128 << 20,
            "native grammar library byte bound"
        );
        let mut hash = Sha256::new();
        let mut buffer = [0; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        ensure!(
            format!("{:x}", hash.finalize()) == expected_sha256,
            "native grammar library identity mismatch"
        );
        // The caller must use an immutable, trusted candidate path: hash then
        // dlopen cannot protect against a concurrent pathname replacement.
        let library = unsafe { Library::new(path)? };
        unsafe {
            let version: unsafe extern "C" fn() -> u32 = *library.get(b"axg_abi_version\0")?;
            let revision: unsafe extern "C" fn() -> *const c_char =
                *library.get(b"axg_source_revision\0")?;
            ensure!(version() == 1, "native grammar ABI mismatch");
            let revision = revision();
            ensure!(
                !revision.is_null() && CStr::from_ptr(revision).to_bytes() == REVISION.as_bytes(),
                "native grammar source mismatch"
            );
            let api = Api {
                compiler_new: *library.get(b"axg_compiler_new\0")?,
                compiler_free: *library.get(b"axg_compiler_free\0")?,
                compile: *library.get(b"axg_compile\0")?,
                grammar_free: *library.get(b"axg_grammar_free\0")?,
                grammar_bytes: *library.get(b"axg_grammar_bytes\0")?,
                compiler_cache_bytes: *library.get(b"axg_compiler_cache_bytes\0")?,
                matcher_new: *library.get(b"axg_matcher_new\0")?,
                matcher_free: *library.get(b"axg_matcher_free\0")?,
                matcher_accept: *library.get(b"axg_matcher_accept\0")?,
                matcher_fill: *library.get(b"axg_matcher_fill\0")?,
                matcher_rollback: *library.get(b"axg_matcher_rollback\0")?,
                matcher_terminated: *library.get(b"axg_matcher_terminated\0")?,
                matcher_fork: *library.get(b"axg_matcher_fork\0")?,
                _library: library,
            };
            Ok(Self(Arc::new(api)))
        }
    }
    /// Entries include empty added/padded IDs; the caller builds the two tokenizer
    /// text/tools views and verifies stop IDs and ByteLevel metadata separately.
    pub(crate) fn compiler(
        &self,
        vocab: &[Vec<u8>],
        byte_level: bool,
        add_prefix_space: bool,
        stops: &[i32],
    ) -> Result<Compiler> {
        ensure!(
            !vocab.is_empty() && vocab.len() <= 154880,
            "grammar vocabulary bound"
        );
        let bytes: Vec<Bytes> = vocab.iter().map(|v| v.as_slice().into()).collect();
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.0.compiler_new)(
                bytes.as_ptr(),
                bytes.len(),
                if byte_level { 2 } else { 0 },
                i32::from(add_prefix_space),
                stops.as_ptr(),
                stops.len(),
                8,
                128 << 20,
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(Compiler {
            api: self.0.clone(),
            pointer: handle(pointer)?,
            vocab: vocab.len(),
            _unique: PhantomData,
        })
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    JsonSchema = 0,
    Regex = 1,
    Ebnf = 2,
    StructuralTag = 3,
}
pub(crate) struct Compiler {
    api: Arc<Api>,
    pointer: NonNull<c_void>,
    vocab: usize,
    _unique: PhantomData<Cell<()>>,
}
impl Compiler {
    pub(crate) fn compile(&mut self, kind: Kind, source: &str) -> Result<Compiled> {
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.compile)(
                self.pointer.as_ptr(),
                kind as i32,
                source.as_bytes().into(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(Compiled {
            api: self.api.clone(),
            pointer: handle(pointer)?,
            vocab: self.vocab,
            _unique: PhantomData,
        })
    }
    pub(crate) fn cache_bytes(&self) -> Result<u64> {
        let mut count = 0;
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.compiler_cache_bytes)(
                self.pointer.as_ptr(),
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(count)
    }
}
impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe { (self.api.compiler_free)(self.pointer.as_ptr()) };
    }
}
pub(crate) struct Compiled {
    api: Arc<Api>,
    pointer: NonNull<c_void>,
    vocab: usize,
    _unique: PhantomData<Cell<()>>,
}
impl Compiled {
    pub(crate) fn bytes(&self) -> Result<u64> {
        let mut count = 0;
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.grammar_bytes)(
                self.pointer.as_ptr(),
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(count)
    }
    pub(crate) fn matcher(&self) -> Result<Matcher> {
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_new)(
                self.pointer.as_ptr(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(Matcher {
            api: self.api.clone(),
            pointer: handle(pointer)?,
            vocab: self.vocab,
            _unique: PhantomData,
        })
    }
}
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { (self.api.grammar_free)(self.pointer.as_ptr()) };
    }
}
pub(crate) struct Matcher {
    api: Arc<Api>,
    pointer: NonNull<c_void>,
    vocab: usize,
    _unique: PhantomData<Cell<()>>,
}
// A matcher and its mutable state have exactly one Rust owner. It may be moved
// to a mask-fill worker and returned, but is !Sync and cannot be shared across
// threads. The immutable compiled grammar uses native shared_ptr ownership.
unsafe impl Send for Matcher {}
impl Matcher {
    pub(crate) fn mask_words(&self) -> usize {
        self.vocab.div_ceil(32)
    }
    pub(crate) fn accept(&mut self, token: i32) -> Result<bool> {
        ensure!(
            token >= 0 && (token as usize) < self.vocab,
            "grammar token bound"
        );
        let mut accepted = 0;
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_accept)(
                self.pointer.as_ptr(),
                token,
                &mut accepted,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        boolean(accepted)
    }
    pub(crate) fn fill(&mut self, words: &mut [i32]) -> Result<bool> {
        ensure!(words.len() == self.mask_words(), "grammar mask word count");
        let mut needs = 0;
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_fill)(
                self.pointer.as_ptr(),
                words.as_mut_ptr(),
                words.len(),
                &mut needs,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        boolean(needs)
    }
    pub(crate) fn rollback(&mut self, count: usize) -> Result<()> {
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_rollback)(
                self.pointer.as_ptr(),
                count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)
    }
    pub(crate) fn terminated(&self) -> Result<bool> {
        let mut terminated = 0;
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_terminated)(
                self.pointer.as_ptr(),
                &mut terminated,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        boolean(terminated)
    }
    #[allow(dead_code)] // Native ownership proof only; serving retains one matcher.
    pub(crate) fn fork(&self) -> Result<Self> {
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        let status = unsafe {
            (self.api.matcher_fork)(
                self.pointer.as_ptr(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        checked(status, &error)?;
        Ok(Self {
            api: self.api.clone(),
            pointer: handle(pointer)?,
            vocab: self.vocab,
            _unique: PhantomData,
        })
    }
}
impl Drop for Matcher {
    fn drop(&mut self) {
        unsafe { (self.api.matcher_free)(self.pointer.as_ptr()) };
    }
}
