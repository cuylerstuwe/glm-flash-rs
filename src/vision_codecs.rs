//! Pinned native JPEG/WebP primitives with Rust policy, rows and ownership.
//! Loaded libraries and resolved dependency symbols must match their manifests.
use crate::vision_decode::{Decoded, MAX_ENCODED_BYTES, MAX_PIXELS, zeroed};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, c_char, c_void},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    ptr::NonNull,
};
type Handle = *mut c_void;
type Free = unsafe extern "C" fn(Handle);
type Version = unsafe extern "C" fn() -> u32;
type JpegOpen = unsafe extern "C" fn(
    *const u8,
    usize,
    *mut Handle,
    *mut u32,
    *mut u32,
    *mut u32,
    *mut c_char,
    usize,
) -> i32;
type JpegStage = unsafe extern "C" fn(Handle, *mut c_char, usize) -> i32;
type JpegRows =
    unsafe extern "C" fn(Handle, *mut u8, usize, u32, *mut u32, *mut c_char, usize) -> i32;
type WebpHeader =
    unsafe extern "C" fn(*const u8, usize, *mut u32, *mut u32, *mut u32, *mut c_char, usize) -> i32;
type WebpOpen = unsafe extern "C" fn(*const u8, usize, *mut Handle, *mut c_char, usize) -> i32;
type WebpFirst =
    unsafe extern "C" fn(Handle, *mut *const u8, *mut u32, *mut u32, *mut c_char, usize) -> i32;
type Symbols = unsafe extern "C" fn(*mut *const c_void, *mut *const c_void, *mut *const c_void);
const ERROR: usize = 512;

pub(crate) struct Codecs {
    jpeg_open: JpegOpen,
    jpeg_start: JpegStage,
    jpeg_rows: JpegRows,
    jpeg_finish: JpegStage,
    jpeg_free: Free,
    webp_header: WebpHeader,
    webp_open: WebpOpen,
    webp_first: WebpFirst,
    webp_free: Free,
    _library: Library,
}
impl Codecs {
    /// Private build manifest pins the shim and all native image dependencies.
    /// Actual linked symbol owners must resolve to those pinned paths too.
    pub(crate) fn load(manifest: &Value) -> Result<Self> {
        ensure!(manifest["abi"] == 1, "image codec manifest ABI");
        let module = pin(&manifest["module"])?;
        let dependencies = manifest["dependencies"]
            .as_object()
            .context("image codec dependencies")?;
        let mut expected = Vec::new();
        ensure!(dependencies.len() <= 8, "image codec dependency count");
        for name in ["jpeg", "webp", "demux"] {
            expected.push(pin(dependencies
                .get(name)
                .context("missing required image codec dependency")?)?);
        }
        // Transitive data/code dependencies can be pinned as well (e.g. the
        // frozen Pillow webp build depends on its separately bundled sharpyuv).
        for (name, value) in dependencies {
            if !["jpeg", "webp", "demux"].contains(&name.as_str()) {
                pin(value)?;
            }
        }
        // SAFETY: trusted private manifest pins each native image before load.
        let library = unsafe { Library::new(module) }.context("load image codec primitives")?;
        unsafe {
            let abi: Version = *library.get(b"avcodec_abi\0")?;
            ensure!(abi() == 1, "image codec primitive ABI");
            let symbols: Symbols = *library.get(b"avcodec_dependency_symbols\0")?;
            let mut owners = [std::ptr::null(); 3];
            symbols(&mut owners[0], &mut owners[1], &mut owners[2]);
            for (symbol, path) in owners.into_iter().zip(expected) {
                let mut info: libc::Dl_info = std::mem::zeroed();
                ensure!(
                    !symbol.is_null()
                        && libc::dladdr(symbol, &mut info) != 0
                        && !info.dli_fname.is_null(),
                    "image codec dependency owner"
                );
                let loaded =
                    PathBuf::from(CStr::from_ptr(info.dli_fname).to_str()?).canonicalize()?;
                ensure!(
                    loaded == path,
                    "image codec resolved dependency differs from pin"
                );
            }
            let jpeg_version: Version = *library.get(b"avcodec_jpeg_abi\0")?;
            let webp_version: Version = *library.get(b"avcodec_webp_version\0")?;
            ensure!(
                manifest["jpeg_abi"].as_u64() == Some(u64::from(jpeg_version()))
                    && manifest["webp_version"].as_u64() == Some(u64::from(webp_version())),
                "image codec native version pin"
            );
            Ok(Self {
                jpeg_open: *library.get(b"avjpeg_open\0")?,
                jpeg_start: *library.get(b"avjpeg_start\0")?,
                jpeg_rows: *library.get(b"avjpeg_rows\0")?,
                jpeg_finish: *library.get(b"avjpeg_finish\0")?,
                jpeg_free: *library.get(b"avjpeg_free\0")?,
                webp_header: *library.get(b"avwebp_header\0")?,
                webp_open: *library.get(b"avwebp_open\0")?,
                webp_first: *library.get(b"avwebp_first\0")?,
                webp_free: *library.get(b"avwebp_free\0")?,
                _library: library,
            })
        }
    }

    pub(crate) fn jpeg(
        &self,
        encoded: &[u8],
        workspace_bytes: usize,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Decoded> {
        check()?;
        ensure!(
            !encoded.is_empty() && encoded.len() <= MAX_ENCODED_BYTES,
            "JPEG encoded byte bound"
        );
        let (mut raw, mut width, mut height, mut channels) = (std::ptr::null_mut(), 0, 0, 0);
        call(|error, size| unsafe {
            (self.jpeg_open)(
                encoded.as_ptr(),
                encoded.len(),
                &mut raw,
                &mut width,
                &mut height,
                &mut channels,
                error,
                size,
            )
        })?;
        let handle = Owner::new(raw, self.jpeg_free)?;
        let (width, height) = (width as usize, height as usize);
        let pixels = area(width, height)?;
        ensure!([3, 4].contains(&channels), "JPEG output channels");
        let workspace_charge = pixels * 3 + width * 4 * 64 + MAX_ENCODED_BYTES + 65 * 1024 * 1024;
        ensure!(
            workspace_charge <= workspace_bytes,
            "JPEG workspace admission bound"
        );
        check()?;
        call(|error, size| unsafe { (self.jpeg_start)(handle.raw(), error, size) })?;
        let mut rgb = zeroed(pixels * 3)?;
        let mut cmyk = zeroed(if channels == 4 { width * 4 * 64 } else { 0 })?;
        let mut position = 0;
        while position < height {
            check()?;
            let rows = (height - position).min(64);
            let mut read = 0;
            let target = if channels == 3 {
                &mut rgb[position * width * 3..(position + rows) * width * 3]
            } else {
                &mut cmyk[..rows * width * 4]
            };
            call(|error, size| unsafe {
                (self.jpeg_rows)(
                    handle.raw(),
                    target.as_mut_ptr(),
                    target.len(),
                    rows as u32,
                    &mut read,
                    error,
                    size,
                )
            })?;
            ensure!(read as usize == rows, "JPEG row count");
            if channels == 4 {
                for (dst, src) in rgb[position * width * 3..(position + rows) * width * 3]
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(cmyk[..rows * width * 4].as_chunks::<4>().0)
                {
                    *dst = cmyk_rgb(*src);
                }
            }
            position += rows;
        }
        call(|error, size| unsafe { (self.jpeg_finish)(handle.raw(), error, size) })?;
        check()?;
        Ok(Decoded {
            width,
            height,
            rgb,
            workspace_charge,
        })
    }

    pub(crate) fn webp(
        &self,
        encoded: &[u8],
        workspace_bytes: usize,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Decoded> {
        check()?;
        ensure!(
            !encoded.is_empty() && encoded.len() <= MAX_ENCODED_BYTES,
            "WebP encoded byte bound"
        );
        let (mut width, mut height, mut alpha) = (0, 0, 0);
        call(|error, size| unsafe {
            (self.webp_header)(
                encoded.as_ptr(),
                encoded.len(),
                &mut width,
                &mut height,
                &mut alpha,
                error,
                size,
            )
        })?;
        let pixels = area(width as usize, height as usize)?;
        // Native animation ownership retains current/previous RGBA canvases;
        // additionally charge codec/frame workspace and the Rust RGB output.
        let workspace_charge = pixels * 24 + MAX_ENCODED_BYTES + 4 * 1024 * 1024;
        ensure!(
            workspace_charge <= workspace_bytes,
            "WebP workspace admission bound"
        );
        check()?;
        let mut raw = std::ptr::null_mut();
        call(|error, size| unsafe {
            (self.webp_open)(encoded.as_ptr(), encoded.len(), &mut raw, error, size)
        })?;
        let handle = Owner::new(raw, self.webp_free)?;
        let (mut data, mut canvas_w, mut canvas_h) = (std::ptr::null(), 0, 0);
        call(|error, size| unsafe {
            (self.webp_first)(
                handle.raw(),
                &mut data,
                &mut canvas_w,
                &mut canvas_h,
                error,
                size,
            )
        })?;
        ensure!(
            !data.is_null() && (canvas_w, canvas_h) == (width, height),
            "WebP decoded canvas identity"
        );
        check()?;
        // Native decoder and the immutable encoded slice stay alive until
        // after all output rows are copied; no borrowed pixels escape.
        let rgba = unsafe { std::slice::from_raw_parts(data, pixels * 4) };
        let mut rgb = zeroed(pixels * 3)?;
        for (out, source) in rgb
            .chunks_exact_mut(width as usize * 3)
            .zip(rgba.chunks_exact(width as usize * 4))
        {
            check()?;
            for (dst, src) in out
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(source.as_chunks::<4>().0)
            {
                *dst = if alpha == 0 {
                    [src[0], src[1], src[2]]
                } else {
                    [
                        white(src[0], src[3]),
                        white(src[1], src[3]),
                        white(src[2], src[3]),
                    ]
                };
            }
        }
        Ok(Decoded {
            width: width as usize,
            height: height as usize,
            rgb,
            workspace_charge,
        })
    }
}

fn pin(value: &Value) -> Result<PathBuf> {
    let path = Path::new(value["path"].as_str().context("image codec path")?);
    ensure!(path.is_absolute(), "image codec path must be absolute");
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= 16 * 1024 * 1024,
        "image codec library size bound"
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
        value["sha256"].as_str() == Some(format!("{:x}", hash.finalize()).as_str()),
        "image codec library hash"
    );
    Ok(path.canonicalize()?)
}
fn area(width: usize, height: usize) -> Result<usize> {
    let pixels = width
        .checked_mul(height)
        .context("image dimensions overflow")?;
    ensure!(
        width > 0 && height > 0 && pixels <= MAX_PIXELS,
        "decoded image pixel bound"
    );
    Ok(pixels)
}
struct Owner {
    raw: NonNull<c_void>,
    free: Free,
}
impl Owner {
    fn new(raw: Handle, free: Free) -> Result<Self> {
        Ok(Self {
            raw: NonNull::new(raw).context("image codec null owner")?,
            free,
        })
    }
    fn raw(&self) -> Handle {
        self.raw.as_ptr()
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.raw.as_ptr());
        }
    }
}
fn call(f: impl FnOnce(*mut c_char, usize) -> i32) -> Result<()> {
    let mut error = [0u8; ERROR];
    let ok = f(error.as_mut_ptr().cast(), ERROR);
    let len = error.iter().position(|v| *v == 0).unwrap_or(ERROR);
    ensure!(
        ok == 1,
        "native image codec: {}",
        String::from_utf8_lossy(&error[..len])
    );
    Ok(())
}
fn white(value: u8, alpha: u8) -> u8 {
    ((u32::from(value) * u32::from(alpha) + 255 * (255 - u32::from(alpha)) + 127) / 255) as u8
}
fn cmyk_rgb(value: [u8; 4]) -> [u8; 3] {
    // Frozen Pillow uses CMYK;I for JPEG then Convert.c's CMYK-to-RGB.
    // The two operations reduce to rounded native component * native K / 255.
    let k = u32::from(value[3]);
    [0, 1, 2].map(|c| ((u32::from(value[c]) * k + 127) / 255) as u8)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cmyk_inversion_matches_frozen_conversion_exhaustively() {
        for c in 0..=255u32 {
            for k in 0..=255u32 {
                let temp = (255 - c) * k + 128;
                let expected = k - (((temp >> 8) + temp) >> 8);
                assert_eq!(
                    cmyk_rgb([c as u8, c as u8, c as u8, k as u8]),
                    [expected as u8; 3]
                );
            }
        }
    }
    #[test]
    fn reject_missing_policy_and_oversized_dimensions_before_native_use() {
        assert!(Codecs::load(&serde_json::json!({})).is_err());
        assert!(area(0, 1).is_err());
        assert!(area(usize::MAX, 2).is_err());
        assert!(area(65536, 65536).is_err());
    }
}
