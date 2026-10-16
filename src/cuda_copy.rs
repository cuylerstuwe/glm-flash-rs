//! Narrow, movable CUDA copy owner. It never owns or mutates a source buffer.
//! Allocate on the model owner at startup; reuse across file jobs. No context-wide
//! synchronization/allocation during a worker's copies or graph capture.
use super::*;

pub struct CopyStream {
    ordinal: c_int,
    ctx: Handle,
    stream: Handle,
    host: *mut u8,
    bytes: usize,
}
// Exclusive move only: every method activates the retained primary context on
// the calling thread. No Rc, borrowed allocation or concurrent access is moved.
unsafe impl Send for CopyStream {}
impl CopyStream {
    pub fn new(device: &Rc<Device>, bytes: usize) -> Result<Self> {
        let mut owner = Self {
            ordinal: device.ordinal,
            ctx: ptr::null_mut(),
            stream: ptr::null_mut(),
            host: ptr::null_mut(),
            bytes,
        };
        unsafe {
            check(cuDevicePrimaryCtxRetain(&mut owner.ctx, owner.ordinal))?;
            check(cuCtxSetCurrent(owner.ctx))?;
            check(cuStreamCreate(&mut owner.stream, 1))?;
            check(cuMemAllocHost_v2(
                (&mut owner.host as *mut *mut u8).cast(),
                bytes,
            ))?;
        }
        Ok(owner)
    }
    /// # Safety
    /// The caller holds an immutable snapshot pin and an admitted copy permit.
    /// Every address/length must lie in that snapshot. Neither may be released
    /// before this batch's stream has drained, including error/unwind paths.
    pub unsafe fn batch(&mut self, copies: &[(usize, u64, usize)]) -> Result<()> {
        for &(dst, src, len) in copies {
            ensure!(
                dst <= self.bytes
                    && len <= self.bytes - dst
                    && src.checked_add(len as u64).is_some(),
                "SSD DMA range"
            );
        }
        unsafe {
            check(cuCtxSetCurrent(self.ctx))?;
        }
        let result = (|| {
            for &(dst, src, len) in copies {
                unsafe {
                    check(cuMemcpyDtoHAsync_v2(
                        self.host.add(dst).cast(),
                        src,
                        len,
                        self.stream,
                    ))?;
                }
            }
            Ok(())
        })();
        // Always drain already submitted copies, even if a later launch failed.
        let drained = unsafe { check(cuStreamSynchronize(self.stream)) };
        result.and(drained)
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host, self.bytes) }
    }
}
impl Drop for CopyStream {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                let _ = cuCtxSetCurrent(self.ctx);
                if !self.stream.is_null() {
                    let _ = cuStreamSynchronize(self.stream);
                    let _ = cuStreamDestroy_v2(self.stream);
                }
                if !self.host.is_null() {
                    let _ = cuMemFreeHost(self.host.cast());
                }
                let _ = cuDevicePrimaryCtxRelease_v2(self.ordinal);
            }
        }
    }
}
