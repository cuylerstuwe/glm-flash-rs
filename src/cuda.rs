//! Small, single-threaded CUDA Driver owner. No Torch, Python, or CUDA runtime dependency.
use anyhow::{Result, bail, ensure};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    ptr,
    rc::Rc,
};
#[path = "cuda_copy.rs"]
mod copy;
pub use copy::CopyStream;
type Handle = *mut c_void;
// CUDA 13 cuda.h: CUlaunchAttributeValue is an eight-byte-aligned 64-byte
// union. This owner only sets programmaticStreamSerializationAllowed (int).
#[repr(C)]
struct LaunchAttribute {
    id: c_int,
    pad: c_int,
    value: [u64; 8],
}
#[repr(C)]
struct LaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    stream: Handle,
    attrs: *mut LaunchAttribute,
    count: u32,
}
const _: () = assert!(size_of::<LaunchAttribute>() == 72);
const _: () = assert!(size_of::<LaunchConfig>() == 56);
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRetain(ctx: *mut Handle, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRelease_v2(device: c_int) -> c_int;
    fn cuCtxSetCurrent(ctx: Handle) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
    fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
    fn cuMemFree_v2(ptr: u64) -> c_int;
    fn cuMemsetD8_v2(ptr: u64, value: u8, bytes: usize) -> c_int;
    fn cuMemsetD8Async(ptr: u64, value: u8, bytes: usize, stream: Handle) -> c_int;
    fn cuMemsetD32Async(ptr: u64, value: u32, count: usize, stream: Handle) -> c_int;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: Handle) -> c_int;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: Handle) -> c_int;
    fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: Handle) -> c_int;
    fn cuMemAllocHost_v2(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cuMemHostGetDevicePointer_v2(dptr: *mut u64, ptr: *mut c_void, flags: u32) -> c_int;
    fn cuMemFreeHost(ptr: *mut c_void) -> c_int;
    fn cuStreamCreate(stream: *mut Handle, flags: c_uint) -> c_int;
    fn cuStreamDestroy_v2(stream: Handle) -> c_int;
    fn cuStreamSynchronize(stream: Handle) -> c_int;
    fn cuStreamWaitEvent(stream: Handle, event: Handle, flags: c_uint) -> c_int;
    fn cuEventCreate(event: *mut Handle, flags: c_uint) -> c_int;
    fn cuEventRecord(event: Handle, stream: Handle) -> c_int;
    fn cuEventSynchronize(event: Handle) -> c_int;
    fn cuEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> c_int;
    fn cuEventDestroy_v2(event: Handle) -> c_int;
    fn cuModuleLoad(module: *mut Handle, filename: *const c_char) -> c_int;
    fn cuModuleUnload(module: Handle) -> c_int;
    fn cuModuleGetFunction(func: *mut Handle, module: Handle, name: *const c_char) -> c_int;
    fn cuFuncSetAttribute(func: Handle, attr: c_int, value: c_int) -> c_int;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut c_int,
        func: Handle,
        block_size: c_int,
        dynamic_smem: usize,
    ) -> c_int;
    fn cuLaunchKernel(
        func: Handle,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared: u32,
        stream: Handle,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    fn cuLaunchKernelEx(
        config: *const LaunchConfig,
        func: Handle,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    fn cuGetErrorName(code: c_int, name: *mut *const c_char) -> c_int;
    fn cuGetErrorString(code: c_int, name: *mut *const c_char) -> c_int;
    fn cuStreamBeginCapture_v2(stream: Handle, mode: c_int) -> c_int;
    fn cuStreamEndCapture(stream: Handle, graph: *mut Handle) -> c_int;
    fn cuGraphInstantiateWithFlags(exec: *mut Handle, graph: Handle, flags: u64) -> c_int;
    fn cuGraphGetNodes(graph: Handle, nodes: *mut Handle, count: *mut usize) -> c_int;
    fn cuGraphLaunch(exec: Handle, stream: Handle) -> c_int;
    fn cuGraphDestroy(graph: Handle) -> c_int;
    fn cuGraphExecDestroy(exec: Handle) -> c_int;
}
fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    unsafe {
        let mut name = ptr::null();
        let mut msg = ptr::null();
        cuGetErrorName(code, &mut name);
        cuGetErrorString(code, &mut msg);
        let name = if name.is_null() {
            "unknown".into()
        } else {
            CStr::from_ptr(name).to_string_lossy()
        };
        let msg = if msg.is_null() {
            "unknown".into()
        } else {
            CStr::from_ptr(msg).to_string_lossy()
        };
        bail!("CUDA {code} {name}: {msg}");
    }
}
pub struct Device {
    ordinal: c_int,
    ctx: Handle,
    pub name: String,
    pub arch: i32,
    integrated: bool,
    copy_gate: std::cell::RefCell<Option<std::sync::Weak<crate::copy_gate::Gate>>>,
}
impl Device {
    pub fn new() -> Result<Rc<Self>> {
        unsafe {
            check(cuInit(0))?;
            let mut ordinal = 0;
            check(cuDeviceGet(&mut ordinal, 0))?;
            let mut name = [0 as c_char; 128];
            check(cuDeviceGetName(name.as_mut_ptr(), 128, ordinal))?;
            let mut major = 0;
            let mut minor = 0;
            check(cuDeviceGetAttribute(&mut major, 75, ordinal))?;
            check(cuDeviceGetAttribute(&mut minor, 76, ordinal))?;
            let mut integrated = 0;
            check(cuDeviceGetAttribute(&mut integrated, 18, ordinal))?;
            let mut ctx = ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut ctx, ordinal))?;
            if let Err(e) = check(cuCtxSetCurrent(ctx)) {
                cuDevicePrimaryCtxRelease_v2(ordinal);
                return Err(e);
            }
            Ok(Rc::new(Self {
                ordinal,
                ctx,
                name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                arch: major * 10 + minor,
                integrated: integrated != 0,
                copy_gate: std::cell::RefCell::new(None),
            }))
        }
    }
    fn activate(&self) -> Result<()> {
        unsafe { check(cuCtxSetCurrent(self.ctx)) }
    }
    pub fn register_copy_gate(&self, gate: &std::sync::Arc<crate::copy_gate::Gate>) -> Result<()> {
        let mut current = self.copy_gate.borrow_mut();
        ensure!(
            current
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .is_none(),
            "device already owns a copy worker"
        );
        *current = Some(std::sync::Arc::downgrade(gate));
        Ok(())
    }
    pub fn sync(&self) -> Result<()> {
        self.activate()?;
        unsafe { check(cuCtxSynchronize()) }
    }
    pub fn memory(&self) -> Result<(usize, usize)> {
        self.activate()?;
        let mut free = 0;
        let mut total = 0;
        unsafe {
            check(cuMemGetInfo_v2(&mut free, &mut total))?;
        }
        Ok((free, total))
    }
    pub fn memory_view(&self) -> Result<crate::memory_budget::View> {
        let (free, total) = self.memory()?;
        Ok(crate::memory_budget::read(free, total, self.integrated))
    }
    pub fn allocatable_memory(&self) -> Result<(usize, usize)> {
        let view = self.memory_view()?;
        Ok((view.usable as usize, view.cuda_total as usize))
    }
    pub fn alloc(self: &Rc<Self>, bytes: usize) -> Result<Buffer> {
        ensure!(bytes > 0, "zero-sized CUDA storage");
        self.activate()?;
        let mut ptr = 0;
        unsafe {
            check(cuMemAlloc_v2(&mut ptr, bytes))?;
        }
        Ok(Buffer {
            device: self.clone(),
            ptr,
            bytes,
        })
    }
    pub fn fence(self: &Rc<Self>) -> Result<Fence> {
        self.activate()?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(cuEventCreate(&mut raw, 2))?;
        }
        Ok(Fence {
            device: self.clone(),
            raw,
            recorded: false,
        })
    }
    pub fn event(self: &Rc<Self>) -> Result<Event> {
        self.activate()?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(cuEventCreate(&mut raw, 0))?;
        }
        Ok(Event {
            device: self.clone(),
            raw,
            recorded: false,
        })
    }
    pub fn stream(self: &Rc<Self>) -> Result<Stream> {
        self.activate()?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(cuStreamCreate(&mut raw, 1))?;
        }
        Ok(Stream {
            device: self.clone(),
            raw,
        })
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            cuCtxSetCurrent(self.ctx);
            cuCtxSynchronize();
            cuDevicePrimaryCtxRelease_v2(self.ordinal);
        }
    }
}
/// Explicit opt-in GPU timing; unused by normal serving and decode.
pub struct Event {
    device: Rc<Device>,
    raw: Handle,
    recorded: bool,
}
impl Event {
    pub fn record(&mut self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "event context mismatch"
        );
        self.device.activate()?;
        unsafe {
            check(cuEventRecord(self.raw, stream.raw))?;
        }
        self.recorded = true;
        Ok(())
    }
    pub fn elapsed_after(&self, start: &Self) -> Result<f32> {
        ensure!(
            self.recorded && start.recorded && Rc::ptr_eq(&self.device, &start.device),
            "unrecorded or foreign timing event"
        );
        self.device.activate()?;
        let mut ms = 0.;
        unsafe {
            check(cuEventSynchronize(self.raw))?;
            check(cuEventElapsedTime(&mut ms, start.raw, self.raw))?;
        }
        ensure!(ms.is_finite() && ms >= 0., "invalid CUDA event duration");
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        let _ = self.device.activate();
        unsafe {
            cuEventDestroy_v2(self.raw);
        }
    }
}
/// Stream ordering without timing, host synchronization, or CPU polling.
pub struct Fence {
    device: Rc<Device>,
    raw: Handle,
    recorded: bool,
}
impl Fence {
    pub fn record(&mut self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "fence context mismatch"
        );
        self.device.activate()?;
        unsafe {
            check(cuEventRecord(self.raw, stream.raw))?;
        }
        self.recorded = true;
        Ok(())
    }
    pub fn wait_on(&self, stream: &Stream) -> Result<()> {
        ensure!(
            self.recorded && Rc::ptr_eq(&self.device, &stream.device),
            "unrecorded/foreign fence"
        );
        self.device.activate()?;
        unsafe { check(cuStreamWaitEvent(stream.raw, self.raw, 0)) }
    }
}
impl Drop for Fence {
    fn drop(&mut self) {
        let _ = self.device.activate();
        unsafe {
            cuEventDestroy_v2(self.raw);
        }
    }
}
pub struct Buffer {
    device: Rc<Device>,
    pub ptr: u64,
    pub bytes: usize,
}
impl Buffer {
    /// Storage must outlive stream completion or the captured graph.
    pub unsafe fn zero_prefix(&self, stream: &Stream, bytes: usize) -> Result<()> {
        unsafe { self.fill_prefix(stream, bytes, 0) }
    }
    /// Storage must outlive stream completion or the captured graph.
    pub unsafe fn fill_prefix(&self, stream: &Stream, bytes: usize, value: u8) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "memset context mismatch"
        );
        let ptr = self.range(0, bytes)?;
        self.device.activate()?;
        unsafe { check(cuMemsetD8Async(ptr, value, bytes, stream.raw)) }
    }
    /// Stream-ordered scalar update. Storage must outlive completion/capture.
    pub unsafe fn set_i32(&self, stream: &Stream, value: i32) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "scalar context mismatch"
        );
        let ptr = self.range(0, 4)?;
        self.device.activate()?;
        unsafe { check(cuMemsetD32Async(ptr, value as u32, 1, stream.raw)) }
    }
    /// Both buffers must outlive stream completion (or the captured graph).
    pub unsafe fn copy_from(
        &self,
        stream: &Stream,
        dst: usize,
        source: &Buffer,
        src: usize,
        bytes: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &source.device) && Rc::ptr_eq(&self.device, &stream.device),
            "copy context mismatch"
        );
        let dst = self.range(dst, bytes)?;
        let src = source.range(src, bytes)?;
        self.device.activate()?;
        unsafe { check(cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.raw)) }
    }
    fn range(&self, off: usize, bytes: usize) -> Result<u64> {
        ensure!(
            off <= self.bytes && bytes <= self.bytes - off,
            "GPU storage range {off}+{bytes} exceeds {}",
            self.bytes
        );
        Ok(self.ptr + off as u64)
    }
    pub fn zero(&self) -> Result<()> {
        self.device.activate()?;
        unsafe {
            check(cuMemsetD8_v2(self.ptr, 0, self.bytes))?;
            check(cuStreamSynchronize(ptr::null_mut()))
        }
    }
    pub fn write(&self, off: usize, data: &[u8]) -> Result<()> {
        let dst = self.range(off, data.len())?;
        self.device.activate()?;
        // Pageable HtoD may return after staging, before its default-stream
        // DMA finishes. This blocking convenience API must be complete before
        // callers consume the buffer on a nonblocking stream. Hot paths use
        // persistent pinned, stream-ordered uploads instead.
        unsafe {
            check(cuMemcpyHtoD_v2(dst, data.as_ptr().cast(), data.len()))?;
            check(cuStreamSynchronize(ptr::null_mut()))
        }
    }
    pub fn read(&self, off: usize, bytes: usize) -> Result<Vec<u8>> {
        let mut data = vec![0u8; bytes];
        self.read_into(off, &mut data)?;
        Ok(data)
    }
    pub fn read_into(&self, off: usize, data: &mut [u8]) -> Result<()> {
        let src = self.range(off, data.len())?;
        self.device.sync()?;
        unsafe {
            check(cuMemcpyDtoH_v2(data.as_mut_ptr().cast(), src, data.len()))?;
        }
        Ok(())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.device.activate();
        unsafe {
            cuMemFree_v2(self.ptr);
        }
    }
}
pub struct Pinned {
    device: Rc<Device>,
    ptr: *mut u8,
    bytes: usize,
}
impl Pinned {
    pub fn new(device: &Rc<Device>, bytes: usize) -> Result<Self> {
        device.activate()?;
        let mut ptr = ptr::null_mut();
        unsafe {
            check(cuMemAllocHost_v2(&mut ptr, bytes))?;
        }
        Ok(Self {
            device: device.clone(),
            ptr: ptr.cast(),
            bytes,
        })
    }
    /// Device address of this page-locked allocation, which kernels may read
    /// and write directly. Valid while `self` lives.
    pub fn device_ptr(&self) -> Result<u64> {
        self.device.activate()?;
        let mut address = 0;
        unsafe {
            check(cuMemHostGetDevicePointer_v2(
                &mut address,
                self.ptr.cast(),
                0,
            ))?;
        }
        Ok(address)
    }
    /// The pinned source and target must outlive completion; do not mutate the
    /// uploaded source range until that stream has completed the copy.
    pub unsafe fn upload_to(
        &self,
        stream: &Stream,
        target: &Buffer,
        dst: usize,
        src: usize,
        bytes: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &target.device) && Rc::ptr_eq(&self.device, &stream.device),
            "upload context mismatch"
        );
        ensure!(
            src <= self.bytes && bytes <= self.bytes - src,
            "pinned upload bounds"
        );
        let dst = target.range(dst, bytes)?;
        self.device.activate()?;
        unsafe {
            check(cuMemcpyHtoDAsync_v2(
                dst,
                self.ptr.add(src).cast(),
                bytes,
                stream.raw,
            ))
        }
    }
    /// Source and destination must outlive completion; neither CPU nor GPU may
    /// access the destination range until this owner's stream has completed.
    pub unsafe fn download_from(
        &mut self,
        stream: &Stream,
        dst: usize,
        source: &Buffer,
        src: usize,
        bytes: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &source.device) && Rc::ptr_eq(&self.device, &stream.device),
            "download context mismatch"
        );
        ensure!(
            dst <= self.bytes && bytes <= self.bytes - dst,
            "pinned download bounds"
        );
        let src = source.range(src, bytes)?;
        self.device.activate()?;
        unsafe {
            check(cuMemcpyDtoHAsync_v2(
                self.ptr.add(dst).cast(),
                src,
                bytes,
                stream.raw,
            ))
        }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}
impl Drop for Pinned {
    fn drop(&mut self) {
        let _ = self.device.activate();
        unsafe {
            cuMemFreeHost(self.ptr.cast());
        }
    }
}
pub struct Stream {
    device: Rc<Device>,
    raw: Handle,
}
impl Stream {
    pub(crate) fn raw(&self) -> Handle {
        self.raw
    }
    pub fn sync(&self) -> Result<()> {
        self.device.activate()?;
        unsafe { check(cuStreamSynchronize(self.raw)) }
    }
    // The caller owns all buffers/modules referenced by captured launches for Graph's entire lifetime.
    pub unsafe fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
        // No cost on replay. Global capture and a helper's CUDA API calls can
        // conflict even on distinct streams; drain/park immutable source copies
        // through capture/instantiation, then restore their previous admission.
        let _copy_pause = self
            .device
            .copy_gate
            .borrow()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map(|g| g.pause());
        self.device.activate()?;
        unsafe {
            check(cuStreamBeginCapture_v2(self.raw, 0))?;
        }
        let result = f();
        let mut raw = ptr::null_mut();
        let end = unsafe { check(cuStreamEndCapture(self.raw, &mut raw)) };
        if let Err(e) = result {
            if !raw.is_null() {
                unsafe {
                    cuGraphDestroy(raw);
                }
            }
            return Err(e);
        }
        end?;
        let mut node_count = 0;
        let counted = unsafe { check(cuGraphGetNodes(raw, ptr::null_mut(), &mut node_count)) };
        if let Err(error) = counted {
            unsafe {
                cuGraphDestroy(raw);
            }
            return Err(error);
        }
        let mut exec = ptr::null_mut();
        let instantiate = unsafe { check(cuGraphInstantiateWithFlags(&mut exec, raw, 0)) };
        unsafe {
            cuGraphDestroy(raw);
        }
        instantiate?;
        Ok(Graph {
            device: self.device.clone(),
            exec,
            node_count,
        })
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.sync();
        unsafe {
            cuStreamDestroy_v2(self.raw);
        }
    }
}
pub struct Graph {
    device: Rc<Device>,
    exec: Handle,
    node_count: usize,
}
impl Graph {
    /// CUDA nodes are a retention bound and telemetry, not an allocation-byte estimate.
    pub fn node_count(&self) -> usize {
        self.node_count
    }
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "CUDA context mismatch"
        );
        self.device.activate()?;
        unsafe { check(cuGraphLaunch(self.exec, stream.raw)) }
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.device.sync();
        unsafe {
            cuGraphExecDestroy(self.exec);
        }
    }
}
pub struct Kernel {
    device: Rc<Device>,
    module: Handle,
    function: Handle,
    pub threads: u32,
    pub shared: u32,
    programmatic: bool,
}
impl Kernel {
    pub fn resident_grid(&self) -> Result<u32> {
        self.device.activate()?;
        let mut blocks = 0;
        let mut sms = 0;
        unsafe {
            check(cuDeviceGetAttribute(&mut sms, 16, self.device.ordinal))?;
            check(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut blocks,
                self.function,
                self.threads as i32,
                self.shared as usize,
            ))?;
        }
        ensure!(
            blocks > 0 && sms > 0,
            "native kernel cannot reside on this GPU"
        );
        Ok((blocks * sms) as u32)
    }
    pub fn load(
        device: &Rc<Device>,
        path: &std::path::Path,
        name: &str,
        threads: u32,
        shared: u32,
    ) -> Result<Self> {
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 module path"))?,
        )?;
        let name = CString::new(name)?;
        device.activate()?;
        let mut module = ptr::null_mut();
        let mut function = ptr::null_mut();
        unsafe {
            check(cuModuleLoad(&mut module, path.as_ptr()))?;
        }
        let result = (|| unsafe {
            check(cuModuleGetFunction(&mut function, module, name.as_ptr()))?;
            // Static shared storage counts toward the default 48 KiB limit too.
            // A kernel requesting exactly 48 KiB dynamically still needs opt-in.
            if shared > 0 {
                check(cuFuncSetAttribute(function, 8, shared as i32))?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            unsafe {
                cuModuleUnload(module);
            }
            return Err(e);
        }
        Ok(Self {
            device: device.clone(),
            module,
            function,
            threads,
            shared,
            programmatic: false,
        })
    }
    /// Enable only for a kernel that waits with griddepcontrol.wait before
    /// touching its predecessor's outputs. Graph capture retains this contract.
    pub fn with_programmatic_dependency(mut self) -> Result<Self> {
        ensure!(self.device.arch >= 90, "PDL requires sm90 or later");
        self.programmatic = true;
        Ok(self)
    }
    /// ABI, pointer bounds, and argument lifetime are the caller's responsibility.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: [u32; 3],
        args: &mut [*mut c_void],
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &stream.device),
            "CUDA context mismatch"
        );
        self.device.activate()?;
        if self.programmatic {
            let mut attr = LaunchAttribute {
                id: 6,
                pad: 0,
                value: [0; 8],
            };
            attr.value[0] = 1;
            let config = LaunchConfig {
                grid,
                block: [self.threads, 1, 1],
                shared: self.shared,
                stream: stream.raw,
                attrs: &mut attr,
                count: 1,
            };
            return unsafe {
                check(cuLaunchKernelEx(
                    &config,
                    self.function,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ))
            };
        }
        unsafe {
            check(cuLaunchKernel(
                self.function,
                grid[0],
                grid[1],
                grid[2],
                self.threads,
                1,
                1,
                self.shared,
                stream.raw,
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        let _ = self.device.sync();
        unsafe {
            cuModuleUnload(self.module);
        }
    }
}
pub fn arg<T>(x: &mut T) -> *mut c_void {
    (x as *mut T).cast()
}
