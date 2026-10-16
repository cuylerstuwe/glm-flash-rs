//! Rust lifecycle for the qualified native RoCE byte transport. One CUDA stream
//! owns every exchange; failure poisons the runtime, never returns stale output.
use crate::roce_config::rails;
use crate::{
    comm::Nccl,
    cuda::{Buffer, Device, Stream},
};
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    ffi::{CStr, CString, c_char, c_void},
    net::{Ipv4Addr, Ipv6Addr, SocketAddr},
    path::Path,
    rc::Rc,
    time::Instant,
};

const LIBRARY_SHA: &str = "3fb78ffee27de9d48bfc058f5f7e42dcb20fe13a807c2534480ae664c179db0f";
const CLASSES: usize = 4; // max grid 8; distinct arrival counters for 1/2/4/8 CTAs
#[repr(C)]
struct Config {
    world: i32,
    rank: i32,
    n_hca: i32,
    cpu: i32,
    traffic_class: i32,
    threads: i32,
    hostreg: i32,
    lean: i32,
    slot_bytes: u64,
    counters: u64,
    stripe_min: u32,
    inline_max: u32,
    lazy_cq: i32,
    copy_mode: i32,
    max_blocks: i32,
    reserved: i32,
}
#[derive(Serialize)]
struct Hca {
    name: String,
    gid: i32,
    ip: Ipv4Addr,
}
// Explicit deployment input: rail order must be the same on every rank.
// No host names, fabric subnets or OS-assigned interface names are assumed.
fn discover() -> Result<Vec<Hca>> {
    let value = std::env::var("ALASKA_ROCE_RAILS")
        .context("set ALASKA_ROCE_RAILS=HCA0:GID0,HCA1:GID1 in consistent physical rail order")?;
    let mut result = Vec::new();
    for (name, gid) in rails(&value)? {
        let port = Path::new("/sys/class/infiniband")
            .join(&name)
            .join("ports/1");
        ensure!(
            std::fs::read_to_string(port.join("state"))?.contains("ACTIVE"),
            "inactive HCA {name}"
        );
        ensure!(
            std::fs::read_to_string(port.join("link_layer"))?.trim() == "Ethernet",
            "HCA is not Ethernet"
        );
        ensure!(
            std::fs::read_to_string(port.join(format!("gid_attrs/types/{gid}")))?.trim()
                == "RoCE v2",
            "selected GID is not RoCE v2"
        );
        let addr: Ipv6Addr = std::fs::read_to_string(port.join(format!("gids/{gid}")))?
            .trim()
            .parse()?;
        let ip = addr
            .to_ipv4_mapped()
            .context("RoCE requires an IPv4-mapped GID")?;
        ensure!(
            !ip.is_unspecified() && !ip.is_loopback() && !ip.is_multicast() && !ip.is_broadcast(),
            "invalid fabric address"
        );
        result.push(Hca { name, gid, ip });
    }
    ensure!(
        result[0].ip != result[1].ip,
        "fabric rails require distinct addresses"
    );
    Ok(result)
}
type ErrorFn = unsafe extern "C" fn() -> *const c_char;
type SizeFn = unsafe extern "C" fn() -> u64;
type NewFn =
    unsafe extern "C" fn(*const Config, *const *const c_char, *const i32, *mut *mut c_void) -> i32;
type BlobFn = unsafe extern "C" fn(*mut c_void, *mut u8, u64) -> i32;
type ConnectFn = unsafe extern "C" fn(*mut c_void, *const u8, u64, i32) -> i32;
type RunFn = unsafe extern "C" fn(*mut c_void, u64, u64, u64, i32, u64, u64, u64, u64) -> i32;
type CheckFn = unsafe extern "C" fn(*mut c_void) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void);

pub struct Roce {
    handle: *mut c_void,
    run: RunFn,
    check: CheckFn,
    destroy: DestroyFn,
    error: ErrorFn,
    blob: BlobFn,
    connect: ConnectFn,
    blob_bytes: usize,
    counters: Buffer,
    device: Rc<Device>,
    stream: Cell<Option<usize>>,
    poisoned: Cell<bool>,
    pub max_bytes: usize,
    timeout_ns: u64,
    settings: [u8; 32],
    hcas: Vec<Hca>,
    _library: Library,
}
impl Roce {
    fn status(&self, code: i32) -> Result<()> {
        if code != 0 {
            let message = unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy();
            bail!("native RoCE: {message}");
        }
        Ok(())
    }
    fn local(device: &Rc<Device>, path: &Path, rank: usize, timeout_ns: u64) -> Result<Self> {
        ensure!(
            rank < 4 && device.arch == 121,
            "RoCE requires the TP4 sm121 fleet"
        );
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(path)?)) == LIBRARY_SHA,
            "RoCE native library hash drift"
        );
        let hcas = discover()?;
        let library = unsafe { Library::new(path) }?;
        let config_size: SizeFn = unsafe { *library.get(b"ak_roce_config_bytes\0")? };
        ensure!(
            unsafe { config_size() } as usize == std::mem::size_of::<Config>(),
            "RoCE C ABI layout mismatch"
        );
        let blob_size: SizeFn = unsafe { *library.get(b"ak_roce_blob_bytes\0")? };
        let new: NewFn = unsafe { *library.get(b"ak_roce_new\0")? };
        let counters = device.alloc((2 + 2 * CLASSES) * 4)?;
        counters.zero()?;
        let mut this = Self {
            handle: std::ptr::null_mut(),
            run: unsafe { *library.get(b"ak_roce_run\0")? },
            check: unsafe { *library.get(b"ak_roce_check\0")? },
            destroy: unsafe { *library.get(b"ak_roce_destroy\0")? },
            error: unsafe { *library.get(b"ak_roce_error\0")? },
            blob: unsafe { *library.get(b"ak_roce_blob\0")? },
            connect: unsafe { *library.get(b"ak_roce_connect\0")? },
            blob_bytes: unsafe { blob_size() } as usize,
            counters,
            device: device.clone(),
            stream: Cell::new(None),
            poisoned: Cell::new(false),
            max_bytes: 8 * 1024 * 1024,
            timeout_ns,
            settings: [0; 32],
            hcas,
            _library: library,
        };
        ensure!(
            this.blob_bytes <= 16384,
            "unreasonable native connection record"
        );
        let names = this
            .hcas
            .iter()
            .map(|h| CString::new(h.name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let ptrs = names.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let gids = this.hcas.iter().map(|h| h.gid).collect::<Vec<_>>();
        let config = Config {
            world: 4,
            rank: rank as i32,
            n_hca: 2,
            cpu: crate::cpu_pin::roce_cpu(),
            traffic_class: 106,
            threads: 512,
            hostreg: 0,
            lean: 0,
            slot_bytes: this.max_bytes as u64,
            counters: this.counters.ptr,
            stripe_min: 0,
            inline_max: 0,
            lazy_cq: 0,
            copy_mode: 2,
            max_blocks: 8,
            reserved: 0,
        };
        let mut fingerprint = Sha256::new();
        fingerprint.update(LIBRARY_SHA.as_bytes());
        for v in [
            config.world as u64,
            config.n_hca as u64,
            config.traffic_class as u64,
            config.threads as u64,
            config.hostreg as u64,
            config.lean as u64,
            config.slot_bytes,
            config.stripe_min as u64,
            config.inline_max as u64,
            config.lazy_cq as u64,
            config.copy_mode as u64,
            config.max_blocks as u64,
            timeout_ns,
        ] {
            fingerprint.update(v.to_le_bytes());
        }
        this.settings = fingerprint.finalize().into();
        let status = unsafe { new(&config, ptrs.as_ptr(), gids.as_ptr(), &mut this.handle) };
        this.status(status)?;
        Ok(this)
    }
    pub fn new(
        device: &Rc<Device>,
        stream: &Stream,
        nccl: &Nccl,
        path: &Path,
        rank: usize,
    ) -> Result<Self> {
        Self::with_timeout(device, stream, nccl, path, rank, 120_000_000_000)
    }
    fn with_timeout(
        device: &Rc<Device>,
        stream: &Stream,
        nccl: &Nccl,
        path: &Path,
        rank: usize,
        timeout_ns: u64,
    ) -> Result<Self> {
        let local = Self::local(device, path, rank, timeout_ns);
        // Even failed setup participates in consensus. Nobody starts a RoCE
        // collective unless every participant has initialized compatible state.
        let mut record = [0u8; 40];
        if let Ok(this) = &local {
            record[..4].copy_from_slice(&1u32.to_le_bytes());
            record[4..8].copy_from_slice(&(this.blob_bytes as u32).to_le_bytes());
            record[8..].copy_from_slice(&this.settings);
        }
        let records = nccl.exchange_host(device, stream, &record)?;
        let failures = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r[..4] != 1u32.to_le_bytes())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            if let Err(e) = local {
                return Err(e.context(format!("RoCE setup failed on ranks {failures:?}")));
            }
            bail!("RoCE setup failed on ranks {failures:?}");
        }
        ensure!(
            records.iter().all(|r| r.as_slice() == record),
            "RoCE setup records disagree"
        );
        let this = local?;
        let mut blob = vec![0; this.blob_bytes];
        this.status(unsafe { (this.blob)(this.handle, blob.as_mut_ptr(), blob.len() as u64) })?;
        let all = nccl.exchange_host(device, stream, &blob)?.concat();
        let connected =
            this.status(unsafe { (this.connect)(this.handle, all.as_ptr(), all.len() as u64, -1) });
        let ready = u32::from(connected.is_ok()).to_le_bytes();
        let ready_all = nccl.exchange_host(device, stream, &ready)?;
        if ready_all.iter().any(|r| r.as_slice() != 1u32.to_le_bytes()) {
            connected?;
            bail!("RoCE peer connection failed; no inference admitted");
        }
        Ok(this)
    }
    pub fn check(&self) -> Result<()> {
        ensure!(!self.poisoned.get(), "RoCE runtime already poisoned");
        let result = self.status(unsafe { (self.check)(self.handle) });
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
    /// Same stream for every eager and graph exchange; callers must retain all
    /// buffers and this runtime until GPU completion/graph destruction.
    pub unsafe fn gather(&self, send: &Buffer, recv: &Buffer, stream: &Stream) -> Result<()> {
        ensure!(recv.bytes == send.bytes * 4, "invalid RoCE buffers");
        unsafe { self.gather_prefix(send, recv, send.bytes, stream) }
    }
    pub unsafe fn gather_prefix(
        &self,
        send: &Buffer,
        recv: &Buffer,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            bytes > 0 && bytes <= self.max_bytes && bytes <= send.bytes && bytes <= recv.bytes / 4,
            "invalid RoCE buffers"
        );
        unsafe { self.gather_raw(send.ptr, recv.ptr, bytes, stream) }
    }
    /// `send` holds `bytes` and `recv` four times as many, at device-accessible
    /// addresses (device or mapped page-locked memory) that outlive completion.
    pub unsafe fn gather_raw(
        &self,
        send: u64,
        recv: u64,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            bytes > 0 && bytes <= self.max_bytes,
            "invalid RoCE gather size"
        );
        ensure!(!self.poisoned.get(), "RoCE runtime poisoned");
        let raw = stream.raw() as usize;
        if let Some(previous) = self.stream.get() {
            ensure!(previous == raw, "RoCE requires one ordered CUDA stream");
        }
        self.stream.set(Some(raw));
        let grid = ((bytes.div_ceil(16)).div_ceil(2 * 512))
            .max(1)
            .next_power_of_two()
            .min(8);
        let class = grid.trailing_zeros() as usize;
        let stage = self.counters.ptr + 4 * (1 + class) as u64;
        let tail = self.counters.ptr + 4 * (1 + CLASSES + class) as u64;
        let status = unsafe {
            (self.run)(
                self.handle,
                send,
                recv,
                bytes as u64,
                grid as i32,
                stage,
                tail,
                self.timeout_ns,
                raw as u64,
            )
        };
        self.status(status)
    }
}
impl Drop for Roce {
    fn drop(&mut self) {
        let _ = self.device.sync();
        if !self.handle.is_null() {
            unsafe {
                (self.destroy)(self.handle);
            }
        }
    }
}

/// Deliberately incomplete setup and a missed exchange on rank 3. Run only in
/// an isolated fleet test; no model request or production marker is involved.
pub fn failure_probe(
    rank: usize,
    native: &Path,
    nccl_path: &Path,
    master: SocketAddr,
) -> Result<serde_json::Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let nccl = Nccl::new(nccl_path, rank, 4, master)?;
    let path = native.join(if rank == 3 {
        "deliberately-missing-roce-library.so"
    } else {
        "libalaska_roce.so"
    });
    let start = Instant::now();
    let failed_setup = Roce::new(&device, &stream, &nccl, &path, rank);
    ensure!(failed_setup.is_err(), "incomplete fleet admitted RoCE");
    let setup_seconds = start.elapsed().as_secs_f64();
    let roce = Roce::with_timeout(
        &device,
        &stream,
        &nccl,
        &native.join("libalaska_roce.so"),
        rank,
        200_000_000,
    )?;
    let send = device.alloc(16)?;
    let recv = device.alloc(64)?;
    send.write(0, &[rank as u8; 16])?;
    unsafe {
        roce.gather(&send, &recv, &stream)?;
    }
    stream.sync()?;
    roce.check()?;
    let start = Instant::now();
    let mut outcome = [0u8; 2];
    let mut error = None;
    if rank != 3 {
        unsafe {
            roce.gather(&send, &recv, &stream)?;
        }
        stream.sync()?;
        let check = roce.check();
        ensure!(check.is_err(), "missing peer did not poison transport");
        error = Some(check.unwrap_err().to_string());
        outcome[0] = 1;
        let retry = Instant::now();
        ensure!(
            unsafe { roce.gather(&send, &recv, &stream) }.is_err(),
            "poisoned runtime admitted another exchange"
        );
        ensure!(
            retry.elapsed().as_secs_f64() < 0.05,
            "poison rejection blocked"
        );
        outcome[1] = 1;
    } else {
        // No GPU work on this rank while its peers wait for its missing flag.
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
    let failure_seconds = start.elapsed().as_secs_f64();
    let records = nccl.exchange_host(&device, &stream, &outcome)?;
    ensure!(
        records[..3].iter().all(|r| r == &[1, 1]) && records[3] == [0, 0],
        "failure consensus mismatch"
    );
    Ok(
        serde_json::json!({"rank":rank,"setup_failed_collectively":true,"setup_seconds":setup_seconds,
        "failure_seconds":failure_seconds,"missing_rank":3,"waiting_ranks_poisoned":true,
        "poisoned_relaunch_rejected":true,"error":error,"no_failed_output_consumed":true}),
    )
}

pub fn probe(
    rank: usize,
    native: &Path,
    nccl_path: &Path,
    master: SocketAddr,
) -> Result<serde_json::Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let nccl = Nccl::new(nccl_path, rank, 4, master)?;
    let roce = Roce::new(
        &device,
        &stream,
        &nccl,
        &native.join("libalaska_roce.so"),
        rank,
    )?;
    let mut cases = Vec::new();
    for size in [16, 17, 4096, 16384, 262144, 8 * 1024 * 1024] {
        let send = device.alloc(size)?;
        let recv = device.alloc(size * 4)?;
        let payload = (0..size)
            .map(|i| (i.wrapping_mul(37) ^ (rank * 71)) as u8)
            .collect::<Vec<_>>();
        send.write(0, &payload)?;
        unsafe {
            roce.gather(&send, &recv, &stream)?;
        }
        stream.sync()?;
        roce.check()?;
        let first = recv.read(0, recv.bytes)?;
        for (peer, row) in first.chunks_exact(size).enumerate() {
            ensure!(
                row.iter()
                    .enumerate()
                    .all(|(i, &b)| b == (i.wrapping_mul(37) ^ (peer * 71)) as u8),
                "RoCE bytes differ at size {size}"
            );
        }
        let graph = unsafe { stream.capture(|| roce.gather(&send, &recv, &stream))? };
        for _ in 0..10 {
            unsafe {
                graph.launch(&stream)?;
            }
        }
        stream.sync()?;
        roce.check()?;
        ensure!(
            first == recv.read(0, recv.bytes)?,
            "RoCE graph/eager bytes differ"
        );
        let mut paired = Vec::new();
        if size.is_multiple_of(4) {
            let reference =
                unsafe { stream.capture(|| nccl.all_gather_f32(&send, &recv, &stream))? };
            for _ in 0..10 {
                unsafe {
                    reference.launch(&stream)?;
                }
            }
            stream.sync()?;
            for round in 0..5 {
                let mut times = [0.; 2];
                for which in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let g = if which == 0 { &reference } else { &graph };
                    let start = Instant::now();
                    for _ in 0..100 {
                        unsafe {
                            g.launch(&stream)?;
                        }
                    }
                    stream.sync()?;
                    roce.check()?;
                    times[which] = start.elapsed().as_secs_f64() * 10000.;
                    ensure!(
                        first == recv.read(0, recv.bytes)?,
                        "transport comparison corrupted bytes"
                    );
                }
                paired.push(serde_json::json!({"nccl_us":times[0],"roce_us":times[1]}));
            }
        }
        cases.push(serde_json::json!({"send_bytes":size,"rank_order_exact":true,"graph_eager_equal":true,"paired":paired}));
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    ensure!(
        !maps.contains("libpython") && !maps.contains("libtorch"),
        "interpreter/Torch mapped"
    );
    Ok(
        serde_json::json!({"rank":rank,"cases":cases,"hcas":roce.hcas,"python_torch_mapped":false,
        "scope":"native transport only; no full-model or production qualification"}),
    )
}
