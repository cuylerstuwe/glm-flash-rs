//! Native NCCL collectives with Rust TCP bootstrap. Never sum partials in NCCL:
//! the model depends on all-gather followed by a deterministic rank-order sum.
use crate::cuda::{Buffer, Device, Pinned, Stream};
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    ptr,
    time::{Duration, Instant},
};
#[repr(C)]
#[derive(Clone, Copy)]
struct UniqueId {
    bytes: [u8; 128],
}
type Comm = *mut c_void;
type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type GetId = unsafe extern "C" fn(*mut UniqueId) -> c_int;
type Init = unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> c_int;
type Gather =
    unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, Comm, *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(Comm) -> c_int;
type Group = unsafe extern "C" fn() -> c_int;
type Send = unsafe extern "C" fn(*const c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int;
type Receive = unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int;
#[path = "nccl_exchange_plan.rs"]
mod exchange_plan;
/// No ownership escapes: every region's Buffer must remain live until stream completion.
#[allow(dead_code)] // Bounded row-split communication owner.
pub struct Region<'a> {
    pub buffer: &'a Buffer,
    pub offset: usize,
    pub bytes: usize,
}
#[allow(dead_code)]
pub struct Exchange<'a> {
    pub peer: usize,
    pub send: Region<'a>,
    pub receive: Region<'a>,
}
pub struct Nccl {
    raw: Comm,
    world: usize,
    rank: usize,
    group_start: Group,
    group_end: Group,
    send: Send,
    receive: Receive,
    gather: Gather,
    destroy: Destroy,
    error: ErrorString,
    _lib: Library,
}
fn check(error: ErrorString, code: i32) -> Result<()> {
    if code != 0 {
        bail!(
            "NCCL {code}: {}",
            unsafe { CStr::from_ptr(error(code)) }.to_string_lossy()
        );
    }
    Ok(())
}
impl Nccl {
    /// Fixed-size bootstrap/control records only. Every rank must pass the same
    /// byte count; returned records preserve rank order and original bytes.
    pub fn exchange_host(
        &self,
        device: &std::rc::Rc<Device>,
        stream: &Stream,
        bytes: &[u8],
    ) -> Result<Vec<Vec<u8>>> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= 65536,
            "bounded control record required"
        );
        let stride = bytes.len().next_multiple_of(4);
        let send = device.alloc(stride)?;
        let recv = device.alloc(stride * self.world)?;
        let mut host = Pinned::new(device, stride * (self.world + 1))?;
        host.bytes_mut()[..stride].fill(0);
        host.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        unsafe {
            host.upload_to(stream, &send, 0, 0, stride)?;
            self.all_gather_f32(&send, &recv, stream)?;
            host.download_from(stream, stride, &recv, 0, recv.bytes)?;
        }
        stream.sync()?;
        let all = &host.bytes_mut()[stride..];
        Ok(all
            .chunks_exact(stride)
            .map(|r| r[..bytes.len()].to_vec())
            .collect())
    }
    pub fn new(library: &Path, rank: usize, world: usize, master: SocketAddr) -> Result<Self> {
        ensure!(world == 4 && rank < world, "experiment requires TP4");
        let lib = unsafe { Library::new(library) }?;
        let error: ErrorString = unsafe { *lib.get(b"ncclGetErrorString\0")? };
        let get_id: GetId = unsafe { *lib.get(b"ncclGetUniqueId\0")? };
        let init: Init = unsafe { *lib.get(b"ncclCommInitRank\0")? };
        let gather: Gather = unsafe { *lib.get(b"ncclAllGather\0")? };
        let destroy: Destroy = unsafe { *lib.get(b"ncclCommDestroy\0")? };
        let group_start: Group = unsafe { *lib.get(b"ncclGroupStart\0")? };
        let group_end: Group = unsafe { *lib.get(b"ncclGroupEnd\0")? };
        let send: Send = unsafe { *lib.get(b"ncclSend\0")? };
        let receive: Receive = unsafe { *lib.get(b"ncclRecv\0")? };
        let mut uid = UniqueId { bytes: [0; 128] };
        if rank == 0 {
            check(error, unsafe { get_id(&mut uid) })?;
            let listener = TcpListener::bind(master)?;
            listener.set_nonblocking(true)?;
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut seen = vec![false; world];
            seen[0] = true;
            while seen.iter().any(|v| !*v) {
                ensure!(
                    Instant::now() < deadline,
                    "bootstrap timed out waiting for ranks"
                );
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        socket.set_read_timeout(Some(Duration::from_secs(5)))?;
                        socket.set_write_timeout(Some(Duration::from_secs(5)))?;
                        let mut hello = [0u8; 8];
                        socket.read_exact(&mut hello)?;
                        ensure!(&hello[..4] == b"AKR1", "bad bootstrap protocol");
                        let peer = u32::from_le_bytes(hello[4..].try_into().unwrap()) as usize;
                        ensure!(
                            peer > 0 && peer < world && !seen[peer],
                            "duplicate/invalid rank"
                        );
                        socket.write_all(&uid.bytes)?;
                        seen[peer] = true;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20))
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        } else {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut socket = loop {
                match TcpStream::connect_timeout(&master, Duration::from_secs(1)) {
                    Ok(s) => break s,
                    Err(e) => {
                        ensure!(
                            Instant::now() < deadline,
                            "bootstrap connection timed out: {e}"
                        );
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            };
            socket.set_read_timeout(Some(Duration::from_secs(60)))?;
            socket.set_write_timeout(Some(Duration::from_secs(5)))?;
            socket.write_all(b"AKR1")?;
            socket.write_all(&(rank as u32).to_le_bytes())?;
            socket.read_exact(&mut uid.bytes)?;
        }
        let mut raw = ptr::null_mut();
        check(error, unsafe {
            init(&mut raw, world as i32, uid, rank as i32)
        })?;
        Ok(Self {
            raw,
            world,
            rank,
            group_start,
            group_end,
            send,
            receive,
            gather,
            destroy,
            error,
            _lib: lib,
        })
    }
    /// Byte copies only: the caller retains deterministic rank-order reduction.
    /// Every participant must submit a matching peer/byte schedule, already
    /// agreed by its common native-work boundary. Local bounds validate before
    /// opening the NCCL group; native failures remain fatal to that model work.
    /// All buffers and the stream must belong to the communicator’s CUDA device
    /// and context. Buffers and the communicator must outlive stream/graph execution.
    #[allow(dead_code)] // Direct transport diagnostic entry point.
    pub unsafe fn exchange_regions(&self, entries: &[Exchange<'_>], stream: &Stream) -> Result<()> {
        ensure!(
            entries.len() <= exchange_plan::MAX_TRANSFERS,
            "NCCL exchange descriptors"
        );
        let mut plans = [exchange_plan::Transfer::default(); exchange_plan::MAX_TRANSFERS];
        for (entry, plan) in entries.iter().zip(&mut plans) {
            let region = |r: &Region<'_>| exchange_plan::Region {
                address: r.buffer.ptr,
                capacity: r.buffer.bytes,
                offset: r.offset,
                bytes: r.bytes,
            };
            *plan = exchange_plan::Transfer {
                peer: entry.peer,
                send: region(&entry.send),
                receive: region(&entry.receive),
            };
        }
        exchange_plan::validate(&plans[..entries.len()], self.rank, self.world)?;
        let status = exchange_plan::grouped(
            entries.len() * 2,
            || unsafe { (self.group_start)() },
            |i| {
                let sending = i < entries.len();
                let plan = plans[if sending { i } else { i - entries.len() }];
                if sending {
                    if plan.send.bytes == 0 {
                        return 0;
                    }
                    unsafe {
                        (self.send)(
                            (plan.send.address + plan.send.offset as u64) as *const c_void,
                            plan.send.bytes,
                            0,
                            plan.peer as c_int,
                            self.raw,
                            stream.raw(),
                        )
                    }
                } else {
                    if plan.receive.bytes == 0 {
                        return 0;
                    }
                    unsafe {
                        (self.receive)(
                            (plan.receive.address + plan.receive.offset as u64) as *mut c_void,
                            plan.receive.bytes,
                            0,
                            plan.peer as c_int,
                            self.raw,
                            stream.raw(),
                        )
                    }
                }
            },
            || unsafe { (self.group_end)() },
        );
        check(self.error, status)
    }
    // Buffers must remain live until stream completion; graph users must retain them for graph lifetime.
    pub unsafe fn all_gather_f32(
        &self,
        send: &Buffer,
        recv: &Buffer,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            recv.bytes == send.bytes * self.world,
            "collective buffer size mismatch"
        );
        unsafe { self.all_gather_prefix(send, recv, send.bytes, stream) }
    }
    pub unsafe fn all_gather_prefix(
        &self,
        send: &Buffer,
        recv: &Buffer,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            bytes > 0
                && bytes.is_multiple_of(4)
                && bytes <= send.bytes
                && bytes <= recv.bytes / self.world,
            "collective buffer size mismatch"
        );
        check(self.error, unsafe {
            (self.gather)(
                send.ptr as *const c_void,
                recv.ptr as *mut c_void,
                bytes / 4,
                7,
                self.raw,
                stream.raw(),
            )
        })
    }
}
impl Drop for Nccl {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.raw);
        }
    }
}
pub fn probe(rank: usize, library: &Path, master: SocketAddr) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let stream = d.stream()?;
    let c = Nccl::new(library, rank, 4, master)?;
    let values: Vec<f32> = (0..4096).map(|i| (rank * 4096 + i) as f32).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let send = d.alloc(bytes.len())?;
    let recv = d.alloc(bytes.len() * 4)?;
    send.write(0, &bytes)?;
    recv.zero()?;
    unsafe {
        c.all_gather_f32(&send, &recv, &stream)?;
    }
    stream.sync()?;
    let actual = recv.read(0, recv.bytes)?;
    for (i, v) in actual.as_chunks::<4>().0.iter().enumerate() {
        ensure!(
            f32::from_le_bytes(*v) == i as f32,
            "rank-order gather mismatch {i}"
        );
    }
    let graph = unsafe { stream.capture(|| c.all_gather_f32(&send, &recv, &stream)) }?;
    for _ in 0..10 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let start = Instant::now();
    for _ in 0..100 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let us = start.elapsed().as_secs_f64() * 10000.;
    ensure!(
        recv.read(0, recv.bytes)? == actual,
        "captured NCCL differs from eager"
    );
    // Arbitrary control bytes (including unaligned payload lengths and float
    // NaN/subnormal encodings) must preserve the current round, not a stale DMA.
    for round in 0..256usize {
        let len = [1usize, 64, 4095, 65533][round % 4];
        let record = |peer: usize| -> Vec<u8> {
            (0..len)
                .map(|i| {
                    (i.wrapping_mul(73) ^ round.wrapping_mul(131) ^ peer.wrapping_mul(47)) as u8
                })
                .collect()
        };
        let all = c.exchange_host(&d, &stream, &record(rank))?;
        ensure!(
            all.iter()
                .enumerate()
                .all(|(peer, bytes)| *bytes == record(peer)),
            "NCCL host record changed/stale round {round}"
        );
    }
    let maps = std::fs::read_to_string("/proc/self/maps").context("runtime dependency audit")?;
    ensure!(
        !maps.contains("libpython") && !maps.contains("libtorch"),
        "unexpected interpreter/Torch library"
    );
    drop(graph);
    stream.sync()?;
    drop(c);
    Ok(
        serde_json::json!({"milestone":"native_tp4_nccl","rank":rank,"rank_order_equal":true,"graph_byte_equal":true,"host_records_byte_exact":true,"host_record_rounds":256,"python_or_torch_loaded":false,"send_bytes_per_rank":send.bytes,"gather_graph_us":us}),
    )
}
