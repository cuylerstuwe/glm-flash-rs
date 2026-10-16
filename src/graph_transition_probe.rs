//! Per-kernel cost of back-to-back launches in one captured graph on this process's own
//! load/capture/launch path (a pinned Triton kernel, router_sum), for comparison with the
//! same cubin launched by a minimal driver-API program. Graphs of `short` and `long`
//! launches; the replay-time difference over the launch-count difference cancels the
//! fixed launch and synchronization cost.
//!
//! `init` reproduces parts of the service's process state first, one letter each:
//! k every pinned Triton kernel loaded, b a cuBLAS handle and one selector GEMM,
//! l the L2 prefetch library (ALASKA_L2_PREFETCH_LIBRARY) queried, m 60 GiB allocated.
use crate::{cuda::Device, dsa::words, front::load_kernel};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, path::Path, rc::Rc, time::Instant};

pub(crate) fn run(
    kernels: &Path,
    short: usize,
    long: usize,
    replays: usize,
    init: &str,
) -> Result<serde_json::Value> {
    ensure!(
        0 < short && short < long && replays >= 10,
        "graph transition probe bounds"
    );
    let d = Device::new()?;
    let s = d.stream()?;
    let mut held_kernels = Vec::new();
    let mut held_buffers = Vec::new();
    let mut held_blas = None;
    let mut held_library = None;
    if init.contains('k') {
        let specs: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(include_str!("../native/kernel-manifest.json"))?;
        for alias in specs.keys() {
            held_kernels.push(load_kernel(&d, kernels, alias)?);
        }
    }
    if init.contains('b') {
        let blas = crate::draft_blas::Blas::new(&d)?;
        let x = d.alloc(8 * 4096 * 2)?;
        let w = d.alloc(256 * 4096 * 2)?;
        let out = d.alloc(8 * 256 * 2)?;
        x.zero()?;
        w.zero()?;
        unsafe { blas.selector(&s, x.ptr, w.ptr, &out, 8)? };
        s.sync()?;
        held_buffers.extend([x, w, out]);
        held_blas = Some(blas);
    }
    if init.contains('l') {
        let path = std::env::var_os("ALASKA_L2_PREFETCH_LIBRARY")
            .context("init l needs ALASKA_L2_PREFETCH_LIBRARY")?;
        let library = unsafe { libloading::Library::new(path)? };
        let properties: unsafe extern "C" fn(*mut i32, *mut i32) -> i32 =
            unsafe { *library.get(b"alaska_l2pf_properties\0")? };
        let (mut sms, mut l2) = (0, 0);
        ensure!(
            unsafe { properties(&mut sms, &mut l2) } == 0,
            "L2 library query"
        );
        held_library = Some(library);
    }
    if init.contains('m') {
        for _ in 0..60 {
            held_buffers.push(d.alloc(1 << 30)?);
        }
    }
    let aliases: &[&str] = if init.contains('x') {
        MIX
    } else {
        &["router_sum"]
    };
    let medians = if init.contains('s') {
        let native = std::env::var_os("ALASKA_PROBE_NATIVE")
            .context("init s needs ALASKA_PROBE_NATIVE (the native cubin directory)")?;
        measure_streaming(&d, kernels, Path::new(&native), short, long, replays)?
    } else if init.contains('n') || init.contains('f') {
        let native = std::env::var_os("ALASKA_PROBE_NATIVE")
            .context("init n/f needs ALASKA_PROBE_NATIVE (the native cubin directory)")?;
        measure_native(
            &d,
            kernels,
            Path::new(&native),
            short,
            long,
            replays,
            init.contains('f'),
        )?
    } else {
        measure(&d, kernels, short, long, replays, aliases)?
    };
    let held = (
        held_kernels.len(),
        held_buffers.len(),
        held_blas.is_some(),
        held_library.is_some(),
    );
    Ok(serde_json::json!({
        "kernels": aliases,
        "init": init,
        "held": [held.0, held.1, held.2, held.3],
        "short": short,
        "long": long,
        "median_ms": [medians[0] * 1e3, medians[1] * 1e3],
        "us_per_kernel": (medians[1] - medians[0]) * 1e6 / (long - short) as f64,
    }))
}

/// ALASKA_TRANSITION_PROBE=1: the same measurement inside the service on its own device,
/// logged with `stage`; never fails the service.
pub(crate) fn service_stage(d: &Rc<Device>, kernels: &Path, stage: &str) {
    if std::env::var("ALASKA_TRANSITION_PROBE").as_deref() != Ok("1") {
        return;
    }
    for (label, aliases) in [("single", &["router_sum"][..]), ("mix", MIX)] {
        match measure(d, kernels, 500, 1500, 100, aliases) {
            Ok(m) => eprintln!(
                "transition probe {stage} {label}: {:.4} us/kernel ({:.4} / {:.4} ms)",
                (m[1] - m[0]) * 1e6 / 1000.0,
                m[0] * 1e3,
                m[1] * 1e3
            ),
            Err(e) => eprintln!("transition probe {stage} {label}: failed: {e:#}"),
        }
    }
}

/// Four pinned kernels from four modules sharing the (part, out, n, 0, 0) launch ABI.
const MIX: &[&str] = &["router_sum", "reduce8_bf16", "reduce2_bf16", "reduce4_bf16"];

/// Before every fifth router_sum, the AOT native draft_rank_sum streams 64 MiB (four times
/// the L2), as weight-streaming kernels do between a real graph's small kernels.
fn measure_streaming(
    d: &Rc<Device>,
    kernels: &Path,
    native: &Path,
    short: usize,
    long: usize,
    replays: usize,
) -> Result<[f64; 2]> {
    let s = d.stream()?;
    let triton = load_kernel(d, kernels, "router_sum")?;
    let stream_sum = crate::front::load_native(
        d,
        &native.join("draft_support.cubin"),
        "draft_rank_sum",
        256,
    )?;
    let n = 4 * 128usize;
    let part = d.alloc(1 << 20)?;
    let out = d.alloc(1 << 20)?;
    let big = d.alloc(64 << 20)?;
    let sink = d.alloc(32 << 20)?;
    part.zero()?;
    out.zero()?;
    big.zero()?;
    // draft_rank_sum reads four rank slices of `count` floats: 64 MiB in all.
    let count = (64usize << 20) / 16;
    let mut medians = [0.0; 2];
    for (slot, launches) in [short, long].into_iter().enumerate() {
        let graph = unsafe {
            s.capture(|| {
                for i in 0..launches {
                    if i % 5 == 0 {
                        let mut all = big.ptr;
                        let mut x = sink.ptr;
                        let mut c = count as i32;
                        let mut args = [
                            crate::cuda::arg(&mut all),
                            crate::cuda::arg(&mut x),
                            crate::cuda::arg(&mut c),
                        ];
                        stream_sum.launch(&s, [count.div_ceil(256) as u32, 1, 1], &mut args)?;
                    }
                    words(
                        &triton,
                        &s,
                        [n.div_ceil(1024) as u32, 1, 1],
                        &[part.ptr, out.ptr, n as u64],
                    )?;
                }
                Ok(())
            })?
        };
        for _ in 0..3 {
            unsafe { graph.launch(&s)? };
            s.sync()?;
        }
        let mut times = Vec::with_capacity(replays);
        for _ in 0..replays {
            let t = Instant::now();
            unsafe { graph.launch(&s)? };
            s.sync()?;
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        medians[slot] = times[times.len() / 2];
    }
    Ok(medians)
}

/// router_sum alternating with the AOT native draft_advance (one thread bumps a counter);
/// with `fork`, router_sum only on the main stream, and before every 8th launch a fork to a
/// side stream that runs draft_advance, joined once at the end (the L2 prefetch pattern).
fn measure_native(
    d: &Rc<Device>,
    kernels: &Path,
    native: &Path,
    short: usize,
    long: usize,
    replays: usize,
    fork: bool,
) -> Result<[f64; 2]> {
    let s = d.stream()?;
    let side = d.stream()?;
    let mut forked = d.fence()?;
    let mut joined = d.fence()?;
    let triton = load_kernel(d, kernels, "router_sum")?;
    let advance =
        crate::front::load_native(d, &native.join("draft_support.cubin"), "draft_advance", 32)?;
    let n = 4 * 128usize;
    let part = d.alloc(1 << 20)?;
    let out = d.alloc(1 << 20)?;
    let counter = d.alloc(64)?;
    part.zero()?;
    out.zero()?;
    counter.zero()?;
    let bump = |stream: &crate::cuda::Stream| -> Result<()> {
        let mut pos = counter.ptr;
        let mut rows = 1i32;
        let mut args = [crate::cuda::arg(&mut pos), crate::cuda::arg(&mut rows)];
        unsafe { advance.launch(stream, [1, 1, 1], &mut args) }
    };
    let mut medians = [0.0; 2];
    for (slot, count) in [short, long].into_iter().enumerate() {
        let graph = unsafe {
            s.capture(|| {
                for i in 0..count {
                    if fork && i % 8 == 0 {
                        forked.record(&s)?;
                        forked.wait_on(&side)?;
                        bump(&side)?;
                    }
                    if fork || i % 2 == 0 {
                        words(
                            &triton,
                            &s,
                            [n.div_ceil(1024) as u32, 1, 1],
                            &[part.ptr, out.ptr, n as u64],
                        )?;
                    } else {
                        bump(&s)?;
                    }
                }
                if fork {
                    joined.record(&side)?;
                    joined.wait_on(&s)?;
                }
                Ok(())
            })?
        };
        for _ in 0..5 {
            unsafe { graph.launch(&s)? };
            s.sync()?;
        }
        let mut times = Vec::with_capacity(replays);
        for _ in 0..replays {
            let t = Instant::now();
            unsafe { graph.launch(&s)? };
            s.sync()?;
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        medians[slot] = times[times.len() / 2];
    }
    Ok(medians)
}

fn measure(
    d: &Rc<Device>,
    kernels: &Path,
    short: usize,
    long: usize,
    replays: usize,
    aliases: &[&str],
) -> Result<[f64; 2]> {
    let s = d.stream()?;
    let ks = aliases
        .iter()
        .map(|alias| load_kernel(d, kernels, alias))
        .collect::<Result<Vec<_>>>()?;
    let n = 4 * 128usize;
    let part = d.alloc(1 << 20)?;
    let out = d.alloc(1 << 20)?;
    part.zero()?;
    out.zero()?;
    let mut medians = [0.0; 2];
    for (slot, count) in [short, long].into_iter().enumerate() {
        let graph = unsafe {
            s.capture(|| {
                for i in 0..count {
                    words(
                        &ks[i % ks.len()],
                        &s,
                        [n.div_ceil(1024) as u32, 1, 1],
                        &[part.ptr, out.ptr, n as u64],
                    )?;
                }
                Ok(())
            })?
        };
        for _ in 0..5 {
            unsafe { graph.launch(&s)? };
            s.sync()?;
        }
        let mut times = Vec::with_capacity(replays);
        for _ in 0..replays {
            let t = Instant::now();
            unsafe { graph.launch(&s)? };
            s.sync()?;
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        medians[slot] = times[times.len() / 2];
    }
    Ok(medians)
}
