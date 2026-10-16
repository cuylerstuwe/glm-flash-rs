//! Exercise actual helper-thread DMA concurrently with lazy graph capture.
use crate::{
    copy_gate::Gate,
    cuda::{CopyStream, Device},
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

pub fn run() -> Result<serde_json::Value> {
    let device = Device::new()?;
    let gate = Gate::new();
    device.register_copy_gate(&gate)?;
    let bytes = 2 * 1024 * 1024;
    let expected: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let source = device.alloc(bytes)?;
    source.write(0, &expected)?;
    let target = device.alloc(bytes)?;
    let stream = device.stream()?;
    let mut copy = CopyStream::new(&device, bytes)?;
    let address = source.ptr;
    let running = Arc::new(AtomicBool::new(true));
    let count = Arc::new(AtomicUsize::new(0));
    gate.hold(false);
    let (worker_gate, worker_running, worker_count) =
        (gate.clone(), running.clone(), count.clone());
    let worker = std::thread::spawn(move || -> Result<()> {
        while worker_running.load(Ordering::Acquire) {
            let _permit = worker_gate.enter()?;
            // Source remains immutable and alive until this thread joins.
            unsafe {
                copy.batch(&[(0, address, bytes)])?;
            }
            ensure!(copy.bytes() == expected, "independent copy mismatch");
            worker_count.fetch_add(1, Ordering::Release);
        }
        Ok(())
    });
    let result = (|| -> Result<()> {
        let start = Instant::now();
        while count.load(Ordering::Acquire) == 0 {
            ensure!(
                start.elapsed() < Duration::from_secs(5),
                "copy startup timeout"
            );
            std::thread::yield_now();
        }
        for _ in 0..64 {
            let graph = unsafe {
                stream.capture(|| {
                    let before = count.load(Ordering::Acquire);
                    target.zero_prefix(&stream, bytes)?;
                    std::thread::sleep(Duration::from_millis(1));
                    ensure!(
                        count.load(Ordering::Acquire) == before,
                        "DMA advanced during capture"
                    );
                    Ok(())
                })?
            };
            unsafe {
                graph.launch(&stream)?;
            }
            stream.sync()?;
            // Require actual DMA progress between captures, not a parked helper.
            let previous = count.load(Ordering::Acquire);
            let start = Instant::now();
            while count.load(Ordering::Acquire) == previous {
                ensure!(
                    start.elapsed() < Duration::from_secs(5),
                    "copy did not resume"
                );
                std::thread::yield_now();
            }
        }
        Ok(())
    })();
    running.store(false, Ordering::Release);
    gate.hold(false);
    let joined = worker
        .join()
        .map_err(|_| anyhow::anyhow!("copy worker panicked"))?;
    result?;
    joined?;
    ensure!(
        target.read(0, bytes)?.iter().all(|b| *b == 0),
        "captured memset mismatch"
    );
    Ok(
        serde_json::json!({"captures":64,"copies":count.load(Ordering::Acquire),"bytes_per_copy":bytes,"copy_bytes_exact":true,"capture_quiescence":true,"resumption":true}),
    )
}
