//! Checksum-verified SSD reads on one bounded CPU worker. No CUDA ownership.
use crate::{
    ssd_format::{CHUNK, Hash},
    ssd_store::{Source, Store},
};
use anyhow::{Context, Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
};

static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
// Metadata is bounded independently of payload bytes, including tiny slices.
const MAX_PARTS: usize = 4096;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected {
    pub key: Hash,
    pub tokens: usize,
}
pub struct Chunk {
    job: u64,
    serial: u64,
    parts: Vec<Part>,
    len: usize,
    buffer: Vec<u8>,
}
#[derive(Clone, Copy)]
struct Part {
    source: Source,
    offset: u64,
    start: usize,
    len: usize,
}
impl Chunk {
    pub fn bytes(&self) -> &[u8] {
        &self.buffer[..self.len]
    }
    pub fn parts(&self) -> impl Iterator<Item = (Source, u64, &[u8])> {
        self.parts
            .iter()
            .map(|p| (p.source, p.offset, &self.buffer[p.start..p.start + p.len]))
    }
}
type Returned = Result<(Vec<u8>, Vec<Part>), String>;
type Done = (Store, Result<Option<Selected>>);
pub struct ReadJob {
    id: u64,
    choice: Receiver<Option<Selected>>,
    begin: Option<SyncSender<()>>,
    chunks: Option<Receiver<Chunk>>,
    replies: Option<SyncSender<Returned>>,
    done: Receiver<Done>,
    handle: Option<JoinHandle<()>>,
    result: Option<Done>,
    waiting: Option<u64>,
    cancelled: Arc<AtomicBool>,
    selected: Option<Option<Selected>>,
    begun: bool,
}
impl ReadJob {
    pub fn start_with_wake(
        store: Store,
        request: Arc<[i32]>,
        wake: Option<crate::request_broker::FileWake>,
    ) -> Result<Self> {
        Self::start_options(
            store,
            request,
            wake,
            std::env::var("ALASKA_SSD_READ_BATCH").as_deref() == Ok("1"),
        )
    }
    fn start_options(
        mut store: Store,
        request: Arc<[i32]>,
        wake: Option<crate::request_broker::FileWake>,
        batched: bool,
    ) -> Result<Self> {
        let id = NEXT_JOB
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("SSD read identity exhausted"))?;
        let (selection, choice) = mpsc::sync_channel(1);
        let (begin, start) = mpsc::sync_channel(1);
        let (send, chunks) = mpsc::sync_channel(1);
        let (replies, returned) = mpsc::sync_channel::<Returned>(1);
        let (complete, done) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = cancelled.clone();
        let handle = thread::Builder::new()
            .name("alaska-ssd-reader".into())
            .spawn(move || {
                crate::cpu_pin::helper_thread();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Option<Selected>> {
                        let selected = store
                            .lookup(&request)?
                            .map(|key| {
                                store
                                    .prefix_len(&key, &request)
                                    .map(|tokens| Selected { key, tokens })
                            })
                            .transpose()?;
                        selection
                            .send(selected)
                            .map_err(|_| anyhow::anyhow!("SSD read selection cancelled"))?;
                        if let Some(wake) = &wake {
                            wake.notify();
                        }
                        let Some(selected) = selected else {
                            return Ok(None);
                        };
                        start
                            .recv()
                            .context("SSD read cancelled before admission")?;
                        let mut exchange = vec![0; CHUNK];
                        let mut parts = Vec::new();
                        let mut used = 0usize;
                        let mut serial = 0u64;
                        let flush = |exchange: &mut Vec<u8>,
                                     parts: &mut Vec<Part>,
                                     used: &mut usize,
                                     serial: &mut u64|
                         -> Result<()> {
                            if parts.is_empty() {
                                return Ok(());
                            }
                            ensure!(!stop.load(Ordering::Relaxed), "SSD read cancelled");
                            send.send(Chunk {
                                job: id,
                                serial: *serial,
                                parts: std::mem::take(parts),
                                len: *used,
                                buffer: std::mem::take(exchange),
                            })
                            .map_err(|_| anyhow::anyhow!("SSD read cancelled"))?;
                            if let Some(wake) = &wake {
                                wake.notify();
                            }
                            let (buffer, metadata) = returned
                                .recv()
                                .context("SSD read cancelled")?
                                .map_err(anyhow::Error::msg)?;
                            *exchange = buffer;
                            *parts = metadata;
                            parts.clear();
                            *used = 0;
                            ensure!(exchange.len() == CHUNK, "SSD read buffer changed");
                            *serial = serial.checked_add(1).context("SSD read serial exhausted")?;
                            Ok(())
                        };
                        let loaded = store
                            .load(&selected.key, &request, |source, offset, bytes| {
                                ensure!(!stop.load(Ordering::Relaxed), "SSD read cancelled");
                                ensure!(bytes.len() <= CHUNK, "SSD read slice bound");
                                if used + bytes.len() > CHUNK || parts.len() == MAX_PARTS {
                                    flush(&mut exchange, &mut parts, &mut used, &mut serial)?;
                                }
                                // Every original typed slice has already passed the full
                                // file-chunk checksum. Preserve its order and boundaries.
                                exchange[used..used + bytes.len()].copy_from_slice(bytes);
                                parts.push(Part {
                                    source,
                                    offset,
                                    start: used,
                                    len: bytes.len(),
                                });
                                used += bytes.len();
                                if !batched || used == CHUNK || parts.len() == MAX_PARTS {
                                    flush(&mut exchange, &mut parts, &mut used, &mut serial)?;
                                }
                                Ok(())
                            })
                            .and_then(|value| {
                                flush(&mut exchange, &mut parts, &mut used, &mut serial)?;
                                Ok(value)
                            });
                        // A known-bad local entry must not keep winning future lookups.
                        // Cancellation leaves immutable cache files available for reuse.
                        if loaded.is_err() && !stop.load(Ordering::Relaxed) {
                            store
                                .forget(&selected.key)
                                .context("invalidating failed SSD read")?;
                        }
                        loaded?;
                        Ok(Some(selected))
                    },
                ))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("SSD reader panicked; reopen required")));
                let _ = complete.send((store, result));
                if let Some(wake) = &wake {
                    wake.notify();
                }
            })?;
        Ok(Self {
            id,
            choice,
            begin: Some(begin),
            chunks: Some(chunks),
            replies: Some(replies),
            done,
            handle: Some(handle),
            result: None,
            waiting: None,
            cancelled,
            selected: None,
            begun: false,
        })
    }
    pub fn selection(&mut self) -> Result<Option<Option<Selected>>> {
        if self.selected.is_none() {
            match self.choice.try_recv() {
                Ok(value) => self.selected = Some(value),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
            }
        }
        Ok(self.selected)
    }
    pub fn begin(&mut self) -> Result<()> {
        ensure!(
            self.selected.flatten().is_some() && !self.begun,
            "SSD read not selected/already begun"
        );
        self.begin
            .take()
            .context("SSD read cancelled")?
            .try_send(())
            .map_err(|_| anyhow::anyhow!("SSD read start unavailable"))?;
        self.begun = true;
        Ok(())
    }
    pub fn poll(&mut self) -> Result<Option<Chunk>> {
        ensure!(
            self.waiting.is_none(),
            "SSD read acknowledgement outstanding"
        );
        let Some(receiver) = &self.chunks else {
            return Ok(None);
        };
        match receiver.try_recv() {
            Ok(chunk) => {
                ensure!(
                    chunk.job == self.id && chunk.len <= CHUNK && chunk.buffer.len() == CHUNK,
                    "SSD read invalid chunk"
                );
                ensure!(
                    !chunk.parts.is_empty() && chunk.parts.len() <= MAX_PARTS,
                    "SSD read metadata bound"
                );
                let mut end = 0;
                for p in &chunk.parts {
                    ensure!(
                        p.start == end && p.len <= chunk.len - end,
                        "SSD read metadata range"
                    );
                    end += p.len;
                }
                ensure!(end == chunk.len, "SSD read payload coverage");
                self.waiting = Some(chunk.serial);
                Ok(Some(chunk))
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => Ok(None),
        }
    }
    pub fn acknowledge(&mut self, chunk: Chunk, copied: Result<()>) -> Result<()> {
        ensure!(
            chunk.job == self.id && self.waiting == Some(chunk.serial),
            "SSD read stale/foreign acknowledgement"
        );
        self.replies
            .as_ref()
            .context("SSD read cancelled")?
            .try_send(
                copied
                    .map(|_| (chunk.buffer, chunk.parts))
                    .map_err(|e| format!("{e:#}")),
            )
            .map_err(|_| anyhow::anyhow!("SSD read reply unavailable"))?;
        self.waiting = None;
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.begin = None;
        self.chunks = None;
        self.replies = None;
        self.waiting = None;
    }
    pub fn try_finish(&mut self) -> Result<Option<Done>> {
        ensure!(self.handle.is_some(), "SSD reader already consumed");
        if self.result.is_none() {
            match self.done.try_recv() {
                Ok(done) => self.result = Some(done),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => anyhow::bail!("SSD reader lost completion"),
            }
        }
        if !self.handle.as_ref().unwrap().is_finished() {
            return Ok(None);
        }
        self.handle
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("SSD reader termination failed"))?;
        self.cancel();
        Ok(self.result.take())
    }
}
impl Drop for ReadJob {
    fn drop(&mut self) {
        self.cancel();
    }
}
#[path = "ssd_reader_tests.rs"]
mod tests;
pub use tests::probe;
