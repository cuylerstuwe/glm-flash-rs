//! One bounded file worker. Only ordinary Rust-owned bytes cross its channels;
//! the caller drives source copies with nonblocking poll/respond operations.
use crate::{
    ssd_format::{CHUNK, Identity, PAGE_BYTES},
    ssd_store::{Outcome, Source, Store},
};
use anyhow::{Context, Result, ensure};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
};

static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy)]
enum CopySource {
    Slice(Source, u64),
    Page(usize),
}
pub struct Request {
    job: u64,
    serial: u64,
    source: CopySource,
    len: usize,
    bytes: Vec<u8>,
}
impl Request {
    /// One bounded owner boundary fills either a layer slice or an entire page.
    /// Reuse the codec's page layout; file bytes and checksums remain unchanged.
    pub fn fill(
        &mut self,
        mut source: impl FnMut(Source, u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let bytes = &mut self.bytes[..self.len];
        match self.source {
            CopySource::Slice(part, offset) => source(part, offset, bytes),
            CopySource::Page(logical) => {
                crate::ssd_store::fill_page(logical, 0, bytes, &mut source)
            }
        }
    }
}
fn page_slice(source: Source, offset: u64, len: usize) -> Result<Option<(usize, usize)>> {
    let Source::Page {
        logical,
        family,
        pooled,
    } = source
    else {
        return Ok(None);
    };
    let latent = crate::cache_pages::PAGE * 528;
    let pool = crate::cache_pages::PAGE / 4 * 256;
    let bytes = if pooled { pool } else { latent };
    ensure!(
        family < 11 && offset <= bytes as u64 && len as u64 <= bytes as u64 - offset,
        "SSD packed-page slice bounds"
    );
    Ok(Some((
        logical,
        family * (latent + pool) + if pooled { latent } else { 0 } + offset as usize,
    )))
}
type Done = (Store, Result<Outcome>);
pub struct WriteJob {
    id: u64,
    requests: Option<Receiver<Request>>,
    replies: Option<SyncSender<Result<Vec<u8>, String>>>,
    done: Receiver<Done>,
    handle: Option<JoinHandle<()>>,
    result: Option<Done>,
    waiting: Option<u64>,
}
impl WriteJob {
    pub fn start(mut store: Store, tokens: Arc<[i32]>) -> Result<Self> {
        let id = NEXT_JOB.fetch_add(1, Ordering::Relaxed);
        ensure!(id != u64::MAX, "SSD job identity exhausted");
        let (send, requests) = mpsc::sync_channel(1);
        let (replies, receive) = mpsc::sync_channel::<Result<Vec<u8>, String>>(1);
        let (complete, done) = mpsc::sync_channel(1);
        let handle = thread::Builder::new()
            .name("alaska-ssd-writer".into())
            .spawn(move || {
                crate::cpu_pin::helper_thread();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut exchange = vec![0; CHUNK];
                    let mut serial = 0u64;
                    let mut cached_page = None;
                    ensure!(
                        PAGE_BYTES as usize <= CHUNK,
                        "SSD packed page exceeds exchange"
                    );
                    store.save(&tokens, false, |source, offset, buf| {
                        ensure!(buf.len() <= CHUNK, "SSD exchange bound");
                        let page = page_slice(source, offset, buf.len())?;
                        let hit = page.is_some_and(|(logical, _)| cached_page == Some(logical));
                        if !hit {
                            let (source, len) = match page {
                                Some((logical, _)) => {
                                    (CopySource::Page(logical), PAGE_BYTES as usize)
                                }
                                None => (CopySource::Slice(source, offset), buf.len()),
                            };
                            send.send(Request {
                                job: id,
                                serial,
                                source,
                                len,
                                bytes: std::mem::take(&mut exchange),
                            })
                            .map_err(|_| anyhow::anyhow!("SSD write cancelled"))?;
                            exchange = receive
                                .recv()
                                .context("SSD write cancelled")?
                                .map_err(anyhow::Error::msg)?;
                            ensure!(exchange.len() == CHUNK, "SSD exchange size changed");
                            serial = serial.checked_add(1).context("SSD slice counter")?;
                            cached_page = page.map(|(logical, _)| logical);
                        }
                        let begin = page.map_or(0, |(_, begin)| begin);
                        buf.copy_from_slice(&exchange[begin..begin + buf.len()]);
                        Ok(())
                    })
                }))
                .unwrap_or_else(|_| {
                    Err(anyhow::anyhow!("SSD file worker panicked; reopen required"))
                });
                let _ = complete.send((store, result));
            })?;
        Ok(Self {
            id,
            requests: Some(requests),
            replies: Some(replies),
            done,
            handle: Some(handle),
            result: None,
            waiting: None,
        })
    }
    pub fn poll(&mut self) -> Result<Option<Request>> {
        ensure!(self.waiting.is_none(), "SSD source reply still outstanding");
        let Some(requests) = &self.requests else {
            return Ok(None);
        };
        match requests.try_recv() {
            Ok(r) => {
                ensure!(
                    r.job == self.id && r.bytes.len() == CHUNK && r.len <= CHUNK,
                    "SSD invalid source request"
                );
                self.waiting = Some(r.serial);
                Ok(Some(r))
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => Ok(None),
        }
    }
    pub fn respond(&mut self, request: Request, result: Result<()>) -> Result<()> {
        ensure!(
            request.job == self.id && self.waiting == Some(request.serial),
            "SSD stale/foreign reply"
        );
        let reply = result.map(|_| request.bytes).map_err(|e| format!("{e:#}"));
        self.replies
            .as_ref()
            .context("SSD write cancelled")?
            .try_send(reply)
            .map_err(|_| anyhow::anyhow!("SSD reply channel unavailable"))?;
        self.waiting = None;
        Ok(())
    }
    /// Signal cancellation without waiting for file I/O or a thread join. The
    /// caller retains the job until try_finish returns the exclusive catalog.
    pub fn cancel(&mut self) {
        self.requests = None;
        self.replies = None;
        self.waiting = None;
    }
    pub fn try_finish(&mut self) -> Result<Option<Done>> {
        ensure!(self.handle.is_some(), "SSD job already consumed");
        if self.result.is_none() {
            match self.done.try_recv() {
                Ok(done) => self.result = Some(done),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => anyhow::bail!("SSD worker lost completion"),
            }
        }
        if !self.handle.as_ref().unwrap().is_finished() {
            return Ok(None);
        }
        self.handle
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("SSD worker termination failed"))?;
        self.cancel();
        Ok(self.result.take())
    }
}
impl Drop for WriteJob {
    fn drop(&mut self) {
        // Closing channels wakes a source wait. A running file syscall may
        // finish later; never block the CUDA owner's destructor on that syscall.
        // The worker owns its catalog lock until it actually exits.
        self.cancel();
    }
}

fn pattern(part: Source, offset: u64, buf: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let code = Sha256::digest(format!("{part:?}").as_bytes())[0];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = ((offset + i as u64) % 251) as u8 ^ code;
    }
}
fn wait_request(job: &mut WriteJob) -> Result<Request> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(r) = job.poll()? {
            return Ok(r);
        }
        ensure!(
            std::time::Instant::now() < deadline,
            "SSD probe request deadline"
        );
        thread::yield_now();
    }
}
pub fn probe(root: &Path, rank: u32) -> Result<serde_json::Value> {
    ensure!(!root.exists(), "SSD worker probe needs new directory");
    let id = Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [44; 32], rank, 1024)?;
    let ids: Arc<[i32]> = (0..1088).map(|n| n % 53).collect::<Vec<_>>().into();
    let budget = 128 << 20;
    let store = Store::open(root, id.clone(), budget)?;
    let mut job = WriteJob::start(store, ids.clone())?;
    let mut r = wait_request(&mut job)?;
    // Hold the only exchange buffer on the owner: the worker cannot advance,
    // but owner progress and cancellation/finish polls remain nonblocking.
    for _ in 0..10_000 {
        ensure!(
            job.try_finish()?.is_none(),
            "worker advanced without source"
        );
    }
    ensure!(
        Store::open(root, id.clone(), budget).is_err(),
        "worker lost exclusive catalog"
    );
    let mut page_requests = usize::from(matches!(r.source, CopySource::Page(_)));
    r.fill(|part, offset, buf| {
        pattern(part, offset, buf);
        Ok(())
    })?;
    job.respond(r, Ok(()))?;
    let mut slices = 1;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut store = loop {
        if let Some(mut r) = job.poll()? {
            page_requests += usize::from(matches!(r.source, CopySource::Page(_)));
            r.fill(|part, offset, buf| {
                pattern(part, offset, buf);
                Ok(())
            })?;
            job.respond(r, Ok(()))?;
            slices += 1;
        }
        if let Some((store, result)) = job.try_finish()? {
            ensure!(result? == Outcome::Saved, "worker save outcome");
            break store;
        }
        ensure!(
            std::time::Instant::now() < deadline,
            "SSD probe finish deadline"
        );
        thread::yield_now();
    };
    ensure!(
        page_requests == 4,
        "expected one owner request per complete page"
    );
    let mut request = ids.to_vec();
    request.push(8);
    let key = store.lookup(&request)?.context("worker saved lookup")?;
    let mut expected = vec![0; CHUNK];
    let mut checked = 0;
    store.load(&key, &request, |part, offset, buf| {
        pattern(part, offset, &mut expected[..buf.len()]);
        ensure!(buf == &expected[..buf.len()], "worker changed source bytes");
        checked += buf.len();
        Ok(())
    })?;
    drop(expected);
    let mut cancelled = WriteJob::start(store, vec![9; 320].into())?;
    let held = wait_request(&mut cancelled)?;
    cancelled.cancel();
    drop(held);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some((mut store, result)) = cancelled.try_finish()? {
            ensure!(
                result.is_err() && store.lookup(&request).is_err(),
                "cancelled mutation must require reopen"
            );
            break;
        }
        ensure!(
            std::time::Instant::now() < deadline,
            "SSD cancellation deadline"
        );
        thread::yield_now();
    }
    let mut store = Store::open(root, id, budget)?;
    ensure!(
        store.lookup(&request)? == Some(key) && store.len() == 1,
        "worker cancellation recovery"
    );
    Ok(
        serde_json::json!({"rank":rank,"source_slices":slices,"page_requests":page_requests,"checked_payload_bytes":checked,
        "file_stage_bytes":CHUNK,"exchange_bytes":CHUNK,"channel_capacity":1,"owner_progress_polls":10000,
        "byte_exact":true,"exclusive_catalog":true,"cancel_waiting_source_reopens":true,
        "scope":"Bounded CPU file worker; GPU source pin/copy and collective scheduling integration are separate."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_worker_bytes_pause_and_cancel() {
        let root = std::env::temp_dir().join(format!(
            "alaska-worker-{}-{}",
            std::process::id(),
            NEXT_JOB.fetch_add(1, Ordering::Relaxed)
        ));
        probe(&root, 0).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
