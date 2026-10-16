//! One file worker with its own bounded CUDA stream/stage. No model-owner
//! round trip per page; immutable source pins stay owned by the controller.
use super::*;
use crate::{copy_gate::Gate, cuda::CopyStream};
use std::{
    sync::{
        Arc,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread::{self, JoinHandle},
};

type Done = (Store, Result<Outcome>, CopyStream);
pub(super) struct CopyJob {
    pub(super) gate: Arc<Gate>,
    done: Receiver<Done>,
    handle: Option<JoinHandle<()>>,
    result: Option<Done>,
}
impl CopyJob {
    pub(super) fn start(
        mut store: Store,
        tokens: Arc<[i32]>,
        mut plan: super::super::super::store::CopyPlan,
        mut stream: CopyStream,
        gate: Arc<Gate>,
    ) -> Result<Self> {
        let (complete, done) = mpsc::sync_channel(1);
        let worker_gate = gate.clone();
        let handle = thread::Builder::new()
            .name("alaska-ssd-copy".into())
            .spawn(move || {
                crate::cpu_pin::helper_thread();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    store.save(&tokens, false, |part, offset, bytes| {
                        plan.fill(&mut stream, &worker_gate, part, offset, bytes)
                    })
                }))
                .unwrap_or_else(|_| {
                    worker_gate
                        .fatal
                        .store(true, std::sync::atomic::Ordering::Release);
                    Err(anyhow::anyhow!("SSD copy worker panicked"))
                });
                if let Err(error) = &result {
                    eprintln!("SSD independent worker result: {error:#}");
                }
                let _ = complete.send((store, result, stream));
            })?;
        Ok(Self {
            gate,
            done,
            handle: Some(handle),
            result: None,
        })
    }
    pub(super) fn cancel(&self) {
        self.gate.cancel();
    }
    pub(super) fn try_finish(&mut self) -> Result<Option<Done>> {
        ensure!(self.handle.is_some(), "SSD copy job already consumed");
        if self.result.is_none() {
            match self.done.try_recv() {
                Ok(done) => self.result = Some(done),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => anyhow::bail!("SSD copy worker lost completion"),
            }
        }
        if !self.handle.as_ref().unwrap().is_finished() {
            return Ok(None);
        }
        self.handle
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("SSD copy worker termination failed"))?;
        Ok(self.result.take())
    }
}
impl Drop for CopyJob {
    fn drop(&mut self) {
        // Wait only for admitted DMA. A blocked file syscall retains its lock
        // and owned bytes; cancellation permanently forbids further source access.
        self.cancel();
    }
}
