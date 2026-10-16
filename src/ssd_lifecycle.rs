//! Bounded off-owner catalog open/scan and common-order application.
use crate::{
    ssd_format::{Hash, Identity},
    ssd_store::Store,
};
use anyhow::{Context, Result, ensure};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread::{self, JoinHandle},
};

pub const MAX_KEYS: usize = 4096;
pub struct CatalogJob {
    inventory: Receiver<Vec<Hash>>,
    apply: Option<SyncSender<Vec<Hash>>>,
    done: Receiver<Result<Store>>,
    handle: Option<JoinHandle<()>>,
    result: Option<Result<Store>>,
    applied: bool,
}
impl CatalogJob {
    pub fn open(root: PathBuf, id: Identity, budget: u64, previous: Option<Store>) -> Result<Self> {
        let (inventory_tx, inventory) = mpsc::sync_channel(1);
        let (apply, commands) = mpsc::sync_channel::<Vec<Hash>>(1);
        let (complete, done) = mpsc::sync_channel(1);
        let handle = thread::Builder::new()
            .name("alaska-ssd-catalog".into())
            .spawn(move || {
                crate::cpu_pin::helper_thread();
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Store> {
                        // Close the old catalog and recover its directory only on this worker.
                        drop(previous);
                        let mut store = Store::open(&root, id, budget)?;
                        let order = store.order();
                        ensure!(order.len() <= MAX_KEYS, "SSD catalog inventory bound");
                        inventory_tx
                            .send(order)
                            .context("SSD catalog inventory cancelled")?;
                        let keep = commands
                            .recv()
                            .context("SSD catalog cancelled before reconciliation")?;
                        ensure!(keep.len() <= MAX_KEYS, "SSD catalog intersection bound");
                        store.retain_order(&keep)?;
                        Ok(store)
                    }))
                    .unwrap_or_else(|_| {
                        Err(anyhow::anyhow!(
                            "SSD catalog worker panicked; reopen required"
                        ))
                    });
                let _ = complete.send(result);
            })?;
        Ok(Self {
            inventory,
            apply: Some(apply),
            done,
            handle: Some(handle),
            result: None,
            applied: false,
        })
    }
    pub fn inventory(&mut self) -> Result<Option<Vec<Hash>>> {
        match self.inventory.try_recv() {
            Ok(order) => {
                ensure!(order.len() <= MAX_KEYS, "SSD inventory protocol bound");
                Ok(Some(order))
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => Ok(None),
        }
    }
    pub fn apply(&mut self, keep: Vec<Hash>) -> Result<()> {
        ensure!(
            !self.applied && keep.len() <= MAX_KEYS,
            "SSD catalog repeated/oversized apply"
        );
        self.apply
            .take()
            .context("SSD catalog cancelled")?
            .try_send(keep)
            .map_err(|_| anyhow::anyhow!("SSD catalog apply unavailable"))?;
        self.applied = true;
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.apply = None;
    }
    pub fn try_finish(&mut self) -> Result<Option<Result<Store>>> {
        ensure!(self.handle.is_some(), "SSD catalog already consumed");
        if self.result.is_none() {
            match self.done.try_recv() {
                Ok(result) => self.result = Some(result),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => anyhow::bail!("SSD catalog lost completion"),
            }
        }
        if !self.handle.as_ref().unwrap().is_finished() {
            return Ok(None);
        }
        self.handle
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("SSD catalog termination failed"))?;
        self.cancel();
        Ok(self.result.take())
    }
}
impl Drop for CatalogJob {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[path = "ssd_lifecycle_tests.rs"]
mod tests;
pub use tests::probe;
