//! Setup-only bounded buffers for retained four-rank prefill selection sharing.
use super::Engine;
use crate::cuda::Buffer;
pub(super) use crate::sparse_select::split_plan as plan;
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::cell::Cell;
#[derive(Clone, Copy, Default, serde::Serialize)]
struct Stats {
    calls: u64,
    selected_rows: u64,
    global_rows: u64,
    gathered_bytes: u64,
}
pub(super) struct Owner {
    pub(super) send: Buffer,
    pub(super) recv: Buffer,
    stats: Cell<Stats>,
}
pub(super) fn identity(enabled: bool) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"retained-select-split-4096p-256r-four-64grid-i64-v1");
    h.update([u8::from(enabled)]);
    h.finalize().into()
}
impl Owner {
    pub(super) fn from_env(e: &Engine<'_>) -> Result<(Option<Self>, [u8; 32])> {
        let parsed = match std::env::var("ALASKA_SELECT_SPLIT") {
            Ok(s) => plan::enabled(&s),
            Err(std::env::VarError::NotPresent) => Ok(false),
            Err(error) => Err(error).context("selection enablement UTF-8"),
        };
        let mut record = [0; 48];
        record[..8].copy_from_slice(b"ALSEL001");
        record[8] = u8::from(parsed.is_ok());
        if let Ok(on) = &parsed {
            record[9..41].copy_from_slice(&identity(*on));
        }
        let all = e.comm.exchange_host(&e.device, &e.stream, &record)?;
        ensure!(
            record[8] == 1 && all.iter().all(|r| r == &record),
            "selection split settings differ/invalid on a rank"
        );
        let hash = record[9..41].try_into().unwrap();
        let owner = if parsed? {
            (|| -> Result<Option<Self>> {
                Ok(Some(Self {
                    send: e.device.alloc(plan::SEND_BYTES)?,
                    recv: e.device.alloc(plan::RECV_BYTES)?,
                    stats: Cell::default(),
                }))
            })()
        } else {
            Ok(None)
        };
        record[41] = u8::from(owner.is_ok());
        let all = e.comm.exchange_host(&e.device, &e.stream, &record)?;
        ensure!(
            record[41] == 1 && all.iter().all(|r| r == &record),
            "selection split setup failed on a rank"
        );
        Ok((owner?, hash))
    }
    pub(super) fn gathered(&self, plan: plan::Plan, rank: usize) -> Result<()> {
        let mut stats = self.stats.get();
        stats.calls += 1;
        stats.selected_rows += plan.own(rank)?.len() as u64;
        stats.global_rows += plan.rows as u64;
        stats.gathered_bytes += (plan.stride * plan::WORLD) as u64;
        self.stats.set(stats);
        Ok(())
    }
    pub(super) fn stats(&self) -> serde_json::Value {
        serde_json::json!(self.stats.get())
    }
}
