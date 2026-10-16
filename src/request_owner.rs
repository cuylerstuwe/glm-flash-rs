//! Blocking CPU ingress/wakeup loop around the single CUDA owner. Followers
//! wait for the head's next collective, bounded by a one-second idle heartbeat.
use super::*;
use crate::request_broker::{Owner, Receipt};
impl Scheduler {
    pub(super) fn wait(&mut self, now: Duration) -> Result<Duration> {
        let p = self.queue.gate.snapshot(now, self.queue.foreground())?;
        if self.last_priority != Some((p.revision, p.hold)) {
            return Ok(Duration::ZERO);
        }
        Ok(self
            .queue
            .wait(now, p.hold)
            .min(self.queue.gate.wait(now)?)
            .min(
                self.disk
                    .as_ref()
                    .map_or(Duration::from_secs(1), |d| d.wait(now, p.hold)),
            )
            .min(self.grammar_wait()))
    }
    pub(super) fn serve(&mut self, long: &mut Long<'_>, owner: Option<&Owner>) -> Result<u64> {
        let result = self.serve_inner(long, owner);
        if let Err(e) = &result {
            long.healthy = false;
            if let Some(owner) = owner {
                owner.fail(&e.to_string());
            }
        }
        result
    }
    fn serve_inner(&mut self, long: &mut Long<'_>, owner: Option<&Owner>) -> Result<u64> {
        ensure!(
            (long.engine.rank == 0) == owner.is_some(),
            "ingress owner/rank mismatch"
        );
        if std::env::var("ALASKA_SSD_READ_WAKE").as_deref() == Ok("1")
            && let (Some(disk), Some(owner)) = (&mut self.disk, owner)
        {
            disk.controller.set_file_wake(owner.file_wake());
        }
        let mut ticks = 0;
        let mut next_telemetry = Duration::ZERO;
        loop {
            let action = if let Some(owner) = owner {
                Some(owner.next(self.wait(owner.now())?))
            } else {
                None
            };
            if let (Some(owner), Some(action)) = (owner, action.as_ref()) {
                owner.image_boundary(action)?;
            }
            let head = action
                .as_ref()
                .map(|a| (a.command(), owner.unwrap().now(), a.control_only()));
            let report = self.pump(long, head)?;
            if let Some(owner) = owner {
                owner.image_hold(report.held)?;
            }
            ticks += 1;
            if let Some(owner) = owner {
                // This never launches model work: all control actions had their
                // common control-only bit checked by the wire protocol.
                ensure!(
                    action.as_ref().is_none_or(|a| !a.control_only()) || report.worked.is_none(),
                    "control receipt delayed by unrelated model work"
                );
                if let Some(key) = report.deferred {
                    owner.defer_submission(action.unwrap(), key)?;
                } else {
                    owner.applied(
                        action.unwrap(),
                        report.prepared,
                        report.submitted,
                        Receipt {
                            applied: report.applied,
                            held: report.held,
                        },
                        report.rejected.clone(),
                    )?;
                }
                if let Some(receipt) = &report.grammar_receipt {
                    owner.grammar_applied(
                        receipt.key,
                        receipt.submitted,
                        receipt.rejected.clone(),
                        receipt.invalid_grammar,
                    )?;
                }
                if let Some(id) = report.admitted {
                    let cached = if let Ok(r) = self.queue.request(id) {
                        r.cached_tokens
                    } else {
                        self.queue
                            .completed
                            .iter()
                            .find(|c| c.status.id == id)
                            .context("missing cache admission")?
                            .status
                            .cached_tokens
                    };
                    owner.cached(id, cached)?;
                }
                if !report.batch_output.is_empty() {
                    for &(id, emitted, reason) in &report.batch_output {
                        let outputs = if reason.is_some() {
                            &self
                                .queue
                                .completed
                                .iter()
                                .find(|c| c.status.id == id)
                                .context("multi completed owner")?
                                .outputs
                        } else {
                            &self.queue.request(id)?.outputs
                        };
                        owner.publish_many(
                            id,
                            &outputs[outputs
                                .len()
                                .checked_sub(emitted)
                                .context("multi output range")?..],
                            reason,
                        )?;
                    }
                } else if let Some(id) = report.completed {
                    let c = self
                        .queue
                        .completed
                        .iter()
                        .find(|c| c.status.id == id)
                        .context("missing completed owner")?;
                    owner.publish_many(
                        id,
                        if report.worked == Some(id) {
                            &c.outputs[c
                                .outputs
                                .len()
                                .checked_sub(report.emitted)
                                .context("completed output range")?..]
                        } else {
                            &[]
                        },
                        Some(c.reason),
                    )?;
                } else if let Some(id) = report.worked {
                    let r = self.queue.request(id)?;
                    if r.done == r.tokens.len() {
                        owner.publish_many(
                            id,
                            &r.outputs[r
                                .outputs
                                .len()
                                .checked_sub(report.emitted)
                                .context("active output range")?..],
                            None,
                        )?;
                    }
                }
            }
            if let Some(owner) = owner
                && (self.stamp >= next_telemetry
                    || report.completed.is_some()
                    || report.admitted.is_some()
                    || report.prepared.is_some()
                    || report.stopped
                    || report.worked.is_none())
            {
                owner.telemetry(serde_json::json!({"observed_ns":self.stamp.as_nanos(),
                    "held":report.held,"applied":report.applied,"queue":self.queue.telemetry(self.stamp),
                    "disk":self.disk.as_ref().map(|d|d.telemetry()),"grammar":self.grammar_telemetry(),
                    "l2_prefetch":long.engine.prefetch_stats(),"select_split":long.split_receipt()}));
                next_telemetry = self.stamp + Duration::from_millis(100);
            }
            if report.stopped {
                return Ok(ticks);
            }
            if report.grammar_receipt.is_none() && self.grammar_pending() {
                std::thread::sleep(Duration::from_millis(1));
            }
            if let Some(owner) = owner
                && (self.disk.as_ref().is_some_and(|d| d.stopping)
                    || self.grammar.as_ref().is_some_and(|g| g.stopping))
            {
                // Broker shutdown returns Stop immediately. Pace only this drain
                // path; all clients are already retired and admission is closed.
                std::thread::sleep(self.wait(owner.now())?);
            }
        }
    }
}
#[path = "request_owner_tests.rs"]
mod tests;
pub use tests::run;
