//! Persistent cold-request scheduler. One bounded command and one model call per
//! tick; rank zero alone chooses command/time, all ranks verify ownership.
use super::{Engine, Long, ROWS, priority::Boundary, store::Control};
use crate::{
    cuda::{Buffer, Pinned},
    output_decode::Table,
    prepared::Prepared,
    request_queue::{Completion, End, Queue},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
const CHUNK: usize = 256 << 10;
use crate::request_broker::Command;
#[derive(Default)]
struct Trace {
    prefill: Option<String>,
    logits: Vec<String>,
    final_state: Option<String>,
}
impl Trace {
    fn value(&self, outputs: &[i32]) -> serde_json::Value {
        serde_json::json!({"prefill_state":self.prefill,"final_state":self.final_state,"logits":self.logits,"outputs":outputs})
    }
}
#[derive(Default, serde::Serialize)]
pub(super) struct Report {
    pub submitted: Option<u64>,
    pub rejected: Option<String>,
    pub completed: Option<u64>,
    pub worked: Option<u64>,
    pub emitted: usize,
    pub admitted: Option<u64>,
    pub prepared: Option<u64>,
    pub applied: u64,
    pub held: bool,
    pub stopped: bool,
    pub deferred: Option<u64>,
    pub grammar_receipt: Option<grammar::Receipt>,
    pub batch_output: Vec<(u64, usize, Option<End>)>,
}
struct Wire {
    control: Control,
    send: Buffer,
    recv: Buffer,
    pinned: Pinned,
    serial: u64,
}
impl Wire {
    fn new(long: &Long<'_>) -> Result<Self> {
        let d = &long.engine.device;
        Ok(Self {
            control: Control::new(d)?,
            send: d.alloc(CHUNK)?,
            recv: d.alloc(CHUNK * 4)?,
            pinned: Pinned::new(d, CHUNK * 2)?,
            serial: 0,
        })
    }
    fn receive(
        &mut self,
        long: &Long<'_>,
        head: Option<(Command, Duration, bool)>,
    ) -> Result<(Command, Duration, bool)> {
        let valid = (long.engine.rank == 0) == head.is_some();
        let mut record = [0; 64];
        record[..8].copy_from_slice(b"ALREQ009");
        record[8..16].copy_from_slice(&self.serial.to_le_bytes());
        record[18] = u8::from(valid);
        let mut payload = Vec::new();
        if let Some((command, now, control_only)) = head {
            record[19] = u8::from(control_only);
            record[24..32].copy_from_slice(
                &u64::try_from(now.as_nanos())
                    .context("owner clock overflow")?
                    .to_le_bytes(),
            );
            match command {
                Command::Tick => {}
                Command::Prepare { background } => {
                    record[16] = 5;
                    record[17] = u8::from(background);
                }
                Command::Release(ticket) => {
                    record[16] = 6;
                    record[32..40].copy_from_slice(&ticket.to_le_bytes());
                }
                Command::Stop => {
                    record[16] = 7;
                }
                Command::CancelGrammar(key) => {
                    record[16] = 9;
                    record[32..40].copy_from_slice(&key.to_le_bytes());
                }
                Command::Cancel(id) => {
                    record[16] = 2;
                    record[32..40].copy_from_slice(&id.to_le_bytes());
                }
                ref submit @ (Command::Submit { .. } | Command::SubmitGrammar { .. }) => {
                    let (tokens, output, background, deadline, policy, key) = match submit {
                        Command::Submit {
                            tokens,
                            output,
                            background,
                            deadline,
                            policy,
                        } => (tokens, *output, *background, *deadline, policy, None),
                        Command::SubmitGrammar {
                            key,
                            tokens,
                            output,
                            background,
                            deadline,
                            policy,
                        } => (tokens, *output, *background, *deadline, policy, Some(*key)),
                        _ => unreachable!(),
                    };
                    record[16] = if key.is_some() { 8 } else { 1 };
                    record[32..40].copy_from_slice(&key.unwrap_or(0).to_le_bytes());
                    record[17] = u8::from(background);
                    record[40..48].copy_from_slice(&(output as u64).to_le_bytes());
                    if tokens.len() <= crate::cache_pages::CONTEXT
                        && deadline.as_nanos() <= u64::MAX as u128
                    {
                        record[48..56].copy_from_slice(&(deadline.as_nanos() as u64).to_le_bytes());
                        payload = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
                        if let Some(policy) = policy {
                            let bytes = policy.encode();
                            record[20..24].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
                            payload.extend_from_slice(&bytes);
                        }
                    } else {
                        record[16] = 4;
                    }
                }
                Command::Lease {
                    owner,
                    epoch,
                    foreground,
                    seconds,
                } => {
                    record[16] = 3;
                    record[17] = u8::from(foreground);
                    record[32..40].copy_from_slice(&epoch.to_le_bytes());
                    record[40..48].copy_from_slice(&seconds.to_bits().to_le_bytes());
                    if owner.len() <= 640 {
                        payload = owner.into_bytes();
                    } else {
                        record[16] = 4;
                    }
                }
            }
            record[56..64].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        }
        let all = self.control.exchange(long, &record)?;
        ensure!(
            valid && all.iter().all(|r| r[..16] == record[..16] && r[18] == 1),
            "request command sequence/leader mismatch"
        );
        let head = all[0];
        ensure!(
            self.control.agree(
                long,
                810,
                Sha256::digest(head).into(),
                long.engine.rank != 0 || head == record
            )?,
            "request header changed in transport"
        );
        ensure!(
            head[16] <= 9 && head[17] <= 1 && head[19] <= 1,
            "request command kind/flag"
        );
        let n = u64::from_le_bytes(head[56..64].try_into().unwrap()) as usize;
        ensure!(
            n <= crate::cache_pages::CONTEXT * 4 + crate::output_policy::WIRE_BYTES,
            "request payload limit"
        );
        if long.engine.rank != 0 {
            payload.resize(n, 0);
        }
        ensure!(payload.len() == n, "request payload size disagreement");
        let original: [u8; 32] = Sha256::digest(&payload).into();
        for start in (0..n).step_by(CHUNK) {
            let count = (n - start).min(CHUNK);
            let stride = count.next_multiple_of(4);
            self.pinned.bytes_mut()[..stride].fill(0);
            if long.engine.rank == 0 {
                self.pinned.bytes_mut()[..count].copy_from_slice(&payload[start..start + count]);
            }
            unsafe {
                self.pinned
                    .upload_to(&long.engine.stream, &self.send, 0, 0, stride)?;
                long.engine.gather_prefix(&self.send, &self.recv, stride)?;
                self.pinned
                    .download_from(&long.engine.stream, CHUNK, &self.recv, 0, count)?;
            }
            long.engine.sync()?;
            payload[start..start + count]
                .copy_from_slice(&self.pinned.bytes_mut()[CHUNK..CHUNK + count]);
        }
        // The header (agreed above) fixes the length; an empty payload has
        // nothing to verify.
        if n > 0 {
            let received: [u8; 32] = Sha256::digest(&payload).into();
            ensure!(
                self.control.agree(
                    long,
                    800,
                    received,
                    long.engine.rank != 0 || received == original
                )?,
                "request payload mismatch"
            );
        }
        self.serial = self
            .serial
            .checked_add(1)
            .context("request protocol exhausted")?;
        let a = u64::from_le_bytes(head[32..40].try_into().unwrap());
        let b = u64::from_le_bytes(head[40..48].try_into().unwrap());
        let policy_bytes = u32::from_le_bytes(head[20..24].try_into().unwrap()) as usize;
        ensure!(
            policy_bytes <= n
                && policy_bytes <= crate::output_policy::WIRE_BYTES
                && (matches!(head[16], 1 | 8) || policy_bytes == 0),
            "generation payload bounds"
        );
        let command = match head[16] {
            0 => Command::Tick,
            2 => Command::Cancel(a),
            9 => Command::CancelGrammar(a),
            5 => Command::Prepare {
                background: head[17] != 0,
            },
            6 => Command::Release(a),
            7 => Command::Stop,
            1 | 8 => {
                let token_bytes = n - policy_bytes;
                ensure!(token_bytes.is_multiple_of(4), "token byte alignment");
                let tokens: Arc<[i32]> = payload[..token_bytes]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|t| i32::from_le_bytes(*t))
                    .collect::<Vec<_>>()
                    .into();
                let output = usize::try_from(b)?;
                let background = head[17] != 0;
                let deadline =
                    Duration::from_nanos(u64::from_le_bytes(head[48..56].try_into().unwrap()));
                let policy = if policy_bytes == 0 {
                    None
                } else {
                    Some(crate::output_policy::Policy::decode(
                        &payload[token_bytes..],
                    )?)
                };
                if head[16] == 8 {
                    Command::SubmitGrammar {
                        key: a,
                        tokens,
                        output,
                        background,
                        deadline,
                        policy,
                    }
                } else {
                    Command::Submit {
                        tokens,
                        output,
                        background,
                        deadline,
                        policy,
                    }
                }
            }
            3 => Command::Lease {
                owner: String::from_utf8(payload)?,
                epoch: a,
                foreground: head[17] != 0,
                seconds: f64::from_bits(b),
            },
            // A rejected input still traverses the common command protocol.
            _ => Command::Submit {
                tokens: Arc::from([]),
                output: 0,
                background: true,
                deadline: Duration::ZERO,
                policy: None,
            },
        };
        Ok((
            command,
            Duration::from_nanos(u64::from_le_bytes(head[24..32].try_into().unwrap())),
            head[19] != 0,
        ))
    }
}
struct Scheduler {
    queue: Queue,
    wire: Wire,
    boundary: Boundary,
    last_priority: Option<(u64, bool)>,
    audit: bool,
    final_prefill_head_only: bool,
    exact_prompt_batch: bool,
    multi_greedy: u8,
    traces: BTreeMap<u64, Trace>,
    stamp: Duration,
    output_table: Option<Arc<Table>>,
    output_streams: BTreeMap<u64, crate::output::Stream>,
    sampler: Option<crate::candidates::Candidates>,
    cache: Option<super::store::Store>,
    disk: Option<disk::RequestDisk>,
    speculation: Option<speculation::State>,
    grammar: Option<grammar::State>,
    /// Chat role token IDs that end the system prompt (snapshot marks).
    share_roles: Vec<i32>,
    /// Rank 0's broker: requests' image rows submitted with their prompts.
    broker: Option<crate::request_broker::Broker>,
    /// Image rows of requests still prefilling, on every rank's device.
    vision_tables: BTreeMap<u64, (std::rc::Rc<[i32]>, crate::cuda::Buffer)>,
}
impl Scheduler {
    fn new(long: &Long<'_>, audit: bool) -> Result<Self> {
        Ok(Self {
            queue: Queue::default(),
            wire: Wire::new(long)?,
            boundary: Boundary::new(long)?,
            last_priority: None,
            audit,
            final_prefill_head_only: false,
            exact_prompt_batch: false,
            multi_greedy: 0,
            traces: BTreeMap::new(),
            stamp: Duration::ZERO,
            output_table: None,
            output_streams: BTreeMap::new(),
            sampler: None,
            cache: None,
            disk: None,
            speculation: None,
            grammar: None,
            share_roles: Vec::new(),
            broker: None,
            vision_tables: BTreeMap::new(),
        })
    }
    fn install_sampler(&mut self, long: &Long<'_>, native: &Path) -> Result<()> {
        ensure!(
            self.queue.count() == 0 && self.sampler.is_none(),
            "sampler installation after admission"
        );
        let sampler = (|| {
            ensure!(
                format!(
                    "{:x}",
                    Sha256::digest(std::fs::read("/lib/aarch64-linux-gnu/libm.so.6")?)
                ) == "5e0391562de36de323fb9cb0f2c1b6bd6d034c840b1dac018499f5fe4ded1b4a",
                "sampler host libm drift"
            );
            crate::candidates::Candidates::new(&long.engine.device, native)
        })();
        ensure!(
            self.wire.control.agree(
                long,
                821,
                Sha256::digest(b"request-sampler-v1/ALSAM001/finite-bf16").into(),
                sampler.is_ok()
            )?,
            "sampler initialization failed on a rank"
        );
        self.sampler = Some(sampler?);
        Ok(())
    }
    fn install_output_table(&mut self, long: &Long<'_>, table: Arc<Table>) -> Result<()> {
        ensure!(
            self.queue.count() == 0
                && self.output_table.is_none()
                && self.output_streams.is_empty(),
            "output table installation after admission"
        );
        ensure!(
            self.wire.control.agree(long, 820, table.identity(), true)?,
            "output decoder identity differs"
        );
        self.output_table = Some(table);
        Ok(())
    }
    fn tick(&mut self, long: &mut Long<'_>, head: Option<(Command, Duration)>) -> Result<Report> {
        self.pump(long, head.map(|(c, t)| (c, t, false)))
    }
    fn pump(
        &mut self,
        long: &mut Long<'_>,
        head: Option<(Command, Duration, bool)>,
    ) -> Result<Report> {
        long.ready()?;
        let result = self.tick_inner(long, head);
        if result.is_err() {
            long.healthy = false;
        }
        result
    }
    fn tick_inner(
        &mut self,
        long: &mut Long<'_>,
        head: Option<(Command, Duration, bool)>,
    ) -> Result<Report> {
        let began = Instant::now();
        crate::round_timing::enter(crate::round_timing::Phase::Receive);
        let (command, now, control_only) = self.wire.receive(long, head)?;
        crate::round_timing::enter(crate::round_timing::Phase::Control);
        self.queue.now(now)?;
        self.stamp = now;
        let mut report = Report::default();
        match command {
            Command::Tick => {}
            Command::Prepare { background } => {
                if !background {
                    match self.queue.gate.enter(now) {
                        Ok(ticket) => report.prepared = Some(ticket),
                        Err(e) => report.rejected = Some(e.to_string()),
                    }
                }
            }
            Command::Release(ticket) => {
                if let Err(e) = self.queue.gate.leave(ticket, now) {
                    report.rejected = Some(e.to_string());
                }
            }
            Command::Stop => {
                if self.queue.count() == 0
                    && self.queue.gate.local_count() == 0
                    && !self.grammar_pending()
                {
                    report.stopped = true;
                } else {
                    report.rejected = Some("owner still has requests/preparations".into());
                }
            }

            Command::Cancel(id) => {
                if !self.queue.cancel(id) {
                    report.rejected = Some("unknown request".into());
                }
            }
            Command::CancelGrammar(key) => {
                self.grammar_cancel(long, key)?;
            }
            Command::SubmitGrammar {
                key,
                tokens,
                output,
                background,
                deadline,
                policy,
            } => {
                if self.grammar_stage(
                    long,
                    key,
                    grammar::Submission {
                        tokens,
                        output,
                        background,
                        deadline,
                        policy,
                    },
                )? {
                    report.deferred = Some(key);
                } else {
                    report.rejected = Some("native grammar preparation unavailable".into());
                }
            }
            Command::Submit {
                tokens,
                output,
                background,
                deadline,
                policy,
            } => {
                if self.audit && (tokens.len() > 16384 || output > 64) {
                    report.rejected = Some("diagnostic request limit".into());
                } else {
                    let stream = (|| -> Result<Option<crate::output::Stream>> {
                        ensure!(
                            policy.as_ref().is_none_or(|p| p.grammar().is_none()),
                            "grammar requests require the prepared grammar serving path"
                        );
                        ensure!(
                            policy
                                .as_ref()
                                .and_then(|p| p.sampling())
                                .is_none_or(|s| s.greedy())
                                || self.sampler.is_some(),
                            "stochastic sampler unavailable"
                        );
                        if let Some(p) = policy.as_ref().filter(|p| p.has_stops()) {
                            let table = self
                                .output_table
                                .clone()
                                .context("string-stop decoder unavailable")?;
                            Ok(Some(
                                crate::output::Stream::new(
                                    table,
                                    p.thinking(),
                                    p.tools(),
                                    &p.stops(),
                                    output,
                                )?
                                .with_ignore_eos(p.ignore_eos()),
                            ))
                        } else {
                            Ok(None)
                        }
                    })();
                    match stream.and_then(|stream| {
                        let id = self.queue.submit_policy(
                            tokens,
                            output,
                            background,
                            deadline,
                            now,
                            policy.as_ref(),
                        )?;
                        if let Some(stream) = stream {
                            self.output_streams.insert(id, stream);
                        }
                        Ok(id)
                    }) {
                        Ok(id) => report.submitted = Some(id),
                        Err(e) => report.rejected = Some(e.to_string()),
                    }
                }
            }
            Command::Lease {
                owner,
                epoch,
                foreground,
                seconds,
            } => {
                if let Err(e) = self
                    .queue
                    .gate
                    .update(&owner, epoch, foreground, seconds, now)
                {
                    report.rejected = Some(e.to_string());
                }
            }
        }
        report.grammar_receipt = self.grammar_poll(long, now)?;
        let cancellation = self.queue.cancellation(now)?;
        let mut ownership = Sha256::new();
        ownership.update(self.queue.digest());
        ownership.update(self.disk.as_ref().map_or([0; 32], |d| d.digest()));
        ownership.update(self.grammar_digest());
        ensure!(
            self.wire
                .control
                .agree(long, 801, ownership.finalize().into(), true)?,
            "request ownership disagreement"
        );
        if let Some((id, _)) =
            cancellation.filter(|(id, _)| self.queue.request(*id).is_ok_and(|r| r.restoring))
        {
            self.disk
                .as_mut()
                .context("restoring request lost controller")?
                .cancel_restore(id)?;
            self.queue.forward_restore_cancel(id)?;
        } else if let Some((id, reason)) = cancellation {
            if let Some(slot) = self.queue.request(id)?.slot {
                long.bind(slot)?;
                let released = long.cancel();
                long.agree_speculation_work(
                    &self.wire.control,
                    802,
                    self.queue.digest(),
                    released,
                )?;
            }
            self.queue.finish(id, reason, now)?;
            self.output_streams.remove(&id);
            self.grammar_remove(id);
            if let Some(speculation) = &mut self.speculation {
                speculation.remove(id);
            }
            report.completed = Some(id);
        }
        let foreground = self.queue.foreground();
        let priority = self.queue.gate.snapshot(now, foreground)?;
        let mut priority_digest = Sha256::new();
        priority_digest.update(self.queue.gate.digest());
        priority_digest.update(priority.revision.to_le_bytes());
        priority_digest.update([u8::from(priority.hold)]);
        priority_digest.update(self.last_priority.map_or(u64::MAX, |p| p.0).to_le_bytes());
        priority_digest.update([self.last_priority.map_or(2, |p| u8::from(p.1))]);
        ensure!(
            self.wire
                .control
                .agree(long, 809, priority_digest.finalize().into(), true)?,
            "request priority revision disagreed"
        );
        if self.last_priority != Some((priority.revision, priority.hold)) {
            self.grammar_drain_masks(long)?;
            let applied = self
                .boundary
                .apply(long, (long.engine.rank == 0).then_some(priority))?;
            self.queue.gate.acknowledge(applied)?;
            self.last_priority = Some((priority.revision, priority.hold));
        }
        self.queue.hold(long.priority_hold, now)?;
        report.applied = self.queue.gate.applied();
        report.held = long.priority_hold;
        let mut maintenance = Sha256::new();
        maintenance.update(self.queue.digest());
        maintenance.update(self.disk.as_ref().map_or([0; 32], |d| d.digest()));
        maintenance.update(self.cache.as_ref().map_or([0; 32], |s| s.digest()));
        ensure!(
            self.wire
                .control
                .agree(long, 828, maintenance.finalize().into(), true)?,
            "request SSD ownership disagreement"
        );
        if report.stopped {
            // A shutdown receipt may precede worker retirement. Never drop or
            // replace a worker/catalog because cancellation was merely requested.
            report.stopped = self.disk_step(long, true, &mut report)? && self.grammar_stop(long)?;
            return Ok(report);
        }
        if report.completed.is_some() || control_only {
            return Ok(report);
        }
        crate::round_timing::enter(crate::round_timing::Phase::Disk);
        self.disk_step(long, false, &mut report)?;
        crate::round_timing::enter(crate::round_timing::Phase::Admit);
        if report.admitted.is_some() || report.completed.is_some() {
            return Ok(report);
        }
        if self.queue.free_slot().is_some()
            && let Some(id) = self.queue.queued(long.priority_hold)
            && !self.share_wait(long, id)?
        {
            let choice = self.cache_choice(long, id);
            ensure!(
                self.wire.control.agree(
                    long,
                    826,
                    choice.as_ref().map_or([0; 32], |c| c.digest(self)),
                    choice.is_ok()
                )?,
                "request cache selection failed on a rank"
            );
            let choice = choice?;
            let slot = choice.slot;
            let request = self.queue.request(id)?;
            long.bind(slot)?;
            let admission = long.admission(
                request.tokens.len(),
                request.output,
                choice.keep(long)?,
                request.background,
            );
            let digest = admission.as_ref().map_or([0; 32], |a| a.digest());
            if self
                .wire
                .control
                .agree(long, 803, digest, admission.is_ok())?
            {
                if matches!(choice.source, cache::Source::Disk) {
                    if self.start_restore(long, id, slot)? {
                        return Ok(report);
                    }
                    // Memory/staging unavailable before any reservation: cold
                    // admission uses the same final idle destination.
                    let r = self.queue.request(id)?;
                    let cold = long.start_mode(r.tokens.len(), r.output, r.background);
                    long.agree_speculation_work(&self.wire.control, 834, digest, cold)?;
                    self.queue.admit_cached(id, slot, 0, now)?;
                    report.admitted = Some(id);
                    return Ok(report);
                }
                let admitted = match choice.source {
                    cache::Source::Cold => long
                        .start_mode(request.tokens.len(), request.output, request.background)
                        .map(|()| 0),
                    cache::Source::Live => {
                        long.continue_request(&request.tokens, request.output, request.background)
                    }
                    cache::Source::Ram(entry) => self.cache.as_mut().unwrap().restore(
                        long,
                        entry,
                        &request.tokens,
                        request.output,
                        request.background,
                    ),
                    cache::Source::Disk => unreachable!(),
                };
                let admitted =
                    long.agree_speculation_work(&self.wire.control, 804, digest, admitted)?;
                ensure!(admitted == choice.cached, "cache admission cursor changed");
                self.queue.admit_cached(id, slot, choice.cached, now)?;
                report.admitted = Some(id);
            } else {
                self.queue.finish(id, End::Rejected, now)?;
                self.output_streams.remove(&id);
                self.grammar_remove(id);
                if let Some(speculation) = &mut self.speculation {
                    speculation.remove(id);
                }
                report.rejected = Some("model page reservation unavailable".into());
                report.completed = Some(id);
                return Ok(report);
            }
        }
        if self.multi_greedy >= 2
            && !self.audit
            && let Some(mut ids) = self.multi_spec_ready(match self.multi_greedy {
                3 => 8,
                4 => 3,
                5 => 2,
                _ => 4,
            })
        {
            if self.multi_greedy == 6 {
                let turn = self.queue.runnable().context("multi cohort turn")?;
                ensure!(
                    ids.contains(&turn),
                    "multi cohort skipped its scheduled owner"
                );
            }
            let window = match self.multi_greedy {
                3 => 8,
                4 => 3,
                5 => 2,
                _ => 4,
            };
            if self.multi_greedy == 3 {
                ids = vec![
                    self.queue.runnable().context("multi fair first")?,
                    self.queue.runnable().context("multi fair second")?,
                ];
                ids.sort_by_key(|id| self.queue.request(*id).unwrap().slot);
            }
            if !ids.is_empty() {
                return self.pump_multi_spec(long, ids, window, now, began, report);
            }
        }
        if self.multi_greedy == 1
            && !self.audit
            && let Some(ids) = self.multi_greedy_ready()
        {
            return self.pump_multi_greedy(long, ids, now, began, report);
        }
        let Some(id) = self.queue.runnable() else {
            return Ok(report);
        };
        let r = self.queue.request(id)?;
        if r.has_grammar() && !self.grammar_mask_ready(long, id)? {
            return Ok(report);
        }
        let r = self.queue.request(id)?;
        let slot = r.slot.unwrap();
        crate::round_timing::enter(crate::round_timing::Phase::Dispatch);
        let mut action = Sha256::new();
        action.update(self.queue.digest());
        action.update(id.to_le_bytes());
        action.update((slot as u64).to_le_bytes());
        action.update((long.prefill_rows as u64).to_le_bytes());
        // Cache and mixed-prefix metadata controls whether ranks enter further
        // collectives. Agree it before any rank can take a different work path.
        let session = &long.sessions[slot];
        let phase = long.memory.book().slot(slot)?.phase();
        action.update([phase as u8, u8::from(long.priority_hold)]);
        action.update([u8::from(session.mixed), u8::from(session.background)]);
        for n in [session.tokens.len(), session.prompt_end, session.total_end] {
            action.update((n as u64).to_le_bytes());
        }
        action.update(session.pending.unwrap_or(-1).to_le_bytes());
        action.update(self.cache.as_ref().map_or([0; 32], |s| s.digest()));
        ensure!(
            self.wire.control.agree(
                long,
                808,
                action.finalize().into(),
                (r.sampling().is_none_or(|s| s.greedy()) || self.sampler.is_some())
                    && session.tokens.len() == r.done + r.decode_tokens
                    && phase == crate::cache_pages::Phase::Active
            )?,
            "request dispatch owner disagreed"
        );
        long.bind(slot)?;
        ensure!(
            long.sessions[slot].tokens.len() == r.done + r.decode_tokens,
            "request/model cursor diverged"
        );
        let prefill = r.done < r.tokens.len();
        let exact_head = prefill && !r.done.is_multiple_of(64);
        ensure!(
            !exact_head || long.sessions[slot].mixed,
            "unaligned pure request prefix"
        );
        if prefill && r.marks.is_none() {
            let marks = self.snapshot_marks(long, id, slot)?;
            self.queue.set_marks(id, marks)?;
        }
        if prefill && !self.vision_tables.contains_key(&id) {
            let r = self.queue.request(id)?;
            let mut keys: Vec<i32> = r.tokens[r.done..]
                .iter()
                .copied()
                .filter(|&t| t >= crate::vision_input::BASE)
                .collect();
            keys.sort_unstable();
            keys.dedup();
            if !keys.is_empty() {
                match self.vision_exchange(long, id, keys)? {
                    Some(table) => {
                        self.vision_tables.insert(id, table);
                    }
                    None => {
                        // Every rank agreed the rows are missing: fail only
                        // this request and release its slot.
                        let released = long.cancel();
                        long.agree_speculation_work(
                            &self.wire.control,
                            871,
                            self.queue.digest(),
                            released,
                        )?;
                        self.queue.finish(id, End::Rejected, now)?;
                        self.output_streams.remove(&id);
                        self.grammar_remove(id);
                        if let Some(speculation) = &mut self.speculation {
                            speculation.remove(id);
                        }
                        report.rejected = Some("prompt image rows unavailable".into());
                        report.completed = Some(id);
                        return Ok(report);
                    }
                }
            }
        }
        let r = self.queue.request(id)?;
        let mut n = if exact_head {
            if self.exact_prompt_batch {
                let planned = long.exact_prompt_rows(&r.tokens[r.done..]);
                let mut plan = Sha256::new();
                plan.update(self.queue.digest());
                plan.update((planned.as_ref().copied().unwrap_or(0) as u64).to_le_bytes());
                // The backlog can shorten a window to one serial row. Agree
                // the plan before choosing paths with different collectives.
                long.agree_speculation_work(
                    &self.wire.control,
                    873,
                    plan.finalize().into(),
                    planned,
                )?
            } else {
                1
            }
        } else if prefill {
            let next = r.marks.iter().flatten().find(|&&m| m > r.done);
            next.map_or(usize::MAX, |m| m - r.done)
                .min(long.prefill_rows)
                .min(r.tokens.len() - r.done)
        } else {
            1
        };
        let speculative = !prefill
            && self.speculation.is_some()
            && r.drafts()
            && !r.has_grammar()
            && r.sampling().is_some_and(|p| {
                self.speculation
                    .as_ref()
                    .is_some_and(|state| state.accepts(p))
            })
            && r.next_output() == crate::thinking_budget::Next::Sample
            && r.output - r.outputs.len() >= 2
            && long.sessions[slot].tokens.len() <= crate::cache_pages::CONTEXT - 8;
        long.engine.set_vision(
            self.vision_tables
                .get(&id)
                .filter(|_| prefill)
                .map(|(keys, rows)| (keys.clone(), rows.ptr)),
        );
        let (mut next, mut elapsed, mut stopped, output_error) = if speculative {
            let round = self.speculative_round(long, id)?;
            n = round.tokens.len();
            (
                round.tokens,
                round.elapsed,
                round.stopped,
                round.output_error,
            )
        } else {
            let started = Instant::now();
            let result = if exact_head {
                if n > 1 {
                    long.exact_prompt_batch(
                        &self.wire.control,
                        self.queue.digest(),
                        &r.tokens[r.done..r.done + n],
                    )
                } else {
                    long.exact_prompt_token(r.tokens[r.done]).map(Some)
                }
            } else if prefill {
                if self.final_prefill_head_only {
                    long.prefill_request(r.done, &r.tokens[r.done..r.done + n])
                } else {
                    long.prefill(r.done, &r.tokens[r.done..r.done + n])
                        .map(Some)
                }
            } else {
                long.step(
                    long.sessions[slot].tokens.len(),
                    *r.outputs.last().context("missing pending output")?,
                )
                .map(Some)
            };
            let elapsed = started.elapsed();
            let mut worked = Sha256::new();
            worked.update(self.queue.digest());
            worked.update([u8::from(long.sessions[slot].mixed)]);
            worked.update((long.sessions[slot].tokens.len() as u64).to_le_bytes());
            let next = long.agree_speculation_work(
                &self.wire.control,
                805,
                worked.finalize().into(),
                result,
            )?;
            (next.into_iter().collect(), elapsed, false, false)
        };
        long.engine.set_vision(None);
        let r = self.queue.request(id)?;
        // Pure snapshots at the planned marks; each ends its own slab.
        if prefill
            && !long.sessions[slot].mixed
            && r.marks.as_ref().is_some_and(|m| m.contains(&(r.done + n)))
            && let Some(store) = &mut self.cache
        {
            let saved_at = Instant::now();
            let saved = if r.background {
                store.save(long)?
            } else {
                store.save_foreground(long)?
            };
            if let Some(disk) = &mut self.disk {
                disk.offer(saved.id, r.background);
            }
            elapsed += saved_at.elapsed();
        }
        // Complete the model-work agreement before another collective. A
        // thinking-budget delimiter replaces this cursor's pending selection
        // before any draw, including the first output of a zero-budget request.
        // Pauses retain it exactly like an ordinary sampled pending token.
        if !speculative
            && (!prefill || r.done + n == r.tokens.len())
            && (r.next_output() != crate::thinking_budget::Next::Sample
                || r.sampling().is_some_and(|s| !s.greedy())
                || r.has_grammar())
        {
            use crate::thinking_budget::Next;
            let selected_at = Instant::now();
            let result = match r.next_output() {
                Next::Force(token) => long.force_pending(token).map(|()| token),
                Next::Sample => {
                    if r.has_grammar() {
                        self.grammar_sample(long, id, r.sampling(), next[0])
                    } else if let Some(policy) = r.sampling().filter(|s| !s.greedy()) {
                        long.sample(
                            self.sampler.as_mut().context("sampler unavailable")?,
                            policy,
                        )
                        .map(|result| result.0 as i32)
                    } else {
                        Ok(next[0])
                    }
                }
                Next::End => Err(anyhow::anyhow!(
                    "request dispatched after output termination"
                )),
            };
            elapsed += selected_at.elapsed();
            next[0] =
                long.agree_speculation_work(&self.wire.control, 822, self.queue.digest(), result)?;
        }
        crate::round_timing::enter(crate::round_timing::Phase::Timing);
        let mut timing = [0; 64];
        timing[..8].copy_from_slice(b"ALTIME01");
        if long.engine.rank == 0 {
            timing[8..16].copy_from_slice(&u64::try_from(elapsed.as_nanos())?.to_le_bytes());
            timing[16..24]
                .copy_from_slice(&u64::try_from((now + began.elapsed()).as_nanos())?.to_le_bytes());
        }
        let all = self.wire.control.exchange(long, &timing)?;
        ensure!(
            all.iter().all(|r| r[..8] == timing[..8]),
            "request timing protocol mismatch"
        );
        let elapsed = Duration::from_nanos(u64::from_le_bytes(all[0][8..16].try_into().unwrap()));
        let end = Duration::from_nanos(u64::from_le_bytes(all[0][16..24].try_into().unwrap()));
        ensure!(
            end >= now && elapsed <= end - now,
            "request work clock invalid"
        );
        self.queue.now(end)?;
        self.stamp = end;
        if output_error {
            self.queue.output_rejected(id, elapsed)?;
            ensure!(
                self.wire
                    .control
                    .agree(long, 807, self.queue.digest(), true)?,
                "request output rejection disagreement"
            );
            // The typed verifier rejection already drained/discarded this slot.
            self.queue.finish(id, End::OutputError, end)?;
            self.output_streams.remove(&id);
            self.grammar_remove(id);
            if let Some(speculation) = &mut self.speculation {
                speculation.remove(id);
            }
            report.completed = Some(id);
            return Ok(report);
        }
        let mut done = if speculative {
            self.queue.worked_many(id, n, &next, elapsed)?
        } else if prefill {
            self.queue
                .worked_prefill(id, n, next.first().copied(), elapsed)?
        } else {
            self.queue.worked(id, n, next[0], elapsed)?
        };
        let mut terminal = End::Complete;
        if self.queue.request(id)?.done == self.queue.request(id)?.tokens.len()
            && self.queue.output_policy(id)?
        {
            let mut failed = false;
            if !speculative && let Some(stream) = self.output_streams.get_mut(&id) {
                match stream.clip_accepted(&[next[0] as u32]) {
                    Ok(clipped) if clipped.keep == 1 => {
                        stopped = clipped.ended && stream.text.stopped;
                    }
                    Ok(_) | Err(_) => failed = true,
                }
            }
            if failed {
                self.queue.output_failed(id)?;
                terminal = End::OutputError;
                done = true;
            } else if self.queue.request(id)?.stops_at(*next.last().unwrap()) {
                self.queue.terminal(id, Completion::Eos)?;
                done = true;
            } else if stopped {
                self.queue.terminal(id, Completion::Stop)?;
                done = true;
            }
        }
        if self.queue.request(id)?.has_grammar()
            && self.queue.request(id)?.done == self.queue.request(id)?.tokens.len()
            && terminal != End::OutputError
        {
            let committed = self.grammar_commit(id, next[0], done);
            long.agree_speculation_work(&self.wire.control, 855, self.queue.digest(), committed)?;
        }
        crate::round_timing::enter(crate::round_timing::Phase::Progress);
        ensure!(
            self.wire.control.agree(
                long,
                807,
                {
                    let mut h = Sha256::new();
                    h.update(self.queue.digest());
                    h.update(self.grammar_digest());
                    h.finalize().into()
                },
                true
            )?,
            "request output/progress disagreement"
        );
        report.worked = Some(id);
        report.emitted = if self.queue.request(id)?.done == self.queue.request(id)?.tokens.len() {
            next.len()
        } else {
            0
        };
        if self.audit {
            let r = self.queue.request(id)?;
            if r.done == r.tokens.len() {
                let trace = self.traces.entry(id).or_default();
                if prefill {
                    trace.prefill = Some(long.state(r.done)?);
                }
                trace.logits.push(long.logits()?);
                if done {
                    trace.final_state = Some(long.state(long.sessions[slot].tokens.len())?);
                }
            }
        }
        if done {
            let finished = if terminal == End::Complete {
                long.finish()
            } else {
                long.cancel()
            };
            long.agree_speculation_work(&self.wire.control, 806, self.queue.digest(), finished)?;
            self.queue.finish(id, terminal, end)?;
            self.output_streams.remove(&id);
            self.grammar_remove(id);
            if let Some(speculation) = &mut self.speculation {
                speculation.remove(id);
            }
            report.completed = Some(id);
        }
        self.traces.retain(|id, _| {
            self.queue.request(*id).is_ok()
                || self.queue.completed.iter().any(|c| c.status.id == *id)
        });
        self.vision_tables.retain(|id, _| {
            self.queue
                .request(*id)
                .is_ok_and(|r| r.done < r.tokens.len())
        });
        if speculative {
            crate::round_timing::round();
        }
        Ok(report)
    }
}
#[path = "request_grammar.rs"]
mod grammar;
#[path = "request_speculation.rs"]
mod speculation;
#[path = "request_scheduler_tests.rs"]
mod tests;
pub use tests::run;
#[path = "request_owner.rs"]
mod owner;
pub use owner::run as run_owner;
#[path = "request_termination_tests.rs"]
mod termination;
pub use termination::run as run_termination;
#[path = "request_sampling_tests.rs"]
mod sampled;
pub use sampled::run as run_sampled_requests;
#[path = "request_cache.rs"]
mod cache;
pub use cache::run as run_cached_requests;
#[path = "request_disk.rs"]
mod disk;
pub use disk::run as run_disk_requests;

pub use disk::run_restore as run_restored_requests;

#[path = "request_service.rs"]
mod service;
pub use service::serve;

impl Scheduler {
    fn multi_greedy_ready(&self) -> Option<[u64; 4]> {
        let mut ids = [0; 4];
        for (slot, id) in ids.iter_mut().enumerate() {
            let request = self.queue.requests_for_slot(slot)?;
            if request.background
                || request.paused
                || request.restoring
                || request.cancellation().is_some()
                || request.has_grammar()
                || !request.sampling().is_some_and(|p| p.greedy())
                || request.done != request.tokens.len()
                || request.outputs.is_empty()
                || request.done + request.decode_tokens >= 512
                || request.next_output() != crate::thinking_budget::Next::Sample
            {
                return None;
            }
            *id = request.id;
        }
        Some(ids)
    }
    fn pump_multi_greedy(
        &mut self,
        long: &mut Long<'_>,
        ids: [u64; 4],
        now: Duration,
        began: Instant,
        mut report: Report,
    ) -> Result<Report> {
        let started = Instant::now();
        let mut digest = Sha256::new();
        digest.update(b"multi-greedy-v1");
        digest.update(self.queue.digest());
        for id in ids {
            digest.update(id.to_le_bytes());
        }
        ensure!(
            self.wire
                .control
                .agree(long, 901, digest.finalize().into(), true)?,
            "multi owner disagreement"
        );
        let output = long.multi_greedy_step();
        let next =
            long.agree_speculation_work(&self.wire.control, 902, self.queue.digest(), output)?;
        ensure!(next.len() == 4, "multi result width");
        let elapsed = started.elapsed();
        let mut timing = [0u8; 64];
        timing[..8].copy_from_slice(b"ALMULT01");
        if long.engine.rank == 0 {
            timing[8..16].copy_from_slice(&u64::try_from(elapsed.as_nanos())?.to_le_bytes());
            timing[16..24]
                .copy_from_slice(&u64::try_from((now + began.elapsed()).as_nanos())?.to_le_bytes());
        }
        let all = self.wire.control.exchange(long, &timing)?;
        ensure!(
            all.iter().all(|r| r[..8] == timing[..8]),
            "multi timing record"
        );
        let elapsed = Duration::from_nanos(u64::from_le_bytes(all[0][8..16].try_into().unwrap()));
        let end = Duration::from_nanos(u64::from_le_bytes(all[0][16..24].try_into().unwrap()));
        ensure!(end >= now && elapsed <= end - now, "multi clock");
        self.queue.now(end)?;
        self.stamp = end;
        for (slot, &id) in ids.iter().enumerate() {
            let token = next[slot];
            let mut done = self.queue.worked(id, 1, token, elapsed / 4)?;
            let mut terminal = End::Complete;
            if self.queue.output_policy(id)? {
                let mut stopped = false;
                let mut failed = false;
                if let Some(stream) = self.output_streams.get_mut(&id) {
                    match stream.clip_accepted(&[token as u32]) {
                        Ok(clipped) if clipped.keep == 1 => {
                            stopped = clipped.ended && stream.text.stopped
                        }
                        _ => failed = true,
                    }
                }
                if failed {
                    self.queue.output_failed(id)?;
                    terminal = End::OutputError;
                    done = true;
                } else if self.queue.request(id)?.stops_at(token) {
                    self.queue.terminal(id, Completion::Eos)?;
                    done = true;
                } else if stopped {
                    self.queue.terminal(id, Completion::Stop)?;
                    done = true;
                }
            }
            if done {
                long.bind(slot)?;
                let completed = if terminal == End::Complete {
                    long.finish()
                } else {
                    long.cancel()
                };
                long.agree_speculation_work(
                    &self.wire.control,
                    903,
                    self.queue.digest(),
                    completed,
                )?;
                self.queue.finish(id, terminal, end)?;
                self.output_streams.remove(&id);
                self.grammar_remove(id);
                if let Some(state) = &mut self.speculation {
                    state.remove(id);
                }
            }
            report.batch_output.push((id, 1, done.then_some(terminal)));
        }
        ensure!(
            self.wire
                .control
                .agree(long, 904, self.queue.digest(), true)?,
            "multi progress disagreement"
        );
        report.worked = Some(ids[0]);
        report.emitted = 1;
        Ok(report)
    }
}

impl Scheduler {
    fn multi_spec_ready(&self, window: usize) -> Option<Vec<u64>> {
        if self.multi_greedy == 6 {
            return self.multi_cohort_ready(window);
        }
        let speculation = self.speculation.as_ref()?;
        let mut ids = Vec::new();
        for slot in 0..4 {
            let Some(request) = self.queue.requests_for_slot(slot) else {
                continue;
            };
            if request.background {
                continue;
            }
            if request.paused
                || request.restoring
                || request.cancellation().is_some()
                || request.has_grammar()
                || !request.drafts()
                || !request.sampling().is_some_and(|p| {
                    (p.greedy() || (self.multi_greedy == 6 && p.gpu_fits(4 * p.local_count())))
                        && speculation.accepts(p)
                })
                || request.done != request.tokens.len()
                || request.outputs.is_empty()
                || request.done + request.decode_tokens + window > 512
                || request.output - request.outputs.len() < window
                || request.next_output() != crate::thinking_budget::Next::Sample
            {
                return None;
            }
            ids.push(request.id);
        }
        if (2..=4).contains(&ids.len()) {
            let first = self.queue.request(ids[0]).ok()?.sampling()?;
            if ids.iter().all(|&id| {
                self.queue
                    .request(id)
                    .ok()
                    .and_then(|r| r.sampling())
                    .is_some_and(|p| {
                        p.greedy() == first.greedy()
                            && (first.greedy() || p.local_count() == first.local_count())
                    })
            }) {
                return Some(ids);
            }
        }
        None
    }
    fn multi_cohort_ready(&self, window: usize) -> Option<Vec<u64>> {
        let speculation = self.speculation.as_ref()?;
        let next = self.queue.peek_runnable()?;
        let first = self.queue.request(next).ok()?;
        let eligible = |r: &crate::request_queue::Request| {
            !r.background
                && !r.paused
                && !r.restoring
                && r.cancellation().is_none()
                && !r.has_grammar()
                && r.drafts()
                && r.done == r.tokens.len()
                && !r.outputs.is_empty()
                && r.done + r.decode_tokens + window <= crate::cache_pages::CONTEXT
                && (r.done + r.decode_tokens >= 2051 || r.done + r.decode_tokens + window <= 2051)
                && r.output - r.outputs.len() >= window
                && r.next_output() == crate::thinking_budget::Next::Sample
                && r.sampling().is_some_and(|p| {
                    speculation.accepts(p) && (p.greedy() || p.gpu_fits(4 * p.local_count()))
                })
        };
        if !eligible(first) {
            return None;
        }
        let policy = first.sampling()?;
        let ids: Vec<_> = (0..4)
            .filter_map(|slot| self.queue.requests_for_slot(slot))
            .filter(|r| {
                eligible(r)
                    && r.sampling().is_some_and(|p| {
                        p.greedy() == policy.greedy()
                            && (policy.greedy() || p.local_count() == policy.local_count())
                    })
            })
            .map(|r| r.id)
            .collect();
        (2..=4).contains(&ids.len()).then_some(ids)
    }
    fn pump_multi_spec(
        &mut self,
        long: &mut Long<'_>,
        ids: Vec<u64>,
        window: usize,
        now: Duration,
        began: Instant,
        mut report: Report,
    ) -> Result<Report> {
        let started = Instant::now();
        crate::round_timing::enter(crate::round_timing::Phase::Plan);
        let mut digest = Sha256::new();
        digest.update(b"multi-spec-v1");
        digest.update(self.queue.digest());
        for id in &ids {
            digest.update(id.to_le_bytes());
        }
        ensure!(
            self.wire
                .control
                .agree(long, 901, digest.finalize().into(), true)?,
            "multi owner disagreement"
        );
        let mut slots = Vec::new();
        let mut inputs = Vec::new();
        let mut policies = Vec::new();
        let mut eos = Vec::new();
        let mut lookup_rows = Vec::new();
        let mut backlogs = Vec::new();
        for &id in &ids {
            let request = self.queue.request(id)?;
            let slot = request.slot.context("multi request slot")?;
            let policy = request.sampling().context("multi request policy")?;
            policies.push(policy);
            let pending = *request.outputs.last().context("multi pending output")?;
            slots.push(slot);
            eos.push(request.stop_eos());
            long.bind(slot)?;
            let lookup = self.multi_lookup(id, window);
            let mut lookup_digest = Sha256::new();
            lookup_digest.update(self.queue.digest());
            lookup_digest.update(id.to_le_bytes());
            if let Ok(found) = &lookup {
                for t in found {
                    lookup_digest.update(t.to_le_bytes());
                }
            }
            let lookup = long.agree_speculation_work(
                &self.wire.control,
                911,
                lookup_digest.finalize().into(),
                lookup,
            )?;
            backlogs.push(long.speculation_backlog_rows(slot)?);
            lookup_rows.push(!lookup.is_empty());
            let proposed = if !lookup.is_empty() {
                long.prepare_speculation(window, false).map(|()| {
                    let mut input = vec![pending];
                    input.extend_from_slice(&lookup);
                    input
                })
            } else {
                crate::round_timing::enter(crate::round_timing::Phase::Draft);
                long.propose_speculation(&self.wire.control, self.queue.digest(), policy)
                    .and_then(|proposal| {
                        ensure!(
                            proposal.tokens.len() >= window - 1
                                && proposal.probabilities.len() >= window - 1,
                            "multi incomplete proposal"
                        );
                        self.multi_dflash_window(id, &proposal.probabilities[..window - 1])?;
                        let mut input = vec![pending];
                        input.extend_from_slice(&proposal.tokens[..window - 1]);
                        Ok(input)
                    })
            };
            let proposed = long.agree_speculation_work(
                &self.wire.control,
                905,
                self.queue.digest(),
                proposed,
            )?;
            if lookup.is_empty() {
                long.drafted();
            }
            inputs.push(proposed);
        }
        let streams = &mut self.output_streams;
        let queue = &self.queue;
        let mut stopped_rows = vec![false; ids.len()];
        let mut accepted_rows = vec![0; ids.len()];
        crate::round_timing::enter(crate::round_timing::Phase::VerifyDispatch);
        let output = long.multi_policy_step(
            &self.wire.control,
            queue.digest(),
            &slots,
            &inputs,
            &eos,
            &policies,
            |row, samples| {
                accepted_rows[row] = samples.len();
                let id = ids[row];
                let request = queue.request(id)?;
                let keep = request.sampled_prefix(samples)?;
                ensure!(keep > 0, "multi sampled prefix");
                if let Some(stream) = streams.get_mut(&id) {
                    let tokens: Vec<u32> = samples[..keep].iter().map(|&v| v as u32).collect();
                    let clipped = stream
                        .clip_accepted(&tokens)
                        .map_err(|_| anyhow::Error::new(super::speculation::ParserFailure))?;
                    stopped_rows[row] = clipped.ended && stream.text.stopped;
                    Ok(clipped.keep)
                } else {
                    Ok(keep)
                }
            },
        );
        let next =
            long.agree_speculation_work(&self.wire.control, 906, self.queue.digest(), output)?;
        ensure!(next.len() == ids.len(), "multi result width");
        let feedback = (|| -> Result<()> {
            for row in 0..ids.len() {
                if !next[row].is_empty() {
                    self.multi_feedback(
                        ids[row],
                        lookup_rows[row],
                        window,
                        backlogs[row],
                        accepted_rows[row],
                    )?;
                }
            }
            Ok(())
        })();
        long.agree_speculation_work(&self.wire.control, 912, self.queue.digest(), feedback)?;
        let elapsed = started.elapsed();
        crate::round_timing::enter(crate::round_timing::Phase::Complete);
        let mut timing = [0u8; 64];
        timing[..8].copy_from_slice(b"ALMULT01");
        if long.engine.rank == 0 {
            timing[8..16].copy_from_slice(&u64::try_from(elapsed.as_nanos())?.to_le_bytes());
            timing[16..24]
                .copy_from_slice(&u64::try_from((now + began.elapsed()).as_nanos())?.to_le_bytes());
        }
        let all = self.wire.control.exchange(long, &timing)?;
        ensure!(
            all.iter().all(|r| r[..8] == timing[..8]),
            "multi timing record"
        );
        let elapsed = Duration::from_nanos(u64::from_le_bytes(all[0][8..16].try_into().unwrap()));
        let end = Duration::from_nanos(u64::from_le_bytes(all[0][16..24].try_into().unwrap()));
        ensure!(end >= now && elapsed <= end - now, "multi clock");
        self.queue.now(end)?;
        self.stamp = end;
        for (slot, &id) in ids.iter().enumerate() {
            let tokens = &next[slot];
            let mut terminal = End::Complete;
            let mut done = if tokens.is_empty() {
                self.queue.output_rejected(id, elapsed / ids.len() as u32)?;
                terminal = End::OutputError;
                true
            } else {
                self.queue
                    .worked_many(id, tokens.len(), tokens, elapsed / ids.len() as u32)?
            };
            if !tokens.is_empty() && self.queue.output_policy(id)? {
                if self.queue.request(id)?.stops_at(*tokens.last().unwrap()) {
                    self.queue.terminal(id, Completion::Eos)?;
                    done = true;
                } else if stopped_rows[slot] {
                    self.queue.terminal(id, Completion::Stop)?;
                    done = true;
                }
            }
            if done {
                long.bind(slots[slot])?;
                let completed = if terminal == End::Complete {
                    long.finish()
                } else {
                    long.cancel()
                };
                long.agree_speculation_work(
                    &self.wire.control,
                    903,
                    self.queue.digest(),
                    completed,
                )?;
                self.queue.finish(id, terminal, end)?;
                self.output_streams.remove(&id);
                self.grammar_remove(id);
                if let Some(state) = &mut self.speculation {
                    state.remove(id);
                }
            }
            report
                .batch_output
                .push((id, tokens.len(), done.then_some(terminal)));
        }
        ensure!(
            self.wire
                .control
                .agree(long, 904, self.queue.digest(), true)?,
            "multi progress disagreement"
        );
        report.worked = Some(ids[0]);
        report.emitted = next[0].len();
        crate::round_timing::round();
        Ok(report)
    }
}
