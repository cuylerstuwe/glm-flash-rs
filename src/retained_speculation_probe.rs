//! Exercises the production ownership/cache seams, without enabling serving.
use super::super::{Engine, disk::Disk, priority::Boundary, store::Store};
use super::*;
use crate::{
    candidates::Candidates, prepared::Prepared, priority::Snapshot as Hold, ssd_store::Outcome,
};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};

#[path = "multi_decode_probe.rs"]
mod multi;

struct Fence {
    control: Control,
    tick: u64,
}
impl Fence {
    fn new(long: &Long<'_>) -> Result<Self> {
        Ok(Self {
            control: Control::new(&long.engine.device)?,
            tick: 0,
        })
    }
    fn work<T>(
        &mut self,
        long: &mut Long<'_>,
        f: impl FnOnce(&mut Long<'_>) -> Result<T>,
    ) -> Result<T> {
        let result = f(long);
        self.complete(long, result)
    }
    fn complete<T>(&mut self, long: &mut Long<'_>, result: Result<T>) -> Result<T> {
        let mut h = Sha256::new();
        h.update(b"retained-probe-v1");
        h.update(self.tick.to_le_bytes());
        self.tick += 1;
        long.agree_speculation_work(&self.control, 850, h.finalize().into(), result)
    }
    fn dispatch(&self) -> [u8; 32] {
        Sha256::digest(self.tick.to_le_bytes()).into()
    }
    fn propose(
        &mut self,
        long: &mut Long<'_>,
        policy: Sampling,
    ) -> Result<crate::draft_chain::Proposal> {
        let result = long.propose_speculation(&self.control, self.dispatch(), policy);
        self.complete(long, result)
    }
    fn verify(
        &mut self,
        long: &mut Long<'_>,
        tokens: &[i32],
        policy: Sampling,
        clip: impl FnOnce(&[i32]) -> Result<usize>,
    ) -> Result<Vec<i32>> {
        let result = long.verify_speculation(&self.control, self.dispatch(), tokens, policy, clip);
        self.complete(long, result)
    }
    fn check(&mut self, long: &mut Long<'_>, condition: bool, label: &str) -> Result<()> {
        self.work(long, |_| {
            ensure!(condition, "{label}");
            Ok(())
        })
    }
}

// Only an entry in this probe's freshly created private cache directory is
// changed. Corrupt the last payload byte so earlier target/drafter chunks have
// already reached the unrunnable destination when checksum validation fails.
fn corrupt_tail(path: &Path, prefix: usize) -> Result<()> {
    let mut found = None;
    for entry in fs::read_dir(path.join("entries"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|ext| ext != "entry") {
            continue;
        }
        ensure!(entry.file_type()?.is_file(), "probe cache entry file type");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(entry.path())?;
        file.seek(SeekFrom::Start(80))?;
        let mut n = [0; 4];
        file.read_exact(&mut n)?;
        if u32::from_le_bytes(n) as usize != prefix {
            continue;
        }
        ensure!(found.is_none(), "duplicate probe prefix files");
        file.seek(SeekFrom::End(-1))?;
        let mut tail = [0];
        file.read_exact(&mut tail)?;
        tail[0] ^= 1;
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&tail)?;
        file.sync_all()?;
        found = Some(());
    }
    found.context("probe cache corruption target")
}
fn prefill(fence: &mut Fence, long: &mut Long<'_>, source: &[i32], end: usize) -> Result<()> {
    while long.sessions[long.active].tokens.len() < end {
        let position = long.sessions[long.active].tokens.len();
        let next = (position + long.prefill_rows).min(end);
        fence.work(long, |long| long.prefill(position, &source[position..next]))?;
    }
    Ok(())
}
fn saved_hash(saved: &Saved) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(saved.metadata());
    for buffer in saved.buffers() {
        hash.update(buffer.read(0, buffer.bytes)?);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn context_hash(fence: &mut Fence, long: &mut Long<'_>) -> Result<String> {
    fence.work(long, |long| long.prepare_speculation(16, true))?;
    fence.work(long, |long| {
        long.with_speculation(|owner, long| {
            let snapshot = owner.draft.capture_context(&long.engine)?;
            let mut h = Sha256::new();
            h.update((snapshot.end as u64).to_le_bytes());
            h.update((snapshot.lo as u64).to_le_bytes());
            for buffer in &snapshot.buffers {
                h.update(buffer.read(0, buffer.bytes)?);
            }
            Ok(format!("{:x}", h.finalize()))
        })?
        .context("retained context owner")
    })
}
fn proposals(
    fence: &mut Fence,
    long: &mut Long<'_>,
    policy: Sampling,
) -> Result<crate::draft_chain::Proposal> {
    fence.propose(long, policy)
}
fn proposal_hash(proposal: &crate::draft_chain::Proposal) -> String {
    let mut h = Sha256::new();
    for token in &proposal.tokens {
        h.update(token.to_le_bytes());
    }
    for probability in &proposal.probabilities {
        h.update(probability.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn sample(
    fence: &mut Fence,
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    policy: Sampling,
) -> Result<i32> {
    Ok(fence.work(long, |long| long.sample(sampler, policy))?.0 as i32)
}
fn check_restored_prefix(
    fence: &mut Fence,
    long: &mut Long<'_>,
    cut: usize,
    target: &str,
    draft: &str,
) -> Result<()> {
    fence.work(long, |long| {
        ensure!(long.state(cut)? == target, "restored target prefix drift");
        let snapshot = long
            .capture_speculation()?
            .context("restored draft snapshot")?;
        ensure!(
            saved_hash(&snapshot)? == draft,
            "restored drafter prefix drift"
        );
        Ok(())
    })
}
fn hold_with_backlog(
    fence: &mut Fence,
    long: &mut Long<'_>,
    boundary: &mut Boundary,
    revision: u64,
) -> Result<()> {
    let slot = long.active;
    let cursor = long.sessions[slot].tokens.len();
    let pending = long.sessions[slot].pending;
    let (before, feature) = fence.work(long, |long| {
        let feature = long.speculation.as_ref().context("held owner")?.slots[slot].status();
        ensure!(
            feature.end > feature.start && feature.phase == FeaturePhase::Ready,
            "hold needs a live nonempty backlog"
        );
        Ok((long.state(cursor)?, feature))
    })?;
    let head = (long.engine.rank == 0).then_some(Hold {
        revision,
        hold: true,
    });
    boundary.apply(long, head)?;
    fence.work(long, |long| {
        let held = long.speculation.as_ref().unwrap().slots[slot].status();
        ensure!(
            held.start == feature.start
                && held.end == feature.end
                && held.epoch == feature.epoch
                && held.phase == FeaturePhase::Paused,
            "hold lost accepted features"
        );
        ensure!(
            long.prepare_speculation(16, true).is_err() && long.healthy,
            "held speculation ran or damaged owner"
        );
        Ok(())
    })?;
    fence.work(long, |long| long.bind(3))?;
    fence.work(long, |long| long.start(64, 4))?;
    fence.work(long, |long| long.prefill(0, &[43; 64]))?;
    let token = fence.work(long, |long| {
        long.sessions[3].pending.context("foreground pending")
    })?;
    fence.work(long, |long| long.step(64, token))?;
    fence.work(long, |long| long.finish())?;
    fence.work(long, |long| long.bind(slot))?;
    let head = (long.engine.rank == 0).then_some(Hold {
        revision: revision + 1,
        hold: false,
    });
    boundary.apply(long, head)?;
    fence.work(long, |long| {
        let resumed = long.speculation.as_ref().unwrap().slots[slot].status();
        ensure!(
            resumed.start == feature.start
                && resumed.end == feature.end
                && resumed.epoch == feature.epoch
                && resumed.phase == FeaturePhase::Ready,
            "release replayed or lost accepted features"
        );
        ensure!(
            long.state(cursor)? == before && long.sessions[slot].pending == pending,
            "held target state/pending changed"
        );
        Ok(())
    })
}

fn parser_rejections(
    fence: &mut Fence,
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    source: &[i32],
    policy: Sampling,
) -> Result<Vec<Value>> {
    let mut cases = Vec::new();
    for all_ranks in [false, true] {
        fence.work(long, |long| long.bind(0))?;
        fence.work(long, |long| long.start(128, 16))?;
        prefill(fence, long, source, 128)?;
        let anchor = sample(fence, long, sampler, policy)?;
        fence.work(long, |long| long.step(128, anchor))?;
        let expected = sample(fence, long, sampler, policy)?;
        let expected_state = fence.work(long, |long| long.state(129))?;
        let expected_draft = context_hash(fence, long)?;
        fence.work(long, |long| long.finish())?;
        fence.work(long, |long| long.start(128, 16))?;
        prefill(fence, long, source, 128)?;
        let pending = sample(fence, long, sampler, policy)?;
        fence.check(long, pending == anchor, "parser peer cold pending drift")?;
        fence.work(long, |long| long.bind(1))?;
        fence.work(long, |long| long.start(128, 16))?;
        prefill(fence, long, source, 128)?;
        let victim_pending = sample(fence, long, sampler, policy)?;
        let reject = all_ranks || long.engine.rank == 2;
        let result = long.verify_speculation(
            &fence.control,
            fence.dispatch(),
            &[victim_pending, 1, 2, 3],
            policy,
            |accepted| {
                if reject {
                    Err(ParserFailure.into())
                } else {
                    Ok(accepted.len())
                }
            },
        );
        let discarded = match result {
            Err(error) => match error.downcast_ref::<DiscardProof>() {
                Some(proof) => long.discard_rejected_output(proof),
                None => Err(error),
            },
            Ok(_) => Err(anyhow::anyhow!("injected parser failure committed output")),
        };
        fence.complete(long, discarded)?;
        fence.work(long, |long| {
            ensure!(
                long.sessions[1].tokens.is_empty()
                    && long.sessions[1].pending.is_none()
                    && long.memory.book().slot(1)?.phase() == crate::cache_pages::Phase::Idle,
                "parser rejected slot remained runnable"
            );
            ensure!(
                long.sessions[0].tokens.len() == 128 && long.sessions[0].pending == Some(anchor),
                "parser discard changed peer ownership"
            );
            Ok(())
        })?;
        fence.work(long, |long| long.bind(0))?;
        fence.work(long, |long| long.step(128, anchor))?;
        let actual = sample(fence, long, sampler, policy)?;
        let actual_draft = context_hash(fence, long)?;
        fence.work(long, |long| {
            ensure!(
                actual == expected
                    && long.state(129)? == expected_state
                    && actual_draft == expected_draft,
                "parser discard changed peer continuation"
            );
            Ok(())
        })?;
        fence.work(long, |long| long.finish())?;
        cases.push(json!({"failed_ranks":if all_ranks { 4 } else { 1 },"fault_rank":if all_ranks { None } else { Some(2) },
            "discarded_before_publish":true,"peer_target_pending_drafter_exact":true}));
    }
    Ok(cases)
}

fn exact_prompt_cases(
    fence: &mut Fence,
    long: &mut Long<'_>,
    sampler: &mut Candidates,
    source: &[i32],
    policy: Sampling,
) -> Result<Vec<Value>> {
    let mut boundary = Boundary::new(long)?;
    let mut cases = Vec::new();
    for cut in [128, 1984, 8192] {
        for offset in [1, 15, 31, 63] {
            let begin = cut + offset;
            let end = begin + 65;
            let mut reference = None;
            for batched in [false, true] {
                fence.work(long, |long| long.bind(0))?;
                fence.work(long, |long| long.start_mode(end, 32, true))?;
                prefill(fence, long, source, cut)?;
                for &token in &source[cut..begin] {
                    fence.work(long, |long| long.exact_prompt_token(token))?;
                }
                let mut checkpoints = Vec::new();
                let mut held = false;
                while long.sessions[0].tokens.len() < end {
                    let at = long.sessions[0].tokens.len();
                    if at.is_multiple_of(64) {
                        fence.work(long, |long| long.prefill(at, &source[at..end]))?;
                    } else {
                        let n = long.exact_prompt_rows(&source[at..end])?;
                        if batched && n > 1 {
                            let result = long.exact_prompt_batch(
                                &fence.control,
                                fence.dispatch(),
                                &source[at..at + n],
                            );
                            fence.complete(long, result)?;
                        } else {
                            for &token in &source[at..at + n] {
                                fence.work(long, |long| long.exact_prompt_token(token))?;
                            }
                        }
                    }
                    checkpoints.push(fence.work(long, |long| {
                        let at = long.sessions[0].tokens.len();
                        let features = long.speculation.as_ref().unwrap().slots[0].status();
                        Ok(json!({"cursor":at,"state":long.state(at)?,
                            "logits":format!("{:x}",Sha256::digest(long.engine.logits.read(0,long.engine.logits.bytes)?)),
                            "pending":long.sessions[0].pending,"start":features.start,"end":features.end}))
                    })?);
                    // Logits are shared scratch, not retained per slot. Capture
                    // this cursor's logits before unrelated foreground work
                    // overwrites them; hold_with_backlog separately proves all
                    // retained target/drafter/pending state survives the hold.
                    if batched && !held {
                        hold_with_backlog(
                            fence,
                            long,
                            &mut boundary,
                            (cases.len() * 2 + 1) as u64,
                        )?;
                        held = true;
                    }
                }
                let draft = context_hash(fence, long)?;
                let pending = sample(fence, long, sampler, policy)?;
                let proposal = proposals(fence, long, policy)?;
                let proposal = proposal_hash(&proposal);
                let mut outputs = vec![pending];
                for _ in 0..4 {
                    let at = long.sessions[0].tokens.len();
                    let token = *outputs.last().unwrap();
                    fence.work(long, |long| long.step(at, token))?;
                    outputs.push(sample(fence, long, sampler, policy)?);
                }
                let followup = fence.work(long, |long| long.state(end + 4))?;
                let final_draft = context_hash(fence, long)?;
                let actual = json!({"checkpoints":checkpoints,"draft":draft,"proposal":proposal,
                    "outputs":outputs,"followup_state":followup,"followup_draft":final_draft});
                if let Some(reference) = &reference {
                    if &actual != reference {
                        eprintln!("exact_prompt_reference={reference}");
                        eprintln!("exact_prompt_actual={actual}");
                    }
                    fence.check(
                        long,
                        &actual == reference,
                        "exact prompt serial/batch drift",
                    )?;
                    ensure!(held, "exact prompt candidate missed native hold");
                } else {
                    reference = Some(actual);
                }
                fence.work(long, |long| long.finish())?;
            }
            cases.push(json!({"cut":cut,"offset":offset,"prompt":end,
                "exact_target_logits_pending":true,"exact_backlog_boundaries":true,
                "exact_draft_proposal_followup":true,"native_hold":true}));
            eprintln!("exact prompt batch matched cut={cut} offset={offset}");
        }
    }
    Ok(cases)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    manifest: &Path,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    sets: &[Vec<i32>],
    use_roce: bool,
    exact_prompt: bool,
) -> Result<Value> {
    let source = sets.last().context("retained speculation fixture")?;
    ensure!(
        source.len() >= 8320,
        "retained fixture needs at least 8320 tokens"
    );
    let fault = match std::env::var("ALASKA_RETAINED_FAULT").as_deref() {
        Err(std::env::VarError::NotPresent) => None,
        Ok("invalid-token") => Some("invalid-token"),
        Ok("missing-pending") => Some("missing-pending"),
        Ok("parser-only") => Some("parser-only"),
        _ => anyhow::bail!(
            "ALASKA_RETAINED_FAULT must be invalid-token/missing-pending/parser-only or unset"
        ),
    };
    let multi_probe = std::env::var("ALASKA_MULTI_PROBE").as_deref() == Ok("1");
    let multi_sampled = std::env::var("ALASKA_MULTI_PROBE_SAMPLED").as_deref() == Ok("1");
    let multi_long = std::env::var("ALASKA_MULTI_PROBE_LONG").as_deref() == Ok("1");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, multi_probe, use_roce)?;
    engine.batch = Some(crate::model::batch::Batch::new_with_rows(
        &engine.device,
        &engine,
        root,
        native,
        8192,
    )?);
    let mut long = Long::new(engine, root)?;
    let temperature = super::super::verifier::diagnostic_temperature(&long)?;
    let mut fence = Fence::new(&long)?;
    fence.work(&mut long, |long| {
        long.enable_speculation(manifest, root, native)
    })?;
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let mut mode_hash = Sha256::new();
    mode_hash.update(b"retained-probe-mode-v1");
    mode_hash.update(fault.unwrap_or("normal").as_bytes());
    mode_hash.update([u8::from(exact_prompt)]);
    mode_hash.update([u8::from(multi_probe)]);
    let multi_window = match std::env::var("ALASKA_MULTI_PROBE_WINDOW").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("4") => 4,
        Ok("8") => 8,
        Ok("2") => 2,
        Ok("3") => 3,
        _ => 0,
    };
    mode_hash.update([multi_window]);
    mode_hash.update([u8::from(multi_sampled)]);
    mode_hash.update([u8::from(multi_long)]);
    ensure!(
        fence
            .control
            .agree(&long, 853, mode_hash.finalize().into(), true)?,
        "retained probe mode disagreement"
    );
    let policy = Sampling::new(731, temperature, 20, 0.95)?;
    if multi_probe {
        ensure!(
            fault.is_none() && !exact_prompt && policy.greedy() && multi_window != 0,
            "multi probe mode"
        );
        return multi::run(&mut fence, &mut long, source, native);
    }
    if exact_prompt {
        ensure!(
            fault.is_none(),
            "exact prompt probe does not accept a fault mode"
        );
        let cases = exact_prompt_cases(&mut fence, &mut long, &mut sampler, source, policy)?;
        return Ok(json!({"rank":rank,"temperature":temperature,"cases":cases,
            "scope":"Serial versus forced batched off-grid prompt heads, exact target/logit/pending/drafter/proposal/followup state, retained background hold and unchanged backlog/grid boundaries. Serving and timing remain separate."}));
    }
    if fault == Some("parser-only") {
        let cases = parser_rejections(&mut fence, &mut long, &mut sampler, source, policy)?;
        fence.work(&mut long, |long| {
            ensure!(
                long.healthy
                    && (0..4).all(|slot| {
                        long.memory
                            .book()
                            .slot(slot)
                            .is_ok_and(|slot| slot.phase() == crate::cache_pages::Phase::Idle)
                    }),
                "parser-only group did not finish healthy and idle"
            );
            Ok(())
        })?;
        return Ok(json!({
            "rank":rank,"temperature":temperature,"mode":"parser-only","prefill_rows":long.prefill_rows,
            "parser_rejections":cases,"healthy_idle":true,
            "scope":"Only one-rank and all-rank parser failures: discard before publication and exact unaffected-peer target/pending/drafter continuation. Cache behavior is covered by separate diagnostic cases."
        }));
    }
    let mut boundary = Boundary::new(&long)?;
    let mut ram = Store::new(&long, 512 << 20)?;
    let path = std::path::PathBuf::from(format!(
        "evidence/retained-model-{}-{}-rank{}",
        if use_roce { "roce" } else { "nccl" },
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        rank
    ));
    let mut disk = Disk::new(&mut long, &path, 512 << 20)?;
    let mut cases = Vec::new();
    for (case, cut) in [128usize, 2112, 8256].into_iter().enumerate() {
        let end = cut + 64;
        let tokens = &source[..end];
        fence.work(&mut long, |long| long.bind(0))?;
        fence.work(&mut long, |long| long.start_mode(end, 32, true))?;
        prefill(&mut fence, &mut long, tokens, cut)?;
        let (prefix_state, draft_hash) = fence.work(&mut long, |long| {
            let saved = long.capture_speculation()?.context("pure prefix draft")?;
            let metadata = saved.metadata();
            ensure!(
                saved.gpu_bytes() == cut.min(WINDOW) * 5120,
                "retained snapshot allocation bound"
            );
            for at in [0usize, 8, 40, 48, 56] {
                let mut bad = metadata;
                bad[at] ^= 1;
                ensure!(
                    long.validate_speculation_metadata(&bad, cut).is_err(),
                    "corrupt draft metadata accepted"
                );
            }
            long.validate_speculation_metadata(&metadata, cut)?;
            Ok((long.state(cut)?, saved_hash(&saved)?))
        })?;
        let saved = ram.save(&mut long)?;
        let id = fence.work(&mut long, |_| {
            saved.id.context("retained RAM snapshot skipped")
        })?;
        let saved = disk.save(&mut long)?;
        fence.check(
            &mut long,
            saved == Outcome::Saved,
            "retained SSD snapshot skipped",
        )?;
        prefill(&mut fence, &mut long, tokens, end)?;
        let anchor = sample(&mut fence, &mut long, &mut sampler, policy)?;
        if let Some(fault) = fault {
            let cursor = long.sessions[0].tokens.len();
            let mut input = vec![anchor, anchor];
            if rank == 2 {
                if fault == "invalid-token" {
                    input[1] = -1;
                } else {
                    long.sessions[0].pending = None;
                }
            }
            let pending = long.sessions[0].pending;
            let rejected = if fault == "invalid-token" {
                long.verify_speculation(&fence.control, [91; 32], &input, policy, |accepted| {
                    Ok(accepted.len())
                })
                .map(|_| ())
            } else {
                long.propose_speculation(&fence.control, [91; 32], policy)
                    .map(|_| ())
            };
            let safe = rejected.is_err()
                && !long.healthy
                && long.sessions[0].tokens.len() == cursor
                && long.sessions[0].pending == pending;
            ensure!(
                fence.control.agree(&long, 852, [91; 32], safe)?,
                "rank-local dispatch fault published or stayed healthy"
            );
            return Ok(
                json!({"rank":rank,"temperature":temperature,"fault":fault,"fault_rank":2,"collective_rejection":true,
                "owner_poisoned":true,"host_cursor_and_pending_unpublished":true,
                "scope":"One-rank invalid dispatch rejected before proposal/verifier native collectives; isolated group terminates after failure proof."}),
            );
        }
        let proposed = proposal_hash(&proposals(&mut fence, &mut long, policy)?);
        let mut serial = vec![anchor];
        let mut states = Vec::new();
        for row in 0..16 {
            fence.work(&mut long, |long| long.step(end + row, serial[row]))?;
            serial.push(sample(&mut fence, &mut long, &mut sampler, policy)?);
            states.push(fence.work(&mut long, |long| long.state(end + row + 1))?);
        }
        let final_draft = context_hash(&mut fence, &mut long)?;
        fence.work(&mut long, |long| long.finish())?;

        fence.work(&mut long, |long| long.bind(1))?;
        let restored = ram.restore(&mut long, id, tokens, 32, true)?;
        fence.check(&mut long, restored == cut, "RAM prefix cursor")?;
        check_restored_prefix(&mut fence, &mut long, cut, &prefix_state, &draft_hash)?;
        prefill(&mut fence, &mut long, tokens, end)?;
        let pending = sample(&mut fence, &mut long, &mut sampler, policy)?;
        fence.check(&mut long, pending == anchor, "RAM pending selection")?;
        let restored_proposal = proposal_hash(&proposals(&mut fence, &mut long, policy)?);
        fence.check(
            &mut long,
            restored_proposal == proposed,
            "RAM proposal drift",
        )?;
        let first = fence.verify(&mut long, &serial[..16], policy, |accepted| {
            ensure!(accepted.len() == 16, "reference path ended early");
            Ok(4)
        })?;
        fence.work(&mut long, |long| {
            ensure!(
                first == serial[1..=4]
                    && long.state(end + 4)? == states[3]
                    && long.sessions[1].pending == Some(serial[4]),
                "clipped retained state/pending drift"
            );
            Ok(())
        })?;
        hold_with_backlog(&mut fence, &mut long, &mut boundary, (case * 2 + 1) as u64)?;
        let rest = fence.verify(&mut long, &serial[4..16], policy, |accepted| {
            Ok(accepted.len())
        })?;
        fence.work(&mut long, |long| {
            ensure!(
                rest == serial[5..=16]
                    && long.state(end + 16)? == states[15]
                    && long.sessions[1].pending == Some(serial[16]),
                "resumed retained state/pending drift"
            );
            Ok(())
        })?;
        let actual_draft = context_hash(&mut fence, &mut long)?;
        fence.check(
            &mut long,
            actual_draft == final_draft,
            "accepted retained feature/KV drift",
        )?;
        fence.work(&mut long, |long| long.finish())?;

        fence.work(&mut long, |long| long.bind(2))?;
        let restored = disk.restore(&mut long, tokens, 32, true, None)?;
        fence.check(&mut long, restored == Some(cut), "SSD prefix cursor")?;
        check_restored_prefix(&mut fence, &mut long, cut, &prefix_state, &draft_hash)?;
        prefill(&mut fence, &mut long, tokens, end)?;
        let pending = sample(&mut fence, &mut long, &mut sampler, policy)?;
        fence.check(&mut long, pending == anchor, "SSD pending selection")?;
        let restored_proposal = proposal_hash(&proposals(&mut fence, &mut long, policy)?);
        fence.check(
            &mut long,
            restored_proposal == proposed,
            "SSD proposal drift",
        )?;
        let mut committed = 0;
        let mut rounds = 0;
        while committed < 16 {
            let proposal = proposals(&mut fence, &mut long, policy)?;
            let rows = (16 - committed).min(8);
            let pending = fence.work(&mut long, |long| {
                long.sessions[2].pending.context("SSD pending")
            })?;
            let mut input = vec![pending];
            input.extend_from_slice(&proposal.tokens[..rows - 1]);
            let actual = fence.verify(&mut long, &input, policy, |accepted| Ok(accepted.len()))?;
            let keep = actual.len();
            fence.work(&mut long, |long| {
                ensure!(
                    keep > 0
                        && keep <= rows
                        && actual == serial[committed + 1..=committed + keep]
                        && long.state(end + committed + keep)? == states[committed + keep - 1],
                    "SSD speculative continuation drift"
                );
                Ok(())
            })?;
            committed += keep;
            rounds += 1;
        }
        let actual_draft = context_hash(&mut fence, &mut long)?;
        fence.check(
            &mut long,
            actual_draft == final_draft,
            "SSD accepted feature/KV drift",
        )?;
        fence.work(&mut long, |long| long.finish())?;
        if case == 2 {
            fence.work(&mut long, |_| {
                if rank == 2 {
                    corrupt_tail(&path, cut)
                } else {
                    Ok(())
                }
            })?;
            let failed = disk.restore(&mut long, tokens, 32, true, None)?;
            fence.work(&mut long, |long| {
                ensure!(
                    failed.is_none()
                        && long.healthy
                        && long.sessions[2].tokens.is_empty()
                        && long.sessions[2].pending.is_none()
                        && long.memory.book().slot(2)?.phase() == crate::cache_pages::Phase::Idle,
                    "partial SSD companion published or poisoned cold fallback"
                );
                Ok(())
            })?;
            fence.work(&mut long, |long| long.start_mode(end, 32, true))?;
            prefill(&mut fence, &mut long, tokens, cut)?;
            prefill(&mut fence, &mut long, tokens, end)?;
            let pending = sample(&mut fence, &mut long, &mut sampler, policy)?;
            fence.check(&mut long, pending == anchor, "cold fallback pending drift")?;
            for row in 0..16 {
                fence.work(&mut long, |long| long.step(end + row, serial[row]))?;
                let pending = sample(&mut fence, &mut long, &mut sampler, policy)?;
                fence.check(
                    &mut long,
                    pending == serial[row + 1],
                    "cold fallback sampled drift",
                )?;
            }
            let actual_draft = context_hash(&mut fence, &mut long)?;
            fence.work(&mut long, |long| {
                ensure!(
                    actual_draft == final_draft && long.state(end + 16)? == states[15],
                    "cold fallback target/drafter drift"
                );
                Ok(())
            })?;
            fence.work(&mut long, |long| long.finish())?;
        }
        cases.push(json!({"prefix":cut,"prompt":end,"snapshot_gpu_bytes":cut.min(WINDOW)*5120,"serial_tokens":16,"ssd_speculative_rounds":rounds,"exact_ram_prefix":true,"exact_ssd_prefix":true,"exact_restored_proposals":true,"exact_clipped_state_pending":true,"exact_retained_draft":true,"nonempty_native_hold":true,"metadata_rejections":5,"rank2_late_ssd_corruption_cold_fallback":case==2}));
        eprintln!("retained speculative caches exact prefix={cut}");
    }
    let parser_rejections = parser_rejections(&mut fence, &mut long, &mut sampler, source, policy)?;
    let cooperative = super::super::disk::run_companion_read_probe(
        &mut long,
        source,
        &path.with_extension("companion-read"),
    )?;
    Ok(
        json!({"rank":rank,"temperature":temperature,"prefill_rows":8192,"backlog_bytes":long.speculation_backlog_bytes(),"cases":cases,
        "cooperative_companion_reads":cooperative,"parser_rejections":parser_rejections,"ssd_evidence":path,
        "scope":"Long-owned retained drafter/verifier, bounded accepted-feature backlogs, real RAM/SSD pure-prefix companions, greedy or sampled clipped and DFlash continuation, native background hold; isolated probe, serving disabled"}),
    )
}
