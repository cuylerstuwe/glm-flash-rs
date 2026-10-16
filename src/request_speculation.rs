//! Opt-in retained speculative dispatch inside the existing native owner.
//! Host planning is bounded; all parser advancement precedes GPU publication.
use super::*;
use crate::{
    depth_policy::{Costs, DepthPolicy},
    prompt_lookup::Lookup,
    sampling::Sampling,
};

struct Session {
    depth: DepthPolicy,
    lookup: Option<Lookup>,
}
pub(super) struct State {
    costs: Costs,
    lookup: bool,
    greedy: bool,
    sessions: BTreeMap<u64, Session>,
}
impl State {
    pub(super) fn accepts(&self, policy: Sampling) -> bool {
        !policy.greedy() || self.greedy
    }
    pub(super) fn remove(&mut self, id: u64) {
        self.sessions.remove(&id);
    }
}
pub(super) struct Round {
    pub(super) tokens: Vec<i32>,
    pub(super) elapsed: Duration,
    pub(super) stopped: bool,
    pub(super) output_error: bool,
}
struct Plan {
    lookup: bool,
    proposals: Vec<i32>,
    backlog: usize,
    room: usize,
    policy: Sampling,
    pending: i32,
}
impl Plan {
    fn digest(&self, queue: [u8; 32], id: u64) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"request-speculative-plan-v1");
        h.update(queue);
        h.update(id.to_le_bytes());
        h.update([u8::from(self.lookup)]);
        h.update(self.policy.encode());
        h.update(self.pending.to_le_bytes());
        for n in [self.backlog, self.room, self.proposals.len()] {
            h.update((n as u64).to_le_bytes());
        }
        for token in &self.proposals {
            h.update(token.to_le_bytes());
        }
        h.finalize().into()
    }
}
impl Scheduler {
    pub(super) fn multi_dflash_window(&mut self, id: u64, probabilities: &[f64]) -> Result<()> {
        self.speculation
            .as_mut()
            .context("multi draft policy")?
            .sessions
            .get_mut(&id)
            .context("multi draft owner")?
            .depth
            .fixed_window(probabilities)
    }
    pub(super) fn multi_lookup(&mut self, id: u64, window: usize) -> Result<Vec<i32>> {
        let request = self.queue.request(id)?;
        let state = self
            .speculation
            .as_mut()
            .context("multi speculation policy")?;
        if !state.sessions.contains_key(&id) {
            ensure!(
                state.sessions.len() < crate::cache_pages::SLOTS,
                "multi lookup owner bound"
            );
            state.sessions.insert(
                id,
                Session {
                    depth: DepthPolicy::new(state.costs.clone())?,
                    lookup: if state.lookup {
                        Some(Lookup::new(&request.tokens)?)
                    } else {
                        None
                    },
                },
            );
        }
        let session = state.sessions.get_mut(&id).unwrap();
        let found = if let Some(lookup) = &mut session.lookup {
            lookup.plan_with_eos(
                &request.outputs,
                window - 1,
                &session.depth,
                request.stop_eos(),
            )?
        } else {
            Vec::new()
        };
        Ok(if found.len() == window - 1 {
            found
        } else {
            Vec::new()
        })
    }
    pub(super) fn multi_feedback(
        &mut self,
        id: u64,
        lookup: bool,
        window: usize,
        backlog: usize,
        accepted: usize,
    ) -> Result<()> {
        let session = self
            .speculation
            .as_mut()
            .context("multi feedback policy")?
            .sessions
            .get_mut(&id)
            .context("multi feedback owner")?;
        session.depth.record(lookup, window, backlog, accepted)?;
        if lookup {
            session
                .lookup
                .as_mut()
                .context("multi lookup feedback")?
                .record(window, accepted)?;
        }
        Ok(())
    }
    pub(super) fn install_speculation(
        &mut self,
        long: &Long<'_>,
        costs: Costs,
        lookup: bool,
        greedy: bool,
    ) -> Result<()> {
        let valid = (|| -> Result<()> {
            ensure!(
                self.queue.count() == 0
                    && self.speculation.is_none()
                    && !self.audit
                    && long.speculation_identity().is_some(),
                "speculative scheduler installation"
            );
            costs.validate()?;
            ensure!(
                costs.verify.len() == if lookup { 16 } else { 8 },
                "speculative service cost width"
            );
            Ok(())
        })();
        let mut h = Sha256::new();
        h.update(b"request-speculation-v2");
        h.update([u8::from(lookup), u8::from(greedy)]);
        h.update(serde_json::to_vec(&costs)?);
        ensure!(
            self.wire
                .control
                .agree(long, 859, h.finalize().into(), valid.is_ok())?,
            "speculative scheduler configuration disagreed"
        );
        valid?;
        self.speculation = Some(State {
            costs,
            lookup,
            greedy,
            sessions: BTreeMap::new(),
        });
        Ok(())
    }
    pub(super) fn speculative_round(&mut self, long: &mut Long<'_>, id: u64) -> Result<Round> {
        let started = Instant::now();
        crate::round_timing::enter(crate::round_timing::Phase::Plan);
        let planned = (|| -> Result<Plan> {
            let request = self.queue.request(id)?;
            ensure!(
                request.done == request.tokens.len()
                    && request.drafts()
                    && !request.has_grammar()
                    && !request.paused
                    && request.cancellation().is_none()
                    && request.next_output() == crate::thinking_budget::Next::Sample,
                "speculative request owner"
            );
            let policy = request
                .sampling()
                .filter(|p| {
                    self.speculation
                        .as_ref()
                        .is_some_and(|state| state.accepts(*p))
                })
                .context("speculative request policy disabled/unavailable")?;
            let slot = request.slot.context("speculative request slot")?;
            let pending = *request
                .outputs
                .last()
                .context("speculative pending output")?;
            ensure!(
                long.active == slot && long.sessions[slot].pending == Some(pending),
                "speculative pending owner"
            );
            let room = (request.output - request.outputs.len()).min(crate::kda_window::MAX_ROWS);
            ensure!(room >= 2, "speculative output room");
            let state = self
                .speculation
                .as_mut()
                .context("speculative scheduler unavailable")?;
            if !state.sessions.contains_key(&id) {
                ensure!(
                    state.sessions.len() < crate::cache_pages::SLOTS,
                    "speculative request count bound"
                );
                let lookup = if state.lookup {
                    Some(Lookup::new(&request.tokens)?)
                } else {
                    None
                };
                state.sessions.insert(
                    id,
                    Session {
                        depth: DepthPolicy::new(state.costs.clone())?,
                        lookup,
                    },
                );
            }
            let session = state.sessions.get_mut(&id).unwrap();
            let proposals = if let Some(lookup) = &mut session.lookup {
                lookup.plan_with_eos(
                    &request.outputs,
                    room - 1,
                    &session.depth,
                    request.stop_eos(),
                )?
            } else {
                Vec::new()
            };
            Ok(Plan {
                lookup: !proposals.is_empty(),
                proposals,
                backlog: long.speculation_backlog_rows(slot)?,
                room,
                policy,
                pending,
            })
        })();
        let queue = self.queue.digest();
        // The first collective of either path below, the drafter or verifier
        // dispatch agreement, covers this plan's digest. A rank whose planning
        // failed joins that exchange with a failure instead.
        let mut plan = match planned {
            Ok(plan) => plan,
            Err(error) => {
                self.wire.control.agree(long, 860, [0; 32], false)?;
                return Err(error);
            }
        };
        let digest = plan.digest(queue, id);
        crate::round_timing::enter(crate::round_timing::Phase::Draft);
        if !plan.lookup {
            // The verifier dispatch agreement covers the chosen tokens and the
            // drafter state; a rank whose proposal failed joins it with a failure.
            let chosen = long
                .propose_speculation(&self.wire.control, digest, plan.policy)
                .and_then(|proposed| {
                    let most = (plan.room - 1).min(7);
                    ensure!(
                        proposed.tokens.len() >= most && proposed.probabilities.len() >= most,
                        "drafter returned incomplete proposals"
                    );
                    let depth = &mut self
                        .speculation
                        .as_mut()
                        .unwrap()
                        .sessions
                        .get_mut(&id)
                        .unwrap()
                        .depth;
                    let keep = depth.dflash(&proposed.probabilities[..most])?;
                    ensure!(keep > 0 && keep <= most, "drafter chosen depth");
                    Ok(proposed.tokens[..keep].to_vec())
                });
            match chosen {
                Ok(tokens) => {
                    long.drafted();
                    plan.proposals = tokens;
                }
                Err(error) => {
                    self.wire.control.agree(long, 862, [0; 32], false)?;
                    return Err(error);
                }
            }
        }
        crate::round_timing::enter(crate::round_timing::Phase::VerifyDispatch);
        let n = plan.proposals.len() + 1;
        let mut tokens = [0; crate::kda_window::MAX_ROWS];
        tokens[0] = plan.pending;
        tokens[1..n].copy_from_slice(&plan.proposals);
        let request = self.queue.request(id)?;
        let stream = self.output_streams.get_mut(&id);
        let mut stopped = false;
        let mut accepted = 0;
        let verified = long.verify_speculation_eos(
            &self.wire.control,
            plan.digest(queue, id),
            &tokens[..n],
            plan.policy,
            request.stop_eos(),
            |samples| {
                accepted = samples.len();
                let keep = request.sampled_prefix(samples)?;
                ensure!(
                    keep > 0,
                    "forced or terminal output entered sampled verifier"
                );
                if let Some(stream) = stream {
                    let mut ids = [0u32; crate::kda_window::MAX_ROWS];
                    for (dst, src) in ids.iter_mut().zip(&samples[..keep]) {
                        *dst = *src as u32;
                    }
                    let clipped = stream.clip_accepted(&ids[..keep]).map_err(|_| {
                        anyhow::Error::new(super::super::speculation::ParserFailure)
                    })?;
                    stopped = clipped.ended && stream.text.stopped;
                    Ok(clipped.keep)
                } else {
                    Ok(keep)
                }
            },
        );
        if let Err(error) = &verified
            && let Some(proof) = error.downcast_ref::<super::super::speculation::DiscardProof>()
        {
            // This type can only originate after every rank reaches the same
            // pre-publication verifier boundary. No native error is recoverable.
            let discarded = long.discard_rejected_output(proof);
            long.agree_speculation_work(&self.wire.control, 864, queue, discarded)?;
            return Ok(Round {
                tokens: Vec::new(),
                elapsed: started.elapsed(),
                stopped: false,
                output_error: true,
            });
        }
        let completed = verified.and_then(|tokens| {
            let session = self
                .speculation
                .as_mut()
                .unwrap()
                .sessions
                .get_mut(&id)
                .unwrap();
            session
                .depth
                .record(plan.lookup, n, plan.backlog, accepted)?;
            if plan.lookup {
                session.lookup.as_mut().unwrap().record(n, accepted)?;
            }
            Ok(tokens)
        });
        crate::round_timing::enter(crate::round_timing::Phase::Complete);
        let tokens = long.agree_speculation_work(&self.wire.control, 863, queue, completed)?;
        Ok(Round {
            tokens,
            elapsed: started.elapsed(),
            stopped,
            output_error: false,
        })
    }
}
