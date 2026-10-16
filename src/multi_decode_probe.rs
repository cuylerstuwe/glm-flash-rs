//! Exact model-state qualification for independent speculative sequences.
use super::*;

struct Reference {
    tokens: Vec<i32>,
    states: Vec<String>,
    contexts: Vec<String>,
    logits: Vec<String>,
}

fn begin(
    fence: &mut Fence,
    long: &mut Long<'_>,
    source: &[i32],
    slot: usize,
    end: usize,
) -> Result<()> {
    fence.work(long, |long| {
        long.bind(slot)?;
        long.start_mode(end, 32, false)
    })?;
    let tokens: Vec<_> = (0..end)
        .map(|i| source[(slot * 71 + i) % source.len()])
        .collect();
    prefill(fence, long, &tokens, end)
}

pub(super) fn run(
    fence: &mut Fence,
    long: &mut Long<'_>,
    source: &[i32],
    native: &Path,
) -> Result<Value> {
    let sampled = std::env::var("ALASKA_MULTI_PROBE_SAMPLED").as_deref() == Ok("1");
    let long_probe = std::env::var("ALASKA_MULTI_PROBE_LONG").as_deref() == Ok("1");
    let policy = |slot: usize| {
        Sampling::new(
            731 + slot as u64,
            if sampled {
                [1., 0.8, 1.2, 1.][slot]
            } else {
                0.
            },
            20,
            [0.95, 0.8, 1., 0.9][slot],
        )
    };
    let mut sampler = Candidates::new(&long.engine.device, native)?;
    let window: usize = std::env::var("ALASKA_MULTI_PROBE_WINDOW")
        .unwrap_or_else(|_| "4".into())
        .parse()?;
    fence.work(long, |long| {
        long.with_speculation(|owner, long| {
            owner.verifier.enable_multi_audit();
            owner.verifier.prewarm_multi(long, window)?;
            Ok(())
        })?
        .context("multi verifier owner")
    })?;
    let mut cases = Vec::new();
    let layouts = if long_probe {
        ensure!(window == 4, "long multi probe window");
        vec![
            (vec![0, 2], vec![511, 1021], vec![1, 4]),
            (
                vec![0, 1, 2, 3],
                vec![2047, 2051, 4093, 4097],
                vec![4, 2, 3, 1],
            ),
            (
                vec![0, 1, 2, 3],
                vec![8190, 8192, 8193, 8195],
                vec![1, 4, 2, 3],
            ),
            (vec![0, 1], vec![33000, 66000], vec![3, 4]),
            (vec![0, 1], vec![33008, 66012], vec![4, 2]),
            (vec![0, 2, 3], vec![4096, 8191, 32769], vec![0, 2, 4]),
        ]
    } else if window == 2 || window == 3 {
        let w = window;
        vec![
            (vec![0, 1, 2, 3], vec![64, 65, 128, 129], vec![1, w, 1, w]),
            (vec![0, 1, 2, 3], vec![253, 254, 255, 256], vec![w, 1, w, 1]),
            (vec![0, 2], vec![503, 512 - w], vec![1, w]),
            (vec![1, 3], vec![127, 130], vec![w - 1, w]),
            (vec![0, 1, 2, 3], vec![63, 64, 65, 66], vec![0, w, 1, w]),
            (vec![0, 1, 2, 3], vec![68, 69, 70, 71], vec![0, 1, w, 1]),
        ]
    } else if window == 8 {
        vec![
            (vec![0, 1], vec![64, 65], vec![1, 8]),
            (vec![2, 3], vec![253, 254], vec![7, 2]),
            (vec![0, 2], vec![499, 504], vec![6, 8]),
            (vec![1, 3], vec![127, 130], vec![4, 5]),
            (vec![0, 1], vec![63, 64], vec![0, 3]),
            (vec![2, 3], vec![68, 69], vec![0, 8]),
        ]
    } else {
        vec![
            (vec![0, 1, 2, 3], vec![64, 65, 128, 129], vec![1, 2, 3, 4]),
            (vec![0, 1, 2, 3], vec![253, 254, 255, 256], vec![4, 3, 2, 1]),
            (vec![0, 2], vec![499, 508], vec![3, 4]),
            (vec![1, 3], vec![127, 130], vec![2, 1]),
            (vec![0, 1, 2, 3], vec![63, 64, 65, 66], vec![0, 4, 1, 3]),
            (vec![0, 1, 2, 3], vec![68, 69, 70, 71], vec![0, 2, 3, 4]),
            (vec![0, 1, 3], vec![65, 130, 255], vec![4, 2, 3]),
            (vec![0, 2, 3], vec![128, 499, 508], vec![1, 3, 4]),
            (vec![1, 2, 3], vec![66, 67, 68], vec![0, 2, 4]),
        ]
    };
    for (case, (slots, ends, keeps)) in layouts.into_iter().enumerate() {
        let mut references = Vec::new();
        for (&slot, &end) in slots.iter().zip(&ends) {
            begin(fence, long, source, slot, end)?;
            if sampled {
                sample(fence, long, &mut sampler, policy(slot)?)?;
            }
            let mut tokens = vec![long.sessions[slot].pending.context("reference pending")?];
            let mut states = vec![fence.work(long, |long| long.state(end))?];
            let mut contexts = vec![context_hash(fence, long)?];
            let mut logits = Vec::new();
            for offset in 0..12 {
                let mut next = fence.work(long, |long| long.step(end + offset, tokens[offset]))?;
                if sampled {
                    next = sample(fence, long, &mut sampler, policy(slot)?)?;
                }
                tokens.push(next);
                states.push(fence.work(long, |long| long.state(end + offset + 1))?);
                contexts.push(context_hash(fence, long)?);
                logits.push(fence.work(long, |long| long.logits())?);
            }
            references.push(Reference {
                tokens,
                states,
                contexts,
                logits,
            });
            fence.work(long, |long| long.finish())?;
        }
        for ((&slot, &end), reference) in slots.iter().zip(&ends).zip(&references) {
            begin(fence, long, source, slot, end)?;
            if sampled {
                sample(fence, long, &mut sampler, policy(slot)?)?;
            }
            fence.check(
                long,
                long.sessions[slot].pending == Some(reference.tokens[0]),
                "multi cold pending",
            )?;
            let state = fence.work(long, |long| long.state(end))?;
            fence.check(
                long,
                state == reference.states[0],
                "multi cold target state",
            )?;
            // The production caller flushes each feature backlog during proposal.
            let context = context_hash(fence, long)?;
            fence.check(
                long,
                context == reference.contexts[0],
                "multi cold drafter state",
            )?;
        }
        let prefix = if (case < 2 && !long_probe) || (long_probe && case == 0) {
            2
        } else {
            0
        };
        if prefix > 0 {
            for (&slot, reference) in slots.iter().zip(&references) {
                fence.work(long, |long| long.bind(slot))?;
                fence.verify(
                    long,
                    &reference.tokens[..prefix],
                    policy(slot)?,
                    |accepted| Ok(accepted.len()),
                )?;
            }
        }
        let inputs: Vec<_> = references
            .iter()
            .map(|r| r.tokens[prefix..prefix + window].to_vec())
            .collect();
        let long_rank = long.engine.rank;
        let policies: Vec<_> = slots
            .iter()
            .map(|&slot| policy(slot))
            .collect::<Result<_>>()?;
        let result = long.multi_policy_step(
            &fence.control,
            fence.dispatch(),
            &slots,
            &inputs,
            &vec![false; slots.len()],
            &policies,
            |row, accepted| {
                ensure!(
                    accepted.len() == window,
                    "multi forced reference acceptance"
                );
                if keeps[row] == 0 && ((case != 5 && case != 8) || long_rank == 2) {
                    return Err(anyhow::Error::new(ParserFailure));
                }
                Ok(if keeps[row] == 0 { window } else { keeps[row] })
            },
        );
        let outputs = fence.complete(long, result)?;
        let logit_hashes = fence.work(long, |long| {
            long.with_speculation(|owner, _| {
                owner.verifier.multi_logits_hashes(slots.len() * window)
            })?
            .context("multi verifier audit")
        })?;
        for (row, ((&slot, &end), (&keep, reference))) in slots
            .iter()
            .zip(&ends)
            .zip(keeps.iter().zip(&references))
            .enumerate()
        {
            fence.work(long, |long| long.bind(slot))?;
            fence.check(
                long,
                outputs[row] == reference.tokens[prefix + 1..=prefix + keep],
                "multi accepted IDs",
            )?;
            fence.check(
                long,
                long.sessions[slot].pending == Some(reference.tokens[prefix + keep]),
                "multi pending selection",
            )?;
            let state = fence.work(long, |long| long.state(end + prefix + keep))?;
            fence.check(
                long,
                state == reference.states[prefix + keep],
                "multi committed full state",
            )?;
            let context = context_hash(fence, long)?;
            fence.check(
                long,
                context == reference.contexts[prefix + keep],
                "multi accepted drafter state",
            )?;
            for token in 0..window {
                fence.check(
                    long,
                    logit_hashes[row * window + token] == reference.logits[prefix + token],
                    "multi full target logits",
                )?;
            }
            if keep > 0 {
                for offset in prefix + keep..prefix + keep + 2 {
                    let mut next = fence.work(long, |long| {
                        long.step(end + offset, reference.tokens[offset])
                    })?;
                    if sampled {
                        next = sample(fence, long, &mut sampler, policy(slot)?)?;
                    }
                    fence.check(
                        long,
                        next == reference.tokens[offset + 1],
                        "multi serial continuation ID",
                    )?;
                    let state = fence.work(long, |long| long.state(end + offset + 1))?;
                    fence.check(
                        long,
                        state == reference.states[offset + 1],
                        "multi serial continuation state",
                    )?;
                }
                let context = context_hash(fence, long)?;
                fence.check(
                    long,
                    context == reference.contexts[prefix + keep + 2],
                    "multi serial continuation drafter",
                )?;
                fence.work(long, |long| long.finish())?;
            } else {
                fence.work(long, |long| long.cancel())?;
            }
        }
        cases.push(
            json!({"case":case,"slots":slots,"prompt_lengths":ends,"accepted":keeps,
            "outputs_full_logits_state_drafter_and_continuation_exact":true}),
        );
    }
    Ok(
        json!({"rank":long.engine.rank,"sampled":sampled,"long_context":long_probe,"cases":cases,"scope":"Independent request policies/seeds, target-sampled or greedy reference windows, every accepted prefix length, parity and nonadjacent slots, attention chunk/bucket boundaries and graph reuse, parser discard, complete state/logits/accepted features and normal serial continuation."}),
    )
}
