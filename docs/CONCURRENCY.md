# Concurrent foreground inference

`ALASKA_MULTI_GREEDY=6` shares target projections, expert work, and verification across two to four compatible foreground requests. Each request contributes a four-token verification window, with at most 16 target rows in a shared batch. Attention and recurrent state, pending tokens, drafter context, sampling policy and random cursor, parser, and output budget remain independently owned. Batching uses the same target weights, precision, and sampling arithmetic as single-request execution.

Despite the variable name, mode 6 supports both greedy and ordinary sampled requests, including long contexts. Set the same value on all ranks. `0` or an absent variable selects the non-batched path. Values 1–5 are experimental variants, not recommended deployment choices. Keep the supplied speculative policy and supporting flags together; this switch does not enable speculation when the service configuration has disabled it.

## Scheduling and transitions

A single request uses the adaptive speculative path. As requests arrive, finish, or change eligibility, the scheduler reconsiders cohorts at safe work boundaries. Each compatible cohort uses the next runnable request's turn. Unsupported peers keep their normal turns instead of disabling batching for every request.

Sampled cohort members may have different positive temperatures, top-p values, and seeds, but must share the candidate layout and fit the bounded GPU sampler. Greedy and sampled requests form separate families. Grammar-constrained requests, unusual sampling overrides that do not fit, forced thinking-budget tokens, partial output tails, and attention-boundary transitions use per-request execution when necessary. They may rejoin a compatible cohort when eligible. This is not universal batching of every request combination.

Foreground and background remain distinct priority classes. Multiple foreground requests may compute concurrently; background model work yields at safe boundaries and retains exact progress. A few seconds of initial handoff can be appropriate for large background prefill pieces. Background requests use per-request scheduling.

The engine has four resident slots total. A paused background owner still occupies one slot, so three foreground requests can be resident while a fourth waits. Shared verification does not add slots or remove the need to budget the live token pool. Preserve the documented total-context/output limits and per-rank RAM/SSD bounds.

## Memory and performance

Shared verification retains separate pinned replay counters and reusable per-request state. Graph keys include each request's relevant attention geometry. The multi-request graph cache is bounded to 96 entries, 1,048,576 aggregate graph nodes, and 192 remembered shapes. These are graph-count limits, not byte limits; measure actual memory at long context and with all resident slots occupied. No extra target quantization is required.

The primary objective is useful sampled interactive work while preserving isolated generation and prefill. C4 is aggregate throughput, not each stream's rate. Batching can improve total completion time without making every member of every mixed group finish sooner. Mixed sampled, grammar-constrained, and greedy requests may receive different amounts of GPU work per turn. Measure both overall completion and each member's latency; batching does not promise equal time slices or a speedup for every member.

With a retained background slot, a four-foreground group can also take longer than the same group with all four slots available. Separate this capacity wait from the initial safe background handoff and from foreground execution after admission. Measure the intended request mix rather than assuming equal time slices or universal per-member speedups.

## Qualification

Use [the full acceptance gates](VALIDATION.md) on the final binary/configuration/artifact set. Published measurements do not qualify an installation automatically.

1. Compare full model logits, recurrent/attention state, drafter state, accepted prefixes, and subsequent serial continuation against independent serial references. Exercise every partial acceptance length, nonadjacent slots, parser rejection of one owner, and unchanged peers.
2. Verify independent random ownership with different seeds, positive temperatures and top-p values, then repeat for greedy cohorts. Test actual HTTP/SSE output, reasoning, tools, EOS, stop strings, output clipping, forced thinking-budget transitions, and grammar fallback.
3. Cover 1→2→3→4→1 request transitions, mixed eligible/ineligible groups, cancellation of either an eligible member or a fallback member, short tails, and requests arriving during prefill or decode. Confirm no starvation or cross-request progress accounting.
4. Exercise dense/sparse attention boundaries, unequal long contexts, live/RAM/SSD prefix reuse, restart, corruption after partial staging, and full-budget pressure without idle gaps. Verify near-window concurrent generation separately from broad long-context semantic quality.
5. Compare foreground prefill, first output, generation, and total completion with background save/restore/prefill active. Check frozen background compute/copy progress under holds, exact resumption, and cancellation while held. Report initial handoff, slot admission, and admitted execution separately.
6. Run matched isolated prefill and sampled generation controls, plus capped C1/C2/C4 and representative long agentic concurrency. Fix diagnostic seeds only for reproducibility; do not introduce a fixed production seed. Retain settings, counts, cache state, raw timings, and output comparisons privately.
7. Test rank/sidecar failure with multiple active clients, explicit terminal errors, coordinated retirement/restart, and actual rollback. Live exact pause/resume is not crash persistence of in-flight requests.

The `model-retained-speculation-roce` and `model-retained-speculation-nccl` diagnostics include the multi-request state matrix. Invoke all four ranks together with the rank-specific target manifest, drafter manifest, kernel/native/NCCL paths, master address, and a synthetic token-set JSON file as described by the CLI argument check in `src/main.rs`.

Set `ALASKA_MULTI_PROBE=1` and `ALASKA_MULTI_PROBE_WINDOW=4` for this matrix; add `ALASKA_MULTI_PROBE_SAMPLED=1` for independent sampled policies and `ALASKA_MULTI_PROBE_LONG=1` for long/boundary layouts. The fixture must contain at least 8,320 valid model tokens. Leave unrelated fault/temperature overrides unset; the matrix installs its own per-request policies. Run both sampled and greedy cases with the intended transport and retain every rank's results. These diagnostics perform expensive state readbacks and are not throughput benchmarks. Never enable `ALASKA_MULTI_PROBE*` variables in the production service environment.
