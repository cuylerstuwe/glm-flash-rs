# Qualification and performance acceptance

Each installation requires qualification on its intended cluster. Verify the final binary, artifacts, environment, configuration, and real client path together. Store raw evidence privately; published examples should use synthetic prompts and omit deployment identities.

See [quality measurements and methodology](QUALITY.md) for the recorded fidelity results and the distinction between distribution similarity, exact execution, and answer correctness.

## Required gates

| Report key | Evidence needed |
| --- | --- |
| `compatibility` | Tokenization/template agreement; HTTP and SSE; tools and argument assembly; reasoning/history; sampling overrides; grammar; configured vision including HTTPS; authentication and the actual client/router adapter |
| `precision_quality` | Target/drafter/model/kernel identities and target precision map; deterministic matched token/logit/state comparisons; sampled serial-vs-speculative checks; representative reasoning/coding/tool/vision quality checks |
| `context_output` | Exact near-window requests up to the 1,048,576 total policy, a separate 131,072-output test with room for input, limit rejection, long decoding and bounded memory |
| `cache_pressure` | RAM and SSD reuse, mixed/fragmented prefixes, more sessions than retained slots, eviction, no-gap writer load, completed persistent entries, restart restore, corruption handling, and bounded memory/storage |
| `native_priority` | Foreground arrival during background prefill, generation, restore, and save/copy; safe yield and exact retained resumption; multiple lease owners/epochs/expiry; cancellation while held; bounded queues and no starvation |
| `fault_stability` | Disconnect/cancel at multiple phases, malformed requests, sidecar/rank/transport failure, disk errors, coherent service retirement/restart, no stale healthy endpoint, no orphan GPU work, and an extended mixed workload |
| `matched_performance` | Comparable prefill, first-output latency, generation and total elapsed time, under normal sampling and representative work; cache/graph/thermal state and initial handoff reported explicitly |
| `reproducible_artifacts` | Source/build/dependency pins, identical rank binaries, a reproducible build check, resolved shared libraries, final config/environment digests, coordinated deployment and tested rollback |

CPU/unit tests and small component probes establish only their own scope. Use the test modules and diagnostic CLI in `src/main.rs` to identify available checks; do not label a simulated broker test as a four-GPU model test. Qualification requires evidence from the installation being tested.

## Reasoning effort qualification

Test both `reasoning_effort` and `chat_template_kwargs.reasoning_effort` against [the API contract](API.md#sampling-reasoning-tools-and-images). The renderer tests and `fixtures/chat/rendering.json` check exact template output; they do not substitute for checking the serving interface.

- Exercise every supported name, case/whitespace normalization, omitted values, and nulls. With the supplied configuration, omission defaults to High, `minimal` keeps thinking enabled at Low, `medium` selects High, and `xhigh` selects Max.
- Compare each alias with its canonical name using the same prompt, sampling settings, and diagnostic seed. Verify prompt token IDs, thinking state, reasoning, and final content; rerendering a normalized request must retain the same prompt.
- Verify valid nested effort takes precedence over valid top-level effort, null leaves the other field effective, and explicit `enable_thinking` keeps its documented override behavior.
- Send unknown names, empty strings, and non-string/non-null values in either field to chat tokenization and chat completion. Require HTTP 400, including when the invalid field is shadowed by a valid field or thinking is disabled.
- Repeat representative checks through the actual client/router path so request translation cannot silently change the requested effort. Include sampled Low, High, and Max model requests alongside the template checks.

## Prefill qualification

Before accepting the cuBLAS dispatch on a new runtime tuple, run the synthetic shape comparison and timing diagnostics on an exclusively owned GPU:

```text
glm-flash-rs prefill-blas-exact /opt/glm-flash-rs/artifacts/kernels /opt/glm-flash-rs/artifacts/native
glm-flash-rs prefill-blas /opt/glm-flash-rs/artifacts/kernels /opt/glm-flash-rs/artifacts/native
```

The exactness probe compares the direct GPU kernels with cuBLAS at every multiple of 64 through 8192 rows and additional odd/boundary sizes. Differences are expected at some sizes and excluded shapes; verify that no differing case is selected by `blas_rows` in `src/prefill.rs`. The timing probe uses synthetic operands and reports GPU event times. Neither probe establishes complete-model equivalence.

Follow with paired `ALASKA_PREFILL_BLAS=0/1` model runs using identical inputs and sampling, comparing both reasoning and answer bytes. Exercise short and partial chunks, odd prompt lengths, 8192-row boundaries, long cold prefill, cached follow-ups, and near-window restore. Separately compare the row-split scheduling path against the non-split execution path, including feature-tap continuations, cancellation, and exact native pause/resume at safe chunk boundaries. Measure memory and foreground handoff with background prefill active. Retain raw evidence privately and requalify whenever the driver, cuBLAS, native kernel pack, or scheduling implementation changes.

## Correctness and performance comparisons

For shared foreground verification, follow [the concurrency qualification matrix](CONCURRENCY.md#qualification). Compare `ALASKA_MULTI_GREEDY=0/6` on the same final binary. Include independent sampled policies, long contexts, mixed constraints, and transitions between one and several active requests. Aggregate throughput alone cannot establish per-request fairness or preservation of isolated speed.

Use the same target weights/precision, tokenizer, template/history, thinking effort, input IDs, output budget, seed where a deterministic check needs one, and sampling settings. Normal usage is temperature 1, top-p 0.95, effective top-k 20, no fixed production seed. Include High and Max reasoning, prose, coding, tool definitions/results, full-file edits, and long continuations. Greedy prose/code remains a controlled diagnostic, but optimizing it must not worsen normal sampled foreground work.

Compare serial target generation with speculative generation, including low-acceptance cases, short responses, stop strings/tokens, tool termination, and constrained outputs. For paths designed to be exact, compare token IDs, output bytes, retained state/logits, and continuation results—not just plausibility. For broad quality comparisons use consistent tasks and inspect failures; aggregate speed cannot excuse changed target precision or incorrect answers.

Record separately:

- Input/output token counts and actual uncached prefill work.
- End-to-end first-output latency and prefill throughput.
- Generation throughput and total request elapsed time, including speculation overhead.
- Cold, RAM-hit, SSD-hit, and retained live-cache cases.
- Warmup/graph preparation, diagnostic hashing/readback settings, memory peaks, clocks/thermals, and competing work.

Use repeats and paired/interleaved ordering where practical. Small gains can add up; report their uncertainty and check the combined configuration. Avoid mixing different prompt families, output lengths, or sampling policies into a misleading single speed claim. A small greedy-only deficit is not grounds for endless tuning if normal sampled work is qualified; preserve the deficit in the evidence.

## Foreground contention

Compare foreground requests alone with the same requests while background restore/save/prefill is active. Use identical inputs and settings in both cases. Measure two intervals:

1. Arrival to safe foreground admission, including the accepted initial background handoff.
2. Foreground work after admission: prefill, first output, generation, and completion.

Report background prefill throughput and total useful work alongside both. A few seconds of initial handoff is acceptable when large safe pieces improve useful work. Do not impose a subsecond cutoff or shrink boundaries solely to improve that metric. Conversely, a quick handoff is not a win if foreground execution is then slowed by unbounded background copying. Throttle/defer lower-priority work where needed while retaining correctness and progress.

Exercise foreground arrival at several points during SSD read/copy/write and background prefill, including full configured RAM/SSD budgets. Validate exact background continuation, cancel during holds and restores, and test multiple queued foreground requests. Account for allocation and I/O costs throughout restoration, including work outside the measured copy interval.

## Deployment report format

`src/production_qualification.rs` is the authoritative schema/check implementation. A completed version-1 report contains:

- `version: 1`, `accepted: true`, the exact `binary_sha256`, `context_tokens: 1048576`, and `max_output_tokens: 131072`.
- A `gates` object with each key above containing `passed: true` and a nonempty `evidence` reference to real completed work.
- A four-entry `deployment` array indexed by rank, each with `config_sha256` and `environment_sha256`.

The configuration digest is SHA-256 over compact JSON with all object keys recursively sorted and only the top-level `qualification` field removed. Match Rust/serde JSON number serialization exactly; use the implementation rather than guessing another language's canonicalization. Hash the **final** production config, including `experimental: false`.

The environment digest covers a sorted map of all `ALASKA_*` and `NCCL_*` variables plus `LD_PRELOAD`, `LD_LIBRARY_PATH`, `OMP_NUM_THREADS`, and `CUBLAS_WORKSPACE_CONFIG`. Presence/absence and exact values matter. Use the exact process launch environment, including per-rank rail configuration. Do not place secrets in this digest-covered environment or publish private values.

Hash the final report file and reference it as `qualification: {"report":"/absolute/private/report.json","sha256":"<actual digest>"}`. The loader rejects a mismatched binary/config/environment or incomplete report. There is deliberately no pre-approved report template in this source release.

This attestation checks consistency with an operator's evidence; it does not execute the tests or prove their truth. `production_ready` is meaningful only when that evidence exists and runtime health is good.
