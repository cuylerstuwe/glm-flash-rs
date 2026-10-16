# Better quality AND speed on an outcome-based coding test

GLM Flash RS completed **14 of 16 attempts correctly**, compared with **13 of 16** for the pinned Mia AI Lab TP4 recipe, and used **26.2 versus 29.4 seconds per fully correct outcome**. That is approximately **11% less time per correct outcome, despite Mia's approximately 5% higher observed API token rate**.

These results describe this battery and these samples. Each recipe solved seven of the eight task types at least once, and each failed a task the other solved twice.

## Combined result

| Measurement | GLM Flash RS | Mia TP4 |
| --- | ---: | ---: |
| Fully correct attempts | **14/16** | 13/16 |
| Total task time, including failures | **366.4s** | 381.9s |
| Total seconds per fully correct outcome | **26.2s** | 29.4s |
| API seconds per correct outcome, excluding external tools/grading | **25.9s** | 29.1s |
| Active prefill + generation seconds per correct outcome³ | **≈25.7s** | ≈29.1s |
| Generated completion tokens, including thinking | **32,121** | 35,312 |
| Observed API completion rate¹ | 88.6 tokens/s | **93.3 tokens/s** |
| Approximate engine-reported decode rate² | 92 tokens/s | **96 tokens/s** |

¹ API rate = total completion tokens / summed API request duration. It includes prefill, reasoning, output and API completion overhead, but excludes time spent executing external tools and grading between requests. Warmup is excluded.

² Decode rates come from each engine's recorded decode counters. The native engine counts active decode work including sampling and snapshot copies; Mia reports per-request `decode_s`. Both rounded values include their three-token warmup. Different accounting makes these approximate rather than a matched kernel benchmark; the API rate uses the same client measurement on both.

Task wall time includes API requests, thinking, output, tools and final validation, including unsuccessful attempts. Setup, loading, warmup and restoration are excluded. Seconds per correct outcome = total task wall time / number of fully correct attempts. A quick incorrect answer consumes time without adding a successful outcome.

Mia's observed API token rate was **5.4% higher**, but it generated **9.9% more completion tokens**, including thinking and assistant tool calls. The additional output outweighed the higher token rate: its total task time was approximately **4% longer**. Token counts, timing and correctness must be considered together.

For example, one historical-database attempt used **2,159 completion tokens and 22.2 seconds** on Mia, versus **1,059 tokens and 11.4 seconds** on GLM Flash RS; both passed. The extra output is recorded, but the test does not establish that it was pathological reasoning loops or that quantization alone caused it.

## Inference time per correct outcome

The same receipts separate inference requests from executing and grading generated code. Only **3.62 seconds for GLM Flash RS and 3.43 seconds for Mia** fell outside API calls across the entire panel. That includes external tool/test execution, sandbox startup, final grading and harness work between requests. Initial service setup, model loading and warmup are already excluded from task wall time.

Removing those external costs gives **25.9 versus 29.1 API seconds per correct outcome**. The comparison therefore does not depend on differences in script/test execution time. This API measure is the most directly comparable inference timing here: both recipes use the same client measurement, including prefill, generation and request/streaming overhead.

Engine phase counters provide a narrower, approximate breakdown:

| Active inference phase per correct outcome³ | GLM Flash RS | Mia TP4 |
| --- | ---: | ---: |
| Prefill | ≈0.8s | ≈0.8s |
| Generation, including thinking | ≈24.9s | ≈28.3s |
| Prefill + generation | **≈25.7s** | ≈29.1s |

**Every per-correct figure counts time spent on failed attempts too**, then divides by the number of fully correct tasks. It is not an average restricted to successful answers; failed work still consumes inference time.

³ Native counters record active prefill/decode phases, including sampling, communication and snapshot copies. Mia reports per-request `prefill_s` and `decode_s`. These exclude external script/test execution and HTTP/client overhead, but include host/GPU orchestration within their respective phase boundaries. They are not pure GPU kernel timings, and their accounting is not identical.

Mia's timed requests total **10.0774 seconds of prefill and 367.8191 seconds of generation**, excluding warmup. Native counter deltas total **11.2897 seconds of prefill and 348.8609 seconds of generation**, including a three-token warmup whose exact phase split was not retained. That warmup's entire measured API time is **0.2202 seconds**, bounding its contribution to **less than 0.0158 seconds per correct outcome**. Removing anywhere from zero to that upper bound leaves every one-decimal figure in the phase table unchanged. The JSON records those bounds rather than inventing an exact warmup subtraction.

This isolates time spent serving inference from external code execution. It does not isolate model weights from engine scheduling, cache behavior, speculative decoding, numerical precision or different generated reasoning lengths.

## Task results

Each row represents two fresh sampled attempts per recipe. Mean time includes both attempts even when one or both failed. Full pass requires every hidden case or trace for that attempt to pass.

| Task | GLM Flash RS correct | Mean task time | Mia correct | Mean task time |
| --- | ---: | ---: | ---: | ---: |
| Lower-bound min-cost flow | 2/2 | 11.3s | 2/2 | 13.5s |
| Historical MVCC and TTL | 2/2 | 11.2s | 2/2 | 16.1s |
| Cyclic package/version resolution | 2/2 | 22.0s | 1/2 | 27.8s |
| Tool-driven build-planner repair | 2/2 | 59.3s | 2/2 | 33.2s |
| Late/deduplicated event windows | 2/2 | 11.9s | 2/2 | 14.1s |
| Tool-driven lease-queue repair | 0/2 | 16.9s | 2/2 | 17.4s |
| Revision-aware temporal SQL | 2/2 | 24.1s | 0/2 | 37.2s |
| Rollback potential equations | 2/2 | 26.6s | 2/2 | 31.8s |

Mia was much faster at the build-planner repair with both answers correct: **33.2 versus 59.3 seconds** on average. The two GLM Flash RS times were **30.8 and 87.9 seconds**, while Mia took **42.2 and 24.2 seconds**. This variation is a reason not to treat the overall timing difference as a fixed engine-speed multiplier.

## What failed

- **GLM Flash RS, both queue repairs:** generated code omitted the required lease-expiration sweep before enqueue. The visible public tests passed, but the omission could leave an expired lease live or alter subsequent job selection. Each answer passed 15/25 hidden traces or targeted checks. Mia passed all 25 twice.
- **Mia, both SQL answers:** one query referenced an unavailable `rowid` and did not execute. The other overcounted overlapping sessions by summing a segment once per matching session instead of once per user. Each failed all 24 hidden cases. GLM Flash RS passed all 24 twice.
- **Mia, one dependency resolver:** the answer found the minimum cost but included an unused, zero-cost, self-dependent package outside the required dependency closure. It passed 38/39 cases. The task explicitly required exactly the reachable closure; the limited scope of this near miss remains relevant when interpreting the full-task failure. Mia's other answer and both GLM Flash RS answers passed all 39.

Minimal explanation checks reproduce the queue and SQL defects: enqueueing a different job after a one-attempt lease expires must mark the old job dead; overlapping sessions `[0,10)` and `[5,15)` have union duration 15, rather than 20. These are defects in generated task solutions, not evidence of broken inference-cache state or service errors.

## Method

- Four NVIDIA GB10 nodes, 128 GB unified memory each, dual 100 Gb/s RoCE rails; one sequential interactive stream.
- Eight synthetic coding, state, SQL and repository-repair tasks, two attempts per recipe. Each round covers 320 hidden cases/traces across the tasks. Checks have different sizes and are correlated; they are not hundreds of independent quality observations.
- **High thinking**, temperature **1**, top-p **0.95**, effective top-k **20**. Fresh explicit 31-bit test seeds were generated before inference; paired per-task/agent-turn lists were used by both recipes. RNG mappings need not match. The seed overrides apply only to test requests.
- Same prompts, order and budgets: **12,288 completion tokens per request**, 240-second request wall guard, at most eight tool-agent turns and a 360-second soft agent deadline checked between turns. Normal prefix caching remained enabled.
- The two repository tasks allowed reading and editing their modules and running public tests. Hidden checks were frozen before inference and supplied no feedback or retries. Known-correct references passed and deliberately broken implementations failed before the panel. No model-based grader or best-answer selection.
- Generated Python and SQL ran in the same networkless, read-only, resource-bounded CPU sandbox. Task wall time includes final grading. No bulk transfers or hash scans ran during timed inference.
- All attempts completed normally. No completion-cap, request-error or agent-deadline failure determined an outcome.

## Recipes and source pins

**GLM Flash RS:** [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f), using TR3/EXL3 4bpw routed experts, FP8/BF16 target non-experts and a BF16 output head.

**Mia AI Lab TP4:** the [canonical TP4 source pin `d220030`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/d22003070575c2774de0f010062aa57bae06f3c7), using the published Q4 non-expert / FP8 KV recipe, [target checkpoint `078455f`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c), and [DFlash2 drafter `bf582e4`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a), with `fnc7:0.3` and copy drafts. Its advertised precision/settings were retained.

[The JSON summary](../benchmarks/outcomes/gb10-tp4-high.json) records exact artifact identities, per-attempt results, counts, timings, formulas' inputs and scope. It is an allowlisted extraction of the recorded panel, excluding private paths, host identities, timestamps, credentials, prompts and operational logs.

## Interpretation and limits

This compares complete recipes: target checkpoint/quantization, non-expert precision, drafter, kernels and scheduling differ. It cannot isolate Rust versus Python, Q4 versus FP8/BF16, or any individual optimization. Similar task outcomes do not establish equal numerical fidelity.

Only two samples per task and one sequential run per recipe were taken. GLM Flash RS ran before Mia; there was no alternating thermal/order-controlled campaign. Sampling changes reasoning length, implementation choices, tool calls and final output, all of which affect time to outcome. The results do not measure long-context quality, vision, concurrency or broad prose quality and do not establish a general quality ranking.

The result on this battery is **more fully correct outcomes in less time for GLM Flash RS, despite Mia's higher token rate**. The task breakdown also shows real advantages for Mia. These findings support measuring useful outcomes alongside TPS.
