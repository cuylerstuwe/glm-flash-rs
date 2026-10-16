# Quality and fidelity

Quality depends on both the target model and correct execution. We report distribution fidelity, implementation correctness, and task outcomes separately; a speed result cannot stand in for any of them.

## Target precision

The selected checkpoint stores routed experts as EXL3/TR3 at 4 bits per weight. Target non-expert weights remain FP8 or BF16, including a BF16 output head. The [artifact contract](ARTIFACTS.md#prepared-target) gives the exact map. This is a quantized model, not a claim of lossless equivalence to BF16.

The drafter uses Q4 representations to propose tokens. The target verifies proposals and determines committed output under the requested sampling policy. Drafter acceptance is a speed metric, not a model-quality score. Changing a drafter or its scheduling still requires sampler, state, stopping, and continuation checks.

## BF16-reference fidelity

The [machine-readable summary](../benchmarks/quality/target-fidelity.json) records measurements of the selected weights and FP8/BF16 profile in a **separate serial TP4 reference runtime**. It is a sanitized extraction of completed measurements, not a newly executed benchmark or a receipt for the full Rust serving path.

| Fixed history | Scored positions | Mean KLD, nats | Top-token agreement |
| --- | ---: | ---: | ---: |
| General | 2,047 | 0.0277963 | 93.84% |
| Code / agentic | 2,047 | 0.0345652 | 94.24% |
| Reasoning / termination | 2,047 | 0.0227251 | 97.31% |
| **Combined** | **6,141** | **0.0283622** | **95.13%** |

The combined 95th-percentile KLD is **0.1205384**, with a maximum of **1.9492641**. The mean alone does not describe every prediction position.

The teacher is the public [BF16 teacher-logit dataset](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/tree/95f4fdd94bf29989db2e0d1054e4931f55edb6aa), tied to BF16 model revision `a6c167b62691b2bac901344b65cb651a70f53e43`. The summary pins the dataset revision, input-token hashes and teacher-logit hashes for `final-0000`, `final-0002` and `final-0003`.

Each history has 2,048 fixed token IDs and 2,047 next-token prediction positions. Both models receive identical histories; generated text is not decoded and retokenized. The target runs one position at a time, with no sampler or drafter. Scoring uses full-distribution FP64 log-softmax and computes `KL(teacher || target)` with natural logarithms over all **154,880** stored vocabulary entries. A second calculation restricts both distributions to the **154,856** tokenizer entries and renormalizes; its mean is **0.0283622145**, with the same top-token agreement. A repeated first-window capture was identical and is excluded from the combined totals.

These are correlated positions from three documents, not 6,141 independent task trials. Divergence includes runtime arithmetic as well as quantization. This diagnostic does not cover fast prefill, speculative serving, vision or long-context sparse attention, and it does not establish the exact Rust binary's BF16-reference KLD. Raw logits and private operational records are not included in the public summary.

The checkpoint publisher also reports a five-cold-run mean KLD of **0.024554564250** on a separate 25-window / 51,175-position panel. That is [checkpoint-level evidence](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/main/docs/HISTORICAL_MODEL_CARD_2026-08-27.md#five-cold-kld-runs), not a measurement of this four-node Rust service. Do not subtract or rank these numbers as though their evaluation paths were identical.

## What the metrics mean

- **KLD:** how much the target's next-token probability distribution differs from the teacher's. Zero means matching distributions at the evaluated positions; lower is closer. A KLD of 0.028 does not mean a 2.8% answer-quality loss.
- **Top-token agreement:** how often the teacher and target rank the same token first when given the same history. Two distributions can agree on the winner while assigning different probabilities to other choices. Conversely, a tiny change can swap two almost-tied tokens without making one answer substantively worse.
- **Exact output/state agreement:** whether two execution paths reproduce the same tested tokens, logits or retained state. It is useful for validating optimizations and pause/resume. It does not prove the model's answer is correct.

Sampling adds another distinction: two runs can draw different valid words from the same probabilities. Preserving a distribution does not inherently promise the same output for a fixed seed across every algorithm or execution schedule. Paths that promise exact retention must still preserve their committed progress and sampling state.

## Rust execution qualification

Shared-verification qualification passed **108/108 rank-cases**, where a case is counted separately on each GPU. Checks covered full target logits, model and drafter state, partial acceptance, parser discard and serial continuation. Paired serving checks matched **24 sampled agentic replies** and **54 controlled RigMark replies**. Reasoning and tool arguments were included where present; generated tool-call IDs were checked for validity separately from semantic equality.

Cache/priority checks compare retained continuations against controls and cover cancellation, SSD corruption and restart. Near-window testing includes two independently seeded requests with **1,044,713 input tokens each**, whose concurrent outputs matched serial controls. Such tests establish execution and retention behavior on their fixtures; they are not a broad one-million-token semantic-quality evaluation.

The [RigMark receipt](../benchmarks/rigmark/gb10-tp4-low.json) separately passes **15/15 basic output gates**. Those completion checks are not a comprehensive reasoning, coding or factual-accuracy benchmark. New installations must qualify the exact binary, kernels, weights, configuration and client path using [VALIDATION.md](VALIDATION.md).
