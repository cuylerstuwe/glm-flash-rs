# GLM Flash RS

https://github.com/user-attachments/assets/114d3a6c-f79d-4f43-8f11-9a40429200e0

# About

GLM Flash RS is a Rust inference service for **GLM 5.3 Flash on a four-node NVIDIA GB10 cluster**.

The opinionated recipe selects a balanced position on the speed/quality curve that **errs on the side of QUALITY**.

It selects [the lowest-KLD EXL3 4bpw quant currently available](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw), and keeps non-expert weights in **FP8 or BF16**.

By contrast, some other recipes follow a singleminded pursuit of maximizing token/s headline numbers. Some recipes swap out the 4bpw quant for 2.5bpw or less. Others quantize attention/etc. to 4-bit, gaining about 5-10% in token/s speed, but losing those gains in real-world use by emitting ~30% more tokens for an equal-or-worse answer:

<img width="523" height="212" alt="Image" src="https://github.com/user-attachments/assets/361b158d-d027-4545-a82b-83596e39338f" />

Some other recipes make claims about the quality/speed of their "alternative quants" that don't actually survive scrutiny. Example:

<img width="523" alt="Image" src="https://github.com/user-attachments/assets/98e04bf8-b70e-4f98-b5d0-bc209f71d95d" />

If you wanted the fastest and most responsive GLM 5.3 Flash stack for 4 DGX Sparks that's actually usable for serious production work, this is the best I've been able to find as of the time of this writing.

# Why does this exist?

When I upgraded my stack to 4x DGX Sparks, the recommended MikroTik CRS804 and CRS812 were out of stock everywhere.

On GPT's suggestion, I decided to use 2x MikroTik CRS504-4XQ-IN switches with 8x 100Gbps cables. This setup was not talked about much online, so I explored optimizations myself.

For more about the hardware setup this recipe has in mind, see [my local AI page](https://stuwe.dev/local-ai/).

Eventually, I decided to convert it from being Python-based to being Rust-based because I was tired of Python GC issues slowing down real-world use at long context lengths.

Unlike some social media influencers, I actually use my local LLMs for real work. I can't afford to statpad token/s stats in exchange for worse _(and ironically, slower)_ results.

## Speed

Results on four GB10s with dual 100 Gb/s RoCE links, using the weights and precision described above. The engine uses shared GPU verification for compatible concurrent foreground requests and adaptive speculative decoding for a single stream.

This RigMark run measures the head's loopback endpoint with **Low** thinking effort, greedy sampling (temperature **0**, top-p **1**), and protocol **1.1.0**. It includes five samples per decode workload, three cold/replay pairs per context size, and three rounds per concurrency level.

```text
╭──────────────────────────────────────────────────────────────────────────────────────────╮
│  R I G M A R K   //   AGENT WORKLOAD RECEIPT                                             │
│  BENCHMARKS LOCAL AI HOW CODING AGENTS ACTUALLY USE IT                                   │
│  ●  15/15 BASIC OUTPUT GATES PASSED                                                      │
├─ SYSTEM ─────────────────────────────────────────────────────────────────────────────────┤
│  MODEL      glm-5.3-flash-exl3                                                           │
│  APPLIANCE  4x NVIDIA GB10, 128 GB unified memory each                                   │
│  RUN        reasoning=low  •  protocol=1.1.0                                             │
│  SOURCE     git:c5a0db01b054  •  clean                                                   │
├─ REAL OUTPUT ────────────────────────────────────────────────────────────────────────────┤
│  WORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE          │
│  CODE             116.0 tok/s      16.7s last    113.8–119.2    ✓ 5/5                    │
│  PROSE             63.0 tok/s      15.7s last     61.8–63.7     ✓ 5/5                    │
│  STRUCTURED*      163.5 tok/s       4.5s last    162.9–164.0    ✓ 5/5                    │
│  * predictable-output ceiling; not a proxy for agent speed                               │
├─ CONTEXT ────────────────────────────────────────────────────────────────────────────────┤
│  64K PREFILL   cold 2,872 tok/s  •  immediate replay 400,802 tok/s                       │
├─ CAPPED CONCURRENT GENERATION ───────────────────────────────────────────────────────────┤
│  SHORT CODE • END-TO-END • 256-TOKEN CAP PER AGENT                                       │
│  AGGREGATE   C1 84.9  •  C2 103.2  •  C4 133.5 tok/s                                     │
│  C4 OUTPUT STATE   normal stop 0/12  •  visible 12/12  •  reasoning may be included      │
├─ RECEIPT ────────────────────────────────────────────────────────────────────────────────┤
│  JSON       sha256:919eb1262328c08c…                                                     │
│  SHARE THE CARD • LINK THE JSON RECEIPT • #RIGMARK                                       │
│  github.com/alexellis/rigmark                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

[View the full JSON receipt](benchmarks/rigmark/gb10-tp4-low.json) · [Plain-text result card](benchmarks/rigmark/gb10-tp4-low.txt)

Receipt SHA-256: `919eb1262328c08c9237276d264e00da0b6f659c43b99a74a30e3a94d71df3c4`. The card's `SOURCE` identifies the RigMark benchmark checkout; the receipt separately records the serving binary hash and the metadata the benchmark did or did not independently verify.

Cold prompt processing at **8,192 / 32,768 / 65,536 input tokens** measured **2,722 / 2,857 / 2,872 tokens/s** median effective prefill, with **3.01 / 11.47 / 22.82 seconds** median to first output. The **400,802 tokens/s** immediate-replay figure reflects prefix-cache reuse: the 64K request's median time to first output was **0.164 seconds**. It is an effective cached-request rate, not the rate of computing a fresh prompt. Client placement, network transport, and proxies affect these timings.

The card's decode estimates exclude prefill. **Capped concurrency rates include prefill and completion overhead** and measure all streams' output divided by group elapsed time; they are not per-stream decode speeds. Every C4 stream reached the 256-token cap, so visible output does not imply a completed coding task. Structured text is a predictable-output ceiling, not a constrained-grammar benchmark or a proxy for agent speed. The 15/15 basic output gates are completion checks, not a comprehensive quality evaluation.

Separate sampled-workload measurements use temperature **1**, top-p **0.95**, effective top-k **20**, and **High** thinking effort. These are different workloads from the greedy RigMark receipt above:

| Workload | Recorded result |
| --- | --- |
| Sampled agentic replay: 24 requests covering reasoning, coding, and tools | **81.3 tokens/s** aggregate generation rate |
| Full-file edits and retries within that replay | **87.2 tokens/s** aggregate generation rate |
| Sampled capped C4, 512 output tokens per stream | **128.2 tokens/s** median aggregate end-to-end rate across three rounds, versus **91.1** with batching disabled |
| Sampled long-context agentic work, two concurrent requests, about 4.7K–57.1K input tokens | **103–105 tokens/s** aggregate generation rate across two passes, versus about **84** with batching disabled |

Sampled generation rates use engine decode timings, include reasoning tokens when emitted, and exclude prefill; capped end-to-end rates include it. RigMark measures streamed output at the client. These are workload-specific measurements, not guaranteed rates for every prompt or request mix. See [concurrent inference](docs/CONCURRENCY.md) for supported batching, retained-slot limits, and mixed-group scheduling tradeoffs.

## Quality

The target uses **4-bit EXL3/TR3 routed experts**, **FP8/BF16 non-expert weights**, and a **BF16 output head**. The recipe keeps target non-experts above 4-bit and uses target verification for speculative output. The drafter's Q4 weights affect its guesses, not the target's precision. See the [precision map](docs/ARTIFACTS.md#prepared-target).

| Fidelity measure | Result | What was measured |
| --- | --- | --- |
| Mean KLD to the BF16 teacher | **0.0283622 nats** | Selected weight/precision recipe in a separate serial TP4 reference runtime; 6,141 prediction positions across three 2,048-token histories |
| Top-token agreement with the BF16 teacher | **95.13%** | The same three-history diagnostic, using all 154,880 stored vocabulary entries |
| Publisher's checkpoint mean KLD | **0.0245546 nats** | A separate 25-window panel, 51,175 positions per run, repeated in five cold runs; [published checkpoint evidence](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/main/docs/HISTORICAL_MODEL_CARD_2026-08-27.md#five-cold-kld-runs) |

**Lower KLD means closer next-token probabilities; higher top-token agreement means more often choosing the same most-likely next token on the same history.** Neither is an answer-accuracy percentage: 95.13% agreement does not mean 95.13% of answers are correct.

The reference-runtime figures characterize the selected precision recipe; **they are not a BF16-reference measurement of the exact Rust serving binary**. The two KLD panels have different coverage and execution paths, so their values are not a controlled comparison. The [fidelity JSON summary](benchmarks/quality/target-fidelity.json) includes per-window results, teacher/token hashes, and measurement scope.

Rust engine qualification separately passed **108/108 rank-cases** checking full logits, retained model/drafter state, partial acceptance, and continuation. Paired serving checks matched **24/24 sampled agentic replies** and **54/54 controlled RigMark replies**, including reasoning and tool arguments where present. These establish preservation of tested behavior, not universal correctness. See [quality methodology and limits](docs/QUALITY.md).

## Better quality AND speed on an outcome-based coding test

On this small coding-and-reasoning battery, **GLM Flash RS returned more fully correct solutions in less total time—despite Mia TP4's approximately 5% higher observed API token rate**.

Eight tasks received two fresh sampled attempts per recipe, using **High** thinking effort, temperature **1**, top-p **0.95**, and effective top-k **20**. Frozen hidden executable checks determined correctness. Failed attempts remain included in the time totals.

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

Mia generated approximately **10% more tokens**, outweighing its token-speed advantage. GLM Flash RS used approximately **11% less time per fully correct outcome**. Excluding external tool and grading execution gives **25.9 versus 29.1 API seconds per correct outcome**, so script/test runtime does not explain the difference. More tokens per second do not necessarily mean a faster—or correct—answer.

¹ Completion tokens divided by summed API wall time, including prefill and excluding external tool/grade time. ² Engine timing scopes differ slightly; these rounded decode rates include a three-token warmup on each stack. Neither rate is interchangeable with the RigMark estimates above.

³ Approximate engine active-phase time, excluding external tools/grading; includes orchestration within each phase. Timing scopes and the small warmup bound are detailed in the report.

This is a **small, workload-specific recipe comparison**, not a general quality ranking or proof that quantization caused the additional tokens. Mia solved both queue repairs that GLM Flash RS missed and completed the build repair faster; GLM Flash RS passed both temporal-SQL tasks that Mia missed. See the [full report, task breakdown and limitations](docs/OUTCOME-COMPARISON.md) and [machine-readable results](benchmarks/outcomes/gb10-tp4-high.json).

## Hardware and software assumptions

| Component | Intended configuration |
| --- | --- |
| Compute | Four GB10 systems, such as DGX Spark or compatible GB10 systems; one GPU/rank each, tensor parallelism of four |
| Memory | 128 GB unified memory per node; capacity is shared by GPU work, host allocations, and the OS |
| CPU / GPU | AArch64 Linux; CUDA compute capability 12.1 (`sm_121`) |
| Network | Two usable RoCE v2 Ethernet rails per node, with consistent rail ordering and connectivity among all ranks; the intended setup uses two 100 Gb/s links per node |
| Storage | Local NVMe for prepared weights and bounded SSD prefix caching; size from the actual artifact manifests, cache budget, and staging/rollback needs |
| Native stack | Compatible CUDA 13 toolchain/driver, NCCL, RDMA userspace libraries, C/C++ toolchain, and pinned image/grammar dependencies |

## Set up with an agent

Give an agent [AGENT-SETUP.md](AGENT-SETUP.md), this repository, and access to your intended cluster. It describes inventory, required artifacts, build/configuration, coordinated startup, qualification, promotion, and rollback.

Supporting documents:

- [Artifact formats and native build requirements](docs/ARTIFACTS.md)
- [API, authentication, and native priority](docs/API.md)
- [Validation and performance acceptance](docs/VALIDATION.md)
- [Configuration templates](config/README.md)
- [Repository development instructions](AGENTS.md)

## Behavior

- OpenAI-style chat and completion endpoints, streaming, tool calls, separate reasoning output, structured outputs, and image input when the vision component is configured.
- A **1,048,576-token total context limit** and **131,072-token maximum output**, with output constrained by remaining context.
- Normal sampling defaults of temperature **1**, top-p **0.95**, and effective top-k **20**, with no fixed production seed. Explicit supported request overrides remain effective. Thinking is enabled in the example configuration with **High** effort. Both effort fields accept `minimal`/`low` → Low, `medium`/`high` → High, `xhigh`/`max` → Max, and `none` → thinking off; unknown names return HTTP 400. See [reasoning options and precedence](docs/API.md#sampling-reasoning-tools-and-images).
- Large prefill batches, CUDA graphs, tensor-parallel RoCE exchanges, target-verified speculative decoding, and shared GPU verification for two to four compatible foreground requests, including ordinary sampling and long contexts.
- Bounded RAM/SSD prefix caches and retained sessions. The example budgets are 16 GiB RAM and 64 GiB SSD **per rank**, in addition to live model/cache/workspace memory.
- Native foreground/background scheduling. Background work yields at a safe boundary and resumes retained progress without cancellation/replay. A few seconds of initial handoff can be worthwhile for efficient background prefill; foreground execution after admission remains the priority.

## Interface

The conventional rank-zero endpoint is `http://<head>:8001/v1`, with model ID **`glm-5.3-flash-exl3`**. Examples bind to loopback until the operator deliberately configures authenticated network access. The service itself speaks HTTP; use an appropriate private network or TLS proxy when needed.

After installation, a foreground request looks like:

```sh
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer $GLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-exl3","messages":[{"role":"user","content":"Explain how a prefix cache helps a coding assistant."}],"temperature":1,"top_p":0.95,"max_tokens":1024,"stream":true}'
```

Add `X-LocalAI-Mode: background` to use idle capacity at lower priority. Ordinary requests default to foreground. Separate authenticated scheduler leases let a router reserve foreground capacity. Read [the API and priority contract](docs/API.md) before implementing a router adapter.
