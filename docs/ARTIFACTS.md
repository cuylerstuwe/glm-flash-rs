# Artifact and build contract

## What is and is not here

The repository includes the Rust host/runtime code, native CUDA/C/C++ shims, kernel and vision identity manifests, prompt templates, and small synthetic fixtures. It excludes weights, prepared binary shards, compiled GPU kernels, shared libraries, deployment credentials, runtime caches, and deployment-specific evidence.

**A fresh install cannot be completed from this repository plus raw Hugging Face weights alone.** There is no included target/drafter shard converter or complete generator for the pinned precompiled kernel pack. Obtain a matching, lawfully redistributable artifact pack from a trusted source, or implement and separately qualify preparation tooling. No complete public pack URL is currently specified. Do not fabricate manifests, bypass pin checks, or silently substitute another quantization.

## Required inventory

| Asset | Required shape / validation |
| --- | --- |
| Target | Four rank-specific prepared manifests and adjacent `data.bin` files, rank 0–3 / world 4, matching `src/prepared.rs` |
| Drafter | Four matching DFlash2 prepared manifests and binary files, validated by `src/drafter_weights.rs` |
| Tokenizer | Matching `tokenizer.json`, including the model's role/special tokens; validate the included template fixtures |
| Kernel pack | The exact `.cubin`, metadata `.json`, and `.ptx` files named/pinned by `native/kernel-manifest.json` and the additional cached kernel call sites |
| Native inference | The standalone `.cubin` files checked by `src/front.rs`, plus shared-library ABIs/hashes checked by the native loader modules |
| Communication | Native RoCE bridge plus compatible NCCL/RDMA libraries; consistent libraries and fabric configuration on all ranks |
| Grammar | XGrammar C ABI bridge and its exact SHA-256 |
| Vision | `config.json`, `processor_config.json`, standalone `vision.safetensors`, native vision operations, pinned codecs and dependencies |
| Runtime tuple | Driver/toolkit, Rust/C++ compilers, libc/libm, NCCL, RDMA, and native dependency versions and hashes |

Keep artifact hashes and revisions in a private deployment inventory. Model and kernel identifiers embedded in source are technical compatibility requirements, not suggestions to download a mutable `main` branch.

## Prepared target

`src/prepared.rs` defines format version 1: file size, chunk size and SHA-256 list, a storage table (`offset`, `nbytes`, device placement), a serialized tensor/object tree, and a preparation key. The loader checks geometry, chunk hashes, storage bounds, rank/world, and model identity. Chunk size is bounded to 128 MiB. The binary payload is adjacent to its manifest.

The pinned target model key is:

```text
Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdbf168
```

The root object tag is `tensorfold.families.glm5_next.cuda.weights:Weights`. Expected geometry includes hidden size 4096, 45 layers, vocabulary 154880, four hyper-connection streams, and linear-attention geometry of 64 heads × 128 dimensions. Read the complete loader checks rather than treating these few numbers as a sufficient converter specification.

The target non-expert policy keeps DSA KV-B and index projections in BF16, most other eligible projections/dense/shared layers in FP8, and the output head in BF16. The detailed preparation map is:

```text
map:dense_down=fp8,dense_gu=fp8,dsa_kvb=bf16,dsa_out=fp8,dsa_proj=fp8,dsa_qb=fp8,idx_proj=bf16,idx_qb=bf16,kda_fb=fp8,kda_gb=fp8,kda_out=fp8,kda_proj=fp8,lm_head=bf16,mtp.dsa_kvb=bf16,mtp.dsa_out=bf16,mtp.dsa_proj=bf16,mtp.dsa_qb=bf16,mtp.idx_proj=bf16,mtp.idx_qb=bf16,mtp.mtp_eh=bf16,mtp.shared_down=bf16,mtp.shared_gu=bf16,shared_down=fp8,shared_gu=fp8
```

Do not re-quantize target non-experts to Q4 or substitute an NVFP4 target. The selected routed-expert checkpoint is already quantized; this policy is not a claim that the whole model is unquantized.

## Prepared drafter

The pinned model key is `incoai--GLM-5.3-Flash-DFlash2-dc77ff1c99ee`. `src/drafter_weights.rs` checks world/rank, `what: drafter`, the preparation-code identity, exact tensor/storage structure, and a 768 MiB payload bound. The drafter uses Q4 representations; this does **not** change the target non-expert policy. Target verification determines accepted output.

Retain the `tensorfold...Q4` and `...DraftLayer` serialization tags and the expected code identity; renaming them without a format migration breaks compatibility. A matching filename alone proves nothing about artifact contents.

## GPU kernel pack

`native/kernel-manifest.json` maps an alias to a cache key, function name, and hashes of three files. The runtime resolves:

```text
<kernels>/<key>/<name>.cubin
<kernels>/<key>/<name>.json
<kernels>/<key>/<name>.ptx
```

Additional kernel references appear in `src/main.rs`, `src/prefill.rs`, and related loaders; inventory all call sites when assembling the pack. The runtime launches precompiled kernels directly. It does not invoke a Python/Triton compiler during inference.

CUDA architecture is `sm_121`. Recompiling with a different compiler or launch metadata can change both bytes and behavior. A hash mismatch is a deployment rejection to investigate, not a reason to delete the check.

## Building Rust and native code

Build on AArch64 Linux with a Rust compiler that accepts edition 2024, the APIs in this tree, and `target-cpu=gb10`. The deployment source used a Rust 1.99/CUDA 13 toolchain family; record the actual toolchain you use and qualify that artifact. `Cargo.lock` fixes Rust dependency resolution. `.cargo/config.toml` targets the GB10 CPU for AArch64 Linux. `build.rs` links the CUDA driver library; `CUDA_LIB_DIR` can override the build-time library search directory. CUDA **stub libraries are for linking only**, never the runtime `LD_LIBRARY_PATH`.

Typical Rust commands, run in the checkout on a build host with the prerequisite toolchain, are:

```sh
cargo check --locked --all-targets
cargo test --locked
cargo build --locked --release
```

`cargo build` does not build the native artifact pack. These are the native source families an agent must map to the pinned libraries:

| Source family | Build requirements |
| --- | --- |
| `native/*.cu` inference helpers | NVCC, `sm_121`, optimized standalone cubins (`--cubin`); retain each kernel's numeric flags. Sampling, candidates, and KDA use `--fmad=false`; do not add fast math. |
| `native/roce.cu` + `roce_bridge.cpp` (which includes the proxy) | CUDA object plus C++17 shared bridge, CUDA runtime and libibverbs, PIC; same C ABI as `src/roce.rs` |
| `native/l2-prefetch` | CUDA/C++17 PIC shared library; prefetch hints must not alter model arithmetic |
| `native/xgrammar/bridge.cpp` | C++17 optimized PIC bridge and XGrammar static library, pthread; build the core without Python bindings |
| `native/vision_ops.cpp` | C++20, ATen/LibTorch CPU/CUDA and c10 libraries, matching C++ ABI; no fast math, floating-point contraction disabled |
| `native/image_codecs.c` | C11 PIC, pinned JPEG/WebP/demux (and transitive libraries); no fast math, floating-point contraction disabled |

The grammar integration targets XGrammar 0.2.8, source revision `97787376faee5ed8466cfad57c99855e4ce2f6aa`, with DLPack revision `bbd2f4d32427e548797929af08cfe2a9cbb3cf12`. Disable XGrammar Python bindings, C++ tests, cpptrace, coverage, and internal checking for the release build; run the bridge qualification separately. These pins identify external build inputs, not bundled source.

The included native sources are useful for rebuilding and auditing shims, but they are not a reproducible recipe for every external kernel. Native loaders pin hashes. If an intentional rebuild changes a hash, review the build/input difference, update the pin only in a candidate branch, then requalify it. Never mass-replace hashes merely to get a health response.

Record and match libc/libm across ranks too: host sampling transformations can depend on libm. Record all resolved shared-library paths (`ldd`/`readelf`), not just environment variables. Verify no Python process/interpreter is required for model loading, grammar, vision, or serving.

## Prefill math and scheduling

The target prefill path can use CUDA 13 cuBLAS through `src/draft_blas.rs`, which loads `/usr/local/cuda/lib64/libcublas.so.13`. Record the resolved cuBLAS library and its dependencies in the runtime inventory. It uses BF16 inputs, FP32 compute, the default GEMM algorithm, and an 8 MiB workspace. Prepared target weights use the specified precision; routed-expert outputs are FP32.

`ALASKA_PREFILL_BLAS` defaults to `1`; `0` selects the retained precompiled GEMM kernels for all shapes. The dispatch in `src/prefill.rs` keeps the 160-by-4096 indexer and 2048-by-128 shapes on those kernels, requires at least 4096 rows for three larger shapes, and at least 256 rows for the other supported shapes. These limits reflect bitwise comparisons and timings on the qualified runtime. They are not a guarantee for another cuBLAS version, GPU, or kernel pack; repeat the checks in docs/VALIDATION.md when assembling a new runtime tuple.

Attention input projection scratch spans the whole prefill chunk. When all input rows are ready, the projection runs once per chunk; when a previous expert layer is still finishing, it runs per attention piece after that piece's share completes. Row-split prefill retains eight dedicated expert exchange slots and overlaps each layer's ordered exchanges, hyper-connection postprocessing, and row sharing with the next layer's attention. Per-piece completion fences protect consumers; arithmetic and summation order remain fixed.

In addition to base model/workspace allocations, budget approximately 88 MiB for projection scratch and 256 MiB for expert exchange slots per rank at the 8192-row chunk/1024-row attention-piece configuration, plus the cuBLAS workspace. Measure full-context memory with the final configuration. Routed-expert outputs remain FP32, and foreground/background yielding occurs at safe work boundaries.

## Vision assets

The rank-zero vision process is started with:

```text
glm-flash-rs vision-serve /run/glm-flash-rs/vision.sock /etc/glm-flash-rs/vision.json
```

The config contains `model_dir`, `library`, `library_sha256`, and `codecs` (the path to a codec manifest). The model directory contains the three files listed above; the vision loader checks tensor identities against `native/vision-manifest.json` and keeps the tower in BF16.

The codec manifest has `abi: 1`, numeric `jpeg_abi` and `webp_version`, `module: {path, sha256}`, and `dependencies` with `jpeg`, `webp`, `demux`, plus any transitive dependencies such as `sharpyuv`. Each dependency is `{path, sha256}`. The loader checks actual symbol ownership against the pinned library paths. Merely copying a shim while allowing it to resolve a different system codec is not equivalent.

## Storage and distribution

Calculate required disk space from actual files, plus the per-rank SSD cache budget, staging, and one recoverable rollback set. Keep one canonical model copy when byte-identical files can be shared locally. Do not commit artifacts or caches to Git.

Before distributing a binary pack, audit embedded build paths/metadata and its redistribution terms separately. The source privacy review is not evidence that arbitrary compiled binaries or prepared manifests are free of personal metadata.
