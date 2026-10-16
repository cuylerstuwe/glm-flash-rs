# Agent setup guide

Use this document as the installation brief for a compatible four-node GB10 cluster. It is intentionally a Markdown workflow rather than an installer script. Resolve each machine-specific value from the operator's environment, maintain a private deployment record, and verify each stage before moving on.

The objective is a reliable, quality-preserving Rust service at the agreed client interface. A green health response is not enough. Do not substitute lower target precision, skip artifact checks, or claim readiness from a short smoke test.

## 1. Establish scope, ownership, and prerequisites

Read README.md, docs/ARTIFACTS.md, docs/API.md, docs/VALIDATION.md, and config/README.md. Note that the original-code license remains undecided; separately acquired artifacts retain their own terms.

Start with read-only inventory. Identify the four intended hosts through the operator's trusted SSH configuration and verify their host identities; do not invent accounts, change SSH host keys, or disable host verification. Confirm AArch64 Linux, GB10/sm_121, memory and NVMe capacity, driver/toolkit versions, existing GPU processes, service managers, listening ports, and active deployment ownership. Preserve unrelated services. If another owner is active, coordinate instead of killing unfamiliar processes.

Establish a cluster-wide exclusive lifecycle lease for startup/shutdown/deployment, using the operator's existing mechanism or an explicitly agreed durable coordinator. Record owner, intended nodes, purpose, active artifact identity, and how to recover an abandoned lease. The runtime's per-state-directory `.service-lock` is useful but does not prevent a second process with a different state directory from competing for the same GPUs.

**Before downloads or GPU startup, verify access to every artifact in docs/ARTIFACTS.md.** This source release does not include the complete prepared model/kernel pack or its generator. If unavailable, report that precise prerequisite and stop dependent deployment work; do not improvise another model or claim that raw checkpoints are immediately loadable. Independent inventory and build checks may continue.

## 2. Record an installation plan privately

Create a private inventory outside the repository containing:

- Host-to-rank mapping (exactly ranks 0–3), verified SSH identities, service user, management interfaces, and two physical fabric rails per rank.
- One immutable release directory, model/artifact roots, private config/state/evidence directories, and rollback target.
- The rank-zero client bind/port/model ID and the shared collective bootstrap address/port.
- Memory/disk budgets, required service capabilities, firewall exposure, and credential ownership.
- The exact source commit, Cargo lock hash, toolchain/container identity, native dependency hashes, model identities, and all relevant environment settings.

Example paths in this repo are generic layout suggestions, not mandatory locations. Keep actual local paths, hostnames, addresses, credentials, and prompts out of Git.

Use separate immutable release and artifact directories, such as `/opt/glm-flash-rs/releases/<release>` and `/srv/models/<artifact>`. Put configuration under `/etc/glm-flash-rs`, sockets under `/run/glm-flash-rs`, and private state under `/var/lib/glm-flash-rs`. Create these with the chosen service user's ownership. Reserve a new state directory for an incompatible binary/model/cache format; do not reuse caches merely because a directory exists.

## 3. Validate networking and memory headroom

Inspect `/sys/class/infiniband`, port states, link layers, RoCE v2 GID types, mapped IPv4 addresses, and netdev associations. Establish the same physical rail order on all ranks. Set **`ALASKA_ROCE_RAILS=HCA0:GID0,HCA1:GID1`** with actual discovered values on each host. The engine requires this explicitly and validates two distinct active Ethernet HCAs and their selected GIDs. It does not guess private subnets or assume OS-assigned HCA names are stable.

Check connectivity, MTU, link rates, routing, and RDMA operation on both rails. A switched or directly connected topology is only suitable if it provides the required connectivity among all four ranks. Discover the correct NCCL management interface, HCA selection, and GID policy too; NCCL is a dependency when the model transport is RoCE. Keep collective bootstrap and RDMA access restricted to the cluster. The documentation-only address `192.0.2.10` in the template must be replaced.

GB10 memory is unified. Account for target/drafter weights, live token pool, retained slots, graphs, prefill scratch, host staging, RAM prefix cache, vision on rank zero, kernel/OS memory, and concurrent file I/O. The intended policy is 1,048,576 total context, 131,072 maximum output, a 2,097,664-token live pool, and four retained slots; inspect `src/cache_pages.rs` before calculating sizes. The configuration example uses 16 GiB RAM and 64 GiB SSD cache per rank. These are cache limits, not whole-process memory limits. Budget free disk for the weights, cache, staging, and rollback set. Do not fill a filesystem to satisfy an example budget.

## 4. Build and verify artifacts

Keep the engine free of a Python interpreter for loading and serving. Install compatible Rust, CUDA, C/C++, NCCL, and RDMA dependencies without replacing unrelated runtime environments. Native ATen/LibTorch for vision is intentional. Use the detailed source/build map in docs/ARTIFACTS.md.

Run `cargo check --locked --all-targets`, the appropriate CPU tests, and `cargo build --locked --release` on the intended AArch64 Linux build environment. Preserve GB10 CPU targeting. The result is `target/release/glm-flash-rs`; hash it and distribute the same binary to all four ranks. Record the complete build environment and retain a reproducible build receipt. Prefer a second isolated build to verify the executable hash before promotion.

Include the resolved CUDA 13 cuBLAS library in that inventory. The enabled prefill dispatch requires shape-specific bitwise qualification on the installed runtime, as described in docs/VALIDATION.md. Account for the additional chunk projection scratch and expert exchange slots described in docs/ARTIFACTS.md when checking memory headroom.

Inventory every native library and kernel against the source pins, including resolved transitive dependencies. Run prepared-target identity checks with:

```text
glm-flash-rs prepared-identity /srv/models/target/rank-0/manifest.json 0
```

Repeat for the proper target manifest/rank on each host. This is a format/identity check, not full model inference qualification. Prepared payloads are named `data.bin` next to their manifests. Drafter loading has its own strict contract.

Rebuilding native code may change hashes. Investigate differences and qualify an intentional pin update as a candidate change; do not bypass checks. If matching external assets cannot be obtained or reproduced, preserve the build/evidence and describe the missing assets honestly.

## 5. Create private configuration

Copy the example JSON and environment into private files. Substitute actual absolute paths, measured SHA-256 values, rank identities, and network settings. JSON configuration does not expand shell variables. Create four configs: change rank and rank-local manifests/state, keep shared model/policy/budgets consistent, and configure vision only for rank zero.

Start with `experimental: true` and `qualification: null`. Use a separate candidate port if an existing production service must remain recoverable. The final conventional endpoint is port **8001**, served model **`glm-5.3-flash-exl3`**. Only rank zero may bind a non-loopback listener with `public_listener: true`; worker listener values remain loopback. Clients must not talk directly to the collective bootstrap port.

Generate distinct random API and scheduler keys locally and store them as private regular files owned by the service account (`0600`, not symlinks). The service config itself must be owned by that account and not group/world-writable. Configure API authentication before off-host access; keep scheduler credentials limited to the router/operator. Do not display keys in evidence or logs.

Use `prefill_tokens: 8192`, the supplied measured speculative cost table, and the documented performance flags as the starting candidate. Keep `audit_logits: false` for serving/performance measurements; enable expensive diagnostics only for their scoped checks. The examples intentionally do not contain valid binary/library hashes or real hardware names, so they cannot accidentally launch as an attested production deployment.

The environment example selects `ALASKA_MULTI_GREEDY=6` for shared verification across compatible foreground requests, including normal sampling and long context. Read [the concurrency contract](docs/CONCURRENCY.md) and qualify this mode against `0` on the new cluster. Keep all ranks consistent, reserve graph/workspace memory, and preserve normal adaptive decoding for a single stream. Do not select the other experimental mode values as production alternatives without separate qualification.

`clear_thinking_file` contains `keep`, `clear`, or `default`; use `keep` when preserving reasoning history is intended. A missing/invalid file falls back to the configured default. Per-request overrides remain supported. Preserve the default normal sampling policy; do not add a fixed production seed or silently select greedy sampling.

Keep the supplied thinking default at **High**. Apply the same effort policy to direct clients and router integrations: `minimal`/`low` → Low, `medium`/`high` → High, `xhigh`/`max` → Max, and `none` → thinking off. Both `reasoning_effort` fields accept these aliases; unknown names must return HTTP 400 instead of falling back to Max. Do not let a router silently replace an invalid effort value. See [the full reasoning contract](docs/API.md#sampling-reasoning-tools-and-images) for nulls, validation, and explicit template precedence.

For full image support, create the private vision and codec manifests from the actual pinned artifacts and configure the text server's `vision.socket`. Set socket-directory ownership so unrelated users cannot replace the socket. Ensure HTTPS certificate verification works in the native fetch stack. Do not weaken private-address protections for ordinary image URLs.

## 6. Start and supervise the candidate coherently

The executable provides `serve CONFIG_JSON` and `vision-serve SOCKET CONFIG_JSON`. It does not include a universal fleet manager. Build appropriate service-manager units or adapt the operator's existing lifecycle manager using these contracts:

1. Acquire the cluster lifecycle lease and verify no other owner uses the GPUs.
2. Verify identical release/policy inputs, artifact pins, and private writable state on all ranks before any rank enters collectives.
3. Start the vision owner on rank zero if configured, then start the four text ranks as one coordinated generation. Launch peers concurrently; do not wait for rank zero to become healthy before starting followers.
4. Capture each rank's logs privately. Advertise readiness only after the native owner and required peers are ready. Test health/models on rank zero with the configured API authentication.
5. If any rank or required sidecar fails, retire the whole generation and its owned children. Do not independently restart one worker into an old collective generation. Do not leave a healthy proxy answering for a dead native owner.
6. For a planned stop, withdraw readiness, stop new ingress, send the coordinated graceful stop, wait for drainage, and terminate only processes proven to belong to that generation if a bounded shutdown fails. Preserve evidence of forced shutdowns.

Use service-manager restart policy only at the coordinated deployment level. Keep unrelated processes and services outside this stop boundary. Persist generation/PID/executable identities to make ownership auditable. Do not mistake stale PID files for live ownership.

## 7. Qualify before promotion

Follow docs/VALIDATION.md. Test actual chat/tools/grammar/images, sampled output correctness, context/output limits, retained progress, RAM/SSD cache pressure, foreground arrival during background prefill/restore/copy, cancellation, and coordinated fault recovery. Use realistic sampled reasoning and coding as the primary performance comparison. Report initial handoff separately from foreground execution; a few seconds at a safe boundary is acceptable, but persistent foreground slowdown or background starvation needs investigation.

Check each effort alias through direct chat completion and chat tokenization, then through the intended client/router path. Confirm High when effort is omitted or null, Low thinking for `minimal`, and HTTP 400 for an unknown or incorrectly typed value in either field, including a shadowed value. Verify exact rendered effort lines and explicit template/thinking precedence as described in [reasoning qualification](docs/VALIDATION.md#reasoning-effort-qualification).

Keep raw evidence and artifact hashes in the private deployment record. Test bounded writer pressure **without idle gaps** as well as idle save/restore. Check that SSD entries actually complete and survive restart; correct but repeated cold prefill is a performance/persistence problem even when output remains exact.

Run full-window and long-output tests separately; a 1M-token input cannot also request 128K output within the total context limit. Verify that runtime memory and cache budgets remain bounded under those conditions. Do not infer these gates from a short prompt.

## 8. Attest, promote, and verify the real client path

Only after completed gates, prepare the pinned qualification report described in docs/VALIDATION.md. The report must match the exact final binary, four configs, and relevant environment. Set `experimental: false` and attach the report path/SHA-256. Config digests exclude the `qualification` field itself, but **include** `experimental`; calculate them for the final production values. Create the report from this installation's completed tests and exact artifact identities.

Coordinate promotion of the binary/configuration/environment/native assets across all four ranks. Keep a recoverable snapshot of the installation's starting state and any existing service configuration. Publish the intended rank-zero interface only when ready; verify `GET /v1/models` and a representative request through the **actual router/client path**, not only localhost.

If integrating with an existing router, preserve its model identity, authentication, foreground/background translation, scheduler lease acknowledgements, and other routes. Adapt only what is necessary for the agreed contract. Confirm streaming, tool calls, sampled generation, image/grammar behavior, foreground hold/release, client disconnect, and service-restart incarnation handling through that path.

## 9. Exercise rollback and leave a durable handoff

Before declaring completion, test coordinated shutdown and recovery using the recorded binary, configuration, and artifact set. Use a compatible state directory and verify the real client path again. If this installation replaces an existing service, also test restoration of that service from its saved configuration and artifacts. For a fresh installation, verify that withdrawing the endpoint and stopping all owned processes returns the cluster to its recorded starting state. Leave the qualified service running after the checks pass, and keep the recovery artifacts available.

Record: current owner/status, source and binary identities, artifacts/environment, which gates passed with evidence, unresolved limitations, live endpoint/model, service-manager start/stop procedure, private credential locations (not values), cache budgets, and exact rollback procedure. Clearly distinguish a running qualified service, an experimental candidate, and a deployment blocked on missing artifacts. Release or transfer lifecycle ownership according to the operator's convention.
