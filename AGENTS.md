# Working on this repository

Read README.md, AGENT-SETUP.md, docs/ARTIFACTS.md, docs/API.md, and docs/VALIDATION.md before deployment work.

- This is a four-rank GB10 engine. Keep the selected target weights and FP8/BF16 non-expert policy. Do not silently substitute an NVFP4 target or compress target non-experts to 4-bit.
- Keep loading and serving free of a Python runtime. CUDA and native C/C++ libraries are intentional dependencies.
- Preserve sampled output behavior, tokenization/templates, exact foreground/background retention, bounded caches, cancellation, and the context/output policy.
- Optimize representative sampled interactive work. Greedy comparisons are useful diagnostics, not the sole objective. Prioritize one foreground stream over aggregate throughput.
- A few seconds of safe initial foreground handoff is acceptable when larger background batches improve useful work. Measure handoff separately from foreground execution; do not equate priority with subsecond preemption.
- Native library/model/kernel hash checks are part of the artifact contract. Do not bypass them to make an incompatible installation start.
- Start with read-only inventory and establish cluster ownership before changes. A local state lock does not grant exclusive ownership of a whole cluster.
- Keep private host inventories, credentials, paths, prompts, runtime logs, and evidence outside Git. Qualification evidence must match the exact binary, configuration, and environment being deployed.
- Retain required third-party notices. The project-wide license is undecided.

Use exact search first, then inspect the implementation and run the relevant checks. A cross-compile check is not GPU qualification. Source changes that affect network discovery or startup need cluster testing before promotion.
