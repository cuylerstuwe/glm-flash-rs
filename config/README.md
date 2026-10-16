# Configuration examples

These are templates, not launch-ready configurations. Replace all `REPLACE_*` values and documentation-only addresses with verified values in **private copies outside Git**. The service JSON does not expand environment variables. The project does not include a setup shell script.

- `service.rank0.example.json`: rank-zero candidate with authentication, large prefill, sampled speculation, grammar and vision. Create three worker copies with ranks 1–3, their own target/drafter manifest paths and private state, loopback listener addresses, and `vision: null`.
- `runtime.env.example`: non-secret performance/transport settings. Select actual interfaces/HCAs/GIDs from the new cluster; do not copy example names blindly. Load this through the chosen service manager after substitution.
- `vision.example.json`: rank-zero native vision owner. Supply the actual library filename/hash and model/codec paths.
- `codecs.example.json`: codec pin structure. Replace version placeholders with integers and use actual resolved dependency filenames/hashes; the example dependency filenames are illustrative.
- `speculation-costs.json`: measured cost policy in milliseconds. Embed its **`costs` object** in each service configuration, as the service example already does. All ranks need identical values. This is a starting policy, not a performance guarantee; change it only with measured qualification.

The JSON examples deliberately contain invalid hash placeholders and start in experimental mode. Do not replace them with arbitrary zeros or invent a completed qualification report.

The default history file content can be `keep`; `clear_thinking_default: false` also preserves history when the file is absent/invalid. API keys and the scheduler key are different files, owned by the service user, normally mode `0600`. State directories must be private (`0700`) and owned by that user; their parent directories must exist.

The supplied 16 GiB RAM / 64 GiB SSD budgets are per-rank cache caps, not total process memory. Reduce them only deliberately and measure the consequences for retained sessions and cache misses. Do not trade away target precision to make an incorrectly budgeted deployment fit.

`ALASKA_MULTI_GREEDY=6` enables shared target verification for two to four compatible foreground requests. Despite its name, this mode includes normal sampled requests and long contexts. `0` or an absent variable disables it; values 1–5 are experimental variants. All ranks must agree. See [concurrent inference](../docs/CONCURRENCY.md) for eligibility, fallback behavior, bounded graph memory, and qualification. The example remains an experimental deployment until its complete acceptance gates pass.

`ALASKA_*` variables configure the runtime. `ALASKA_ROCE_RAILS` explicitly selects the two fabric rails and their GIDs. The scalar NCCL GID setting must be verified independently against the selected HCA topology. If the NCCL setup needs different GIDs across devices, resolve that with the installed NCCL version rather than assuming the same example index fits all interfaces.

The engine's CUDA driver link needs the real driver at runtime. Add only verified native library directories to `LD_LIBRARY_PATH`; never add CUDA stubs. Record the exact resolved libc/libm and all shared libraries. Those choices can affect reproducibility and are part of deployment qualification.
