# HTTP and priority contract

The conventional base URL is `http://<head>:8001/v1`. Only rank zero serves clients. A different port or model alias can be configured, but integrations expecting the standard interface should use `8001` and `glm-5.3-flash-exl3`.

This implements an OpenAI-compatible subset, not every OpenAI endpoint or parameter. The definitive request handling is in `src/http_api.rs`, `src/http_numbers.rs`, `src/chat.rs`, and `src/grammar_request.rs`.

## Authentication and exposure

When `api_key_file` is set, all ordinary routes, including health and metrics, require `Authorization: Bearer <API key>`. The scheduler instead requires **`X-Scheduler-Token: <scheduler key>`**, even if inference authentication is disabled. Keys must differ. Keep control access private to your router/operator.

Store each key in a regular non-symlink file owned by the service user, with no group/world permissions (normally `0600`). Keys are 16–512 characters after trailing newlines are removed, with no whitespace/control characters. Do not put key values in tracked configuration, example requests, logs, or process arguments.

Loopback is the safe example default. An off-host rank-zero listener requires `public_listener: true`; this means permission to bind a non-loopback address, **not** that exposing an unauthenticated service to the public Internet is appropriate. Configure network access and TLS at the deployment boundary. The service itself uses HTTP.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health`, `/v1/health` | Runtime availability, errors, cache/service state, and qualification status |
| GET | `/models`, `/v1/models` | Served model identity and context limit |
| GET | `/metrics` | Text metrics |
| POST | `/v1/chat/completions` | Chat, tools, reasoning, images, structured output, streaming |
| POST | `/v1/completions` | Raw prompt text or token IDs, with streaming support |
| POST | `/tokenize`, `/v1/tokenize` | Count and return token IDs using the same prompt preparation |
| GET / POST | `/v1/scheduler` | Authenticated native foreground leases and scheduler state |

`/tokenize` accepts a chat body with `messages`, or a completion body with `prompt`. It returns `count`, `tokens`, `max_model_len`, and `token_strs: null`. Image tokenization also uses bounded image preparation and scheduling.

The standard model ID is `glm-5.3-flash-exl3`. The limit is **1,048,576 total tokens**, including requested output space, with an output cap of **131,072 tokens**. An explicit request whose prompt plus requested output exceeds context is rejected; an omitted output limit uses remaining space up to the output cap. Do not assume that requesting a large output allows the service to silently discard input.

## Sampling, reasoning, tools, and images

Normal sampling uses temperature `1`, top-p `0.95`, and effective top-k `20`. No fixed seed is imposed in production; `seed` can be supplied for reproducible diagnostic comparisons. Supported numeric overrides retain their own validation. Temperature at or below zero selects greedy behavior.

Thinking defaults come from service configuration. The supplied configuration enables thinking with **High** effort. Both top-level `reasoning_effort` and `chat_template_kwargs.reasoning_effort` use the same policy:

| Request value | Effective behavior |
| --- | --- |
| `none` | Thinking disabled |
| `minimal`, `low` | Thinking enabled, Low effort |
| `medium`, `high` | Thinking enabled, High effort |
| `xhigh`, `max` | Thinking enabled, Max effort |
| Omitted or `null` | Use the other field if supplied, otherwise the configured default (High in the supplied configuration) |

Names are case-insensitive and surrounding whitespace is ignored. Unknown names (including `off`), empty strings, and values other than a string or `null` return **HTTP 400**. Both fields are validated, even when one is overridden or thinking is explicitly disabled. Chat tokenization uses the same validation as chat completion.

When both fields contain valid non-null values, `chat_template_kwargs.reasoning_effort` takes precedence. An explicit `chat_template_kwargs.enable_thinking` overrides the effort-implied thinking switch; effort controls the thinking level only when thinking is enabled. To disable thinking, use `none` without a conflicting thinking override, or set `enable_thinking: false`. `minimal` selects Low thinking rather than disabling it.

For example, `"reasoning_effort": "medium"` and `"chat_template_kwargs": {"reasoning_effort": "high"}` both request High thinking. `chat_template_kwargs` also supports `clear_thinking`. Live history policy can be controlled by the configured private `clear_thinking_file`; this is a local file policy, not an HTTP endpoint.

Reasoning is returned in **`reasoning`**, separate from final `content`, including streaming deltas. Do not assume a duplicate `reasoning_content` field. Streaming uses SSE and terminates with `[DONE]`. Tool calls preserve the tool-call schema and incremental arguments; clients must assemble deltas instead of interpreting each chunk as a complete JSON argument object.

Configured XGrammar supports constrained output. Constrained requests may take the serial target path rather than speculative decoding; correctness is the contract, not a promise of identical speed to unconstrained generation. See `src/grammar_request.rs` for supported grammar/JSON-schema inputs and rejection behavior.

Vision requires the rank-zero native vision service. Image message parts use `image_url` with supported inline data URLs or HTTPS URLs when fetching is enabled. Fetching is bounded and validates destinations/redirects; it rejects unsupported schemes and disallowed private destinations. Do not disable address checks to work around a deployment mistake. Missing vision support is an explicit error, never a silent omission of images.

## Foreground and background

Use **`X-LocalAI-Mode: foreground`** or **`X-LocalAI-Mode: background`**. Missing mode defaults to foreground. Other values are rejected. A header used by some external router is not automatically equivalent; adapt it explicitly to this backend header.

Multiple foreground requests are supported concurrently. Priority selects which class may compute; it is not a single-request gate. With the documented batching mode, compatible foreground requests share GPU verification while retaining independent sampling and model state. There are four resident slots in total: a paused background request still occupies one, so a fourth foreground request can queue until a slot is released. See [concurrent inference](CONCURRENCY.md) for eligibility and scheduling limits.

Foreground arrivals hold lower-priority background work at a safe boundary. Retained state includes model progress and the decoding state needed to resume exactly. This is different from canceling and replaying a request. Disconnect/cancellation, on the other hand, abandons that request and releases its resources.

The policy protects foreground correctness, compatibility, quality, and execution speed. An already-running background prefill piece can take **a few seconds** to reach a safe handoff; this is an intentional tradeoff for useful background throughput. Compare that initial wait separately from foreground prefill/generation after admission. Background work must still yield safely without starvation or loss of progress.

This does not promise resumable HTTP requests across a process crash. In-flight work fails when its owner dies; valid persisted prefix entries can accelerate a new request after coordinated recovery.

## Router leases

Ordinary foreground requests trigger priority automatically. A router can additionally reserve foreground capacity while it prepares a request or coordinates several backends:

```json
{"owner":"router-a","epoch":1,"foreground":true,"lease_seconds":30}
```

Send this JSON to `POST /v1/scheduler` with `X-Scheduler-Token`. Owner identifiers are bounded; use a stable distinct owner per coordinator. Epochs are unsigned and monotonic within that owner. Lease duration is bounded to 1–300 seconds (30 by default). Renew before expiry, and release with a higher epoch and `foreground: false`. One owner cannot release another owner's hold.

Successful responses include `owner`, `epoch`, `foreground`, `applied_revision`, `foreground_active`, and `incarnation`. Treat the acknowledgement as evidence that native control applied the revision; do not replace it with a proxy-only queue. The incarnation changes on service restart, so reset/reconcile router state accordingly. Inspect `GET /v1/scheduler` for native state. Router adapters must use this scheduler's documented fields and acknowledgement rules.

`X-Request-Id` optionally supplies a request identifier (at most 160 characters, using letters, digits, `-`, `_`, `.`, or `:`). An active collision returns `409`. IDs do not create durable request resumption or deduplicate completed inference.

## Health and compatibility details

Health returns `503` when unavailable. `production_ready` combines current health with a pinned deployment attestation; it is **not a benchmark or self-test**. A new binary must earn its own report as described in [VALIDATION.md](VALIDATION.md).

Some response extensions, metric names, engine strings, and internal identifiers use fixed `tensorfold` or `alaska-rust` spellings in the wire format. Their presence does not select a remote service or leak a deployment address. Consumers should rely on documented fields rather than treating those strings as configuration.
