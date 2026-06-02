# Operations

How to add, remove, or change a backend model while keeping `llm-mode` and
`llm-relay` in sync. Everything lives in `config/*.yaml` — the goal is that
each operational change is a single edit in a single file.

## Single source of truth

| File | What it owns |
|---|---|
| `config/providers.yaml` | Backend hosts (`base_url`, polling cadence, circuit breaker). |
| `config/models.yaml` | Concrete model definitions and **aliases**. For local models, the `port` and `service` fields are read by both llm-relay (for routing) and `llm-mode` (for systemd lifecycle). |
| `config/modes.yaml` | `llm-mode` profiles. Lists the *models* a mode should run; ports and units are derived from `models.yaml`. |
| `config/policy.yaml` | Routing policy: privacy defaults, fallback graph, `llm-mode` hint messages. |

If you find yourself editing the same fact in two files, that is a bug — open
an issue.

## Add a new local model

Example: a new GGUF you've packaged on your LLM host as `example-llm-30b.service`,
listening on port 8084.

1. **Add the model to `config/models.yaml`** (one entry — its `use_cases` tags
   place it in every category at once; there is no separate list to edit):
   ```yaml
   qwen3.5-30b:
     provider: local-llm
     class: local-30b
     port: 8084
     service: example-llm-30b.service
     context_window: 65536
     capabilities: [tool_use, structured_output]
     tags: [local, medium-speed]
     preference: 0.85
     # Categories this model serves + priority (higher = preferred earlier).
     use_cases: {main: 3, high-quality: 1, code_medium: 2}
   ```
   The relay derives the category map from these tags at load, so the model is
   immediately a ranked member of `main`, `high-quality`, and `code_medium` with
   no separate alias list to keep in sync. (An explicit `aliases:` block still
   works as a deprecated override.)
2. **(Optional) Tune priorities or add a quality gate** — adjust the `use_cases`
   priorities above, or set a `categories.<name>.reasoning_floor` (a minimum
   `preference`) to refuse models below a quality bar for that category.
3. **(Optional) Add it to a `llm-mode` profile** in `config/modes.yaml`:
   ```yaml
   modes:
     multi:
       models: [qwen3.5-9b, qwen3.5-35b, qwen3.5-30b]
   ```
4. **Restart `llm-relay`** so it picks up the new model entry:
   ```bash
   systemctl --user restart llm-relay.service
   ```
   llm-mode does not need to be restarted — it reads YAML on every invocation.
5. **Verify**:
   ```bash
   llm-mode models                                     # new entry shows up
   ./.venv/bin/llm-relay resolve qwen3.5-30b           # provider+port resolved
   curl -s http://127.0.0.1:8090/health | jq           # new (provider,port) polled
   ```

You do **not** need to edit `llm-mode` itself or the systemd unit. The relay
polls the new port automatically; OpenAI-compatible clients pick up the new
model on their next `/v1/models` refresh.

## Remove or rename a model

1. Remove (or rename) the entry in `config/models.yaml`. Its `use_cases` tags go
   with it, so it drops from every category automatically — nothing else to edit.
2. Remove any reference in an explicit `aliases:` block, if you use that
   deprecated form.
3. Remove from any mode in `config/modes.yaml` (or leave — `llm-mode` will
   error helpfully if a mode references an unknown model).
4. Remove from `config/policy.yaml` fallback graphs if present.
5. `systemctl --user restart llm-relay.service`.

## Change which mode is active on the LLM host

```bash
llm-mode large-context           # start 9B + 35B, stop the rest
llm-mode big                     # start 9B + 70B, stop 35B + trinity
llm-mode multi                   # start 9B + 35B + trinity
llm-mode status                  # see what's running and which is the default
llm-mode set-default qwen3.5-9b  # change the Caddy default without changing services
```

When you switch modes, the relay sees the change on the next 15s discovery
poll. Aliases that include both the old default and the new one (e.g.
`high-quality: [qwen3.5-35b, llama-3.3-70b]`) keep routing correctly with no
intervention: `llm-mode big` makes 35B unavailable and 70B available, and
`high-quality` requests start hitting 70B automatically.

## Change which service is running, ad-hoc

`llm-mode` is the right tool. If you bypass it and start/stop services on
the LLM host by hand, the relay will still notice on the next poll — but the
Caddy default won't move with you, so any other consumers pointed at the
default upstream may break. Prefer `llm-mode <mode>` or `llm-mode set-default`.

## Add a cloud provider

1. Enable in `config/providers.yaml`:
   ```yaml
   anthropic:
     enabled: true
     auth_source: vault    # see README for credential handling
   ```
2. Add or amend models in `config/models.yaml` (cloud models need
   `privacy: cloud_ok`).
3. Decide which aliases should fall through to cloud. Cross-tier aliases are
   ordered just like local ones; e.g. `fast: [qwen3.5-9b, claude-3-5-haiku]`
   keeps local first.
4. Cross the privacy boundary explicitly on a per-request basis via
   `X-Llm-Relay-Privacy: cloud_ok`. The default is `local_only` and cloud
   models are filtered out otherwise — that is intentional.
5. Restart the relay.

## When does a change need a restart?

| Change | Restart `llm-relay`? | Restart `llm-mode`? |
|---|---|---|
| `llm-mode <mode>` (services start/stop on the LLM host) | No (polling picks it up in ≤15s) | n/a |
| Edit `models.yaml` (new/removed/changed model) | Yes | No (reads YAML every call) |
| Edit `modes.yaml` | No (relay does not use modes) | No |
| Edit `policy.yaml` (fallback graph, privacy) | Yes | No |
| Edit `providers.yaml` | Yes | No |

Restart is cheap (~2s):

```bash
systemctl --user restart llm-relay.service
```

## Consumer integration

Any OpenAI-compatible client can use llm-relay as a provider. The two
endpoints that make integration easy:

- `GET /v1/available-models` — returns every concrete model and every alias
  with metadata, so a client can populate a model picker.
- `GET /routing-table` — returns the fallback chains, useful for clients
  that want to surface "which model will I actually hit" before sending.

Aliases are the recommended way to address the relay: a client that asks
for `subagent` keeps working when you reshuffle which concrete model serves
that role.

## Observability (Prometheus metrics)

The relay exposes Prometheus metrics at `GET /metrics` (a direct route with no
trailing slash, so there's no 307 in front of the scrape endpoint). Disable
with `LLM_RELAY_METRICS=0`.

### Request metrics (recorded per chat-completion)

| Series | Type | Labels | Meaning |
|---|---|---|---|
| `llm_relay_requests_total` | counter | provider, model, alias, outcome, client | Routed requests. `outcome` ∈ success / upstream_error / saturated / network_error / no_candidate / backend_error. |
| `llm_relay_tokens_total` | counter | provider, model, direction, client | Tokens, `direction` ∈ prompt / completion. |
| `llm_relay_fallbacks_total` | counter | alias, model, client | Requests served by a non-preferred candidate (router fell back). |
| `llm_relay_request_duration_seconds` | histogram | provider, model | End-to-end relay request duration. |

The `client` label is the calling agent, resolved from the `X-Llm-Relay-Client`
header (falling back to a distinctive `User-Agent`); unknown values bucket to
`other` so the label can't explode cardinality.

### Backend metrics (pull-based, read off discovery at scrape time)

| Series | Type | Labels | Meaning |
|---|---|---|---|
| `llm_relay_backend_up` | gauge | backend, provider | 1 if the backend is healthy/degraded, else 0. |
| `llm_relay_inflight_requests` | gauge | backend, provider | In-flight slots currently held on a bounded backend. |
| `llm_relay_backend_max_concurrent` | gauge | backend, provider | Configured `max_concurrent` (0 = unbounded). |
| `llm_relay_circuit_breaker_state` | gauge | backend, provider | 1 if the circuit breaker is open, else 0. |
| `llm_relay_slot_reconciliations_total` | counter | backend, provider | Times the poll loop force-reset a stranded in-flight counter (leaked-slot containment). |
| `llm_relay_backend_resets_total` | counter | backend, provider | Times a backend recovery (circuit recovery or model reload) wiped stale in-flight state. |

The last two are *containment* signals, not normal operation. A steadily-rising
`slot_reconciliations_total` means in-flight slots are leaking faster than the
synchronous release path should allow — worth investigating. `backend_resets_total`
just tracks how often a backend restarted/reloaded out from under the relay.
