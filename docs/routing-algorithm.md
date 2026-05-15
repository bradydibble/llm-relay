# Routing algorithm and fallback semantics

How `llm-relay` decides which backend a request goes to.

## Overview

The algorithm is `build → filter → order → select → forward`. Every step is
deterministic and rule-based — there is no LLM in the routing path.

## Step 1 — Build candidates

The `model` field in the incoming request can be one of four things; each
produces a candidate list, and each tells us whether that list is **ordered**
(the caller chose the priority) or **unordered** (we have to pick).

| `model` is | Candidate list | Ordered? |
|---|---|---|
| An alias (e.g. `subagent`) | The alias's member list | Yes |
| An explicit model (e.g. `qwen3.5-35b`) | `[model] + fallback_graph[model]` | Yes |
| A fallback-graph key (e.g. `high-quality` not also an alias) | `fallback_graph[key]` | Yes |
| Unknown | All currently-healthy models | No |

In the homelab config aliases and fallback-graph keys can overlap — aliases
win.

## Step 2 — Filter

Apply hard constraints. A candidate that fails any of these is removed
entirely:

- **Privacy**: `local_only` (the default) excludes any model with
  `privacy: cloud_ok`. Cross this boundary by setting the
  `X-Llm-Relay-Privacy: cloud_ok` header.
- **Tool requirement**: with `X-Llm-Relay-Require-Tools: true`, models without
  `tool_use` in their capabilities are dropped.
- **Context window**: with `X-Llm-Relay-Min-Context: N`, models whose
  `context_window < N` are dropped.

## Step 3 — Order

- **Ordered candidates** (aliases, explicit, fallback-graph key): the list
  order *is* the priority. llm-relay does not rerank by preference. This means
  `subagent: [qwen3.5-9b, qwen3.5-35b]` will pick `qwen3.5-9b` whenever it is
  up, even though `qwen3.5-35b` has higher preference, latency, and quality
  scores.
- **Unordered candidates** (unknown model): ranked by
  `quality × preference + latency_bonus + cost_bonus + availability` where
  weights come from `policy.yaml` (or the `X-Llm-Relay-Weights` header).

## Step 4 — Select

Walk the ordered (or ranked) list and return the first candidate whose
discovery state is `available`. Skip `degraded` and `unavailable`. If no
candidate is currently available, return 503 with the decision trace in the
error body.

## Step 5 — Forward

- Compute the backend URL from the chosen model's `provider`, `port`, and
  optional `path`: `{provider.base_url}:{port}{path}/v1/chat/completions`.
- Rewrite the request body's `model` field to the resolved name (so the
  upstream sees the concrete model id, not the alias).
- POST. Return the upstream response with a `llm-relay` metadata block added.

## Circuit breaker

Each backend has a circuit breaker controlled by `providers.yaml`:

```yaml
circuit_breaker:
  failure_threshold: 3
  recovery_timeout: 30s
```

After `failure_threshold` consecutive polls fail, the breaker opens and the
poll loop short-circuits to a `[]` model list. After `recovery_timeout`
seconds, the next poll resets the breaker and attempts a real probe; if the
probe fails again, the breaker re-trips immediately.

## llm-mode dynamics

`llm-mode` switches services on the LLM host. llm-relay does *not* invoke
`llm-mode`. The interaction is one-way: when `llm-mode big` stops the 35B
service and starts the 70B service, the per-port discovery polls notice on the
next 15s tick. Aliases that include both models (e.g.
`high-quality: [qwen3.5-35b, llama-3.3-70b]`) will route to whichever is
available at request time, with no human intervention.

## Examples

### Alias with both candidates up

`POST /v1/chat/completions { "model": "subagent" }`

1. Build: `[qwen3.5-9b, qwen3.5-35b]` (ordered, from alias)
2. Filter (privacy=local_only): both pass
3. Order: alias order — `[qwen3.5-9b, qwen3.5-35b]`
4. Select: `qwen3.5-9b` is healthy → pick it
5. Forward to `http://127.0.0.1:8080/v1/chat/completions`

### Explicit model, backend down

`POST /v1/chat/completions { "model": "llama-3.3-70b" }`

1. Build: `[llama-3.3-70b]` (no fallback chain configured for this id)
2. Filter: passes
3. Order: trivially `[llama-3.3-70b]`
4. Select: not available → 503 with `decision.ranked` in the body

### Alias spanning local and cloud, default privacy

`POST /v1/chat/completions { "model": "fast" }`
with `aliases.fast: [qwen3.5-9b, claude-3-5-haiku]` and default
`privacy: local_only`:

1. Build: `[qwen3.5-9b, claude-3-5-haiku]`
2. Filter: claude drops out (cloud_ok)
3. Order: `[qwen3.5-9b]`
4. Select: pick whichever local 9B status is — fail if down (no cloud
   fallback because privacy=local_only)
