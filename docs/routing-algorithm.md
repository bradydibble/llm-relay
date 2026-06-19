# Routing algorithm and fallback semantics

How `llm-relay` decides which backend a request goes to.

## Overview

The algorithm is `build → filter → order → select → forward`. Every step is
deterministic and rule-based — there is no LLM in the routing path.

## Step 1 — Build candidates

Categories (aliases) are **derived from per-model `use_cases` tags** at load:
`aliases[category] = models tagged with it, ordered by priority desc, then
preference desc, then name`. The `model` field
in the incoming request can be one of four things; each produces a candidate list,
and each tells us whether that list is **ordered** (the caller chose the priority)
or **unordered** (we have to pick).

| `model` is | Candidate list | Ordered? |
|---|---|---|
| A category / alias (e.g. `subagent`) | The category's tagged members (priority prefix) **+ every other live model, preference-ranked** (open fallthrough) | Yes |
| An explicit model (e.g. `qwen3.5-35b`) | `[model] + fallback_graph[model]` | Yes |
| A fallback-graph key (e.g. `high-quality` not also an alias) | `fallback_graph[key]` | Yes |
| Unknown | All currently-healthy models | No |

**Open by default — a category is a priority order, not a whitelist.** A category
request prefers its tagged members in order, then falls through to *any other live
model* (preference-ranked) that survives the filters. So a category degrades to
whatever is currently up rather than dead-ending in a 503 when its named members
are down — the only honest refusal is when *nothing live* can serve the request.
Fallthrough is for categories only; explicit / host-pinned requests stay strict
(that tier is deliberately specific).

**Aliases vs the fallback graph — one mental model.** Both express ordered
fallback, but for different request shapes, with a strict precedence:

1. **Category alias** (request a *category*, e.g. `main`): the derived member list
   plus the open-fallthrough tail is the chain. Categories always win — if a name
   is both a category and a `fallback.graph` key, the category is used.
2. **Explicit concrete model** (e.g. `qwen3.5-35b`): the chain is the model itself
   followed by its `fallback.graph[model]` entry, if any — a per-model safety net
   for direct requests that categories cannot express. No open fallthrough here.
3. **Bare `fallback.graph` key** that is neither a category nor a concrete model:
   its graph chain is used directly.
4. **Unknown name**: discovery-ranked candidates (preference sort).

So curate **categories via `use_cases` tags** and reserve **`fallback.graph` for
per-concrete-model fallback**. When the same name lives in both, the graph entry
is reachable only via the explicit-model path (2), never the category path (1).

## Step 2 — Filter

Apply hard constraints. A candidate that fails any of these is removed
entirely:

- **Privacy**: `local_only` (the default) excludes any model with
  `privacy: cloud_ok`. Cross this boundary by setting the
  `X-Llm-Relay-Privacy: cloud_ok` header.
- **Tool requirement**: with `X-Llm-Relay-Require-Tools: true`, models without
  `tool_use` in their capabilities are dropped.
- **Context window**: models whose **live** `context_window` cannot hold the
  request's *prompt* are dropped. The floor is `max(X-Llm-Relay-Min-Context,
  prompt estimate + small output headroom)`; `max_tokens` is *not* part of it (it
  is clamped to the chosen model's headroom at forward time) — see the context-fit
  contract below.
- **Reasoning floor** (opt-in, off by default): a category may declare
  `categories.<name>.reasoning_floor`, a minimum `preference`; models below it are
  dropped for that category — including the open-fallthrough tail — so a
  quality-sensitive category refuses rather than serves below the bar.

## Step 3 — Order

- **Ordered candidates** (aliases, explicit, fallback-graph key): the list
  order *is* the priority — llm-relay never reranks by preference. It does
  apply **load-aware spill**: among candidates of equal priority the
  least-loaded backend wins, and a saturated backend (no free slot) is skipped
  in favour of a free one. So the `subagent` category (tag-derived order
  `qwen3.5-9b` then `qwen3.5-35b`) picks `qwen3.5-9b` while it has a free slot, but
  spills to `qwen3.5-35b` when the 9B backend has none — preferring an available
  alternate over a slot wait.
- **Unordered candidates** (unknown model): sorted by `preference`
  (descending), ties broken by name — the same ordering the MCP
  `select_for_capability` tool returns, so the two surfaces never disagree.

## Step 4 — Select

Walk the ordered (or ranked) list and return the first candidate whose discovery
state is `available` (or `degraded`). Open fallthrough makes "no candidate" rare —
it happens only when nothing live survives the filters. When it does, return 503
with the decision trace in the error body. If the binding constraint was
**context** (no live model is big enough for the request's prompt), the 503 body
also carries a structured `context` diagnosis distinguishing **`oversize_for_now`** (a
big-enough model exists in the catalog but is currently down → back off and retry)
from **`oversize_period`** (nothing in the fleet is big enough → resize or defer),
with the estimated tokens, the max available now, and the max in catalog.

If context is **not** the binding constraint, the relay still distinguishes a
*transient* shortage from a genuine one. When the constraints would be satisfied by
a configured model that is merely down or paused right now (a discovery blip or a
maintenance pause), the 503 carries a `Retry-After` header and the `no_backend`
outcome, so a batch caller waits and retries rather than failing hard. Only a
genuine mismatch — no configured model can ever satisfy the constraints (e.g. tools
required but none is tool-capable, or a privacy / reasoning floor that nothing
meets) — is terminal (`no_candidate`), since retrying cannot help.

## Step 5 — Forward

- Compute the backend URL from the chosen model's `provider`, `port`, and
  optional `path`: `{provider.base_url}:{port}{path}/v1/chat/completions`.
- Rewrite the request body's `model` field to the resolved name (so the
  upstream sees the concrete model id, not the alias).
- POST. Return the upstream response with a `llm-relay` metadata block added.

## Context-fit contract

The relay routes; it does not chunk or truncate. The contract lets a client size
requests correctly and adapt deterministically (no model-side decision):

- **Advertise the live-servable ceiling.** `/v1/available-models` (`alias_info`)
  and `/v1/models` report, per category, the *largest context the category can
  serve right now* — the max live window among the models it would actually route
  to (members + open fallthrough), not a down primary's nominal window.
- **Size on the prompt.** Eligibility is gated on the request's **prompt**
  (`chars/3` — a deliberate over-count at `~3 chars/token` — plus a small output
  floor), *not* on `max_tokens`. So size a prompt to `chars/3 <= context_window`
  and it routes; the relay estimates the same way, so "sized it right, still
  503'd" can't happen. `max_tokens` is an output *ceiling*, not context the model
  must reserve: counting it toward eligibility would pin every request carrying a
  generous `max_tokens` to the single largest-context backend, defeating open
  fallthrough.
- **Clamp the output, don't exclude the model.** At forward time the relay caps
  `max_tokens` to the chosen model's remaining headroom (`window - prompt`), so
  the output never overflows the window it was routed to (llama.cpp truncates
  silently; vLLM hard-rejects `prompt + max_tokens > max_model_len` with a 400). A
  clamp can return a shorter completion than requested (`finish_reason=length`) —
  honest graceful degradation, not a failure.
- **Refuse informatively.** When no live model can hold the **prompt**, the 503
  carries the `oversize_for_now` vs `oversize_period` signal (see Step 4). The
  relay never silently truncates the prompt; the client waits, resizes, or
  defers — all deterministic arithmetic, no second LLM call.

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
next 15s tick. A category that includes both models (e.g. `high-quality`, with
both the 35B and the 70B tagged into it) routes to whichever is available at
request time, with no human intervention.

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

### Category spanning local and cloud, default privacy

`POST /v1/chat/completions { "model": "fast" }`
with `fast` tagged on `[qwen3.5-9b, claude-3-5-haiku]` and default
`privacy: local_only`:

1. Build: `[qwen3.5-9b, claude-3-5-haiku]` + open-fallthrough tail (other live
   local models, preference-ranked)
2. Filter: claude drops out (cloud_ok under local_only)
3. Order: `[qwen3.5-9b, <other live local models that fit>]`
4. Select: `qwen3.5-9b` if up; otherwise fall through to the next live local
   model. Only 503s if no live local model can serve the request — and then with
   the context diagnosis if the cause was size.
