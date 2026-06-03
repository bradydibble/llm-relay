# llm-relay

Lightweight routing control plane for homelab + cloud LLM backends.

> **Heads up:** I wrote this for my own homelab as I was getting tired of
> manually switching out the models used across my agents. I am switching
> models constantly for tweaking and benchmarking, and it started to become
> a problem. Existing tools like llm router were just too heavy for my needs.
> It is published in case the design is useful, not as a polished general-purpose
> product. Configs in `config/` are generic placeholders. Real values (hostnames,
> IPs, systemd service names, model registry) belong in a private override directory.
> See [Configuration overrides](#configuration-overrides) below.

## What it does

`llm-relay` sits in front of your LLM backends and:
- Discovers available models via `/v1/models` polling
- Routes requests to the best healthy backend based on policy
- Provides introspection endpoints for observability
- Falls back to alternative models on failure

It is **not** a full platform. It's a router with explicit, rule-based routing.

## How it relates to `llm-mode`

`llm-mode` ships in this repo as a sibling script (`./llm-mode`) to swap model configs.

- `llm-mode` = **capacity manager**. It starts/stops systemd-managed LLM
  services on a target host and keeps a Caddy default upstream in sync.
- `llm-relay` = **request router**. It forwards requests to available backends.

They work together but stay separate:
- `llm-relay` does NOT invoke `llm-mode` on the request path
- `llm-relay` can **suggest** mode changes when a model is unavailable
- `llm-mode` remains the source of truth for what's actually running

## Quick start

```bash
# Install
pip install -e .

# Run as HTTP server (default port 8090)
llm-relay run

# CLI introspection
llm-relay models
llm-relay resolve high-quality
llm-relay route qwen3.5-35b --privacy cloud_ok
```

## Configuration

### `config/providers.yaml`

Define your backends:

```yaml
providers:
  local-llm:
    type: openai
    base_url: http://127.0.0.1
    enabled: true
    poll_interval: 15s

  anthropic:
    type: anthropic
    base_url: https://api.anthropic.com
    enabled: false
    model_overrides:
      - claude-3-5-sonnet-20241022
```

### `config/models.yaml`

Define models and the use-cases (categories) each serves; the relay **derives**
the category map from these tags at load:

```yaml
models:
  qwen3.5-9b:
    provider: local-llm
    class: local-9b
    port: 8080        # one polling client is created per (provider, port)
    context_window: 32768
    capabilities: [tool_use, structured_output]
    tags: [local, fast]
    preference: 0.7
    # Categories this model serves, with a priority per category. The category
    # map is DERIVED: aliases[uc] = models tagged uc, ordered by priority desc,
    # then preference desc, then name. Re-rank a model everywhere with one edit.
    use_cases: {subagent: 2, fast: 1, main: 1}

  qwen3.5-35b:
    provider: local-llm
    class: local-35b
    port: 8081
    context_window: 262144
    capabilities: [tool_use, structured_output, long_context]
    tags: [local]
    preference: 0.9
    use_cases: {main: 4, high-quality: 3, subagent: 1}

  # Optional per-category quality gate, off by default. When set, models below
  # the floor preference are refused for that category rather than served.
  # categories:
  #   high-quality: {reasoning_floor: 0.85}
```

A category is a **priority order over the live fleet, not a whitelist**: the
relay prefers the tagged models in order, then falls through to any *other* live
model that fits the request, so traffic is served whenever anything live can hold
it (context permitting). An explicit `aliases:` block still works as a deprecated
override (it wins per-category and logs a warning).

### `config/policy.yaml`

Define routing policy and fallback:

```yaml
policy:
  constraints:
    privacy:
      default: local_only

  fallback:
    graph:
      high-quality: [qwen3.5-35b, llama-3.3-70b, claude-3-5-sonnet]
      fast: [qwen3.5-9b, claude-3-5-haiku]
```

### `config/modes.yaml` (optional)

Reference `llm-mode` modes for model availability hints:

```yaml
modes:
  big:
    description: "9B + 70B (default: 70B)"
    ports: [8080, 8083]
    models: [qwen3.5-9b, llama-3.3-70b]
    default: llama-3.3-70b
```

## API

### Forward requests

```bash
curl -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Llm-Relay-Privacy: cloud_ok" \
  -d '{
    "model": "high-quality",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Add `"stream": true` for an SSE token stream — the relay proxies it and records
how it ended (clean finish vs. mid-stream error). If no live model can hold the
request the relay returns 503 with an `oversize_for_now` / `oversize_period`
signal rather than truncating; size requests against the live ceiling reported by
`/v1/available-models`.

### Routing hints (headers)

| Header | Values | Description |
|--------|--------|-------------|
| `X-Llm-Relay-Privacy` | `local_only`, `cloud_ok` | Privacy constraint |
| `X-Llm-Relay-Require-Tools` | `true`, `false` | Require tool_use capability |
| `X-Llm-Relay-Min-Context` | `131072` | Minimum context window |

### Introspection

```bash
GET /health                       # Endpoint health status
GET /status                       # Live routing state: active mode, alias resolutions, backend health
GET /v1/available-models          # All models with rich metadata + aliases (canonical)
GET /available-models             # Deprecated alias of the above (same payload)
GET /v1/models                    # OpenAI-compatible model list (concrete + aliases)
GET /v1/models/{model}            # Model card; resolves a bare or provider:model id
GET /routing-table                # Fallback graphs
GET /routing-table/qwen3.5-35b    # Fallback chain for a model
```

## Observability

Prometheus metrics are exposed at `GET /metrics` (on by default; set
`LLM_RELAY_METRICS=0` to disable). They cover request / token / fallback
counts, request and time-to-first-token latency, per-cause streaming outcomes,
and per-backend health, saturation, and circuit-breaker gauges.

OpenTelemetry tracing to an OTLP endpoint (e.g. Arize Phoenix, for per-request
prompt/completion inspection) is **opt-in and off by default**. Enable it with
the `otel` extra and an env flag:

```bash
pip install -e .[otel]
export LLM_RELAY_TELEMETRY=1
export LLM_RELAY_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces   # default
export PHOENIX_PROJECT_NAME=llm-relay                            # default
```

Metrics and tracing are independent: tracing being disabled or unreachable
never affects metrics or request handling.

## MCP server

With the `mcp` extra installed (`pip install -e .[mcp]`), the relay mounts a
Model Context Protocol server at `/mcp/mcp` (Streamable HTTP transport) so an
agent can inspect routing state before it picks a model:

- `relay_status` — active mode, alias resolutions, backend health
- `list_models` — every configured model with current availability
- `describe_alias` — a category's resolved model, context window, members, and
  saturation flag
- `select_for_capability` — models matching context / capability / privacy
  constraints, ranked by preference

## CLI

```bash
llm-relay models                   # List configured models + aliases
llm-relay resolve high-quality     # Resolve alias
llm-relay health                   # Check endpoints
llm-relay route qwen3.5-35b        # Simulate routing decision
llm-relay config                   # Print config
```

## Routing algorithm

`build → filter → order → select → forward`, fully deterministic — no LLM in the
routing path. A request's `model` may be a **category** (use-case alias), an
**explicit model**, a host-qualified **`provider:model`** id, or an unknown name.

- **Open by default** — a category is a *priority order over the live fleet*, not a
  whitelist: it prefers its tagged members, then falls through to any other live
  model that fits, degrading to whatever is up instead of dead-ending in a 503.
- **Two floors** gate each candidate: **context-fit** (hard — the model must be
  able to hold the request) and an optional **reasoning floor** (off by default).
  When nothing live can hold the request, the 503 carries an `oversize_for_now`
  vs `oversize_period` signal so a client can back off, resize, or defer.
- **Order** — configured/derived priority wins, with load-aware spill to a free
  backend among equal-priority candidates; unknown names rank by `preference`.
- **Fallback** — on a retryable upstream status (502/503/504) the router walks to
  the next candidate (pre-first-byte for streams); a tripped circuit breaker skips
  a backend entirely; the `local_only` privacy default is never crossed to cloud.

See **[docs/routing-algorithm.md](docs/routing-algorithm.md)** for the full
algorithm — the alias-vs-`fallback.graph` precedence, the context-fit contract,
and worked examples.

## Configuration overrides

The committed `config/` directory contains **generic templates**. To use
your own values without committing them to this repo, keep them in a
separate directory and point `llm-relay` (and `llm-mode`) at it via env
vars:

```bash
export LLM_RELAY_CONFIG_DIR=~/homelab/configs/llm-relay
export LLM_MODE_CONFIG_DIR=~/homelab/configs/llm-relay
export LLM_MODE_HOST=your-llm-host          # default: localhost
export LLM_MODE_UPSTREAM_IP=192.0.2.10      # default: 127.0.0.1
export CADDY_CONFIG=/etc/caddy/your.conf    # default: /etc/caddy/llm-relay.conf
```

That directory should contain its own `providers.yaml`, `models.yaml`,
`modes.yaml`, `policy.yaml` with your real hosts and model registry. The
public repo never sees them.

A convenient pattern is to keep a `env.sh` in the private dir that
exports the variables, then `source` it from your shell rc or via
`direnv`:

```bash
# ~/homelab/configs/llm-relay/env.sh
export LLM_RELAY_CONFIG_DIR="$(dirname "${BASH_SOURCE[0]}")"
export LLM_MODE_CONFIG_DIR="$LLM_RELAY_CONFIG_DIR"
export LLM_MODE_HOST=...
# etc.
```

## Why this design?

- **Simple**: One command to start, config-driven
- **Auditable**: All decisions logged, YAML config is human-readable
- **Deterministic**: No AI deciding routing, just rules
- **Separate concerns**: `llm-mode` handles capacity, `llm-relay` handles routing

## v1 scope

**Included:**
- OpenAI-compatible endpoint polling, model discovery, and health tracking
- Deterministic policy-based routing (privacy, capability, context-fit, load-aware spill)
- Open-by-default category aliases with per-model fallback chains
- Streaming (SSE) pass-through with pre-first-byte failover
- Introspection API + CLI, Prometheus metrics, and an optional MCP server / OTel tracing

**Not included (planned):**
- Automatic mode switching via `llm-mode`
- Prompt-intent tool routing (capability *filtering* via headers is supported)
- Cost tracking and budgeting
- A/B testing between models
- Custom provider adapters beyond OpenAI/Anthropic
