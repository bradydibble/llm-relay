# llm-relay

Lightweight routing control plane for homelab + cloud LLM backends.

> **Heads up:** I wrote this for my own homelab as I was getting tired of
> manually switching out the models used across my agents. I am switching
> models constantly for tweaking and benchmarking, and it started to become
> a problem. Exising tools like llm router were just too heavy for my needs.
> It is published in case the design is useful, not as a polished general-purpose
> product. Configs in `config/` are generic placeholders. real values (hostnames,
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
llm-relay models --available
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

Define models, aliases, and metadata:

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

  # Aliases are ORDERED priority lists. llm-relay walks them at request time
  # and picks the first available candidate — so the alias `subagent: [9b, 35b]`
  # always prefers 9b when it is up, falling back to 35b when it is not.
  aliases:
    subagent:     [qwen3.5-9b, qwen3.5-35b]
    main:         [qwen3.5-35b, llama-3.3-70b, qwen3.5-9b]
    fast:         [qwen3.5-9b]
    high-quality: [qwen3.5-35b, llama-3.3-70b]
```

### `config/policy.yaml`

Define routing policy and fallback:

```yaml
policy:
  ranking:
    quality: 0.4
    latency: 0.3
    cost: 0.1
    availability: 0.2

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

### Routing hints (headers)

| Header | Values | Description |
|--------|--------|-------------|
| `X-Llm-Relay-Privacy` | `local_only`, `cloud_ok` | Privacy constraint |
| `X-Llm-Relay-Weights` | `quality=0.4,latency=0.3,...` | Override ranking weights |
| `X-Llm-Relay-Require-Tools` | `true`, `false` | Require tool_use capability |
| `X-Llm-Relay-Min-Context` | `131072` | Minimum context window |

### Introspection

```bash
GET /health                       # Endpoint health status
GET /available-models             # All models with rich metadata + aliases
GET /v1/available-models          # Same payload, /v1-prefixed for clients
GET /v1/models                    # OpenAI-compatible model list (concrete + aliases)
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

## CLI

```bash
llm-relay models --available       # List models
llm-relay resolve high-quality     # Resolve alias
llm-relay health                   # Check endpoints
llm-relay route qwen3.5-35b        # Simulate routing decision
llm-relay config                   # Print config
```

## Routing algorithm

1. **Build candidates**: If the requested model is an alias, an explicit model
   with a fallback chain, or a fallback graph key, the candidate list is
   **ordered** (the user/config specified the priority). If the requested name
   is unknown, candidates are drawn from currently-discovered models.
2. **Filter**: Apply hard constraints (privacy, tools, context).
3. **Order**: For ordered candidates, the configured order *is* the ranking —
   llm-relay does not re-rank by preference. For the unknown case, candidates
   are ranked by `quality + latency + cost + availability`.
4. **Select**: Walk the ordered (or ranked) list and return the first candidate
   the discovery layer currently reports as available.
5. **Fallback**: On backend HTTP 5xx the router moves on to the next available
   candidate in the same list.

## Fallback semantics

- **On HTTP 5xx**: Retry next candidate in fallback chain
- **On circuit breaker**: Skip unavailable endpoints entirely
- **Privacy boundaries**: Never cross `local_only` → cloud

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
- OpenAI-compatible endpoint polling
- Model discovery and health tracking
- Policy-based routing (privacy, quality, latency)
- Fallback chain execution
- Introspection API and CLI

**Not included (v2):**
- Automatic mode switching via `llm-mode`
- Tool/function call routing
- Cost tracking and budgeting
- A/B testing between models
- Custom provider adapters beyond OpenAI/Anthropic
