# llm-relay v2 Features (Out of Scope for v1)

This document lists features planned for v2 that are **explicitly excluded** from v1.

## v1 Scope (Completed)

✅ Basic OpenAI-compatible endpoint polling
✅ Model discovery and health tracking
✅ Policy-based routing (privacy, quality, latency)
✅ Fallback chain execution
✅ Introspection API and CLI
✅ Circuit breaker behavior
✅ Deterministic, rule-based routing

## v2 Features (Not Implemented)

### 1. Automatic mode switching via `llm-mode`

**Description:** `llm-relay` could invoke `llm-mode` to switch local modes when a requested model is unavailable.

**Why not in v1:**
- Adds coupling between router and actuator
- Mode switches are slow (seconds) vs routing (milliseconds)
- Creates unpredictable latency
- Operator should control mode switches explicitly

**v2 design:** Opt-in feature with explicit configuration:

```yaml
auto_mode_switch:
  enabled: false  # false in v2 default
  triggers:
    - when_requesting: llama-3.3-70b
      action: prewarm
      mode: big
      timeout: 30s
  approval: manual  # or auto (risky!)
```

### 2. Tool/function call routing

**Description:** Routes requests based on tool requirements (e.g., "this prompt needs function calling, route to a model that supports it").

**Why not in v1:**
- Requires parsing prompts to detect tool usage intent
- Adds latency and complexity
- Tool support detection can be done via capability tags (already in v1)

**v2 design:**

```yaml
tool_routing:
  enabled: false
  models:
    function_calling: [qwen3.5-35b, claude-3-5-sonnet]
    structured_output: [qwen3.5-9b, llama-3.3-70b]
```

### 3. Cost tracking and budgeting

**Description:** Track token costs per request/model and enforce budget limits.

**Why not in v1:**
- Requires cost metadata per model
- Requires usage tracking and aggregation
- Out of scope for a router (should be a separate cost-aware layer)

**v2 design:**

```yaml
cost_tracking:
  enabled: false
  models:
    llama-3.3-70b:
      input_cost: 0.00  # local, free
      output_cost: 0.00
    claude-3-5-sonnet:
      input_cost: 0.003
      output_cost: 0.015
  budgets:
    daily: 10.00
    monthly: 100.00
```

### 4. A/B testing between models

**Description:** Route a percentage of traffic to different models for comparison.

**Why not in v1:**
- Requires traffic splitting logic
- Requires result collection and comparison
- Out of scope for production routing

**v2 design:**

```yaml
ab_testing:
  enabled: false
  tests:
    - name: sonnet_vs_70b
      variants:
        - model: claude-3-5-sonnet
          weight: 0.5
        - model: llama-3.3-70b
          weight: 0.5
      evaluation: latency, quality_score
```

### 5. Custom provider adapters

**Description:** Support for non-OpenAI/Anthropic backends (e.g., vLLM, TGI, custom APIs).

**Why not in v1:**
- Adds abstraction complexity
- OpenAI and Anthropic cover most use cases

**v2 design:**

```yaml
providers:
  vllm-cluster:
    type: vllm
    base_url: http://vllm.local
    adapter: vllm_adapter
```

### 6. Prompt-based model selection

**Description:** Use an LLM to analyze the prompt and suggest the best model.

**Why not in v1:**
- **Explicitly forbidden** in the requirements
- "AI decides routing" is a bad pattern (opaque, unpredictable, expensive)
- Rule-based routing is auditable and deterministic

**v2 design:** (if absolutely necessary)

```yaml
ai_routing:
  enabled: false
  model: qwen3.5-9b
  prompt: "Given this prompt, suggest the best model from [list of models]"
  strict: true  # always use the AI's suggestion
```

### 7. Multi-region failover

**Description:** Failover to a different region/datacenter if all local models are down.

**Why not in v1:**
- Requires region configuration
- Adds latency and complexity
- Should be handled at the infrastructure layer (Caddy, DNS)

**v2 design:**

```yaml
regions:
  local:
    providers: [local-llm]
  cloud:
    providers: [anthropic, openrouter]
  failover:
    primary: local
    secondary: cloud
```

### 8. Usage analytics and metrics

**Description:** Collect and expose usage metrics (requests per model, latency, errors).

**Status: shipped.** Prometheus metrics are exposed at `GET /metrics` (on by
default): request / token / fallback counts, request and time-to-first-token
latency, per-cause streaming outcomes, and per-backend health, saturation, and
circuit-breaker gauges. Optional OpenTelemetry→Phoenix tracing is opt-in (see
the README "Observability" section). This item is no longer out of scope.

### 9. WebSocket streaming support

**Description:** Support streaming responses for chat completions.

**Why not in v1:**
- Adds async streaming complexity
- Most use cases don't require streaming

**v2 design:**

```bash
# Streaming endpoint
POST /v1/chat/completions?stream=true
```

### 10. Rate limiting and quotas

**Description:** Enforce rate limits per client/API key.

**Why not in v1:**
- Out of scope for a router
- Should be handled at the API gateway layer

**v2 design:**

```yaml
rate_limiting:
  enabled: false
  default:
    requests_per_minute: 60
    tokens_per_minute: 10000
```

## Decision rationale

**Principle:** Keep v1 boring, simple, and auditable.

- **No AI in the loop:** Routing is rule-based, not LLM-decided
- **No auto-switching:** Operator controls mode changes via `llm-mode`
- **No complexity:** One command to start, YAML config, clear behavior
- **Separate concerns:** `llm-mode` = capacity, `llm-relay` = routing

**v2 philosophy:** Add features only if they:
1. Don't add significant complexity
2. Have clear operational value
3. Are requested by users

**v2 guardrails:** Never add:
1. AI-based routing decisions
2. Automatic infrastructure changes
3. Opaque "magic" behavior
