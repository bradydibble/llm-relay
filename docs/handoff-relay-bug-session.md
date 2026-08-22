# Handoff: LLM Relay Bug Fix Session (2026-08-21/22)

> **RESOLVED 2026-08-22.** Release `27a7dd6` (the complete fix stack) went live
> on llm-gateway-01 at 18:03 UTC and was verified against the live relay:
> qwen3.6-35b reproducer 10/10 consecutive non-streaming passes (10.6–14.1s,
> `finish_reason: stop`, no cap hits) plus a clean streaming pass (16s, 1775
> chunks, `[DONE]`). ornith-35b turned out NOT to be the loop bug: llama-server
> on llama-01 already defaults `repeat_penalty=1.1` and honors the relay's
> max_tokens cap — it is simply slow (~120s for a ~1250-token reasoning answer),
> so client timeouts under ~3 min read as hangs there. The "deployed vs pending"
> section below is retained as history; it is no longer current.

## Objective

Reproduce and fix the ciq-relay bug report: qwen3.6-35b and ornith-35b hang indefinitely on certain `/v1/chat/completions` requests. Expand test suite, improve health checks, get advising from Claude/ChatGPT.

## What was the bug (root cause, found by Jason Rodriguez)

Two separate bugs, both in the relay's non-streaming path:

1. **Unbounded generation:** vLLM's default when `max_tokens` is unset = `max_model_len - prompt` = ~245K tokens = ~6 hours at 12 tok/s. The relay buffered the entire non-streaming response, so the client got HTTP 200 + keepalive whitespace for hours.

2. **Repetition loop:** vLLM's default `repetition_penalty` = 1.0 (no penalty). Certain prompts (e.g. `+const int64 x = 1;`) trigger the model to repeat the same token sequence forever (1300+ chunks of "int64 vs int64"). Ollama's default is 1.1, which breaks the loop. Nobody misconfigured anything — two frameworks, two defaults.

**How we missed #2:** Never streamed the response to watch what the model was generating. The vLLM logs showed "30 tok/s generation" — read as productive, was a loop. Three AI reviews (Claude Opus, ChatGPT, ciq-reviewer) confirmed our wrong framing ("prefill stall") instead of independently investigating. The Ollama comparison ("same weights work in 40s") was dismissed as a max_tokens difference instead of bisecting ALL default differences.

**A third bug found late:** the streaming path in `route_and_forward` did NOT apply `repetition_penalty` or `max_tokens` defaults — only the non-streaming path did. If the client sent `stream=true`, the request hit vLLM with no penalty and no cap. Same payload, different `stream` flag = intermittent hangs (2 streaming hangs, 3 non-streaming successes). Found by ChatGPT gpt-5.6 gap audit.

## All commits (llm-relay repo, pushed to both bradydibble/llm-relay and ctrliq/llm-relay)

| Commit | What |
|---|---|
| `cd3d70a` | default max_tokens=1024 for non-streaming (initial fix) |
| `ac0611a` | Raised to 8192, made configurable via policy.yaml |
| `cb8c6ef` | default repetition_penalty=1.1 for non-streaming |
| `7cca75e` | Degeneracy detector library + golden fixture tests |
| `91d21cf` | L2 health probe with circuit breaker (wedge detection) |
| `ad8f60b` | L2 probe: max_tokens=1 (avoid false positives on reasoning models) |
| `fe484ca` | L2 circuit breaker actually works (L0 no longer overwrites degraded; selector excludes degraded) |
| `b75a790` | L2 probe timeout 30s + cooldown to prevent flapping |
| `d851411` | MCP list_models/select_for_capability use /v1/models (host-qualified IDs) |
| `730dd6c` | L2 probe uses streaming (avoid false positives on busy backends like GB200) |
| `02f2ab2` | In-band degeneracy detector on streaming path + config-drift detector |
| `62bc2e0` | Apply repetition_penalty to streaming path too |
| `f693368` | Streaming gets max_tokens cap too + repetition_penalty enforced as floor (not just default) |

NOTE: Fable is fixing health.py in a PARALLEL SESSION. His commits may be interleaved. Check `git log --oneline` for his work before deploying.

## What's deployed vs pending (HISTORICAL — superseded by the banner above)

**Deployed (live on llm-gateway-01):** Through commit `730dd6c` (the streaming L2 probe fix). This was the last relay restart I did.
*(2026-08-22 update: `27a7dd6` deployed 18:03 UTC — everything below shipped.)*

**NOT deployed (committed and pushed, but NOT restarted):**
- `02f2ab2` — in-band degeneracy detector + config-drift detector
- `62bc2e0` — repetition_penalty on streaming path
- `f693368` — streaming max_tokens cap + repetition_penalty as floor (clamps explicit 1.0 up to 1.1)
- Whatever fable pushed for health.py

**DO NOT restart the relay until fable's health.py fix is ready.** Then deploy ALL pending commits in one restart.

## The GB200 503 issue

The user keeps getting `glm-5.2-nvfp4 is not available` in ciq-harness. Cause: the L2 health probe was false-positiving on the GB200 (marking it degraded when it was busy serving real requests). The streaming probe fix (`730dd6c`) helped but the GB200 may still get false-positived under certain load patterns. Fable is fixing health.py to address this.

The deployed relay has `730dd6c` (streaming probe) but NOT `f693368` (the latest fixes). When fable's fix + `f693368` are deployed together, the GB200 issue should be fully resolved.

## llm-relay-ops PR

PR #1 (https://github.com/ctrliq/llm-relay-ops/pull/1) adds GCP deployment docs + Fuzzball amd-mi300x context. **Blocked by branch protection** — needs review approval from someone other than the pusher. The repo requires PRs.

## Config changes on the gateway (not in any repo)

- `structured_output` added to 6 models in `/srv/llm/config/relay/models.yaml` on llm-gateway-01: trinity-large-thinking, glm-5.2-nvfp4, qwen3.8-27b, qwen3.6-35b, glimmer-vllm, deepseek-v4-flash. All tested and confirmed supporting both `json_object` and `json_schema` response_format.
- Backup: `models.yaml.bak-pre-structured-output-20260821`
- qwen2.5-14b and qwen2.5-coder-7b (llama.cpp) do NOT get structured_output — they produce JSON wrapped in markdown, not strict JSON.

## Remaining gaps (identified by ChatGPT gap audit, not yet fixed)

1. **`max_completion_tokens` not canonicalized** — the new OpenAI field name is not handled by `_clamp_max_tokens`. A client sending `max_completion_tokens` instead of `max_tokens` bypasses the cap. Medium risk.

2. **`set_params` in models.yaml can override safety defaults** — a model config with `set_params: {repetition_penalty: 1.0}` would undo the floor. Need config-load validation to reject/warn. Medium risk.

3. **No conformance test matrix** — the streaming-vs-non-streaming gap slipped through because there's no parameterized test covering `{streaming, non-streaming} x {defaults applied, defaults not applied} x {params present, absent, null}`. This is the detection gap that let the bug ship. High priority for prevention.

4. **Alias reasoning intersection** — `main` and other aliases show `reasoning: False` because some members don't have `reasoning` in their config. This is correct intersection behavior. The MCP fix (commit `d851411`) lets clients select specific models by host-qualified ID instead of relying on alias capabilities. No code fix needed — the solution is to select `llama-01:ornith-35b` directly when reasoning is needed.

5. **llama-01 only runs one model at a time** — the other ports (8081-8084, 8086-8090) being "unavailable" is NORMAL, not a bug. Strix Halo runs one model at a time. ornith-35b on 8088 is the active serve. Do not try to "fix" the unavailable ports.

6. **ornith-35b's non-streaming slot was wedged** — from the original Aug 16 incident (pre-fix unbounded generation occupied the single `--parallel 1` slot). The wedge eventually cleared on its own (the generation finished after ~5.7 hours). The L2 probe correctly detected and excluded it during the wedge, then closed the circuit after recovery.

## Key files

- `llm_relay/routing/router.py` — `_clamp_max_tokens`, `_apply_repetition_penalty_default`, `route_and_forward` (both streaming and non-streaming paths)
- `llm_relay/health.py` — L2HealthProbe (background circuit breaker). BEING MODIFIED BY FABLE.
- `llm_relay/degeneracy.py` — compression-ratio + cycle detection
- `llm_relay/config_drift.py` — config-drift detector (hashes /v1/models, alerts on change)
- `llm_relay/api/app.py` — lifespan wiring (L2 probe, config drift, degeneracy detector), in-band degeneracy on streaming path, MCP tools
- `llm_relay/mcp/server.py` — list_models and select_for_capability (now use /v1/models)
- `tests/test_context_clamp.py` — max_tokens + repetition_penalty default tests
- `tests/test_degeneracy.py` — degeneracy detector tests
- `tests/test_health_probe.py` — L2 circuit breaker tests
- `tests/test_config_drift.py` — config-drift detector tests

## Deploy procedure (do NOT restart until fable is done)

```bash
SHA=$(cd ~/Projects/llm-relay && git rev-parse HEAD)
cd ~/Projects/llm-relay && tar czf /tmp/llm-relay-$SHA.tar.gz \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  --exclude='*.pyc' --exclude='*.egg-info' .
scp /tmp/llm-relay-$SHA.tar.gz head-01:/tmp/
ssh head-01 "gcloud compute scp /tmp/llm-relay-$SHA.tar.gz \
  llm-gateway-01:/tmp/ --project product-tooling-501914 --zone us-central1-a --internal-ip"
ssh head-01 "gcloud compute ssh llm-gateway-01 \
  --project product-tooling-501914 --zone us-central1-a --internal-ip \
  --command 'sudo mkdir -p /srv/llm/releases/llm-relay-$SHA && \
    sudo tar xzf /tmp/llm-relay-$SHA.tar.gz -C /srv/llm/releases/llm-relay-$SHA && \
    sudo chown -R llm:llm /srv/llm/releases/llm-relay-$SHA && \
    sudo ln -sfn /srv/llm/releases/llm-relay-$SHA /srv/llm/current/llm-relay && \
    sudo systemctl restart llm-relay && sleep 5 && \
    systemctl is-active llm-relay && \
    systemctl show llm-relay -p NRestarts --value'"
```

## Advising protocols used

- `claude -p --model opus` (Claude Opus) — for architectural review and post-mortem
- `claude -p --model fable` (Claude Fable) — for wedge diagnosis (per user request)
- `codex exec -s read-only -c model_reasoning_effort=high` (ChatGPT gpt-5.6) — for gap audits and wedge diagnosis

Lessons learned (from Claude Opus post-mortem):
- Never observe only proxy metrics (throughput, status codes) — observe the primary artifact (generated tokens)
- Bisect ALL differences in a working control (Ollama), not just the first plausible one
- Three reviews with identical framing = 1 opinion, not 3 — vary the framing, withhold diagnosis, provide raw artifacts
- `finish_reason: "length"` at exactly the cap = not a success signal
- Apply defaults to ALL code paths, not just the one you tested
