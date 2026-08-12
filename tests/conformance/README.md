# Backend conformance harness

`conformance.py` runs the standard client use cases against a **live** relay and
reports, per model, what actually works. It is not a unit test and does not run
under pytest: it needs a real key and real backends, and it costs tokens.

```bash
export CIQ_RELAY_KEY=llmr_...
python3 tests/conformance/conformance.py --all                  # every discovered model
python3 tests/conformance/conformance.py --model main --model fast
python3 tests/conformance/conformance.py --all --json > matrix.json
python3 tests/conformance/conformance.py --all --expect tests/conformance/expected.json
```

## Why this exists

Backends behind one alias surface do not behave the same, and the differences are
invisible until a user hits them in an editor. Measured 2026-08-11, all four
served models disagreed about something that matters to a client:

- `ornith-35b` (llama.cpp) writes its chain-of-thought into `message.content` as
  literal `<think>…</think>`, and leaks tool syntax into `content` as well.
- `qwen3-14b-awq` cannot emit a tool call at all — with `tool_choice` forcing one
  it returns `finish_reason: "stop"` and empty content.
- `glm-5.2-nvfp4` and `trinity-large-thinking` disagree with each other about
  whether reasoning lands in `reasoning`, `reasoning_content`, or both.
- `max_tokens` is floored up to `REASONING_OUTPUT_FLOOR` for reasoning models, so
  a client asking for 16 tokens can receive 1111 — while `max_completion_tokens`
  is honoured exactly. Two parameters OpenAI treats as synonyms behave oppositely.

A capability claim that has not been run against the model is a guess. This
harness is how a claim in `/v1/models` earns the right to be published.

## What each case proves

| Case | Why a client cares |
|---|---|
| `chat` | Baseline: a 200 with non-empty content. |
| `content_clean` | No `<think>`, no tool syntax in `content`. An editor renders content verbatim. |
| `reasoning_separated` | Reasoning arrives in a field, not in the answer. |
| `stream_content` | Deltas arrive and assemble; no reasoning leaks into visible deltas. |
| `tool_single` | Correct tool name, parseable JSON arguments, `finish_reason: tool_calls`, call id present. |
| `tool_forced` | `tool_choice` actually forces a call rather than returning empty. |
| `tool_parallel` | Two calls when two are warranted. |
| `tool_stream` | Streaming tool-call argument deltas assemble into valid JSON. |
| `tool_loop` | The real agent loop: feed a tool result back, get an answer that uses it. |
| `structured_output` | `response_format: json_schema` returns parseable JSON. |
| `max_tokens_honoured` | A client's cost/latency ceiling is respected. |
| `max_completion_tokens_honoured` | The modern spelling of the same ceiling. |

`--expect` compares the run against a committed baseline and exits non-zero on any
regression, so this can gate a backend or template change. Record a new baseline
only with a deliberate note about what changed and why.

Baseline recorded 2026-08-12, after reasoning separation (`set_params:
reasoning_format=deepseek` on ornith-35b), body-derived tool routing, and the
removal of the max_tokens floor: main, fast, and trinity all clean. Known flake:
`tool_parallel` on `main` occasionally emits one call instead of two (model
nondeterminism, 4/4 on retry) — if it is the only regression in a run, retry
before treating the run as red.
