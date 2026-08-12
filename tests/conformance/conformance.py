#!/usr/bin/env python3
"""Run the standard client use cases against a live relay, per model.

Stdlib only, matching the rest of this repo. Not a pytest module: it needs a real
key and real backends. See README.md in this directory for why it exists.

Every case answers one question a client actually asks. A case never raises: a
backend that times out or 503s is reported, not fatal, because the point is to
produce a complete matrix even when part of the fleet is unhealthy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("CIQ_RELAY_URL", "https://llm.internal.ciq.com/v1")
TIMEOUT = int(os.environ.get("CONFORMANCE_TIMEOUT", "240"))

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

# Markers that must never appear in visible content. Chain-of-thought delimiters
# and raw tool syntax are both "the model's internals rendered at the user".
LEAK_MARKERS = ("<think>", "</think>", "<|channel|>", "<tool_call", "</tool_call",
                "<function=", "functions.", "<|tool")


class Result:
    __slots__ = ("case", "status", "detail")

    def __init__(self, case: str, status: str, detail: str = "") -> None:
        self.case, self.status, self.detail = case, status, detail


def post(base: str, key: str, model: str, body: dict, nonconfidential: bool,
         stream: bool = False):
    """Returns (http_status, parsed_json_or_raw_text). Never raises."""
    payload = dict(body, model=model)
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **({"X-Llm-Relay-Confidentiality": "non_confidential"} if nonconfidential else {}),
            "X-Llm-Relay-Client": "conformance",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", "replace")
            if stream:
                return r.status, raw
            try:
                return r.status, json.loads(raw)
            except ValueError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw
    except Exception as e:  # timeout, DNS, TLS, reset
        return 0, f"{type(e).__name__}: {e}"


def msg_of(doc) -> dict:
    if not isinstance(doc, dict):
        return {}
    try:
        return doc["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError):
        return {}


def finish_of(doc) -> str:
    if not isinstance(doc, dict):
        return "?"
    try:
        return doc["choices"][0].get("finish_reason") or "?"
    except (KeyError, IndexError, TypeError):
        return "?"


def leaks(text: str) -> str | None:
    low = (text or "").lower()
    for m in LEAK_MARKERS:
        if m in low:
            return m
    return None


# --- cases ------------------------------------------------------------------
# Each takes (base, key, model, nonconf) and returns a list of Result.

def case_chat(base, key, model, nc):
    st, doc = post(base, key, model, {
        "messages": [{"role": "user", "content": "What is 17*3? Answer briefly."}],
        "max_completion_tokens": 600,
    }, nc)
    if st != 200:
        return [Result("chat", "FAIL", f"http {st}"),
                Result("content_clean", "SKIP", "no response"),
                Result("reasoning_separated", "SKIP", "no response")]
    m = msg_of(doc)
    content = m.get("content") or ""
    out = [Result("chat", "PASS" if content.strip() else "FAIL",
                  "" if content.strip() else "200 but empty content")]
    leak = leaks(content)
    out.append(Result("content_clean", "FAIL" if leak else "PASS",
                      f"leaked {leak!r} into content" if leak else ""))
    fields = [f for f in ("reasoning", "reasoning_content") if m.get(f)]
    if fields:
        out.append(Result("reasoning_separated", "PASS", "+".join(fields)))
    elif leak:
        out.append(Result("reasoning_separated", "FAIL", "reasoning is inside content"))
    else:
        out.append(Result("reasoning_separated", "SKIP", "model emitted no reasoning"))
    return out


def case_stream(base, key, model, nc):
    # 900 for the same reason as case_structured: heavy-reasoning aliases
    # (code_heavy burns ~150 reasoning deltas on "count to 5") can consume a
    # tight ceiling before the first content delta, which scores as "no content
    # deltas" and reads like a streaming bug. Measured twice on code_heavy
    # before this was raised.
    st, raw = post(base, key, model, {
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_completion_tokens": 900, "stream": True,
    }, nc, stream=True)
    if st != 200 or not isinstance(raw, str):
        return [Result("stream_content", "FAIL", f"http {st}")]
    text = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        blob = line[5:].strip()
        if blob == "[DONE]":
            break
        try:
            ev = json.loads(blob)
            d = ev["choices"][0].get("delta") or {}
            if isinstance(d.get("content"), str):
                text.append(d["content"])
        except Exception:
            continue
    joined = "".join(text)
    if not joined.strip():
        detail = "no content deltas"
        if '"finish_reason":"length"' in raw or '"finish_reason": "length"' in raw:
            detail = ("truncated (finish_reason=length) before any content - "
                      "reasoning consumed the ceiling; raise max tokens")
        return [Result("stream_content", "FAIL", detail)]
    leak = leaks(joined)
    return [Result("stream_content", "FAIL" if leak else "PASS",
                   f"leaked {leak!r} into deltas" if leak else f"{len(text)} deltas")]


def case_tool_single(base, key, model, nc):
    st, doc = post(base, key, model, {
        "messages": [{"role": "user", "content": "What's the weather in Denver?"}],
        "tools": WEATHER_TOOL, "max_completion_tokens": 600,
    }, nc)
    if st != 200:
        return [Result("tool_single", "FAIL", f"http {st}")]
    m = msg_of(doc)
    calls = m.get("tool_calls") or []
    if not calls:
        return [Result("tool_single", "FAIL",
                       f"no tool_calls (finish={finish_of(doc)})")]
    c = calls[0]
    problems = []
    if (c.get("function") or {}).get("name") != "get_weather":
        problems.append(f"wrong name {(c.get('function') or {}).get('name')!r}")
    try:
        args = json.loads((c.get("function") or {}).get("arguments") or "")
        if "city" not in args:
            problems.append("arguments missing required 'city'")
    except ValueError:
        problems.append("arguments are not valid JSON")
    if not c.get("id"):
        problems.append("call id missing")
    if finish_of(doc) != "tool_calls":
        problems.append(f"finish_reason={finish_of(doc)!r} not 'tool_calls'")
    leak = leaks(m.get("content") or "")
    if leak:
        problems.append(f"tool syntax leaked into content ({leak!r})")
    return [Result("tool_single", "FAIL" if problems else "PASS", "; ".join(problems))]


def case_tool_forced(base, key, model, nc):
    st, doc = post(base, key, model, {
        "messages": [{"role": "user", "content": "Weather in Reno?"}],
        "tools": WEATHER_TOOL,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "max_completion_tokens": 400,
    }, nc)
    if st != 200:
        return [Result("tool_forced", "FAIL", f"http {st}")]
    m = msg_of(doc)
    if m.get("tool_calls"):
        return [Result("tool_forced", "PASS")]
    # Forcing a tool and getting empty content is the worst outcome: the client
    # sees a successful, meaningless response rather than an error it can act on.
    body = (m.get("content") or "").strip()
    return [Result("tool_forced", "FAIL",
                   "forced call ignored; " + ("empty content" if not body else "answered as prose"))]


def case_tool_parallel(base, key, model, nc):
    st, doc = post(base, key, model, {
        # Explicit on purpose: this case measures whether the serve CAN return two
        # calls in one response, not whether the model spontaneously parallelises
        # a vague ask. The soft phrasing scored glimmer (and occasionally ornith)
        # as incapable when a firm instruction yields 2/2 every time - a harness
        # must not report prompt sensitivity as a missing capability.
        "messages": [{"role": "user",
                      "content": "Get the weather for Denver AND Austin. "
                                 "You must call get_weather twice, once per city, "
                                 "in a single response."}],
        "tools": WEATHER_TOOL, "max_completion_tokens": 700,
    }, nc)
    if st != 200:
        return [Result("tool_parallel", "FAIL", f"http {st}")]
    n = len(msg_of(doc).get("tool_calls") or [])
    return [Result("tool_parallel", "PASS" if n >= 2 else "FAIL", f"{n} call(s)")]


def case_tool_stream(base, key, model, nc):
    st, raw = post(base, key, model, {
        "messages": [{"role": "user", "content": "Weather in Boise?"}],
        "tools": WEATHER_TOOL, "max_completion_tokens": 400, "stream": True,
    }, nc, stream=True)
    if st != 200 or not isinstance(raw, str):
        return [Result("tool_stream", "FAIL", f"http {st}")]
    frags: dict[int, list[str]] = {}
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        blob = line[5:].strip()
        if blob == "[DONE]":
            break
        try:
            d = json.loads(blob)["choices"][0].get("delta") or {}
        except Exception:
            continue
        for tc in d.get("tool_calls") or []:
            idx = tc.get("index", 0)
            piece = (tc.get("function") or {}).get("arguments")
            if isinstance(piece, str):
                frags.setdefault(idx, []).append(piece)
    if not frags:
        return [Result("tool_stream", "FAIL", "no tool-call deltas")]
    for idx, parts in frags.items():
        try:
            json.loads("".join(parts))
        except ValueError:
            return [Result("tool_stream", "FAIL",
                           f"call {idx} arguments never assembled into JSON")]
    return [Result("tool_stream", "PASS", f"{len(frags)} call(s) assembled")]


def case_tool_loop(base, key, model, nc):
    """The real agent loop. If this fails, agent mode is unusable regardless."""
    msgs = [{"role": "user",
             "content": "What's the weather in Denver? Use the tool, then answer in one sentence."}]
    st, doc = post(base, key, model,
                   {"messages": msgs, "tools": WEATHER_TOOL, "max_completion_tokens": 800}, nc)
    if st != 200:
        return [Result("tool_loop", "FAIL", f"step 1 http {st}")]
    a = msg_of(doc)
    calls = a.get("tool_calls") or []
    if not calls:
        return [Result("tool_loop", "SKIP", "no tool call to continue from")]
    msgs.append({"role": "assistant", "content": a.get("content"), "tool_calls": calls})
    msgs.append({"role": "tool", "tool_call_id": calls[0].get("id"),
                 "content": json.dumps({"tempF": 41, "sky": "snow"})})
    st2, doc2 = post(base, key, model,
                     {"messages": msgs, "tools": WEATHER_TOOL, "max_completion_tokens": 800}, nc)
    if st2 != 200:
        return [Result("tool_loop", "FAIL", f"step 2 http {st2}")]
    final = msg_of(doc2).get("content") or ""
    problems = []
    if not ("41" in final or "snow" in final.lower()):
        problems.append("final answer ignored the tool result")
    leak = leaks(final)
    if leak:
        problems.append(f"leaked {leak!r} into the final answer")
    return [Result("tool_loop", "FAIL" if problems else "PASS", "; ".join(problems))]


def case_structured(base, key, model, nc):
    # 900, not 400: reasoning tokens count toward the ceiling, so a reasoning
    # model under load can burn a tight budget and truncate mid-JSON - which
    # then scores as "not JSON" and reads like a schema failure. The case
    # measures whether the model CAN produce schema-conformant JSON, so it gets
    # fair room; truncation is reported as its own distinct outcome below.
    st, doc = post(base, key, model, {
        "messages": [{"role": "user", "content": "Give me a city and its country."}],
        "max_completion_tokens": 900,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "loc", "strict": True,
            "schema": {"type": "object",
                       "properties": {"city": {"type": "string"},
                                      "country": {"type": "string"}},
                       "required": ["city", "country"],
                       "additionalProperties": False}}},
    }, nc)
    if st != 200:
        return [Result("structured_output", "FAIL", f"http {st}")]
    try:
        json.loads(msg_of(doc).get("content") or "")
        return [Result("structured_output", "PASS")]
    except ValueError:
        if finish_of(doc) == "length":
            return [Result("structured_output", "FAIL",
                           "truncated (finish_reason=length) - reasoning consumed the "
                           "ceiling; raise max tokens rather than blaming the schema")]
        return [Result("structured_output", "FAIL",
                       "200 but content is not JSON (schema ignored)")]


def case_ceilings(base, key, model, nc):
    """A client's output ceiling is a cost and latency control. Ignoring it
    silently is worse than rejecting it, because the client believes it is capped."""
    out = []
    for case, param in (("max_tokens_honoured", "max_tokens"),
                        ("max_completion_tokens_honoured", "max_completion_tokens")):
        st, doc = post(base, key, model, {
            "messages": [{"role": "user", "content": "Write a 300 word essay about rivers."}],
            param: 16,
        }, nc)
        if st != 200:
            out.append(Result(case, "FAIL", f"http {st}"))
            continue
        used = ((doc.get("usage") or {}).get("completion_tokens")
                if isinstance(doc, dict) else None)
        if used is None:
            out.append(Result(case, "SKIP", "no usage reported"))
        elif used <= 32:
            out.append(Result(case, "PASS", f"{used} tokens"))
        else:
            out.append(Result(case, "FAIL", f"asked 16, got {used} — silently overridden"))
    return out


CASES = [case_chat, case_stream, case_tool_single, case_tool_forced,
         case_tool_parallel, case_tool_stream, case_tool_loop,
         case_structured, case_ceilings]

CASE_ORDER = ["chat", "content_clean", "reasoning_separated", "stream_content",
              "tool_single", "tool_forced", "tool_parallel", "tool_stream",
              "tool_loop", "structured_output",
              "max_tokens_honoured", "max_completion_tokens_honoured"]


def discover(base: str, key: str) -> list[str]:
    req = urllib.request.Request(f"{base.rstrip('/')}/models",
                                headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            doc = json.loads(r.read().decode())
        return [m["id"] for m in doc.get("data", []) if isinstance(m.get("id"), str)]
    except Exception as e:
        print(f"could not discover models: {e}", file=sys.stderr)
        return []


def is_third_party(model: str, nodes: tuple[str, ...]) -> bool:
    return any(model == n or model.startswith(n + ":") for n in nodes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", action="append", default=[],
                    help="repeatable; omit with --all to test everything discovered")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable matrix")
    ap.add_argument("--expect", help="baseline JSON to compare against; regressions exit 1")
    ap.add_argument("--third-party-nodes", default="amd-mi300x,gb200,amd-dev",
                    help="models on these nodes need the non_confidential declaration")
    args = ap.parse_args()

    key = os.environ.get("CIQ_RELAY_KEY") or os.environ.get("LLM_RELAY_API_KEY") or ""
    if not key:
        print("set CIQ_RELAY_KEY", file=sys.stderr)
        return 2

    nodes = tuple(n.strip() for n in args.third_party_nodes.split(",") if n.strip())
    models = args.model or (discover(args.base, key) if args.all else [])
    if not models:
        print("nothing to test: pass --model or --all", file=sys.stderr)
        return 2

    matrix: dict[str, dict[str, dict]] = {}
    for model in models:
        nc = is_third_party(model, nodes)
        results: dict[str, dict] = {}
        for case in CASES:
            for r in case(args.base, key, model, nc):
                results[r.case] = {"status": r.status, "detail": r.detail}
        # One labeled retry for failures. Models are nondeterministic and the
        # fleet serves real traffic during a run, so single-shot reds mix hard
        # failures with load flakes. A pass-on-retry is recorded as PASS with
        # the flake noted - visible, not hidden - while a double failure keeps
        # the first detail. Never more than one retry: a case that needs three
        # tries IS a reliability finding.
        failed = {c for c, v in results.items() if v["status"] == "FAIL"}
        if failed:
            for case in CASES:
                retry = {r.case: r for r in case(args.base, key, model, nc)}
                for c, r in retry.items():
                    if c in failed and r.status == "PASS":
                        results[c] = {"status": "PASS",
                                      "detail": "flaky: failed once, passed on retry"}
        matrix[model] = results
        if not args.json:
            fails = [c for c, v in results.items() if v["status"] == "FAIL"]
            print(f"  {model}: {len(fails)} failing", file=sys.stderr)

    if args.json:
        print(json.dumps(matrix, indent=2))
    else:
        width = max(len(m) for m in matrix) + 2
        print()
        print("| Case".ljust(34) + "".join(f"| {m} " for m in matrix) + "|")
        print("|" + "-" * 33 + "".join("|" + "-" * (width + 1) for _ in matrix) + "|")
        for case in CASE_ORDER:
            row = f"| `{case}`".ljust(34)
            for m in matrix:
                v = matrix[m].get(case, {"status": "-"})
                mark = {"PASS": "pass", "FAIL": "**FAIL**", "SKIP": "skip"}.get(v["status"], "-")
                row += f"| {mark} ".ljust(width + 2)
            print(row + "|")
        print()
        for m, res in matrix.items():
            bad = [(c, v["detail"]) for c, v in res.items() if v["status"] == "FAIL"]
            if bad:
                print(f"{m}:")
                for c, d in bad:
                    print(f"  {c}: {d}")

    if args.expect:
        try:
            with open(args.expect) as fh:
                expected = json.load(fh)
        except OSError as e:
            print(f"could not read baseline: {e}", file=sys.stderr)
            return 2
        regressions = []
        for m, res in matrix.items():
            for case, v in res.items():
                was = (expected.get(m) or {}).get(case, {}).get("status")
                if was == "PASS" and v["status"] == "FAIL":
                    regressions.append(f"{m}/{case}: {v['detail']}")
        if regressions:
            print("\nREGRESSIONS vs baseline:", file=sys.stderr)
            for r in regressions:
                print(f"  {r}", file=sys.stderr)
            return 1
        print("\nno regressions vs baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
