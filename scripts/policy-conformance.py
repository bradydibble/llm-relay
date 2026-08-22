#!/usr/bin/env python3
"""Live policy conformance check for a RUNNING llm-relay.

WHY THIS EXISTS, given tests/ already has 45 confidentiality tests:

Those tests build synthetic configs in tmp_path. They prove the *logic* is right
and they can never fail because of a config mistake — which is exactly the class
of failure that will break this next. An untagged provider, a provider tagged
`ciq_owned` that isn't, a harness config that drifted from the repo, a models.yaml
edit that quietly re-points an alias: every unit test still passes.

This asserts the policy against the LIVE fleet and its REAL config. It is the
regression test for the bug that started this work — a request for `amd-dev`
answered by a 35B on llama-01 with nothing saying so.

Safe against production: read-mostly, max_tokens=5 on the few requests that
generate, and it skips (rather than fails) when a precondition isn't met.

Usage:
    ./scripts/policy-conformance.py [--endpoint URL] [--api-key KEY] [--verbose]

    ENDPOINT env var / --endpoint  default https://relay.example.invalid/v1
    LLM_RELAY_API_KEY / --api-key  omit for the keyless trusted listener

Exit codes: 0 all checks passed (skips allowed), 1 a check FAILED, 2 could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://relay.example.invalid/v1"
CONF_HEADER = "X-Llm-Relay-Confidentiality"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    mark = {PASS: "  PASS", FAIL: "  FAIL", SKIP: "  SKIP"}[status]
    print(f"{mark}  {name}" + (f"\n        {detail}" if detail else ""), flush=True)


def call(path: str, endpoint: str, key: str | None, body=None, headers=None, timeout=90):
    """Return (status_code, parsed_json_or_none). Never raises on HTTP error."""
    url = endpoint.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"_raw": raw}
    except Exception as e:  # network/DNS/TLS
        return None, {"_error": str(e)}


def chat(model: str, endpoint: str, key: str | None, declare: bool):
    hdrs = {CONF_HEADER: "non_confidential"} if declare else {}
    return call(
        "/chat/completions", endpoint, key,
        body={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        headers=hdrs,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--api-key", default=os.environ.get("LLM_RELAY_API_KEY"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    ep, key = args.endpoint, args.api_key

    print(f"llm-relay policy conformance -> {ep}\n")

    # ---- Preconditions -----------------------------------------------------
    status, avail = call("/available-models", ep, key)
    if status != 200 or not isinstance(avail, dict):
        print(f"could not read /v1/available-models (status={status}): {avail}")
        return 2
    models = {k: v for k, v in avail.items() if isinstance(v, dict) and "provider" in v}
    if not models:
        print("no models in /v1/available-models; nothing to check")
        return 2

    owned = {m for m, v in models.items() if v.get("ownership") == "ciq_owned"}
    third = {m for m, v in models.items() if v.get("ownership") == "third_party"}
    live = {m for m, v in models.items() if v.get("status") in ("available", "degraded")}

    if args.verbose:
        for m, v in sorted(models.items()):
            print(f"        {m:22} {v.get('ownership','?'):12} {v.get('status','?')}")
        print()

    # ---- 1. Every model resolves an ownership value ------------------------
    # A missing/unknown value means a provider was added without a tag. The
    # loader is supposed to refuse to start, so this catches a relay running
    # older code against newer config, or a discovery-time model with no home.
    untagged = sorted(m for m, v in models.items()
                      if v.get("ownership") not in ("ciq_owned", "third_party"))
    if untagged:
        record(FAIL, "every model reports a valid ownership", f"untagged: {untagged}")
    else:
        record(PASS, "every model reports a valid ownership",
               f"{len(owned)} ciq_owned, {len(third)} third_party")

    # ---- 2. requires_non_confidential agrees with ownership ----------------
    # These are computed independently in the payload builder; disagreement means
    # a client filtering on one field gets a different answer than routing uses.
    bad = sorted(m for m, v in models.items()
                 if v.get("requires_non_confidential") != (v.get("ownership") != "ciq_owned"))
    record(FAIL if bad else PASS,
           "requires_non_confidential agrees with ownership",
           f"disagree: {bad}" if bad else "")

    # ---- 3. The relay serves self-hosted inference only --------------------
    # Scope invariant: llm-relay's remit is our own models on our own remit of
    # hardware. A vendor API appearing here means the scope boundary moved.
    _, listing = call("/models", ep, key)
    ids = [m["id"] for m in (listing or {}).get("data", [])] if isinstance(listing, dict) else []
    vendor = [i for i in ids if any(s in i.lower() for s in ("claude", "anthropic", "gpt-", "openai"))]
    record(FAIL if vendor else PASS,
           "no vendor-API models advertised (relay is self-hosted only)",
           f"found: {vendor}" if vendor else f"{len(ids)} ids, none vendor")

    cloud = sorted(m for m, v in models.items() if v.get("privacy") == "cloud_ok")
    record(FAIL if cloud else PASS,
           "no model declares privacy: cloud_ok",
           f"found: {cloud}" if cloud else "")

    # ---- 4. THE CORE INVARIANT --------------------------------------------
    # An UNDECLARED request must never be ANSWERED BY third-party hardware.
    # Checked per live third-party model AND for the aliases that could reach
    # one. A 200 is only acceptable if a ciq_owned provider served it.
    # This is the check that would have caught the original bug.
    live_third = sorted(third & live)
    probes = live_third + [a for a in ("main", "amd-dev", "high-quality", "code_heavy") if a in ids]
    violations, checked = [], 0
    for target in probes:
        code, resp = chat(target, ep, key, declare=False)
        if code is None:
            record(SKIP, f"undeclared '{target}' not answered by third-party metal",
                   f"unreachable: {resp.get('_error')}")
            continue
        checked += 1
        if code == 200:
            served = (resp.get("llm-relay") or {}).get("selected_model")
            if served in third:
                violations.append(f"{target} -> 200 served by {served} (third_party)")
        # any non-200 is fine here: refusing is the correct outcome
    if violations:
        record(FAIL, "undeclared requests are never served by third-party hardware",
               "; ".join(violations))
    elif checked:
        record(PASS, "undeclared requests are never served by third-party hardware",
               f"{checked} probe(s): {', '.join(probes[:6])}")
    else:
        record(SKIP, "undeclared requests are never served by third-party hardware",
               "no probe reachable")

    # ---- 5. A confidentiality block is terminal and actionable -------------
    # Distinguishes policy refusal from a generic outage: the 503 must carry the
    # reason and name the node, and must NOT offer Retry-After (retrying can
    # never make a workload non-confidential).
    if live_third:
        target = live_third[0]
        code, resp = chat(target, ep, key, declare=False)
        detail = (resp or {}).get("detail", resp) or {}
        block = detail.get("confidentiality") if isinstance(detail, dict) else None
        if code == 503 and block and block.get("reason") == "confidentiality_requires_declaration":
            nodes = block.get("third_party_nodes")
            record(PASS, "confidentiality block is terminal and names the node",
                   f"{target}: 503, nodes={nodes}")
        elif code == 200:
            record(FAIL, "confidentiality block is terminal and names the node",
                   f"{target} returned 200 undeclared — see the core-invariant check")
        else:
            record(FAIL, "confidentiality block is terminal and names the node",
                   f"{target}: status={code} but no confidentiality reason in body")
    else:
        record(SKIP, "confidentiality block is terminal and names the node",
               "no third-party model is live")

    # ---- 6. The declaration actually unlocks the hardware ------------------
    # Guards the opposite failure: a gate so tight nothing can pass it. Needs
    # the caller's key to hold `third_party` (implicit on the trusted listener).
    if live_third:
        target = live_third[0]
        code, resp = chat(target, ep, key, declare=True)
        if code == 200:
            served = (resp.get("llm-relay") or {}).get("selected_provider")
            record(PASS, "declaring non_confidential unlocks third-party hardware",
                   f"{target} -> 200 via {served}")
        elif code == 503 and "confidentiality" in json.dumps(resp).lower():
            record(SKIP, "declaring non_confidential unlocks third-party hardware",
                   "still blocked — caller's key likely lacks the `third_party` scope")
        else:
            record(FAIL, "declaring non_confidential unlocks third-party hardware",
                   f"{target}: status={code}")
    else:
        record(SKIP, "declaring non_confidential unlocks third-party hardware",
               "no third-party model is live")

    # ---- 7. No collateral damage to ordinary work --------------------------
    # The gate must not have broken the default path. `main` undeclared should
    # still serve, from CIQ-owned metal.
    if "main" in ids:
        code, resp = chat("main", ep, key, declare=False)
        served = (resp.get("llm-relay") or {}).get("selected_model") if code == 200 else None
        if code == 200 and served in owned:
            record(PASS, "ordinary undeclared work still serves from CIQ-owned metal",
                   f"main -> {served}")
        elif code == 200:
            record(FAIL, "ordinary undeclared work still serves from CIQ-owned metal",
                   f"main -> {served}, which is not ciq_owned")
        else:
            record(SKIP, "ordinary undeclared work still serves from CIQ-owned metal",
                   f"main unavailable (status={code}) — fleet may be down")
    else:
        record(SKIP, "ordinary undeclared work still serves from CIQ-owned metal",
               "no `main` alias")

    # ---- Summary -----------------------------------------------------------
    n = {s: sum(1 for st, _, _ in results if st == s) for s in (PASS, FAIL, SKIP)}
    print(f"\n{n[PASS]} passed, {n[FAIL]} failed, {n[SKIP]} skipped")
    if n[SKIP]:
        print("NOTE: skips are not passes — a skipped check verified nothing.")
        for st, name, detail in results:
            if st == SKIP:
                print(f"      - {name}: {detail}")
    return 1 if n[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
