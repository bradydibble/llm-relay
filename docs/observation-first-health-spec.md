# Observation-first health — spec

**Status:** approved direction (Brady, 2026-08-22). **Implemented 2026-08-22**
— see §8 for what shipped and the two deliberate deviations.

## 1. Problem

The relay keeps a *predictive* model of backend availability — the L0
`/v1/models` poll, the L2 generation probe, and their circuit breakers — and
routes against that model. Every prediction layer is a place false positives
live, and on this fleet's topology they fire constantly: the health signals
share one thin, saturable pipe with the traffic they judge. The gb200 path is
the worst case (Atmos VPN → ssh forward → loopback; one 512k-context upload
saturates it), but llama-01 over the tailnet flaps the same way under load.

The night of 2026-08-21/22 made the cost concrete. The relay 503'd every
named-model request for `gb200:glm-5.2-nvfp4` across repeated multi-minute
windows — `"status": "unavailable"` — while the same backend was returning
200s to real completions *during the windows*:

- L2 probes (30s first-token deadline) starved behind long-context prefills
  and opened the circuit — fixed by `b7059c4` (probe failures are not counted
  while a real completion landed within `TRAFFIC_FRESH_S`).
- The L0 poll (5s GET) starved the same way. ONE empty poll set
  `status=unavailable` and wiped the model list; three opened the silent L0
  breaker — fixed by `567a2c1` (same evidence rule, shared implementation in
  `discovery.endpoint`).
- Every relay redeploy (six that evening) reset all state and re-opened the
  windows at cold start.
- A real vLLM restart (~20 min: 724s weight load + autotune + CUDA graphs)
  produced *correct* unavailability that was indistinguishable, in the error
  payload and the logs, from all of the above.

The two fixes landed the principle: **a completed real request is proof of
liveness that outranks any starved probe.** This spec finishes the inversion.

Why not drop prediction entirely, like LiteLLM does (send, fail, fall back)?
Because two of our constraints make its posture wrong here:

1. **Named-model requests never substitute** (confidentiality and
   borrowed-hardware custody). A gateway that discovers failures by falling
   back can hide its own mistakes; ours turns every availability error into a
   user-visible 503, so the availability data must be *right*, not papered
   over.
2. **Wedge detection is real.** A wedged slot (stuck request, repetition
   loop, GPU hang) accepts connections and returns 200 headers while
   generating nothing. Reactive gateways "solve" this by hanging the user's
   request (the 2026-08-16 narf-agent hang). The L2 generation probe exists
   for this and stays.

So: observation first, probes demoted to where observation is blind — idle
backends and wedge detection.

## 2. Signal hierarchy (target state)

Authority over a backend's availability, highest first:

1. **Real request outcomes** (passive, continuous when traffic flows):
   - a completed successful request ⇒ up, immediately and unconditionally;
   - consecutive *transport* failures of real requests ⇒ down, faster than
     any probe could conclude it.
2. **Active probes** (L0 poll, L2 generation probe): authoritative only for
   backends with no recent traffic. Skipped entirely while traffic evidence
   is fresh.
3. **Cached state**: never the last word for a named-model refusal — a 503
   must reflect a live attempt, not a stale poll.

`TRAFFIC_FRESH_S = 120.0` (in `discovery.endpoint`, single source) remains
the evidence window everywhere.

## 3. Design

### 3.1 Passive failure evidence (symmetric stamping)

Today the router stamps only success (`last_traffic_success_ns`). Add the
failure half, on `EndpointClient`:

```
consecutive_traffic_failures: int      # transport failures of REAL requests
last_traffic_failure_ns: int | None
```

Stamped by `RequestRouter` (`_note_traffic_failure(backend_key)`) at the same
altitude as `_note_traffic_success`, on the request paths only — never by
probes or polls.

**What counts as a traffic failure** (transport-level only — the request
never got a well-formed answer):
- connect refused / connect timeout,
- read timeout / reset mid-body before completion,
- HTTP 502/503/504 *from the backend itself*.

**What must NOT count:**
- 4xx of any kind (client's problem; context-overflow 400s especially — they
  prove the backend is alive and parsing),
- client-side cancellation / disconnect of a streamed response (we chose to
  stop reading; says nothing about the backend),
- failures of requests the relay refused before dispatch.

**Effect:** `consecutive_traffic_failures >= 3` marks the backend
unavailable (and opens the L0 breaker), regardless of what probes think. Any
traffic success zeroes it and closes circuits — one real 200 outranks
everything. Rationale for 3: one reset can be a single dropped ssh channel;
three in a row with zero successes interleaved is a dead path. This closes
the known asymmetry where a backend that hangs *real* requests coasts up to
120s on its last success: hanging requests now accumulate failure evidence
and take it down without waiting for a probe cycle.

### 3.2 Probe demotion (skip what observation already answers)

- `DiscoveryManager._poll_client_once`: if `traffic_is_fresh(client)` and the
  catalog is non-empty, skip the GET entirely (today we perform it and ignore
  its failure). The model list refreshes on the next *idle* poll; a live
  model-set change on a busy backend is already handled by config-drift
  detection and, in the worst case, waits ≤120s of quiet.
- `L2HealthProbe._probe_all`: skip backends with fresh traffic. Today a
  starved probe is merely not counted; not sending it at all also stops
  probes competing with real prefills for the thin pipe.
- Idle backends keep today's behavior bit-for-bit: poll every interval, L2
  probe every 30s, breakers as configured.

### 3.3 Optimistic dispatch for named models (no refusal from cache)

In the named-model path, when the model's only backend is marked
unavailable/degraded, do not 503 from state. Instead:

- attempt the actual request against the backend with
  `httpx.Timeout(connect=3.0, …)` (normal read timeout — if it connects, it
  is a real request and gets to run);
- success ⇒ serve it, stamp evidence (state heals as a side effect);
- transport failure ⇒ 503 as today, but the decision payload now states a
  live attempt was made (§3.4), and the failure stamps §3.1 evidence.

**Stampede guard:** at most ONE optimistic in-flight attempt per backend
(`asyncio.Lock` per `EndpointClient`); concurrent named requests during an
outage get the current 503 with `"checked_live": false` and the age of the
last live attempt. A dead backend costs one 3s connect per storm, not one per
request.

Aliases are unchanged: they route over live candidates and never needed this.

### 3.4 Reason-coded refusals

The `named_model` error payload and a new structured log line carry:

```
"reason":        "refused" | "timeout" | "traffic_failures" | "starting" | "never_seen"
"checked_live":  true | false          # was §3.3 attempted for THIS response
"last_traffic_success_age_s": float | null
```

- `refused` — TCP refusal at the backend (dead or booting; on gb200 this is
  "vLLM not bound yet", the ~20-min restart window),
- `timeout` — connect/read starvation (stalled tunnel or saturation),
- `traffic_failures` — taken down by §3.1 evidence,
- `starting` — refused AND the backend completed a request within the last
  hour (heuristic for "restarting, come back in ~20 min" on gb200-class
  backends; best-effort, never load-bearing),
- `never_seen` — no evidence this process lifetime and probes failing.

Metrics: `llm_relay_requests_total` outcome gains
`named_unavailable_<reason>` (or a separate `reason` label on a new
`llm_relay_named_refusals_total` counter — implementer's choice; keep
cardinality ≤ the five reasons). This is what turns "insanely disruptive" into
a graph, and it is what the portal's Tests/Status pages will read (portal
work tracked there, not here).

### 3.5 State persistence across restarts

Cold starts caused real 503 windows six times in one evening. Persist, per
backend, on change (debounced ≥5s), to
`$LLM_RELAY_STATE_DIR/backend-state.json` (the relay already has the
writable state dir):

```
{ "<backend key>": { "status": "...", "models": [...],
                      "last_traffic_success_ns": ... } }
```

At boot, load entries newer than 10 minutes as priors: status and catalog
adopted as-is, traffic evidence adopted so §3.1/§3.2 behave as if the restart
never happened. Older entries are ignored (cold start as today). The file is
advisory — corrupt/missing/stale means "no priors", never a crash
(`try/except` around the whole load; see the api_keys.yaml handling for the
write-temp-then-rename pattern).

## 4. Invariants (each becomes a test)

1. A backend with a completed real request in the last `TRAFFIC_FRESH_S` is
   never marked unavailable by any probe or poll, and never has its catalog
   wiped by one. (Exists: `test_l0_traffic_evidence`, `test_health_probe`.)
2. Three consecutive transport failures of real requests mark a backend
   unavailable even if every probe passes; one success reverses it.
3. 4xx responses, context-overflow rejections, and client aborts never
   change availability state.
4. A named-model request never receives a 503 whose `checked_live` is false
   unless another live attempt is already in flight for that backend.
5. A wedged-but-accepting idle backend is still detected by the L2 probe
   within today's bounds (no regression on the original wedge guarantee).
6. A relay restarted under load reaches the pre-restart availability state
   without a user-visible window (priors ≤10 min old).
7. Fail closed: no code path widens a named request onto a different model —
   observation changes *when* we refuse, never *what* we serve.

## 5. Non-goals

- No change to no-substitution semantics, confidentiality gating, or alias
  routing policy.
- No removal of L0/L2 probes (demotion only).
- No cross-process sharing of evidence (single-gateway reality; revisit if a
  second relay instance ever exists).
- Portal/ops surfacing (uptime strips, WBR `no_candidate` KPI relabel, alert
  rules) — tracked in llm-fleet-portal / llm-relay-ops, consumes §3.4.

## 6. Rollout order

Each step lands with its tests and deploys via `llm-relay-deploy` before the
next starts. Steps 1+3 remove essentially all remaining false 503s; 2 is
mostly deletion; 4 is observability; 5 kills the redeploy windows.

1. §3.1 passive failure evidence (touches `routing/router.py` — coordinate
   with any in-flight router work; the token-tracking workstream owns edits
   there as of this writing).
2. §3.3 optimistic dispatch + stampede guard.
3. §3.2 probe demotion.
4. §3.4 reason-coded refusals + metrics.
5. §3.5 state persistence.

## 8. Implementation notes (landed 2026-08-22, single change set)

All five steps shipped together (tests: `tests/test_observation_first.py`,
plus the pre-existing `test_l0_traffic_evidence.py` / `test_health_probe.py`
guards). Two deliberate deviations from the letter of the spec:

1. **§3.3 uses a live catalog check, not the request itself, as the probe.**
   `RequestRouter._named_live_check` GETs the backend's `/v1/models` with a
   3s connect timeout under the per-backend `optimistic_lock`; on success it
   heals L0 state in place, re-runs selection, and the request flows through
   the ONE normal dispatch path — the real request follows the check within
   milliseconds and stamps real evidence itself. Rationale: dispatching the
   user's request as the probe would have forked the streaming/non-streaming
   dispatch loop into a second implementation; the live check preserves a
   single dispatch path at the cost of one extra round trip on the (rare)
   recovery edge. A live `/v1/models` hit never overrules an L2 `degraded`
   verdict (invariant 5): listening and listing is not generating.
2. **§3.4 reason vocabulary grew** to match what the live check can actually
   distinguish: `refused | starting | timeout | not_loaded | degraded |
   error | never_seen | check_in_flight`, carried in
   `named_model.availability` alongside `checked_live` and
   `last_traffic_success_age_s`. The metric landed as
   `llm_relay_requests_total{outcome="named_unavailable_<reason>"}` with the
   refused model and provider (b7da7df) — closing the mislabel that hid
   named refusals inside `no_candidate`/`model=None` (and, downstream, inside
   the WBR's "context rejects").

§3.5 note: the state file lives beside the audit log
(`dirname($LLM_RELAY_AUDIT_LOG)/backend-state.json`) unless
`LLM_RELAY_STATE_DIR` overrides it, so no unit-file change was needed.

## 7. Prior art, for the reviewer who asks "why not just use X"

Envoy calls §3.1 *outlier detection* and §3.3's family *panic/slow-start
modes*; HAProxy calls §3.1 `observe layer7`. LiteLLM has neither active
generation probes nor named-model strictness, so it neither has this problem
nor our reasons for the machinery. We are not inventing a novel health model;
we are adopting the boring, proven one on top of the two constraints
(no-substitution, wedge detection) that are genuinely ours.
