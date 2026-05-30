# Host-qualified model identity

Status: accepted (2026-05-30)

## Problem

The relay identifies a model by its bare config name. When the same upstream
model runs on more than one backend host, clients cannot tell the instances
apart in `/v1/models`, and discovery's name→backend map (`model_to_client`) is
1:1, so registering the same name twice silently hides one host. Operators also
have no stable way to address "this model on this specific node" — e.g. to
verify a single deployment in isolation.

## Principles

- **Local-first.** Optimize for a small, curated local fleet; cloud support is
  incidental, never the design center.
- **Categories are the front door.** Clients ask for a use-case alias ("give me
  fast code"), and the relay resolves to whatever is deployed now. Clients
  depend only on the category name; the operator curates what it maps to.

## Two addressing methods (only)

1. **Category / use-case alias** (preferred) — the existing alias chains;
   unchanged by this work.
2. **Host-qualified id `provider:model`** — a unique, self-describing handle for
   one model on one node, used for testing/pinning a deployment and for
   client-side differentiation in `/v1/models`.

There is intentionally **no** host-agnostic logical (load-balanced) name: the
same model on different hosts is tuned differently (slot count, context, quant)
and is therefore not interchangeable. Cross-host balancing is intentionally
absent; an alias chain provides ordered fallback across deployments.

## Design

**Identity.** The canonical id for every configured model is
`f"{provider}:{name}"`, derived from the model's existing `provider` field — no
config migration. Example: model `model-x` on provider `prov-a` →
`prov-a:model-x`.

**`/v1/models` and `/v1/models/{model}`.** The list advertises qualified ids
plus aliases (categories). `_model_entry` emits `id = provider:name`, while
context metadata is still resolved by the bare name. `/v1/models/{model}`
resolves either a bare or qualified id (echoing the id it was asked for) and
404s otherwise. Bare names are no longer *advertised* but still **resolve**
(backward compatibility).

**Routing resolution.** The requested model resolves in order: alias → exact
`provider:model` → bare name → unknown. A `provider:model` id is split on the
first `:` and accepted only when `name` is a configured model whose provider
equals `provider`; otherwise it is unknown (→ 404 / no candidate). No current
provider, model, or alias name contains `:`, so the split is unambiguous.

**Discovery.** No change. A qualified id normalizes to the existing bare-name
lookups, which are keyed by distinct config names, so nothing collides. *Known
limitation (out of scope):* `get_available_models()` keys by the id a backend
*reports*, so two backends reporting the same upstream id collapse there — this
affects only the dynamic-discovery fallback, never the config-driven surface or
routing.

**Errors.** An unknown model or a mismatched `provider:model` pair returns 404,
matching the existing model-card path.

## Out of scope

Tier-2 logical/load-balanced names; dynamic capability-selector categories;
auto-resolving the reported-id collision; any cloud-first behavior.

## Test plan (TDD)

- A `provider:model` id round-trips through `/v1/models` (qualified id present,
  aliases present, bare name not advertised).
- A request for `provider:model` routes to that backend; a mismatched pair 404s
  / yields no candidate.
- The same base model on two providers yields two distinct, individually
  routable qualified ids.
- A bare model name still resolves (routing and the `/v1/models/{model}` card) —
  backward compatibility.
- Aliases are unchanged (still resolve and rank as before).
- Cloud models qualify uniformly (`prov-cloud:model-y`); bare still resolves.
