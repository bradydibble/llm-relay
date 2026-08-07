# Credential management for the llm-relay cockpit — spec

Status: **proposal** (not yet implemented). Awaiting Brady's review.

The "cockpit" is the operator management surface of llm-relay: the `/admin/*`
and introspection HTTP endpoints (consumed by the external Reno fleet dashboard)
plus the `llm-relay` CLI. Today it can pause/resume providers and tail logs, but
it has **no credential story**: inbound API keys are CLI-only, outbound upstream
keys are a single shared env var, and admin endpoints have no authorization
(any valid key works, scopes are dead).

This spec adds a unified credential-management surface to the cockpit covering
both directions, plus the authorization model that gates it.

## Goals

1. Manage **inbound** relay API keys (principals) over HTTP, not just the CLI —
   list, mint, revoke, enable/disable, edit priority/scopes — so the dashboard
   can do it without shelling out.
2. Manage **outbound** upstream provider credentials per-provider, replacing the
   single shared env-var key, with the `auth_source` field already on
   `ProviderConfig` finally made real.
3. Add **scope-based authorization** so admin/credential endpoints require an
   `relay:admin` principal, not just any key. Wire up the `Principal.scopes`
   field that already exists but is never checked.
4. **Rotate** an upstream credential live (hot-swap the bearer) without a relay
   restart.
5. **Never expose plaintext** inbound or outbound — the cockpit returns only
   fingerprints and "configured" flags.
6. **Audit** every credential change (who/when/what) to a durable log readable
   from the cockpit.

## Non-goals

- A new secrets backend. We resolve credential *references* to existing sources
  (env var, file path, a `get-secret`-style key). We are not building a vault.
- Encrypting outbound keys at rest in this iteration. Ref-based resolution keeps
  the relay out of the key-custody business. (Encrypted inline store is a named
  future extension, below.)
- Host/network authz. Stays key-based, not host-based (see `auth.py` docstring).
- Merging `llm-mode` capacity credentials. Out of scope; `llm-mode` keeps its own
  upstream-key story for systemd units.

## Current state (grounding)

| Concern | Today |
|---|---|
| Inbound key store | `api_keys.yaml`, sha256-hashed, 0600. `llm-relay keys add\|list\|revoke`. |
| Inbound enforcement | `AuthMiddleware` (raw ASGI). `Principal.scopes` parsed, **never checked**. |
| Admin authz | None. `/admin/pause`, `/admin/resume`, `/v1/jobs`, `/v1/jobs/{id}/cancel` accept any key (or anonymous when auth off). |
| Outbound key | One shared bearer from `LLM_RELAY_UPSTREAM_API_KEY` / `LLM_API_KEY` env (`endpoint._shared_upstream_bearer`). Used for every backend's `/v1/models` probe and (implicitly) the shared upstream. |
| `ProviderConfig.auth_source` | Parsed from yaml (e.g. `vault`), **never resolved**. Pure documentation hint today. |
| Per-provider credentials | None. |
| Rotation | Restart the relay (and re-read env). |
| Audit | None. |

## Design

### Two credential domains

Keep them separate — different threat models and storage shapes.

- **Inbound (relay API keys / principals)**: the relay is the relying party.
  Keys are **hashed** at rest (existing). Plaintext shown once at mint, never
  recoverable. This is unchanged; we add an HTTP management surface on top of
  the existing `auth.py` primitives.
- **Outbound (upstream provider keys)**: the relay is a *consumer* of someone
  else's secret. It needs the plaintext to send upstream, so it cannot hash it.
  We therefore store **references** to secrets held elsewhere, resolved at
  probe/forward time. The relay never becomes the system of record for upstream
  key material.

### Authorization model

Give `Principal.scopes` meaning. Scopes are strings, checked as a set with no
wildcards in v1.

| Scope | Grants |
|---|---|
| `relay:read` | `GET /status`, `/v1/available-models`, `/v1/models*`, `/routing-table*`, `/logs*`, `/health`. |
| `relay:route` | `POST /v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`. |
| `relay:jobs` | `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel` (own jobs; `relay:admin` for any). |
| `relay:admin` | `POST /admin/pause`, `/admin/resume`, and all `/admin/credentials/*`. |

Enforcement: a FastAPI dependency `require_scopes(*scopes)` that reads
`request.state.principal` (set by `AuthMiddleware`) and raises 403 when the
principal lacks any required scope. When auth is **disabled**, the dependency
passes (anonymous-allowed) — preserves the open-by-default posture.

Backward-compat: existing principals have empty `scopes`. On upgrade, an empty
`scopes` list is treated as **full access** (equivalent to all four scopes) so
no key silently loses capability. A new key minted with no scopes gets nothing;
the cockpit forces a scope selection at mint time. This is a one-line escape
hatch documented in `auth.yaml.example`.

### Inbound key management API

All under `/admin/credentials/principals`, all require `relay:admin`.

| Method + path | Body / result |
|---|---|
| `GET  /admin/credentials/principals` | List principals. Never returns key material; returns `id`, `priority_weight`, `scopes`, `enabled`, `key_count`, and a `fingerprint` per key (first 8 hex of the sha256) so an operator can match a held key to a row. |
| `POST /admin/credentials/principals` | `{id, priority_weight?, scopes?, enabled?}` -> `{plaintext}` shown **once** (201). Persists the hash. |
| `PATCH /admin/credentials/principals/{id}` | `{priority_weight?, scopes?, enabled?}` -> updated principal (200). Partial update. |
| `DELETE /admin/credentials/principals/{id}` | Revoke all keys for `id` (200, `{removed}` count). Mirrors `revoke_id`. |

Reuses `auth.mint_key` / `load_keys` / `write_keys` / `revoke_id` unchanged.
The CLI `llm-relay keys` subcommands keep working and become thin callers over
the same `auth.py` functions (no HTTP round-trip from the CLI — it writes the
file directly, as today).

### Outbound provider credential model

Formalize `ProviderConfig.auth_source` into a structured `credential` block.
Keep `auth_source` as a deprecated alias for the `ref` source for one release.

```yaml
providers:
  local-llm:
    base_url: http://127.0.0.1
    credential:
      source: env          # env | file | pass | none
      ref: LLM_RELAY_LOCAL_KEY   # env var name / file path / get-secret key
  example-cloud:
    base_url: https://cloud.example.invalid
    enabled: false
    credential:
      source: pass         # resolves via ~/bin/get-secret <ref> (homelab only)
      ref: example-cloud/shared/api-key
```

Sources:
- `env` — `os.environ[ref]`. Simplest; matches today's shared-key pattern.
- `file` — read `ref` path (0600 expected), one key, trailing whitespace
  stripped. Lets the systemd unit write a tmpfs file the relay reads.
- `pass` — shell out to `get-secret <ref>` (homelab convention). Cached in
  memory with a TTL; re-resolved on rotation or expiry. The public repo treats
  this as an opt-in backend; the example config ships `env`/`file` only.
- `none` / absent — no bearer sent (local backends with no auth), today's
  default.

Resolution lives in a new `llm_relay/credentials.py` (pure module, no FastAPI):
`resolve_credential(cred: CredentialRef) -> str | None` with an in-process cache
keyed by `(source, ref)`. The `EndpointClient` drops `_shared_upstream_bearer()`
and instead receives its bearer through a resolver bound at registration time in
`create_app`. Backward-compat: a provider with no `credential` block but a set
`auth_source` string, **or** the legacy shared env var present, falls back to
`LLM_RELAY_UPSTREAM_API_KEY` / `LLM_API_KEY` so nothing breaks on upgrade.

### Outbound credential management API

All require `relay:admin`. Never returns plaintext.

| Method + path | Result |
|---|---|
| `GET  /admin/credentials/providers` | Per provider: `source`, `ref` (ref shown — it is a name, not a secret), `configured` (bool: resolved to a non-empty value), `fingerprint` (last 4 chars + sha256 first 8 hex of the resolved secret), `last_resolved`. |
| `PUT  /admin/credentials/providers/{provider}` | `{source, ref}`. Validates the provider exists; resolves once to confirm; persists to `providers.yaml` (or a sibling `provider-credentials.yaml` so we don't rewrite the hand-edited file — see open question). |
| `POST /admin/credentials/providers/{provider}/rotate` | Force re-resolve (clears the cache entry) and re-probe. Used after the secret was rotated at its source. Returns new `fingerprint`. |
| `POST /admin/credentials/providers/{provider}/test` | One-shot probe: hit the backend `/v1/models` with the currently-resolved bearer, return `{ok, status, models_count}`. Does not persist. |

`rotate` is the hot-swap path: clearing the cache entry makes the next probe and
the next forwarded request pick up the new value, no restart.

### Audit log

Append-only JSONL at `<config_dir>/credential-audit.jsonl`, 0600. One line per
credential event:

```json
{"ts": "2026-06-24T12:00:00Z", "actor": "butler", "action": "mint_principal",
 "target": "pi", "detail": {"scopes": ["relay:route","relay:read"]}}
```

Actions: `mint_principal`, `revoke_principal`, `patch_principal`,
`set_provider_credential`, `rotate_provider_credential`, `test_provider_credential`.

`GET /admin/credentials/audit?limit=` tails it for the dashboard. Actor is the
calling principal's `id` (`"anonymous"` when auth off — logged loudly so an
unauthenticated admin path is visible).

## Data model changes

`config/types.py`:

```python
@dataclass
class CredentialRef:
    source: str          # env | file | pass | none
    ref: str | None = None

@dataclass
class ProviderConfig:
    # ... existing ...
    credential: CredentialRef | None = None
    # auth_source kept as deprecated alias -> credential{source:"ref"? ...}
```

`Principal` is unchanged (already has `scopes`, `enabled`). No new fields there.

`config/loader.py`:
- Parse `credential` block; map legacy `auth_source: <str>` to
  `CredentialRef(source=<str>, ref=None)` only for the `pass`/`vault` hint case
  (warn it is unresolved until a real `ref` is set).
- New `save_provider_credentials()` for the PUT path (or a separate
  `provider-credentials.yaml` — see open questions).

## Security considerations

- **Plaintext exposure**: no endpoint returns an inbound key (post-mint) or an
  outbound secret. `fingerprint` is sha256-prefix / last-4 only — enough to
  confirm "is this the key I think it is," not enough to recover it.
- **Outbound in memory**: resolved secrets live in the process cache only. The
  cache is keyed by `(source, ref)`, never logged, never put in a metric label.
  `pass`-source values are re-resolved on TTL expiry so a rotated `pass` entry
  is picked up without a relay restart.
- **File perms**: `api_keys.yaml` and `credential-audit.jsonl` stay 0600
  (existing `write_keys` already chmods). `provider-credentials.yaml` if added:
  0600, gitignored, lives in the off-repo config dir.
- **Auth-off posture unchanged**: when `LLM_RELAY_AUTH` is off, the relay stays
  an open loopback proxy. Credential endpoints are still reachable anonymously
  in that mode — documented as "do not run auth-off on a routable interface,"
  matching the existing README warning. The audit log records `actor:
  "anonymous"` so the path is at least visible.
- **No new inbound secrets on disk**: inbound keys stay hashed (sha256). We are
  not adding an encrypted-store dependency.
- **Scope escalation**: a principal cannot edit its own scopes (PATCH validates
  that the caller's scopes are a superset of the target's new scopes, or the
  caller has `relay:admin`). Prevents a `relay:route` key minting itself admin.

## Migration / rollout

1. **No-op upgrade.** Ship the scope dependency with the "empty scopes = full
   access" escape hatch. Existing keys keep working; `auth_source` still falls
   back to the shared env var. No config edit required.
2. **Opt-in authz.** Set `auth.enforce_scopes: true` in `auth.yaml` to flip
   empty-scopes from "full access" to "no access." Mint a `relay:admin` key for
   the dashboard first. Default `false` for one release.
3. **Per-provider credentials.** Add `credential` blocks to `providers.yaml` as
   you rotate. Each provider migrated off the shared env var. `rotate` confirms.
4. **Audit on by default.** The audit log writes from day one of the feature;
   backfill is not needed.

## CLI

Extend `llm-relay keys`:
- `llm-relay keys add <id> --scope relay:route --scope relay:read [--priority 2]`
  (scopes already on the parser, just exposed).
- `llm-relay keys disable <id>` / `enable <id>` / `set-scopes <id> --scope ...`.
- `llm-relay credentials list` — providers + configured/fingerprint.
- `llm-relay credentials set <provider> --source env --ref VAR`
  / `--source pass --ref service/identity/credential`.
- `llm-relay credentials rotate <provider>` / `test <provider>`.

## Tests (new)

- `test_scope_authz.py` — each scope gate; empty-scopes escape hatch on/off;
  self-escalation blocked.
- `test_credentials_api.py` — principal CRUD over HTTP; plaintext returned once;
  list never leaks material; provider GET shows `configured`/`fingerprint`
  without plaintext.
- `test_credential_resolver.py` — env/file/pass sources; cache hit/miss; rotate
  clears cache; legacy `auth_source` + shared-env fallback.
- `test_credential_audit.py` — every mutating API appends a line; actor recorded.
- Extend `test_provider_config.py` — `credential` block parse; `auth_source`
  deprecation alias.

## Open questions

1. **Where do provider credential refs persist?** Rewriting the hand-edited
   `providers.yaml` from the API is risky (comments, ordering). Preference: a
   sibling `provider-credentials.yaml` in the config dir that overlays
   `providers.yaml`'s `credential` blocks. Confirm.
2. **`pass` source in the public repo.** Shipping a `get-secret` shelling-out
   backend in a public repo is odd. Keep it, but document it as an
   opt-in/homelab convention and ship only `env`/`file` in the example config?
3. **Encrypted inline store.** A future iteration could add `source: inline`
   with AES-GCM at rest under `LLM_RELAY_CREDENTIAL_KEY`. Defer until someone
   has a deployment that genuinely cannot use env/file/pass.
4. **Scope for `/v1/jobs` ownership.** Should `relay:jobs` only see/cancel its
   own jobs, with `relay:admin` for cross-principal? Lean yes; confirm.
5. **Audit log retention.** Append-only JSONL grows forever. Rotate by size
   (e.g. 10MB) keeping the tail? Or leave it to logrotate?
