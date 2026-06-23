"""Per-user API-key auth model + resolution.

Pure module: no FastAPI imports, so the auth logic is unit-testable without a
server. Enforcement (middleware) lives in ``llm_relay.api.middleware``.

Keys are stored hashed (sha256 hex), so the on-disk key store never holds a
plaintext key. Resolution is not host-based: the relay typically runs behind a
loopback reverse proxy, so trusting the peer address would bypass auth for all
proxied traffic. A valid key is required, full stop (with exempt paths handled
by the middleware).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Principal:
    """An authenticated caller. ``priority_weight`` is carried for the scheduler
    (QoS fair-share); ``scopes`` are reserved for later enforcement."""

    id: str
    priority_weight: float = 1.0
    scopes: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AuthConfig:
    enabled: bool = False
    # Paths served without a key. The default keeps a Prometheus scrape and a
    # liveness probe working; everything else needs a key when enabled. Not
    # host-based on purpose -- see the module docstring.
    exempt_paths: list[str] = field(default_factory=lambda: ["/health", "/metrics"])
    principals_by_hash: dict[str, "Principal"] = field(default_factory=dict)


class AuthError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def hash_key(plaintext: str) -> str:
    """sha256 hex of a key. The key store holds only these digests."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def load_keys(path: Path) -> dict[str, Principal]:
    """Parse the key store (``{key_hash: principal}``). ``{}`` if absent."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, Principal] = {}
    for key_hash, meta in (raw.get("keys") or {}).items():
        meta = meta or {}
        out[str(key_hash)] = Principal(
            id=str(meta.get("id", "unknown")),
            priority_weight=float(meta.get("priority_weight", 1.0)),
            scopes=list(meta.get("scopes", []) or []),
            enabled=bool(meta.get("enabled", True)),
        )
    return out


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the presented key from ``Authorization: Bearer <key>`` (preferred)
    or ``X-API-Key: <key>``."""
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return None


def authenticate(
    authorization: str | None,
    x_api_key: str | None,
    cfg: AuthConfig,
) -> Principal | None:
    """Resolve a presented key to a ``Principal``.

    Returns ``None`` when auth is disabled (caller treats as anonymous-allowed).
    Raises ``AuthError`` when enabled and the key is missing, unknown, or
    disabled. Returns the ``Principal`` on success.
    """
    if not cfg.enabled:
        return None
    key = _extract_key(authorization, x_api_key)
    if key is None:
        raise AuthError("missing API key")
    presented = hash_key(key)
    for stored_hash, principal in cfg.principals_by_hash.items():
        if hmac.compare_digest(stored_hash, presented):
            if not principal.enabled:
                raise AuthError("key disabled")
            return principal
    raise AuthError("unknown API key")


def mint_key(
    id: str,
    priority_weight: float = 1.0,
    scopes: list[str] | None = None,
) -> tuple[str, Principal]:
    """Mint a new key. Returns ``(plaintext, principal)``; the plaintext is shown
    once by the CLI and never persisted (only its hash is stored)."""
    plaintext = "llmr_" + secrets.token_urlsafe(32)
    return plaintext, Principal(id=id, priority_weight=priority_weight, scopes=scopes or [])


def write_keys(path: Path, principals_by_hash: dict[str, Principal]) -> None:
    """Persist the key store (hashes only)."""
    doc = {
        "keys": {
            h: {
                "id": p.id,
                "priority_weight": p.priority_weight,
                "scopes": p.scopes,
                "enabled": p.enabled,
            }
            for h, p in principals_by_hash.items()
        }
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=True))


def revoke_id(path: Path, id: str) -> int:
    """Remove every key whose principal id matches. Returns the count removed."""
    principals = load_keys(path)
    kept = {h: p for h, p in principals.items() if p.id != id}
    removed = len(principals) - len(kept)
    write_keys(path, kept)
    return removed
