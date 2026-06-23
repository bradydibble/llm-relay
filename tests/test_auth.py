"""Auth + identity: principal model, key store, and resolution."""
from __future__ import annotations

import sys

import pytest

from llm_relay.auth import (
    AuthConfig,
    AuthError,
    Principal,
    authenticate,
    hash_key,
    load_keys,
    mint_key,
    revoke_id,
    write_keys,
)
from fastapi.testclient import TestClient

from llm_relay.api.app import create_app
from llm_relay.config.loader import ConfigLoader


# --- Task 1: model + key store --------------------------------------------

def test_hash_key_is_stable_and_hex():
    h = hash_key("llmr_abc")
    assert h == hash_key("llmr_abc")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_load_keys_absent_file_returns_empty(tmp_path):
    assert load_keys(tmp_path / "nope.yaml") == {}


def test_load_keys_parses_principals(tmp_path):
    p = tmp_path / "api_keys.yaml"
    p.write_text(
        "keys:\n"
        f"  {hash_key('llmr_secret')}:\n"
        "    id: alice\n"
        "    priority_weight: 2.0\n"
        "    scopes: [chat]\n"
    )
    principals = load_keys(p)
    pr = principals[hash_key("llmr_secret")]
    assert pr.id == "alice"
    assert pr.priority_weight == 2.0
    assert pr.scopes == ["chat"]
    assert pr.enabled is True


# --- Task 2: authenticate() ------------------------------------------------

def _cfg():
    return AuthConfig(
        enabled=True,
        principals_by_hash={hash_key("llmr_good"): Principal(id="alice")},
    )


def test_authenticate_disabled_returns_none():
    assert authenticate("Bearer x", None, AuthConfig(enabled=False)) is None


def test_authenticate_bearer_ok():
    pr = authenticate("Bearer llmr_good", None, _cfg())
    assert pr.id == "alice"


def test_authenticate_x_api_key_ok():
    pr = authenticate(None, "llmr_good", _cfg())
    assert pr.id == "alice"


def test_authenticate_missing_raises():
    with pytest.raises(AuthError):
        authenticate(None, None, _cfg())


def test_authenticate_unknown_raises():
    with pytest.raises(AuthError):
        authenticate("Bearer llmr_bad", None, _cfg())


def test_authenticate_disabled_principal_raises():
    cfg = AuthConfig(
        enabled=True,
        principals_by_hash={hash_key("llmr_x"): Principal(id="bob", enabled=False)},
    )
    with pytest.raises(AuthError):
        authenticate("Bearer llmr_x", None, cfg)


# --- Task 6: key store mint / write / revoke -------------------------------

def test_mint_and_roundtrip(tmp_path):
    path = tmp_path / "api_keys.yaml"
    plaintext, principal = mint_key("dave", priority_weight=3.0, scopes=["chat"])
    write_keys(path, {hash_key(plaintext): principal})
    assert plaintext.startswith("llmr_")
    loaded = load_keys(path)
    assert loaded[hash_key(plaintext)].id == "dave"
    assert loaded[hash_key(plaintext)].priority_weight == 3.0
    # The plaintext key is never written to disk, only its hash.
    assert plaintext not in path.read_text()
    # Owner-only permissions on the key store.
    assert (path.stat().st_mode & 0o777) == 0o600


def test_revoke_removes_id(tmp_path):
    path = tmp_path / "api_keys.yaml"
    pt, pr = mint_key("erin")
    write_keys(path, {hash_key(pt): pr})
    assert revoke_id(path, "erin") == 1
    assert load_keys(path) == {}


# --- Task 3: loader wires the key store into config.auth -------------------

def test_loader_loads_auth_keys(tmp_path, monkeypatch):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    (tmp_path / "api_keys.yaml").write_text(
        "keys:\n"
        f"  {hash_key('llmr_live')}:\n"
        "    id: carol\n"
    )
    monkeypatch.setenv("LLM_RELAY_AUTH", "1")
    cfg = ConfigLoader(config_dir=tmp_path)
    cfg.load()
    assert cfg.auth.enabled is True
    assert cfg.auth.principals_by_hash[hash_key("llmr_live")].id == "carol"


def test_loader_auth_disabled_by_default(tmp_path, monkeypatch):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    cfg = ConfigLoader(config_dir=tmp_path)
    cfg.load()
    assert cfg.auth.enabled is False


def test_loader_auth_attr_exists_before_load(tmp_path):
    """config.auth must exist even before load() (create_app installs middleware
    that reads it; a missing attr would be an AttributeError at request time)."""
    cfg = ConfigLoader(config_dir=tmp_path)
    assert cfg.auth.enabled is False


# --- Task 4: HTTP auth middleware ------------------------------------------

def _app_with_auth(tmp_path, monkeypatch):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    (tmp_path / "api_keys.yaml").write_text(
        "keys:\n  " + hash_key("llmr_live") + ":\n    id: carol\n"
    )
    monkeypatch.setenv("LLM_RELAY_AUTH", "1")
    return create_app(config_dir=tmp_path)


def test_exempt_health_no_key(tmp_path, monkeypatch):
    client = TestClient(_app_with_auth(tmp_path, monkeypatch))
    assert client.get("/health").status_code == 200


def test_protected_path_requires_key(tmp_path, monkeypatch):
    client = TestClient(_app_with_auth(tmp_path, monkeypatch))
    r = client.get("/v1/available-models")
    assert r.status_code == 401
    assert "Bearer" in r.headers.get("www-authenticate", "")


def test_protected_path_accepts_valid_key(tmp_path, monkeypatch):
    client = TestClient(_app_with_auth(tmp_path, monkeypatch))
    r = client.get("/v1/available-models", headers={"Authorization": "Bearer llmr_live"})
    assert r.status_code == 200


def test_auth_disabled_allows_all(tmp_path, monkeypatch):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    client = TestClient(create_app(config_dir=tmp_path))
    assert client.get("/v1/available-models").status_code == 200


# --- Task 5: minimal /health when auth enabled (no topology leak) ----------

def test_health_minimal_when_auth_enabled(tmp_path, monkeypatch):
    client = TestClient(_app_with_auth(tmp_path, monkeypatch))
    assert client.get("/health").json() == {"status": "ok"}


def test_health_full_when_auth_disabled(tmp_path, monkeypatch):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    client = TestClient(create_app(config_dir=tmp_path))
    assert "endpoints" in client.get("/health").json()


# --- Task 6: keys CLI (add / list / revoke) --------------------------------

def test_keys_cli_add_then_revoke(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_RELAY_CONFIG_DIR", str(tmp_path))
    from llm_relay.cli import main

    monkeypatch.setattr(sys, "argv", ["llm-relay", "keys", "add", "frank", "--priority", "2"])
    assert main() == 0
    principals = load_keys(tmp_path / "api_keys.yaml")
    assert any(p.id == "frank" and p.priority_weight == 2.0 for p in principals.values())

    monkeypatch.setattr(sys, "argv", ["llm-relay", "keys", "list"])
    assert main() == 0

    monkeypatch.setattr(sys, "argv", ["llm-relay", "keys", "revoke", "frank"])
    assert main() == 0
    assert load_keys(tmp_path / "api_keys.yaml") == {}


def test_insecure_bind_warning():
    from llm_relay.cli import _insecure_bind_warning

    # Loopback or auth-on: no warning.
    assert _insecure_bind_warning("127.0.0.1", auth_enabled=False) is None
    assert _insecure_bind_warning("localhost", auth_enabled=False) is None
    assert _insecure_bind_warning("0.0.0.0", auth_enabled=True) is None
    # Routable bind with auth off: warn.
    w = _insecure_bind_warning("0.0.0.0", auth_enabled=False)
    assert w is not None and "auth" in w.lower()
