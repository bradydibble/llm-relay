"""Auth + identity: principal model, key store, and resolution."""
from __future__ import annotations

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


def test_revoke_removes_id(tmp_path):
    path = tmp_path / "api_keys.yaml"
    pt, pr = mint_key("erin")
    write_keys(path, {hash_key(pt): pr})
    assert revoke_id(path, "erin") == 1
    assert load_keys(path) == {}
