"""Secrets must not survive into the prompt store.

Retention is indefinite, so this pass is the primary control keeping a prompt
archive from becoming a credential archive. It is deliberately lossy: a false
positive costs one unreadable substring, a false negative costs a live secret.
"""
from __future__ import annotations

from llm_relay.redaction import redact


def test_bearer_token_is_redacted():
    clean, hit = redact("curl -H 'Authorization: Bearer sk-abc123def456ghi789jkl' url")
    assert hit is True
    assert "sk-abc123def456ghi789jkl" not in clean
    assert "REDACTED" in clean


def test_relay_key_is_redacted():
    clean, hit = redact("my key is llmr_9f8e7d6c5b4a3928170654321fedcba0")
    assert hit is True
    assert "llmr_9f8e7d6c5b4a3928170654321fedcba0" not in clean


def test_github_token_is_redacted():
    clean, hit = redact("GH_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz")
    assert hit is True
    assert "ghp_1234567890abcdefghijklmnopqrstuvwxyz" not in clean


def test_aws_access_key_is_redacted():
    clean, hit = redact("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
    assert hit is True
    assert "AKIAIOSFODNN7EXAMPLE" not in clean


def test_private_key_block_is_redacted_entirely():
    text = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n"
        "AAAAMwAAAAtzc2gtZW\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    clean, hit = redact(text)
    assert hit is True
    assert "b3BlbnNzaC1rZXktdjEA" not in clean


def test_env_assignment_secrets_are_redacted():
    clean, hit = redact("DATABASE_PASSWORD=hunter2supersecret\nAPI_SECRET=abcd1234efgh5678")
    assert hit is True
    assert "hunter2supersecret" not in clean
    assert "abcd1234efgh5678" not in clean


def test_jwt_is_redacted():
    jwt = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
           "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
           "dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk")
    clean, hit = redact(f"token: {jwt}")
    assert hit is True
    assert jwt not in clean


def test_ordinary_code_is_left_alone():
    # A false positive here costs real analytical value, so the common case of
    # plain source code must pass through untouched.
    code = (
        "def add(a, b):\n"
        "    total = a + b  # sum them\n"
        "    return total\n"
        "result = add(2, 3)\n"
    )
    clean, hit = redact(code)
    assert hit is False
    assert clean == code


def test_prose_is_left_alone():
    text = "Can you explain why the deploy failed on llm-gateway-01 last night?"
    clean, hit = redact(text)
    assert hit is False
    assert clean == text


def test_empty_and_none_safe():
    assert redact("") == ("", False)
    assert redact(None) == ("", False)


def test_redaction_is_idempotent():
    once, _ = redact("Authorization: Bearer sk-abc123def456ghi789jkl")
    twice, hit = redact(once)
    assert twice == once
    assert hit is False
