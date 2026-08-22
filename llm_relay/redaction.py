"""Strip credential-shaped substrings before prompt text is stored.

Runs BEFORE hashing, so the stored bytes, the content-address, and the search
index all describe the redacted text.

Deliberately lossy and deliberately conservative in opposite directions: the
patterns target shapes that are almost never legitimate prose or code, because
a false positive costs one unreadable substring while a false negative costs a
live credential sitting in an indefinitely-retained store.

The asymmetry has one hard limit. ``test_ordinary_code_is_left_alone`` and
``test_prose_is_left_alone`` exist to keep the store analytically useful: if a
pattern starts matching plain source or plain English, tighten the pattern
rather than the test.
"""
from __future__ import annotations

import re

_PLACEHOLDER = "[REDACTED:{}]"

PATTERNS = (
    # Whole PEM/OpenSSH private-key blocks, body included.
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL)),
    # JWTs: three base64url segments separated by dots, starting with a header.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    # Relay-issued keys.
    ("relay_key", re.compile(r"\bllmr_[A-Za-z0-9_-]{16,}\b")),
    # Provider keys: OpenAI-style and GitHub's prefixed tokens.
    ("provider_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    # AWS access key ids.
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # Bearer/authorization header values.
    ("bearer", re.compile(r"(?i)\b(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    # KEY=value / SECRET: value assignments. The name must itself look secret,
    # so ordinary assignments (total = a + b) are untouched.
    ("assignment", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY)[A-Z0-9_]*)"
        r"(\s*[:=]\s*)"
        r"(\"[^\"\n]{6,}\"|'[^'\n]{6,}'|[^\s\"'#;,)]{6,})")),
)


def redact(text: str) -> tuple[str, bool]:
    """Return ``(clean_text, did_redact)``.

    Never raises: unparseable input yields ``("", False)`` rather than an error
    in the request path.
    """
    if not text or not isinstance(text, str):
        return "", False
    clean = text
    hit = False
    for name, pattern in PATTERNS:
        if name == "assignment":
            # Keep the name and the separator so the reader can still see WHAT
            # was set; only the value goes.
            def _sub(m, _n=name):
                return f"{m.group(1)}{m.group(2)}{_PLACEHOLDER.format(_n)}"

            clean, n = pattern.subn(_sub, clean)
        else:
            clean, n = pattern.subn(_PLACEHOLDER.format(name), clean)
        if n:
            hit = True
    return clean, hit
