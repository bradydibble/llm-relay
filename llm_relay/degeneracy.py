"""Degeneracy detector: identifies repetition loops in model output.

Used in three places:
1. In-band on the relay's streaming path (every ~64 tokens, abort on trip)
2. Health probe (L2/L3 — test a backend for repetition-loop susceptibility)
3. Test suite (golden fixture corpus — assert known-bad prompts don't loop)

The detector uses two signals, both cheap and language-agnostic:
  - Compression ratio: zlib over the last ~4KB. Degenerate loops compress
    below ~0.1 (the same bytes over and over). Normal text is 0.3-0.7.
  - Exact cycle detection: for period p in [1,128], test if the last p bytes
    repeat the p bytes before them. Requires the cycle to persist >=256
    tokens to avoid false positives on legitimate repetitive code (enum
    lists, generated bindings).

A degeneracy score of 0.0 = normal, 1.0 = definite loop. The threshold
for action is 0.5 (either signal fires).
"""
from __future__ import annotations

import zlib


def compression_ratio(text: str) -> float:
    """zlib compressed size / raw size. Low = repetitive."""
    if len(text) < 64:
        return 1.0
    raw = text[-4096:].encode("utf-8", "replace")
    compressed = zlib.compress(raw, 6)
    return len(compressed) / len(raw) if raw else 1.0


def detect_cycle(text: str, min_period: int = 1, max_period: int = 128,
                 min_repeats: int = 20) -> int | None:
    """Detect an exact repeating cycle in the tail of text.

    Returns the period (in chars) if a cycle of length p persists for at
    least min_repeats repetitions, else None. A cycle means:
      text[-p:] == text[-2p:-p] == text[-3p:-2p] == ...

    This catches "int64 vs int64 vs int64 ..." (period ~15 chars, 200+ reps)
    and "the the the the ..." (period ~4 chars, 100+ reps) while ignoring
    legitimate repetitive text like "The quick brown fox" x10 (only 10 reps).
    min_repeats=20 means the cycle must repeat 20+ times — a real loop does
    hundreds; legitimate repetition rarely exceeds 10.
    """
    if len(text) < min_period * min_repeats:
        return None
    for p in range(min_period, min(max_period, len(text) // 2) + 1):
        tail = text[-p:]
        # Check if the last min_repeats repetitions of tail match
        repeats = len(text) // p
        if repeats < min_repeats:
            continue
        # Verify: the last (repeats * p) chars should all be tail repeated
        expected = tail * repeats
        actual = text[-(repeats * p):]
        if actual == expected and repeats >= min_repeats:
            return p
    return None


def degeneracy_score(text: str) -> float:
    """Combined degeneracy score: 0.0 = normal, 1.0 = definite loop.

    Fires on either signal:
    - compression_ratio < 0.1 (highly repetitive)
    - exact cycle detected with period <= 128
    """
    if len(text) < 64:
        return 0.0
    cr = compression_ratio(text)
    cycle = detect_cycle(text)
    if cycle is not None:
        return 1.0
    if cr < 0.1:
        return 0.7
    if cr < 0.2:
        return 0.3
    return 0.0


def is_degenerate(text: str, threshold: float = 0.5) -> bool:
    """True if the text is likely a repetition loop."""
    return degeneracy_score(text) >= threshold
