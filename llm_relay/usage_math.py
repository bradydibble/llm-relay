"""How many tokens did this request use, and how well do we know?

Pure functions, no I/O. Separated from metrics/instrumentation so the
arithmetic that money and capacity decisions rest on can be tested directly.

Naming follows the Claude/OpenAI convention:
``output_tokens`` INCLUDES ``reasoning_tokens``. Total = input + output.
Reasoning is an of-which subset and is never added a second time.
"""
from __future__ import annotations

from dataclasses import dataclass

# How we learned the token counts, best first.
SOURCE_INCREMENTAL = "upstream_incremental"  # usage on every chunk; exact even if aborted
SOURCE_FINAL = "upstream_final"              # usage from the terminal chunk / body; exact
SOURCE_FRAMES = "frame_count"                # aborted stream; one SSE frame ~= one token
SOURCE_ESTIMATE = "tokenizer_estimate"       # aborted before any frame; tokenized text
SOURCE_NONE = "none"                         # no tokens were consumed

# How we learned the reasoning/content split.
REASON_DETAILS = "upstream_details"  # backend reported reasoning_tokens
REASON_SPLIT = "char_split"          # proportional split of an exact output total
REASON_NONE = "none"                 # no reasoning in this response


@dataclass(frozen=True)
class UsageCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    usage_source: str = SOURCE_NONE
    reasoning_source: str = REASON_NONE


def _int(value) -> int:
    """Coerce an upstream-supplied number to a non-negative int. Upstream JSON
    is not trusted: absent, null, negative, and non-numeric all mean zero."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def estimate_tokens(text: str) -> int:
    """Approximate token count for text we have but have no count for.

    Uses tiktoken (already a relay dependency) rather than a characters/4
    heuristic. It is a foreign BPE, not the served model's tokenizer, so this
    is still an estimate — callers flag it as SOURCE_ESTIMATE.
    """
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # tiktoken unavailable or its data files unreachable: fall back to the
        # crude ratio rather than losing the request entirely.
        return max(1, len(text) // 4)


def _merged_usage(usage: dict | None, response_body: dict | None) -> dict:
    """Token usage lives in the streaming-reassembled ``usage`` dict OR, for
    non-streaming responses, in ``response_body['usage']``. Prefer the former."""
    if usage:
        return usage
    if isinstance(response_body, dict):
        u = response_body.get("usage")
        if isinstance(u, dict):
            return u
    return {}


def _split_reasoning(output_tokens: int, eff: dict,
                     content_text: str, reasoning_text: str) -> tuple[int, str]:
    """Reasoning tokens as an of-which subset of an already-known output total."""
    details = eff.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        reported = _int(details.get("reasoning_tokens"))
        if reported:
            # Clamp: a bad upstream number must not break the subset invariant.
            return min(reported, output_tokens), REASON_DETAILS

    if not reasoning_text or output_tokens <= 0:
        return 0, REASON_NONE

    # Proportional split of an EXACT total: only the split is approximate, so
    # totals and cost stay correct.
    r_chars = len(reasoning_text)
    total_chars = r_chars + len(content_text)
    if total_chars <= 0:
        return 0, REASON_NONE
    share = int(round(output_tokens * (r_chars / total_chars)))
    return min(max(share, 0), output_tokens), REASON_SPLIT


def resolve_usage(*, usage: dict | None, response_body: dict | None,
                  streamed: bool, frame_count: int = 0,
                  content_text: str = "", reasoning_text: str = "",
                  saw_incremental: bool = False) -> UsageCounts:
    """Resolve token counts and record how they were obtained.

    ``saw_incremental`` is True when the upstream sent usage on intermediate
    chunks (vLLM ``continuous_usage_stats``), which makes the numbers exact
    even for a stream the client aborted.
    """
    eff = _merged_usage(usage, response_body)
    input_tokens = _int(eff.get("prompt_tokens"))
    output_tokens = _int(eff.get("completion_tokens"))

    if output_tokens or input_tokens:
        source = SOURCE_INCREMENTAL if (streamed and saw_incremental) else SOURCE_FINAL
        # An aborted stream can carry prompt_tokens (llama.cpp timings) with no
        # completion count. Recover output from frames/text instead of zero.
        if not output_tokens and streamed:
            if frame_count > 0:
                output_tokens, source = frame_count, SOURCE_FRAMES
            elif content_text or reasoning_text:
                output_tokens = estimate_tokens(content_text) + estimate_tokens(reasoning_text)
                source = SOURCE_ESTIMATE
    elif streamed and frame_count > 0:
        output_tokens, source = frame_count, SOURCE_FRAMES
    elif content_text or reasoning_text:
        output_tokens = estimate_tokens(content_text) + estimate_tokens(reasoning_text)
        source = SOURCE_ESTIMATE
    else:
        return UsageCounts()  # nothing consumed; SOURCE_NONE / REASON_NONE

    reasoning_tokens, reasoning_source = _split_reasoning(
        output_tokens, eff, content_text, reasoning_text
    )

    cache_read = 0
    if isinstance(response_body, dict):
        timings = response_body.get("timings")
        if isinstance(timings, dict):
            cache_read = _int(timings.get("cache_n"))

    return UsageCounts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read,
        usage_source=source,
        reasoning_source=reasoning_source,
    )
