"""Tests for the degeneracy detector and default-sampling protection."""
from llm_relay.degeneracy import (
    compression_ratio,
    degeneracy_score,
    detect_cycle,
    is_degenerate,
)
from llm_relay.routing.router import (
    DEFAULT_REPETITION_PENALTY,
    _apply_repetition_penalty_default,
    _clamp_max_tokens,
    DEFAULT_NON_STREAM_MAX_TOKENS,
)


# --- degeneracy detector ---

def test_normal_text_not_degenerate():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    assert not is_degenerate(text)
    assert degeneracy_score(text) < 0.5


def test_code_not_degenerate():
    """Legitimate repetitive code (enum list) should NOT trigger."""
    text = "\n".join(f"    FOO_{i} = {i}," for i in range(100))
    assert not is_degenerate(text), "enum list falsely flagged as degenerate"


def test_repetition_loop_detected():
    """The exact failure mode: 'int64 vs int64' repeated forever."""
    text = "int64 vs int64 " * 200  # ~3200 chars, period ~15
    assert is_degenerate(text)
    cycle = detect_cycle(text)
    assert cycle is not None, "cycle not detected in repetition loop"


def test_short_repetition_not_flagged():
    """A few repetitions is normal — need >=20 reps of the cycle."""
    text = "int64 vs int64 " * 5  # 5 reps — too few
    assert not is_degenerate(text)


def test_compression_ratio_normal():
    """Varied text should have a moderate compression ratio."""
    text = (
        "The quick brown fox jumps over the lazy dog.\n"
        "A journey of a thousand miles begins with a single step.\n"
        "To be or not to be, that is the question.\n"
        "All that glitters is not gold.\n"
        "The only thing we have to fear is fear itself.\n"
        "Ask not what your country can do for you.\n"
        "I think therefore I am.\n"
        "The unexamined life is not worth living.\n"
    ) * 3
    cr = compression_ratio(text)
    assert 0.1 < cr < 1.0, f"expected moderate compression, got {cr}"


def test_compression_ratio_degenerate():
    text = "a" * 4000
    cr = compression_ratio(text)
    assert cr < 0.1  # highly compressible


def test_empty_text_safe():
    assert not is_degenerate("")
    assert not is_degenerate("hi")
    assert degeneracy_score("") == 0.0


def test_long_normal_text_not_degenerate():
    """A long, varied response should not be flagged."""
    text = (
        "Here's an analysis of the CVE patch:\n\n"
        "1. The upstream commit modifies src/sg_inq.c to sanitize output.\n"
        "2. The candidate patch covers the same changes.\n"
        "3. The security property is: udev-conforming character encoding.\n"
        "4. The fix prevents control character injection in SCSI name strings.\n"
        "5. Verdict: PASS — the patch is complete and faithful.\n"
    )
    assert not is_degenerate(text)


# --- golden fixture: the exact prompt that triggered the original bug ---

def test_diff_marker_prompt_is_not_degenerate_with_penalty():
    """The diff-marker prompt '+const int64 x = 1;' triggered a repetition
    loop with rep_pen=1.0. With rep_pen=1.1 (our default), the model
    should produce normal output. This test doesn't call the model — it
    verifies the relay APPLIES the default so the model never sees 1.0."""
    # Simulate what the relay does: client sends no repetition_penalty,
    # relay applies 1.1 before forwarding.
    client_request = {"model": "qwen3.6-35b", "messages": [{"role": "user", "content": "+const int64 x = 1;"}]}
    forwarded = _apply_repetition_penalty_default(client_request, default=DEFAULT_REPETITION_PENALTY)
    assert forwarded["repetition_penalty"] == 1.1, \
        "relay must apply repetition_penalty=1.1 when client sets none"


def test_simulated_repetition_loop_output_is_degenerate():
    """Simulate what the model produced WITHOUT the penalty: 1300+ chunks
    of 'int64 vs int64'. Verify the detector catches it."""
    loop_output = "Comparing int64 vs int64: int64 is the same as int64. " * 100
    assert is_degenerate(loop_output), \
        "degeneracy detector must catch the 'int64 vs int64' repetition loop"


def test_simulated_normal_output_not_degenerate():
    """Simulate what the model produced WITH the penalty: a clean explanation."""
    normal_output = (
        "That line appears to declare a constant 64-bit integer, but its "
        "validity depends on the programming language and context. The `+` "
        "prefix suggests this is a line from a unified diff. In C, `const "
        "int64_t x = 1;` declares a read-only 64-bit signed integer."
    )
    assert not is_degenerate(normal_output)


# --- default_max_tokens + repetition_penalty combined ---

def test_both_defaults_applied_together():
    """Non-streaming request with neither max_tokens nor repetition_penalty
    should get both defaults applied."""
    req = {"model": "m", "messages": []}
    # Apply max_tokens default
    with_mt = _clamp_max_tokens(req, prompt_est=1000, window=262144,
                               default=DEFAULT_NON_STREAM_MAX_TOKENS)
    assert with_mt["max_tokens"] == DEFAULT_NON_STREAM_MAX_TOKENS
    # Apply repetition_penalty default
    with_both = _apply_repetition_penalty_default(with_mt, default=DEFAULT_REPETITION_PENALTY)
    assert with_both["repetition_penalty"] == DEFAULT_REPETITION_PENALTY
    assert with_both["max_tokens"] == DEFAULT_NON_STREAM_MAX_TOKENS


def test_client_values_never_overridden():
    """Client-set values for max_tokens AND repetition_penalty are never
    overridden by defaults."""
    req = {"model": "m", "messages": [], "max_tokens": 500, "repetition_penalty": 1.3}
    with_mt = _clamp_max_tokens(req, prompt_est=1000, window=262144,
                               default=DEFAULT_NON_STREAM_MAX_TOKENS)
    assert with_mt is req  # unchanged — client ceiling respected
    with_rp = _apply_repetition_penalty_default(req, default=DEFAULT_REPETITION_PENALTY)
    assert with_rp is req  # unchanged — client penalty respected
    assert with_rp["repetition_penalty"] == 1.3
