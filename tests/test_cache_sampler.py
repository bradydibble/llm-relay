"""Fleet-level prefix-cache reuse, sampled from each backend's ``/metrics``.

This is the AGGREGATE lane, deliberately per (day, model) and nothing finer.
Per-request attribution is a different mechanism entirely (vLLM's
``usage.prompt_tokens_details.cached_tokens`` behind ``--enable-prompt-tokens-details``,
already read by ``usage_math``); these numbers are not expected to agree with
it and are never reconciled against it. See the module docstring.

These tests pin the three things that make a cumulative counter safe to
account from: reset detection (a restarted backend must not emit a negative
delta), idempotence (a re-run must not double-count), and the difference
between "not reported" and "no reuse" -- conflating those two is the bug being
fixed, so a genuine zero and an absent metric are asserted separately.

No test touches the network: the fleet is represented by fixture metrics text
and an injected fetcher.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_relay.cache_sampler import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    OUTCOME_BASELINE,
    OUTCOME_COUNTED,
    OUTCOME_NOT_REPORTED,
    OUTCOME_REJECTED,
    OUTCOME_RESET,
    OUTCOME_UNCHANGED,
    Backend,
    CacheCounters,
    backends_from_config,
    cache_by_model,
    cache_rollup,
    counter_delta,
    daily_id,
    ensure_schema,
    metrics_url,
    open_cache_db,
    parse_prefix_cache_metrics,
    priced_input_tokens,
    record_sample,
    sample_backends,
)
from llm_relay.config.types import ModelConfig, Ownership, ProviderConfig, ProviderType
from llm_relay.usage_store import open_db

# A real vLLM exposition, including the scientific notation it actually emits
# and the HELP/TYPE comment lines a naive line split would mistake for data.
VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen3.6-35b"} 1.0
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3.6-35b"} 2.05165845e+08
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in number of hit tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3.6-35b"} 1.89851904e+08
"""

# llama.cpp: reports per-request (usage.prompt_tokens_details.cached_tokens and
# timings.cache_n, both already read by usage_math), so it exposes no
# vllm:prefix_cache_* series at all. Not the same fact as "no reuse".
LLAMACPP_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12345
"""

ZERO_REUSE_METRICS = """\
vllm:prefix_cache_queries_total{engine="0",model_name="trinity-large-thinking"} 4096.0
vllm:prefix_cache_hits_total{engine="0",model_name="trinity-large-thinking"} 0.0
"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parses_scientific_notation_and_keys_on_model_name():
    reading = parse_prefix_cache_metrics(VLLM_METRICS)
    assert reading.reported is True
    assert set(reading.by_model) == {"qwen3.6-35b"}
    counters = reading.by_model["qwen3.6-35b"]
    assert counters.queried == 205165845
    assert counters.hits == 189851904


def test_multiple_models_on_one_backend_are_kept_apart():
    text = (
        'vllm:prefix_cache_queries_total{model_name="a"} 100\n'
        'vllm:prefix_cache_hits_total{model_name="a"} 40\n'
        'vllm:prefix_cache_queries_total{model_name="b"} 10\n'
        'vllm:prefix_cache_hits_total{model_name="b"} 9\n'
    )
    reading = parse_prefix_cache_metrics(text)
    assert reading.by_model["a"] == CacheCounters(100, 40)
    assert reading.by_model["b"] == CacheCounters(10, 9)


def test_backend_with_no_prefix_cache_series_is_not_reported():
    reading = parse_prefix_cache_metrics(LLAMACPP_METRICS)
    assert reading.reported is False
    assert reading.by_model == {}


def test_a_genuine_zero_is_reported_and_distinct_from_not_reported():
    reading = parse_prefix_cache_metrics(ZERO_REUSE_METRICS)
    assert reading.reported is True  # the backend answered: zero reuse
    assert reading.by_model["trinity-large-thinking"] == CacheCounters(4096, 0)


def test_unparseable_values_are_ignored_not_guessed():
    text = (
        'vllm:prefix_cache_queries_total{model_name="a"} NaN\n'
        'vllm:prefix_cache_hits_total{model_name="a"} +Inf\n'
    )
    reading = parse_prefix_cache_metrics(text)
    assert reading.by_model == {}


def test_metrics_url_hangs_off_the_backend_root_not_the_v1_prefix():
    assert metrics_url("http://10.0.0.5:8000") == "http://10.0.0.5:8000/metrics"
    assert metrics_url("http://10.0.0.5:8000/") == "http://10.0.0.5:8000/metrics"


# --------------------------------------------------------------------------
# Reset detection -- the whole reason a cursor exists
# --------------------------------------------------------------------------

def test_counter_delta_subtracts_the_previous_sample():
    delta, was_reset = counter_delta(CacheCounters(1000, 400), CacheCounters(1500, 600))
    assert was_reset is False
    assert delta == CacheCounters(500, 200)


def test_counter_delta_treats_a_lower_value_as_a_backend_restart():
    # The counters count from process start. After a restart the current value
    # is LOWER than the last sample; a naive subtraction would emit a negative.
    delta, was_reset = counter_delta(CacheCounters(1000, 400), CacheCounters(200, 90))
    assert was_reset is True
    assert delta == CacheCounters(200, 90)  # tokens since the restart


def test_counter_delta_detects_a_reset_visible_only_in_hits():
    delta, was_reset = counter_delta(CacheCounters(1000, 400), CacheCounters(1200, 90))
    assert was_reset is True
    assert delta == CacheCounters(1200, 90)


def test_first_sight_seeds_a_baseline_and_attributes_nothing():
    delta, was_reset = counter_delta(None, CacheCounters(205165845, 189851904))
    assert was_reset is False
    assert delta == CacheCounters(0, 0)


def test_backend_restart_never_writes_a_negative_daily_delta(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    kw = {"day": "2026-08-21", "backend": "gb200:8000", "model": "glm-5.2-nvfp4"}
    assert record_sample(conn, current=CacheCounters(1000, 400), **kw) == OUTCOME_BASELINE
    assert record_sample(conn, current=CacheCounters(1500, 600), **kw) == OUTCOME_COUNTED
    assert record_sample(conn, current=CacheCounters(200, 90), **kw) == OUTCOME_RESET
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert len(rows) == 1
    assert rows[0]["queried_tokens"] == 500 + 200      # never 500 + (200 - 1500)
    assert rows[0]["cache_read_tokens"] == 200 + 90
    assert rows[0]["queried_tokens"] > 0
    assert rows[0]["resets"] == 1


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------

def test_daily_id_is_deterministic_and_keyed_on_day_and_model():
    a = daily_id("2026-08-21", "glm-5.2-nvfp4")
    assert a == daily_id("2026-08-21", "glm-5.2-nvfp4")
    assert a != daily_id("2026-08-22", "glm-5.2-nvfp4")
    assert a != daily_id("2026-08-21", "qwen3.6-35b")


def test_resampling_the_same_counters_adds_nothing(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    kw = {"day": "2026-08-21", "backend": "gb200:8000", "model": "glm-5.2-nvfp4"}
    record_sample(conn, current=CacheCounters(1000, 400), **kw)
    record_sample(conn, current=CacheCounters(1500, 600), **kw)
    assert record_sample(conn, current=CacheCounters(1500, 600), **kw) == OUTCOME_UNCHANGED
    assert record_sample(conn, current=CacheCounters(1500, 600), **kw) == OUTCOME_UNCHANGED
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert rows[0]["queried_tokens"] == 500
    assert rows[0]["cache_read_tokens"] == 200


def test_two_backends_serving_one_model_sum_into_one_daily_row(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    for backend in ("llama-01:18402", "llama-01:18412"):
        kw = {"day": "2026-08-21", "backend": backend, "model": "glimmer-vllm"}
        record_sample(conn, current=CacheCounters(100, 90), **kw)
        record_sample(conn, current=CacheCounters(200, 180), **kw)
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert len(rows) == 1                       # keyed by (day, model)
    assert rows[0]["queried_tokens"] == 200     # 100 from each backend's delta
    assert rows[0]["cache_read_tokens"] == 180


def test_hits_exceeding_queries_is_rejected_not_clamped(tmp_path):
    # Untrusted upstream numbers: a delta with more hits than queries would
    # violate the CHECK, and INSERT OR IGNORE hides EVERY constraint failure --
    # so validate first and count the rejection instead of losing it silently.
    conn = open_cache_db(str(tmp_path / "u.db"))
    kw = {"day": "2026-08-21", "backend": "b", "model": "m"}
    record_sample(conn, current=CacheCounters(100, 50), **kw)
    assert record_sample(conn, current=CacheCounters(110, 200), **kw) == OUTCOME_REJECTED
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert rows[0]["queried_tokens"] == 0    # the rejection added nothing
    assert rows[0]["cache_read_tokens"] == 0
    # ...and the cursor did not advance, so the next legal reading is still
    # measured against the last trusted sample.
    assert record_sample(conn, current=CacheCounters(150, 70), **kw) == OUTCOME_COUNTED
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert rows[0]["queried_tokens"] == 50
    assert rows[0]["cache_read_tokens"] == 20


# --------------------------------------------------------------------------
# Schema is additive: a live database upgrades in place
# --------------------------------------------------------------------------

def test_schema_creation_leaves_an_existing_usage_database_intact(tmp_path):
    path = str(tmp_path / "live.db")
    conn = open_db(path)
    conn.execute(
        "INSERT INTO requests (request_id, ts, day, principal, client, model, "
        "provider, outcome, streamed, input_tokens, output_tokens, "
        "reasoning_tokens, usage_source, reasoning_source) VALUES "
        "('live-1', 1.0, '2026-08-20', 'brady', 'cc', 'glm-5.2', 'gb200', "
        "'success', 1, 99, 9, 0, 'upstream_final', 'none')"
    )
    conn.close()

    conn = open_cache_db(path)
    ensure_schema(conn)  # additive: safe to run again
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    assert conn.execute(
        "SELECT input_tokens FROM requests WHERE request_id = 'live-1'"
    ).fetchone()[0] == 99
    record_sample(conn, day="2026-08-21", backend="b", model="m",
                  current=CacheCounters(10, 5))
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


# --------------------------------------------------------------------------
# Backend discovery from the relay's own config
# --------------------------------------------------------------------------

def _fake_config():
    providers = {
        "llama-01": ProviderConfig(
            type=ProviderType.openai, base_url="http://192.168.1.76/",
            ownership=Ownership.ciq_owned, discover_ports=[18404, 18400],
        ),
        "gb200": ProviderConfig(
            type=ProviderType.openai, base_url="http://10.128.0.5:8000",
            ownership=Ownership.third_party,
        ),
        "retired": ProviderConfig(
            type=ProviderType.openai, base_url="http://10.9.9.9",
            ownership=Ownership.ciq_owned, enabled=False,
        ),
    }
    models = {
        "glimmer-vllm": ModelConfig(provider="llama-01", port=18402),
        "qwen3.8-27b": ModelConfig(provider="llama-01", port=18401),
        "glm-5.2-nvfp4": ModelConfig(provider="gb200"),
        "gone": ModelConfig(provider="retired", port=9999),
    }
    return SimpleNamespace(
        providers=providers,
        models=SimpleNamespace(models=models, aliases={}),
    )


def test_backends_come_from_provider_config_not_hardcoded_addresses():
    backends = backends_from_config(_fake_config())
    urls = {b.base_url for b in backends}
    assert "http://192.168.1.76:18402" in urls   # provider base_url + model port
    assert "http://192.168.1.76:18401" in urls
    assert "http://10.128.0.5:8000" in urls      # no port on the model entry
    assert "http://192.168.1.76:18404" in urls   # a bare discover_port
    assert "http://192.168.1.76:18400" in urls
    assert not any("10.9.9.9" in u for u in urls)  # disabled provider skipped
    by_url = {b.base_url: b for b in backends}
    assert by_url["http://192.168.1.76:18402"].models == ("glimmer-vllm",)
    assert by_url["http://192.168.1.76:18404"].models == ()


# --------------------------------------------------------------------------
# Sampling many backends: one bad backend must not lose the others
# --------------------------------------------------------------------------

def _fetcher(mapping):
    def fetch(base_url, *, timeout=5.0):
        value = mapping[base_url]
        if isinstance(value, Exception):
            raise value
        return value
    return fetch


def test_one_unreachable_backend_does_not_abort_the_sample(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    backends = [
        Backend(key="a", provider="p", base_url="http://a", models=("qwen3.6-35b",)),
        Backend(key="dead", provider="p", base_url="http://dead", models=("x",)),
    ]
    fetch = _fetcher({
        "http://a": VLLM_METRICS,
        "http://dead": OSError("connection refused"),
    })
    first = sample_backends(conn, backends, day="2026-08-21", fetcher=fetch)
    assert first.unreachable == ["dead"]
    assert first.baselined == 1
    second = sample_backends(conn, backends, day="2026-08-21", fetcher=fetch)
    assert second.unreachable == ["dead"]
    assert second.counted == 0  # counters had not moved, but 'a' was still read


def test_a_not_reporting_backend_is_recorded_distinctly_from_zero_reuse(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    backends = [
        Backend(key="cpp", provider="p", base_url="http://cpp",
                models=("glimmer-llamacpp",)),
        Backend(key="tlt", provider="p", base_url="http://tlt",
                models=("trinity-large-thinking",)),
    ]
    fetch = _fetcher({
        "http://cpp": LLAMACPP_METRICS,
        "http://tlt": ZERO_REUSE_METRICS,
    })
    result = sample_backends(conn, backends, day="2026-08-21", fetcher=fetch)
    assert result.not_reported == ["cpp"]
    rows = {r["model"]: r for r in cache_rollup(conn, "2026-08-21", "2026-08-21")}
    assert rows["glimmer-llamacpp"]["reported"] is False
    assert rows["glimmer-llamacpp"]["hit_rate"] is None       # unknown, not 0%
    assert rows["trinity-large-thinking"]["reported"] is True
    assert rows["trinity-large-thinking"]["hit_rate"] is None  # nothing queried yet


def test_a_backend_that_starts_reporting_promotes_the_not_reported_row(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    backends = [Backend(key="b", provider="p", base_url="http://b", models=("m",))]
    sample_backends(conn, backends, day="2026-08-21",
                    fetcher=_fetcher({"http://b": LLAMACPP_METRICS}))
    assert cache_rollup(conn, "2026-08-21", "2026-08-21")[0]["reported"] is False
    on = ('vllm:prefix_cache_queries_total{model_name="m"} 100\n'
          'vllm:prefix_cache_hits_total{model_name="m"} 60\n')
    sample_backends(conn, backends, day="2026-08-21",
                    fetcher=_fetcher({"http://b": on}))
    assert cache_rollup(conn, "2026-08-21", "2026-08-21")[0]["reported"] is True


def test_unlabelled_metrics_fall_back_to_the_single_configured_model(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    backends = [Backend(key="b", provider="p", base_url="http://b",
                        models=("only-model",))]
    text = ("vllm:prefix_cache_queries_total 100\n"
            "vllm:prefix_cache_hits_total 60\n")
    fetch = _fetcher({"http://b": text})
    sample_backends(conn, backends, day="2026-08-21", fetcher=fetch)
    text2 = ("vllm:prefix_cache_queries_total 300\n"
             "vllm:prefix_cache_hits_total 160\n")
    sample_backends(conn, backends, day="2026-08-21",
                    fetcher=_fetcher({"http://b": text2}))
    rows = cache_rollup(conn, "2026-08-21", "2026-08-21")
    assert rows[0]["model"] == "only-model"
    assert rows[0]["queried_tokens"] == 200


# --------------------------------------------------------------------------
# Read side: what the portal will price from
# --------------------------------------------------------------------------

def test_rollup_reports_a_hit_rate_per_model(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    kw = {"backend": "b", "model": "qwen3.6-35b"}
    record_sample(conn, day="2026-08-20", current=CacheCounters(0, 0), **kw)
    record_sample(conn, day="2026-08-20", current=CacheCounters(1000, 900), **kw)
    record_sample(conn, day="2026-08-21", current=CacheCounters(3000, 2700), **kw)
    rows = cache_rollup(conn, "2026-08-20", "2026-08-21")
    assert [r["day"] for r in rows] == ["2026-08-20", "2026-08-21"]
    assert rows[0]["hit_rate"] == pytest.approx(0.9)
    assert rows[1]["queried_tokens"] == 2000
    assert rows[1]["hit_rate"] == pytest.approx(0.9)

    totals = cache_by_model(conn, "2026-08-20", "2026-08-21")
    assert len(totals) == 1
    assert totals[0]["model"] == "qwen3.6-35b"
    assert totals[0]["queried_tokens"] == 3000
    assert totals[0]["cache_read_tokens"] == 2700
    assert totals[0]["hit_rate"] == pytest.approx(0.9)
    assert totals[0]["days"] == 2


def test_rollup_window_excludes_days_outside_it(tmp_path):
    conn = open_cache_db(str(tmp_path / "u.db"))
    kw = {"backend": "b", "model": "m"}
    record_sample(conn, day="2026-08-19", current=CacheCounters(0, 0), **kw)
    record_sample(conn, day="2026-08-19", current=CacheCounters(10, 5), **kw)
    record_sample(conn, day="2026-08-21", current=CacheCounters(30, 15), **kw)
    rows = cache_rollup(conn, "2026-08-20", "2026-08-21")
    assert [r["day"] for r in rows] == ["2026-08-21"]


def test_priced_input_tokens_expresses_the_cache_read_discount():
    # Anthropic bills a cache read at 10% of input and a cache write at 125%.
    assert CACHE_READ_MULTIPLIER == pytest.approx(0.10)
    assert CACHE_WRITE_MULTIPLIER == pytest.approx(1.25)
    # 1000 queried, 900 of them cache hits: 100 fresh + 900 at a tenth.
    assert priced_input_tokens(1000, 900) == pytest.approx(100 + 90)
    # No reuse prices at face value.
    assert priced_input_tokens(1000, 0) == pytest.approx(1000)
    # A caller modelling explicit cache writes pays the write premium on misses.
    assert priced_input_tokens(
        1000, 900, write_multiplier=CACHE_WRITE_MULTIPLIER
    ) == pytest.approx(125 + 90)


def test_priced_input_tokens_never_prices_more_reads_than_were_queried():
    assert priced_input_tokens(100, 5000) == pytest.approx(10.0)


def test_the_daily_table_stays_at_the_day_model_grain(tmp_path):
    # Structural guard on the grain. The per-request lane (requests.cache_read_
    # tokens, fed by prompt_tokens_details.cached_tokens) is where per-principal
    # attribution belongs; this table is a fleet-level trend and must not grow
    # a principal, client, or request column.
    conn = open_cache_db(str(tmp_path / "u.db"))
    columns = {r[1] for r in conn.execute("PRAGMA table_info(cache_daily)")}
    assert {"day", "model", "queried_tokens", "cache_read_tokens"} <= columns
    assert not columns & {"principal", "client", "request_id", "requests"}


def test_outcome_constants_are_distinct():
    outcomes = {OUTCOME_BASELINE, OUTCOME_COUNTED, OUTCOME_RESET,
                OUTCOME_REJECTED, OUTCOME_UNCHANGED, OUTCOME_NOT_REPORTED}
    assert len(outcomes) == 6
