"""principal label on request/token counters."""
from llm_relay import metrics


def test_record_request_emits_principal_label():
    m = metrics.get_metrics()
    m.record_request(
        alias="main", model="m", provider="p", outcome="success",
        client="jdoe", principal="jdoe",
        usage={"prompt_tokens": 1, "completion_tokens": 2},
        response_body=None, duration_s=0.1, fell_back=False,
    )
    body, _ = metrics.render_exposition()
    assert b'principal="jdoe"' in body


def test_record_request_defaults_principal_anonymous():
    m = metrics.get_metrics()
    m.record_request(
        alias="main", model="m2", provider="p", outcome="success",
        client="someone", usage=None,
        response_body=None, duration_s=0.1, fell_back=False,
    )
    body, _ = metrics.render_exposition()
    assert b'model="m2"' in body and b'principal="anonymous"' in body
