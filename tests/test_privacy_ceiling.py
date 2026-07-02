"""Privacy ceiling: unscoped keys can never escalate to cloud_ok."""
from llm_relay.api.app import _clamp_privacy
from llm_relay.auth import Principal


def test_clamp_when_no_cloud_scope():
    h = {"X-Llm-Relay-Privacy": "cloud_ok"}
    _clamp_privacy(Principal(id="jdoe"), True, h)
    assert h["X-Llm-Relay-Privacy"] == "local_only"


def test_no_clamp_with_cloud_scope():
    h = {"X-Llm-Relay-Privacy": "cloud_ok"}
    _clamp_privacy(Principal(id="internal", scopes=["admin", "cloud"]), True, h)
    assert h["X-Llm-Relay-Privacy"] == "cloud_ok"


def test_no_clamp_when_auth_disabled():
    h = {"X-Llm-Relay-Privacy": "cloud_ok"}
    _clamp_privacy(Principal(id="anonymous"), False, h)
    assert h["X-Llm-Relay-Privacy"] == "cloud_ok"


def test_clamp_handles_missing_header():
    h = {}
    _clamp_privacy(Principal(id="jdoe"), True, h)
    assert h.get("X-Llm-Relay-Privacy", "local_only") == "local_only"


def test_clamp_handles_none_principal():
    h = {"X-Llm-Relay-Privacy": "cloud_ok"}
    _clamp_privacy(None, True, h)
    assert h["X-Llm-Relay-Privacy"] == "local_only"
