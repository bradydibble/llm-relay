"""QoS admission (plan 4, slice 1): shed low-urgency work under contention."""
from __future__ import annotations

import httpx

from llm_relay.api.app import create_app
from llm_relay.scheduler import AdmissionController


class _FakeClient:
    def __init__(self, inflight_used, max_concurrent):
        self.inflight_used = inflight_used
        self.max_concurrent = max_concurrent


class _FakeDisc:
    def __init__(self, clients):
        self.clients = {f"k{i}": c for i, c in enumerate(clients)}


def test_global_load_computes_ratio():
    a = AdmissionController()
    assert a.global_load(_FakeDisc([_FakeClient(2, 4), _FakeClient(2, 4)])) == 0.5


def test_global_load_zero_when_no_bounded_capacity():
    a = AdmissionController()
    assert a.global_load(_FakeDisc([_FakeClient(0, None)])) == 0.0
    assert a.global_load(_FakeDisc([])) == 0.0


def test_should_shed_only_low_urgency_and_only_under_contention():
    a = AdmissionController(contention_threshold=0.85)
    busy = _FakeDisc([_FakeClient(4, 4)])  # load 1.0
    idle = _FakeDisc([_FakeClient(0, 4)])  # load 0.0
    assert a.should_shed("low", busy) is True
    assert a.should_shed("low", idle) is False
    assert a.should_shed("normal", busy) is False
    assert a.should_shed("high", busy) is False
    assert a.should_shed(None, busy) is False


def _cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models:\n  m:\n    provider: p\n    port: 8000\n")
    return tmp_path


async def test_low_urgency_request_shed_with_429_under_contention(tmp_path, monkeypatch):
    app = create_app(config_dir=_cfg(tmp_path))
    monkeypatch.setattr(app.state.admission, "global_load", lambda disc: 1.0)  # force contention
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Llm-Relay-Urgency": "low"},
        )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


async def test_normal_urgency_not_shed_under_contention(tmp_path, monkeypatch):
    app = create_app(config_dir=_cfg(tmp_path))
    monkeypatch.setattr(app.state.admission, "global_load", lambda disc: 1.0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Llm-Relay-Urgency": "normal"},
        )
    # Not shed: it proceeds to routing (no live backend -> 503), but not a 429 shed.
    assert resp.status_code != 429
