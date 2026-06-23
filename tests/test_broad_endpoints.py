"""Broad endpoints (plan 6): embeddings + rerank via a generic simple proxy."""
from __future__ import annotations

import httpx

from llm_relay.api.app import create_app
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.router import RequestRouter


def _write_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers:\n  p:\n    base_url: http://127.0.0.1\n")
    (tmp_path / "models.yaml").write_text("models:\n  emb:\n    provider: p\n    port: 8000\n")


def _seed(disc: DiscoveryManager, model: str, provider: str = "p") -> None:
    key = f"k:{model}"
    disc.clients[key] = EndpointClient(
        provider_name=provider, base_url="http://127.0.0.1:8000",
        state=EndpointState(provider=provider, status=EndpointStatus.healthy, models=[model]),
    )
    disc.model_to_client[model] = key


async def test_route_simple_forwards_to_given_upstream_path(tmp_path, monkeypatch):
    _write_cfg(tmp_path)
    cfg = ConfigLoader(config_dir=tmp_path)
    cfg.load()
    disc = DiscoveryManager()
    _seed(disc, "emb")
    r = RequestRouter(cfg, disc)

    captured: dict = {}

    async def fake_forward(backend_url, model_name, request_data, headers=None,
                           backend_key=None, slot_wait_timeout=30.0, upstream_path="chat/completions"):
        captured["path"] = upstream_path
        captured["model"] = model_name
        return httpx.Response(200, json={"object": "list", "data": []})

    monkeypatch.setattr(r, "forward_request", fake_forward)
    resp, result = await r.route_simple({"model": "emb", "input": "hi"}, {}, "embeddings")
    assert captured["path"] == "embeddings"
    assert captured["model"] == "emb"
    assert resp.status_code == 200
    assert result.selected_model == "emb"


async def test_embeddings_endpoint_returns_upstream_payload(tmp_path, monkeypatch):
    _write_cfg(tmp_path)
    app = create_app(config_dir=tmp_path)

    async def fake_route_simple(request_data, headers=None, upstream_path="embeddings"):
        from llm_relay.routing.router import RouteResult
        assert upstream_path == "embeddings"
        return httpx.Response(200, json={"object": "list", "data": [{"embedding": [0.1]}]}), \
            RouteResult(success=True, selected_model="emb", backend_url="x", provider_name="p")

    monkeypatch.setattr(app.state.router, "route_simple", fake_route_simple)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={"model": "emb", "input": "hi"})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["embedding"] == [0.1]
    assert resp.headers.get("X-Llm-Relay-Selected-Model") == "emb"


async def test_rerank_endpoint_routes_to_rerank_path(tmp_path, monkeypatch):
    _write_cfg(tmp_path)
    app = create_app(config_dir=tmp_path)
    seen: dict = {}

    async def fake_route_simple(request_data, headers=None, upstream_path="embeddings"):
        from llm_relay.routing.router import RouteResult
        seen["path"] = upstream_path
        return httpx.Response(200, json={"results": []}), \
            RouteResult(success=True, selected_model="emb", backend_url="x", provider_name="p")

    monkeypatch.setattr(app.state.router, "route_simple", fake_route_simple)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/rerank", json={"model": "emb", "query": "q", "documents": ["a"]})
    assert resp.status_code == 200
    assert seen["path"] == "rerank"


async def test_embeddings_no_candidate_is_503(tmp_path):
    _write_cfg(tmp_path)
    app = create_app(config_dir=tmp_path)  # nothing seeded -> no live model
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={"model": "emb", "input": "hi"})
    assert resp.status_code == 503
