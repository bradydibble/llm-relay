"""Async job endpoints + lifespan wiring (plan 4 slice 2)."""
from __future__ import annotations

import asyncio

import httpx

from llm_relay.api.app import create_app
from llm_relay.jobs import STATUS_INTERRUPTED, STATUS_RUNNING, JobStore


def _min_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    return tmp_path


async def test_submit_returns_202_and_queues(tmp_path):
    app = create_app(config_dir=_min_cfg(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/jobs",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Llm-Relay-Urgency": "low", "X-Llm-Relay-SLA-Class": "agentic"},
        )
        assert r.status_code == 202
        jid = r.json()["job_id"]
        assert r.json()["status"] == "queued"
        g = await client.get(f"/v1/jobs/{jid}")
    assert g.status_code == 200
    assert g.json()["status"] == "queued"
    assert g.json()["urgency"] == "low"
    assert g.json()["sla_class"] == "agentic"


async def test_get_unknown_job_404(tmp_path):
    app = create_app(config_dir=_min_cfg(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/v1/jobs/does-not-exist")
    assert r.status_code == 404


async def test_cancel_queued_job(tmp_path):
    app = create_app(config_dir=_min_cfg(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        jid = (await client.post("/v1/jobs", json={"model": "m", "messages": []})).json()["job_id"]
        c = await client.post(f"/v1/jobs/{jid}/cancel")
        assert c.status_code == 200
        assert c.json()["cancelled"] is True
        g = await client.get(f"/v1/jobs/{jid}")
    assert g.json()["status"] == "cancelled"


async def test_lifespan_reconciles_running_jobs(tmp_path):
    # Pre-seed a job orphaned in 'running' by a simulated crash.
    seed = JobStore(tmp_path / "jobs.json")
    j = seed.create(principal="a", body={}, sla_class=None, urgency=None,
                    priority_weight=1.0, created_ts=1.0)
    j.status = STATUS_RUNNING
    seed.update(j)
    app = create_app(config_dir=_min_cfg(tmp_path))

    async def _startup_status():
        async with app.router.lifespan_context(app):
            return app.state.job_store.get(j.id).status

    status = await asyncio.wait_for(_startup_status(), timeout=15)
    assert status == STATUS_INTERRUPTED
