"""Async job worker (plan 4 slice 2): runs queued jobs via route_and_forward."""
from __future__ import annotations

import asyncio

import httpx

from llm_relay.jobs import STATUS_DONE, STATUS_ERROR, STATUS_QUEUED, STATUS_RUNNING, JobStore
from llm_relay.jobworker import run_worker


class _FakeRouter:
    def __init__(self, status=200, body=None, exc=None):
        self.status = status
        self.body = body if body is not None else {"ok": True}
        self.exc = exc
        self.calls: list = []

    async def route_and_forward(self, request_data, headers=None, stream=False):
        self.calls.append((request_data, headers))
        if self.exc:
            raise self.exc
        return httpx.Response(self.status, json=self.body), None


async def _drain(store, router, timeout=3.0):
    stop = asyncio.Event()
    task = asyncio.create_task(run_worker(store, router, stop, poll_interval=0.01))

    async def _wait():
        while any(j.status in (STATUS_QUEUED, STATUS_RUNNING) for j in store.jobs.values()):
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(_wait(), timeout)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout)


async def test_worker_runs_job_to_done(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.create(principal="a", body={"model": "m"}, sla_class="agentic",
                       urgency="normal", priority_weight=1.0, created_ts=1.0)
    router = _FakeRouter(status=200, body={"choices": [{"text": "hi"}]})
    await _drain(store, router)
    done = store.get(job.id)
    assert done.status == STATUS_DONE
    assert done.result == {"choices": [{"text": "hi"}]}
    # Intent flowed through as headers.
    assert router.calls[0][1].get("X-Llm-Relay-Urgency") == "normal"


async def test_worker_marks_error_on_exception(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.create(principal="a", body={}, sla_class=None, urgency=None,
                       priority_weight=1.0, created_ts=1.0)
    await _drain(store, _FakeRouter(exc=RuntimeError("boom")))
    j = store.get(job.id)
    assert j.status == STATUS_ERROR
    assert "boom" in (j.error or "")


async def test_worker_marks_error_on_upstream_4xx(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.create(principal="a", body={}, sla_class=None, urgency=None,
                       priority_weight=1.0, created_ts=1.0)
    await _drain(store, _FakeRouter(status=400, body={"error": "bad"}))
    assert store.get(job.id).status == STATUS_ERROR


async def test_worker_skips_cancelled_job(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    keep = store.create(principal="a", body={}, sla_class=None, urgency=None,
                        priority_weight=1.0, created_ts=1.0)
    drop = store.create(principal="a", body={}, sla_class=None, urgency=None,
                        priority_weight=1.0, created_ts=2.0)
    store.cancel(drop.id)
    router = _FakeRouter(status=200)
    await _drain(store, router)
    assert store.get(keep.id).status == STATUS_DONE
    assert store.get(drop.id).status == "cancelled"
    assert len(router.calls) == 1  # the cancelled job never ran
