"""Async job worker (plan 4 slice 2).

Pops queued jobs in priority order and runs them through the existing
``route_and_forward`` (non-streaming). Concurrency is bounded by an ephemeral
task set, NOT a pool with persistent counters: the per-backend ``acquire_slot``
machinery inside ``route_and_forward`` does the real "how many run" bounding, so
there is no parallel concurrency state to keep coherent across restarts. A job is
marked ``running`` synchronously before its task is launched, so the next
``next_queued()`` can never re-pick it.
"""
from __future__ import annotations

import asyncio

from .jobs import STATUS_DONE, STATUS_ERROR, STATUS_RUNNING, Job, JobStore


def _headers_for(job: Job) -> dict[str, str]:
    headers: dict[str, str] = {}
    if job.sla_class:
        headers["X-Llm-Relay-SLA-Class"] = job.sla_class
    if job.urgency:
        headers["X-Llm-Relay-Urgency"] = job.urgency
    return headers


async def _execute(store: JobStore, router, job: Job) -> None:
    try:
        resp, _ = await router.route_and_forward(
            job.body, headers=_headers_for(job), stream=False,
        )
        try:
            content = resp.json()
        except Exception:
            content = {"raw": resp.text}
        if resp.status_code < 400:
            job.status = STATUS_DONE
            job.result = content
        else:
            job.status = STATUS_ERROR
            job.error = f"upstream status {resp.status_code}"
            job.result = content
    except Exception as e:  # noqa: BLE001 - any failure becomes a terminal job error
        job.status = STATUS_ERROR
        job.error = str(e)
    store.update(job)


async def run_worker(
    store: JobStore,
    router,
    stop_event: asyncio.Event,
    *,
    concurrency: int = 4,
    poll_interval: float = 0.5,
) -> None:
    """Run queued jobs until ``stop_event`` is set, then drain in-flight tasks.

    NOTE (v1): concurrency is capped at ``concurrency`` and route_and_forward's
    per-backend slots bound it further. Head-of-line fairness within a tier is
    FIFO by submit order; richer fair-share is the next refinement.
    """
    running: set[asyncio.Task] = set()
    while not stop_event.is_set():
        while len(running) < concurrency:
            # Soft fair-share: when more than one principal has queued work, cap
            # each at an even slice of the worker's concurrency (>=1) so no single
            # principal monopolizes the lane. Counts come from the store.
            principals = store.queued_principals()
            cap = max(1, concurrency // len(principals)) if len(principals) > 1 else None
            job = store.next_queued(max_per_principal=cap, running_counts=store.running_counts())
            if job is None:
                break
            # Mark running synchronously (no await) so the next next_queued()
            # cannot re-pick this job before its task starts.
            job.status = STATUS_RUNNING
            store.update(job)
            task = asyncio.create_task(_execute(store, router, job))
            running.add(task)
            task.add_done_callback(running.discard)
        if not running:
            await asyncio.sleep(poll_interval)
            continue
        await asyncio.wait(running, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED)
    if running:
        await asyncio.gather(*running, return_exceptions=True)
