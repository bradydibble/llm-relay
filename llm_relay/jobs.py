"""Durable async job store (plan 4 slice 2).

Holds agentic jobs in memory and mirrors them to an atomic JSON file so a
submitted job survives a relay restart (the whole reason the async lane exists
over the synchronous one). Priority dequeue orders by effective priority
(principal weight x urgency). On restart, jobs left ``running`` by a crash are
reconciled to ``interrupted`` (terminal) rather than silently requeued, because
chat completions are not idempotent and a double-run could double-charge or
double-act.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_INTERRUPTED = "interrupted"
STATUS_CANCELLED = "cancelled"


def _urgency_factor(urgency: str | None) -> float:
    return {"high": 2.0, "normal": 1.0, "low": 0.5}.get((urgency or "normal").lower(), 1.0)


@dataclass
class Job:
    id: str
    principal: str
    body: dict
    sla_class: str | None
    urgency: str | None
    priority_weight: float
    submit_seq: int
    status: str = STATUS_QUEUED
    result: dict | None = None
    error: str | None = None
    created_ts: float = 0.0

    def public(self) -> dict:
        """Client-facing view (drops nothing sensitive; the body is the client's own)."""
        return asdict(self)


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.jobs: dict[str, Job] = {}
        self._seq = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for jd in data.get("jobs", []):
            try:
                j = Job(**jd)
            except TypeError:
                continue  # skip a row written by an incompatible older schema
            self.jobs[j.id] = j
            self._seq = max(self._seq, j.submit_seq)

    def _persist(self) -> None:
        doc = {"jobs": [asdict(j) for j in self.jobs.values()]}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def create(self, *, principal: str, body: dict, sla_class: str | None,
               urgency: str | None, priority_weight: float, created_ts: float) -> Job:
        self._seq += 1
        job = Job(
            id=uuid.uuid4().hex,
            principal=principal,
            body=body,
            sla_class=sla_class,
            urgency=urgency,
            priority_weight=priority_weight,
            submit_seq=self._seq,
            created_ts=created_ts,
        )
        self.jobs[job.id] = job
        self._persist()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def update(self, job: Job) -> None:
        self.jobs[job.id] = job
        self._persist()

    def next_queued(
        self,
        max_per_principal: int | None = None,
        running_counts: dict[str, int] | None = None,
    ) -> Job | None:
        """The highest-priority queued job: order by effective priority
        (weight x urgency) desc, then submit order asc (FIFO within a tier).

        Soft fair-share: when ``max_per_principal`` is set, skip a principal that
        already has that many jobs running (per ``running_counts``), so the next
        eligible principal's job is served instead. Counts come from the durable
        store (the source of truth), so there is no separate in-flight counter to
        leak. Returns None when every queued job's principal is at its cap."""
        queued = [j for j in self.jobs.values() if j.status == STATUS_QUEUED]
        if not queued:
            return None
        queued.sort(key=lambda j: (-(j.priority_weight * _urgency_factor(j.urgency)), j.submit_seq))
        if max_per_principal is None:
            return queued[0]
        rc = running_counts or {}
        for job in queued:
            if rc.get(job.principal, 0) < max_per_principal:
                return job
        return None

    def running_counts(self) -> dict[str, int]:
        """How many jobs each principal currently has running (for fair-share)."""
        counts: dict[str, int] = {}
        for j in self.jobs.values():
            if j.status == STATUS_RUNNING:
                counts[j.principal] = counts.get(j.principal, 0) + 1
        return counts

    def queued_principals(self) -> set[str]:
        return {j.principal for j in self.jobs.values() if j.status == STATUS_QUEUED}

    def cancel(self, job_id: str) -> bool:
        """Cancel a still-queued job (admission-time only). A running generation
        cannot be cancelled. Returns True if it was queued and is now cancelled."""
        job = self.jobs.get(job_id)
        if job is not None and job.status == STATUS_QUEUED:
            job.status = STATUS_CANCELLED
            self._persist()
            return True
        return False

    def reconcile_on_start(self) -> None:
        """Mark jobs orphaned ``running`` by a crash as ``interrupted`` (terminal).
        Never requeues them: re-running a non-idempotent completion is unsafe."""
        changed = False
        for job in self.jobs.values():
            if job.status == STATUS_RUNNING:
                job.status = STATUS_INTERRUPTED
                job.error = "interrupted by relay restart"
                changed = True
        if changed:
            self._persist()
