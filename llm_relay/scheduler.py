"""Synchronous QoS admission (plan 4, slice 1).

The agentic-first scheduler's first, safe slice: a stateless admission gate that
sheds explicitly low-urgency work under fleet contention, so high-urgency and
interactive work keeps flowing. It gates on the EXISTING per-backend slot signals
(``inflight_used`` / ``max_concurrent`` already tracked by discovery) -- no new
worker pool, no persistence, no per-request lifecycle state.

Deliberately deferred to the next slice (see the plan-4 design doc), because they
are stateful / concurrent / persistent and must not be rushed onto the live
router: soft per-principal fair-share (needs in-flight counting across the
streaming completion lifecycle) and the async ``/v1/jobs`` store + worker.
"""
from __future__ import annotations


class AdmissionController:
    def __init__(self, contention_threshold: float = 0.85) -> None:
        # Fraction of total fleet slots in use above which we start shedding
        # low-urgency work. Below it there is spare capacity, so nothing is shed.
        self.contention_threshold = contention_threshold

    def global_load(self, discovery) -> float:
        """Fleet in-flight load: sum(inflight_used) / sum(max_concurrent) across
        backends that declare a slot count. 0.0 when no backend bounds concurrency
        (nothing to contend for)."""
        used = 0
        total = 0
        for c in discovery.clients.values():
            mc = getattr(c, "max_concurrent", None)
            if mc:
                used += getattr(c, "inflight_used", 0)
                total += mc
        return (used / total) if total else 0.0

    def should_shed(self, urgency: str | None, discovery) -> bool:
        """Shed only requests that explicitly declared ``urgency: low``, and only
        when the fleet is contended (load >= threshold). Everything else is
        admitted, subject to the existing per-backend saturation handling."""
        if (urgency or "").lower() != "low":
            return False
        return self.global_load(discovery) >= self.contention_threshold
