"""Async job store (plan 4 slice 2): durable, atomic, priority-ordered."""
from __future__ import annotations

from llm_relay.jobs import (
    STATUS_CANCELLED,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobStore,
)


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.json")


def test_create_and_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    j = s.create(principal="alice", body={"model": "m"}, sla_class="agentic",
                 urgency="normal", priority_weight=1.0, created_ts=1.0)
    assert j.status == STATUS_QUEUED
    assert s.get(j.id).body == {"model": "m"}


def test_persistence_survives_reload(tmp_path):
    s = _store(tmp_path)
    j = s.create(principal="bob", body={}, sla_class=None, urgency=None,
                 priority_weight=1.0, created_ts=1.0)
    s2 = JobStore(tmp_path / "jobs.json")  # fresh instance, same file
    assert s2.get(j.id) is not None
    assert s2.get(j.id).principal == "bob"


def test_next_queued_priority_order(tmp_path):
    s = _store(tmp_path)
    # Lower weight, submitted first.
    a = s.create(principal="a", body={}, sla_class=None, urgency="normal",
                 priority_weight=1.0, created_ts=1.0)
    # Higher effective priority (weight x urgency) -> served first despite later submit.
    b = s.create(principal="b", body={}, sla_class=None, urgency="high",
                 priority_weight=2.0, created_ts=2.0)
    assert s.next_queued().id == b.id
    b.status = STATUS_RUNNING
    s.update(b)
    assert s.next_queued().id == a.id


def test_cancel_only_queued(tmp_path):
    s = _store(tmp_path)
    j = s.create(principal="a", body={}, sla_class=None, urgency=None,
                 priority_weight=1.0, created_ts=1.0)
    assert s.cancel(j.id) is True
    assert s.get(j.id).status == STATUS_CANCELLED
    # A running job cannot be cancelled (admission-time only).
    j2 = s.create(principal="a", body={}, sla_class=None, urgency=None,
                  priority_weight=1.0, created_ts=2.0)
    j2.status = STATUS_RUNNING
    s.update(j2)
    assert s.cancel(j2.id) is False


def test_reconcile_marks_running_interrupted(tmp_path):
    s = _store(tmp_path)
    j = s.create(principal="a", body={}, sla_class=None, urgency=None,
                 priority_weight=1.0, created_ts=1.0)
    j.status = STATUS_RUNNING
    s.update(j)
    # A fresh store (simulating a restart) reconciles the orphaned running job.
    s2 = JobStore(tmp_path / "jobs.json")
    s2.reconcile_on_start()
    assert s2.get(j.id).status == STATUS_INTERRUPTED
    # Not re-run: it is terminal, never silently requeued.
    assert s2.next_queued() is None


def test_next_queued_fair_share_cap(tmp_path):
    s = _store(tmp_path)
    a1 = s.create(principal="a", body={}, sla_class=None, urgency=None,
                  priority_weight=1.0, created_ts=1.0)
    s.create(principal="a", body={}, sla_class=None, urgency=None,
             priority_weight=1.0, created_ts=2.0)
    b1 = s.create(principal="b", body={}, sla_class=None, urgency=None,
                  priority_weight=1.0, created_ts=3.0)
    # 'a' already has 1 running and cap=1 -> a's queued jobs are skipped for b1.
    assert s.next_queued(max_per_principal=1, running_counts={"a": 1}).id == b1.id
    # No cap -> plain priority/FIFO order (a1 first).
    assert s.next_queued().id == a1.id


def test_running_counts_and_queued_principals(tmp_path):
    s = _store(tmp_path)
    a = s.create(principal="a", body={}, sla_class=None, urgency=None,
                 priority_weight=1.0, created_ts=1.0)
    s.create(principal="b", body={}, sla_class=None, urgency=None,
             priority_weight=1.0, created_ts=2.0)
    assert s.queued_principals() == {"a", "b"}
    a.status = STATUS_RUNNING
    s.update(a)
    assert s.running_counts() == {"a": 1}
    assert s.queued_principals() == {"b"}
