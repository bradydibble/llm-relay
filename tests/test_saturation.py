"""Verify per-backend semaphore acquire/release and timeout behavior."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import yaml

from llm_relay.config.types import CircuitBreaker, EndpointState, ModelStatus, SaturationError
from llm_relay.discovery import endpoint as endpoint_mod
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager


def _make_client(max_concurrent: int | None) -> EndpointClient:
    return EndpointClient(
        provider_name="test",
        base_url="http://nope",
        state=EndpointState(provider="test"),
        circuit_breaker=CircuitBreaker(),
        max_concurrent=max_concurrent,
    )


async def test_acquire_slot_returns_immediately_when_unbounded():
    """max_concurrent=None means no semaphore — acquire is a no-op."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=None)

    async with disc.acquire_slot("k", wait_timeout=0.1):
        pass  # no error, no wait


async def test_acquire_slot_blocks_when_saturated_then_releases():
    """With max_concurrent=1, a second acquire must wait until the first releases.

    Uses an event to coordinate first-acquired so the test doesn't rely on
    arbitrary asyncio.sleep timing.
    """
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)

    first_acquired = asyncio.Event()
    release_first = asyncio.Event()

    async def first():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            first_acquired.set()
            await release_first.wait()

    async def second():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            return "got it"

    t1 = asyncio.create_task(first())
    await first_acquired.wait()  # deterministic: first really has the permit

    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.01)  # short tick to let t2's acquire actually start blocking
    assert not t2.done(), "second acquire should still be waiting"

    release_first.set()
    result = await asyncio.wait_for(t2, timeout=1.0)
    await t1
    assert result == "got it"


async def test_acquire_slot_raises_saturation_error_on_timeout():
    """If wait_timeout elapses with no slot free, raise SaturationError carrying retry_after."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)

    hold_event = asyncio.Event()

    async def holder():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            await hold_event.wait()

    t = asyncio.create_task(holder())
    await asyncio.sleep(0.05)

    with pytest.raises(SaturationError) as excinfo:
        async with disc.acquire_slot("k", wait_timeout=0.1):
            pytest.fail("should not have acquired")

    assert excinfo.value.backend_key == "k"
    assert excinfo.value.retry_after_seconds > 0

    hold_event.set()
    await t


async def test_acquire_slot_releases_on_inner_exception():
    """If the body of `async with acquire_slot` raises, the slot is still released."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with disc.acquire_slot("k", wait_timeout=0.5):
            raise _Boom()

    # Next acquire should succeed immediately — slot was released.
    async with disc.acquire_slot("k", wait_timeout=0.1):
        pass


async def test_acquire_slot_does_not_leak_permit_on_timeout():
    """A timeout during acquire() must NOT leak the permit.

    Scenario: max_concurrent=2. Holder takes 1 permit. Caller times out
    waiting for the 2nd permit (because we starve the loop). After holder
    releases, the second permit should be reusable — if the timeout leaked
    one, we'd be unable to acquire even with the holder released.
    """
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=2)

    # Hold permit 1 for the duration of the test.
    holder_release = asyncio.Event()

    async def holder():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            await holder_release.wait()

    t = asyncio.create_task(holder())
    await asyncio.sleep(0.05)  # let holder acquire

    # Saturate permit 2 with a separate slow holder.
    slow_release = asyncio.Event()

    async def slow_holder():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            await slow_release.wait()

    t_slow = asyncio.create_task(slow_holder())
    await asyncio.sleep(0.05)

    # Now both permits are held. Try to acquire with a short timeout — must fail.
    with pytest.raises(SaturationError):
        async with disc.acquire_slot("k", wait_timeout=0.05):
            pytest.fail("should not have acquired")

    # Release the slow holder; permit 2 should now be available.
    slow_release.set()
    await t_slow

    # A fresh acquire on the released permit must succeed quickly.
    # If the prior timeout had leaked a permit, this would block / timeout.
    async with disc.acquire_slot("k", wait_timeout=0.5):
        pass  # success means no leak

    # Cleanup.
    holder_release.set()
    await t


# ---------------------------------------------------------------------------
# API-layer: 503 + Retry-After on SaturationError
# ---------------------------------------------------------------------------

def _make_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal providers.yaml + models.yaml so create_app can load."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {
                "type": "openai",
                "base_url": "http://127.0.0.1",
                "enabled": True,
                "max_concurrent": 1,
                "slot_wait_timeout": 5.0,
            }
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "test-model": {
                "provider": "local-llm",
                "class": "unknown",
                "privacy": "local_only",
            }
        }
    }))
    return cfg_dir


async def test_chat_completions_returns_503_retry_after_on_saturation(tmp_path, monkeypatch):
    """When the router raises SaturationError, /v1/chat/completions returns
    503 with a Retry-After header hinting the retry window.

    Patches route_and_forward (the new entry point the API handler calls) to
    raise SaturationError directly, exercising the API's exception handling.
    """
    import httpx
    from httpx import ASGITransport

    from llm_relay.api.app import create_app

    cfg_dir = _make_minimal_config(tmp_path)
    app = create_app(config_dir=cfg_dir)

    # Patch route_and_forward — the method the API handler now calls — to raise
    # SaturationError, bypassing discovery/selection entirely.
    async def _fake_route_and_forward(request_data, headers=None, stream=False):
        raise SaturationError(backend_key="local-llm", retry_after_seconds=5.0)

    monkeypatch.setattr(app.state.router, "route_and_forward", _fake_route_and_forward)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 503
    assert "Retry-After" in resp.headers
    retry_after = int(resp.headers["Retry-After"])
    assert retry_after >= 1
    body = resp.json()
    assert body["detail"]["error"] == "backend saturated"
    assert body["detail"]["backend"] == "local-llm"


async def test_register_backend_propagates_max_concurrent_from_provider_config():
    """ProviderConfig.max_concurrent must reach EndpointClient.inflight_sem via register_backend.

    Regression for the Task 5 wiring gap where register_backend ignored
    provider.max_concurrent and silently disabled saturation handling.
    """
    disc = DiscoveryManager()
    await disc.register_backend(
        key="test-backend",
        provider_name="test",
        base_url="http://nope",
        models_hint=["dummy-model"],
        poll_interval=999999,  # avoid actually polling
        max_concurrent=4,
    )
    try:
        client = disc.clients["test-backend"]
        assert client.max_concurrent == 4
        assert client.inflight_sem is not None
        # Behaviorally prove the semaphore has 4 slots: acquire 4 times (must succeed)
        # then verify the 5th times out (capacity exhausted).
        for _ in range(4):
            await asyncio.wait_for(client.inflight_sem.acquire(), timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.inflight_sem.acquire(), timeout=0.05)
        # Clean up: release the 4 we acquired so the disc.shutdown() path is tidy.
        for _ in range(4):
            client.inflight_sem.release()
    finally:
        await disc.shutdown()


async def test_inflight_used_increments_on_acquire_and_decrements_on_release():
    """inflight_used must reflect the number of slots currently held."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=2)
    client = disc.clients["k"]

    assert client.inflight_used == 0
    async with disc.acquire_slot("k", wait_timeout=0.5):
        assert client.inflight_used == 1
        async with disc.acquire_slot("k", wait_timeout=0.5):
            assert client.inflight_used == 2
        assert client.inflight_used == 1
    assert client.inflight_used == 0


async def test_inflight_used_decrements_when_body_raises():
    """If the with-body raises, inflight_used must still decrement."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)
    client = disc.clients["k"]

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with disc.acquire_slot("k", wait_timeout=0.5):
            assert client.inflight_used == 1
            raise _Boom()
    assert client.inflight_used == 0


async def test_inflight_used_unchanged_on_saturation_timeout():
    """A saturation timeout never acquired a slot, so inflight_used must not
    drift — the holder still owns its slot count, and the timed-out caller
    contributes nothing."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)
    client = disc.clients["k"]

    release = asyncio.Event()

    async def holder():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            await release.wait()

    t = asyncio.create_task(holder())
    await asyncio.sleep(0.05)
    assert client.inflight_used == 1

    with pytest.raises(SaturationError):
        async with disc.acquire_slot("k", wait_timeout=0.05):
            pytest.fail("should not have acquired")

    # The failed acquire must NOT have nudged the counter — still exactly 1.
    assert client.inflight_used == 1

    release.set()
    await t
    assert client.inflight_used == 0


async def test_inflight_used_stays_zero_when_max_concurrent_is_none():
    """Unbounded backends don't track inflight — counter stays at 0."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=None)
    client = disc.clients["k"]

    assert client.inflight_used == 0
    async with disc.acquire_slot("k", wait_timeout=0.1):
        # No semaphore means no tracking. Capacity is None, so consumers
        # interpret "no saturation data" rather than "0/None saturated".
        assert client.inflight_used == 0
    assert client.inflight_used == 0


async def test_status_endpoint_emits_inflight_fields(tmp_path):
    """/status backend payload must include inflight_used and inflight_capacity
    so the describe_alias MCP tool can surface saturation to consumers.

    Bypasses lifespan — registers one backend manually on app.state.discovery,
    then asserts the GET /status response shape.
    """
    import httpx
    from httpx import ASGITransport

    from llm_relay.api.app import create_app

    cfg_dir = _make_minimal_config(tmp_path)
    app = create_app(config_dir=cfg_dir)

    # Manually register a backend (no real polling — large interval keeps it idle).
    await app.state.discovery.register_backend(
        key="test-backend",
        provider_name="local-llm",
        base_url="http://nope",
        models_hint=["test-model"],
        poll_interval=999999,
        max_concurrent=3,
    )
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/status")
    finally:
        await app.state.discovery.shutdown()

    assert resp.status_code == 200
    backends = resp.json()["backends"]
    assert "test-backend" in backends, backends
    payload = backends["test-backend"]
    assert payload["inflight_used"] == 0
    assert payload["inflight_capacity"] == 3


async def test_register_backend_without_max_concurrent_yields_no_semaphore():
    """When max_concurrent is omitted, inflight_sem stays None (unbounded)."""
    disc = DiscoveryManager()
    await disc.register_backend(
        key="unbounded",
        provider_name="test",
        base_url="http://nope",
        models_hint=["dummy-model"],
        poll_interval=999999,
    )
    try:
        client = disc.clients["unbounded"]
        assert client.max_concurrent is None
        assert client.inflight_sem is None
    finally:
        await disc.shutdown()


# ---------------------------------------------------------------------------
# Fix #2: periodic reconciliation of a stuck (leaked) in-flight counter.
# Containment, not a cure — the real fix is the synchronous release in
# stream_request. This keeps a missed release from permanently shrinking
# capacity by bounding the blast radius to one polling cycle.
# ---------------------------------------------------------------------------

async def test_acquire_slot_records_last_dispatched_at():
    """Acquiring a slot stamps last_dispatched_at so the poll loop can tell a
    live backend from one whose counter is stranded."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)
    client = disc.clients["k"]

    assert client.last_dispatched_at is None
    async with disc.acquire_slot("k", wait_timeout=0.5):
        assert client.last_dispatched_at is not None
        first = client.last_dispatched_at

    await asyncio.sleep(0.01)
    async with disc.acquire_slot("k", wait_timeout=0.5):
        assert client.last_dispatched_at >= first


async def test_reconcile_resets_stuck_inflight_after_idle():
    """inflight_used>0 with no dispatch inside the idle window == leaked slots:
    zero the counter, re-init the semaphore at full capacity, record the event."""
    disc = DiscoveryManager(slot_reconcile_idle_seconds=10.0)
    client = _make_client(max_concurrent=2)
    disc.clients["k"] = client
    client.inflight_used = 2
    client.last_dispatched_at = time.monotonic() - 11.0  # past the idle window

    disc._reconcile_stuck_slots(client)

    assert client.inflight_used == 0
    assert client.slot_reconciliations == 1
    # Fresh semaphore at full capacity: two acquires succeed, a third blocks.
    for _ in range(2):
        await asyncio.wait_for(client.inflight_sem.acquire(), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(client.inflight_sem.acquire(), timeout=0.05)


async def test_reconcile_skips_when_recently_dispatched():
    """A recent dispatch means the backend is presumed legitimately busy (e.g. a
    long stream) — never reconciled, so live requests aren't disrupted."""
    disc = DiscoveryManager(slot_reconcile_idle_seconds=10.0)
    client = _make_client(max_concurrent=2)
    disc.clients["k"] = client
    client.inflight_used = 1
    client.last_dispatched_at = time.monotonic()  # just now

    disc._reconcile_stuck_slots(client)

    assert client.inflight_used == 1
    assert client.slot_reconciliations == 0


async def test_reconcile_noop_when_counter_is_zero():
    """Nothing stranded → no reconcile even if the backend has been idle."""
    disc = DiscoveryManager(slot_reconcile_idle_seconds=10.0)
    client = _make_client(max_concurrent=2)
    disc.clients["k"] = client
    client.inflight_used = 0
    client.last_dispatched_at = time.monotonic() - 100.0

    disc._reconcile_stuck_slots(client)

    assert client.inflight_used == 0
    assert client.slot_reconciliations == 0


# ---------------------------------------------------------------------------
# Fix #3: backend-wipe lifecycle hook. A backend that was down (circuit tripped)
# or reloaded (model set changed) has effectively restarted — its pre-outage
# in-flight slots are dead. fetch_models wipes the stale accounting on recovery.
# ---------------------------------------------------------------------------

def _patch_models_fetch(monkeypatch, model_ids):
    """Make EndpointClient.fetch_models' httpx GET return the given model ids."""
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"object": "list", "data": [{"id": m} for m in model_ids]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr(endpoint_mod.httpx, "AsyncClient", lambda *a, **k: _Client())


async def test_backend_reset_on_circuit_recovery_wipes_inflight(monkeypatch):
    """Circuit was open (sustained outage) → the first successful poll wipes any
    in-flight slots stranded across the outage."""
    _patch_models_fetch(monkeypatch, ["m1"])
    client = _make_client(max_concurrent=2)
    client.state.models = ["m1"]  # same model set returns
    client.inflight_used = 2  # stranded across the outage
    client.state.circuit_open = True
    client.state.circuit_opened_at = time.monotonic() - 9999  # recovery window elapsed

    models = await client.fetch_models()

    assert models == ["m1"]
    assert client.inflight_used == 0, "stale in-flight slots must be wiped on recovery"
    assert client.backend_resets == 1
    assert client.state.circuit_open is False


async def test_backend_reset_on_model_change_wipes_inflight(monkeypatch):
    """A freshly-loaded model (model set changed) means the backend reloaded —
    wipe stale in-flight accounting even without a circuit trip."""
    _patch_models_fetch(monkeypatch, ["new-model"])
    client = _make_client(max_concurrent=2)
    client.state.models = ["old-model"]  # previous poll saw a different model
    client.inflight_used = 1

    models = await client.fetch_models()

    assert models == ["new-model"]
    assert client.inflight_used == 0
    assert client.backend_resets == 1


async def test_no_backend_reset_on_first_successful_poll(monkeypatch):
    """Initial discovery (no prior models, healthy from the start) is not a
    reset — don't wipe or count it."""
    _patch_models_fetch(monkeypatch, ["m1"])
    client = _make_client(max_concurrent=2)
    assert client.state.models == []  # fresh
    client.inflight_used = 1  # e.g. a request already in flight

    models = await client.fetch_models()

    assert models == ["m1"]
    assert client.inflight_used == 1, "first poll must not wipe a legitimately in-flight slot"
    assert client.backend_resets == 0


async def test_brief_blip_below_threshold_does_not_wipe(monkeypatch):
    """A couple of failures that never tripped the circuit, then success with the
    same models, is a transient blip — in-flight slots stay (request may still
    be valid)."""
    _patch_models_fetch(monkeypatch, ["m1"])
    client = _make_client(max_concurrent=2)
    client.state.models = ["m1"]
    client.inflight_used = 1
    client.state.consecutive_failures = 1  # below threshold; circuit never opened

    models = await client.fetch_models()

    assert models == ["m1"]
    assert client.inflight_used == 1
    assert client.backend_resets == 0
