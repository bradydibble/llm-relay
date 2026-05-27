"""Verify per-backend semaphore acquire/release and timeout behavior."""
from __future__ import annotations

import asyncio

import pytest

from llm_relay.config.types import CircuitBreaker, EndpointState, SaturationError
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
