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
    """With max_concurrent=1, a second acquire must wait until the first releases."""
    disc = DiscoveryManager()
    disc.clients["k"] = _make_client(max_concurrent=1)

    release_first = asyncio.Event()

    async def first():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            await release_first.wait()

    async def second():
        async with disc.acquire_slot("k", wait_timeout=2.0):
            return "got it"

    t1 = asyncio.create_task(first())
    await asyncio.sleep(0.05)  # let first() acquire
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.05)
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
