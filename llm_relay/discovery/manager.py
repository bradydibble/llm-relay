"""Health polling and model discovery manager."""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config.types import CircuitBreaker, EndpointState, EndpointStatus, ModelStatus, SaturationError
from .endpoint import EndpointClient


@dataclass
class DiscoveryManager:
    """Track many backends (provider+port/path combos) and per-model availability."""

    clients: dict[str, EndpointClient] = field(default_factory=dict)
    model_to_client: dict[str, str] = field(default_factory=dict)
    _tasks: list[asyncio.Task] = field(default_factory=list)

    async def register_backend(
        self,
        key: str,
        provider_name: str,
        base_url: str,
        models_hint: list[str],
        health_endpoint: str = "/v1/models",
        poll_interval: int = 15,
        circuit_breaker: CircuitBreaker | None = None,
        timeout: float = 5.0,
        max_concurrent: int | None = None,
    ) -> None:
        state = EndpointState(provider=provider_name)
        client = EndpointClient(
            provider_name=provider_name,
            base_url=base_url,
            health_endpoint=health_endpoint,
            timeout=timeout,
            state=state,
            circuit_breaker=circuit_breaker or CircuitBreaker(),
            max_concurrent=max_concurrent,
        )
        self.clients[key] = client
        for m in models_hint:
            self.model_to_client[m] = key
        self._tasks.append(asyncio.create_task(self._poll_loop(client, poll_interval)))

    @contextlib.asynccontextmanager
    async def acquire_slot(self, key: str, wait_timeout: float):
        """Acquire an in-flight slot for backend `key`, releasing on exit.

        If the backend was registered without max_concurrent (or doesn't exist),
        this is a no-op. Raises SaturationError if no slot becomes available
        within wait_timeout, carrying a retry_after_seconds hint.
        """
        client = self.clients.get(key)
        if client is None or client.inflight_sem is None:
            yield
            return

        try:
            await asyncio.wait_for(client.inflight_sem.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError as e:
            raise SaturationError(backend_key=key, retry_after_seconds=wait_timeout) from e

        try:
            yield
        finally:
            client.inflight_sem.release()

    async def _poll_loop(self, client: EndpointClient, interval: int) -> None:
        while True:
            try:
                models = await client.fetch_models()
                client.state.last_poll = datetime.now(timezone.utc).isoformat()
                if models:
                    client.state.status = EndpointStatus.healthy
                    client.state.models = models
                else:
                    client.state.status = EndpointStatus.unavailable
                    client.state.models = []
            except Exception:
                client.state.status = EndpointStatus.unavailable
            await asyncio.sleep(interval)

    def get_model_state(self, model_name: str) -> ModelStatus:
        key = self.model_to_client.get(model_name)
        if key:
            client = self.clients.get(key)
            if client:
                if client.state.status == EndpointStatus.healthy:
                    return ModelStatus.available
                if client.state.status == EndpointStatus.degraded:
                    return ModelStatus.degraded
                return ModelStatus.unavailable
        for client in self.clients.values():
            if model_name in client.state.models:
                if client.state.status == EndpointStatus.healthy:
                    return ModelStatus.available
                return ModelStatus.degraded
        return ModelStatus.unavailable

    def get_client_for_model(self, model_name: str) -> EndpointClient | None:
        key = self.model_to_client.get(model_name)
        if key:
            return self.clients.get(key)
        for client in self.clients.values():
            if model_name in client.state.models:
                return client
        return None

    def get_available_models(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for client in self.clients.values():
            for m in client.state.models:
                result[m] = {
                    "provider": client.provider_name,
                    "status": client.state.status.value,
                    "last_poll": client.state.last_poll,
                }
        return result

    def get_endpoint_status(self, key: str) -> EndpointState | None:
        c = self.clients.get(key)
        return c.state if c else None

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
