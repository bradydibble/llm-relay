"""Request routing and upstream forwarding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from ..config.loader import ConfigLoader
from ..config.types import ModelStatus, Privacy, SaturationError
from ..discovery.endpoint import _shared_upstream_bearer
from ..discovery.manager import DiscoveryManager
from .selector import ModelSelector, RoutingContext


def _compose_backend_key(provider_name: str, port: int | None, path: str) -> str:
    """Build the discovery-client key for a (provider, port, path) triple.

    Matches the key format used by ``create_app`` when calling
    ``register_backend``.  No models → just provider_name; port/path
    components are appended with ':' as separator.
    """
    parts = [provider_name]
    if port:
        parts.append(str(port))
    if path:
        parts.append(path.strip("/"))
    return ":".join(parts)


@dataclass
class RouteResult:
    success: bool
    selected_model: str | None
    backend_url: str | None
    provider_name: str | None
    error: str | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    backend_key: str | None = None
    slot_wait_timeout: float = 30.0


class RequestRouter:
    def __init__(self, config: ConfigLoader, discovery: DiscoveryManager):
        self.config = config
        self.discovery = discovery
        self.selector = ModelSelector(config, discovery)

    def _backend_url(self, model_name: str) -> str | None:
        cfg = self.config.models.models.get(model_name)
        if not cfg:
            return None
        provider = self.config.providers.get(cfg.provider)
        if not provider:
            return None
        url = provider.base_url.rstrip("/")
        if cfg.port:
            url = f"{url}:{cfg.port}"
        if cfg.path:
            url = f"{url}/{cfg.path.lstrip('/')}"
        return f"{url}/v1"

    async def route_request(
        self,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> RouteResult:
        headers = headers or {}
        privacy_str = headers.get("X-Llm-Relay-Privacy", "local_only")
        privacy = Privacy(privacy_str if privacy_str in ("local_only", "cloud_ok") else "local_only")

        weights: dict[str, float] = {}
        weights_str = headers.get("X-Llm-Relay-Weights", "")
        if weights_str:
            for pair in weights_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        weights[k.strip()] = float(v.strip())
                    except ValueError:
                        pass

        ctx = RoutingContext(
            requested_model=request_data.get("model", "") or "",
            privacy=privacy,
            weights=weights,
            require_tools=headers.get("X-Llm-Relay-Require-Tools", "false").lower() == "true",
            min_context=int(headers.get("X-Llm-Relay-Min-Context", "0") or 0) or None,
        )

        selected = self.selector.select_best(ctx)
        if not selected:
            return RouteResult(
                success=False,
                selected_model=None,
                backend_url=None,
                provider_name=None,
                error="No model matches constraints",
                decision={"requested": ctx.requested_model, "candidates": ctx.candidates, "filtered": ctx.filtered},
            )

        for candidate in ctx.ranked:
            cfg = self.config.models.models.get(candidate)
            if not cfg:
                continue
            if self.discovery.get_model_state(candidate) != ModelStatus.available:
                continue
            url = self._backend_url(candidate)
            if not url:
                continue
            backend_key = _compose_backend_key(cfg.provider, cfg.port, cfg.path or "")
            provider_cfg = self.config.providers.get(cfg.provider)
            slot_wait_timeout = provider_cfg.slot_wait_timeout if provider_cfg else 30.0
            return RouteResult(
                success=True,
                selected_model=candidate,
                backend_url=url,
                provider_name=cfg.provider,
                decision={
                    "requested": ctx.requested_model,
                    "selected": candidate,
                    "candidates": ctx.candidates,
                    "ranked": ctx.ranked[:5],
                    "privacy": ctx.privacy.value,
                },
                backend_key=backend_key,
                slot_wait_timeout=slot_wait_timeout,
            )

        return RouteResult(
            success=False,
            selected_model=selected,
            backend_url=None,
            provider_name=(self.config.models.models.get(selected).provider if selected in self.config.models.models else None),
            error="No candidate currently available",
            decision={"requested": ctx.requested_model, "ranked": ctx.ranked, "filtered": ctx.filtered},
        )

    async def forward_request(
        self,
        backend_url: str,
        model_name: str,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        backend_key: str | None = None,
        slot_wait_timeout: float = 30.0,
    ) -> httpx.Response:
        body = dict(request_data)
        body["model"] = model_name
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        # Authenticate to api-key'd upstreams using the shared homelab bearer
        # (see endpoint.py for env-var resolution). Caller-provided
        # Authorization wins so a future per-request auth path can override.
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # backend_key="" / None → acquire_slot is a no-op (no semaphore registered).
        async with self.discovery.acquire_slot(backend_key or "", wait_timeout=slot_wait_timeout):
            async with httpx.AsyncClient(timeout=300.0) as client:
                return await client.post(
                    f"{backend_url}/chat/completions",
                    json=body,
                    headers=merged_headers,
                )

    async def stream_request(
        self,
        backend_url: str,
        model_name: str,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        backend_key: str | None = None,
        slot_wait_timeout: float = 30.0,
    ) -> tuple[httpx.Response, AsyncIterator[bytes]]:
        """Open a streaming upstream connection.

        Returns the response (so the caller can read status/headers) and an
        async iterator over raw bytes. The iterator owns the client lifecycle;
        consume it to completion (or call .aclose() on it) to release the
        connection.

        The in-flight slot is held from acquire (here) until the iterator is
        exhausted or aborted.  We manually enter/exit the context manager
        rather than using ``async with`` because the slot lifetime must span
        the returned generator — a plain ``async with`` block would release
        the slot as soon as this coroutine returns.
        """
        body = dict(request_data)
        body["model"] = model_name
        # Ask the upstream to include token usage in the final SSE event. Standard
        # OpenAI streaming omits usage; this opts back in so cross-provider captures
        # (Anthropic fallback, etc.) have token counts without relying on
        # llama-server's non-standard `timings` field. Preserve any user override.
        existing_opts = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        body["stream_options"] = {"include_usage": True, **existing_opts}
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # 600s per-chunk read timeout — SSE may legitimately stall between tokens
        # on slow models within that window, but a truly dead upstream gets canceled.
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)

        # Acquire the slot BEFORE building the client so a SaturationError never
        # leaks an open connection.  The slot is released in the iterator's finally.
        slot_cm = self.discovery.acquire_slot(backend_key or "", wait_timeout=slot_wait_timeout)
        await slot_cm.__aenter__()  # may raise SaturationError; propagates to caller
        slot_released = False

        async def _release_slot() -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                await slot_cm.__aexit__(None, None, None)

        client = httpx.AsyncClient(timeout=timeout)
        try:
            req = client.build_request(
                "POST",
                f"{backend_url}/chat/completions",
                json=body,
                headers=merged_headers,
            )
            resp = await client.send(req, stream=True)
        except BaseException:
            await client.aclose()
            await _release_slot()
            raise

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
                await _release_slot()

        return resp, _iter()
