"""Pure-ASGI auth middleware.

Implemented as raw ASGI rather than Starlette ``BaseHTTPMiddleware`` on purpose:
``BaseHTTPMiddleware`` can buffer a ``StreamingResponse`` and interfere with a
response's ``BackgroundTask``, and this gate sits in front of the streaming
``/v1/chat/completions`` proxy whose ``BackgroundTask(cleanup)`` frees in-flight
slots. On a pass-through (auth disabled, exempt path, or a valid key) it hands
the untouched ASGI scope straight to the app, so the response stream and its
background task are never touched; on a failure it short-circuits with a 401
before the app runs.

Not host-based: the relay typically runs behind a loopback reverse proxy, so
trusting the peer address would bypass auth for all proxied traffic.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..auth import AuthError, Principal, authenticate


class AuthMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cfg = scope["app"].state.config.auth
        state = scope.setdefault("state", {})
        if not cfg.enabled:
            state["principal"] = Principal(id="anonymous")
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in cfg.exempt_paths:
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        try:
            principal = authenticate(
                headers.get("authorization"),
                headers.get("x-api-key"),
                cfg,
            )
        except AuthError as e:
            response = JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": e.reason},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        state["principal"] = principal
        await self.app(scope, receive, send)


def install_auth_middleware(app: FastAPI) -> None:
    app.add_middleware(AuthMiddleware)
