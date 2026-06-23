"""HTTP auth middleware.

Not host-based: the relay typically runs behind a loopback reverse proxy, so
trusting the peer address would bypass auth for all proxied traffic. A valid key
is required on every non-exempt path when auth is enabled.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..auth import AuthError, Principal, authenticate


def install_auth_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        cfg = request.app.state.config.auth
        if not cfg.enabled:
            request.state.principal = Principal(id="anonymous")
            return await call_next(request)
        if request.url.path in cfg.exempt_paths:
            return await call_next(request)
        try:
            principal = authenticate(
                request.headers.get("authorization"),
                request.headers.get("x-api-key"),
                cfg,
            )
        except AuthError as e:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": e.reason},
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state.principal = principal
        return await call_next(request)
