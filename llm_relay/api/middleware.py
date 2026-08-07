"""Pure-ASGI auth middleware.

Implemented as raw ASGI rather than Starlette ``BaseHTTPMiddleware`` on purpose:
``BaseHTTPMiddleware`` can buffer a ``StreamingResponse`` and interfere with a
response's ``BackgroundTask``, and this gate sits in front of the streaming
``/v1/chat/completions`` proxy whose ``BackgroundTask(cleanup)`` frees in-flight
slots. On a pass-through (auth disabled, trusted listener, exempt path, or a
valid key) it hands the untouched ASGI scope straight to the app, so the
response stream and its background task are never touched; on a failure it
short-circuits with a 401/403 before the app runs.

Trust model (see also the README security model):
- Never peer-address based: the relay typically runs behind a loopback reverse
  proxy, so trusting the peer would bypass auth for all proxied traffic.
- LISTENER based instead: requests arriving on a port listed in
  ``auth.trusted_ports`` (the local socket they connected to, from
  ``scope["server"]``) are implicitly the deployment's own local consumers.
  They are attributed to ``auth.trusted_principal`` with admin+cloud+third_party
  scopes — the deployment's own local consumers are fully privileged, including
  the ability to declare a workload non-confidential and reach hardware CIQ does
  not own. Bind trusted ports to loopback and never route external traffic to
  them: that binding IS the access control for these scopes.
- Every other listener enforces a key, and the ``admin`` scope gates
  ``/admin/*`` and ``/logs*`` (fleet-wide operator surfaces).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..audit import audit
from ..auth import AuthError, Principal, authenticate
from ..metrics import record_auth_failure


def _admin_gated(path: str) -> bool:
    return path.startswith("/admin") or path == "/logs" or path.startswith("/logs/")


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
        server = scope.get("server") or (None, None)
        if server[1] in cfg.trusted_ports:
            state["principal"] = Principal(
                id=cfg.trusted_principal,
                priority_weight=1.0,
                # `third_party` keeps on-box/tailnet agents able to reach the
                # MI300X tray as they could before the confidentiality axis
                # existed. Omitting it here would have silently revoked that
                # access for every trusted-listener consumer.
                scopes=["admin", "cloud", "third_party"],
            )
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in cfg.exempt_paths:
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
            record_auth_failure()
            audit("auth_failure", path=path, reason=e.reason)
            response = JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": e.reason},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        if _admin_gated(path) and "admin" not in principal.scopes:
            audit("scope_denied", path=path, principal=principal.id)
            response = JSONResponse(
                status_code=403,
                content={"error": "forbidden", "detail": "admin scope required"},
            )
            await response(scope, receive, send)
            return
        state["principal"] = principal
        await self.app(scope, receive, send)


def install_auth_middleware(app: FastAPI) -> None:
    app.add_middleware(AuthMiddleware)
