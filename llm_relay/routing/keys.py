"""Shared backend-key and backend-URL helpers.

Both ``selector.py`` and ``router.py`` need to produce a backend key (used to
look up a registered slot semaphore) and a backend URL.  This module is the
single source of truth so the two call sites can never drift apart.

Key format
----------
``<provider_name>[:<port>][:<path_without_leading_slash>]``

Examples::

    local-llm
    local-llm:8080
    local-llm:8080:v2

URL format
----------
``<provider.base_url>[:<port>][/<path>]/v1``
"""
from __future__ import annotations


def compose_backend_key(provider_name: str, port: int | None, path: str) -> str:
    """Build the discovery-client key for a (provider, port, path) triple.

    Matches the key format used by ``create_app`` when calling
    ``register_backend``.  No port/path → just provider_name; port/path
    components are appended with ':' as separator.
    """
    parts = [provider_name]
    if port:
        parts.append(str(port))
    if path:
        parts.append(path.strip("/"))
    return ":".join(parts)


def compose_backend_url(base_url: str, port: int | None, path: str | None) -> str:
    """Build the full backend URL (up to and including the /v1 prefix).

    Appends ``:<port>`` and ``/<path>`` to *base_url* when present, then
    appends ``/v1`` so callers only need to add the endpoint suffix.
    """
    url = base_url.rstrip("/")
    if port:
        url = f"{url}:{port}"
    if path:
        url = f"{url}/{path.lstrip('/')}"
    return f"{url}/v1"
