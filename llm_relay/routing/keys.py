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

Model identity
--------------
``compose_model_id`` / ``resolve_model_id`` handle the host-qualified model id
``<provider>:<model>`` (distinct from the backend key above, which also encodes
port/path). Kept here so the /v1/models surface and the routing resolver share
one definition of the separator and the parse rule.
"""
from __future__ import annotations

from typing import Mapping

from ..config.types import ModelConfig


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


def compose_model_id(provider_name: str, model: str) -> str:
    """Canonical host-qualified model id: ``<provider>:<model>``.

    Single source of the provider/model separator so the ``/v1/models`` surface
    and the routing resolver can never drift.
    """
    return f"{provider_name}:{model}"


def resolve_model_id(models: Mapping[str, ModelConfig], requested: str) -> str | None:
    """Map a requested model id to a bare config model name, or ``None``.

    Accepts either a bare name (returned as-is when configured) or a
    host-qualified ``provider:model`` id (returned as the bare ``model`` only
    when ``model`` is configured AND served by ``provider``). A qualified id
    whose pair doesn't match a configured model, or any unknown name, returns
    ``None``. A bare name is matched first, so a model whose name happens to
    contain ``:`` still resolves.
    """
    if requested in models:
        return requested
    provider, sep, name = requested.partition(":")
    if sep:
        m = models.get(name)
        if m is not None and m.provider == provider:
            return name
    return None
