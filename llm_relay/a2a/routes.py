"""Build and return the A2A Starlette route lists for mounting into FastAPI."""
from __future__ import annotations

from typing import Any

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

from .card import build_agent_card
from .executor import LlmRelayExecutor


def build_a2a_routes(
    base_url: str,
    default_alias: str = "main",
) -> tuple[list[Any], list[Any], list[Any]]:
    """Return ``(card_routes, jsonrpc_routes, rest_routes)`` ready to extend a FastAPI app.

    Usage in ``create_app``::

        card_r, jsonrpc_r, rest_r = build_a2a_routes(base_url)
        app.routes.extend(card_r)
        app.routes.extend(jsonrpc_r)
        app.routes.extend(rest_r)

    Args:
        base_url: Externally-reachable root URL of this llm-relay instance.
        default_alias: llm-relay model alias used for plain chat tasks.
    """
    agent_card = build_agent_card(base_url)
    task_store = InMemoryTaskStore()
    executor = LlmRelayExecutor(base_url=base_url, default_alias=default_alias)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    card_routes = create_agent_card_routes(agent_card=agent_card)
    jsonrpc_routes = create_jsonrpc_routes(
        request_handler=request_handler,
        rpc_url="/a2a/jsonrpc",
    )
    rest_routes = create_rest_routes(
        request_handler=request_handler,
        path_prefix="/a2a/rest",
    )

    return card_routes, jsonrpc_routes, rest_routes
