"""A2A AgentExecutor that routes tasks through llm-relay's own chat endpoint."""
from __future__ import annotations

import json
import logging

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, TaskState

logger = logging.getLogger(__name__)

_STATUS_KEYWORDS = frozenset(
    ["status", "mode", "model", "models", "available", "running", "what is", "what's"]
)


class LlmRelayExecutor(AgentExecutor):
    """Execute A2A tasks by forwarding them to llm-relay's own endpoints.

    * Queries that look like status/model-list requests are answered via
      the relay's ``/status`` and ``/v1/available-models`` endpoints so the
      reply is always accurate without incurring an LLM round-trip.
    * All other messages are forwarded as chat completions to
      ``POST /v1/chat/completions`` using the ``main`` alias, which llm-relay
      resolves to the best available model at call time.
    """

    def __init__(self, base_url: str, default_alias: str = "main") -> None:
        self._base_url = base_url.rstrip("/")
        self._default_alias = default_alias

    # ------------------------------------------------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        query = (context.get_user_input() or "").strip()
        if not query:
            await updater.complete()
            return

        await event_queue.enqueue_event(
            __import__("a2a.types", fromlist=["Task"]).Task(
                id=task_id,
                context_id=context_id,
                status=__import__("a2a.types", fromlist=["TaskStatus"]).TaskStatus(
                    state=TaskState.TASK_STATE_SUBMITTED
                ),
                history=[context.message] if context.message else [],
            )
        )
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="Routing…")])
        )

        try:
            reply = await self._dispatch(query)
        except Exception as exc:
            logger.exception("LlmRelayExecutor dispatch failed: %s", exc)
            await updater.add_artifact(
                parts=[Part(text=f"Error routing request: {exc}")],
                name="error",
                last_chunk=True,
            )
            await updater.complete()
            return

        await updater.add_artifact(
            parts=[Part(text=reply)],
            name="response",
            last_chunk=True,
        )
        await updater.complete()

    # ------------------------------------------------------------------
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()

    # ------------------------------------------------------------------
    async def _dispatch(self, query: str) -> str:
        ql = query.lower()

        # Status / model-list shortcut — answer directly from relay APIs
        if any(kw in ql for kw in _STATUS_KEYWORDS) and len(query) < 200:
            return await self._handle_status_query(query)

        return await self._chat(query)

    async def _handle_status_query(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=5.0) as client:
            status_r = await client.get(f"{self._base_url}/status")
            status_r.raise_for_status()
            status = status_r.json()

        mode = ", ".join(status.get("mode", ["custom"]))
        local_models = status.get("available_local_models", [])
        aliases = status.get("aliases", {})
        main_model = aliases.get("main", "unknown")

        lines = [
            f"**llm-relay status**",
            f"Mode: {mode}",
            f"Main alias → {main_model}",
            f"Available local models: {', '.join(local_models) or 'none'}",
        ]

        # Show key aliases concisely
        key_aliases = ["main", "daily", "fast", "high-quality", "reasoning"]
        alias_lines = []
        for a in key_aliases:
            if a in aliases and aliases[a]:
                alias_lines.append(f"  {a}: {aliases[a]}")
        if alias_lines:
            lines.append("Alias resolutions:")
            lines.extend(alias_lines)

        return "\n".join(lines)

    async def _chat(self, query: str) -> str:
        payload = {
            "model": self._default_alias,
            "messages": [{"role": "user", "content": query}],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return json.dumps(data, indent=2)
