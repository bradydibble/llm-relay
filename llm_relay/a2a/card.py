"""Build the A2A AgentCard for llm-relay."""
from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)


def build_agent_card(base_url: str) -> AgentCard:
    """Return the AgentCard that describes this llm-relay instance.

    ``base_url`` should be the externally-reachable root URL of the relay
    (e.g. ``http://127.0.0.1:8090`` for local use, or a Tailscale address
    for cross-agent use).  It must *not* have a trailing slash.
    """
    return AgentCard(
        name="llm-relay",
        description=(
            "LLM routing control plane.  Routes chat-completion requests to "
            "the best available local or cloud model based on privacy "
            "constraints, context requirements, and backend health.  Agents "
            "can ask it to complete a task, list available models, or report "
            "the current routing mode."
        ),
        provider=AgentProvider(
            organization="llm-relay",
            url=base_url,
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="route_chat",
                name="Route chat completion",
                description=(
                    "Accept a natural-language message and route it to the "
                    "best available model.  Respects privacy, context-window, "
                    "and capability requirements supplied via the task metadata."
                ),
                tags=["llm", "chat", "routing"],
                examples=[
                    "Summarise this document in three bullet points.",
                    "Write a Python function that reverses a string.",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="list_models",
                name="List available models",
                description=(
                    "Return the current set of configured models, their "
                    "availability status, context windows, and which routing "
                    "aliases they belong to."
                ),
                tags=["llm", "routing", "status"],
                examples=[
                    "What models are available?",
                    "Which model handles the 'main' alias right now?",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="relay_status",
                name="Relay routing status",
                description=(
                    "Report the active llm-mode preset (e.g. qwen36, "
                    "large-context, big), which backends are healthy, and "
                    "what each routing alias currently resolves to."
                ),
                tags=["llm", "routing", "status"],
                examples=[
                    "What mode is llm-relay in?",
                    "Is the 35B model running?",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"{base_url}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"{base_url}/a2a/rest",
            ),
        ],
    )
