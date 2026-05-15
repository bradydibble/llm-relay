"""Discovery module for llm-relay."""

from .endpoint import EndpointClient
from .manager import DiscoveryManager

__all__ = ["EndpointClient", "DiscoveryManager"]
