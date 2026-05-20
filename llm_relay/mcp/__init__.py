"""MCP (Model Context Protocol) server for llm-relay."""
from .server import build_mcp_app, build_mcp_server

__all__ = ["build_mcp_app", "build_mcp_server"]
