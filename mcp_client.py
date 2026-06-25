"""The agent's bridge to the MCP server.

The LangGraph nodes reach the outside world exclusively through this client:
it spawns the MCP server as a subprocess over stdio, lists the available tools,
and forwards tool calls. Nothing here imports Playwright — that lives behind
the MCP boundary.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

#: Project root, used as the working directory for the spawned server.
PROJECT_ROOT = Path(__file__).resolve().parent


class MCPToolError(RuntimeError):
    """Raised when an MCP tool reports an execution error."""


def _content_to_text(content: Sequence[Any]) -> str:
    """Flatten MCP content blocks into a single text string."""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(parts).strip()


class MCPClient:
    """An async MCP client connected to the Conductor automation server."""

    def __init__(self, command: str | None = None, args: list[str] | None = None) -> None:
        """Configure how the MCP server subprocess will be launched."""
        self._params = StdioServerParameters(
            command=command or sys.executable,
            args=args or ["-m", "mcp_server.server"],
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """Spawn the server and open an initialised MCP session."""
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Connected to MCP server via stdio")

    async def aclose(self) -> None:
        """Close the session and terminate the server subprocess."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        """Return the active session or raise if not connected."""
        if self._session is None:
            raise RuntimeError("MCP client is not connected; call connect() first.")
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the schema (name, description, input schema) of every tool."""
        result = await self._require_session().list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Call an MCP tool and return its text result.

        Args:
            name: The tool name (must exist on the server).
            args: Keyword arguments for the tool.

        Returns:
            The tool's text output.

        Raises:
            MCPToolError: if the tool reports an error.
        """
        result = await self._require_session().call_tool(name, args or {})
        text = _content_to_text(result.content)
        if result.isError:
            raise MCPToolError(text or f"Tool {name!r} failed with no message")
        return text

    async def __aenter__(self) -> "MCPClient":
        """Connect on entering an ``async with`` block."""
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Disconnect on leaving an ``async with`` block."""
        await self.aclose()
