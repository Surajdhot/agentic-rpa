"""The Conductor MCP server.

A real Model Context Protocol server built with the official FastMCP SDK. It
exposes one MCP tool per allow-listed automation action; each tool delegates to
the sandboxed :class:`PlaywrightBackend`. The agent discovers and calls these
tools over MCP and never touches Playwright directly.

Run as a standalone stdio process::

    python -m mcp_server.server
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

import config
from mcp_server.playwright_backend import PlaywrightBackend

config.configure_logging()
logger = logging.getLogger(__name__)

#: Single shared browser backend for the lifetime of the server process.
backend = PlaywrightBackend()


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Start the browser when the server boots and stop it on shutdown."""
    await backend.start()
    try:
        yield
    finally:
        await backend.stop()


mcp = FastMCP("conductor-automation", lifespan=lifespan)


@mcp.tool()
async def navigate(url: str) -> str:
    """Open a web page in the browser. Provide a full URL including scheme."""
    logger.info("AUDIT tool=navigate url=%s", url)
    return await backend.navigate(url)


@mcp.tool()
async def click(selector: str) -> str:
    """Click the first element matching a CSS selector on the current page."""
    logger.info("AUDIT tool=click selector=%s", selector)
    return await backend.click(selector)


@mcp.tool()
async def type_text(selector: str, text: str) -> str:
    """Type text into the form field matching a CSS selector."""
    logger.info("AUDIT tool=type_text selector=%s len=%d", selector, len(text))
    return await backend.type_text(selector, text)


@mcp.tool()
async def read_text(selector: str) -> str:
    """Return the visible inner text of the element matching a CSS selector."""
    logger.info("AUDIT tool=read_text selector=%s", selector)
    return await backend.read_text(selector)


@mcp.tool()
async def screenshot(name: str) -> str:
    """Save a PNG screenshot of the current page into the output sandbox."""
    logger.info("AUDIT tool=screenshot name=%s", name)
    return await backend.screenshot(name)


@mcp.tool()
async def save_to_file(filename: str, content: str) -> str:
    """Save text content to a file inside the output sandbox and return its path."""
    logger.info("AUDIT tool=save_to_file filename=%s len=%d", filename, len(content))
    return await backend.save_to_file(filename, content)


def registered_tool_names() -> list[str]:
    """Return the sorted names of every tool registered on the server.

    Exposed for tests so they can assert the allow-list is honoured without
    spawning the server process or opening a browser.
    """
    return sorted(tool.name for tool in mcp._tool_manager.list_tools())


if __name__ == "__main__":
    mcp.run()
