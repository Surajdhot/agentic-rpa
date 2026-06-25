"""Tests for the MCP server and the sandboxed Playwright backend.

These tests introspect the registered tools and exercise the path-traversal
guard. The Playwright backend is never started, so no real browser is opened.
"""

import pytest

import config
from mcp_server import server
from mcp_server.playwright_backend import PlaywrightBackend


def test_server_registers_exactly_the_allowlist() -> None:
    """The server must expose precisely the allow-listed tools, no more."""
    assert server.registered_tool_names() == sorted(config.ALLOWED_TOOLS)


def test_tool_schemas_declare_expected_arguments() -> None:
    """Each tool's input schema must declare the right typed arguments."""
    tools = {tool.name: tool.parameters for tool in server.mcp._tool_manager.list_tools()}
    assert "url" in tools["navigate"]["properties"]
    assert {"selector", "text"} <= set(tools["type_text"]["properties"])
    assert {"filename", "content"} <= set(tools["save_to_file"]["properties"])
    assert "name" in tools["screenshot"]["properties"]


async def test_save_to_file_rejects_path_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Writing outside the output sandbox must raise, without touching disk."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    backend = PlaywrightBackend()
    with pytest.raises(ValueError):
        await backend.save_to_file("../escape.txt", "x")
    with pytest.raises(ValueError):
        await backend.save_to_file("/etc/passwd", "x")


async def test_save_to_file_writes_inside_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A plain filename must be written inside the output sandbox."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    backend = PlaywrightBackend()
    path = await backend.save_to_file("result.txt", "hello")
    assert path == str(tmp_path / "result.txt")
    assert (tmp_path / "result.txt").read_text() == "hello"


async def test_navigate_tool_delegates_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The navigate MCP tool must delegate to the backend (which we mock)."""
    seen: dict[str, str] = {}

    async def fake_navigate(url: str) -> str:
        seen["url"] = url
        return "navigated"

    monkeypatch.setattr(server.backend, "navigate", fake_navigate)
    result = await server.navigate("https://example.com")
    assert seen["url"] == "https://example.com"
    assert result == "navigated"
