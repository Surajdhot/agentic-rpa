"""Tests for the plan node.

The LLM is mocked and the MCP tool list is provided by a fake client, so the
planner is exercised without any network or browser access.
"""

import json

import pytest

import config
from nodes.plan_node import plan_node


class FakeClient:
    """A stand-in MCP client exposing the allow-listed tools."""

    async def list_tools(self) -> list[dict]:
        """Return schemas for every allow-listed tool."""
        return [
            {"name": name, "description": "d", "input_schema": {"properties": {}}}
            for name in config.ALLOWED_TOOLS
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        """Unused by the planner; present for interface completeness."""
        return "ok"


def _config(client: FakeClient) -> dict:
    """Build a runnable config injecting the fake MCP client."""
    return {"configurable": {"mcp_client": client}}


async def test_plan_keeps_only_allowlisted_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Steps using non-allow-listed tools must be dropped."""
    raw = json.dumps(
        [
            {"tool": "navigate", "args": {"url": "https://x"}, "description": "open"},
            {"tool": "run_shell", "args": {"cmd": "rm -rf /"}, "description": "evil"},
            {"tool": "read_text", "args": {"selector": "h1"}, "description": "read"},
        ]
    )

    async def fake_llm(_prompt: str) -> str:
        return raw

    monkeypatch.setattr("nodes.plan_node.ainvoke_text", fake_llm)
    out = await plan_node({"instruction": "do a thing"}, _config(FakeClient()))

    tools = [step.tool for step in out["steps"]]
    assert tools == ["navigate", "read_text"]
    assert all(tool in config.ALLOWED_TOOLS for tool in tools)
    assert out["current_index"] == 0


async def test_plan_respects_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan must be capped at MAX_STEPS even if the LLM returns more."""
    raw = json.dumps(
        [
            {"tool": "navigate", "args": {"url": "https://x"}, "description": str(i)}
            for i in range(config.MAX_STEPS + 5)
        ]
    )

    async def fake_llm(_prompt: str) -> str:
        return raw

    monkeypatch.setattr("nodes.plan_node.ainvoke_text", fake_llm)
    out = await plan_node({"instruction": "do a thing"}, _config(FakeClient()))

    assert len(out["steps"]) == config.MAX_STEPS
