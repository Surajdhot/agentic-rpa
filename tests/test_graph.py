"""Tests for the LangGraph state machine and its conditional router.

The LLM is mocked and a fake MCP client supplies tool results, so the full
graph runs end to end without touching the network or a browser.
"""

import json

import pytest

import config
from graph import GRAPH
from models import Step, StepResult
from nodes.check_node import check_node
from state import initial_state

ALL_TOOLS = [
    {"name": name, "description": "d", "input_schema": {"properties": {}}}
    for name in config.ALLOWED_TOOLS
]


def _state(results: list, steps: list, index: int, replans: int) -> dict:
    """Build a minimal state dict for router unit tests."""
    return {
        "instruction": "g",
        "steps": steps,
        "current_index": index,
        "results": results,
        "replans_used": replans,
        "final": None,
    }


def test_check_finishes_when_all_steps_done() -> None:
    step = Step(0, "navigate", {"url": "x"}, "go")
    ok = StepResult(step, "success", "ok")
    assert check_node(_state([ok], [step], 1, 0)) == "finish"


def test_check_continues_when_steps_remain() -> None:
    a, b = Step(0, "navigate", {"url": "x"}, "go"), Step(1, "read_text", {"selector": "h1"}, "r")
    ok = StepResult(a, "success", "ok")
    assert check_node(_state([ok], [a, b], 1, 0)) == "execute"


def test_check_replans_on_failure_with_budget() -> None:
    step = Step(0, "navigate", {"url": "x"}, "go")
    bad = StepResult(step, "failed", error="boom")
    assert check_node(_state([bad], [step], 1, 0)) == "replan"


def test_check_finishes_when_replans_exhausted() -> None:
    step = Step(0, "navigate", {"url": "x"}, "go")
    bad = StepResult(step, "failed", error="boom")
    assert check_node(_state([bad], [step], 1, config.MAX_REPLANS)) == "finish"


class FailOnceClient:
    """Fake client whose read_text fails on the first call, then succeeds."""

    def __init__(self) -> None:
        self.read_attempts = 0

    async def list_tools(self) -> list[dict]:
        return ALL_TOOLS

    async def call_tool(self, name: str, _args: dict) -> str:
        if name == "read_text":
            self.read_attempts += 1
            if self.read_attempts == 1:
                raise RuntimeError("selector not found")
            return "recovered text"
        return f"ok:{name}"


class AlwaysFailClient:
    """Fake client whose read_text always fails."""

    async def list_tools(self) -> list[dict]:
        return ALL_TOOLS

    async def call_tool(self, name: str, _args: dict) -> str:
        if name == "read_text":
            raise RuntimeError("always missing")
        return f"ok:{name}"


def _patch_llm(monkeypatch: pytest.MonkeyPatch, plan: str, replan: str) -> None:
    """Patch the plan and replan nodes' LLM calls with fixed JSON outputs."""

    async def fake_plan(_prompt: str) -> str:
        return plan

    async def fake_replan(_prompt: str) -> str:
        return replan

    monkeypatch.setattr("nodes.plan_node.ainvoke_text", fake_plan)
    monkeypatch.setattr("nodes.replan_node.ainvoke_text", fake_replan)


_PLAN = json.dumps(
    [
        {"tool": "navigate", "args": {"url": "https://x"}, "description": "open"},
        {"tool": "read_text", "args": {"selector": ".title"}, "description": "read"},
    ]
)
_REPLAN = json.dumps(
    [{"tool": "read_text", "args": {"selector": "h1"}, "description": "read (retry)"}]
)


async def test_failing_step_triggers_exactly_one_replan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single failure should cause exactly one replan, then recover."""
    _patch_llm(monkeypatch, _PLAN, _REPLAN)
    cfg = {"configurable": {"mcp_client": FailOnceClient()}, "recursion_limit": 100}
    final = await GRAPH.ainvoke(initial_state("read the title"), cfg)
    assert final["replans_used"] == 1
    assert final["final"].final_status in ("success", "partial")


async def test_replans_stop_at_max_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent failures must stop at MAX_REPLANS and finish gracefully."""
    _patch_llm(monkeypatch, _PLAN, _REPLAN)
    cfg = {"configurable": {"mcp_client": AlwaysFailClient()}, "recursion_limit": 100}
    final = await GRAPH.ainvoke(initial_state("read the title"), cfg)
    assert final["replans_used"] == config.MAX_REPLANS
    assert final["final"] is not None
    assert final["final"].final_status in ("partial", "failed")
