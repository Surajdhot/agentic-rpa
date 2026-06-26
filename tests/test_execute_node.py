"""Tests for the execute node and step-to-step data flow.

The MCP client is a capturing fake, so no browser is opened. These tests focus
on the ``{{...output}}`` substitution that lets one step reuse an earlier
step's result.
"""

from models import Step, StepResult
from nodes.execute_node import execute_node
from nodes.utils import resolve_args


def _success(index: int, tool: str, output: str) -> StepResult:
    """Build a successful StepResult for the given step index."""
    return StepResult(Step(index, tool, {}, "d"), "success", output)


def test_resolve_args_substitutes_last_output() -> None:
    """``{{last_output}}`` must become the previous step's output."""
    results = [_success(0, "read_text", "HELLO")]
    out = resolve_args({"filename": "f.txt", "content": "{{last_output}}"}, results)
    assert out == {"filename": "f.txt", "content": "HELLO"}


def test_resolve_args_substitutes_by_step_index() -> None:
    """``{{step_N_output}}`` must resolve the step at index N."""
    results = [_success(0, "read_text", "A"), _success(1, "read_text", "B")]
    assert resolve_args({"content": "{{step_0_output}}"}, results)["content"] == "A"


def test_resolve_args_unknown_reference_is_empty() -> None:
    """A reference with no matching result resolves to an empty string."""
    assert resolve_args({"content": "{{last_output}}"}, [])["content"] == ""


class CaptureClient:
    """Fake MCP client that records the arguments it is called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[dict]:
        return []

    async def call_tool(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return "saved"


async def test_execute_resolves_references_before_calling_tool() -> None:
    """execute_node must pass resolved (not templated) args to the tool."""
    prior = [_success(0, "read_text", "PAGE TEXT")]
    step = Step(1, "save_to_file", {"filename": "out.txt", "content": "{{last_output}}"}, "save")
    state = {
        "instruction": "i",
        "steps": [Step(0, "read_text", {}, "r"), step],
        "current_index": 1,
        "results": prior,
        "replans_used": 0,
        "final": None,
    }
    client = CaptureClient()
    update = await execute_node(state, {"configurable": {"mcp_client": client}})

    assert client.calls == [("save_to_file", {"filename": "out.txt", "content": "PAGE TEXT"})]
    assert update["results"][0].status == "success"
    assert update["current_index"] == 2
