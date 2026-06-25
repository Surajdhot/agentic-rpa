"""The Conductor LangGraph state machine.

This module defines the whole orchestration as a real ``StateGraph``:

    START -> plan -> execute -> [check_node router] -> execute | replan | finalize
                          ^                                  |
                          |__________________________________|
                                   replan -> execute
    finalize -> END

``check_node`` is wired as a conditional-edge router (it only reads state and
returns a branch name); it is never a state-mutating node. The graph is the
sole owner of the loop — there is no hand-written while-loop anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import config
from mcp_client import MCPClient
from models import RunResult, StepResult
from nodes.check_node import check_node
from nodes.execute_node import execute_node
from nodes.plan_node import plan_node
from nodes.replan_node import replan_node
from state import ConductorState, initial_state

logger = logging.getLogger(__name__)


def _final_status(state: ConductorState) -> str:
    """Derive the overall run status from the executed results."""
    results = state["results"]
    if not results:
        return "failed"
    completed_all = state["current_index"] >= len(state["steps"])
    if completed_all and results[-1].status != "failed":
        return "success"
    if any(result.status == "success" for result in results):
        return "partial"
    return "failed"


def _summary(state: ConductorState, status: str) -> str:
    """Build a short human-readable summary of the run."""
    results: list[StepResult] = state["results"]
    successes = sum(1 for result in results if result.status == "success")
    return (
        f"{status.upper()}: {successes}/{len(results)} executed step(s) "
        f"succeeded after {state['replans_used']} replan(s)."
    )


async def finalize_node(state: ConductorState) -> dict[str, object]:
    """Assemble the immutable :class:`RunResult` at the end of the run."""
    status = _final_status(state)
    run = RunResult(
        instruction=state["instruction"],
        steps=state["steps"],
        results=state["results"],
        final_status=status,
        summary=_summary(state, status),
    )
    logger.info("Run finished: %s", run.summary)
    return {"final": run}


def build_graph():
    """Construct and compile the Conductor state graph.

    Returns:
        The compiled LangGraph application.
    """
    builder = StateGraph(ConductorState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("replan", replan_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_conditional_edges(
        "execute",
        check_node,
        {"execute": "execute", "replan": "replan", "finish": "finalize"},
    )
    builder.add_edge("replan", "execute")
    builder.add_edge("finalize", END)
    return builder.compile()


#: The compiled graph, reused across runs.
GRAPH = build_graph()


def _run_config(client: MCPClient) -> RunnableConfig:
    """Build the runnable config that injects the MCP client into nodes."""
    return {
        "configurable": {"mcp_client": client},
        "recursion_limit": config.RECURSION_LIMIT,
    }


async def run(instruction: str) -> RunResult:
    """Execute an instruction end to end and return the final result.

    Args:
        instruction: The plain-English automation goal.

    Returns:
        The completed :class:`RunResult`.
    """
    config.validate_config()
    async with MCPClient() as client:
        final_state = await GRAPH.ainvoke(initial_state(instruction), _run_config(client))
    result = final_state.get("final")
    if result is None:
        raise RuntimeError("Graph finished without producing a RunResult.")
    return result


async def arun_events(instruction: str) -> AsyncIterator[ConductorState]:
    """Stream full state snapshots as the graph progresses, for live UIs.

    Args:
        instruction: The plain-English automation goal.

    Yields:
        The full :class:`ConductorState` after each super-step.
    """
    config.validate_config()
    async with MCPClient() as client:
        cfg = _run_config(client)
        async for snapshot in GRAPH.astream(initial_state(instruction), cfg, stream_mode="values"):
            yield snapshot
