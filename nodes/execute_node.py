"""The execute node: run the current step by calling its MCP tool."""

import logging

from langchain_core.runnables import RunnableConfig

from models import StepResult
from nodes.utils import client_from_config
from state import ConductorState

logger = logging.getLogger(__name__)


async def execute_node(state: ConductorState, config: RunnableConfig) -> dict[str, object]:
    """Execute the step at ``current_index`` via the MCP client.

    The step is logged before execution for auditing. Success and failure are
    both recorded as a :class:`StepResult`; failures do not raise so the graph
    can decide whether to replan.

    Args:
        state: The current graph state.
        config: LangGraph runnable config carrying the MCP client.

    Returns:
        A partial state update appending one result and advancing the cursor.
    """
    step = state["steps"][state["current_index"]]
    logger.info("AUDIT execute step=%d tool=%s args=%s", step.index, step.tool, step.args)
    client = client_from_config(config)
    try:
        output = await client.call_tool(step.tool, step.args)
        result = StepResult(step=step, status="success", output=output)
    except Exception as exc:  # noqa: BLE001 - failures are recorded, not raised
        logger.warning("Step %d (%s) failed: %s", step.index, step.tool, exc)
        result = StepResult(step=step, status="failed", error=str(exc))
    return {"results": [result], "current_index": state["current_index"] + 1}
