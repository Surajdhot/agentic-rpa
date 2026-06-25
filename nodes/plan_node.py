"""The plan node: turn the instruction into an ordered list of steps."""

import logging

from langchain_core.runnables import RunnableConfig

from config import MAX_STEPS
from llm import ainvoke_text
from nodes.utils import client_from_config, extract_json_array, format_tools, render_prompt, to_steps
from state import ConductorState

logger = logging.getLogger(__name__)


async def plan_node(state: ConductorState, config: RunnableConfig) -> dict[str, object]:
    """Produce the initial plan using the LLM and the live MCP tool list.

    The planner is given the real tool schemas discovered over MCP, so it can
    only choose from the allow-listed actions. The result is written to state
    with the cursor reset to the first step.

    Args:
        state: The current graph state (must contain ``instruction``).
        config: LangGraph runnable config carrying the MCP client.

    Returns:
        A partial state update with the planned steps and ``current_index`` 0.
    """
    client = client_from_config(config)
    tools = await client.list_tools()
    prompt = render_prompt(
        "plan.txt",
        instruction=state["instruction"],
        tools=format_tools(tools),
    )
    raw = await ainvoke_text(prompt)
    steps = to_steps(extract_json_array(raw), start_index=0, max_new=MAX_STEPS)
    logger.info("Planned %d step(s) for instruction", len(steps))
    return {"steps": steps, "current_index": 0}
