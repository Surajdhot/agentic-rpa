"""The replan node: revise the remaining plan after a step fails."""

import logging

from langchain_core.runnables import RunnableConfig

from config import MAX_STEPS
from llm import ainvoke_text
from models import StepResult
from nodes.utils import client_from_config, extract_json_array, format_tools, render_prompt, to_steps
from state import ConductorState

logger = logging.getLogger(__name__)


def _summarise_results(results: list[StepResult]) -> str:
    """Render completed step outcomes as a compact, prompt-friendly list."""
    if not results:
        return "(none)"
    lines = [
        f"{r.step.index}. [{r.status}] {r.step.description or r.step.tool}"
        for r in results
    ]
    return "\n".join(lines)


async def replan_node(state: ConductorState, config: RunnableConfig) -> dict[str, object]:
    """Replace the failed step and everything after it with a revised plan.

    Steps that already succeeded are kept; the failed step onward is regenerated
    by the LLM. The cursor is moved back to the first revised step and the
    replan counter is incremented.

    Args:
        state: The current graph state (last result must be a failure).
        config: LangGraph runnable config carrying the MCP client.

    Returns:
        A partial state update with revised steps, cursor, and replan count.
    """
    failed_index = state["current_index"] - 1
    kept = state["steps"][:failed_index]
    failure = state["results"][-1]
    client = client_from_config(config)
    prompt = render_prompt(
        "replan.txt",
        instruction=state["instruction"],
        tools=format_tools(await client.list_tools()),
        completed=_summarise_results(state["results"][:failed_index]),
        failed_step=failure.step.description or failure.step.tool,
        error=failure.error or "unknown error",
    )
    raw = await ainvoke_text(prompt)
    budget = MAX_STEPS - len(kept)
    revised = to_steps(extract_json_array(raw), start_index=failed_index, max_new=budget)
    logger.info("Replanned: kept %d, revised %d step(s)", len(kept), len(revised))
    return {
        "steps": kept + revised,
        "current_index": failed_index,
        "replans_used": state["replans_used"] + 1,
    }
