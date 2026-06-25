"""The check router: decide where the graph goes after each executed step.

This is wired as a LangGraph conditional-edge function, not as a node. It is
pure: it only reads state and returns the name of the next branch.
"""

from __future__ import annotations

import logging
from typing import Literal

import config
from state import ConductorState

logger = logging.getLogger(__name__)

#: The branches this router can select.
Route = Literal["execute", "replan", "finish"]


def check_node(state: ConductorState) -> Route:
    """Route to the next graph branch based on the latest result.

    Decision order:
        * If the last step failed and replans remain -> ``replan``.
        * If the last step failed and replans are exhausted -> ``finish``.
        * If every planned step has run -> ``finish``.
        * Otherwise -> ``execute`` the next step.

    Args:
        state: The current graph state.

    Returns:
        The name of the next branch.
    """
    results = state["results"]
    if results and results[-1].status == "failed":
        if state["replans_used"] < config.MAX_REPLANS:
            logger.info("Last step failed; routing to replan")
            return "replan"
        logger.info("Last step failed and replans exhausted; finishing")
        return "finish"
    if state["current_index"] >= len(state["steps"]):
        logger.info("All steps complete; finishing")
        return "finish"
    return "execute"
