"""The single typed state object that flows through the LangGraph graph.

Using a ``TypedDict`` (rather than a free-form dict) is what lets LangGraph
validate the state and apply reducers. The ``results`` field uses an
``Annotated`` reducer so each node can append its outcome without clobbering
earlier results.
"""

from operator import add
from typing import Annotated, Optional, TypedDict

from models import RunResult, Step, StepResult


class ConductorState(TypedDict):
    """State carried between the plan, execute, check, and replan nodes.

    Attributes:
        instruction: The original plain-English goal.
        steps: The current ordered plan.
        current_index: Index of the next step to execute.
        results: Accumulated step outcomes (appended via the ``add`` reducer).
        replans_used: How many times the plan has been revised so far.
        final: The finished :class:`RunResult`, set by the finalize node.
    """

    instruction: str
    steps: list[Step]
    current_index: int
    results: Annotated[list[StepResult], add]
    replans_used: int
    final: Optional[RunResult]


def initial_state(instruction: str) -> ConductorState:
    """Build the starting state for a fresh run.

    Args:
        instruction: The user's plain-English instruction.

    Returns:
        A fully-populated initial :class:`ConductorState`.
    """
    return ConductorState(
        instruction=instruction,
        steps=[],
        current_index=0,
        results=[],
        replans_used=0,
        final=None,
    )
