"""Typed data structures passed between the agent, graph, and UI.

These dataclasses are deliberately serialisable plain-old-data so they can be
stored in LangGraph state and rendered by Streamlit without coupling to the
LLM or MCP layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import ALLOWED_TOOLS

#: Allowed lifecycle states for an executed step.
STEP_STATUSES: frozenset[str] = frozenset({"success", "failed", "skipped"})

#: Allowed overall outcomes for a run.
RUN_STATUSES: frozenset[str] = frozenset({"success", "failed", "partial"})


@dataclass
class Step:
    """A single planned automation action.

    Attributes:
        index: Position of the step within the ordered plan.
        tool: Name of the MCP tool to call; must be allow-listed.
        args: Keyword arguments forwarded to the tool.
        description: Human-readable explanation of the step's intent.
    """

    index: int
    tool: str
    args: dict[str, object]
    description: str

    def __post_init__(self) -> None:
        """Reject any tool outside the safety allow-list."""
        if self.tool not in ALLOWED_TOOLS:
            raise ValueError(
                f"Tool {self.tool!r} is not allow-listed; "
                f"allowed tools are: {', '.join(ALLOWED_TOOLS)}"
            )


@dataclass
class StepResult:
    """The outcome of executing a single :class:`Step`.

    Attributes:
        step: The step that was executed.
        status: One of ``success``, ``failed`` or ``skipped``.
        output: Text returned by the MCP tool (may be a file path).
        error: Error message when the step failed, otherwise ``None``.
    """

    step: Step
    status: str
    output: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate that ``status`` is a recognised value."""
        if self.status not in STEP_STATUSES:
            raise ValueError(
                f"Status {self.status!r} is invalid; "
                f"must be one of: {', '.join(sorted(STEP_STATUSES))}"
            )


@dataclass
class RunResult:
    """The full record of a single Conductor run.

    Attributes:
        instruction: The original plain-English instruction.
        steps: The final plan that was executed.
        results: The outcome of each executed step, in order.
        final_status: Overall outcome (``success``/``failed``/``partial``).
        summary: Short human-readable summary of what happened.
    """

    instruction: str
    steps: list[Step] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    final_status: str = "partial"
    summary: str = ""

    def __post_init__(self) -> None:
        """Validate that ``final_status`` is a recognised value."""
        if self.final_status not in RUN_STATUSES:
            raise ValueError(
                f"Run status {self.final_status!r} is invalid; "
                f"must be one of: {', '.join(sorted(RUN_STATUSES))}"
            )
