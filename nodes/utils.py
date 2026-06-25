"""Shared helpers used by the planning nodes.

Kept in one place so the plan and replan nodes do not duplicate prompt
loading, tool formatting, JSON parsing, or dependency access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

import config
from mcp_client import MCPClient
from models import Step

logger = logging.getLogger(__name__)

#: Directory holding the .txt prompt templates.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def render_prompt(filename: str, **values: str) -> str:
    """Load a prompt template and substitute ``{key}`` placeholders.

    Uses plain string replacement (not ``str.format``) so JSON braces inside
    the template are left untouched.

    Args:
        filename: Template filename within the prompts directory.
        **values: Replacement values keyed by placeholder name.

    Returns:
        The rendered prompt text.
    """
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def format_tools(tools: list[dict[str, Any]]) -> str:
    """Render MCP tool schemas as a readable bullet list for the prompt."""
    lines = []
    for tool in tools:
        schema = tool.get("input_schema", {}) or {}
        params = ", ".join((schema.get("properties") or {}).keys()) or "none"
        lines.append(f"- {tool['name']}({params}): {tool['description']}")
    return "\n".join(lines)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from raw LLM text, tolerating markdown fences.

    Args:
        text: The raw model response.

    Returns:
        The parsed list of step dictionaries.

    Raises:
        ValueError: if no JSON array can be found or parsed.
    """
    cleaned = text.replace("```json", "```").replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in model output: {text[:200]!r}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Model output was not a JSON array of steps.")
    return parsed


def to_steps(items: list[dict[str, Any]], start_index: int, max_new: int) -> list[Step]:
    """Convert raw step dicts into validated, re-indexed :class:`Step` objects.

    Steps whose tool is not allow-listed are dropped, and at most ``max_new``
    steps are produced.

    Args:
        items: Raw step dictionaries from the LLM.
        start_index: Index to assign to the first produced step.
        max_new: Maximum number of steps to keep.

    Returns:
        The validated list of steps.
    """
    steps: list[Step] = []
    for raw in items:
        tool = str(raw.get("tool", ""))
        if tool not in config.ALLOWED_TOOLS:
            logger.warning("Dropping step with non-allow-listed tool %r", tool)
            continue
        steps.append(
            Step(
                index=start_index + len(steps),
                tool=tool,
                args=dict(raw.get("args") or {}),
                description=str(raw.get("description", "")),
            )
        )
        if len(steps) >= max(max_new, 1):
            break
    return steps


def client_from_config(runnable_config: RunnableConfig) -> MCPClient:
    """Pull the MCP client injected into the LangGraph runnable config.

    Raises:
        RuntimeError: if no client was provided.
    """
    client = (runnable_config.get("configurable") or {}).get("mcp_client")
    if not isinstance(client, MCPClient):
        raise RuntimeError("No MCP client found in runnable config.")
    return client
