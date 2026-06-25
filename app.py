"""Streamlit front-end for Conductor.

Provides an instruction box and a Run button, then streams the LangGraph run
live: each step's description and outcome, inline screenshots, clearly-marked
re-planning events, and a final summary. All actions are limited to the
MCP-exposed allow-list and everything runs locally on free tooling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import streamlit as st

import config
from graph import arun_events
from state import ConductorState

_STATUS_BADGE = {"success": "✅", "partial": "🟡", "failed": "❌"}


def _render_plan(state: ConductorState, seen: dict[str, Any]) -> None:
    """Announce the plan once it has been produced."""
    if state["steps"] and not seen["planned"]:
        st.info(f"📋 Planned {len(state['steps'])} step(s).")
        seen["planned"] = True


def _render_replan(state: ConductorState, seen: dict[str, Any]) -> None:
    """Show a clear notice whenever the agent re-plans after a failure."""
    if state["replans_used"] > seen["replans"]:
        st.warning(f"⚠️ A step failed → re-planning (revision {state['replans_used']}).")
        seen["replans"] = state["replans_used"]


def _render_result_detail(result: Any) -> None:
    """Render a single step's output: screenshot, saved file, or text."""
    step = result.step
    st.caption(f"tool: `{step.tool}`  ·  args: `{step.args}`")
    if result.status != "success" or not result.output:
        return
    if step.tool == "screenshot" and Path(result.output).exists():
        st.image(result.output, caption=result.output)
    elif step.tool == "save_to_file":
        st.caption(f"💾 saved → {result.output}")
    else:
        with st.expander("output"):
            st.text(result.output[:2000])


def _render_results(state: ConductorState, seen: dict[str, Any]) -> None:
    """Render any step results that have appeared since the last snapshot."""
    results = state["results"]
    for result in results[seen["rendered"] :]:
        label = f"Step {result.step.index + 1}: {result.step.description or result.step.tool}"
        if result.status == "success":
            st.success(f"{label} — done")
        else:
            st.error(f"{label} — failed: {result.error}")
        _render_result_detail(result)
    seen["rendered"] = len(results)


def _render_final(state: ConductorState) -> None:
    """Render the overall status and summary once the run finishes."""
    final = state.get("final")
    if final is None:
        return
    badge = _STATUS_BADGE.get(final.final_status, "")
    st.subheader(f"{badge} {final.final_status.upper()}")
    st.write(final.summary)


async def _stream_run(instruction: str) -> None:
    """Drive the graph and render each live snapshot to the page."""
    seen: dict[str, Any] = {"planned": False, "replans": 0, "rendered": 0}
    async for state in arun_events(instruction):
        _render_plan(state, seen)
        _render_replan(state, seen)
        _render_results(state, seen)
        _render_final(state)


def _sidebar() -> None:
    """Render the informational sidebar."""
    st.sidebar.header("About")
    st.sidebar.markdown(
        "**Conductor** plans automation with LangGraph and acts only through "
        "tools exposed over **MCP**. It runs locally on free tooling "
        "(Google Gemini + Playwright)."
    )
    st.sidebar.subheader("Allowed actions")
    st.sidebar.write(", ".join(config.ALLOWED_TOOLS))


def main() -> None:
    """Render the Conductor UI and handle a run request."""
    st.set_page_config(page_title="Conductor", page_icon="🎼")
    st.title("🎼 Conductor")
    st.caption("AI-powered RPA: plain English → planned, MCP-driven automation.")
    _sidebar()

    instruction = st.text_area(
        "Instruction",
        placeholder="e.g. Go to example.com, read the page heading, and save it to result.txt",
        height=110,
    )
    if not st.button("Run", type="primary"):
        return
    if not instruction.strip():
        st.warning("Please enter an instruction first.")
        return
    try:
        config.validate_config()
    except EnvironmentError as exc:
        st.error(str(exc))
        return
    with st.spinner("Running…"):
        asyncio.run(_stream_run(instruction.strip()))


if __name__ == "__main__":
    config.configure_logging()
    main()
