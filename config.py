"""Central configuration for Conductor.

All tunable constants, the tool allow-list, and environment-driven settings
live here so the rest of the codebase imports a single source of truth.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LLM settings -----------------------------------------------------------
MODEL: str = os.getenv("MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = 0.0
LLM_MAX_RETRIES: int = 3
LLM_BACKOFF_SECONDS: tuple[int, ...] = (4, 8, 16)
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

# --- Orchestration limits ---------------------------------------------------
MAX_STEPS: int = 15
MAX_REPLANS: int = 2
# Generous ceiling so the execute/replan loop never trips LangGraph's guard.
RECURSION_LIMIT: int = 2 * MAX_STEPS * (MAX_REPLANS + 1) + 10

# --- Filesystem sandbox -----------------------------------------------------
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()

# --- Safety allow-list ------------------------------------------------------
# The MCP server exposes ONLY these actions; the planner may use no others.
ALLOWED_TOOLS: tuple[str, ...] = (
    "navigate",
    "click",
    "type_text",
    "read_text",
    "screenshot",
    "save_to_file",
)

# --- Logging ----------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def configure_logging() -> None:
    """Initialise root logging once, using the configured level."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def validate_config() -> None:
    """Fail fast with a clear message if required settings are missing.

    Raises:
        EnvironmentError: when GOOGLE_API_KEY is not set.
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a "
            "free Google AI Studio key (https://aistudio.google.com/apikey)."
        )


def ensure_output_dir() -> Path:
    """Create the sandboxed output directory if needed and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
