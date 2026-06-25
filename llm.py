"""The single LLM factory and access point for Conductor.

Everything that needs the language model goes through :func:`get_llm` and
:func:`ainvoke_text`. This is the one place the provider (Google Gemini) is
named, so swapping providers means editing only this file.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached, low-temperature Gemini chat model.

    Returns:
        The configured :class:`ChatGoogleGenerativeAI` instance.

    Raises:
        EnvironmentError: if no Google API key is configured.
    """
    config.validate_config()
    return ChatGoogleGenerativeAI(
        model=config.MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )


def _is_rate_limited(exc: Exception) -> bool:
    """Heuristically detect a Gemini 429 / quota-exhausted error."""
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


def _as_text(response: BaseMessage | str) -> str:
    """Normalise an LLM response into plain text."""
    if isinstance(response, str):
        return response
    content = response.content
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else str(part) for part in content)
    return str(content)


async def ainvoke_text(prompt: str) -> str:
    """Invoke the LLM and return its text, retrying on rate limits.

    Retries up to ``LLM_MAX_RETRIES`` times with the configured backoff
    schedule (4s, 8s, 16s) whenever a 429/quota error is seen.

    Args:
        prompt: The fully-rendered prompt to send to the model.

    Returns:
        The model's response as plain text.

    Raises:
        Exception: the last error if every attempt fails.
    """
    llm = get_llm()
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            return _as_text(await llm.ainvoke(prompt))
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            last_attempt = attempt == config.LLM_MAX_RETRIES
            if last_attempt or not _is_rate_limited(exc):
                raise
            delay = config.LLM_BACKOFF_SECONDS[attempt]
            logger.warning("Rate limited; retrying in %ss (attempt %s)", delay, attempt + 1)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: ainvoke_text retry loop exited")
