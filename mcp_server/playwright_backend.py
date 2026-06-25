"""The Playwright automation backend.

This is the ONLY module in Conductor that imports Playwright. Every browser
action is a small async method here; the MCP server in :mod:`mcp_server.server`
wraps these methods as MCP tools. File-producing actions are sandboxed to
``OUTPUT_DIR`` and reject any path that escapes it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import Browser, Page, Playwright, async_playwright

import config

logger = logging.getLogger(__name__)

#: Default per-action timeout in milliseconds.
ACTION_TIMEOUT_MS: int = 15_000


class PlaywrightBackend:
    """A headless-Chromium wrapper exposing a fixed set of safe actions."""

    def __init__(self) -> None:
        """Create an unstarted backend; call :meth:`start` before use."""
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def start(self) -> None:
        """Launch headless Chromium and open a single page."""
        if self._page is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(ACTION_TIMEOUT_MS)
        logger.info("Playwright backend started (headless chromium)")

    async def stop(self) -> None:
        """Close the page, browser, and Playwright driver."""
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None
        logger.info("Playwright backend stopped")

    def _require_page(self) -> Page:
        """Return the active page or raise if the backend is not started."""
        if self._page is None:
            raise RuntimeError("Backend not started; call start() first.")
        return self._page

    def _resolve_output_path(self, name: str) -> Path:
        """Resolve ``name`` inside OUTPUT_DIR, rejecting path traversal.

        Args:
            name: A relative filename supplied by the agent.

        Returns:
            The absolute, sandboxed path to write to.

        Raises:
            ValueError: if the resolved path escapes OUTPUT_DIR.
        """
        base = config.ensure_output_dir()
        candidate = (base / name).resolve()
        if not candidate.is_relative_to(base):
            raise ValueError(f"Path {name!r} escapes the output sandbox.")
        return candidate

    async def navigate(self, url: str) -> str:
        """Navigate the page to ``url`` and report the resulting title."""
        page = self._require_page()
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url} (title: {await page.title()!r})"

    async def click(self, selector: str) -> str:
        """Click the first element matching ``selector``."""
        page = self._require_page()
        await page.click(selector)
        return f"Clicked element {selector!r}"

    async def type_text(self, selector: str, text: str) -> str:
        """Fill the element matching ``selector`` with ``text``."""
        page = self._require_page()
        await page.fill(selector, text)
        return f"Typed {len(text)} characters into {selector!r}"

    async def read_text(self, selector: str) -> str:
        """Return the visible inner text of the matched element."""
        page = self._require_page()
        return await page.inner_text(selector)

    async def screenshot(self, name: str) -> str:
        """Capture a PNG screenshot into OUTPUT_DIR and return its path."""
        page = self._require_page()
        filename = name if name.lower().endswith(".png") else f"{name}.png"
        path = self._resolve_output_path(filename)
        await page.screenshot(path=str(path))
        return str(path)

    async def save_to_file(self, filename: str, content: str) -> str:
        """Write ``content`` to a sandboxed file and return its path."""
        path = self._resolve_output_path(filename)
        path.write_text(content, encoding="utf-8")
        return str(path)
