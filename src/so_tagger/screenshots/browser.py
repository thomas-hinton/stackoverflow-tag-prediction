"""Drive headless Chromium through Playwright to take each capture.

Every capture gets a fresh browser context with deterministic settings: its fixed viewport,
device scale factor 1, light colour scheme, reduced motion, an English locale and the UTC time
zone. Animations and the text caret are disabled while taking the screenshot.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from so_tagger.screenshots.captures import Capture, Click, Fill, WaitFor

if TYPE_CHECKING:
    from playwright.sync_api import Browser

NAVIGATION_TIMEOUT_MS = 30_000
INSTALL_HINT = "uv sync --group docs && uv run --group docs playwright install chromium"


class BrowserError(RuntimeError):
    """Playwright or Chromium is missing, or a page could not be captured."""


def take_screenshots(
    captures: Sequence[Capture], base_urls: Mapping[str, str], output_dir: Path
) -> list[Path]:
    """Capture each page into ``output_dir/<name>.png`` and return the written files."""
    try:
        from playwright.sync_api import Error, sync_playwright
    except ModuleNotFoundError as error:
        raise BrowserError(f"Playwright is not installed. Run: {INSTALL_HINT}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Error as error:
            raise BrowserError(
                f"Could not launch Chromium ({error.message.splitlines()[0]}). Run: {INSTALL_HINT}"
            ) from error
        try:
            for capture in captures:
                path = output_dir / capture.file_name
                try:
                    _take_screenshot(browser, capture, capture.resolve_url(base_urls), path)
                except Error as error:
                    raise BrowserError(
                        f"Capture '{capture.name}' failed: {error.message.splitlines()[0]}"
                    ) from error
                written.append(path)
        finally:
            browser.close()
    return written


def _take_screenshot(browser: "Browser", capture: Capture, url: str, path: Path) -> None:
    context = browser.new_context(
        viewport={"width": capture.viewport.width, "height": capture.viewport.height},
        device_scale_factor=1,
        color_scheme="light",
        reduced_motion="reduce",
        locale="en-US",
        timezone_id="UTC",
    )
    try:
        context.set_default_timeout(NAVIGATION_TIMEOUT_MS)
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        for action in capture.actions:
            match action:
                case Fill(selector=selector, value=value):
                    page.fill(selector, value)
                case Click(selector=selector):
                    page.click(selector)
                case WaitFor(selector=selector):
                    page.wait_for_selector(selector, state="visible")
        page.wait_for_load_state("networkidle")
        target = page.locator(capture.selector) if capture.selector else page
        target.screenshot(path=path, animations="disabled", caret="hide")
    finally:
        context.close()
