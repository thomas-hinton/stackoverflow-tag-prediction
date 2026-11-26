"""The capture list: which pages `docs-screenshots` captures, and how.

The list is versioned in ``docs/screenshots.toml``, one ``[[capture]]`` table per image. A
capture targets a page served by one of the project's servers (a path such as ``/docs``) or a
public page (an absolute ``https://`` URL), at a fixed viewport, optionally after a few actions
and optionally cropped to one element.
"""

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DEFAULT_CAPTURES_PATH = Path("docs/screenshots.toml")

SERVERS: Final = ("api",)
"""Servers a capture can target by path; `docs-screenshots` starts each on a fixture Tagger.
The demo (#15) joins this list, and ``cli.SERVER_APPS``, once it exists."""

DEFAULT_SERVER: Final = "api"
DEFAULT_VIEWPORT_WIDTH: Final = 1280
DEFAULT_VIEWPORT_HEIGHT: Final = 800

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
_ABSOLUTE_URL_PATTERN = re.compile(r"https?://\S+")
_FIELDS = frozenset({"name", "url", "server", "viewport", "selector", "actions"})


class CaptureError(ValueError):
    """The capture list is missing, malformed or holds an invalid capture."""


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Fill:
    """Type ``value`` into the field matching ``selector``."""

    selector: str
    value: str


@dataclass(frozen=True, slots=True)
class Click:
    """Click the element matching ``selector``."""

    selector: str


@dataclass(frozen=True, slots=True)
class WaitFor:
    """Wait until an element matching ``selector`` is visible."""

    selector: str


Action = Fill | Click | WaitFor
_ACTION_KINDS: Final = ("fill", "click", "wait_for")


@dataclass(frozen=True, slots=True)
class Capture:
    """One screenshot: the page to open, what to do on it, and what to keep."""

    name: str
    url: str
    server: str | None
    viewport: Viewport
    actions: tuple[Action, ...]
    selector: str | None

    @property
    def file_name(self) -> str:
        return f"{self.name}.png"

    def resolve_url(self, base_urls: Mapping[str, str]) -> str:
        """The URL to open, given the base URL of each running server."""
        if self.server is None:
            return self.url
        return base_urls[self.server].rstrip("/") + self.url


def load_captures(path: Path = DEFAULT_CAPTURES_PATH) -> tuple[Capture, ...]:
    """Read and validate the capture list; raise ``CaptureError`` on the first problem."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CaptureError(f"Capture list not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise CaptureError(f"The capture list {path} is not valid TOML: {error}") from error
    return parse_captures(raw)


def parse_captures(raw: Mapping[str, Any]) -> tuple[Capture, ...]:
    """Validate the parsed TOML of a capture list."""
    tables = raw.get("capture", [])
    if not isinstance(tables, list) or not tables:
        raise CaptureError("The capture list declares no capture: add a [[capture]] table.")
    captures: list[Capture] = []
    seen: set[str] = set()
    for index, table in enumerate(tables, start=1):
        capture = _capture(table, f"capture #{index}")
        if capture.name in seen:
            raise CaptureError(f"Duplicate capture name '{capture.name}'.")
        seen.add(capture.name)
        captures.append(capture)
    return tuple(captures)


def _capture(table: Any, where: str) -> Capture:
    if not isinstance(table, dict):
        raise CaptureError(f"{where} must be a table.")
    name = _string(table, "name", where)
    if not _NAME_PATTERN.fullmatch(name):
        raise CaptureError(
            f"{where}: invalid name '{name}'; use lowercase letters, digits and hyphens."
        )
    where = f"capture '{name}'"
    unknown = sorted(set(table) - _FIELDS)
    if unknown:
        raise CaptureError(
            f"{where}: unknown field(s) {', '.join(unknown)}; "
            f"expected some of: {', '.join(sorted(_FIELDS))}."
        )

    url = _string(table, "url", where)
    server: str | None
    if _ABSOLUTE_URL_PATTERN.fullmatch(url):
        if "server" in table:
            raise CaptureError(f"{where}: an absolute URL targets no server; remove 'server'.")
        server = None
    elif url.startswith("/"):
        server = table.get("server", DEFAULT_SERVER)
        if server not in SERVERS:
            raise CaptureError(
                f"{where}: unknown server {server!r}; expected one of: {', '.join(SERVERS)}."
            )
    else:
        raise CaptureError(
            f"{where}: 'url' must be an absolute URL or a path starting with '/', got {url!r}."
        )

    selector = table.get("selector")
    if selector is not None and (not isinstance(selector, str) or not selector.strip()):
        raise CaptureError(f"{where}: 'selector' must be a non-empty string.")

    return Capture(
        name=name,
        url=url,
        server=server,
        viewport=_viewport(table.get("viewport"), where),
        actions=_actions(table.get("actions", []), where),
        selector=selector,
    )


def _viewport(raw: Any, where: str) -> Viewport:
    if raw is None:
        return Viewport(DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT)
    if not isinstance(raw, dict):
        raise CaptureError(f"{where}: 'viewport' must be a table with 'width' and 'height'.")
    sizes: dict[str, int] = {}
    for key in ("width", "height"):
        if key not in raw:
            raise CaptureError(f"{where}: the viewport is missing {key!r}.")
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CaptureError(f"{where}: the viewport {key} must be a positive integer.")
        sizes[key] = value
    return Viewport(width=sizes["width"], height=sizes["height"])


def _actions(raw: Any, where: str) -> tuple[Action, ...]:
    if not isinstance(raw, list):
        raise CaptureError(f"{where}: 'actions' must be a list of tables.")
    return tuple(_action(item, f"{where}, action #{i}") for i, item in enumerate(raw, start=1))


def _action(raw: Any, where: str) -> Action:
    if not isinstance(raw, dict):
        raise CaptureError(f'{where} must be a table such as {{ click = "#id" }}.')
    kinds = [key for key in raw if key in _ACTION_KINDS]
    expected = ", ".join(_ACTION_KINDS)
    if not kinds:
        raise CaptureError(f"{where}: unknown action {sorted(raw)}; expected one of: {expected}.")
    if len(kinds) > 1:
        raise CaptureError(f"{where}: an action holds exactly one of: {expected}.")
    kind = kinds[0]
    allowed = {kind, "value"} if kind == "fill" else {kind}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise CaptureError(f"{where}: unexpected key(s) {', '.join(extra)} for a {kind} action.")
    selector = _string(raw, kind, where)
    if kind == "fill":
        return Fill(selector=selector, value=_string(raw, "value", where, allow_empty=True))
    if kind == "click":
        return Click(selector=selector)
    return WaitFor(selector=selector)


def _string(table: Mapping[str, Any], key: str, where: str, *, allow_empty: bool = False) -> str:
    if key not in table:
        raise CaptureError(f"{where}: missing {key!r}.")
    value = table[key]
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CaptureError(f"{where}: {key!r} must be a non-empty string.")
    return value
