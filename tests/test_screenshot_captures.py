"""Seam: the capture list of the `docs-screenshots` command, parsed and validated from TOML."""

from pathlib import Path

import pytest

from so_tagger.screenshots.captures import (
    DEFAULT_CAPTURES_PATH,
    CaptureError,
    Click,
    Fill,
    Viewport,
    WaitFor,
    load_captures,
)


def write_captures(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "screenshots.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_minimal_capture_targets_the_api_with_the_default_viewport(tmp_path: Path) -> None:
    path = write_captures(
        tmp_path,
        """
        [[capture]]
        name = "api-docs"
        url = "/docs"
        """,
    )

    (capture,) = load_captures(path)

    assert capture.name == "api-docs"
    assert capture.file_name == "api-docs.png"
    assert capture.server == "api"
    assert capture.url == "/docs"
    assert capture.viewport == Viewport(width=1280, height=800)
    assert capture.actions == ()
    assert capture.selector is None
    assert capture.resolve_url({"api": "http://127.0.0.1:5000"}) == "http://127.0.0.1:5000/docs"


def test_a_capture_runs_actions_in_order_and_crops_to_a_selector(tmp_path: Path) -> None:
    path = write_captures(
        tmp_path,
        """
        [[capture]]
        name = "typed-title"
        url = "/"
        viewport = { width = 1024, height = 600 }
        selector = "#result"
        actions = [
          { fill = "#title", value = "How do I parse JSON in Python?" },
          { click = "button[type=submit]" },
          { wait_for = "#result li" },
        ]
        """,
    )

    (capture,) = load_captures(path)

    assert capture.viewport == Viewport(width=1024, height=600)
    assert capture.selector == "#result"
    assert capture.actions == (
        Fill(selector="#title", value="How do I parse JSON in Python?"),
        Click(selector="button[type=submit]"),
        WaitFor(selector="#result li"),
    )


def test_an_absolute_url_needs_no_server(tmp_path: Path) -> None:
    url = "https://github.com/thomas-hinton/stackoverflow-tag-prediction/releases"
    path = write_captures(
        tmp_path,
        f"""
        [[capture]]
        name = "releases"
        url = "{url}"
        """,
    )

    (capture,) = load_captures(path)

    assert capture.server is None
    assert capture.resolve_url({}) == url


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('url = "/docs"', "'name'"),
        ('name = "x"', "'url'"),
        ('name = "x"\nurl = "/docs"\nactions = [{ hover = "#a" }]', "Unknown action"),
        ('name = "x"\nurl = "/docs"\nactions = [{ fill = "#a" }]', "'value'"),
        ('name = "x"\nurl = "/docs"\nactions = [{ click = "#a", wait_for = "#b" }]', "one of"),
        ('name = "x"\nurl = "/docs"\nserver = "nowhere"', "Unknown server"),
        ('name = "x"\nurl = "https://github.com"\nserver = "api"', "absolute URL"),
        ('name = "x"\nurl = "docs"', "absolute URL or a path"),
        ('name = "x"\nurl = "/docs"\nviewport = { width = 0, height = 800 }', "width"),
        ('name = "x"\nurl = "/docs"\nviewport = { width = 1280 }', "'height'"),
        ('name = "../x"\nurl = "/docs"', "name"),
        ('name = "x"\nurl = "/docs"\ncolour = "red"', "Unknown field"),
    ],
)
def test_an_invalid_capture_is_rejected_with_its_name_and_the_reason(
    tmp_path: Path, body: str, message: str
) -> None:
    path = write_captures(tmp_path, f"[[capture]]\n{body}\n")

    with pytest.raises(CaptureError, match=message):
        load_captures(path)


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    path = write_captures(
        tmp_path,
        """
        [[capture]]
        name = "api-docs"
        url = "/docs"

        [[capture]]
        name = "api-docs"
        url = "/redoc"
        """,
    )

    with pytest.raises(CaptureError, match="Duplicate capture name 'api-docs'"):
        load_captures(path)


def test_a_file_without_captures_or_with_bad_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CaptureError, match="no capture"):
        load_captures(write_captures(tmp_path, "# nothing\n"))
    with pytest.raises(CaptureError, match="TOML"):
        load_captures(write_captures(tmp_path, "[[capture]\n"))
    with pytest.raises(CaptureError, match="not found"):
        load_captures(tmp_path / "missing.toml")


def test_the_versioned_capture_list_holds_the_expanded_suggest_operation() -> None:
    captures = {capture.name: capture for capture in load_captures(DEFAULT_CAPTURES_PATH)}

    api_docs = captures["api-docs-suggest"]
    assert api_docs.server == "api"
    assert api_docs.url == "/docs"
    assert api_docs.actions, "the POST /suggest operation must be expanded before capture"
