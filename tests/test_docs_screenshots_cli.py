"""Seam 6: the `docs-screenshots` command, driven through its CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from so_tagger.screenshots.captures import SERVERS
from so_tagger.screenshots.cli import MAX_IMAGE_BYTES, SERVER_APPS, app

runner = CliRunner()

CAPTURES = """
[[capture]]
name = "api-docs"
url = "/docs"

[[capture]]
name = "releases"
url = "https://github.com/thomas-hinton/stackoverflow-tag-prediction/releases"
viewport = { width = 1024, height = 768 }
"""


@pytest.fixture
def captures_file(tmp_path: Path) -> Path:
    path = tmp_path / "screenshots.toml"
    path.write_text(CAPTURES, encoding="utf-8")
    return path


def test_list_shows_every_declared_capture_and_its_target(captures_file: Path) -> None:
    result = runner.invoke(app, ["--list", "--captures", str(captures_file)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert any("api-docs" in line and "api /docs" in line and "1280x800" in line for line in lines)
    assert any(
        "releases" in line and "https://github.com/" in line and "1024x768" in line
        for line in lines
    )


def test_every_server_a_capture_can_target_can_be_started() -> None:
    assert set(SERVER_APPS) == set(SERVERS)


def test_check_passes_when_every_png_is_declared(captures_file: Path, tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "api-docs.png").write_bytes(b"png")

    result = runner.invoke(
        app, ["--check", "--captures", str(captures_file), "--output-dir", str(images)]
    )

    assert result.exit_code == 0, result.output
    assert "No orphan" in result.output


def test_check_reports_orphan_pngs_and_fails(captures_file: Path, tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "api-docs.png").write_bytes(b"png")
    (images / "old-demo.png").write_bytes(b"png")

    result = runner.invoke(
        app, ["--check", "--captures", str(captures_file), "--output-dir", str(images)]
    )

    assert result.exit_code == 1
    assert "old-demo.png" in result.output
    assert "api-docs.png" not in result.output


def test_only_rejects_an_undeclared_capture(captures_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--only", "nope", "--captures", str(captures_file), "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Unknown capture 'nope'" in result.output
    assert "api-docs" in result.output


def test_an_invalid_capture_list_is_a_clean_error(tmp_path: Path) -> None:
    path = tmp_path / "screenshots.toml"
    path.write_text('[[capture]]\nname = "x"\n', encoding="utf-8")

    result = runner.invoke(app, ["--list", "--captures", str(path)])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "'url'" in result.output


@pytest.mark.slow
def test_the_api_docs_page_is_captured_as_a_small_png(tmp_path: Path) -> None:
    pytest.importorskip("playwright")
    output_dir = tmp_path / "images"

    result = runner.invoke(app, ["--only", "api-docs-suggest", "--output-dir", str(output_dir)])

    assert result.exit_code == 0, result.output
    image = output_dir / "api-docs-suggest.png"
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert 0 < image.stat().st_size <= MAX_IMAGE_BYTES
    assert "api-docs-suggest.png" in result.output
