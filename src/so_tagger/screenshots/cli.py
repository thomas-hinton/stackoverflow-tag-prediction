"""The `docs-screenshots` command: regenerate the documentation screenshots.

It trains a small Linear tagger on the fixture dataset, serves the API on it on a free local
port, captures every declared page with headless Chromium, then optimises each PNG and fails
when one exceeds ``MAX_IMAGE_BYTES``. Needs the `docs` dependency group and Chromium:
``uv sync --group docs && uv run --group docs playwright install chromium``.
"""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated, Final

import typer

from so_tagger.api import create_app
from so_tagger.config import ConfigError, load_config
from so_tagger.dataset import DatasetError, prepare_questions, read_raw_questions
from so_tagger.screenshots.browser import BrowserError, take_screenshots
from so_tagger.screenshots.captures import (
    DEFAULT_CAPTURES_PATH,
    Capture,
    CaptureError,
    load_captures,
)
from so_tagger.screenshots.images import optimise_png, orphan_images
from so_tagger.screenshots.server import ServerError, running_server
from so_tagger.split import SplitError, split_questions
from so_tagger.taggers import Tagger, TaggerError, train_tagger

DEFAULT_OUTPUT_DIR = Path("docs/images/screenshots")
FIXTURE_DATASET = Path("tests/fixtures/stackoverflow_posts_fixture.csv")
FIXTURE_CONFIG = Path("tests/fixtures/config.toml")
MAX_IMAGE_BYTES: Final = 300 * 1024

app = typer.Typer(add_completion=False)


@contextmanager
def _user_errors() -> Iterator[None]:
    try:
        yield
    except (
        BrowserError,
        CaptureError,
        ConfigError,
        DatasetError,
        ServerError,
        SplitError,
        TaggerError,
    ) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command()
def main(
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Run only this capture; repeat to run several."),
    ] = None,
    list_captures: Annotated[
        bool, typer.Option("--list", help="Show the declared captures and exit.")
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Report PNGs in the output folder that no capture declares, and exit."
        ),
    ] = False,
    captures_path: Annotated[
        Path, typer.Option("--captures", help="Capture list.", dir_okay=False)
    ] = DEFAULT_CAPTURES_PATH,
    output_dir: Annotated[
        Path, typer.Option(help="Where to write the PNGs.", file_okay=False)
    ] = DEFAULT_OUTPUT_DIR,
    dataset: Annotated[
        Path, typer.Option(help="Raw dataset the fixture Tagger is trained on.", dir_okay=False)
    ] = FIXTURE_DATASET,
    config: Annotated[
        Path, typer.Option(help="Configuration of the fixture Tagger.", dir_okay=False)
    ] = FIXTURE_CONFIG,
) -> None:
    """Regenerate the documentation screenshots declared in the capture list.

    Starts the API on a Linear tagger trained on the fixture dataset, captures each page with
    headless Chromium, optimises the PNGs and fails if one exceeds 300 KB. The server is always
    stopped, even when a capture fails.
    """
    with _user_errors():
        captures = load_captures(captures_path)
        if list_captures:
            _show(captures)
            return
        if check:
            _check(captures, output_dir)
            return
        selected = _select(captures, only)
        written = _capture(selected, output_dir, dataset, config)
    _report(written)


def _show(captures: Sequence[Capture]) -> None:
    width = max(len(capture.name) for capture in captures)
    for capture in captures:
        target = capture.url if capture.server is None else f"{capture.server} {capture.url}"
        size = f"{capture.viewport.width}x{capture.viewport.height}"
        extras = [f"{len(capture.actions)} action(s)"] if capture.actions else []
        if capture.selector:
            extras.append(f"cropped to {capture.selector}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        typer.echo(f"{capture.name:<{width}}  {size:>9}  {target}{suffix}")


def _check(captures: Sequence[Capture], output_dir: Path) -> None:
    orphans = orphan_images(output_dir, captures)
    if not orphans:
        typer.echo(f"No orphan PNG in {output_dir}.")
        return
    typer.echo(f"PNGs in {output_dir} that no capture declares:", err=True)
    for path in orphans:
        typer.echo(f"  {path}", err=True)
    raise typer.Exit(code=1)


def _select(captures: Sequence[Capture], only: Sequence[str] | None) -> list[Capture]:
    if not only:
        return list(captures)
    by_name = {capture.name: capture for capture in captures}
    unknown = [name for name in only if name not in by_name]
    if unknown:
        raise CaptureError(f"Unknown capture '{unknown[0]}'; declared: {', '.join(by_name)}.")
    return [by_name[name] for name in dict.fromkeys(only)]


def _capture(
    captures: Sequence[Capture], output_dir: Path, dataset: Path, config: Path
) -> list[tuple[Path, int]]:
    servers = {capture.server for capture in captures if capture.server is not None}
    with ExitStack() as stack:
        base_urls: dict[str, str] = {}
        if "api" in servers:
            typer.echo(f"Training the fixture Linear tagger on {dataset}...")
            app = create_app({"linear": _fixture_tagger(dataset, config)})
            base_urls["api"] = stack.enter_context(running_server(app))
            typer.echo(f"API serving at {base_urls['api']}")
        paths = take_screenshots(captures, base_urls, output_dir)
    written = [(path, optimise_png(path)) for path in paths]
    oversized = [(path, size) for path, size in written if size > MAX_IMAGE_BYTES]
    if oversized:
        _report(written)
        names = ", ".join(f"{path.name} ({size / 1024:.0f} KB)" for path, size in oversized)
        raise BrowserError(
            f"Image(s) over the {MAX_IMAGE_BYTES // 1024} KB limit: {names}. "
            "Crop to a selector or use a smaller viewport."
        )
    return written


def _fixture_tagger(dataset: Path, config_path: Path) -> Tagger:
    config = load_config(config_path)
    prepared = prepare_questions(read_raw_questions(dataset), config.data.frequency_cutoff)
    dataset_split = split_questions(
        prepared.questions, prepared.supported_tags, config.split, config.seed
    )
    return train_tagger("linear", dataset_split.train, dataset_split.validation, config)


def _report(written: Sequence[tuple[Path, int]]) -> None:
    for path, size in written:
        typer.echo(f"{path}  {size / 1024:6.1f} KB")
