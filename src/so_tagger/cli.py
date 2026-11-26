"""Command-line interface of the project."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from so_tagger import __version__
from so_tagger.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from so_tagger.dataset import (
    DatasetError,
    prepare_questions,
    read_prepared,
    read_questions,
    read_raw_questions,
    write_prepared,
)
from so_tagger.domain import SuggestionList
from so_tagger.split import SplitError, split_questions, write_split
from so_tagger.taggers import TAGGER_KINDS, TaggerError, load_tagger, train_tagger

DEFAULT_RAW_PATH = Path("data/stackoverflow_posts.csv")
DEFAULT_PREPARED_DIR = Path("data/prepared")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_TAGGERS_DIR = Path("artifacts")

app = typer.Typer(
    help="Suggest Tags for a StackOverflow Question from its Title.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Project configuration file.", dir_okay=False)
]


def _known_tagger_kind(kind: str) -> str:
    if kind not in TAGGER_KINDS:
        raise typer.BadParameter(
            f"Unknown Tagger kind '{kind}'; expected one of: {', '.join(TAGGER_KINDS)}."
        )
    return kind


def _non_empty_title(title: str) -> str:
    if not title.strip():
        raise typer.BadParameter("The Title must not be empty.")
    return title


TaggerOption = Annotated[
    str,
    typer.Option(
        "--tagger",
        help=f"Tagger kind: {', '.join(TAGGER_KINDS)}.",
        callback=_known_tagger_kind,
    ),
]


@contextmanager
def _user_errors() -> Iterator[None]:
    """Turn expected failures (bad configuration, bad data) into a clean error and exit code."""
    try:
        yield
    except (ConfigError, DatasetError, SplitError, TaggerError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"so-tagger {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Suggest Tags for a StackOverflow Question from its Title."""


@app.command()
def prepare(
    raw: Annotated[
        Path, typer.Option(help="Raw dataset CSV, one row per (Question, Tag) pair.")
    ] = DEFAULT_RAW_PATH,
    output_dir: Annotated[
        Path, typer.Option(help="Where to write the prepared Questions and Supported tags.")
    ] = DEFAULT_PREPARED_DIR,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Group the raw rows into Questions and keep only their Supported tags.

    A Tag is a Supported tag when at least `data.frequency_cutoff` distinct Questions carry
    it; Questions left without any Supported tag are dropped.
    """
    with _user_errors():
        project = load_config(config)
        dataset = prepare_questions(read_raw_questions(raw), project.data.frequency_cutoff)
        write_prepared(dataset, output_dir)
    typer.echo(dataset.summary())
    typer.echo(f"Prepared dataset written to {output_dir}")


@app.command()
def split(
    prepared_dir: Annotated[
        Path, typer.Option(help="Prepared dataset written by `prepare`.")
    ] = DEFAULT_PREPARED_DIR,
    output_dir: Annotated[
        Path, typer.Option(help="Where to write train.jsonl, validation.jsonl and test.jsonl.")
    ] = DEFAULT_SPLITS_DIR,
    seed: Annotated[
        int | None, typer.Option(help="Override the seed from the configuration.")
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Split the prepared Questions into the frozen train, validation and test sets.

    The split is stratified on Tags, uses the ratios in `[split]` and the configured seed, and
    is byte-identical across runs with the same input and seed. It fails if a Supported tag
    would be absent from the training split.
    """
    with _user_errors():
        project = load_config(config)
        questions, supported_tags = read_prepared(prepared_dir)
        dataset_split = split_questions(
            questions, supported_tags, project.split, project.seed if seed is None else seed
        )
        write_split(dataset_split, output_dir)
    typer.echo(dataset_split.summary())
    typer.echo(f"Split written to {output_dir}")


@app.command()
def train(
    tagger: TaggerOption = "linear",
    splits_dir: Annotated[
        Path, typer.Option(help="Split written by `split` (train and validation sets are used).")
    ] = DEFAULT_SPLITS_DIR,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Where to save the Tagger directory. Defaults to artifacts/<tagger>."),
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Train a Tagger on the training split and save it as a Tagger directory.

    Hyperparameters and the Score cut-off come from the Tagger's section of the configuration.
    """
    directory = output_dir or DEFAULT_TAGGERS_DIR / tagger
    with _user_errors():
        project = load_config(config)
        train_set = read_questions(splits_dir / "train.jsonl")
        validation_set = read_questions(splits_dir / "validation.jsonl")
        trained = train_tagger(tagger, train_set, validation_set, project)
        trained.save(directory)
    typer.echo(
        f"Trained the {tagger} Tagger on {len(train_set)} Questions "
        f"({len(trained.supported_tags)} Supported tags, Score cut-off {trained.score_cutoff})."
    )
    typer.echo(f"Tagger saved to {directory}")


@app.command()
def suggest(
    title: Annotated[
        str,
        typer.Argument(
            help="Title of the Question to suggest Tags for.", callback=_non_empty_title
        ),
    ],
    tagger: TaggerOption = "linear",
    tagger_dir: Annotated[
        Path | None,
        typer.Option(help="Tagger directory written by `train`. Defaults to artifacts/<tagger>."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the Suggestion list as JSON.")
    ] = False,
) -> None:
    """Print the ranked Suggestion list for a Title, with the Score of each Tag."""
    with _user_errors():
        loaded = load_tagger(tagger_dir or DEFAULT_TAGGERS_DIR / tagger, kind=tagger)
        suggestions = loaded.suggest(title)
    typer.echo(_as_json(suggestions) if as_json else _as_table(suggestions))


def _as_json(suggestions: SuggestionList) -> str:
    return json.dumps([{"tag": s.tag, "score": s.score} for s in suggestions], indent=2)


def _as_table(suggestions: SuggestionList) -> str:
    width = max(len("Tag"), *(len(s.tag) for s in suggestions))
    rows = [f"{'Tag':<{width}}  Score"]
    rows += [f"{s.tag:<{width}}  {s.score:.3f}" for s in suggestions]
    return "\n".join(rows)
