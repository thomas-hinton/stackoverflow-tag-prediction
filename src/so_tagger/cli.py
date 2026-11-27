"""Command-line interface of the project."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from so_tagger import __version__
from so_tagger.api import create_app, load_taggers
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
from so_tagger.evaluation import (
    EvaluationError,
    current_git_commit,
    evaluate_tagger,
    metrics_report,
    summary,
)
from so_tagger.quality_gate import QualityGateError, check_quality_gate, read_metrics_file
from so_tagger.release import ReleaseError, package_tagger, release_notes
from so_tagger.split import SPLIT_NAMES, SplitError, split_questions, write_split
from so_tagger.taggers import TAGGER_KINDS, TaggerError, load_tagger, train_tagger

DEFAULT_RAW_PATH = Path("data/stackoverflow_posts.csv")
DEFAULT_PREPARED_DIR = Path("data/prepared")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_TAGGERS_DIR = Path("artifacts")
DEFAULT_METRICS_DIR = Path("metrics")
DEFAULT_RELEASE_DIR = Path("dist")

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


def _known_split(name: str) -> str:
    if name not in SPLIT_NAMES:
        raise typer.BadParameter(
            f"Unknown split '{name}'; expected one of: {', '.join(SPLIT_NAMES)}."
        )
    return name


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
    except (
        ConfigError,
        DatasetError,
        EvaluationError,
        QualityGateError,
        ReleaseError,
        SplitError,
        TaggerError,
    ) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


class _EchoHandler(logging.Handler):
    """Print log records to the standard error of the running command."""

    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(self.format(record), err=True)


def _show_progress_messages() -> None:
    """Show the package's progress messages, such as the Transformer tagger's epoch log."""
    package_logger = logging.getLogger("so_tagger")
    package_logger.setLevel(logging.INFO)
    if not any(isinstance(handler, _EchoHandler) for handler in package_logger.handlers):
        package_logger.addHandler(_EchoHandler())


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
    _show_progress_messages()


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

    Hyperparameters come from the Tagger's section of the configuration. The transformer
    Tagger needs the `transformer` extra (`uv sync --extra transformer`). The Score cut-off is
    then tuned on the validation split: the cut-off from 0.01 to 0.99 (step 0.01) whose
    Suggestion lists reach the best micro-F1, the smallest one on ties. The test split is never
    read.
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
        f"({len(trained.supported_tags)} Supported tags)."
    )
    typer.echo(
        f"Tuned the Score cut-off on {len(validation_set)} validation Questions: "
        f"{trained.score_cutoff:.2f} (validation micro-F1 "
        f"{trained.metadata.validation_micro_f1:.3f})."
    )
    typer.echo(f"Tagger saved to {directory}")


@app.command()
def evaluate(
    tagger: TaggerOption = "linear",
    tagger_dir: Annotated[
        Path | None,
        typer.Option(help="Tagger directory written by `train`. Defaults to artifacts/<tagger>."),
    ] = None,
    split_name: Annotated[
        str,
        typer.Option(
            "--split",
            help=f"Split to evaluate on: {', '.join(SPLIT_NAMES)}.",
            callback=_known_split,
        ),
    ] = "test",
    splits_dir: Annotated[
        Path, typer.Option(help="Split written by `split`.")
    ] = DEFAULT_SPLITS_DIR,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Metrics file to write. Defaults to metrics/<tagger>-<split>.json.",
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Measure a saved Tagger on a split and write a metrics file (JSON).

    Suggestions come from the Suggestion policy at the Tagger's tuned Score cut-off. The file
    holds micro and macro scores, precision@k and recall@k, per-Tag scores and everything
    needed to trace the run; its schema is documented in docs/metrics.md.
    """
    metrics_path = output or DEFAULT_METRICS_DIR / f"{tagger}-{split_name}.json"
    with _user_errors():
        loaded = load_tagger(tagger_dir or DEFAULT_TAGGERS_DIR / tagger, kind=tagger)
        questions = read_questions(splits_dir / f"{split_name}.jsonl")
        metrics = evaluate_tagger(loaded, questions)
        report = metrics_report(loaded, split_name, len(questions), metrics, current_git_commit())
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"Evaluated the {tagger} Tagger on {len(questions)} {split_name} Questions "
        f"(Score cut-off {loaded.score_cutoff:.2f})."
    )
    typer.echo(summary(metrics))
    typer.echo(f"Metrics written to {metrics_path}")


@app.command()
def gate(
    tagger: TaggerOption = "linear",
    metrics: Annotated[
        Path | None,
        typer.Option(
            help="Metrics file written by `evaluate` on the test split. "
            "Defaults to metrics/<tagger>-test.json.",
            dir_okay=False,
        ),
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Fail unless the Tagger's test micro-F1 reaches its quality-gate floor.

    The floor of each Tagger kind is in `[quality_gate.min_test_micro_f1]` of the
    configuration. Exits with code 1, showing the value, the floor and the difference, when the
    test micro-F1 is below the floor.
    """
    with _user_errors():
        floors = load_config(config).quality_gate.min_test_micro_f1
        report = read_metrics_file(metrics or DEFAULT_METRICS_DIR / f"{tagger}-test.json")
        if report.get("tagger") != tagger:
            raise QualityGateError(
                f"The metrics file measures the {report.get('tagger')!r} Tagger, "
                f"not the '{tagger}' one."
            )
        result = check_quality_gate(report, floors)
    if not result.passed:
        typer.echo(f"Error: {result.message()}", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.message())


@app.command()
def package(
    release_tag: Annotated[
        str,
        typer.Option(help="Release tag the Tagger is published under: vMAJOR.MINOR.PATCH."),
    ],
    tagger: TaggerOption = "linear",
    tagger_dir: Annotated[
        Path | None,
        typer.Option(help="Tagger directory written by `train`. Defaults to artifacts/<tagger>."),
    ] = None,
    metrics: Annotated[
        Path | None,
        typer.Option(
            help="Metrics file written by `evaluate`, copied as <tagger>-metrics.json.",
            dir_okay=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Where to write the Release assets.", file_okay=False)
    ] = DEFAULT_RELEASE_DIR,
) -> None:
    """Turn a Tagger directory into the Release assets of a Release.

    Writes <tagger>-tagger.tar.gz, the Tagger directory whose metadata records the Release tag
    as its version, so that the API reports it. With --metrics, also writes
    <tagger>-metrics.json. The Tagger directory itself is left untouched.
    """
    with _user_errors():
        assets = package_tagger(
            tagger_dir or DEFAULT_TAGGERS_DIR / tagger,
            release_tag,
            output_dir,
            kind=tagger,
            metrics=metrics,
        )
    typer.echo(f"Packaged the {tagger} Tagger as version {release_tag}: {assets.archive}")
    if assets.metrics is not None:
        typer.echo(f"Metrics file copied to {assets.metrics}")


@app.command("release-notes")
def release_notes_command(
    metrics: Annotated[
        list[Path],
        typer.Option(
            help="Metrics file written by `evaluate`, one per Tagger; repeat the option to "
            "compare several Taggers side by side.",
            dir_okay=False,
        ),
    ],
    run_url: Annotated[
        str | None, typer.Option(help="Link to the workflow run that trained the Taggers.")
    ] = None,
) -> None:
    """Print the Markdown notes of a Release, generated from its Taggers' metrics files.

    They hold a table comparing the Taggers side by side (test metrics, Score cut-offs,
    training times), the frequency cut-off, the dataset sizes, the commit and, with --run-url,
    a link to the workflow run. The metrics files must share the split, dataset and commit.
    """
    with _user_errors():
        notes = release_notes([read_metrics_file(path) for path in metrics], run_url)
    typer.echo(notes, nl=False)


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(envvar="SO_TAGGER_HOST", help="Interface to listen on.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(envvar="SO_TAGGER_PORT", help="Port to listen on.")] = 8000,
    linear_dir: Annotated[
        Path,
        typer.Option(
            envvar="SO_TAGGER_LINEAR_DIR",
            help="Linear tagger directory written by `train`.",
            file_okay=False,
        ),
    ] = DEFAULT_TAGGERS_DIR / "linear",
    transformer_dir: Annotated[
        Path,
        typer.Option(
            envvar="SO_TAGGER_TRANSFORMER_DIR",
            help="Transformer tagger directory written by `train` (needs the transformer extra).",
            file_okay=False,
        ),
    ] = DEFAULT_TAGGERS_DIR / "transformer",
    torch_threads: Annotated[
        int,
        typer.Option(
            envvar="SO_TAGGER_TORCH_THREADS",
            min=0,
            help="CPU threads PyTorch uses for the Transformer tagger; 0 keeps its default.",
        ),
    ] = 0,
) -> None:
    """Serve the HTTP API; the interactive documentation is at /docs.

    Both Taggers are loaded once at startup. A missing Tagger directory leaves that Tagger
    unloaded: the API still starts, requests for it get a 503, and GET /health answers 503
    while the Linear tagger (the default) is missing. Each Tagger reports the version
    recorded in its metadata: the Release tag set by `package`, or `dev`.
    """
    with _user_errors():
        taggers = load_taggers({"linear": linear_dir, "transformer": transformer_dir})
    if torch_threads > 0 and "transformer" in taggers:
        import torch  # installed with the `transformer` extra, which loading it required

        torch.set_num_threads(torch_threads)
        typer.echo(f"PyTorch uses {torch_threads} CPU threads.")
    uvicorn.run(create_app(taggers), host=host, port=port)


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
