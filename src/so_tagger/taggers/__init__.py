"""Every Tagger kind behind the shared Tagger contract."""

import importlib
import time
from collections.abc import Sequence
from pathlib import Path

from so_tagger.config import ProjectConfig
from so_tagger.domain import Question
from so_tagger.evaluation import tune_score_cutoff
from so_tagger.taggers.base import DEV_VERSION, ScoreMatrix, Tagger, TaggerError, TaggerMetadata

# Each kind's module is imported only when that kind is used, so that the base install, which
# lacks the `transformer` extra (torch, transformers), still trains and serves the Linear tagger.
_TAGGERS: dict[str, tuple[str, str]] = {
    "linear": ("so_tagger.taggers.linear", "LinearTagger"),
    "transformer": ("so_tagger.taggers.transformer", "TransformerTagger"),
}

_EXTRAS: dict[str, str] = {"transformer": "transformer"}
"""The optional dependency group (extra) a Tagger kind needs beyond the base install."""

TAGGER_KINDS: tuple[str, ...] = tuple(_TAGGERS)
"""The Tagger kinds the project can train and load."""

__all__ = [
    "DEV_VERSION",
    "TAGGER_KINDS",
    "ScoreMatrix",
    "Tagger",
    "TaggerError",
    "TaggerMetadata",
    "load_tagger",
    "train_tagger",
]


def train_tagger(
    kind: str,
    train: Sequence[Question],
    validation: Sequence[Question],
    config: ProjectConfig,
) -> Tagger:
    """Train a Tagger of the given kind, then tune its Score cut-off on the validation split.

    The test split is never passed here: neither training nor tuning can read it.
    """
    start = time.perf_counter()
    tagger = _tagger_class(kind).train(train, validation, config)
    tuning = tune_score_cutoff(tagger, validation)
    tagger.record_tuning(
        tuning.score_cutoff, tuning.validation_micro_f1, time.perf_counter() - start
    )
    return tagger


def load_tagger(directory: Path, kind: str | None = None) -> Tagger:
    """Load the Tagger saved in ``directory``, checking its kind when one is given."""
    return _tagger_class(kind or TaggerMetadata.read(directory).kind).load(directory)


def _tagger_class(kind: str) -> type[Tagger]:
    try:
        module_name, class_name = _TAGGERS[kind]
    except KeyError as error:
        known = ", ".join(TAGGER_KINDS)
        raise TaggerError(f"Unknown Tagger kind '{kind}'; expected one of: {known}.") from error
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if kind not in _EXTRAS or error.name == module_name:
            raise
        raise TaggerError(
            f"The {kind} Tagger needs the '{_EXTRAS[kind]}' extra ({error.name} is missing): "
            f"run `uv sync --extra {_EXTRAS[kind]}`."
        ) from error
    tagger_class: type[Tagger] = getattr(module, class_name)
    return tagger_class
