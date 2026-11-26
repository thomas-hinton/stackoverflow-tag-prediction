"""Every Tagger kind behind the shared Tagger contract."""

import time
from collections.abc import Sequence
from pathlib import Path

from so_tagger.config import ProjectConfig
from so_tagger.domain import Question
from so_tagger.evaluation import tune_score_cutoff
from so_tagger.taggers.base import ScoreMatrix, Tagger, TaggerError, TaggerMetadata
from so_tagger.taggers.linear import LinearTagger

_TAGGERS: dict[str, type[Tagger]] = {LinearTagger.kind: LinearTagger}

TAGGER_KINDS: tuple[str, ...] = tuple(_TAGGERS)
"""The Tagger kinds the project can train and load."""

__all__ = [
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
        return _TAGGERS[kind]
    except KeyError as error:
        known = ", ".join(TAGGER_KINDS)
        raise TaggerError(f"Unknown Tagger kind '{kind}'; expected one of: {known}.") from error
