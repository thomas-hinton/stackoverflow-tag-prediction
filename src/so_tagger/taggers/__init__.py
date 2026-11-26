"""Every Tagger kind behind the shared Tagger contract."""

from collections.abc import Sequence
from pathlib import Path

from so_tagger.config import ProjectConfig
from so_tagger.domain import Question
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
    """Train a Tagger of the given kind."""
    return _tagger_class(kind).train(train, validation, config)


def load_tagger(directory: Path, kind: str | None = None) -> Tagger:
    """Load the Tagger saved in ``directory``, checking its kind when one is given."""
    metadata = TaggerMetadata.read(directory)
    if kind is not None and metadata.kind != kind:
        raise TaggerError(f"{directory} holds a '{metadata.kind}' Tagger, not a '{kind}' one.")
    return _tagger_class(metadata.kind).load(directory)


def _tagger_class(kind: str) -> type[Tagger]:
    try:
        return _TAGGERS[kind]
    except KeyError as error:
        known = ", ".join(TAGGER_KINDS)
        raise TaggerError(f"Unknown Tagger kind '{kind}'; expected one of: {known}.") from error
