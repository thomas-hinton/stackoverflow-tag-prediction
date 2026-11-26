"""The Tagger contract shared by every Tagger kind.

A Tagger is trained from the training Questions (and the validation Questions, used for
tuning) plus the project configuration. It exposes its Supported tags, fixed at training time,
and its Score cut-off; it scores batches of Titles and turns one Title into a Suggestion list
through the shared Suggestion policy. It is saved to a directory holding its model files and a
``metadata.json`` file, and loaded back with ``so_tagger.taggers.load_tagger``.

A new Tagger kind subclasses ``Tagger`` and implements ``train``, ``score``, ``_save_model`` and
``_load_model``; everything else comes from this class.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

from so_tagger import __version__
from so_tagger.config import ProjectConfig
from so_tagger.domain import Question, SuggestionList
from so_tagger.suggestion_policy import apply_suggestion_policy

METADATA_FILE = "metadata.json"

ScoreMatrix = npt.NDArray[np.float64]
"""Per-Tag Scores in [0, 1]: one row per Title, one column per Supported tag, in order."""


class TaggerError(ValueError):
    """A Tagger directory is missing, incomplete or of an unexpected kind."""


@dataclass(frozen=True, slots=True)
class TaggerMetadata:
    """What a saved Tagger directory records about the Tagger it holds."""

    kind: str
    supported_tags: tuple[str, ...]
    score_cutoff: float
    frequency_cutoff: int
    training_config: dict[str, Any]
    package_version: str
    created_at: str

    @classmethod
    def at_training(
        cls,
        kind: str,
        train: Sequence[Question],
        score_cutoff: float,
        config: ProjectConfig,
        training_config: dict[str, Any],
    ) -> Self:
        """Metadata of a Tagger trained now: its Supported tags are the training Tags."""
        return cls(
            kind=kind,
            supported_tags=tuple(sorted({tag for question in train for tag in question.tags})),
            score_cutoff=score_cutoff,
            frequency_cutoff=config.data.frequency_cutoff,
            training_config=training_config,
            package_version=__version__,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def write(self, directory: Path) -> None:
        text = json.dumps(asdict(self), indent=2, ensure_ascii=False)
        (directory / METADATA_FILE).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def read(cls, directory: Path) -> Self:
        path = directory / METADATA_FILE
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise TaggerError(
                f"No Tagger in {directory}: {METADATA_FILE} is missing; run `train` first."
            ) from error
        try:
            return cls(**{**raw, "supported_tags": tuple(raw["supported_tags"])})
        except (KeyError, TypeError) as error:
            raise TaggerError(f"{path} is not valid Tagger metadata: {error}") from error


class Tagger(ABC):
    """A trained model that turns a Title into a Suggestion list."""

    kind: ClassVar[str]

    def __init__(self, metadata: TaggerMetadata) -> None:
        self._metadata = metadata

    @classmethod
    @abstractmethod
    def train(
        cls,
        train: Sequence[Question],
        validation: Sequence[Question],
        config: ProjectConfig,
    ) -> Self:
        """Train a Tagger whose Supported tags are the Tags of the training Questions."""

    @abstractmethod
    def score(self, titles: Sequence[str]) -> ScoreMatrix:
        """Per-Tag Scores for a batch of Titles, columns ordered as ``supported_tags``."""

    @abstractmethod
    def _save_model(self, directory: Path) -> None:
        """Write the model files (everything but the metadata) to ``directory``."""

    @classmethod
    @abstractmethod
    def _load_model(cls, directory: Path, metadata: TaggerMetadata) -> Self:
        """Rebuild the Tagger from the files written by ``_save_model``."""

    @property
    def metadata(self) -> TaggerMetadata:
        return self._metadata

    @property
    def supported_tags(self) -> tuple[str, ...]:
        return self._metadata.supported_tags

    @property
    def score_cutoff(self) -> float:
        return self._metadata.score_cutoff

    def suggest(self, title: str) -> SuggestionList:
        """The Suggestion list for one Title, from its Scores and the Suggestion policy."""
        scores = self.score([title])[0]
        return apply_suggestion_policy(
            dict(zip(self.supported_tags, scores.tolist(), strict=True)), self.score_cutoff
        )

    def save(self, directory: Path) -> None:
        """Save the model files and ``metadata.json`` to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        self._save_model(directory)
        self._metadata.write(directory)

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load a Tagger of this kind saved with ``save``."""
        metadata = TaggerMetadata.read(directory)
        if metadata.kind != cls.kind:
            raise TaggerError(
                f"{directory} holds a '{metadata.kind}' Tagger, not a '{cls.kind}' one."
            )
        return cls._load_model(directory, metadata)
