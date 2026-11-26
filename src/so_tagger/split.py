"""The frozen train/validation/test split of the prepared Questions.

A Question and all its Tags go to exactly one set. The split uses multi-label iterative
stratification (Sechidis, Tsoumakas and Vlahavas, 2011): the rarest Tags are placed first, each
of their Questions going to the set that still lacks the most of that Tag. This keeps even the
rarest Supported tags present in the test split, as ADR 0003 requires. Set sizes are fixed up
front from the ratios, and every random choice comes from one seeded generator, so the same
input and seed always give the same split.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so_tagger.config import SplitConfig
from so_tagger.dataset import write_questions
from so_tagger.domain import Question

SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    """The split cannot be produced or breaks the validation rule."""


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[Question, ...]
    validation: tuple[Question, ...]
    test: tuple[Question, ...]

    def sets(self) -> dict[str, tuple[Question, ...]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def summary(self) -> str:
        sizes = ", ".join(f"{len(questions)} {name}" for name, questions in self.sets().items())
        return f"Split into {sizes} Questions."


def split_questions(
    questions: Sequence[Question],
    supported_tags: Sequence[str],
    split_config: SplitConfig,
    seed: int,
) -> DatasetSplit:
    """Split Questions into train, validation and test sets.

    Raises ``SplitError`` naming the offending Tags if a Supported tag is absent from the
    training split.
    """
    rng = np.random.default_rng(seed)
    by_id = sorted(questions, key=lambda question: question.id)
    unassigned = [by_id[index] for index in rng.permutation(len(by_id))]

    capacity = _set_sizes(len(unassigned), split_config.ratios())
    tag_counts = Counter(tag for question in unassigned for tag in set(question.tags))
    wanted = [
        {tag: ratio * count for tag, count in tag_counts.items()} for ratio in split_config.ratios()
    ]
    assigned: list[list[Question]] = [[] for _ in SPLIT_NAMES]

    while unassigned:
        remaining = Counter(tag for question in unassigned for tag in set(question.tags))
        rarest = min(remaining, key=lambda tag: (remaining[tag], tag))
        for question in (q for q in unassigned if rarest in q.tags):
            target = _choose_set(rarest, wanted, capacity, rng)
            assigned[target].append(question)
            capacity[target] -= 1
            for tag in question.tags:
                wanted[target][tag] -= 1
        unassigned = [question for question in unassigned if rarest not in question.tags]

    train, validation, test = (
        tuple(sorted(questions, key=lambda question: question.id)) for questions in assigned
    )
    _check_training_covers(train, supported_tags)
    return DatasetSplit(train=train, validation=validation, test=test)


def write_split(split: DatasetSplit, directory: Path) -> None:
    """Write one JSON Lines file per set, Questions sorted by id."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, questions in split.sets().items():
        write_questions(questions, directory / f"{name}.jsonl")


def _set_sizes(total: int, ratios: tuple[float, ...]) -> list[int]:
    """Round ``total * ratio`` per set with the largest remainder method, so sizes sum up."""
    exact = [total * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    by_remainder = sorted(range(len(ratios)), key=lambda i: (sizes[i] - exact[i], i))
    for index in by_remainder[: total - sum(sizes)]:
        sizes[index] += 1
    return sizes


def _choose_set(
    tag: str, wanted: list[dict[str, float]], capacity: list[int], rng: np.random.Generator
) -> int:
    """The open set that wants the Tag most, then the one with most room, then at random."""
    open_sets = [index for index, room in enumerate(capacity) if room > 0]
    best = max((wanted[index][tag], capacity[index]) for index in open_sets)
    tied = [index for index in open_sets if (wanted[index][tag], capacity[index]) == best]
    return tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]


def _check_training_covers(train: Sequence[Question], supported_tags: Sequence[str]) -> None:
    present = {tag for question in train for tag in question.tags}
    missing = sorted(set(supported_tags) - present)
    if missing:
        raise SplitError(
            f"Supported tags absent from the training split: {', '.join(missing)}. "
            "Every Supported tag must appear at least once in training; check the "
            "frequency cut-off and the split ratios."
        )
