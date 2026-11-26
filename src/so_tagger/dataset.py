"""Data preparation: from the raw dataset to Questions carrying only Supported tags.

The raw dataset has one row per (Question, Tag) pair. Its anomalies and the rules applied to
them are documented in ``notebooks/01_data_exploration.ipynb``:

- the Tag literally named ``null`` must stay a Tag, so the file is read without NA detection;
- ``tag_position`` has gaps and does not always start at 0, so it is only used to order a
  Question's Tags, never as an index;
- no duplicated (Question, Tag) pair, no Title disagreement within a Question and a one-to-one
  ``tag_id`` / ``tag_name`` mapping are asserted on load, so a new export that breaks them fails.
"""

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from so_tagger.domain import Question

RAW_COLUMNS = ("post_id", "tag_name", "tag_id", "tag_position", "title")
QUESTIONS_FILE = "questions.jsonl"
SUPPORTED_TAGS_FILE = "supported_tags.json"


class DatasetError(ValueError):
    """The dataset breaks one of the invariants the pipeline relies on."""


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Questions filtered to Supported tags, with what was kept and dropped."""

    questions: tuple[Question, ...]
    supported_tags: tuple[str, ...]
    frequency_cutoff: int
    questions_read: int

    @property
    def questions_dropped(self) -> int:
        return self.questions_read - len(self.questions)

    def summary(self) -> str:
        return (
            f"{self.questions_read} Questions read, {len(self.questions)} kept, "
            f"{self.questions_dropped} dropped (no Supported tag left); "
            f"{len(self.supported_tags)} Supported tags "
            f"(frequency cut-off {self.frequency_cutoff})."
        )


def read_raw_questions(path: Path) -> list[Question]:
    """Group the raw rows by post id into Questions, checking the dataset invariants."""
    titles: dict[int, str] = {}
    tag_rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
    id_of_name: dict[str, str] = {}
    name_of_id: dict[str, str] = {}

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise DatasetError(f"Expected columns {RAW_COLUMNS}, got {reader.fieldnames}.")
        for line, row in enumerate(reader, start=2):
            post_id, tag, title = int(row["post_id"]), row["tag_name"], row["title"].strip()
            if not tag or not title:
                raise DatasetError(f"Line {line}: empty Tag or Title for Question {post_id}.")
            if titles.setdefault(post_id, title) != title:
                raise DatasetError(f"Question {post_id} has rows with different Titles.")
            if any(existing == tag for _, existing in tag_rows[post_id]):
                raise DatasetError(f"Question {post_id} carries the Tag '{tag}' twice.")
            tag_id = row["tag_id"]
            if id_of_name.setdefault(tag, tag_id) != tag_id or (
                name_of_id.setdefault(tag_id, tag) != tag
            ):
                raise DatasetError(f"tag_id and tag_name are not one-to-one around '{tag}'.")
            tag_rows[post_id].append((int(row["tag_position"]), tag))

    return [
        Question(id=post_id, title=title, tags=tuple(tag for _, tag in sorted(tag_rows[post_id])))
        for post_id, title in sorted(titles.items())
    ]


def prepare_questions(questions: Sequence[Question], frequency_cutoff: int) -> PreparedDataset:
    """Keep the Tags carried by at least ``frequency_cutoff`` Questions over the whole dataset.

    Questions left with no Supported tag are dropped.
    """
    frequency = Counter(tag for question in questions for tag in set(question.tags))
    supported = {tag for tag, count in frequency.items() if count >= frequency_cutoff}
    kept = []
    for question in questions:
        tags = tuple(tag for tag in question.tags if tag in supported)
        if tags:
            kept.append(Question(id=question.id, title=question.title, tags=tags))
    return PreparedDataset(
        questions=tuple(kept),
        supported_tags=tuple(sorted(supported)),
        frequency_cutoff=frequency_cutoff,
        questions_read=len(questions),
    )


def write_prepared(dataset: PreparedDataset, directory: Path) -> None:
    """Write the prepared Questions and the Supported tags to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    write_questions(dataset.questions, directory / QUESTIONS_FILE)
    supported = {"frequency_cutoff": dataset.frequency_cutoff, "tags": dataset.supported_tags}
    (directory / SUPPORTED_TAGS_FILE).write_text(
        json.dumps(supported, indent=2) + "\n", encoding="utf-8"
    )


def read_prepared(directory: Path) -> tuple[list[Question], tuple[str, ...]]:
    """Read back the prepared Questions and the Supported tags written by ``write_prepared``."""
    try:
        supported = json.loads((directory / SUPPORTED_TAGS_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"No prepared dataset in {directory}; run `prepare` first.") from error
    return read_questions(directory / QUESTIONS_FILE), tuple(supported["tags"])


def write_questions(questions: Iterable[Question], path: Path) -> None:
    """Write Questions as JSON Lines: one ``{question_id, title, tags}`` object per line."""
    lines = (
        json.dumps(
            {"question_id": question.id, "title": question.title, "tags": list(question.tags)},
            ensure_ascii=False,
        )
        for question in questions
    )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read_questions(path: Path) -> list[Question]:
    """Read Questions written by ``write_questions``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise DatasetError(f"Questions file not found: {path}") from error
    questions = []
    for line in text.splitlines():
        record = json.loads(line)
        questions.append(
            Question(id=record["question_id"], title=record["title"], tags=tuple(record["tags"]))
        )
    return questions
