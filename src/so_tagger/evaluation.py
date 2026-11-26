"""Score cut-off tuning and evaluation metrics, shared by every Tagger kind.

Everything here goes through the Tagger contract only: batch Scores (``Tagger.score``), the
Supported tags and the Score cut-off, combined with the shared Suggestion policy. Nothing
depends on how a Tagger computes its Scores, so the Transformer tagger reuses it unchanged.

The metrics file written by ``evaluate`` is built by ``metrics_report``; its schema is
documented in ``docs/metrics.md``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from so_tagger import __version__
from so_tagger.domain import Question
from so_tagger.suggestion_policy import MAX_SUGGESTIONS, MIN_SUGGESTIONS

if TYPE_CHECKING:
    from so_tagger.taggers.base import ScoreMatrix, Tagger

METRICS_SCHEMA_VERSION = 1

SCORE_CUTOFF_GRID: tuple[float, ...] = tuple(round(step / 100, 2) for step in range(1, 100))
"""Candidate Score cut-offs tried on the validation split: 0.01 to 0.99 in steps of 0.01."""

TOP_K: tuple[int, ...] = (1, 3, 5)
"""The k of precision@k and recall@k."""

CONVENTIONS = {
    "averaging": (
        "micro-* pool every (Question, Supported tag) decision; macro-F1 is the unweighted "
        "mean of the per-Tag F1 over all Supported tags."
    ),
    "zero_division": (
        "A precision, recall or F1 whose denominator is 0 counts as 0, never as an error: a "
        "Supported tag with no true and no predicted example has F1 = 0 and still counts in "
        "macro-F1."
    ),
    "suggestions": (
        "Predicted Tags are the Suggestion list given by the Suggestion policy at the Tagger's "
        "Score cut-off (at least 1, at most 5 Suggestions)."
    ),
    "at_k": (
        "precision@k and recall@k use the top k Tags ranked by Score, independent of the "
        "Score cut-off: P@k = |top-k ∩ true| / k and R@k = |top-k ∩ true| / |true|, each "
        "averaged over Questions."
    ),
}
"""How the metrics are computed, copied into every metrics file."""

BoolMatrix = npt.NDArray[np.bool_]
"""One row per Question, one column per Supported tag."""


class EvaluationError(ValueError):
    """The Questions cannot be evaluated against the Tagger's Supported tags."""


@dataclass(frozen=True, slots=True)
class ScoreCutoffTuning:
    """The Score cut-off chosen on the validation split and the micro-F1 it reaches there."""

    score_cutoff: float
    validation_micro_f1: float


@dataclass(frozen=True, slots=True)
class TagMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class Metrics:
    """Quality of a Tagger's Suggestions on one set of Questions."""

    micro_f1: float
    micro_precision: float
    micro_recall: float
    macro_f1: float
    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    mean_suggestions_per_question: float
    per_tag: dict[str, TagMetrics]


def true_tag_matrix(questions: Sequence[Question], supported_tags: Sequence[str]) -> BoolMatrix:
    """Which Supported tags each Question carries; Tags that are not supported are ignored.

    Raises ``EvaluationError`` if a Question carries no Supported tag, since recall and
    recall@k are undefined for it.
    """
    column = {tag: index for index, tag in enumerate(supported_tags)}
    matrix = np.zeros((len(questions), len(supported_tags)), dtype=np.bool_)
    for row, question in enumerate(questions):
        columns = [column[tag] for tag in question.tags if tag in column]
        if not columns:
            raise EvaluationError(
                f"Question {question.id} carries no Supported tag of this Tagger: "
                f"{', '.join(question.tags)}."
            )
        matrix[row, columns] = True
    return matrix


def score_ranks(scores: ScoreMatrix, supported_tags: Sequence[str]) -> npt.NDArray[np.intp]:
    """Rank of each Tag within its row: 0 for the best Score, ties broken by Tag name.

    This is the ranking of the Suggestion policy.
    """
    name_order = np.argsort(np.argsort(np.asarray(supported_tags, dtype=object), kind="stable"))
    names = np.broadcast_to(name_order, scores.shape)
    order = np.lexsort((names, -scores), axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(scores.shape[1]), axis=-1)
    return ranks


def suggestion_matrix(
    scores: ScoreMatrix, supported_tags: Sequence[str], score_cutoff: float
) -> BoolMatrix:
    """Which Tags the Suggestion policy keeps for each row of Scores.

    Equivalent to ``apply_suggestion_policy`` on every row, computed for a whole batch at once.
    """
    return _suggested(score_ranks(scores, supported_tags), scores, score_cutoff)


def tune_score_cutoff(tagger: Tagger, validation: Sequence[Question]) -> ScoreCutoffTuning:
    """Choose the Score cut-off with the best micro-F1 on the validation Questions.

    Each candidate of ``SCORE_CUTOFF_GRID`` is applied through the full Suggestion policy. Ties
    go to the smallest cut-off, so the choice is deterministic. Only the validation Questions
    are read.
    """
    true = true_tag_matrix(validation, tagger.supported_tags)
    scores = tagger.score([question.title for question in validation])
    ranks = score_ranks(scores, tagger.supported_tags)
    best = ScoreCutoffTuning(score_cutoff=SCORE_CUTOFF_GRID[0], validation_micro_f1=-1.0)
    for cutoff in SCORE_CUTOFF_GRID:
        micro_f1 = _micro(_suggested(ranks, scores, cutoff), true)[2]
        if micro_f1 > best.validation_micro_f1:
            best = ScoreCutoffTuning(score_cutoff=cutoff, validation_micro_f1=micro_f1)
    return best


def evaluate_tagger(tagger: Tagger, questions: Sequence[Question]) -> Metrics:
    """Score the Questions with the Tagger and measure its Suggestions against their Tags."""
    scores = tagger.score([question.title for question in questions])
    return compute_metrics(
        scores,
        true_tag_matrix(questions, tagger.supported_tags),
        tagger.supported_tags,
        tagger.score_cutoff,
    )


def compute_metrics(
    scores: ScoreMatrix,
    true: BoolMatrix,
    supported_tags: Sequence[str],
    score_cutoff: float,
) -> Metrics:
    """Every metric of the metrics file, from Scores and true Tags (see ``CONVENTIONS``)."""
    if scores.shape != true.shape or scores.shape[1] != len(supported_tags):
        raise EvaluationError(
            f"Scores {scores.shape} and true Tags {true.shape} must both have one column per "
            f"Supported tag ({len(supported_tags)})."
        )
    if scores.shape[0] == 0:
        raise EvaluationError("Cannot evaluate a Tagger on zero Questions.")
    ranks = score_ranks(scores, supported_tags)
    predicted = _suggested(ranks, scores, score_cutoff)
    micro_precision, micro_recall, micro_f1 = _micro(predicted, true)

    true_positives = (predicted & true).sum(axis=0)
    predicted_counts = predicted.sum(axis=0)
    support = true.sum(axis=0)
    per_tag = {
        tag: TagMetrics(
            precision=_ratio(true_positives[i], predicted_counts[i]),
            recall=_ratio(true_positives[i], support[i]),
            f1=_ratio(2 * true_positives[i], predicted_counts[i] + support[i]),
            support=int(support[i]),
        )
        for i, tag in enumerate(supported_tags)
    }

    n_true = true.sum(axis=1)
    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    for k in TOP_K:
        hits = (true & (ranks < k)).sum(axis=1)
        precision_at_k[str(k)] = float(np.mean(hits / k))
        recall_at_k[str(k)] = float(np.mean(hits / n_true))

    return Metrics(
        micro_f1=micro_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        macro_f1=float(np.mean([tag.f1 for tag in per_tag.values()])),
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mean_suggestions_per_question=float(predicted.sum(axis=1).mean()),
        per_tag=per_tag,
    )


def metrics_report(
    tagger: Tagger,
    split: str,
    n_questions: int,
    metrics: Metrics,
    git_commit: str | None,
) -> dict[str, Any]:
    """The JSON object written to the metrics file (schema in ``docs/metrics.md``)."""
    metadata = tagger.metadata
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "tagger": metadata.kind,
        "split": split,
        "metrics": asdict(metrics),
        "conventions": CONVENTIONS,
        "score_cutoff": tagger.score_cutoff,
        "validation_micro_f1": metadata.validation_micro_f1,
        "frequency_cutoff": metadata.frequency_cutoff,
        "n_supported_tags": len(metadata.supported_tags),
        "split_sizes": {**metadata.split_sizes, split: n_questions},
        "hyperparameters": metadata.training_config,
        "seed": metadata.seed,
        "training_seconds": metadata.training_seconds,
        "git_commit": git_commit,
        "package_version": __version__,
        "tagger_package_version": metadata.package_version,
        "tagger_created_at": metadata.created_at,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def current_git_commit() -> str | None:
    """The commit checked out in the working directory, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def summary(metrics: Metrics) -> str:
    """A compact, human-readable view of the headline metrics."""
    at_k = "  ".join(
        f"P@{k} {metrics.precision_at_k[str(k)]:.3f}  R@{k} {metrics.recall_at_k[str(k)]:.3f}"
        for k in TOP_K
    )
    return "\n".join(
        [
            f"micro-F1  {metrics.micro_f1:.3f}  (precision {metrics.micro_precision:.3f}, "
            f"recall {metrics.micro_recall:.3f})",
            f"macro-F1  {metrics.macro_f1:.3f}",
            at_k,
            f"Suggestions per Question: {metrics.mean_suggestions_per_question:.2f}",
        ]
    )


def _suggested(ranks: npt.NDArray[np.intp], scores: ScoreMatrix, score_cutoff: float) -> BoolMatrix:
    passing = (scores >= score_cutoff).sum(axis=1)
    kept = np.clip(passing, MIN_SUGGESTIONS, MAX_SUGGESTIONS)
    suggested: BoolMatrix = ranks < kept[:, np.newaxis]
    return suggested


def _micro(predicted: BoolMatrix, true: BoolMatrix) -> tuple[float, float, float]:
    """Micro precision, recall and F1."""
    true_positives = int((predicted & true).sum())
    n_predicted = int(predicted.sum())
    n_true = int(true.sum())
    return (
        _ratio(true_positives, n_predicted),
        _ratio(true_positives, n_true),
        _ratio(2 * true_positives, n_predicted + n_true),
    )


def _ratio(numerator: float | np.integer[Any], denominator: float | np.integer[Any]) -> float:
    return float(numerator / denominator) if denominator else 0.0
