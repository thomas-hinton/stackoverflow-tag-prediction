"""Metrics on small hand-computed cases, and Score cut-off tuning on the fixture data."""

import numpy as np
import pytest

from so_tagger.config import load_config
from so_tagger.dataset import prepare_questions, read_raw_questions
from so_tagger.domain import Question
from so_tagger.evaluation import (
    SCORE_CUTOFF_GRID,
    EvaluationError,
    TagMetrics,
    compute_metrics,
    score_ranks,
    suggestion_matrix,
    true_tag_matrix,
)
from so_tagger.split import split_questions
from so_tagger.suggestion_policy import apply_suggestion_policy
from so_tagger.taggers import TAGGER_KINDS, train_tagger
from tests.conftest import FIXTURE_CONFIG, FIXTURE_DATASET

# Two Questions, four Supported tags. `d` is never true and never suggested at cut-off 0.5.
TAGS = ("a", "b", "c", "d")
SCORES = np.array(
    [
        [0.9, 0.6, 0.1, 0.0],  # true {a}: suggests a, b
        [0.2, 0.3, 0.8, 0.05],  # true {b, c}: suggests c
    ]
)
TRUE = np.array([[True, False, False, False], [False, True, True, False]])


def test_micro_scores_pool_every_decision() -> None:
    metrics = compute_metrics(SCORES, TRUE, TAGS, score_cutoff=0.5)

    # 2 true positives (a in Q1, c in Q2), 3 predicted, 3 true.
    assert metrics.micro_precision == pytest.approx(2 / 3)
    assert metrics.micro_recall == pytest.approx(2 / 3)
    assert metrics.micro_f1 == pytest.approx(2 / 3)
    assert metrics.mean_suggestions_per_question == pytest.approx(1.5)


def test_macro_f1_counts_a_tag_with_no_true_and_no_predicted_example_as_zero() -> None:
    metrics = compute_metrics(SCORES, TRUE, TAGS, score_cutoff=0.5)

    assert metrics.per_tag == {
        "a": TagMetrics(precision=1.0, recall=1.0, f1=1.0, support=1),
        "b": TagMetrics(precision=0.0, recall=0.0, f1=0.0, support=1),
        "c": TagMetrics(precision=1.0, recall=1.0, f1=1.0, support=1),
        "d": TagMetrics(precision=0.0, recall=0.0, f1=0.0, support=0),
    }
    assert metrics.macro_f1 == pytest.approx((1 + 0 + 1 + 0) / 4)


def test_precision_and_recall_at_k_use_the_score_ranking_not_the_cutoff() -> None:
    # A cut-off of 0.99 keeps a single Suggestion per Question; @k must not change.
    metrics = compute_metrics(SCORES, TRUE, TAGS, score_cutoff=0.99)

    # Top-1: a | c -> 1 and 1 hit. Top-3: a, b, c | c, b, a -> 1 and 2 hits.
    # Top-5 holds all 4 Tags, still divided by k = 5.
    assert metrics.precision_at_k == pytest.approx({"1": 1.0, "3": 0.5, "5": 0.3})
    assert metrics.recall_at_k == pytest.approx({"1": 0.75, "3": 1.0, "5": 1.0})


def test_ranking_breaks_score_ties_by_tag_name() -> None:
    ranks = score_ranks(np.array([[0.5, 0.5, 0.9]]), ("zeta", "alpha", "mid"))

    assert ranks.tolist() == [[2, 1, 0]]


@pytest.mark.parametrize("cutoff", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_batch_suggestions_match_the_suggestion_policy(cutoff: float) -> None:
    tags = tuple(f"tag{i:02d}" for i in range(12))
    scores = np.random.default_rng(0).random((50, len(tags)))
    scores[:, 3] = scores[:, 4]  # force ties

    predicted = suggestion_matrix(scores, tags, cutoff)

    for row, suggested in zip(scores, predicted, strict=True):
        policy = apply_suggestion_policy(dict(zip(tags, row.tolist(), strict=True)), cutoff)
        assert {tags[i] for i in np.flatnonzero(suggested)} == {s.tag for s in policy}


def test_true_tags_ignore_unsupported_tags_and_reject_questions_without_any() -> None:
    questions = [Question(1, "t", ("b", "rust")), Question(2, "t", ("a",))]

    assert true_tag_matrix(questions, ("a", "b")).tolist() == [[False, True], [True, False]]
    with pytest.raises(EvaluationError, match="Question 3 carries no Supported tag"):
        true_tag_matrix([Question(3, "t", ("rust",))], ("a", "b"))


def _independent_best_cutoff(
    scores: list[dict[str, float]], true: list[set[str]]
) -> tuple[float, float]:
    """Brute force over the grid with the scalar Suggestion policy and set arithmetic."""
    best_cutoff, best_f1 = 0.0, -1.0
    for cutoff in SCORE_CUTOFF_GRID:
        predicted = [{s.tag for s in apply_suggestion_policy(row, cutoff)} for row in scores]
        hits = sum(len(p & t) for p, t in zip(predicted, true, strict=True))
        total = sum(len(p) for p in predicted) + sum(len(t) for t in true)
        f1 = 2 * hits / total
        if f1 > best_f1:  # strict: ties keep the smallest cut-off
            best_cutoff, best_f1 = cutoff, f1
    return best_cutoff, best_f1


@pytest.mark.parametrize("kind", TAGGER_KINDS)
def test_tuned_cutoff_is_the_best_validation_micro_f1(
    kind: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    config_path = tmp_path_factory.mktemp("config") / "config.toml"
    config_path.write_text(FIXTURE_CONFIG, encoding="utf-8")
    config = load_config(config_path)
    dataset = prepare_questions(read_raw_questions(FIXTURE_DATASET), config.data.frequency_cutoff)
    split = split_questions(dataset.questions, dataset.supported_tags, config.split, config.seed)

    tagger = train_tagger(kind, split.train, split.validation, config)

    scores = tagger.score([question.title for question in split.validation])
    rows = [dict(zip(tagger.supported_tags, row.tolist(), strict=True)) for row in scores]
    expected_cutoff, expected_f1 = _independent_best_cutoff(
        rows, [set(question.tags) for question in split.validation]
    )
    assert tagger.score_cutoff == expected_cutoff
    assert tagger.metadata.validation_micro_f1 == pytest.approx(expected_f1)
