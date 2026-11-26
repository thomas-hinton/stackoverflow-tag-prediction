import pytest

from so_tagger.domain import Suggestion
from so_tagger.suggestion_policy import (
    apply_suggestion_policy,
)


def test_keeps_tags_at_or_above_the_cutoff_ranked_by_descending_score() -> None:
    scores = {"python": 0.9, "json": 0.5, "sql": 0.49, "css": 0.1}

    suggestions = apply_suggestion_policy(scores, score_cutoff=0.5)

    assert suggestions == (Suggestion("python", 0.9), Suggestion("json", 0.5))


def test_returns_the_single_best_tag_when_none_passes_the_cutoff() -> None:
    scores = {"python": 0.2, "json": 0.3, "sql": 0.1}

    suggestions = apply_suggestion_policy(scores, score_cutoff=0.8)

    assert suggestions == (Suggestion("json", 0.3),)


def test_never_returns_more_than_five_suggestions() -> None:
    scores = {f"tag{i}": 0.9 - i / 100 for i in range(8)}

    suggestions = apply_suggestion_policy(scores, score_cutoff=0.0)

    assert [s.tag for s in suggestions] == ["tag0", "tag1", "tag2", "tag3", "tag4"]


def test_breaks_score_ties_by_tag_name() -> None:
    scores = {"sql": 0.7, "css": 0.7, "python": 0.7, "java": 0.2}

    suggestions = apply_suggestion_policy(scores, score_cutoff=0.5)

    assert [s.tag for s in suggestions] == ["css", "python", "sql"]


def test_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        apply_suggestion_policy({}, score_cutoff=0.5)
