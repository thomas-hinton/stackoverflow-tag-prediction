"""The Suggestion policy: how per-Tag Scores become a Suggestion list.

It is a pure function with no knowledge of any model, shared by every Tagger.
"""

from collections.abc import Mapping

from so_tagger.domain import Suggestion, SuggestionList

MIN_SUGGESTIONS = 1
MAX_SUGGESTIONS = 5


def apply_suggestion_policy(
    scores: Mapping[str, float],
    score_cutoff: float,
    *,
    min_suggestions: int = MIN_SUGGESTIONS,
    max_suggestions: int = MAX_SUGGESTIONS,
) -> SuggestionList:
    """Rank Tags by descending Score and keep those whose Score reaches the cut-off.

    Ties are broken by Tag name so the result is deterministic. If fewer than
    ``min_suggestions`` Tags pass the cut-off, the best-ranked Tags fill the list up to that
    minimum. The list is truncated to ``max_suggestions``.
    """
    if not scores:
        raise ValueError("The Suggestion policy needs Scores for at least one Tag.")
    if not 1 <= min_suggestions <= max_suggestions:
        raise ValueError(
            f"Expected 1 <= min_suggestions <= max_suggestions, "
            f"got {min_suggestions} and {max_suggestions}."
        )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    passing = sum(1 for _, score in ranked if score >= score_cutoff)
    kept = min(max(passing, min_suggestions), max_suggestions)
    return tuple(Suggestion(tag, float(score)) for tag, score in ranked[:kept])
