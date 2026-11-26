"""Domain types shared by data preparation, the Taggers and the interfaces.

Names follow GLOSSARY.md: a Question has a Title and Tags, a Tagger turns a Title into a
Suggestion list, and each Suggestion is a Supported tag with its Score.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Question:
    """A StackOverflow Question: its post id, its Title and its ordered Tags."""

    id: int
    title: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A Supported tag proposed for a Title, with its Score in [0, 1]."""

    tag: str
    score: float


SuggestionList = tuple[Suggestion, ...]
"""Suggestions for one Title, ranked by descending Score; between one and five entries."""
