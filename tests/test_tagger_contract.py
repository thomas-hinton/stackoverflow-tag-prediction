"""Seam 2: the Tagger contract, one suite run against every Tagger kind.

#10 adds the Transformer tagger to TAGGER_KINDS; this suite then covers it unchanged.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

import so_tagger
from so_tagger.config import load_config
from so_tagger.dataset import prepare_questions, read_raw_questions
from so_tagger.evaluation import SCORE_CUTOFF_GRID
from so_tagger.split import split_questions
from so_tagger.suggestion_policy import (
    MAX_SUGGESTIONS,
    MIN_SUGGESTIONS,
    apply_suggestion_policy,
)
from so_tagger.taggers import TAGGER_KINDS, Tagger, TaggerError, load_tagger, train_tagger
from tests.conftest import FIXTURE_CONFIG, FIXTURE_DATASET, FIXTURE_SUPPORTED_TAGS

TITLES = [
    "How do I parse JSON in Python?",
    "Center a div with CSS flexbox",
    "SQL query to join two tables",
    "JavaScript fetch returns a promise",
]
OFF_TOPIC_TITLE = "Hello, how are you?"


@pytest.fixture(scope="module", params=TAGGER_KINDS)
def tagger(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Tagger:
    config_path = tmp_path_factory.mktemp("config") / "config.toml"
    config_path.write_text(FIXTURE_CONFIG, encoding="utf-8")
    config = load_config(config_path)
    dataset = prepare_questions(read_raw_questions(FIXTURE_DATASET), config.data.frequency_cutoff)
    dataset_split = split_questions(
        dataset.questions, dataset.supported_tags, config.split, config.seed
    )
    kind: str = request.param
    return train_tagger(kind, dataset_split.train, dataset_split.validation, config)


@pytest.mark.parametrize("title", TITLES)
def test_a_title_yields_one_to_five_suggestions_by_descending_score(
    tagger: Tagger, title: str
) -> None:
    suggestions = tagger.suggest(title)

    assert MIN_SUGGESTIONS <= len(suggestions) <= MAX_SUGGESTIONS
    scores = [suggestion.score for suggestion in suggestions]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("title", TITLES)
def test_every_suggestion_is_a_supported_tag_with_a_score_in_unit_range(
    tagger: Tagger, title: str
) -> None:
    for suggestion in tagger.suggest(title):
        assert suggestion.tag in tagger.supported_tags
        assert 0.0 <= suggestion.score <= 1.0


def test_supported_tags_are_those_of_the_training_questions(tagger: Tagger) -> None:
    assert tagger.supported_tags == FIXTURE_SUPPORTED_TAGS


def test_an_off_topic_title_still_yields_a_suggestion(tagger: Tagger) -> None:
    assert len(tagger.suggest(OFF_TOPIC_TITLE)) >= 1


def test_batch_scores_have_one_row_per_title_and_one_column_per_supported_tag(
    tagger: Tagger,
) -> None:
    scores = tagger.score(TITLES)

    assert scores.shape == (len(TITLES), len(tagger.supported_tags))
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_saving_then_loading_gives_identical_suggestions(tagger: Tagger, tmp_path: Path) -> None:
    tagger.save(tmp_path / "tagger")

    loaded = load_tagger(tmp_path / "tagger")

    assert loaded.kind == tagger.kind
    assert loaded.supported_tags == tagger.supported_tags
    assert loaded.score_cutoff == tagger.score_cutoff
    for title in [*TITLES, OFF_TOPIC_TITLE]:
        assert loaded.suggest(title) == tagger.suggest(title)


def test_saved_directory_holds_the_tagger_metadata(tagger: Tagger, tmp_path: Path) -> None:
    tagger.save(tmp_path / "tagger")

    metadata = json.loads((tmp_path / "tagger" / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["kind"] == tagger.kind
    assert tuple(metadata["supported_tags"]) == FIXTURE_SUPPORTED_TAGS
    assert metadata["score_cutoff"] == tagger.score_cutoff
    assert 0.0 <= metadata["validation_micro_f1"] <= 1.0
    assert metadata["frequency_cutoff"] == 5
    assert metadata["seed"] == 7
    assert metadata["split_sizes"] == {"train": 50, "validation": 11}
    assert metadata["training_seconds"] > 0
    assert metadata["package_version"] == so_tagger.__version__
    assert isinstance(metadata["training_config"], dict) and metadata["training_config"]
    assert datetime.fromisoformat(metadata["created_at"]).tzinfo is not None


def test_a_loaded_tagger_exposes_the_tuned_cutoff_and_suggest_uses_it(
    tagger: Tagger, tmp_path: Path
) -> None:
    tagger.save(tmp_path / "tagger")

    loaded = load_tagger(tmp_path / "tagger")

    assert loaded.score_cutoff in SCORE_CUTOFF_GRID
    assert loaded.score_cutoff == tagger.score_cutoff
    for title in [*TITLES, OFF_TOPIC_TITLE]:
        scores = dict(zip(loaded.supported_tags, loaded.score([title])[0].tolist(), strict=True))
        assert loaded.suggest(title) == apply_suggestion_policy(scores, loaded.score_cutoff)


def test_a_tagger_directory_without_a_tuned_cutoff_is_rejected(
    tagger: Tagger, tmp_path: Path
) -> None:
    tagger.save(tmp_path / "tagger")
    path = tmp_path / "tagger" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**metadata, "score_cutoff": None}), encoding="utf-8")

    with pytest.raises(TaggerError, match="no tuned Score cut-off"):
        load_tagger(tmp_path / "tagger")
