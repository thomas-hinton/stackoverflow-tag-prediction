"""What is specific to the Transformer tagger: maximum length, epochs, seeding, offline loading.

The shared behaviour (Suggestions, save and load, tuned Score cut-off) is covered for every
Tagger kind by the Tagger contract suite.
"""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from so_tagger.config import ProjectConfig, load_config
from so_tagger.dataset import prepare_questions, read_raw_questions
from so_tagger.split import DatasetSplit, split_questions
from so_tagger.taggers import TaggerError, load_tagger, train_tagger
from tests.conftest import FIXTURE_DATASET

pytest.importorskip("torch")
pytest.importorskip("transformers")

TITLES = ["How do I parse JSON in Python?", "Center a div with CSS flexbox"]


def configure(tmp_path: Path, config_toml: str, **overrides: str) -> ProjectConfig:
    for key, value in overrides.items():
        lines = [
            f"{key} = {value}" if line.startswith(f"{key} = ") else line
            for line in config_toml.splitlines()
        ]
        config_toml = "\n".join(lines)
    path = tmp_path / "config.toml"
    path.write_text(config_toml, encoding="utf-8")
    return load_config(path)


@pytest.fixture(scope="module")
def dataset_split(
    fixture_config_toml: str, tmp_path_factory: pytest.TempPathFactory
) -> DatasetSplit:
    config = configure(tmp_path_factory.mktemp("config"), fixture_config_toml)
    dataset = prepare_questions(read_raw_questions(FIXTURE_DATASET), config.data.frequency_cutoff)
    return split_questions(dataset.questions, dataset.supported_tags, config.split, config.seed)


def test_maximum_length_covers_the_99th_percentile_of_training_titles(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit, tiny_checkpoint: Path
) -> None:
    from transformers import AutoTokenizer

    config = configure(tmp_path, fixture_config_toml)

    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    tokenizer = AutoTokenizer.from_pretrained(tiny_checkpoint)
    lengths = [len(tokenizer(question.title)["input_ids"]) for question in dataset_split.train]
    max_length = tagger.metadata.training_details["max_length"]
    assert max_length == math.ceil(np.percentile(lengths, 99))
    assert np.mean(np.asarray(lengths) <= max_length) >= 0.99


def test_the_best_validation_epoch_is_kept(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit
) -> None:
    config = configure(tmp_path, fixture_config_toml, epochs="3", patience="3")

    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    details = tagger.metadata.training_details
    f1s = [entry["validation_micro_f1"] for entry in details["history"]]
    assert details["epochs_run"] == 3
    assert details["best_epoch"] == 1 + f1s.index(max(f1s))
    # The kept weights are those of the best epoch: tuning on them reaches its micro-F1.
    assert tagger.metadata.validation_micro_f1 == pytest.approx(max(f1s), abs=1e-6)
    for entry in details["history"]:
        assert set(entry) == {"epoch", "loss", "validation_micro_f1", "elapsed_seconds"}


def test_training_stops_early_after_patience_epochs_without_improvement(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit
) -> None:
    # A learning rate this small leaves the validation micro-F1 unchanged after epoch 1.
    config = configure(tmp_path, fixture_config_toml, epochs="5", learning_rate="1e-12")

    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    details = tagger.metadata.training_details
    assert (details["best_epoch"], details["epochs_run"]) == (1, 2)


def test_training_twice_with_the_same_seed_gives_the_same_scores(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit
) -> None:
    config = configure(tmp_path, fixture_config_toml)

    first = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)
    second = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    np.testing.assert_array_equal(first.score(TITLES), second.score(TITLES))
    assert first.score_cutoff == second.score_cutoff


def test_a_saved_tagger_loads_offline_from_its_directory_alone(
    tmp_path: Path,
    fixture_config_toml: str,
    dataset_split: DatasetSplit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = configure(tmp_path, fixture_config_toml)
    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)
    tagger.save(tmp_path / "saved")
    shutil.move(tmp_path / "saved", tmp_path / "moved")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    loaded = load_tagger(tmp_path / "moved")

    np.testing.assert_array_equal(loaded.score(TITLES), tagger.score(TITLES))


def test_scoring_no_title_gives_an_empty_score_matrix(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit
) -> None:
    config = configure(tmp_path, fixture_config_toml, epochs="1")
    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    assert tagger.score([]).shape == (0, len(tagger.supported_tags))


def test_without_the_transformer_extra_the_error_says_how_to_install_it(
    tmp_path: Path,
    fixture_config_toml: str,
    dataset_split: DatasetSplit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = configure(tmp_path, fixture_config_toml)
    monkeypatch.delitem(sys.modules, "so_tagger.taggers.transformer")
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(TaggerError, match="uv sync --extra transformer"):
        train_tagger("transformer", dataset_split.train, dataset_split.validation, config)


@pytest.mark.slow
def test_the_full_distilbert_checkpoint_fine_tunes_on_the_fixture_dataset(
    tmp_path: Path, fixture_config_toml: str, dataset_split: DatasetSplit
) -> None:
    """Downloads `distilbert-base-uncased` (about 250 MB) on first run."""
    config = configure(
        tmp_path,
        fixture_config_toml,
        checkpoint='"distilbert-base-uncased"',
        learning_rate="5e-5",
        epochs="1",
    )

    tagger = train_tagger("transformer", dataset_split.train, dataset_split.validation, config)

    assert tagger.metadata.training_details["max_length"] <= 64
    assert 1 <= len(tagger.suggest(TITLES[0])) <= 5
