from pathlib import Path

import pytest

from so_tagger.config import ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def test_project_config_loads_with_the_documented_defaults() -> None:
    config = load_config(CONFIG_PATH)

    assert config.split.ratios() == (0.70, 0.15, 0.15)
    assert config.data.frequency_cutoff == 20
    assert config.linear_tagger.C > 0


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("test_ratio = 0.15", "test_ratio = 0.2")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)


def test_project_config_holds_the_transformer_tagger_defaults() -> None:
    settings = load_config(CONFIG_PATH).transformer_tagger

    assert settings.checkpoint == "distilbert-base-uncased"
    assert (settings.learning_rate, settings.batch_size, settings.patience) == (1e-4, 16, 2)
    assert (settings.warmup_ratio, settings.weight_decay) == (0.1, 0.01)


@pytest.mark.parametrize(
    ("line", "replacement", "message"),
    [
        ("learning_rate = 1e-4", "learning_rate = 0", "'learning_rate' must be positive"),
        ("warmup_ratio = 0.1", "warmup_ratio = 1.0", "'warmup_ratio' must be in"),
        ("weight_decay = 0.01", "weight_decay = -0.1", "'weight_decay' must be >= 0"),
        ("max_length_percentile = 99", "max_length_percentile = 0", "'max_length_percentile'"),
        ('checkpoint = "distilbert-base-uncased"', 'checkpoint = ""', "'checkpoint' must be"),
        ("num_threads = 0", "num_threads = -1", "'num_threads' must be an integer >= 0"),
    ],
)
def test_invalid_transformer_tagger_settings_are_rejected(
    tmp_path: Path, line: str, replacement: str, message: str
) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert line in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(line, replacement), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_project_config_holds_a_quality_gate_floor_per_tagger_kind() -> None:
    floors = load_config(CONFIG_PATH).quality_gate.min_test_micro_f1

    assert set(floors) == {"linear", "transformer"}
    assert all(0 < floor < 1 for floor in floors.values())


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("linear = 1.5", "'linear' quality-gate floor must be in \\[0, 1\\]"),
        ('linear = "high"', "'linear' must be a number"),
    ],
)
def test_invalid_quality_gate_floors_are_rejected(
    tmp_path: Path, replacement: str, message: str
) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(text.replace("linear = 0.20", replacement), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_a_configuration_without_quality_gate_floors_is_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(text.split("[quality_gate.min_test_micro_f1]")[0], encoding="utf-8")

    with pytest.raises(ConfigError, match="quality_gate"):
        load_config(path)
