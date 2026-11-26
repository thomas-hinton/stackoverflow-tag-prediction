import tomllib
from pathlib import Path

import pytest

from so_tagger.config import ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def test_config_defines_a_positive_frequency_cutoff() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    cutoff = config["data"]["frequency_cutoff"]

    assert isinstance(cutoff, int) and not isinstance(cutoff, bool)
    assert cutoff >= 1


def test_project_config_loads_with_the_documented_defaults() -> None:
    config = load_config(CONFIG_PATH)

    assert config.split.ratios() == (0.70, 0.15, 0.15)
    assert config.data.frequency_cutoff == 20
    assert 0.0 <= config.linear_tagger.score_cutoff <= 1.0


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("test_ratio = 0.15", "test_ratio = 0.2")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)
