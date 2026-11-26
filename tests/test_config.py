import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def test_config_defines_a_positive_frequency_cutoff() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    cutoff = config["data"]["frequency_cutoff"]

    assert isinstance(cutoff, int) and not isinstance(cutoff, bool)
    assert cutoff >= 1
