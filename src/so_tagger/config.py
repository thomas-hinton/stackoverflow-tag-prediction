"""Typed access to the versioned project configuration (``config.toml``)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.toml")


class ConfigError(ValueError):
    """The project configuration is missing a value or holds an invalid one."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    frequency_cutoff: int


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        ratios = self.ratios()
        if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
            raise ConfigError(f"Split ratios must be positive and sum to 1, got {ratios}.")

    def ratios(self) -> tuple[float, float, float]:
        """The train, validation and test ratios, in that order."""
        return (self.train_ratio, self.validation_ratio, self.test_ratio)


@dataclass(frozen=True, slots=True)
class LinearTaggerConfig:
    """Hyperparameters of the Linear tagger and its provisional Score cut-off."""

    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    C: float
    class_weight: str | None
    max_iter: int
    score_cutoff: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_cutoff <= 1.0:
            raise ConfigError(f"'score_cutoff' must be in [0, 1], got {self.score_cutoff}.")
        if self.C <= 0:
            raise ConfigError(f"'C' must be positive, got {self.C}.")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    split: SplitConfig
    linear_tagger: LinearTaggerConfig


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read and validate the project configuration file."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {path}") from error
    return ProjectConfig(
        seed=_int(raw, "seed", minimum=0),
        data=DataConfig(frequency_cutoff=_int(_section(raw, "data"), "frequency_cutoff", 1)),
        split=SplitConfig(
            train_ratio=_float(_section(raw, "split"), "train_ratio"),
            validation_ratio=_float(_section(raw, "split"), "validation_ratio"),
            test_ratio=_float(_section(raw, "split"), "test_ratio"),
        ),
        linear_tagger=_linear_tagger(_section(raw, "linear_tagger")),
    )


def _linear_tagger(section: dict[str, Any]) -> LinearTaggerConfig:
    class_weight = section.get("class_weight")
    if class_weight not in ("balanced", "none"):
        raise ConfigError(f'\'class_weight\' must be "balanced" or "none", got {class_weight!r}.')
    sublinear_tf = section.get("sublinear_tf")
    if not isinstance(sublinear_tf, bool):
        raise ConfigError(f"'sublinear_tf' must be a boolean, got {sublinear_tf!r}.")
    return LinearTaggerConfig(
        word_ngram_range=_ngram_range(section, "word_ngram_range"),
        char_ngram_range=_ngram_range(section, "char_ngram_range"),
        min_df=_int(section, "min_df", minimum=1),
        sublinear_tf=sublinear_tf,
        C=_float(section, "C"),
        class_weight=None if class_weight == "none" else class_weight,
        max_iter=_int(section, "max_iter", minimum=1),
        score_cutoff=_float(section, "score_cutoff"),
    )


def _ngram_range(section: dict[str, Any], key: str) -> tuple[int, int]:
    value = section.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in value)
        or not 1 <= value[0] <= value[1]
    ):
        raise ConfigError(f"'{key}' must be [low, high] with 1 <= low <= high, got {value!r}.")
    return (value[0], value[1])


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing [{name}] section in the configuration.")
    return section


def _float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a number, got {value!r}.")
    return float(value)


def _int(section: dict[str, Any], key: str, minimum: int) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"'{key}' must be an integer >= {minimum}, got {value!r}.")
    return value
