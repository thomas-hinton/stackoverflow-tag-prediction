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
    """Hyperparameters of the Linear tagger (its Score cut-off is tuned, not configured)."""

    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    C: float
    class_weight: str | None
    max_iter: int

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ConfigError(f"'C' must be positive, got {self.C}.")


@dataclass(frozen=True, slots=True)
class TransformerTaggerConfig:
    """Hyperparameters of the Transformer tagger (its Score cut-off is tuned, not configured)."""

    checkpoint: str
    """Pretrained model to fine-tune: a Hugging Face Hub id or a local directory.

    Tests point it at a tiny local checkpoint, which sets the architecture size too.
    """
    learning_rate: float
    batch_size: int
    epochs: int
    """Maximum number of epochs; early stopping may end training sooner."""
    warmup_ratio: float
    weight_decay: float
    patience: int
    """Epochs without a better validation micro-F1 before training stops."""
    max_length_percentile: float
    """Titles are truncated to the token length covering this percentile of training Titles."""
    inference_batch_size: int
    num_threads: int
    """CPU threads PyTorch may use; 0 keeps PyTorch's default."""

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigError(f"'learning_rate' must be positive, got {self.learning_rate}.")
        if not 0 <= self.warmup_ratio < 1:
            raise ConfigError(f"'warmup_ratio' must be in [0, 1), got {self.warmup_ratio}.")
        if self.weight_decay < 0:
            raise ConfigError(f"'weight_decay' must be >= 0, got {self.weight_decay}.")
        if not 0 < self.max_length_percentile <= 100:
            raise ConfigError(
                f"'max_length_percentile' must be in (0, 100], got {self.max_length_percentile}."
            )


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    split: SplitConfig
    linear_tagger: LinearTaggerConfig
    transformer_tagger: TransformerTaggerConfig


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
        transformer_tagger=_transformer_tagger(_section(raw, "transformer_tagger")),
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
    )


def _transformer_tagger(section: dict[str, Any]) -> TransformerTaggerConfig:
    checkpoint = section.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ConfigError(f"'checkpoint' must be a non-empty string, got {checkpoint!r}.")
    return TransformerTaggerConfig(
        checkpoint=checkpoint,
        learning_rate=_float(section, "learning_rate"),
        batch_size=_int(section, "batch_size", minimum=1),
        epochs=_int(section, "epochs", minimum=1),
        warmup_ratio=_float(section, "warmup_ratio"),
        weight_decay=_float(section, "weight_decay"),
        patience=_int(section, "patience", minimum=1),
        max_length_percentile=_float(section, "max_length_percentile"),
        inference_batch_size=_int(section, "inference_batch_size", minimum=1),
        num_threads=_int(section, "num_threads", minimum=0),
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
