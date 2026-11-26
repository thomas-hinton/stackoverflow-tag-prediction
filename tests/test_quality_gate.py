"""The quality gate: `gate` passes a Tagger whose test micro-F1 reaches its floor, fails others."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from so_tagger.cli import app
from so_tagger.quality_gate import QualityGateError, check_quality_gate

runner = CliRunner()
FLOORS = {"linear": 0.20, "transformer": 0.20}


def metrics_report(micro_f1: float, tagger: str = "linear", split: str = "test") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tagger": tagger,
        "split": split,
        "metrics": {"micro_f1": micro_f1},
    }


def write_metrics(path: Path, micro_f1: float, **kwargs: str) -> Path:
    path.write_text(json.dumps(metrics_report(micro_f1, **kwargs)), encoding="utf-8")
    return path


def gate(*args: str | Path) -> Result:
    return runner.invoke(app, ["gate", *(str(arg) for arg in args)])


def test_a_tagger_above_its_floor_passes() -> None:
    result = check_quality_gate(metrics_report(0.51), FLOORS)

    assert result.passed
    assert result.difference == pytest.approx(0.31)


def test_a_tagger_exactly_at_its_floor_passes() -> None:
    assert check_quality_gate(metrics_report(0.20), FLOORS).passed


def test_a_tagger_below_its_floor_fails_with_both_numbers_and_the_difference() -> None:
    result = check_quality_gate(metrics_report(0.15), FLOORS)

    assert not result.passed
    message = result.message()
    assert "failed" in message
    assert "0.1500" in message
    assert "0.2000" in message
    assert "-0.0500" in message


def test_each_tagger_kind_is_held_to_its_own_floor() -> None:
    floors = {"linear": 0.20, "transformer": 0.60}

    assert check_quality_gate(metrics_report(0.5, tagger="linear"), floors).passed
    assert not check_quality_gate(metrics_report(0.5, tagger="transformer"), floors).passed


def test_the_gate_only_checks_the_test_split() -> None:
    with pytest.raises(QualityGateError, match="test split"):
        check_quality_gate(metrics_report(0.9, split="validation"), FLOORS)


def test_a_tagger_kind_without_a_floor_is_rejected() -> None:
    with pytest.raises(QualityGateError, match="No quality-gate floor for the 'linear' Tagger"):
        check_quality_gate(metrics_report(0.9), {"transformer": 0.2})


def test_gate_command_passes_above_the_configured_floor(
    tmp_path: Path, fixture_config: Path
) -> None:
    metrics = write_metrics(tmp_path / "linear-test.json", 0.51)

    result = gate("--metrics", metrics, "--config", fixture_config)

    assert result.exit_code == 0, result.output
    assert "Quality gate passed for the linear Tagger" in result.output
    assert "0.5100" in result.output
    assert "0.2000" in result.output


def test_gate_command_fails_below_the_configured_floor_and_shows_both_numbers(
    tmp_path: Path, fixture_config: Path
) -> None:
    metrics = write_metrics(tmp_path / "linear-test.json", 0.12)

    result = gate("--metrics", metrics, "--config", fixture_config)

    assert result.exit_code == 1
    assert "Quality gate failed for the linear Tagger" in result.output
    assert "0.1200" in result.output
    assert "0.2000" in result.output
    assert "-0.0800" in result.output


def test_gate_command_reads_the_floor_from_the_configuration(
    tmp_path: Path, fixture_config_toml: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        fixture_config_toml.replace("linear = 0.20", "linear = 0.60"), encoding="utf-8"
    )
    metrics = write_metrics(tmp_path / "linear-test.json", 0.51)

    result = gate("--metrics", metrics, "--config", config)

    assert result.exit_code == 1
    assert "0.6000" in result.output


def test_gate_command_rejects_a_metrics_file_of_another_tagger(
    tmp_path: Path, fixture_config: Path
) -> None:
    metrics = write_metrics(tmp_path / "metrics.json", 0.9, tagger="transformer")

    result = gate("--tagger", "linear", "--metrics", metrics, "--config", fixture_config)

    assert result.exit_code == 1
    assert "not the 'linear' one" in result.output


def test_gate_command_fails_clearly_without_a_metrics_file(
    tmp_path: Path, fixture_config: Path
) -> None:
    result = gate("--metrics", tmp_path / "missing.json", "--config", fixture_config)

    assert result.exit_code == 1
    assert "run `evaluate` first" in result.output


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ({"split": "test", "metrics": {"micro_f1": 0.5}}, "no 'tagger' key"),
        ({"tagger": "linear", "split": "test", "metrics": {}}, "no numeric 'metrics.micro_f1'"),
        ({"tagger": "linear", "split": "test", "metrics": {"micro_f1": "0.5"}}, "numeric"),
    ],
)
def test_an_incomplete_metrics_file_is_rejected(report: dict[str, Any], message: str) -> None:
    with pytest.raises(QualityGateError, match=message):
        check_quality_gate(report, FLOORS)


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_gate_command_rejects_a_file_that_is_not_a_metrics_file(
    tmp_path: Path, fixture_config: Path, content: str
) -> None:
    metrics = tmp_path / "linear-test.json"
    metrics.write_text(content, encoding="utf-8")

    result = gate("--metrics", metrics, "--config", fixture_config)

    assert result.exit_code == 1
    assert "is not a metrics file" in result.output
