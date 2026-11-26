"""The quality gate: a Tagger is released only if its test micro-F1 reaches a floor.

The floor of each Tagger kind lives in ``[quality_gate.min_test_micro_f1]`` of the project
configuration. ``gate`` reads a metrics file written by ``evaluate`` on the test split and
compares its micro-F1 with that floor; the training workflow publishes no Release when it fails.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class QualityGateError(ValueError):
    """The metrics file cannot be checked against the quality gate."""


@dataclass(frozen=True, slots=True)
class GateResult:
    """A Tagger's test micro-F1 next to the floor of its kind."""

    tagger: str
    test_micro_f1: float
    floor: float

    @property
    def passed(self) -> bool:
        return self.test_micro_f1 >= self.floor

    @property
    def difference(self) -> float:
        """How far the test micro-F1 is above the floor (negative when below)."""
        return self.test_micro_f1 - self.floor

    def message(self) -> str:
        verdict = "passed" if self.passed else "failed"
        relation = "reaches" if self.passed else "is below"
        return (
            f"Quality gate {verdict} for the {self.tagger} Tagger: test micro-F1 "
            f"{self.test_micro_f1:.4f} {relation} the floor {self.floor:.4f} "
            f"(difference {self.difference:+.4f})."
        )


def read_metrics_file(path: Path) -> dict[str, Any]:
    """Read a metrics file written by ``evaluate``."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise QualityGateError(f"No metrics file at {path}: run `evaluate` first.") from error
    except json.JSONDecodeError as error:
        raise QualityGateError(f"{path} is not a metrics file: {error}.") from error
    if not isinstance(report, dict):
        raise QualityGateError(f"{path} is not a metrics file: it is not a JSON object.")
    return report


def check_quality_gate(report: Mapping[str, Any], floors: Mapping[str, float]) -> GateResult:
    """Compare the test micro-F1 of a metrics file with the floor of its Tagger kind.

    ``floors`` maps each Tagger kind to its minimum test micro-F1. Raises ``QualityGateError``
    when the metrics file does not measure the test split or its Tagger kind has no floor.
    """
    tagger = report.get("tagger")
    if not isinstance(tagger, str):
        raise QualityGateError("The metrics file has no 'tagger' key.")
    if report.get("split") != "test":
        raise QualityGateError(
            f"The quality gate checks the test split, but the metrics file measures the "
            f"{report.get('split')!r} split: run `evaluate --split test`."
        )
    if tagger not in floors:
        raise QualityGateError(
            f"No quality-gate floor for the '{tagger}' Tagger: add it to "
            f"[quality_gate.min_test_micro_f1] in the configuration."
        )
    metrics = report.get("metrics")
    micro_f1 = metrics.get("micro_f1") if isinstance(metrics, dict) else None
    if not isinstance(micro_f1, int | float) or isinstance(micro_f1, bool):
        raise QualityGateError("The metrics file has no numeric 'metrics.micro_f1'.")
    return GateResult(tagger=tagger, test_micro_f1=float(micro_f1), floor=floors[tagger])
