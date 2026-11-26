"""Plot the Transformer tagger's validation micro-F1 per epoch against the Linear tagger.

Reads the per-epoch ``training_details.history`` saved in the Transformer tagger's
``metadata.json`` and draws its validation micro-F1 after each epoch, with the Linear tagger's
``validation_micro_f1`` as a horizontal reference. The image is embedded in ``docs/metrics.md``.

Regenerate from the repository root (needs the ``explore`` dependency group for matplotlib):

    uv run --group explore python scripts/plot_training_curve.py

Options: ``--transformer-dir`` (default ``artifacts/transformer``), ``--linear-dir`` (default
``artifacts/linear``) and ``--output`` (default ``docs/images/training-curve.png``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
TEXT_PRIMARY = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e5e5e5"
SERIES = "#257de0"  # Reference categorical palette, first slot.


def read_metadata(tagger_dir: Path) -> dict[str, Any]:
    with (tagger_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata: dict[str, Any] = json.load(file)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transformer-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--linear-dir", type=Path, default=Path("artifacts/linear"))
    parser.add_argument("--output", type=Path, default=Path("docs/images/training-curve.png"))
    args = parser.parse_args()

    transformer = read_metadata(args.transformer_dir)
    history: list[dict[str, Any]] = transformer["training_details"]["history"]
    best_epoch: int = transformer["training_details"]["best_epoch"]
    linear_f1: float = read_metadata(args.linear_dir)["validation_micro_f1"]

    epochs = [int(entry["epoch"]) for entry in history]
    f1 = [float(entry["validation_micro_f1"]) for entry in history]
    minutes = float(history[-1]["elapsed_seconds"]) / 60
    best_f1 = f1[epochs.index(best_epoch)]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(linear_f1, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(
        f"Linear tagger {linear_f1:.3f}",
        xy=(epochs[0], linear_f1),
        xytext=(0, 6),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
    )

    ax.plot(epochs, f1, color=SERIES, linewidth=2, marker="o", markersize=8, mec=SURFACE)
    ax.annotate(
        f"Transformer, best epoch {best_epoch}\nmicro-F1 {best_f1:.3f}",
        xy=(best_epoch, best_f1),
        xytext=(-10, -34),
        textcoords="offset points",
        ha="right",
        color=TEXT_PRIMARY,
        fontsize=9,
    )

    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Epoch", color=TEXT_PRIMARY)
    ax.set_ylabel("Validation micro-F1", color=TEXT_PRIMARY)
    ax.set_title(
        f"Transformer tagger validation micro-F1 per epoch ({minutes:.0f} min on CPU)",
        color=TEXT_PRIMARY,
        fontsize=11,
        loc="left",
    )
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"Wrote {args.output} (best epoch {best_epoch}, micro-F1 {best_f1:.3f})")


if __name__ == "__main__":
    main()
