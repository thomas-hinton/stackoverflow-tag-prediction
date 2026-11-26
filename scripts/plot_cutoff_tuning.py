"""Plot validation micro-F1, precision and recall against the Score cut-off.

Draws the tuning curve that ``train`` walks through (``SCORE_CUTOFF_GRID`` on the validation
split, full Suggestion policy) for a trained Tagger, and marks its saved Score cut-off. The
image is embedded in ``docs/metrics.md``.

Regenerate from the repository root (needs the ``explore`` dependency group for matplotlib):

    uv run --group explore python scripts/plot_cutoff_tuning.py

Options: ``--tagger-dir`` (default ``artifacts/linear``), ``--validation`` (default
``data/splits/validation.jsonl``) and ``--output`` (default ``docs/images/cutoff-tuning.png``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from so_tagger.dataset import read_questions  # noqa: E402
from so_tagger.evaluation import (  # noqa: E402
    SCORE_CUTOFF_GRID,
    compute_metrics,
    true_tag_matrix,
)
from so_tagger.taggers import load_tagger  # noqa: E402

SURFACE = "#ffffff"
TEXT_PRIMARY = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e5e5e5"
SERIES = {  # Reference categorical palette, first three slots (validated as a set).
    "micro-F1": "#257de0",
    "micro-precision": "#f47230",
    "micro-recall": "#00b894",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tagger-dir", type=Path, default=Path("artifacts/linear"))
    parser.add_argument("--validation", type=Path, default=Path("data/splits/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/images/cutoff-tuning.png"))
    args = parser.parse_args()

    tagger = load_tagger(args.tagger_dir)
    questions = read_questions(args.validation)
    scores = tagger.score([question.title for question in questions])
    true = true_tag_matrix(questions, tagger.supported_tags)

    curves: dict[str, list[float]] = {name: [] for name in SERIES}
    for cutoff in SCORE_CUTOFF_GRID:
        metrics = compute_metrics(scores, true, tagger.supported_tags, cutoff)
        curves["micro-F1"].append(metrics.micro_f1)
        curves["micro-precision"].append(metrics.micro_precision)
        curves["micro-recall"].append(metrics.micro_recall)

    chosen = tagger.score_cutoff
    chosen_f1 = curves["micro-F1"][SCORE_CUTOFF_GRID.index(chosen)]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for name, values in curves.items():
        ax.plot(SCORE_CUTOFF_GRID, values, color=SERIES[name], linewidth=2, label=name)
        ax.annotate(
            name,
            xy=(SCORE_CUTOFF_GRID[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
            fontsize=9,
        )

    ax.axvline(chosen, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.plot([chosen], [chosen_f1], "o", color=SERIES["micro-F1"], markersize=8, mec=SURFACE)
    ax.annotate(
        f"chosen cut-off {chosen:.2f}\nmicro-F1 {chosen_f1:.3f}",
        xy=(chosen, chosen_f1),
        xytext=(10, 18),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Score cut-off", color=TEXT_PRIMARY)
    ax.set_ylabel("Validation score", color=TEXT_PRIMARY)
    ax.set_title(
        f"Score cut-off tuning, {tagger.metadata.kind} tagger "
        f"({len(questions)} validation Questions)",
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
    ax.legend(loc="lower left", frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"Wrote {args.output} (chosen cut-off {chosen:.2f}, micro-F1 {chosen_f1:.3f})")


if __name__ == "__main__":
    main()
