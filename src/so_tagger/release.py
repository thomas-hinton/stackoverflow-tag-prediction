"""The Release asset convention: how a trained Tagger travels from training to the image.

A Release is tagged ``vMAJOR.MINOR.PATCH`` and carries, per Tagger kind:

- ``<kind>-tagger.tar.gz``: the Tagger directory, files at the archive root, with the Release
  tag written into ``metadata.json`` as the Tagger's ``version``;
- ``<kind>-metrics.json``: the metrics file ``evaluate`` wrote for that Tagger.

``package_tagger`` builds both assets from a Tagger directory and a metrics file. The Docker
image downloads the archives from the Release and extracts each into its own directory, which
``so-tagger serve`` then loads.
"""

import io
import json
import re
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from so_tagger.taggers.base import METADATA_FILE, TaggerMetadata

RELEASE_TAG_PATTERN = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
"""A Release tag: semantic versioning with a leading ``v``, e.g. ``v0.1.0``."""


class ReleaseError(ValueError):
    """A Release tag or a Release asset does not follow the Release asset convention."""


@dataclass(frozen=True, slots=True)
class ReleaseAssets:
    """The files ``package_tagger`` wrote, ready to attach to a Release."""

    archive: Path
    metrics: Path | None


def tagger_archive_name(kind: str) -> str:
    return f"{kind}-tagger.tar.gz"


def metrics_asset_name(kind: str) -> str:
    return f"{kind}-metrics.json"


def check_release_tag(release_tag: str) -> str:
    """Return ``release_tag`` if it is ``vMAJOR.MINOR.PATCH``, raise ``ReleaseError`` if not."""
    if not RELEASE_TAG_PATTERN.fullmatch(release_tag):
        raise ReleaseError(
            f"Invalid Release tag '{release_tag}': expected vMAJOR.MINOR.PATCH, e.g. v0.1.0."
        )
    return release_tag


def package_tagger(
    tagger_dir: Path,
    release_tag: str,
    output_dir: Path,
    *,
    kind: str | None = None,
    metrics: Path | None = None,
) -> ReleaseAssets:
    """Write the Release assets of the Tagger saved in ``tagger_dir`` to ``output_dir``.

    The archive holds every file of the Tagger directory; its ``metadata.json`` records
    ``release_tag`` as the Tagger's version. ``tagger_dir`` itself is left untouched. When
    ``kind`` is given, the Tagger must be of that kind. When ``metrics`` is given, it must be a
    metrics file of the same Tagger kind, and it is copied as ``<kind>-metrics.json``.
    """
    check_release_tag(release_tag)
    metadata = replace(TaggerMetadata.read(tagger_dir), version=release_tag)
    if kind is not None and metadata.kind != kind:
        raise ReleaseError(f"{tagger_dir} holds a '{metadata.kind}' Tagger, not a '{kind}' one.")
    if metrics is not None:
        _check_metrics(metrics, metadata.kind)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / tagger_archive_name(metadata.kind)
    _write_archive(tagger_dir, metadata, archive)
    metrics_asset = None
    if metrics is not None:
        metrics_asset = output_dir / metrics_asset_name(metadata.kind)
        shutil.copyfile(metrics, metrics_asset)
    return ReleaseAssets(archive=archive, metrics=metrics_asset)


_NOTES_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("micro-F1", ("micro_f1",)),
    ("micro-precision", ("micro_precision",)),
    ("micro-recall", ("micro_recall",)),
    ("macro-F1", ("macro_f1",)),
    ("P@1", ("precision_at_k", "1")),
    ("P@3", ("precision_at_k", "3")),
    ("P@5", ("precision_at_k", "5")),
    ("R@1", ("recall_at_k", "1")),
    ("R@3", ("recall_at_k", "3")),
    ("R@5", ("recall_at_k", "5")),
)
"""The rows of the metrics table in the Release notes: label and path inside ``metrics``."""


def release_notes(reports: Sequence[Mapping[str, Any]], run_url: str | None = None) -> str:
    """The Markdown notes of a Release, generated from the metrics files of its Taggers.

    A table compares the Taggers side by side: one column per Tagger, in the order given, with
    their test metrics, Score cut-offs and training times. Below it come what the Taggers
    share: the Supported tag frequency cut-off, the dataset sizes, the commit they were
    evaluated on and, when given, the link to the workflow run that trained them. Raises
    ``ReleaseError`` when a metrics file lacks a value, when two files measure the same Tagger
    kind, or when the files were not measured on the same split, dataset and commit.
    """
    if not reports:
        raise ReleaseError("The Release notes need at least one metrics file.")
    try:
        kinds = [str(report["tagger"]) for report in reports]
        rows = [
            _row(label, [f"{_dig(report['metrics'], path):.3f}" for report in reports])
            for label, path in _NOTES_METRICS
        ]
        rows += [
            _row(
                "Suggestions per Question",
                [f"{report['metrics']['mean_suggestions_per_question']:.2f}" for report in reports],
            ),
            _row("Score cut-off", [f"{report['score_cutoff']:.2f}" for report in reports]),
            _row(
                "Validation micro-F1",
                [f"{report['validation_micro_f1']:.3f}" for report in reports],
            ),
            _row(
                "Training time",
                [_duration(report.get("training_seconds")) for report in reports],
            ),
        ]
        shared = {
            name: _shared_value(reports, name)
            for name in ("split", "frequency_cutoff", "n_supported_tags", "git_commit")
        }
        sizes = _shared_value(reports, "split_sizes")
        details = [
            f"- **Supported tag frequency cut-off**: {shared['frequency_cutoff']} Questions "
            f"({shared['n_supported_tags']} Supported tags)",
            "- **Dataset sizes**: "
            + ", ".join(f"{sizes[name]} {name}" for name in ("train", "validation", "test"))
            + " Questions",
            f"- **Commit**: {shared['git_commit'] or 'unknown'}",
        ]
    except ReleaseError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseError(
            f"The metrics file lacks what the Release notes need: {error}."
        ) from error
    if len(set(kinds)) != len(kinds):
        raise ReleaseError(f"Each Tagger kind needs exactly one metrics file, got {kinds}.")
    if run_url:
        details.append(f"- **Workflow run**: {run_url}")
    names = [f"{kind.capitalize()} tagger" for kind in kinds]
    title = (
        names[0] if len(names) == 1 else " and ".join(k.capitalize() for k in kinds) + " taggers"
    )
    return "\n".join(
        [
            f"## {title}",
            "",
            f"Measured on the {shared['split']} split.",
            "",
            _row("Metric", names),
            _row("---", ["---"] * len(names)),
            *rows,
            "",
            *details,
            "",
        ]
    )


def _row(label: str, cells: Sequence[str]) -> str:
    return "| " + " | ".join([label, *cells]) + " |"


def _shared_value(reports: Sequence[Mapping[str, Any]], name: str) -> Any:
    """The value of ``name`` in every metrics file, which must be the same in all of them."""
    values = [report.get(name) for report in reports]
    if name != "git_commit" and values[0] is None:
        raise KeyError(name)
    if any(value != values[0] for value in values[1:]):
        raise ReleaseError(
            f"The Taggers cannot be compared: their metrics files differ in '{name}' "
            f"({', '.join(map(str, values))}). Evaluate them on the same split and commit."
        )
    return values[0]


def _duration(seconds: float | None) -> str:
    """A training time such as ``12 s``, ``2 min 05 s`` or ``1 h 02 min 05 s``."""
    if seconds is None:
        return "unknown"
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _dig(values: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = values
    for key in path:
        value = value[key]
    return float(value)


def _check_metrics(path: Path, kind: str) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReleaseError(f"No metrics file at {path}: run `evaluate` first.") from error
    except json.JSONDecodeError as error:
        raise ReleaseError(f"{path} is not a metrics file: {error}.") from error
    if not isinstance(report, dict) or "tagger" not in report:
        raise ReleaseError(f"{path} is not a metrics file: it has no 'tagger' key.")
    if report["tagger"] != kind:
        raise ReleaseError(
            f"{path} measures the '{report['tagger']}' Tagger, not the '{kind}' one."
        )


def _write_archive(tagger_dir: Path, metadata: TaggerMetadata, archive: Path) -> None:
    """Tar and gzip the Tagger directory, replacing its metadata with ``metadata``.

    Entries are sorted and carry no owner, so that the archive does not depend on who built it.
    """

    def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    files = sorted(
        path
        for path in tagger_dir.rglob("*")
        if path.is_file() and path.relative_to(tagger_dir).as_posix() != METADATA_FILE
    )
    metadata_bytes = metadata.to_json().encode()
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(tagger_dir).as_posix(), filter=anonymous)
        info = anonymous(tarfile.TarInfo(METADATA_FILE))
        info.size = len(metadata_bytes)
        info.mtime = int((tagger_dir / METADATA_FILE).stat().st_mtime)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(metadata_bytes))
