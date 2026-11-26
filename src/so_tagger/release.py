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
from dataclasses import dataclass, replace
from pathlib import Path

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
