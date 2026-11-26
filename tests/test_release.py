"""The Release asset convention: `package` turns a Tagger directory into Release assets."""

import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from so_tagger.cli import app
from so_tagger.taggers import Tagger, load_tagger
from tests.conftest import extract_archive

runner = CliRunner()
TITLES = ["How do I parse JSON in Python?", "Center a div with CSS", "Hello, how are you?"]


@pytest.fixture
def tagger_dir(linear_tagger: Tagger, tmp_path: Path) -> Path:
    directory = tmp_path / "linear"
    linear_tagger.save(directory)
    return directory


def package(*args: str | Path) -> Result:
    return runner.invoke(app, ["package", *(str(arg) for arg in args)])


def write_metrics(path: Path, tagger: str) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "tagger": tagger}), encoding="utf-8")
    return path


def test_package_writes_an_archive_whose_tagger_reports_the_release_tag(
    tagger_dir: Path, linear_tagger: Tagger, tmp_path: Path
) -> None:
    result = package(
        "--release-tag", "v0.1.0", "--tagger-dir", tagger_dir, "--output-dir", tmp_path
    )

    assert result.exit_code == 0, result.output
    archive = tmp_path / "linear-tagger.tar.gz"
    assert str(archive) in result.output
    released = load_tagger(extract_archive(archive, tmp_path / "extracted"), kind="linear")
    assert released.metadata.version == "v0.1.0"
    assert released.score_cutoff == linear_tagger.score_cutoff
    for title in TITLES:
        assert released.suggest(title) == linear_tagger.suggest(title)


def test_package_leaves_the_tagger_directory_untouched(tagger_dir: Path, tmp_path: Path) -> None:
    before = {path.name: path.read_bytes() for path in tagger_dir.iterdir()}

    package("--release-tag", "v0.1.0", "--tagger-dir", tagger_dir, "--output-dir", tmp_path / "d")

    assert {path.name: path.read_bytes() for path in tagger_dir.iterdir()} == before
    assert load_tagger(tagger_dir).metadata.version == "dev"


def test_the_archive_holds_the_tagger_files_at_its_root_without_owner(
    tagger_dir: Path, tmp_path: Path
) -> None:
    package("--release-tag", "v0.1.0", "--tagger-dir", tagger_dir, "--output-dir", tmp_path)

    with tarfile.open(tmp_path / "linear-tagger.tar.gz") as tar:
        members = tar.getmembers()
    assert sorted(member.name for member in members) == ["metadata.json", "model.joblib"]
    assert {(member.uid, member.gid, member.uname, member.gname) for member in members} == {
        (0, 0, "", "")
    }


def test_package_copies_the_metrics_file_under_its_asset_name(
    tagger_dir: Path, tmp_path: Path
) -> None:
    metrics = write_metrics(tmp_path / "linear-test.json", "linear")

    result = package(
        "--release-tag", "v1.0.0", "--tagger-dir", tagger_dir, "--metrics", metrics,
        "--output-dir", tmp_path / "dist",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (tmp_path / "dist").iterdir()) == [
        "linear-metrics.json",
        "linear-tagger.tar.gz",
    ]
    assert (tmp_path / "dist" / "linear-metrics.json").read_bytes() == metrics.read_bytes()


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "v01.2.3", "v1.2.3-rc1", "latest"])
def test_package_rejects_a_tag_that_is_not_semantic_versioning(
    tag: str, tagger_dir: Path, tmp_path: Path
) -> None:
    result = package("--release-tag", tag, "--tagger-dir", tagger_dir, "--output-dir", tmp_path)

    assert result.exit_code == 1
    assert "expected vMAJOR.MINOR.PATCH" in result.output
    assert not list(tmp_path.glob("*.tar.gz"))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"tagger": "transformer"}', "measures the 'transformer' Tagger"),
        ('{"schema_version": 1}', "it has no 'tagger' key"),
        ("not json", "is not a metrics file"),
        (None, "run `evaluate` first"),
    ],
    ids=["other-tagger", "no-tagger-key", "not-json", "missing"],
)
def test_package_rejects_a_metrics_file_that_does_not_measure_the_tagger(
    content: str | None, message: str, tagger_dir: Path, tmp_path: Path
) -> None:
    metrics = tmp_path / "metrics.json"
    if content is not None:
        metrics.write_text(content, encoding="utf-8")

    result = package(
        "--release-tag", "v0.1.0", "--tagger-dir", tagger_dir, "--metrics", metrics,
        "--output-dir", tmp_path / "dist",
    )  # fmt: skip

    assert result.exit_code == 1
    assert message in result.output
    assert not (tmp_path / "dist").exists()


def test_package_rejects_a_tagger_directory_of_another_kind(
    tagger_dir: Path, tmp_path: Path
) -> None:
    result = package(
        "--release-tag", "v0.1.0", "--tagger", "transformer", "--tagger-dir", tagger_dir,
        "--output-dir", tmp_path,
    )  # fmt: skip

    assert result.exit_code == 1
    assert "not a 'transformer' one" in result.output


def test_package_fails_clearly_without_a_trained_tagger(tmp_path: Path) -> None:
    result = package("--release-tag", "v0.1.0", "--tagger-dir", tmp_path / "none")

    assert result.exit_code == 1
    assert "run `train` first" in result.output
