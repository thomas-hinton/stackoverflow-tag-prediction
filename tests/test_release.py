"""The Release asset convention: `package` turns a Tagger directory into Release assets."""

import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from so_tagger.cli import app
from so_tagger.domain import Question
from so_tagger.evaluation import evaluate_tagger, metrics_report
from so_tagger.release import ReleaseError, release_notes
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


@pytest.fixture
def linear_metrics(linear_tagger: Tagger, tmp_path: Path) -> Path:
    """A real metrics file of the fixture Linear tagger, as `evaluate` writes it."""
    questions = [
        Question(id=1, title="How do I parse JSON in Python?", tags=("python",)),
        Question(id=2, title="Center a div with CSS", tags=("css",)),
        Question(id=3, title="Select rows with SQL", tags=("sql",)),
    ]
    metrics = evaluate_tagger(linear_tagger, questions)
    report = metrics_report(linear_tagger, "test", len(questions), metrics, "abc123")
    path = tmp_path / "linear-test.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_release_notes_hold_the_metrics_cutoffs_sizes_commit_and_run(linear_metrics: Path) -> None:
    report = json.loads(linear_metrics.read_text(encoding="utf-8"))

    notes = release_notes([report], run_url="https://github.com/o/r/actions/runs/42")

    assert notes.startswith("## Linear tagger\n")
    assert "| Metric | Linear tagger |" in notes
    assert f"| micro-F1 | {report['metrics']['micro_f1']:.3f} |" in notes
    assert f"| macro-F1 | {report['metrics']['macro_f1']:.3f} |" in notes
    assert f"| P@1 | {report['metrics']['precision_at_k']['1']:.3f} |" in notes
    assert f"| Score cut-off | {report['score_cutoff']:.2f} |" in notes
    assert f"**Supported tag frequency cut-off**: {report['frequency_cutoff']} Questions" in notes
    sizes = report["split_sizes"]
    assert f"{sizes['train']} train, {sizes['validation']} validation, 3 test" in notes
    assert "**Commit**: abc123" in notes
    assert "**Workflow run**: https://github.com/o/r/actions/runs/42" in notes


def test_release_notes_command_prints_the_notes(linear_metrics: Path) -> None:
    result = runner.invoke(app, ["release-notes", "--metrics", str(linear_metrics)])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("## Linear tagger")
    assert "Workflow run" not in result.output


def test_release_notes_reject_an_incomplete_metrics_file() -> None:
    with pytest.raises(ReleaseError, match="lacks what the Release notes need"):
        release_notes([{"schema_version": 1, "tagger": "linear", "split": "test"}])


def a_report(tagger: str, micro_f1: float, **changes: object) -> dict[str, object]:
    """A complete metrics file with literal values, as `evaluate` writes it on the test split."""
    report: dict[str, object] = {
        "schema_version": 1,
        "tagger": tagger,
        "split": "test",
        "metrics": {
            "micro_f1": micro_f1,
            "micro_precision": 0.5,
            "micro_recall": 0.25,
            "macro_f1": 0.125,
            "precision_at_k": {"1": 0.75, "3": 0.5, "5": 0.375},
            "recall_at_k": {"1": 0.25, "3": 0.5, "5": 0.625},
            "mean_suggestions_per_question": 1.5,
        },
        "score_cutoff": 0.3,
        "validation_micro_f1": 0.6,
        "frequency_cutoff": 20,
        "n_supported_tags": 86,
        "split_sizes": {"train": 2870, "validation": 615, "test": 616},
        "training_seconds": 12.4,
        "git_commit": "abc123",
    }
    report.update(changes)
    return report


def test_release_notes_compare_both_taggers_side_by_side() -> None:
    linear = a_report("linear", 0.4567, score_cutoff=0.27, training_seconds=12.4)
    transformer = a_report("transformer", 0.5321, score_cutoff=0.18, training_seconds=3725.0)

    notes = release_notes([linear, transformer], run_url="https://github.com/o/r/actions/runs/7")

    assert "| Metric | Linear tagger | Transformer tagger |" in notes
    assert "| micro-F1 | 0.457 | 0.532 |" in notes
    assert "| P@1 | 0.750 | 0.750 |" in notes
    assert "| Score cut-off | 0.27 | 0.18 |" in notes
    assert "| Training time | 12 s | 1 h 02 min 05 s |" in notes
    assert "**Supported tag frequency cut-off**: 20 Questions (86 Supported tags)" in notes
    assert "2870 train, 615 validation, 616 test Questions" in notes
    assert "**Commit**: abc123" in notes
    assert "**Workflow run**: https://github.com/o/r/actions/runs/7" in notes


def test_release_notes_command_compares_every_metrics_file_given(tmp_path: Path) -> None:
    paths = []
    for report in (a_report("linear", 0.4567), a_report("transformer", 0.5321)):
        path = tmp_path / f"{report['tagger']}-test.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths += ["--metrics", str(path)]

    result = runner.invoke(app, ["release-notes", *paths])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("## Linear and Transformer taggers\n")
    assert "| micro-F1 | 0.457 | 0.532 |" in result.output


@pytest.mark.parametrize(
    ("other", "changes", "message"),
    [
        ("linear", {}, "exactly one metrics file"),
        ("transformer", {"split_sizes": {"train": 1, "validation": 1, "test": 1}}, "'split_sizes'"),
        ("transformer", {"git_commit": "def456"}, "differ in 'git_commit'"),
        ("transformer", {"frequency_cutoff": 10}, "differ in 'frequency_cutoff'"),
    ],
    ids=["same-kind", "other-dataset", "other-commit", "other-cutoff"],
)
def test_release_notes_refuse_taggers_that_cannot_be_compared(
    other: str, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ReleaseError, match=message):
        release_notes([a_report("linear", 0.4), a_report(other, 0.5, **changes)])
