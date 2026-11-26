"""Seam 3: the CLI pipeline, run end to end on the fixture dataset in a temporary directory."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from so_tagger.cli import app
from tests.conftest import FIXTURE_DATASET, FIXTURE_SUPPORTED_TAGS, needs_transformer_extra

runner = CliRunner()


def invoke(*args: str | Path) -> Result:
    result = runner.invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(tmp_path: Path, config: Path) -> Path:
    prepared = tmp_path / "prepared"
    invoke("prepare", "--raw", FIXTURE_DATASET, "--output-dir", prepared, "--config", config)
    return prepared


def test_prepare_keeps_only_supported_tags_and_drops_questions_without_any(
    tmp_path: Path, fixture_config: Path
) -> None:
    prepared = prepare(tmp_path, fixture_config)

    questions = read_jsonl(prepared / "questions.jsonl")
    kept_tags = {tag for question in questions for tag in question["tags"]}
    titles = {question["title"] for question in questions}
    assert kept_tags == set(FIXTURE_SUPPORTED_TAGS)
    # The only Question tagged `haskell` alone loses its single Tag and is dropped,
    # while the `haskell` + `python` Question is kept with `python` only.
    assert "Haskell type class instance for Maybe" not in titles
    assert {"title": "Haskell monad equivalent in Python", "tags": ["python"]}.items() <= next(
        q for q in questions if q["title"].startswith("Haskell monad")
    ).items()
    assert len(questions) == 72


def test_prepare_records_the_supported_tags_and_logs_a_summary(
    tmp_path: Path, fixture_config: Path
) -> None:
    prepared = tmp_path / "prepared"
    result = invoke(
        "prepare", "--raw", FIXTURE_DATASET, "--output-dir", prepared, "--config", fixture_config
    )

    supported = json.loads((prepared / "supported_tags.json").read_text(encoding="utf-8"))
    assert supported["frequency_cutoff"] == 5
    assert tuple(supported["tags"]) == FIXTURE_SUPPORTED_TAGS
    assert "73 Questions read" in result.output
    assert "72 kept" in result.output
    assert "4 Supported tags" in result.output


SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def split(prepared: Path, output: Path, config: Path, *extra: str) -> Path:
    invoke("split", "--prepared-dir", prepared, "--output-dir", output, "--config", config, *extra)
    return output


def split_bytes(directory: Path) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in SPLIT_FILES}


def test_split_with_the_same_seed_is_byte_identical(tmp_path: Path, fixture_config: Path) -> None:
    prepared = prepare(tmp_path, fixture_config)

    first = split(prepared, tmp_path / "first", fixture_config)
    second = split(prepared, tmp_path / "second", fixture_config)

    assert split_bytes(first) == split_bytes(second)


def test_split_with_another_seed_differs(tmp_path: Path, fixture_config: Path) -> None:
    prepared = prepare(tmp_path, fixture_config)

    default = split(prepared, tmp_path / "default", fixture_config)
    other = split(prepared, tmp_path / "other", fixture_config, "--seed", "8")

    assert split_bytes(default) != split_bytes(other)


def test_split_puts_each_question_in_exactly_one_set_with_the_configured_ratios(
    tmp_path: Path, fixture_config: Path
) -> None:
    prepared = prepare(tmp_path, fixture_config)
    output = split(prepared, tmp_path / "splits", fixture_config)

    ids = {name: [q["question_id"] for q in read_jsonl(output / name)] for name in SPLIT_FILES}
    all_ids = [question_id for split_ids in ids.values() for question_id in split_ids]
    prepared_ids = {q["question_id"] for q in read_jsonl(prepared / "questions.jsonl")}
    assert len(all_ids) == len(set(all_ids)) == len(prepared_ids) == 72
    assert set(all_ids) == prepared_ids
    # 72 Questions at 70/15/15: 50.4 / 10.8 / 10.8, rounded by largest remainder.
    assert [len(ids[name]) for name in SPLIT_FILES] == [50, 11, 11]


def test_split_keeps_every_supported_tag_in_training(tmp_path: Path, fixture_config: Path) -> None:
    prepared = prepare(tmp_path, fixture_config)
    output = split(prepared, tmp_path / "splits", fixture_config)

    train_tags = {tag for q in read_jsonl(output / "train.jsonl") for tag in q["tags"]}
    assert train_tags == set(FIXTURE_SUPPORTED_TAGS)


def test_split_fails_clearly_when_a_supported_tag_is_absent_from_training(
    tmp_path: Path, fixture_config: Path
) -> None:
    prepared = prepare(tmp_path, fixture_config)
    supported_path = prepared / "supported_tags.json"
    supported = json.loads(supported_path.read_text(encoding="utf-8"))
    supported["tags"] = [*supported["tags"], "rust"]
    supported_path.write_text(json.dumps(supported), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "split",
            "--prepared-dir",
            str(prepared),
            "--output-dir",
            str(tmp_path / "splits"),
            "--config",
            str(fixture_config),
        ],
    )

    assert result.exit_code == 1
    assert "absent from the training split" in result.output
    assert "rust" in result.output
    assert not (tmp_path / "splits" / "train.jsonl").exists()


def train(
    tmp_path: Path, config: Path, tagger_dir: Path | None = None, kind: str = "linear"
) -> Path:
    splits = tmp_path / "splits"
    if not splits.exists():
        split(prepare(tmp_path, config), splits, config)
    tagger_dir = tagger_dir or tmp_path / "tagger"
    invoke(
        "train",
        "--tagger",
        kind,
        "--splits-dir",
        splits,
        "--output-dir",
        tagger_dir,
        "--config",
        config,
    )
    return tagger_dir


def test_train_then_suggest_prints_a_ranked_suggestion_table(
    tmp_path: Path, fixture_config: Path
) -> None:
    tagger_dir = train(tmp_path, fixture_config)

    result = invoke("suggest", "How do I parse JSON in Python?", "--tagger-dir", tagger_dir)

    lines = result.output.strip().splitlines()
    assert lines[0].split() == ["Tag", "Score"]
    first_tag, first_score = lines[1].split()
    assert first_tag == "python"
    assert 0.0 <= float(first_score) <= 1.0


def test_suggest_json_prints_valid_suggestions(tmp_path: Path, fixture_config: Path) -> None:
    tagger_dir = train(tmp_path, fixture_config)

    result = invoke(
        "suggest", "Hello, how are you?", "--tagger", "linear", "--tagger-dir", tagger_dir, "--json"
    )

    suggestions = json.loads(result.output)
    assert 1 <= len(suggestions) <= 5
    scores = [suggestion["score"] for suggestion in suggestions]
    assert scores == sorted(scores, reverse=True)
    for suggestion in suggestions:
        assert suggestion["tag"] in FIXTURE_SUPPORTED_TAGS
        assert 0.0 <= suggestion["score"] <= 1.0


def test_suggest_fails_clearly_without_a_trained_tagger(tmp_path: Path) -> None:
    result = runner.invoke(app, ["suggest", "Any title", "--tagger-dir", str(tmp_path / "none")])

    assert result.exit_code == 1
    assert "run `train` first" in result.output


HEADER = "post_id,tag_name,tag_id,tag_position,title\n"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1,python,16,0,Read a file\n1,python,16,1,Read a file\n", "'python' twice"),
        ("1,python,16,0,Read a file\n1,io,17,1,Write a file\n", "different Titles"),
        ("1,,16,0,Read a file\n", "empty Tag or Title"),
        ("1,python,16,0,Read a file\n2,python,99,0,Parse JSON\n", "not one-to-one"),
    ],
)
def test_prepare_fails_clearly_on_a_raw_dataset_that_breaks_an_invariant(
    tmp_path: Path, fixture_config: Path, rows: str, message: str
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(HEADER + rows, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "prepare",
            "--raw",
            str(raw),
            "--output-dir",
            str(tmp_path / "out"),
            "--config",
            str(fixture_config),
        ],
    )

    assert result.exit_code == 1
    assert message in result.output


def test_prepare_keeps_the_tag_named_null(tmp_path: Path, fixture_config: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        HEADER + "".join(f"{i},null,3392,0,Check for null in JavaScript {i}\n" for i in range(5)),
        encoding="utf-8",
    )

    invoke("prepare", "--raw", raw, "--output-dir", tmp_path / "out", "--config", fixture_config)

    assert {tag for q in read_jsonl(tmp_path / "out" / "questions.jsonl") for tag in q["tags"]} == {
        "null"
    }


def test_train_tunes_the_cutoff_without_ever_reading_the_test_split(
    tmp_path: Path, fixture_config: Path
) -> None:
    first = json.loads((train(tmp_path, fixture_config) / "metadata.json").read_text("utf-8"))
    # Replace the test split with garbage: any read of it would now fail.
    (tmp_path / "splits" / "test.jsonl").write_text("not a Question\n", encoding="utf-8")

    second_dir = train(tmp_path, fixture_config, tmp_path / "second")
    second = json.loads((second_dir / "metadata.json").read_text("utf-8"))

    assert second["score_cutoff"] == first["score_cutoff"]
    assert second["validation_micro_f1"] == first["validation_micro_f1"]


def test_train_logs_the_tuned_cutoff_and_its_validation_micro_f1(
    tmp_path: Path, fixture_config: Path
) -> None:
    splits = split(prepare(tmp_path, fixture_config), tmp_path / "splits", fixture_config)

    result = invoke(
        "train", "--splits-dir", splits, "--output-dir", tmp_path / "t", "--config", fixture_config
    )

    metadata = json.loads((tmp_path / "t" / "metadata.json").read_text("utf-8"))
    assert f"Score cut-off on 11 validation Questions: {metadata['score_cutoff']:.2f}" in (
        result.output
    )
    assert f"validation micro-F1 {metadata['validation_micro_f1']:.3f}" in result.output


METRICS_KEYS = {
    "schema_version",
    "tagger",
    "split",
    "metrics",
    "conventions",
    "score_cutoff",
    "validation_micro_f1",
    "frequency_cutoff",
    "n_supported_tags",
    "split_sizes",
    "hyperparameters",
    "seed",
    "training_seconds",
    "git_commit",
    "package_version",
    "tagger_package_version",
    "tagger_created_at",
    "created_at",
    "training_details",
}


def test_evaluate_writes_a_complete_metrics_file_and_prints_a_summary(
    tmp_path: Path, fixture_config: Path
) -> None:
    tagger_dir = train(tmp_path, fixture_config)
    metrics_path = tmp_path / "metrics" / "linear-test.json"

    result = invoke(
        "evaluate",
        "--tagger-dir",
        tagger_dir,
        "--splits-dir",
        tmp_path / "splits",
        "--output",
        metrics_path,
    )

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(report) == METRICS_KEYS
    assert report["schema_version"] == 1
    assert (report["tagger"], report["split"]) == ("linear", "test")
    assert report["split_sizes"] == {"train": 50, "validation": 11, "test": 11}
    assert (report["frequency_cutoff"], report["n_supported_tags"], report["seed"]) == (5, 4, 7)
    assert report["hyperparameters"]["C"] == 10.0
    assert 0.0 < report["score_cutoff"] < 1.0
    assert 0.0 <= report["validation_micro_f1"] <= 1.0
    assert report["training_seconds"] > 0
    assert report["git_commit"] is None or len(report["git_commit"]) == 40

    metrics = report["metrics"]
    for key in ("micro_f1", "micro_precision", "micro_recall", "macro_f1"):
        assert 0.0 <= metrics[key] <= 1.0
    for key in ("precision_at_k", "recall_at_k"):
        assert set(metrics[key]) == {"1", "3", "5"}
        assert all(0.0 <= value <= 1.0 for value in metrics[key].values())
    assert 1.0 <= metrics["mean_suggestions_per_question"] <= 5.0
    assert set(metrics["per_tag"]) == set(FIXTURE_SUPPORTED_TAGS)
    for tag_metrics in metrics["per_tag"].values():
        assert set(tag_metrics) == {"precision", "recall", "f1", "support"}
    assert sum(t["support"] for t in metrics["per_tag"].values()) >= 11

    assert "Evaluated the linear Tagger on 11 test Questions" in result.output
    assert f"micro-F1  {metrics['micro_f1']:.3f}" in result.output
    assert "P@1" in result.output and "R@5" in result.output
    assert f"Metrics written to {metrics_path}" in result.output


def test_evaluate_on_another_split_records_its_size(tmp_path: Path, fixture_config: Path) -> None:
    tagger_dir = train(tmp_path, fixture_config)
    metrics_path = tmp_path / "validation.json"

    invoke(
        "evaluate",
        "--tagger-dir",
        tagger_dir,
        "--splits-dir",
        tmp_path / "splits",
        "--split",
        "validation",
        "--output",
        metrics_path,
    )

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["split"] == "validation"
    assert report["split_sizes"] == {"train": 50, "validation": 11}
    # On the split it was tuned on, the Tagger reproduces its validation micro-F1.
    assert report["metrics"]["micro_f1"] == report["validation_micro_f1"]


def test_evaluate_rejects_an_unknown_split() -> None:
    result = runner.invoke(app, ["evaluate", "--split", "holdout"])

    assert result.exit_code == 2
    assert "Unknown split 'holdout'" in result.output


def test_evaluate_fails_clearly_without_a_trained_tagger(tmp_path: Path) -> None:
    result = runner.invoke(app, ["evaluate", "--tagger-dir", str(tmp_path / "none")])

    assert result.exit_code == 1
    assert "run `train` first" in result.output


@pytest.mark.parametrize("command", [["train"], ["suggest", "Any title"], ["evaluate"]])
def test_an_unknown_tagger_kind_is_rejected_with_the_valid_kinds(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--tagger", "forest"])

    assert result.exit_code == 2
    assert "Unknown Tagger kind 'forest'" in result.output
    assert "linear" in result.output


@pytest.mark.parametrize("title", ["", "   "])
def test_suggest_rejects_an_empty_title(title: str) -> None:
    result = runner.invoke(app, ["suggest", title])

    assert result.exit_code == 2
    assert "Title must not be empty" in result.output


def test_transformer_train_then_evaluate_writes_a_valid_metrics_file(
    tmp_path: Path, fixture_config: Path
) -> None:
    needs_transformer_extra("transformer")
    splits = split(prepare(tmp_path, fixture_config), tmp_path / "splits", fixture_config)
    tagger_dir = tmp_path / "transformer"
    trained = invoke(
        "train",
        "--tagger",
        "transformer",
        "--splits-dir",
        splits,
        "--output-dir",
        tagger_dir,
        "--config",
        fixture_config,
    )
    metrics_path = tmp_path / "transformer-test.json"

    evaluated = invoke(
        "evaluate",
        "--tagger",
        "transformer",
        "--tagger-dir",
        tagger_dir,
        "--splits-dir",
        splits,
        "--output",
        metrics_path,
    )

    assert "Trained the transformer Tagger on 50 Questions (4 Supported tags)." in trained.output
    assert "Epoch 1/2: loss" in trained.output
    assert "Maximum sequence length:" in trained.output
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(report) == METRICS_KEYS
    assert (report["tagger"], report["split"], report["schema_version"]) == (
        "transformer",
        "test",
        1,
    )
    assert report["split_sizes"] == {"train": 50, "validation": 11, "test": 11}
    assert report["training_seconds"] > 0
    assert report["hyperparameters"]["learning_rate"] == 1e-3
    details = report["training_details"]
    assert details["best_epoch"] in (1, 2)
    assert details["epochs_run"] == len(details["history"])
    assert details["max_length"] >= 3
    assert 0.0 <= report["metrics"]["micro_f1"] <= 1.0
    assert set(report["metrics"]["per_tag"]) == set(FIXTURE_SUPPORTED_TAGS)
    assert "Evaluated the transformer Tagger on 11 test Questions" in evaluated.output


def test_transformer_suggest_prints_valid_suggestions(tmp_path: Path, fixture_config: Path) -> None:
    needs_transformer_extra("transformer")
    tagger_dir = train(tmp_path, fixture_config, kind="transformer")

    result = invoke(
        "suggest",
        "How do I parse JSON in Python?",
        "--tagger",
        "transformer",
        "--tagger-dir",
        tagger_dir,
        "--json",
    )

    suggestions = json.loads(result.stdout)
    assert 1 <= len(suggestions) <= 5
    scores = [suggestion["score"] for suggestion in suggestions]
    assert scores == sorted(scores, reverse=True)
    for suggestion in suggestions:
        assert suggestion["tag"] in FIXTURE_SUPPORTED_TAGS
        assert 0.0 <= suggestion["score"] <= 1.0


def test_suggest_rejects_a_tagger_directory_of_another_kind(
    tmp_path: Path, fixture_config: Path
) -> None:
    tagger_dir = train(tmp_path, fixture_config)

    result = runner.invoke(
        app, ["suggest", "Any title", "--tagger", "transformer", "--tagger-dir", str(tagger_dir)]
    )

    assert result.exit_code == 1
    assert "holds a 'linear' Tagger, not a 'transformer' one" in result.output
