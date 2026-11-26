"""Seam 3: the CLI pipeline, run end to end on the fixture dataset in a temporary directory."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from so_tagger.cli import app
from tests.conftest import FIXTURE_DATASET, FIXTURE_SUPPORTED_TAGS

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


def train(tmp_path: Path, config: Path) -> Path:
    splits = split(prepare(tmp_path, config), tmp_path / "splits", config)
    tagger_dir = tmp_path / "tagger"
    invoke(
        "train",
        "--tagger",
        "linear",
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
