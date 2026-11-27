"""Seam 1: the HTTP API, driven in process with Taggers trained on the fixture dataset.

The Transformer tagger is fine-tuned from a tiny local checkpoint; its tests skip themselves
without the `transformer` extra.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, get_args

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so_tagger import __version__
from so_tagger.api import MAX_TITLE_LENGTH, TaggerKind, create_app, load_taggers
from so_tagger.cli import app as cli_app
from so_tagger.suggestion_policy import MAX_SUGGESTIONS, MIN_SUGGESTIONS
from so_tagger.taggers import TAGGER_KINDS, Tagger, TaggerError
from tests.conftest import FIXTURE_SUPPORTED_TAGS, RELEASE_TAG

TITLE = "How do I parse JSON in Python?"
OFF_TOPIC_TITLE = "Hello, how are you?"


@pytest.fixture(scope="module")
def client(released_linear_tagger: Tagger) -> TestClient:
    return TestClient(create_app({"linear": released_linear_tagger}))


@pytest.fixture(scope="module")
def both_taggers(
    released_linear_tagger: Tagger, released_transformer_tagger: Tagger
) -> dict[str, Tagger]:
    return {"linear": released_linear_tagger, "transformer": released_transformer_tagger}


@pytest.fixture(scope="module")
def client_with_both(both_taggers: dict[str, Tagger]) -> TestClient:
    return TestClient(create_app(both_taggers))


@pytest.fixture(scope="module")
def client_without_taggers() -> TestClient:
    return TestClient(create_app({}))


def suggest(client: TestClient, payload: dict[str, Any], expected_status: int = 200) -> Any:
    response = client.post("/suggest", json=payload)
    assert response.status_code == expected_status, response.text
    return response.json()


def error_messages(body: Any) -> str:
    return " ".join(error["msg"] for error in body["detail"])


def assert_valid_suggestion_list(body: Any) -> None:
    assert set(body) == {"tagger", "version", "suggestions"}
    suggestions = body["suggestions"]
    assert MIN_SUGGESTIONS <= len(suggestions) <= MAX_SUGGESTIONS
    for suggestion in suggestions:
        assert set(suggestion) == {"tag", "score"}
        assert suggestion["tag"] in FIXTURE_SUPPORTED_TAGS
        assert 0.0 <= suggestion["score"] <= 1.0
    scores = [suggestion["score"] for suggestion in suggestions]
    assert scores == sorted(scores, reverse=True)


def test_without_a_tagger_the_linear_tagger_answers(client: TestClient) -> None:
    body = suggest(client, {"title": TITLE})

    assert body["tagger"] == "linear"
    assert body["version"] == RELEASE_TAG
    assert_valid_suggestion_list(body)


def test_the_linear_tagger_can_be_asked_explicitly(client: TestClient) -> None:
    body = suggest(client, {"title": TITLE, "tagger": "linear"})

    assert body["tagger"] == "linear"
    assert_valid_suggestion_list(body)


def test_suggestions_are_the_taggers_suggestion_list(
    client: TestClient, linear_tagger: Tagger
) -> None:
    body = suggest(client, {"title": TITLE})

    expected = [{"tag": s.tag, "score": s.score} for s in linear_tagger.suggest(TITLE)]
    assert body["suggestions"] == expected


def test_the_title_is_stripped_before_scoring(client: TestClient) -> None:
    assert suggest(client, {"title": f"  \t{TITLE}\n "}) == suggest(client, {"title": TITLE})


def test_an_off_topic_title_still_gets_the_guaranteed_minimum(client: TestClient) -> None:
    body = suggest(client, {"title": OFF_TOPIC_TITLE})

    assert len(body["suggestions"]) >= MIN_SUGGESTIONS
    assert_valid_suggestion_list(body)


@pytest.mark.parametrize("title", ["", "   \t\n "], ids=["empty", "whitespace-only"])
def test_an_empty_title_is_rejected(client: TestClient, title: str) -> None:
    body = suggest(client, {"title": title}, expected_status=422)

    assert "must not be empty" in error_messages(body)


def test_a_title_of_the_maximum_length_is_accepted(client: TestClient) -> None:
    assert_valid_suggestion_list(suggest(client, {"title": "a" * MAX_TITLE_LENGTH}))


def test_a_too_long_title_is_rejected(client: TestClient) -> None:
    body = suggest(client, {"title": "a" * (MAX_TITLE_LENGTH + 1)}, expected_status=422)

    assert f"at most {MAX_TITLE_LENGTH} characters" in error_messages(body)


def test_a_missing_title_is_rejected(client: TestClient) -> None:
    suggest(client, {"tagger": "linear"}, expected_status=422)


def test_an_unknown_tagger_is_rejected_with_the_valid_names(client: TestClient) -> None:
    body = suggest(client, {"title": TITLE, "tagger": "bert"}, expected_status=422)

    message = error_messages(body)
    for kind in TAGGER_KINDS:
        assert f"'{kind}'" in message


def test_a_known_but_unloaded_tagger_is_unavailable(client: TestClient) -> None:
    body = suggest(client, {"title": TITLE, "tagger": "transformer"}, expected_status=503)

    assert body["detail"] == "The transformer Tagger is not loaded on this server."


def test_the_transformer_tagger_answers_when_asked(
    client_with_both: TestClient, transformer_tagger: Tagger
) -> None:
    body = suggest(client_with_both, {"title": TITLE, "tagger": "transformer"})

    assert body["tagger"] == "transformer"
    assert body["version"] == RELEASE_TAG
    assert_valid_suggestion_list(body)
    expected = transformer_tagger.suggest(TITLE)
    assert [s["tag"] for s in body["suggestions"]] == [s.tag for s in expected]
    assert [s["score"] for s in body["suggestions"]] == pytest.approx([s.score for s in expected])


def test_with_both_taggers_loaded_the_default_is_still_linear(
    client_with_both: TestClient,
) -> None:
    assert suggest(client_with_both, {"title": TITLE})["tagger"] == "linear"


def test_health_lists_both_loaded_taggers_with_their_versions(
    client_with_both: TestClient,
) -> None:
    response = client_with_both.get("/health")

    assert response.status_code == 200
    loaded = {
        "loaded": True,
        "version": RELEASE_TAG,
        "n_supported_tags": len(FIXTURE_SUPPORTED_TAGS),
    }
    assert response.json() == {
        "status": "ok",
        "taggers": {"linear": loaded, "transformer": loaded},
    }


def test_removing_the_transformer_tagger_makes_it_unavailable_again(
    both_taggers: dict[str, Tagger],
) -> None:
    registry = {kind: t for kind, t in both_taggers.items() if kind != "transformer"}
    client = TestClient(create_app(registry))

    suggest(client, {"title": TITLE, "tagger": "transformer"}, expected_status=503)
    assert client.get("/health").json()["taggers"]["transformer"]["loaded"] is False


def test_concurrent_requests_get_the_same_answers_as_sequential_ones(
    client_with_both: TestClient,
) -> None:
    titles = [TITLE, OFF_TOPIC_TITLE, "Center a div with CSS flexbox", "SQL join two tables"]
    payloads = [{"title": t, "tagger": k} for t in titles for k in TAGGER_KINDS]
    expected = [suggest(client_with_both, payload) for payload in payloads]

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda p: suggest(client_with_both, p), payloads * 8))

    assert answers == expected * 8


def test_the_latency_is_logged_per_tagger(
    client_with_both: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="so_tagger.api"):
        for kind in TAGGER_KINDS:
            suggest(client_with_both, {"title": TITLE, "tagger": kind})

    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "so_tagger.api"]
    assert [line["tagger"] for line in lines] == list(TAGGER_KINDS)
    assert [line["version"] for line in lines] == [RELEASE_TAG] * len(TAGGER_KINDS)
    assert all(line["latency_ms"] >= 0 for line in lines)


def test_the_known_kinds_of_the_api_are_the_tagger_kinds() -> None:
    assert get_args(TaggerKind) == TAGGER_KINDS


def test_health_lists_every_tagger_with_the_version_of_its_release(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "taggers": {
            "linear": {
                "loaded": True,
                "version": RELEASE_TAG,
                "n_supported_tags": len(FIXTURE_SUPPORTED_TAGS),
            },
            "transformer": {"loaded": False, "version": None, "n_supported_tags": 0},
        },
    }


def test_health_is_unavailable_without_the_default_tagger(
    client_without_taggers: TestClient,
) -> None:
    response = client_without_taggers.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["taggers"]["linear"] == {"loaded": False, "version": None, "n_supported_tags": 0}


def test_a_tagger_fresh_from_training_reports_the_dev_version(linear_tagger: Tagger) -> None:
    client = TestClient(create_app({"linear": linear_tagger}))

    assert suggest(client, {"title": TITLE})["version"] == "dev"
    assert client.get("/health").json()["taggers"]["linear"]["version"] == "dev"


def test_the_default_tagger_is_unavailable_when_not_loaded(
    client_without_taggers: TestClient,
) -> None:
    suggest(client_without_taggers, {"title": TITLE}, expected_status=503)


def test_the_openapi_schema_documents_the_contract(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "StackOverflow Tag Suggestion API"
    assert schema["info"]["version"] == __version__
    assert schema["info"]["description"]
    assert {"/suggest", "/health"} <= set(schema["paths"])
    components = schema["components"]["schemas"]
    request = components["SuggestRequest"]
    assert request["examples"]
    assert request["properties"]["title"]["maxLength"] == MAX_TITLE_LENGTH
    assert request["properties"]["tagger"]["enum"] == list(TAGGER_KINDS)
    for name in ("SuggestRequest", "SuggestResponse", "SuggestionModel", "HealthResponse"):
        for field in components[name]["properties"].values():
            assert field.get("description"), (name, field)
    assert client.get("/docs").status_code == 200


def test_one_log_line_per_request_without_the_title_at_info_level(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret_title = "Unique words zanzibar quokka in a JSON title"
    with caplog.at_level(logging.INFO, logger="so_tagger.api"):
        body = suggest(client, {"title": secret_title})

    records = [r for r in caplog.records if r.name == "so_tagger.api"]
    assert len(records) == 1
    assert "zanzibar" not in records[0].getMessage()
    fields = json.loads(records[0].getMessage())
    assert fields["event"] == "suggest"
    assert fields["tagger"] == "linear"
    assert fields["title_length"] == len(secret_title)
    assert fields["n_suggestions"] == len(body["suggestions"])
    assert fields["latency_ms"] >= 0


def test_a_registry_with_a_tagger_under_another_kind_is_refused(linear_tagger: Tagger) -> None:
    with pytest.raises(ValueError, match="is a 'linear' Tagger"):
        create_app({"transformer": linear_tagger})
    with pytest.raises(ValueError, match="Unknown Tagger kinds: bert"):
        create_app({"bert": linear_tagger})


def test_load_taggers_skips_a_missing_directory(linear_tagger: Tagger, tmp_path: Path) -> None:
    linear_tagger.save(tmp_path / "linear")

    taggers = load_taggers({"linear": tmp_path / "linear", "transformer": tmp_path / "missing"})

    assert set(taggers) == {"linear"}
    assert taggers["linear"].supported_tags == FIXTURE_SUPPORTED_TAGS


def test_load_taggers_refuses_a_directory_of_another_kind(
    linear_tagger: Tagger, tmp_path: Path
) -> None:
    linear_tagger.save(tmp_path / "linear")

    with pytest.raises(TaggerError, match="not a 'transformer' one"):
        load_taggers({"transformer": tmp_path / "linear"})


def test_serve_loads_the_tagger_directory_from_the_environment(
    released_linear_tagger: Tagger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released_linear_tagger.save(tmp_path / "linear")
    served: dict[str, Any] = {}

    def fake_run(app: FastAPI, host: str, port: int) -> None:
        served.update(app=app, host=host, port=port)

    monkeypatch.setattr("so_tagger.cli.uvicorn.run", fake_run)
    result = CliRunner().invoke(
        cli_app,
        ["serve", "--port", "9000"],
        env={"SO_TAGGER_LINEAR_DIR": str(tmp_path / "linear")},
    )

    assert result.exit_code == 0, result.output
    assert (served["host"], served["port"]) == ("127.0.0.1", 9000)
    health = TestClient(served["app"]).get("/health").json()
    assert health["taggers"]["linear"]["loaded"] is True
    assert health["taggers"]["linear"]["version"] == RELEASE_TAG


def test_serve_loads_both_taggers_and_sets_the_torch_threads_from_the_environment(
    released_linear_tagger: Tagger,
    released_transformer_tagger: Tagger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    released_linear_tagger.save(tmp_path / "linear")
    released_transformer_tagger.save(tmp_path / "transformer")
    served: dict[str, Any] = {}
    monkeypatch.setattr("so_tagger.cli.uvicorn.run", lambda app, host, port: served.update(app=app))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: served.update(threads=n))

    result = CliRunner().invoke(
        cli_app,
        ["serve"],
        env={
            "SO_TAGGER_LINEAR_DIR": str(tmp_path / "linear"),
            "SO_TAGGER_TRANSFORMER_DIR": str(tmp_path / "transformer"),
            "SO_TAGGER_TORCH_THREADS": "3",
        },
    )

    assert result.exit_code == 0, result.output
    assert served["threads"] == 3
    client = TestClient(served["app"])
    taggers = client.get("/health").json()["taggers"]
    assert taggers["linear"]["loaded"] and taggers["transformer"]["loaded"]
    assert suggest(client, {"title": TITLE, "tagger": "transformer"})["tagger"] == "transformer"


def test_serve_starts_without_a_tagger_and_reports_it_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr("so_tagger.cli.uvicorn.run", lambda app, host, port: served.update(app=app))

    result = CliRunner().invoke(
        cli_app,
        [
            "serve",
            "--linear-dir",
            str(tmp_path / "none"),
            "--transformer-dir",
            str(tmp_path / "none"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "No linear Tagger" in result.output
    assert TestClient(served["app"]).get("/health").status_code == 503
