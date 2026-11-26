"""Seam 5: the helper that runs the API on a free local port for `docs-screenshots`, no browser."""

import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from so_tagger.api import create_app
from so_tagger.screenshots.server import ServerError, running_server
from so_tagger.taggers import Tagger


def status_of(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return error.code


def accepts_connections(base_url: str) -> bool:
    parts = urlsplit(base_url)
    assert parts.hostname is not None and parts.port is not None
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout=1):
            return True
    except OSError:
        return False


def test_the_server_is_healthy_inside_the_block_and_stopped_after(linear_tagger: Tagger) -> None:
    with running_server(create_app({"linear": linear_tagger})) as base_url:
        assert base_url.startswith("http://127.0.0.1:")
        assert status_of(f"{base_url}/health") == 200
        assert status_of(f"{base_url}/docs") == 200

    assert not accepts_connections(base_url)


def test_the_server_is_stopped_when_the_block_fails(linear_tagger: Tagger) -> None:
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        running_server(create_app({"linear": linear_tagger})) as base_url,
    ):
        raise RuntimeError("capture failed")

    assert not accepts_connections(base_url)


def test_a_server_that_never_turns_healthy_fails_and_is_stopped() -> None:
    # Without the Linear tagger, GET /health answers 503.
    app = create_app({})

    with (
        pytest.raises(ServerError, match="did not answer 200 on /health"),
        running_server(app, startup_timeout=1.0),
    ):
        pytest.fail("the block must not run")
