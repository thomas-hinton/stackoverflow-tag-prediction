"""Run an ASGI app on a free local port for the duration of a ``with`` block.

`docs-screenshots` serves the API (and later the demo) in process, in a background thread, so
that the browser can open it. The server is always stopped on leaving the block, whether the
block succeeds, fails, or the server never turns healthy.
"""

import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from starlette.types import ASGIApp

HOST = "127.0.0.1"
POLL_INTERVAL = 0.05
SHUTDOWN_TIMEOUT = 10.0


class ServerError(RuntimeError):
    """The server did not start, or did not turn healthy in time."""


@contextmanager
def running_server(
    app: ASGIApp,
    *,
    health_path: str = "/health",
    startup_timeout: float = 30.0,
) -> Iterator[str]:
    """Serve ``app`` on a free port of ``127.0.0.1`` and yield its base URL.

    The block runs once ``health_path`` answers 200. Raises ``ServerError`` when it does not
    within ``startup_timeout`` seconds.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]
    base_url = f"http://{HOST}:{port}"

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        _wait_until_healthy(base_url, health_path, thread, startup_timeout)
        yield base_url
    finally:
        server.should_exit = True
        thread.join(SHUTDOWN_TIMEOUT)
        sock.close()


def _wait_until_healthy(
    base_url: str, health_path: str, thread: threading.Thread, timeout: float
) -> None:
    url = f"{base_url}{health_path}"
    deadline = time.monotonic() + timeout
    last_answer = "no answer"
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise ServerError(f"The server stopped before answering on {url}.")
        try:
            with urllib.request.urlopen(url, timeout=POLL_INTERVAL * 20) as response:
                if response.status == 200:
                    return
                last_answer = f"status {response.status}"
        except urllib.error.HTTPError as error:
            last_answer = f"status {error.code}"
        except OSError as error:
            last_answer = str(error)
        time.sleep(POLL_INTERVAL)
    raise ServerError(
        f"The server did not answer 200 on {health_path} within {timeout:g} s "
        f"(last: {last_answer})."
    )
