"""Reproducible documentation screenshots: the `docs-screenshots` command and its parts.

Playwright and Pillow come from the `docs` dependency group (`uv sync --group docs`); only the
modules that drive the browser or touch images import them, so the capture list and the server
helper work, and are tested, without them.
"""
