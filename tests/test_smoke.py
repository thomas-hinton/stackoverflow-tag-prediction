from typer.testing import CliRunner

import so_tagger
from so_tagger.cli import app


def test_version_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert so_tagger.__version__ in result.output
