"""Command-line interface of the project."""

from typing import Annotated

import typer

from so_tagger import __version__

app = typer.Typer(
    help="Suggest Tags for a StackOverflow Question from its Title.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"so-tagger {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Suggest Tags for a StackOverflow Question from its Title."""
