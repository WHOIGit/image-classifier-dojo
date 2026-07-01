"""dojo CLI entrypoint."""

from __future__ import annotations

import typer

from dojo.cli.inspect import app as inspect_app
from dojo.cli.train import TRAIN_CONTEXT_SETTINGS, train_command


app = typer.Typer(no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")
app.command("train", context_settings=TRAIN_CONTEXT_SETTINGS)(train_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
