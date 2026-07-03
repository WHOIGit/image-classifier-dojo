"""dojo CLI entrypoint."""

from __future__ import annotations

import typer

from dojo.cli.eval import app as eval_app
from dojo.cli.infer import app as infer_app
from dojo.cli.inspect import app as inspect_app
from dojo.cli.init import init_command
from dojo.cli.train import TRAIN_CONTEXT_SETTINGS, train_command


app = typer.Typer(no_args_is_help=True)
app.command("init")(init_command)
app.add_typer(infer_app, name="infer")
app.add_typer(eval_app, name="eval")
app.add_typer(inspect_app, name="inspect")
app.command("train", context_settings=TRAIN_CONTEXT_SETTINGS)(train_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
