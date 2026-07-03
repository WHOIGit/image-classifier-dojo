"""`dojo infer` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dojo.config_loader import ConfigCompositionError, compose_and_resolve
from dojo.inference import execute_infer

app = typer.Typer(no_args_is_help=True)
console = Console()


def _abort(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


def _run(ctx: typer.Context, *, checkpoint: Path, output_kind: str) -> None:
    try:
        resolved = compose_and_resolve(overrides=list(ctx.args))
        result = execute_infer(
            resolved.config,
            checkpoint_path=checkpoint,
            output_kind=output_kind,  # type: ignore[arg-type]
        )
    except (ConfigCompositionError, Exception) as exc:
        _abort(str(exc))
    console.print(f"wrote {result.result_record_count} rows to {result.results_dir}")
    console.print(f"manifest: {result.manifest_path}")


@app.command(
    "predictions",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def predictions(
    ctx: typer.Context,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True)],
) -> None:
    """Write `stage=infer` classification prediction rows."""

    _run(ctx, checkpoint=checkpoint, output_kind="predictions")


@app.command(
    "embeddings",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def embeddings(
    ctx: typer.Context,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True)],
) -> None:
    """Write `stage=infer` embedding rows."""

    _run(ctx, checkpoint=checkpoint, output_kind="embeddings")
