"""``dojo train`` CLI command: a thin Typer wrapper over the training run.

Parses CLI options + Hydra overrides, composes/resolves the config, calls
:func:`dojo.training.run.execute_train`, and prints a summary. All training
logic lives in :mod:`dojo.training.run` so non-CLI callers (tests, a Prefect
flow) reuse it without depending on Typer / Rich.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from dojo.config_loader import ConfigCompositionError, compose_and_resolve
from dojo.training.run import execute_train

console = Console()

# Lets dash-free Hydra overrides (e.g. ``runtime.num_workers=0``) reach the
# composer instead of being rejected as unexpected arguments.
TRAIN_CONTEXT_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}


def train_command(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Authored root YAML file to compose."),
    ] = None,
    resolved_config: Annotated[
        Path | None,
        typer.Option("--resolved-config", help="Resolved config artifact to run."),
    ] = None,
    config_dir: Annotated[
        list[Path] | None,
        typer.Option("--config-dir", help="Additional config search directory."),
    ] = None,
) -> None:
    """Compose, resolve, and run one supervised training job."""

    try:
        resolved = compose_and_resolve(
            overrides=list(ctx.args),
            config_file=config,
            resolved_config_file=resolved_config,
            config_dirs=config_dir or (),
        )
    except (ConfigCompositionError, ValidationError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    cfg = resolved.config
    for warning in resolved.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    def _status(message: str) -> None:
        console.print(f"[dim]status:[/dim] {message}")

    result = execute_train(cfg, status_callback=_status)

    console.print(f"[green]Run complete:[/green] {result.run_dir}")
    console.print(f"  config_hash:     {result.config_hash}")
    console.print(f"  dataset_hash:    {result.dataset_hash}")
    console.print(f"  checkpoint:      {result.best_checkpoint}")
    console.print(f"  checkpoint_hash: {result.checkpoint_hash}")
    if result.results_dir is not None:
        console.print(
            f"  results:         {result.results_dir} "
            f"({result.result_record_count} rows)"
        )
