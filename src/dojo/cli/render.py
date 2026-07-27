"""Artifact-only figure regeneration commands."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich.console import Console

from dojo.config_loader.compositor import compose_config
from dojo.config_schemas import RootConfig
from dojo.training.figures import write_training_figures

app = typer.Typer(no_args_is_help=True)
console = Console()


def _abort(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


def _load_objective_to_head(config_path: Path) -> dict[str, str] | None:
    """Load the stored resolved config through the shared resolved-config path."""

    if not config_path.exists():
        return None
    composed = compose_config(resolved_config_file=config_path)
    raw = OmegaConf.to_container(composed.config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("resolved config root must be a mapping")
    cfg = RootConfig.model_validate(raw)
    return {
        name: objective.head or name
        for name, objective in cfg.objectives.items()
        if objective.enabled
    }


def _backup_figures(figures_dir: Path) -> Path | None:
    if not figures_dir.exists():
        return None
    index = 1
    while (candidate := figures_dir.with_name(f"{figures_dir.name}.{index}")).exists():
        index += 1
    shutil.copytree(figures_dir, candidate)
    return candidate


@app.command("run")
def render_run(
    run_dir: Annotated[Path, typer.Argument(help="Completed run's top-level directory.")],
    backup: Annotated[
        bool,
        typer.Option("--backup", help="Copy existing figures to figures.N before overwrite."),
    ] = False,
) -> None:
    """Regenerate training figures from persisted run artifacts."""

    metrics_csv = run_dir / "metrics" / "metrics.csv"
    results_dir = run_dir / "results"
    if not metrics_csv.exists():
        _abort(f"missing metrics artifact: {metrics_csv}")
    if not results_dir.is_dir():
        _abort(f"missing results artifact: {results_dir}")

    figures_dir = run_dir / "figures"
    try:
        objective_to_head = _load_objective_to_head(run_dir / "config" / "resolved.yaml")
        backup_dir = _backup_figures(figures_dir) if backup else None
        write_training_figures(
            metrics_csv=metrics_csv,
            figures_dir=figures_dir,
            results_dir=results_dir,
            objective_to_head=objective_to_head,
        )
    except Exception as exc:
        _abort(str(exc))

    if backup_dir is not None:
        console.print(f"backed up figures to {backup_dir}")
    console.print(f"wrote figures to {figures_dir}")
