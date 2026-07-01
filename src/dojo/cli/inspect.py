"""`dojo inspect` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from dojo.config_loader import ConfigCompositionError, compose_and_resolve
from dojo.config_schemas import RootConfig, config_hash


app = typer.Typer(no_args_is_help=True)
console = Console()


def _abort(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


def _build_report(cfg: RootConfig, warnings: tuple[str, ...]) -> dict:
    enabled_outputs = {
        "logging": [sink.type for sink in cfg.training_outputs.logging.sinks],
        "results": cfg.training_outputs.results.enabled,
        "metrics": cfg.training_outputs.metrics.enabled,
        "figures": cfg.training_outputs.figures.enabled,
        "export": cfg.training_outputs.export.enabled,
    }
    paths = {
        "training_outputs.dir": cfg.training_outputs.dir,
        "training_outputs.results.dir": cfg.training_outputs.results.dir,
        "training_outputs.metrics.dir": cfg.training_outputs.metrics.dir,
        "training_outputs.figures.dir": cfg.training_outputs.figures.dir,
        "training_outputs.export.dir": cfg.training_outputs.export.dir,
    }
    return {
        "valid": True,
        "experiment": cfg.experiment.name,
        "task": cfg.task.type,
        "run_id": cfg.runtime.run_id,
        "config_hash": config_hash(cfg),
        "paths": paths,
        "enabled_outputs": enabled_outputs,
        "result_record_types": cfg.supervised_record_types(),
        "result_stages": cfg.result_stages(),
        "result_partition_by": cfg.training_outputs.results.partition_by,
        "warnings": list(warnings),
        "resolved_config": cfg.model_dump(mode="json", exclude_none=True),
    }


def _print_text_report(report: dict) -> None:
    console.print("[bold]Config[/bold]")
    console.print(f"  experiment: {report['experiment']}")
    console.print(f"  task:       {report['task']}")
    console.print(f"  run_id:     {report['run_id']}")
    console.print(f"  hash:       {report['config_hash']}")

    paths = Table(title="Resolved Output Paths", show_header=True)
    paths.add_column("Key")
    paths.add_column("Path")
    for key, value in report["paths"].items():
        paths.add_row(key, str(value))
    console.print(paths)

    outputs = Table(title="Enabled Outputs", show_header=True)
    outputs.add_column("Output")
    outputs.add_column("Status")
    enabled = report["enabled_outputs"]
    outputs.add_row("logging", ", ".join(enabled["logging"]) or "disabled")
    for key in ("results", "metrics", "figures", "export"):
        outputs.add_row(key, "enabled" if enabled[key] else "disabled")
    console.print(outputs)

    results = Table(title="P1 Result Contract", show_header=True)
    results.add_column("Field")
    results.add_column("Value")
    results.add_row("stages", ", ".join(report["result_stages"]))
    results.add_row("record types", ", ".join(report["result_record_types"]))
    results.add_row("partition_by", ", ".join(report["result_partition_by"]))
    console.print(results)

    if report["warnings"]:
        console.print("[yellow]Warnings[/yellow]")
        for warning in report["warnings"]:
            console.print(f"  - {warning}")


@app.command(
    "config",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def inspect_config(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Authored root YAML file to compose."),
    ] = None,
    resolved_config: Annotated[
        Path | None,
        typer.Option("--resolved-config", help="Resolved config artifact to inspect."),
    ] = None,
    config_dir: Annotated[
        list[Path] | None,
        typer.Option("--config-dir", help="Additional config search directory."),
    ] = None,
    check_remote: Annotated[
        bool,
        typer.Option("--check-remote", help="Reserved for remote HEAD checks."),
    ] = False,
    output_format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Output format."),
    ] = "text",
) -> None:
    """Compose, validate, and render a Dojo config."""

    if check_remote:
        _abort("--check-remote is not implemented in the P1 inspect-config slice")

    overrides = list(ctx.args)
    try:
        resolved = compose_and_resolve(
            overrides=overrides,
            config_file=config,
            resolved_config_file=resolved_config,
            config_dirs=config_dir or (),
        )
        report = _build_report(resolved.config, resolved.warnings)
    except ConfigCompositionError as exc:
        _abort(str(exc))
    except ValidationError as exc:
        if output_format == "json":
            console.print_json(
                json.dumps({"valid": False, "errors": exc.errors(include_url=False)})
            )
            raise typer.Exit(1)
        _abort(str(exc))
    except Exception as exc:
        _abort(str(exc))

    if output_format == "json":
        console.print_json(json.dumps(report))
    else:
        _print_text_report(report)
