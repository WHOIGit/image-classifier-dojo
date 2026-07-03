"""`dojo inspect` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from dojo.config_loader import ConfigCompositionError, compare_config_files, compose_and_resolve
from dojo.config_schemas import RootConfig, config_hash
from dojo.data.inspect import inspect_dataset as inspect_dataset_config
from dojo.model import build_supervised_model
from dojo.training.checkpoint import checkpoint_hash


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


def _display_diff_value(value: object) -> str:
    if isinstance(value, dict) and value.get("__dojo_missing__") is True:
        return "<missing>"
    text = json.dumps(value, sort_keys=True)
    if len(text) > 96:
        return f"{text[:93]}..."
    return text


def _print_config_compare_report(report: dict) -> None:
    console.print("[bold]Config Compare[/bold]")
    console.print(
        f"  A: {report['config_a']['path']} ({report['config_a']['source_kind']})"
    )
    console.print(
        f"  B: {report['config_b']['path']} ({report['config_b']['source_kind']})"
    )

    summary = Table(title="Hash Source Sections", show_header=True)
    summary.add_column("Section")
    summary.add_column("Equal")
    summary.add_column("Diffs")
    summary.add_column("Hash A")
    summary.add_column("Hash B")
    for section_name, section in report["sections"].items():
        if section_name == "head_hashes":
            for head_name, head_section in section.items():
                summary.add_row(
                    f"head_hash:{head_name}",
                    "yes" if head_section["equal"] else "no",
                    str(len(head_section["diffs"])),
                    str(head_section.get("hash_a")),
                    str(head_section.get("hash_b")),
                )
            continue
        summary.add_row(
            section_name,
            "yes" if section["equal"] else "no",
            str(len(section["diffs"])),
            str(section.get("hash_a", "")),
            str(section.get("hash_b", "")),
        )
    console.print(summary)

    for section_name, section in report["sections"].items():
        nested_sections = section.items() if section_name == "head_hashes" else [(section_name, section)]
        for display_name, item in nested_sections:
            diffs = item["diffs"]
            if not diffs:
                continue
            table = Table(title=f"Diffs: {display_name}", show_header=True)
            table.add_column("Path")
            table.add_column("A")
            table.add_column("B")
            for diff in diffs[:20]:
                table.add_row(
                    diff["path"],
                    _display_diff_value(diff["a"]),
                    _display_diff_value(diff["b"]),
                )
            console.print(table)
            if len(diffs) > 20:
                console.print(f"  ... {len(diffs) - 20} more diffs")

    if report["notes"]:
        console.print("[yellow]Notes[/yellow]")
        for note in report["notes"]:
            console.print(f"  - {note}")


def _print_dataset_report(report: dict, *, wrote_stats_cache: bool, stats_cache_uri: str | None) -> None:
    console.print("[bold]Dataset[/bold]")
    console.print(f"  backend:      {report['backend']}")
    console.print(f"  manifest_uri: {report['manifest_uri']}")
    console.print(f"  hash:         {report['dataset_hash']}")
    console.print(f"  provenance:   {report['dataset_hash_provenance']}")
    console.print(f"  splits:       {', '.join(report['splits_present'])}")

    targets = Table(title="Targets", show_header=True)
    targets.add_column("Split")
    targets.add_column("Target")
    targets.add_column("Valid")
    targets.add_column("Missing")
    targets.add_column("Policy failure")
    for split, per_target in report["targets"].items():
        for target_name, summary in per_target.items():
            targets.add_row(
                split,
                target_name,
                str(summary["valid_count"]),
                str(summary["missing_count"]),
                "yes" if summary["would_fail"] else "no",
            )
    console.print(targets)

    counts = Table(title="Class Counts", show_header=True)
    counts.add_column("Head")
    counts.add_column("Num Classes")
    counts.add_column("Total")
    for head_name, summary in report["aspects"]["class_counts"]["per_head"].items():
        counts.add_row(
            head_name,
            str(summary["num_classes"]),
            str(summary["total"]),
        )
    console.print(counts)

    class_mapping = Table(title="Class Mapping", show_header=True)
    class_mapping.add_column("Head")
    class_mapping.add_column("Index", justify="right")
    class_mapping.add_column("Label")
    class_mapping.add_column("Train Count", justify="right")
    mappings = report["aspects"].get("class_mapping", {}).get("per_head", {})
    per_head_counts = report["aspects"]["class_counts"]["per_head"]
    for head_name, labels in mappings.items():
        counts_for_head = per_head_counts.get(head_name, {}).get("counts", {})
        for index in sorted(labels, key=lambda value: int(value)):
            class_mapping.add_row(
                head_name,
                str(index),
                str(labels[index]),
                str(counts_for_head.get(str(index), 0)),
            )
    console.print(class_mapping)

    if stats_cache_uri is not None:
        status = "wrote" if wrote_stats_cache else "configured"
        console.print(f"  stats cache:  {status} {stats_cache_uri}")


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


@app.command("config-compare")
def inspect_config_compare(
    config_a: Annotated[
        Path,
        typer.Option("--config-a", exists=True, help="First authored or resolved YAML."),
    ],
    config_b: Annotated[
        Path,
        typer.Option("--config-b", exists=True, help="Second authored or resolved YAML."),
    ],
    config_dir: Annotated[
        list[Path] | None,
        typer.Option("--config-dir", help="Additional config search directory."),
    ] = None,
    output_format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Output format."),
    ] = "text",
) -> None:
    """Compare two authored or resolved config YAML files by hash-source sections."""

    try:
        report = compare_config_files(
            config_a=config_a,
            config_b=config_b,
            config_dirs=tuple(config_dir or ()),
        )
    except Exception as exc:
        _abort(str(exc))

    if output_format == "json":
        console.print_json(json.dumps(report))
    else:
        _print_config_compare_report(report)


@app.command(
    "dataset",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def inspect_dataset(
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
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Write configured dataset stats cache."),
    ] = False,
    dimensions: Annotated[
        bool,
        typer.Option("--dimensions", help="Include per-sample native dimensions."),
    ] = False,
    normalization: Annotated[
        bool,
        typer.Option("--normalization", help="Compute train-split image mean/std."),
    ] = False,
    output_format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Output format."),
    ] = "text",
) -> None:
    """Inspect the configured dataset and optionally write its stats cache."""

    overrides = list(ctx.args)
    try:
        resolved = compose_and_resolve(
            overrides=overrides,
            config_file=config,
            resolved_config_file=resolved_config,
            config_dirs=config_dir or (),
        )
        result = inspect_dataset_config(
            resolved.config,
            write_stats=stats,
            include_dimensions=stats or dimensions,
            include_normalization=normalization,
        )
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
        console.print_json(
            json.dumps(
                {
                    **result.report,
                    "stats_cache_uri": result.stats_cache_uri,
                    "wrote_stats_cache": result.wrote_stats_cache,
                }
            )
        )
    else:
        _print_dataset_report(
            result.report,
            wrote_stats_cache=result.wrote_stats_cache,
            stats_cache_uri=result.stats_cache_uri,
        )


@app.command(
    "backbone",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def inspect_backbone(
    ctx: typer.Context,
    output_format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Output format."),
    ] = "text",
) -> None:
    """Inspect the configured backbone and freeze policy."""

    try:
        resolved = compose_and_resolve(overrides=list(ctx.args))
        model = build_supervised_model(
            resolved.config.model,
            freeze_cfg=resolved.config.training.freeze,
        )
        total_params = sum(p.numel() for p in model.backbone.parameters())
        trainable_params = sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        )
        report = {
            "source": resolved.config.model.image_input.backbone.architecture.source,
            "name": resolved.config.model.image_input.backbone.architecture.name,
            "output_dim": model.backbone.output_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
    except Exception as exc:
        _abort(str(exc))

    if output_format == "json":
        console.print_json(json.dumps(report))
    else:
        console.print("[bold]Backbone[/bold]")
        for key, value in report.items():
            console.print(f"  {key}: {value}")


@app.command("checkpoint")
def inspect_checkpoint(
    checkpoint: Annotated[Path, typer.Argument(exists=True)],
    output_format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Output format."),
    ] = "text",
) -> None:
    """Inspect a Dojo checkpoint's embedded inference contract."""

    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        contract = payload.get("dojo_inference_contract")
        report = {
            "checkpoint": str(checkpoint),
            "checkpoint_hash": checkpoint_hash(checkpoint),
            "has_inference_contract": isinstance(contract, dict),
            "compatibility": contract.get("compatibility", {}) if isinstance(contract, dict) else {},
        }
    except Exception as exc:
        _abort(str(exc))

    if output_format == "json":
        console.print_json(json.dumps(report))
    else:
        console.print("[bold]Checkpoint[/bold]")
        console.print(f"  checkpoint_hash: {report['checkpoint_hash']}")
        console.print(f"  has_inference_contract: {report['has_inference_contract']}")
        compatibility = report["compatibility"]
        for key in (
            "target_schema_hash",
            "class_mapping_hash",
            "model_config_hash",
            "preprocessing_hash",
        ):
            if key in compatibility:
                console.print(f"  {key}: {compatibility[key]}")
