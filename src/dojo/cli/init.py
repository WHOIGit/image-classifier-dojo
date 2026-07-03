"""`dojo init` project materialization."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table


console = Console()


@dataclass(frozen=True)
class InitAction:
    action: str
    path: Path


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    dry_run: bool,
    clobber: bool,
) -> list[InitAction]:
    actions: list[InitAction] = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = path.relative_to(source)
        target = destination / rel
        if target.exists() and not clobber:
            actions.append(InitAction("skip", target))
            continue
        actions.append(InitAction("overwrite" if target.exists() else "create", target))
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return actions


def _copy_fixture_data(
    project_dir: Path,
    *,
    dry_run: bool,
    clobber: bool,
) -> list[InitAction]:
    fixture = Path.cwd() / "tests" / "fixtures" / "plankton-toyset"
    if not fixture.exists():
        return []
    return _copy_tree(
        fixture,
        project_dir / "example-data" / "plankton-toyset",
        dry_run=dry_run,
        clobber=clobber,
    )


def _print_actions(actions: list[InitAction]) -> None:
    table = Table(title="Init Actions", show_header=True)
    table.add_column("Action")
    table.add_column("Path")
    for item in actions:
        table.add_row(item.action, str(item.path))
    console.print(table)


def init_command(
    project_dir: Annotated[
        Path,
        typer.Argument(help="Project directory to initialize."),
    ] = Path("."),
    minimal: Annotated[
        bool,
        typer.Option("--minimal", help="Copy the supervised starter config set."),
    ] = False,
    supervised: Annotated[
        bool,
        typer.Option("--supervised", help="Include supervised starter configs."),
    ] = False,
    data: Annotated[
        bool,
        typer.Option("--data", help="Copy the small example fixture dataset."),
    ] = False,
    all_configs: Annotated[
        bool,
        typer.Option("--all", help="Copy all packaged config groups and fixture data."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print actions without writing files."),
    ] = False,
    clobber: Annotated[
        bool,
        typer.Option("--clobber", help="Overwrite existing materialized files."),
    ] = False,
) -> None:
    """Materialize packaged Dojo configs into an editable local project."""

    selected_default = not any((minimal, supervised, data, all_configs))
    if selected_default:
        minimal = True
        supervised = True

    if all_configs and (minimal or supervised):
        console.print("[red]Error:[/red] --all is mutually exclusive with scope flags")
        raise typer.Exit(1)

    project_dir = project_dir.resolve()
    actions: list[InitAction] = []

    with as_file(files("dojo").joinpath("configs")) as packaged_configs:
        source = Path(packaged_configs)
        actions.extend(
            _copy_tree(
                source,
                project_dir / "configs",
                dry_run=dry_run,
                clobber=clobber,
            )
        )

    if all_configs or data:
        actions.extend(
            _copy_fixture_data(project_dir, dry_run=dry_run, clobber=clobber)
        )

    _print_actions(actions)
