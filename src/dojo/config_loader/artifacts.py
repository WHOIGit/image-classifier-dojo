"""Run config artifacts: the ``config/`` directory every run writes.

`DESIGN-DOC/03-configuration.md` ("Run config artifacts") specifies the set:

- ``composed.yaml`` — after composition and CLI overrides, before resolution
- ``resolved.yaml`` / ``resolved.json`` — fully resolved, with generated values
- ``cli.txt`` — the invoked command
- ``overrides.txt`` — the overrides applied

The resolved pair is always written. The other three come from composition
provenance, so they are written only when a caller supplies it — a run built from
a bare ``RootConfig`` (a test, an external orchestrator, ``--resolved-config``)
has no composition history, and reconstructing one would fabricate a
pre-resolution input that was never composed. A missing file is unambiguous.

This lives outside ``dojo.training`` because ``eval_outputs.dir`` carries the same
layout (`DESIGN-DOC/06-results-artifacts-and-metadata.md`), so `dojo eval` /
`dojo infer` can reuse it.

``sweep_values.txt`` is sweep-member-only and is not written here.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from dojo.config_loader.resolver import ConfigProvenance
from dojo.config_schemas import RootConfig

RESOLVED_YAML = "resolved.yaml"
RESOLVED_JSON = "resolved.json"
COMPOSED_YAML = "composed.yaml"
CLI_TXT = "cli.txt"
OVERRIDES_TXT = "overrides.txt"


def write_run_config_artifacts(
    config_dir: Path,
    cfg: RootConfig,
    *,
    provenance: ConfigProvenance | None = None,
) -> tuple[Path, ...]:
    """Write a run's config artifacts into ``config_dir``; return what was written.

    ``resolved.yaml`` and ``resolved.json`` are always written and share one
    serialization pass, so the two cannot drift.
    """

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    resolved = cfg.model_dump(mode="json", exclude_none=True)

    resolved_yaml = config_dir / RESOLVED_YAML
    resolved_yaml.write_text(
        OmegaConf.to_yaml(OmegaConf.create(resolved)), encoding="utf-8"
    )
    written.append(resolved_yaml)

    resolved_json = config_dir / RESOLVED_JSON
    resolved_json.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    written.append(resolved_json)

    if provenance is None:
        return tuple(written)

    composed_yaml = config_dir / COMPOSED_YAML
    composed_yaml.write_text(OmegaConf.to_yaml(provenance.composed), encoding="utf-8")
    written.append(composed_yaml)

    overrides_txt = config_dir / OVERRIDES_TXT
    overrides_txt.write_text(
        "".join(f"{override}\n" for override in provenance.overrides), encoding="utf-8"
    )
    written.append(overrides_txt)

    if provenance.invoked_command is not None:
        cli_txt = config_dir / CLI_TXT
        cli_txt.write_text(provenance.invoked_command + "\n", encoding="utf-8")
        written.append(cli_txt)

    return tuple(written)
