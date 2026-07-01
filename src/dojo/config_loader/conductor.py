"""The config-loader conductor: compose -> validate -> resolve in one call.

Sequences the three steps that turn user input (overrides + optional
``--config`` / ``--resolved-config`` / ``--config-dir``) into a single validated,
resolved :class:`RootConfig`: :func:`compose_config` (compositor),
:meth:`RootConfig.model_validate` (the schema contract), and
:func:`resolve_runtime_and_paths` (resolver), plus the one authored-only-field
guard. The single entry point every caller (CLI, tests, external orchestrators)
uses so none of them drift on composition or validation order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from dojo.config_loader.compositor import compose_config
from dojo.config_loader.resolver import ResolutionResult, resolve_runtime_and_paths
from dojo.config_schemas import RootConfig


def validate_authored_only_fields(raw: dict, *, resolved_source: bool) -> None:
    """Reject resolved-only keys in authored input (skipped for resolved sources)."""

    if resolved_source:
        return
    transforms = raw.get("transforms")
    if isinstance(transforms, dict) and "inference_pipeline" in transforms:
        raise ValueError(
            "transforms.inference_pipeline is resolved-only; configure "
            "transforms.pipeline instead"
        )


def compose_and_resolve(
    *,
    overrides: Sequence[str] = (),
    config_file: Path | None = None,
    resolved_config_file: Path | None = None,
    config_dirs: Sequence[Path] = (),
) -> ResolutionResult:
    """Compose, validate, and resolve a Dojo config into a :class:`ResolutionResult`."""

    composed = compose_config(
        overrides=list(overrides),
        config_file=config_file,
        resolved_config_file=resolved_config_file,
        config_dirs=tuple(config_dirs),
    )
    raw = OmegaConf.to_container(composed.config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("composed config root must be a mapping")
    validate_authored_only_fields(raw, resolved_source=resolved_config_file is not None)
    authored = RootConfig.model_validate(raw)
    return resolve_runtime_and_paths(authored)
