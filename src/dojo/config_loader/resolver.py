"""Dojo output path and template resolution."""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import coolname

from dojo.config_schemas.root import RootConfig


_TOKEN_RE = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class ResolutionResult:
    config: RootConfig
    warnings: tuple[str, ...]


def _slug(value: Any) -> str:
    text = str(value)
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _get_path(mapping: dict[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _seeded_slug(seed_material: str) -> str:
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    default_generator = coolname.generate.__self__
    previous_random = default_generator.random
    try:
        coolname.replace_random(rng)
        return coolname.generate_slug(2)
    finally:
        coolname.replace_random(previous_random)


def render_template(template: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "timestamp":
            return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if token == "coolname:noseed":
            return coolname.generate_slug(2)
        if token == "coolname":
            seed = _get_path(values, "runtime.seed")
            return _seeded_slug(f"runtime.seed:{seed}")
        if token.startswith("coolname:"):
            source = token.split(":", 1)[1]
            try:
                seed_value = _get_path(values, source)
            except KeyError:
                seed_value = source
            return _seeded_slug(f"{source}:{seed_value}")

        path, sep, fmt = token.partition(":")
        value = _get_path(values, path)
        if not sep:
            return str(value)
        if fmt == "slug":
            return _slug(value)
        return format(value, fmt)

    return _TOKEN_RE.sub(replace, template)


def _resolve_path(value: str, parent_base: Path, cwd: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if value == "." or value.startswith("./"):
        return str((cwd / path).resolve())
    return str((parent_base / path).resolve())


def _resolve_output_root(output_root: str, cwd: Path) -> Path:
    path = Path(output_root)
    if path.is_absolute():
        return path
    return (cwd / path).resolve()


def _derive_inference_pipeline(cfg_dict: dict[str, Any]) -> None:
    transforms = cfg_dict["transforms"]
    pipeline = transforms.get("pipeline") or []
    transforms["inference_pipeline"] = [
        copy.deepcopy(step)
        for step in pipeline
        if step.get("enabled", True) and not step.get("train_only", False)
    ]


def resolve_runtime_and_paths(cfg: RootConfig, *, cwd: Path | None = None) -> ResolutionResult:
    cwd = (cwd or Path.cwd()).resolve()
    warnings: list[str] = []
    cfg_dict = cfg.model_dump(mode="json", exclude_none=True)

    run_id_template = cfg_dict["runtime"].get("run_id") or "{coolname:noseed}"
    if "{" in run_id_template:
        cfg_dict["runtime"]["run_id"] = render_template(run_id_template, cfg_dict)

    _derive_inference_pipeline(cfg_dict)

    output_root = _resolve_output_root(cfg_dict["output_root"], cwd)
    cfg_dict["output_root"] = str(output_root)

    training_outputs = cfg_dict["training_outputs"]
    if training_outputs.get("dir_template"):
        rendered = render_template(training_outputs["dir_template"], cfg_dict)
        training_dir = Path(_resolve_path(rendered, output_root, cwd))
    elif training_outputs.get("dir"):
        training_dir = Path(_resolve_path(training_outputs["dir"], output_root, cwd))
    else:
        raise ValueError("training_outputs.dir or dir_template is required")
    training_outputs["dir"] = str(training_dir)

    for block_name in ("results", "metrics", "figures", "export"):
        block = training_outputs.get(block_name)
        if not isinstance(block, dict):
            continue
        block_dir = block.get("dir")
        if block_dir:
            block["dir"] = _resolve_path(str(block_dir), training_dir, cwd)

    if training_dir.exists():
        non_config_contents = [
            path
            for path in training_dir.iterdir()
            if path.name != "config" and not path.name.startswith(".")
        ]
        if non_config_contents:
            warnings.append(
                f"training output directory is not empty: {training_dir}"
            )

    return ResolutionResult(
        config=RootConfig.model_validate(cfg_dict),
        warnings=tuple(warnings),
    )
