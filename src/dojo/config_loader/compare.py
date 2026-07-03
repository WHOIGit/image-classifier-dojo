"""Compare two composed Dojo configs by hash-source sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import ValidationError

from dojo.config_loader.compositor import ConfigCompositionError, compose_config
from dojo.config_loader.conductor import validate_authored_only_fields
from dojo.config_loader.resolver import _derive_inference_pipeline, _resolve_model_shapes
from dojo.config_schemas import (
    RootConfig,
    class_mapping_hash_source,
    config_hash_source,
    head_hash_source,
    model_config_hash_source,
    preprocessing_hash_source,
    target_schema_hash_source,
)
from dojo.config_schemas.hashing import sha256_json

_MISSING = object()


def _jsonable(value: Any) -> Any:
    if value is _MISSING:
        return {"__dojo_missing__": True}
    return value


def _diff_values(a: Any, b: Any, *, path: str = "") -> list[dict[str, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs: list[dict[str, Any]] = []
        for key in sorted(set(a) | set(b), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            diffs.extend(
                _diff_values(
                    a.get(key, _MISSING),
                    b.get(key, _MISSING),
                    path=child_path,
                )
            )
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        diffs = []
        for index in range(max(len(a), len(b))):
            child_path = f"{path}[{index}]"
            diffs.extend(
                _diff_values(
                    a[index] if index < len(a) else _MISSING,
                    b[index] if index < len(b) else _MISSING,
                    path=child_path,
                )
            )
        return diffs
    if a == b:
        return []
    return [{"path": path, "a": _jsonable(a), "b": _jsonable(b)}]


def _path_is_under(path: str, prefixes: set[str]) -> bool:
    return any(
        path == prefix
        or path.startswith(f"{prefix}.")
        or path.startswith(f"{prefix}[")
        or prefix.startswith(f"{path}.")
        or prefix.startswith(f"{path}[")
        for prefix in prefixes
    )


def _leaf_paths(value: Any, *, path: str = "") -> set[str]:
    if isinstance(value, dict):
        if not value:
            return {path}
        paths: set[str] = set()
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            paths.update(_leaf_paths(child, path=child_path))
        return paths
    if isinstance(value, list):
        if not value:
            return {path}
        paths = set()
        for index, child in enumerate(value):
            paths.update(_leaf_paths(child, path=f"{path}[{index}]"))
        return paths
    return {path}


def _hash_ready_config(cfg: RootConfig) -> RootConfig:
    """Derive hash-time shapes without rendering paths or run-id templates."""

    cfg_dict = cfg.model_dump(mode="json", exclude_none=True)
    _derive_inference_pipeline(cfg_dict)
    _resolve_model_shapes(cfg_dict)
    return RootConfig.model_validate(cfg_dict)


def _root_from_path(path: Path, *, config_dirs: tuple[Path, ...]) -> tuple[RootConfig, str]:
    authored_error: Exception | None = None
    try:
        composed = compose_config(config_file=path, config_dirs=config_dirs)
        raw = OmegaConf.to_container(composed.config, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("composed config root must be a mapping")
        validate_authored_only_fields(raw, resolved_source=False)
        return _hash_ready_config(RootConfig.model_validate(raw)), "authored"
    except (ConfigCompositionError, ValidationError, ValueError) as exc:
        authored_error = exc

    try:
        composed = compose_config(resolved_config_file=path, config_dirs=config_dirs)
        raw = OmegaConf.to_container(composed.config, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("resolved config root must be a mapping")
        return RootConfig.model_validate(raw), "resolved"
    except Exception as resolved_error:
        raise ConfigCompositionError(
            f"could not load {path} as authored or resolved config; "
            f"authored error: {authored_error}; resolved error: {resolved_error}"
        ) from resolved_error


def _fallback_class_mapping(cfg: RootConfig) -> dict[int, str]:
    max_classes = max((head.num_classes for head in cfg.model.heads.values()), default=0)
    return {index: str(index) for index in range(max_classes)}


def _section(source_a: Any, source_b: Any) -> dict[str, Any]:
    diffs = _diff_values(source_a, source_b)
    return {
        "hash_a": sha256_json(source_a),
        "hash_b": sha256_json(source_b),
        "equal": not diffs,
        "diffs": diffs,
    }


def _hash_sections(cfg_a: RootConfig, cfg_b: RootConfig) -> dict[str, Any]:
    mapping_a = _fallback_class_mapping(cfg_a)
    mapping_b = _fallback_class_mapping(cfg_b)
    sections: dict[str, Any] = {
        "config_hash": _section(
            config_hash_source(cfg_a),
            config_hash_source(cfg_b),
        ),
        "target_schema_hash": _section(
            target_schema_hash_source(cfg_a),
            target_schema_hash_source(cfg_b),
        ),
        "class_mapping_hash": _section(
            class_mapping_hash_source(cfg_a, class_mapping=mapping_a),
            class_mapping_hash_source(cfg_b, class_mapping=mapping_b),
        ),
        "model_config_hash": _section(
            model_config_hash_source(cfg_a),
            model_config_hash_source(cfg_b),
        ),
        "preprocessing_hash": _section(
            preprocessing_hash_source(cfg_a),
            preprocessing_hash_source(cfg_b),
        ),
        "head_hashes": {},
    }
    for head_name in sorted(set(cfg_a.model.heads) | set(cfg_b.model.heads)):
        if head_name in cfg_a.model.heads:
            source_a = head_hash_source(
                cfg_a,
                head_name=head_name,
                class_mapping=mapping_a,
            )
        else:
            source_a = _MISSING
        if head_name in cfg_b.model.heads:
            source_b = head_hash_source(
                cfg_b,
                head_name=head_name,
                class_mapping=mapping_b,
            )
        else:
            source_b = _MISSING
        if source_a is _MISSING or source_b is _MISSING:
            diffs = _diff_values(source_a, source_b, path=head_name)
            sections["head_hashes"][head_name] = {
                "hash_a": None if source_a is _MISSING else sha256_json(source_a),
                "hash_b": None if source_b is _MISSING else sha256_json(source_b),
                "equal": False,
                "diffs": diffs,
            }
        else:
            sections["head_hashes"][head_name] = _section(
                source_a,
                source_b,
            )
    return sections


def _used_config_paths(cfg_a: RootConfig, cfg_b: RootConfig) -> set[str]:
    used = _leaf_paths(config_hash_source(cfg_a)) | _leaf_paths(config_hash_source(cfg_b))
    used.update(
        {
            "transforms.image_mode",
            "transforms.input_bit_depth",
            "transforms.inference_pipeline",
        }
    )
    return used


def compare_config_files(
    *,
    config_a: Path,
    config_b: Path,
    config_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Return a structured comparison of two authored or resolved config YAMLs."""

    cfg_a, source_kind_a = _root_from_path(config_a, config_dirs=config_dirs)
    cfg_b, source_kind_b = _root_from_path(config_b, config_dirs=config_dirs)
    sections = _hash_sections(cfg_a, cfg_b)

    full_a = cfg_a.model_dump(mode="json", exclude_none=True)
    full_b = cfg_b.model_dump(mode="json", exclude_none=True)
    used_paths = _used_config_paths(cfg_a, cfg_b)
    other_diffs = [
        diff
        for diff in _diff_values(full_a, full_b)
        if not _path_is_under(diff["path"], used_paths)
    ]
    sections["other"] = {
        "equal": not other_diffs,
        "diffs": other_diffs,
    }
    return {
        "valid": True,
        "config_a": {"path": str(config_a), "source_kind": source_kind_a},
        "config_b": {"path": str(config_b), "source_kind": source_kind_b},
        "sections": sections,
        "notes": [
            "class_mapping_hash and head_hash sections use index-string fallback "
            "labels because config-compare does not read datasets."
        ],
    }
