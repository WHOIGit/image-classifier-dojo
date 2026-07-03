"""Canonical JSON hashing helpers for config identity."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .root import RootConfig


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(f"{value:.12g}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def config_hash_source(cfg: RootConfig) -> dict[str, Any]:
    source = cfg.model_dump(mode="json", exclude_none=True)
    source.pop("output_root", None)
    source.pop("training_outputs", None)
    source.pop("eval_outputs", None)
    runtime = source.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("run_id", None)
        runtime.pop("sweep_id", None)
    transforms = source.get("transforms")
    if isinstance(transforms, dict):
        transforms.pop("inference_pipeline", None)
    return source


def config_hash(cfg: RootConfig) -> str:
    return sha256_json(config_hash_source(cfg))


def head_hash_source(
    cfg: RootConfig,
    *,
    head_name: str,
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    """Canonical P2.5a source block for one supervised head identity hash."""

    head = cfg.model.heads[head_name]
    target = cfg.data.targets[head.target]
    source: dict[str, Any] = {
        "version": "1",
        "head": {
            "type": head.type,
            "target": head.target,
            "target_type": target.type,
            "num_classes": head.num_classes,
            "network": head.network.model_dump(mode="json", exclude_none=True),
        },
    }
    if head.type == "multiclass_classification":
        source["head"]["class_mapping"] = [
            {"index": index, "label": class_mapping.get(index, str(index))}
            for index in range(head.num_classes)
        ]
    return source


def head_hash(
    cfg: RootConfig,
    *,
    head_name: str,
    class_mapping: dict[int, str],
) -> str:
    """Return the per-head content hash written on per-head result rows."""

    return sha256_json(
        head_hash_source(cfg, head_name=head_name, class_mapping=class_mapping)
    )


def _ordered_labels(
    *,
    num_classes: int,
    class_mapping: dict[int, str],
) -> list[dict[str, Any]]:
    return [
        {"index": index, "label": class_mapping.get(index, str(index))}
        for index in range(num_classes)
    ]


def target_schema_hash_source(cfg: RootConfig) -> dict[str, Any]:
    return {
        "version": "1",
        "heads": {
            name: {
                "type": head.type,
                "target": head.target,
                "target_type": cfg.data.targets[head.target].type,
                "target_transform": None,
                "output_dim": None,
                "num_classes": head.num_classes,
                "distribution": None,
                "ordinal": None,
            }
            for name, head in sorted(cfg.model.heads.items())
        },
    }


def class_mapping_hash_source(
    cfg: RootConfig,
    *,
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    return {
        "version": "1",
        "heads": {
            name: {
                "target": head.target,
                "ordered_labels": _ordered_labels(
                    num_classes=head.num_classes,
                    class_mapping=class_mapping,
                ),
            }
            for name, head in sorted(cfg.model.heads.items())
            if head.type == "multiclass_classification"
        },
    }


def model_config_hash_source(cfg: RootConfig) -> dict[str, Any]:
    model = cfg.model
    return {
        "version": "1",
        "model": {
            "image_input": {
                "backbone": {
                    "architecture": {
                        "source": model.image_input.backbone.architecture.source,
                        "name": model.image_input.backbone.architecture.name,
                        "output_dim": model.image_input.backbone.architecture.output_dim,
                        "params": {
                            "input_channels": model.image_input.backbone.architecture.input_channels
                        },
                    }
                }
            },
            "tabular_input": model.tabular_input.model_dump(mode="json", exclude_none=True),
            "embedding_adapter": model.embedding_adapter.model_dump(
                mode="json", exclude_none=True
            ),
            "heads": {
                name: {
                    "type": head.type,
                    "target": head.target,
                    "network": head.network.model_dump(mode="json", exclude_none=True),
                    "num_classes": head.num_classes,
                    "output_dim": None,
                    "distribution": None,
                }
                for name, head in sorted(model.heads.items())
            },
        },
    }


def preprocessing_hash_source(cfg: RootConfig) -> dict[str, Any]:
    return {
        "version": "1",
        "transforms": {
            "image_mode": cfg.transforms.image_mode,
            "input_bit_depth": cfg.transforms.input_bit_depth,
            "inference_pipeline": [
                step.model_dump(mode="json", exclude_none=True)
                for step in (cfg.transforms.inference_pipeline or [])
            ],
        },
        "tabular_preprocessing": {
            "selected_columns": [],
            "encodings": {},
            "imputation": {},
            "normalization": {},
        },
    }


def compatibility_hashes(
    cfg: RootConfig,
    *,
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    """Return P2.5b compatibility hashes and their canonical source blocks."""

    target_source = target_schema_hash_source(cfg)
    class_source = class_mapping_hash_source(cfg, class_mapping=class_mapping)
    model_source = model_config_hash_source(cfg)
    preprocessing_source = preprocessing_hash_source(cfg)
    return {
        "target_schema_hash": sha256_json(target_source),
        "target_schema_source": target_source,
        "class_mapping_hash": sha256_json(class_source),
        "class_mapping_source": class_source,
        "model_config_hash": sha256_json(model_source),
        "model_config_source": model_source,
        "preprocessing_hash": sha256_json(preprocessing_source),
        "preprocessing_source": preprocessing_source,
    }
