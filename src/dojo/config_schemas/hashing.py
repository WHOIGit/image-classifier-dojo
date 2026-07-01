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
