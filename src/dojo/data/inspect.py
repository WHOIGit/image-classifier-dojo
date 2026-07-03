"""Dataset inspection and stats-cache writing for current manifest backends."""

from __future__ import annotations

import io
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dojo.config_schemas.root import RootConfig, TargetConfig
from dojo.data.parquet_images import (
    _discover_files,
    _integer_targets,
    _needed_columns,
    _resolve_class_mapping,
    _tables_by_split,
    build_datasets,
)
from dojo.data.parquet_rows import (
    ROW_FILE_PATH,
    ROW_GROUP_INDEX,
    ROW_IN_GROUP_INDEX,
    LazyParquetImageReader,
)
from dojo.data.transforms import _decode_to_unit_chw
from dojo.storage import Storage, get_storage


@dataclass(frozen=True)
class DatasetInspection:
    report: dict[str, Any]
    stats_cache_uri: str | None = None
    wrote_stats_cache: bool = False


def _target_report(table: pa.Table, target: TargetConfig) -> dict[str, int | bool]:
    candidate_columns = [
        col
        for col in (target.label_index_column, target.label_name_column)
        if col is not None
    ]
    total = table.num_rows
    valid = 0
    for row_index in range(total):
        if any(table.column(col)[row_index].as_py() is not None for col in candidate_columns):
            valid += 1
    missing = total - valid
    return {
        "total_count": total,
        "valid_count": valid,
        "missing_count": missing,
        "drop_sample_count": missing if target.missing_policy == "drop_sample" else 0,
        "mask_objective_count": missing
        if target.missing_policy == "mask_objective"
        else 0,
        "would_fail": missing > 0 and target.missing_policy == "error",
    }


def _class_counts(
    tables: dict[str, pa.Table],
    target: TargetConfig,
    class_index_by_name: dict[str, int] | None,
) -> dict[str, dict[str, int]]:
    return {
        split: {
            str(index): count
            for index, count in sorted(
                Counter(_integer_targets(table, target, class_index_by_name)).items()
            )
        }
        for split, table in tables.items()
    }


def _tabular_feature_columns(value: list[str] | dict[str, Any] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return list(value)


def _decode_image_bytes(
    *,
    table: pa.Table,
    row_index: int,
    cfg: RootConfig,
    storage: Storage,
    manifest_root: Path,
    parquet_reader: LazyParquetImageReader | None = None,
) -> bytes:
    if cfg.data.backend == "parquet_images":
        image = cfg.data.images
        assert image is not None
        assert parquet_reader is not None
        return parquet_reader.read_bytes(
            file_path=str(table.column(ROW_FILE_PATH)[row_index].as_py()),
            row_group=int(table.column(ROW_GROUP_INDEX)[row_index].as_py()),
            row_in_group=int(table.column(ROW_IN_GROUP_INDEX)[row_index].as_py()),
        )
    assert cfg.data.image_uri_column is not None
    uri = table.column(cfg.data.image_uri_column)[row_index].as_py()
    if "://" in uri or str(uri).startswith("/"):
        resolved = str(uri)
    else:
        resolved = str((manifest_root / str(uri)).resolve())
    return storage.read_bytes(resolved)


def _update_content_hash(
    digest: "hashlib._Hash",
    *,
    split: str,
    sample_id: str,
    image_bytes: bytes,
    tabular_values: dict[str, Any],
) -> None:
    digest.update(split.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0image\0")
    digest.update(len(image_bytes).to_bytes(8, "big"))
    digest.update(image_bytes)
    if tabular_values:
        digest.update(b"\0tabular\0")
        digest.update(
            json.dumps(tabular_values, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )


def _dimensions_normalization_and_content_hash(
    *,
    cfg: RootConfig,
    tables: dict[str, pa.Table],
    storage: Storage,
    manifest_root: Path,
    include_dimensions: bool,
    include_normalization: bool,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, str | None]:
    dimensions: list[dict[str, Any]] = []
    channel_sum = None
    channel_sum_sq = None
    pixel_count = 0
    content_digest = hashlib.sha256()
    read_content = include_dimensions or include_normalization
    tabular_columns = _tabular_feature_columns(cfg.data.tabular_feature_columns)
    parquet_reader = None
    if cfg.data.backend == "parquet_images":
        image = cfg.data.images
        assert image is not None
        parquet_reader = LazyParquetImageReader(
            image_column=image.column,
            bytes_field=image.bytes_field,
        )

    for split in sorted(tables):
        table = tables[split]
        for row_index in range(table.num_rows):
            image_bytes = _decode_image_bytes(
                table=table,
                row_index=row_index,
                cfg=cfg,
                storage=storage,
                manifest_root=manifest_root,
                parquet_reader=parquet_reader,
            )
            sample_id = str(table.column(cfg.data.sample_id_column)[row_index].as_py())
            if read_content:
                tabular_values = {
                    column: table.column(column)[row_index].as_py()
                    for column in tabular_columns
                    if column in table.column_names
                }
                _update_content_hash(
                    content_digest,
                    split=split,
                    sample_id=sample_id,
                    image_bytes=image_bytes,
                    tabular_values=tabular_values,
                )
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
            native_w, native_h = img.size
            if include_dimensions:
                dimensions.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "native_width_px": int(native_w),
                        "native_height_px": int(native_h),
                    }
                )
            if include_normalization and split == "train":
                tensor = _decode_to_unit_chw(
                    img,
                    cfg.transforms.image_mode,
                    cfg.transforms.input_bit_depth,
                )
                flat = tensor.reshape(tensor.shape[0], -1)
                sums = flat.sum(dim=1, dtype=flat.dtype)
                sums_sq = (flat * flat).sum(dim=1, dtype=flat.dtype)
                channel_sum = sums if channel_sum is None else channel_sum + sums
                channel_sum_sq = (
                    sums_sq if channel_sum_sq is None else channel_sum_sq + sums_sq
                )
                pixel_count += flat.shape[1]

    normalization = None
    if include_normalization and channel_sum is not None and pixel_count > 0:
        mean = channel_sum / pixel_count
        variance = (channel_sum_sq / pixel_count) - (mean * mean)
        std = variance.clamp_min(0).sqrt()
        normalization = {
            "split": "train",
            "estimated": False,
            "image_mode": cfg.transforms.image_mode,
            "mean": [float(v) for v in mean.tolist()],
            "std": [float(v) for v in std.tolist()],
        }

    dataset_content_hash = (
        f"sha256:{content_digest.hexdigest()}" if read_content else None
    )
    return (dimensions if include_dimensions else None), normalization, dataset_content_hash


def _sidecar_uri(stats_cache_uri: str, suffix: str) -> str:
    path = Path(stats_cache_uri)
    return str(path.with_name(f"{path.stem}.{suffix}.parquet"))


def _write_dimensions_sidecar(
    *,
    storage: Storage,
    stats_cache_uri: str,
    rows: list[dict[str, Any]],
) -> str:
    sidecar_uri = _sidecar_uri(stats_cache_uri, "dimensions")
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("sample_id", pa.string()),
                ("split", pa.string()),
                ("native_width_px", pa.int64()),
                ("native_height_px", pa.int64()),
            ]
        ),
    )
    sidecar_path = storage.localize(sidecar_uri)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, sidecar_path)
    return Path(sidecar_uri).name


def inspect_dataset(
    cfg: RootConfig,
    *,
    storage: Storage | None = None,
    write_stats: bool = False,
    include_dimensions: bool = False,
    include_normalization: bool = False,
) -> DatasetInspection:
    """Inspect a resolved dataset config and optionally write its stats cache."""

    storage = storage or get_storage(cfg.storage)
    data_cfg = cfg.data
    root, files = _discover_files(data_cfg, storage)
    tables = _tables_by_split(data_cfg, files, _needed_columns(data_cfg))
    bundle = build_datasets(cfg, storage)
    class_index_by_name: dict[str, dict[str, int] | None] = {}
    class_mapping_by_target: dict[str, dict[int, str]] = {}
    for target_name, target in data_cfg.targets.items():
        mapping, name_to_index = _resolve_class_mapping(target, list(tables.values()))
        class_mapping_by_target[target_name] = mapping
        class_index_by_name[target_name] = name_to_index

    target_reports = {
        split: {
            target_name: _target_report(table, target)
            for target_name, target in data_cfg.targets.items()
        }
        for split, table in tables.items()
    }
    class_counts_by_target = {
        target_name: _class_counts(tables, target, class_index_by_name[target_name])
        for target_name, target in data_cfg.targets.items()
    }

    counts_per_head: dict[str, dict[str, Any]] = {}
    mapping_per_head: dict[str, dict[str, str]] = {}
    for head_name, head in cfg.model.heads.items():
        target_counts = class_counts_by_target[head.target]
        target_mapping = class_mapping_by_target[head.target]
        ordered_mapping = {
            str(index): target_mapping.get(index, str(index))
            for index in range(head.num_classes)
        }
        counts_per_head[head_name] = {
            "num_classes": head.num_classes,
            "total": sum(target_counts.get("train", {}).values()),
            "counts": target_counts.get("train", {}),
        }
        mapping_per_head[head_name] = ordered_mapping
    bit_depth = 8 if cfg.transforms.input_bit_depth == "auto" else cfg.transforms.input_bit_depth

    dimensions, normalization, dataset_content_hash = _dimensions_normalization_and_content_hash(
        cfg=cfg,
        tables=tables,
        storage=storage,
        manifest_root=root,
        include_dimensions=include_dimensions,
        include_normalization=include_normalization,
    )

    aspects: dict[str, Any] = {
        "class_counts": {
            "split": "train",
            "estimated": False,
            "per_head": counts_per_head,
        },
        "class_mapping": {"per_head": mapping_per_head},
        "bit_depth": {
            "split": "all",
            "value": bit_depth,
            "heterogeneous": False,
        },
    }
    if dimensions is not None:
        aspects["dimensions"] = {
            "split": "all",
            "rows": dimensions,
            "columns": ["sample_id", "split", "native_width_px", "native_height_px"],
        }
    if normalization is not None:
        aspects["normalization"] = normalization

    report = {
        "valid": True,
        "backend": data_cfg.backend,
        "manifest_uri": data_cfg.manifest_uri,
        "dataset_hash": bundle.dataset_hash,
        "dataset_hash_provenance": bundle.dataset_hash_provenance,
        "dataset_content_hash": dataset_content_hash,
        "files": [{"path": path, "size": size} for path, size in bundle.files],
        "splits_present": sorted(tables),
        "targets": target_reports,
        "aspects": aspects,
    }

    wrote = False
    if write_stats and data_cfg.stats_cache_uri is not None:
        cache_aspects = dict(aspects)
        if dimensions is not None:
            dimensions_uri = _write_dimensions_sidecar(
                storage=storage,
                stats_cache_uri=data_cfg.stats_cache_uri,
                rows=dimensions,
            )
            cache_aspects["dimensions"] = {
                "split": "all",
                "parquet_uri": dimensions_uri,
                "columns": [
                    "sample_id",
                    "split",
                    "native_width_px",
                    "native_height_px",
                ],
            }
        payload = {
            "schema_version": "1.0.0",
            "dataset_hash": bundle.dataset_hash,
            "dataset_hash_provenance": bundle.dataset_hash_provenance,
            "dataset_content_hash": dataset_content_hash,
            "created_by": "dojo inspect dataset",
            "dojo_version": "0.3.0.dev0",
            "splits_present": sorted(tables),
            "aspects": cache_aspects,
        }
        storage.write_bytes(
            data_cfg.stats_cache_uri,
            json.dumps(payload, indent=2).encode("utf-8"),
        )
        wrote = True

    return DatasetInspection(
        report=report,
        stats_cache_uri=data_cfg.stats_cache_uri,
        wrote_stats_cache=wrote,
    )
