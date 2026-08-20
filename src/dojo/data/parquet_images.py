"""Manifest backends: discover files, route splits, build datasets.

Supports the two split sources from the schema: ``split_column`` (canonical
split values in a manifest column) and ``split_from_filename`` (assign each
file's rows a split by matching its basename against per-split globs). Rows are
held in memory for the current supervised platform slice; image bytes are
decoded lazily by the dataset classes.
"""

from __future__ import annotations

import fnmatch
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from dojo.config_schemas.root import DataConfig, RootConfig, TargetConfig
from dojo.data.dataset import ManifestImagesDataset, ParquetImagesDataset
from dojo.data.identity import compute_dataset_hash
from dojo.data.materialize import materialize_image_cache
from dojo.data.parquet_rows import (
    ROW_FILE_PATH,
    ROW_GROUP_INDEX,
    ROW_IMAGE_PATH,
    ROW_IN_GROUP_INDEX,
    parquet_row_reference_columns,
)
from dojo.data.transforms import build_image_transform, find_aspect_bucket_step
from dojo.storage import Storage, get_storage


class DatasetConfigError(ValueError):
    """Raised for backend / manifest configuration problems."""


@dataclass(frozen=True)
class DataBundle:
    datasets: dict[str, ParquetImagesDataset | ManifestImagesDataset]
    dataset_hash: str
    dataset_hash_provenance: str
    files: list[tuple[str, int]]
    dataset_content_hash: str | None
    materialized_image_cache_dir: Path | None
    class_counts: dict[str, dict[int, int]]
    class_counts_by_target: dict[str, dict[str, dict[int, int]]]
    target_name: str
    target_names: tuple[str, ...]
    class_mapping: dict[int, str]
    class_mapping_by_target: dict[str, dict[int, str]]


def _discover_files(cfg: DataConfig, storage: Storage) -> tuple[Path, list[Path]]:
    root = storage.localize(cfg.manifest_uri)
    if not root.exists():
        raise DatasetConfigError(f"data.manifest_uri does not exist: {root}")
    if root.is_file():
        return root.parent, [root]
    default_pattern = "*.csv" if cfg.backend == "csv_manifest" else "*.parquet"
    pattern = cfg.file_pattern or default_pattern
    # Sort on the POSIX-relative name: Path ordering is case-insensitive on
    # Windows, which would give the manifest files a different order (and so a
    # different dataset_hash / row order) than on Linux.
    files = sorted(root.glob(pattern), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise DatasetConfigError(
            f"no files matching {pattern!r} under data.manifest_uri: {root}"
        )
    return root, files


def _needed_columns(cfg: DataConfig) -> list[str]:
    columns = [cfg.sample_id_column]
    if cfg.backend != "parquet_images":
        assert cfg.image_uri_column is not None
        columns.append(cfg.image_uri_column)
    for target in cfg.targets.values():
        if target.label_index_column is not None:
            columns.append(target.label_index_column)
        if target.label_name_column is not None:
            columns.append(target.label_name_column)
    columns.extend(cfg.source_extra_columns)
    if cfg.split_column is not None:
        columns.append(cfg.split_column)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return [c for c in columns if not (c in seen or seen.add(c))]


def _split_for_filename(name: str, patterns: dict[str, str]) -> str | None:
    matches = [split for split, glob in patterns.items() if fnmatch.fnmatch(name, glob)]
    if len(matches) > 1:
        raise DatasetConfigError(
            f"file {name!r} matches multiple split_from_filename patterns: {matches}"
        )
    return matches[0] if matches else None


def _tables_by_split(
    cfg: DataConfig,
    files: list[Path],
    columns: list[str],
) -> dict[str, pa.Table]:
    def read_parquet_images_table(path: Path) -> pa.Table:
        image = cfg.images
        assert image is not None
        table = pq.read_table(path, columns=columns)
        if image.path_field is None:
            image_path = pa.nulls(table.num_rows, type=pa.string())
        else:
            path_table = pq.read_table(
                path,
                columns=[f"{image.column}.{image.path_field}"],
            )
            image_path = path_table.column(0)

        parquet_file = pq.ParquetFile(path)
        row_group_sizes = [
            parquet_file.metadata.row_group(index).num_rows
            for index in range(parquet_file.metadata.num_row_groups)
        ]
        row_refs = parquet_row_reference_columns(
            str(path.resolve()),
            row_group_sizes=row_group_sizes,
        )
        if sum(row_group_sizes) != table.num_rows:
            raise DatasetConfigError(
                f"row-reference length mismatch for parquet_images file: {path}"
            )
        return table.append_column(ROW_FILE_PATH, row_refs[ROW_FILE_PATH]).append_column(
            ROW_GROUP_INDEX,
            row_refs[ROW_GROUP_INDEX],
        ).append_column(
            ROW_IN_GROUP_INDEX,
            row_refs[ROW_IN_GROUP_INDEX],
        ).append_column(
            ROW_IMAGE_PATH,
            image_path,
        )

    def read_table(path: Path) -> pa.Table:
        if cfg.backend == "csv_manifest":
            return pacsv.read_csv(
                path,
                convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
            ).select(columns)
        if cfg.backend == "parquet_images":
            return read_parquet_images_table(path)
        return pq.read_table(path, columns=columns)

    if cfg.split_from_filename is not None:
        grouped: dict[str, list[pa.Table]] = defaultdict(list)
        for path in files:
            split = _split_for_filename(path.name, cfg.split_from_filename)
            if split is None:
                continue
            grouped[split].append(read_table(path))
        return {split: pa.concat_tables(parts) for split, parts in grouped.items()}

    # split_column mode: one combined table, partitioned by the column values.
    combined = pa.concat_tables([read_table(p) for p in files])
    split_values = combined.column(cfg.split_column).to_pylist()
    tables: dict[str, pa.Table] = {}
    for split in sorted(set(split_values)):
        mask = pa.array([v == split for v in split_values])
        tables[split] = combined.filter(mask)
    return tables


def _resolve_class_mapping(
    target: "TargetConfig", tables: list[pa.Table]
) -> tuple[dict[int, str], dict[str, int] | None]:
    """Resolve ``(class_mapping, class_index_by_name)`` across all split tables.

    ``class_mapping`` is class index -> readable name. ``class_index_by_name`` is
    non-None only when the target has no index column, in which case class indices
    are assigned to distinct names alphabetically and the dataset maps names ->
    indices.
    """
    index_col, name_col = target.label_index_column, target.label_name_column

    if index_col is None:
        assert name_col is not None  # guaranteed by TargetConfig validation
        names: set[str] = set()
        for table in tables:
            names.update(str(n) for n in table.column(name_col).to_pylist() if n is not None)
        class_index_by_name = {name: index for index, name in enumerate(sorted(names))}
        return {index: name for name, index in class_index_by_name.items()}, class_index_by_name

    if name_col is not None:
        class_mapping: dict[int, str] = {}
        for table in tables:
            indices = table.column(index_col).to_pylist()
            names_col = table.column(name_col).to_pylist()
            for index, name in zip(indices, names_col):
                if index is None or name is None:
                    continue
                index, name = int(index), str(name)
                existing = class_mapping.setdefault(index, name)
                if existing != name:
                    raise DatasetConfigError(
                        f"class index {index} maps to multiple names: "
                        f"{existing!r} and {name!r}"
                    )
        metadata_mapping = _class_mapping_from_dojo_metadata(index_col, tables)
        for index, name in metadata_mapping.items():
            class_mapping.setdefault(index, name)
        return class_mapping, None

    metadata_mapping = _class_mapping_from_schema_metadata(index_col, tables)
    return metadata_mapping, None


def _extract_names_from_feature(value: object) -> list[str] | None:
    if not isinstance(value, dict):
        return None
    names = value.get("names")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    if isinstance(names, dict):
        ordered = sorted(names.items(), key=lambda item: int(item[0]))
        if all(isinstance(name, str) for _, name in ordered):
            return [name for _, name in ordered]
    return None


def _metadata_class_names(metadata: dict[bytes, bytes] | None, index_col: str) -> list[str] | None:
    if not metadata:
        return None
    for key in (b"dojo:class_names", b"class_names"):
        raw_names = metadata.get(key)
        if raw_names is None:
            continue
        try:
            names = json.loads(raw_names.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return list(names)

    for key in (b"huggingface", b"features"):
        raw = metadata.get(key)
        if raw is None:
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        features = payload.get("features") if isinstance(payload, dict) else None
        if features is None and isinstance(payload, dict):
            features = payload.get("info", {}).get("features")
        if not isinstance(features, dict):
            continue
        names = _extract_names_from_feature(features.get(index_col))
        if names is not None:
            return names
    return None


def _dojo_class_names(metadata: dict[bytes, bytes] | None) -> list[str] | None:
    if not metadata:
        return None
    raw_names = metadata.get(b"dojo:class_names")
    if raw_names is None:
        return None
    try:
        names = json.loads(raw_names.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    return None


def _class_mapping_from_dojo_metadata(
    index_col: str | None,
    tables: list[pa.Table],
) -> dict[int, str]:
    if index_col is None:
        return {}
    for table in tables:
        if index_col not in table.column_names:
            continue
        field = table.schema.field(index_col)
        names = _dojo_class_names(field.metadata)
        if names is None:
            names = _dojo_class_names(table.schema.metadata)
        if names is None:
            continue
        return {index: label for index, label in enumerate(names)}
    return {}


def _class_mapping_from_schema_metadata(
    index_col: str | None,
    tables: list[pa.Table],
) -> dict[int, str]:
    if index_col is None:
        return {}
    for table in tables:
        if index_col not in table.column_names:
            continue
        field = table.schema.field(index_col)
        names = _metadata_class_names(field.metadata, index_col)
        if names is None:
            names = _metadata_class_names(table.schema.metadata, index_col)
        if names is None:
            continue
        return {index: label for index, label in enumerate(names)}
    return {}


def _integer_targets(
    table: pa.Table, target: "TargetConfig", class_index_by_name: dict[str, int] | None
) -> list[int]:
    if target.label_index_column is not None:
        return [
            int(v)
            for v in table.column(target.label_index_column).to_pylist()
            if v is not None
        ]
    assert class_index_by_name is not None
    names = table.column(target.label_name_column).to_pylist()
    return [
        class_index_by_name[str(n)]
        for n in names
        if n is not None
    ]


def _target_is_missing(table: pa.Table, target: TargetConfig, row_index: int) -> bool:
    candidate_columns = [
        column
        for column in (target.label_index_column, target.label_name_column)
        if column is not None
    ]
    return all(table.column(column)[row_index].as_py() is None for column in candidate_columns)


def _apply_missing_policies(
    tables: dict[str, pa.Table],
    data_cfg: DataConfig,
) -> dict[str, pa.Table]:
    drop_targets = [
        target
        for target in data_cfg.targets.values()
        if target.missing_policy == "drop_sample"
    ]
    if not drop_targets:
        return tables

    filtered: dict[str, pa.Table] = {}
    for split, table in tables.items():
        keep = [
            not any(
                _target_is_missing(table, target, row_index)
                for target in drop_targets
            )
            for row_index in range(table.num_rows)
        ]
        filtered[split] = table.filter(pa.array(keep))
    return filtered


def build_datasets(cfg: RootConfig, storage: Storage | None = None) -> DataBundle:
    """Build per-split datasets, the ``dataset_hash``, and per-split class counts.

    ``cfg`` must be resolved (``transforms.inference_pipeline`` populated): the
    ``train`` split uses ``transforms.pipeline`` and every other split uses the
    derived inference pipeline.
    """

    storage = storage or get_storage(cfg.storage)
    data_cfg = cfg.data
    if cfg.transforms.inference_pipeline is None:
        raise DatasetConfigError(
            "transforms.inference_pipeline is unresolved; resolve the config first"
        )

    target_names = tuple(data_cfg.targets)
    target_name = target_names[0]
    root, files = _discover_files(data_cfg, storage)
    # as_posix(), not str(): a Windows "sub\\part-0.parquet" would otherwise
    # hash differently from the same dataset's "sub/part-0.parquet" on Linux.
    file_ids = sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size) for path in files
    )
    dataset_hash, provenance = compute_dataset_hash(data_cfg, file_ids)

    tables = _apply_missing_policies(
        _tables_by_split(data_cfg, files, _needed_columns(data_cfg)),
        data_cfg,
    )
    dataset_content_hash = None
    materialized_image_cache_dir = None
    if data_cfg.backend == "parquet_images" and data_cfg.image_cache.enabled:
        tables, dataset_content_hash, materialized_image_cache_dir = (
            materialize_image_cache(
                cfg=data_cfg,
                storage_cfg=cfg.storage,
                root=root,
                files=files,
                file_ids=file_ids,
                tables=tables,
            )
        )

    train_transform = build_image_transform(
        cfg.transforms.pipeline,
        image_mode=cfg.transforms.image_mode,
        input_bit_depth=cfg.transforms.input_bit_depth,
    )
    eval_transform = build_image_transform(
        cfg.transforms.inference_pipeline,
        image_mode=cfg.transforms.image_mode,
        input_bit_depth=cfg.transforms.input_bit_depth,
    )
    train_aspect_bucket_step = find_aspect_bucket_step(cfg.transforms.pipeline)
    eval_aspect_bucket_step = find_aspect_bucket_step(cfg.transforms.inference_pipeline)

    class_mapping_by_target: dict[str, dict[int, str]] = {}
    class_index_by_name: dict[str, dict[str, int] | None] = {}
    for name in target_names:
        mapping, name_to_index = _resolve_class_mapping(
            data_cfg.targets[name],
            list(tables.values()),
        )
        class_mapping_by_target[name] = mapping
        class_index_by_name[name] = name_to_index

    datasets: dict[str, ParquetImagesDataset | ManifestImagesDataset] = {}
    class_counts_by_target: dict[str, dict[str, dict[int, int]]] = {}
    for split, table in tables.items():
        transform = train_transform if split == "train" else eval_transform
        aspect_bucket_step = (
            train_aspect_bucket_step if split == "train" else eval_aspect_bucket_step
        )
        if data_cfg.backend == "parquet_images":
            datasets[split] = ParquetImagesDataset(
                table,
                cfg=data_cfg,
                target_names=target_names,
                split=split,
                transform=transform,
                aspect_bucket_step=aspect_bucket_step,
                class_index_by_name=class_index_by_name,
            )
        else:
            datasets[split] = ManifestImagesDataset(
                table,
                cfg=data_cfg,
                target_names=target_names,
                split=split,
                transform=transform,
                storage=storage,
                manifest_root=str(root),
                aspect_bucket_step=aspect_bucket_step,
                class_index_by_name=class_index_by_name,
            )
        class_counts_by_target[split] = {}
        for name in target_names:
            counts = Counter(
                _integer_targets(table, data_cfg.targets[name], class_index_by_name[name])
            )
            class_counts_by_target[split][name] = dict(sorted(counts.items()))

    return DataBundle(
        datasets=datasets,
        dataset_hash=dataset_hash,
        dataset_hash_provenance=provenance,
        files=file_ids,
        dataset_content_hash=dataset_content_hash,
        materialized_image_cache_dir=materialized_image_cache_dir,
        class_counts={
            split: per_target[target_name]
            for split, per_target in class_counts_by_target.items()
        },
        class_counts_by_target=class_counts_by_target,
        target_name=target_name,
        target_names=target_names,
        class_mapping=class_mapping_by_target[target_name],
        class_mapping_by_target=class_mapping_by_target,
    )
