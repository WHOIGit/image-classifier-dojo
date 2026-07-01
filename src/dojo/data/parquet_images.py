"""``parquet_images`` backend: discover manifest files, route splits, build datasets.

P1 supports the two split sources from the schema: ``split_column`` (canonical
split values in a manifest column) and ``split_from_filename`` (assign each
file's rows a split by matching its basename against per-split globs). Rows are
held in memory (the backend is meant for inlined-image Parquet — fixtures and
modest datasets); image bytes are decoded lazily by
:class:`ParquetImagesDataset`.
"""

from __future__ import annotations

import fnmatch
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dojo.config_schemas.root import DataConfig, RootConfig, TargetConfig
from dojo.data.dataset import ParquetImagesDataset
from dojo.data.identity import compute_dataset_hash
from dojo.data.transforms import build_image_transform
from dojo.storage import Storage, get_storage


class DatasetConfigError(ValueError):
    """Raised for backend / manifest configuration problems."""


@dataclass(frozen=True)
class DataBundle:
    datasets: dict[str, ParquetImagesDataset]
    dataset_hash: str
    dataset_hash_provenance: str
    files: list[tuple[str, int]]
    class_counts: dict[str, dict[int, int]]
    target_name: str
    # Resolved class index -> readable name for the sole P1 target. Missing
    # indices (no name available) fall back to their index string downstream.
    class_mapping: dict[int, str]


def _discover_files(cfg: DataConfig, storage: Storage) -> tuple[Path, list[Path]]:
    root = storage.localize(cfg.manifest_uri)
    if not root.exists():
        raise DatasetConfigError(f"data.manifest_uri does not exist: {root}")
    if root.is_file():
        return root.parent, [root]
    pattern = cfg.file_pattern or "*.parquet"
    files = sorted(root.glob(pattern))
    if not files:
        raise DatasetConfigError(
            f"no files matching {pattern!r} under data.manifest_uri: {root}"
        )
    return root, files


def _needed_columns(cfg: DataConfig) -> list[str]:
    assert cfg.images is not None
    columns = [cfg.sample_id_column, cfg.images.column]
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
    if cfg.split_from_filename is not None:
        grouped: dict[str, list[pa.Table]] = defaultdict(list)
        for path in files:
            split = _split_for_filename(path.name, cfg.split_from_filename)
            if split is None:
                continue
            grouped[split].append(pq.read_table(path, columns=columns))
        return {split: pa.concat_tables(parts) for split, parts in grouped.items()}

    # split_column mode: one combined table, partitioned by the column values.
    combined = pa.concat_tables([pq.read_table(p, columns=columns) for p in files])
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
        return class_mapping, None

    return {}, None  # index-only: names fall back to index strings downstream


def _integer_targets(
    table: pa.Table, target: "TargetConfig", class_index_by_name: dict[str, int] | None
) -> list[int]:
    if target.label_index_column is not None:
        return [int(v) for v in table.column(target.label_index_column).to_pylist()]
    assert class_index_by_name is not None
    names = table.column(target.label_name_column).to_pylist()
    return [class_index_by_name[str(n)] for n in names]


def _sole_target(cfg: DataConfig) -> str:
    if len(cfg.targets) != 1:
        raise DatasetConfigError(
            "P1 supports exactly one data target; "
            f"got {sorted(cfg.targets)}"
        )
    return next(iter(cfg.targets))


def build_datasets(cfg: RootConfig, storage: Storage | None = None) -> DataBundle:
    """Build per-split datasets, the ``dataset_hash``, and per-split class counts.

    ``cfg`` must be resolved (``transforms.inference_pipeline`` populated): the
    ``train`` split uses ``transforms.pipeline`` and every other split uses the
    derived inference pipeline.
    """

    storage = storage or get_storage(cfg.storage)
    data_cfg = cfg.data
    if data_cfg.backend != "parquet_images":
        raise DatasetConfigError(
            f"parquet_images backend required; got {data_cfg.backend!r}"
        )
    if cfg.transforms.inference_pipeline is None:
        raise DatasetConfigError(
            "transforms.inference_pipeline is unresolved; resolve the config first"
        )

    target_name = _sole_target(data_cfg)
    root, files = _discover_files(data_cfg, storage)
    file_ids = sorted(
        (str(path.relative_to(root)), path.stat().st_size) for path in files
    )
    dataset_hash, provenance = compute_dataset_hash(data_cfg, file_ids)

    tables = _tables_by_split(data_cfg, files, _needed_columns(data_cfg))

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

    target = data_cfg.targets[target_name]
    class_mapping, class_index_by_name = _resolve_class_mapping(target, list(tables.values()))

    datasets: dict[str, ParquetImagesDataset] = {}
    class_counts: dict[str, dict[int, int]] = {}
    for split, table in tables.items():
        datasets[split] = ParquetImagesDataset(
            table,
            cfg=data_cfg,
            target_name=target_name,
            split=split,
            transform=train_transform if split == "train" else eval_transform,
            class_index_by_name=class_index_by_name,
        )
        counts = Counter(_integer_targets(table, target, class_index_by_name))
        class_counts[split] = dict(sorted(counts.items()))

    return DataBundle(
        datasets=datasets,
        dataset_hash=dataset_hash,
        dataset_hash_provenance=provenance,
        files=file_ids,
        class_counts=class_counts,
        target_name=target_name,
        class_mapping=class_mapping,
    )
