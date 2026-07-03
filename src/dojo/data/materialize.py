"""Materialize embedded Parquet images into a content-addressed disk cache."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from dojo.config_schemas.root import DataConfig, StorageConfig
from dojo.data.parquet_rows import (
    ROW_FILE_PATH,
    ROW_GROUP_INDEX,
    ROW_IMAGE_PATH,
    ROW_IN_GROUP_INDEX,
    ROW_MATERIALIZED_IMAGE_PATH,
)

_CACHE_SCHEMA_VERSION = "1.0.0"
_CHUNK_SIZE = 1024 * 1024


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def _progress(enabled: bool, *, mode: str) -> Progress:
    if mode == "bytes":
        columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
        )
    else:
        columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
    return Progress(*columns, disable=not enabled)


def compute_dataset_content_hash(
    *,
    root: Path,
    files: list[Path],
    progress_enabled: bool,
) -> str:
    """Hash source Parquet file names and bytes for cache identity."""

    digest = hashlib.sha256()
    total = sum(path.stat().st_size for path in files)
    with _progress(progress_enabled, mode="bytes") as progress:
        task = progress.add_task("Hashing dataset content", total=total)
        for path in sorted(files):
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    progress.update(task, advance=len(chunk))
    return f"sha256:{digest.hexdigest()}"


def _cache_base_dir(cfg: DataConfig, storage_cfg: StorageConfig) -> Path:
    configured = cfg.image_cache.dir
    if configured is None:
        configured = str(Path(storage_cfg.local_cache_dir) / "materialized_images")
    return Path(configured).expanduser().resolve()


def _cache_dir_for_hash(
    base: Path,
    dataset_content_hash: str,
    *,
    cache_bust: str | None,
) -> Path:
    component = dataset_content_hash.removeprefix("sha256:")
    if cache_bust is not None:
        component = f"{component}-{_safe_component(cache_bust)}"
    return base / component


def _marker_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _marker_matches(
    *,
    cache_dir: Path,
    dataset_content_hash: str,
    files: list[tuple[str, int]],
    cache_bust: str | None,
) -> bool:
    marker = _marker_path(cache_dir)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("schema_version") == _CACHE_SCHEMA_VERSION
        and payload.get("dataset_content_hash") == dataset_content_hash
        and payload.get("cache_bust") == cache_bust
        and payload.get("complete") is True
        and payload.get("files") == [
            {"path": rel, "size": size} for rel, size in files
        ]
    )


def _materialized_path(
    *,
    cache_dir: Path,
    source_file: Path,
    row_group: int,
    row_in_group: int,
    image_path: str | None,
) -> Path:
    suffix = Path(image_path or "").suffix or ".img"
    return (
        cache_dir
        / "images"
        / _safe_component(source_file.stem)
        / f"rg{row_group:05d}_row{row_in_group:06d}{suffix}"
    )


def _write_marker(
    *,
    cache_dir: Path,
    dataset_content_hash: str,
    files: list[tuple[str, int]],
    image_count: int,
    cache_bust: str | None,
) -> None:
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset_content_hash": dataset_content_hash,
        "cache_bust": cache_bust,
        "complete": True,
        "image_count": image_count,
        "files": [{"path": rel, "size": size} for rel, size in files],
    }
    _marker_path(cache_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _materialize_files(
    *,
    cfg: DataConfig,
    root: Path,
    files: list[Path],
    cache_dir: Path,
    progress_enabled: bool,
) -> int:
    image = cfg.images
    assert image is not None
    image_count = 0
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    with _progress(progress_enabled, mode="count") as progress:
        task = progress.add_task("Materializing image cache", total=total_rows)
        for path in sorted(files):
            parquet_file = pq.ParquetFile(path)
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                columns = [f"{image.column}.{image.bytes_field}"]
                if image.path_field is not None:
                    columns.append(f"{image.column}.{image.path_field}")
                table = parquet_file.read_row_group(row_group_index, columns=columns)
                struct = table.column(image.column)
                bytes_column = pc.struct_field(struct, image.bytes_field)
                if image.path_field is None:
                    path_values: list[str | None] = [None] * len(bytes_column)
                else:
                    path_values = [
                        None if value is None else str(value)
                        for value in pc.struct_field(struct, image.path_field).to_pylist()
                    ]

                for row_in_group in range(len(bytes_column)):
                    image_bytes = bytes_column[row_in_group].as_py()
                    out = _materialized_path(
                        cache_dir=cache_dir,
                        source_file=path.relative_to(root),
                        row_group=row_group_index,
                        row_in_group=row_in_group,
                        image_path=path_values[row_in_group],
                    )
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(image_bytes)
                    image_count += 1
                    progress.update(task, advance=1)
                del table
                pa.default_memory_pool().release_unused()
    return image_count


def _cache_column_for_table(table: pa.Table, cache_dir: Path) -> pa.Array:
    paths: list[str] = []
    source_files = table.column(ROW_FILE_PATH).to_pylist()
    row_groups = table.column(ROW_GROUP_INDEX).to_pylist()
    rows_in_group = table.column(ROW_IN_GROUP_INDEX).to_pylist()
    image_paths = table.column(ROW_IMAGE_PATH).to_pylist()
    for source_file, row_group, row_in_group, image_path in zip(
        source_files,
        row_groups,
        rows_in_group,
        image_paths,
    ):
        paths.append(
            str(
                _materialized_path(
                    cache_dir=cache_dir,
                    source_file=Path(str(source_file)).resolve(),
                    row_group=int(row_group),
                    row_in_group=int(row_in_group),
                    image_path=None if image_path is None else str(image_path),
                )
            )
        )
    return pa.array(paths, type=pa.string())


def materialize_image_cache(
    *,
    cfg: DataConfig,
    storage_cfg: StorageConfig,
    root: Path,
    files: list[Path],
    file_ids: list[tuple[str, int]],
    tables: dict[str, pa.Table],
) -> tuple[dict[str, pa.Table], str, Path]:
    """Ensure image files exist on disk and add cache paths to split tables."""

    dataset_content_hash = compute_dataset_content_hash(
        root=root,
        files=files,
        progress_enabled=cfg.image_cache.progress,
    )
    cache_dir = _cache_dir_for_hash(
        _cache_base_dir(cfg, storage_cfg),
        dataset_content_hash,
        cache_bust=cfg.image_cache.cache_bust,
    )
    if cfg.image_cache.clobber and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.image_cache.force_rebuild or not _marker_matches(
        cache_dir=cache_dir,
        dataset_content_hash=dataset_content_hash,
        files=file_ids,
        cache_bust=cfg.image_cache.cache_bust,
    ):
        image_count = _materialize_files(
            cfg=cfg,
            root=root,
            files=files,
            cache_dir=cache_dir,
            progress_enabled=cfg.image_cache.progress,
        )
        _write_marker(
            cache_dir=cache_dir,
            dataset_content_hash=dataset_content_hash,
            files=file_ids,
            image_count=image_count,
            cache_bust=cfg.image_cache.cache_bust,
        )

    cached_tables = {
        split: table.append_column(
            ROW_MATERIALIZED_IMAGE_PATH,
            _cache_column_for_table(table, cache_dir),
        )
        for split, table in tables.items()
    }
    return cached_tables, dataset_content_hash, cache_dir
