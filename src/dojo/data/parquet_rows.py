"""Lightweight row references for embedded-image Parquet datasets."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROW_FILE_PATH = "__dojo_file_path"
ROW_GROUP_INDEX = "__dojo_row_group"
ROW_IN_GROUP_INDEX = "__dojo_row_in_group"
ROW_IMAGE_PATH = "__dojo_image_path"
ROW_MATERIALIZED_IMAGE_PATH = "__dojo_materialized_image_path"


class LazyParquetImageReader:
    """Read embedded image bytes by row reference without retaining image columns.

    PyArrow's Parquet random access is row-group based, so reading one embedded
    image still transiently reads the row group's image column. The important
    memory invariant is that workers must not retain those row-group tables
    after a sample is decoded.
    """

    def __init__(self, *, image_column: str, bytes_field: str) -> None:
        self._image_column = image_column
        self._bytes_field = bytes_field
        self._files: dict[str, pq.ParquetFile] = {}

    def _parquet_file(self, file_path: str) -> pq.ParquetFile:
        parquet_file = self._files.get(file_path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(file_path)
            self._files[file_path] = parquet_file
        return parquet_file

    def read_bytes(
        self,
        *,
        file_path: str,
        row_group: int,
        row_in_group: int,
    ) -> bytes:
        table = self._parquet_file(file_path).read_row_group(
            row_group,
            columns=[f"{self._image_column}.{self._bytes_field}"],
        )
        struct = table.column(self._image_column)
        image_bytes = pc.struct_field(struct, self._bytes_field)[row_in_group].as_py()
        del table
        pa.default_memory_pool().release_unused()
        return image_bytes


def parquet_row_reference_columns(path: str, *, row_group_sizes: list[int]) -> dict[str, pa.Array]:
    """Return hidden columns that locate each logical row in a Parquet file."""

    file_paths: list[str] = []
    row_groups: list[int] = []
    row_in_groups: list[int] = []
    for row_group, rows in enumerate(row_group_sizes):
        file_paths.extend([path] * rows)
        row_groups.extend([row_group] * rows)
        row_in_groups.extend(range(rows))
    return {
        ROW_FILE_PATH: pa.array(file_paths, type=pa.string()),
        ROW_GROUP_INDEX: pa.array(row_groups, type=pa.int32()),
        ROW_IN_GROUP_INDEX: pa.array(row_in_groups, type=pa.int32()),
    }
