"""Dojo-owned reader for the canonical result table.

A thin wrapper over an ``amplify_db_utils`` :class:`DuckDBParquetStore` for the
hive-partitioned Parquet written by :class:`ResultWriter`. A separate "reader"
process can open an existing results dir and filter by ``stage`` /
``record_type`` without any writer state.

Filter semantics come straight from the store: equality ``{"field": value}``,
set ``{"field": {"in": [...]}}``, and range ``{"field": {"gte": x, "lt": y}}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from amplify_db_utils import DuckDBParquetConfig, DuckDBParquetStore

_TABLE = "results"


class ResultReader:
    """Read / filter / count rows from a run's canonical result table."""

    def __init__(
        self,
        results_dir: str | os.PathLike[str],
        *,
        table: str = _TABLE,
    ) -> None:
        self._table = table
        self._store = DuckDBParquetStore(
            DuckDBParquetConfig(root=str(Path(results_dir)))
        )

    def read(self, filters: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        return self._store.read(self._table, filters)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return self._store.count(self._table, filters)
