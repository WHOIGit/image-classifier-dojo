"""Canonical tall-Parquet result writer over ``amplify-db-utils``.

Wraps a single :class:`DuckDBParquetStore` table rooted at the run's
``results/`` directory. Rows for every P1 record type share one union schema
(:func:`result_table_schema`) and are partitioned per
``training_outputs.results.partition_by`` (default ``[record_type]``), so a
reader can prune / filter by ``stage`` and ``record_type``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
from amplify_db_utils import DuckDBParquetConfig, DuckDBParquetStore

from dojo.results.schemas import result_table_schema

_TABLE = "results"


def _records_to_table(records: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a ``pa.Table`` from dict rows using the explicit result schema.
    """

    columns = {
        field.name: pa.array([row.get(field.name) for row in records], type=field.type)
        for field in schema
    }
    return pa.table(columns, schema=schema)


class ResultWriter:
    """Append-only writer for one run's canonical result table."""

    def __init__(
        self,
        results_dir: str | os.PathLike[str],
        *,
        partition_by: Iterable[str] = ("record_type",),
        table: str = _TABLE,
    ) -> None:
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._table = table
        self._partition_by = list(partition_by)
        self._schema = result_table_schema()
        self._store = DuckDBParquetStore(DuckDBParquetConfig(root=str(self._dir)))
        self._store.create_table(
            self._table,
            self._schema,
            partition_by=self._partition_by or None,
        )

    @property
    def results_dir(self) -> Path:
        return self._dir

    @property
    def table(self) -> str:
        return self._table

    def write_records(self, records: list[dict[str, Any]]) -> None:
        """Append result rows. Unknown keys are dropped; missing cols null."""

        if not records:
            return
        self._store.write(self._table, _records_to_table(records, self._schema))

    def read(self, filters: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Iterate rows, optionally filtered (e.g. ``{"record_type": "..."}``)."""

        return self._store.read(self._table, filters)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return self._store.count(self._table, filters)
