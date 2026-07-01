"""A separate reader can open an existing results dir and filter it.

:class:`ResultReader` wraps an ``amplify_db_utils`` ``DuckDBParquetStore``. The
store reloads its ``SchemaRegistry`` in ``__init__``, which round-trips Arrow
``list<…>`` types (``logits`` / ``probabilities``), so a fresh store on an
existing results dir opens cleanly and a separate "reader" can filter by
``stage`` / ``record_type``.
"""

from __future__ import annotations

import pytest
from amplify_db_utils import DuckDBParquetConfig, DuckDBParquetStore

from dojo.results import (
    Provenance,
    ResultReader,
    ResultWriter,
    classification_output_record,
    sample_metadata_record,
)
from dojo.results.schemas import (
    RECORD_TYPE_CLASSIFICATION_OUTPUT,
    RECORD_TYPE_SAMPLE_METADATA,
    STAGE_TRAIN_VALIDATION,
)


def _write(results_dir) -> None:
    prov = Provenance(run_id="r", config_hash="sha256:c", dataset_hash="sha256:d")
    writer = ResultWriter(results_dir, partition_by=["record_type"])
    writer.write_records(
        [
            sample_metadata_record(prov, sample_id="s0", split="val"),
            classification_output_record(
                prov,
                sample_id="s0",
                split="val",
                head_name="species",
                prediction_index=1,
                prediction_label="B",
                prediction_confidence=0.9,
                logits=[0.1, 0.2],
                probabilities=[0.4, 0.6],
                epoch=2,
            ),
        ]
    )


def test_fresh_reader_filters_existing_results(tmp_path):
    results_dir = tmp_path / "results"
    _write(results_dir)

    # A brand-new reader object (no writer state) reads the persisted dir.
    reader = ResultReader(results_dir)
    assert reader.count() == 2
    assert reader.count({"record_type": RECORD_TYPE_SAMPLE_METADATA}) == 1
    assert reader.count({"stage": STAGE_TRAIN_VALIDATION}) == 2

    cls = list(reader.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))
    assert len(cls) == 1
    assert cls[0]["probabilities"] == pytest.approx([0.4, 0.6])


def test_fresh_store_reloads_list_columns(tmp_path):
    """A fresh DuckDBParquetStore opens an existing list-column results dir.

    Guards the registry list-type round-trip that ResultReader relies on for
    out-of-process reads: opening the store reloads the schema (with ``logits``
    / ``probabilities`` list columns) and the list values read back intact.
    """
    results_dir = tmp_path / "results"
    _write(results_dir)

    store = DuckDBParquetStore(DuckDBParquetConfig(root=str(results_dir)))
    assert store.count("results") == 2
    cls = list(store.read("results", filters={"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))
    assert len(cls) == 1
    assert cls[0]["probabilities"] == pytest.approx([0.4, 0.6])
