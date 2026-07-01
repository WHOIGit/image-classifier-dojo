"""Explicit pyarrow schema for the P1 canonical result table.

P1 writes a single tall-Parquet table whose rows are discriminated by
``record_type`` and partitioned per ``training_outputs.results.partition_by``.
The schema is the union of the common provenance columns and the columns
specific to the two P1 record types, ``sample_metadata`` and
``classification_output``. Columns not applicable to a given row are null.

Vector columns (``logits`` / ``probabilities``) use explicit Arrow list types
so the round-trip through Parquet is lossless.
"""

from __future__ import annotations

import pyarrow as pa

# Bumped together with the sidecar schema_version; both share this value so a
# reader can correlate result rows with their `_metadata.json`.
RESULTS_SCHEMA_VERSION = "1.0.0"

# P1 record types (the supervised subset of the record_type taxonomy).
RECORD_TYPE_SAMPLE_METADATA = "sample_metadata"
RECORD_TYPE_CLASSIFICATION_OUTPUT = "classification_output"
RECORD_TYPES = (RECORD_TYPE_SAMPLE_METADATA, RECORD_TYPE_CLASSIFICATION_OUTPUT)

# P1 produces only train-time validation rows.
STAGE_TRAIN_VALIDATION = "train_validation"

_VECTOR = pa.list_(pa.float32())

# Common provenance columns shared by every supervised result record. Only the
# true identity keys are non-nullable; per-row-optional provenance (epoch,
# checkpoint/model/sweep identity) is nullable because static sample metadata
# and non-exported / non-swept P1 runs leave them empty.
_PROVENANCE_FIELDS = [
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("uri", pa.string(), nullable=True),
    pa.field("split", pa.string(), nullable=False),
    pa.field("stage", pa.string(), nullable=False),
    pa.field("record_type", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("config_id", pa.string(), nullable=True),
    pa.field("config_hash", pa.string(), nullable=False),
    pa.field("dataset_id", pa.string(), nullable=True),
    pa.field("dataset_hash", pa.string(), nullable=False),
    pa.field("epoch", pa.int64(), nullable=True),
    pa.field("global_step", pa.int64(), nullable=True),
    pa.field("checkpoint_hash", pa.string(), nullable=True),
    pa.field("model_id", pa.string(), nullable=True),
    pa.field("model_hash", pa.string(), nullable=True),
    pa.field("sweep_id", pa.string(), nullable=True),
    pa.field("sweep_hash", pa.string(), nullable=True),
    pa.field("schema_version", pa.string(), nullable=False),
]

# `record_type=sample_metadata` columns.
_SAMPLE_METADATA_FIELDS = [
    pa.field("native_width_px", pa.int32(), nullable=True),
    pa.field("native_height_px", pa.int32(), nullable=True),
    pa.field("resize_width_px", pa.int32(), nullable=True),
    pa.field("resize_height_px", pa.int32(), nullable=True),
    pa.field("microns_per_pixel", pa.float64(), nullable=True),
    pa.field("aspect_bucket", pa.string(), nullable=True),
    pa.field("bin_id", pa.string(), nullable=True),
    pa.field("bin_uri", pa.string(), nullable=True),
    pa.field("roi_number", pa.int64(), nullable=True),
    pa.field("source_extra_json", pa.string(), nullable=True),
    pa.field("tabular_features_json", pa.string(), nullable=True),
]

# `record_type=classification_output` columns. The row is scoped by `head_name`
# (the head->target link lives in `_metadata.json`); `target_index` / `target_name`
# ground-truth columns are reserved for P2.
_CLASSIFICATION_OUTPUT_FIELDS = [
    pa.field("head_name", pa.string(), nullable=True),
    pa.field("prediction_index", pa.int64(), nullable=True),
    pa.field("prediction_label", pa.string(), nullable=True),
    pa.field("prediction_confidence", pa.float64(), nullable=True),
    pa.field("logits", _VECTOR, nullable=True),
    pa.field("probabilities", _VECTOR, nullable=True),
]


def result_table_schema() -> pa.Schema:
    """Return the union pyarrow schema for the P1 result table."""

    return pa.schema(
        _PROVENANCE_FIELDS + _SAMPLE_METADATA_FIELDS + _CLASSIFICATION_OUTPUT_FIELDS
    )
