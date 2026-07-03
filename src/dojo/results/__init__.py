"""Canonical tall-Parquet result writing and the ``_metadata.json`` sidecar."""

from dojo.results.metadata import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
    CompatibilityMeta,
    JsonColumnMeta,
    RecordTypesMeta,
    ResultsMetadata,
    SampleMetadataMeta,
    write_metadata_json,
)
from dojo.results.reader import ResultReader
from dojo.results.records import (
    Provenance,
    classification_output_record,
    embedding_record,
    sample_metadata_record,
)
from dojo.results.schemas import (
    RECORD_TYPES,
    RECORD_TYPE_EMBEDDING,
    RESULTS_SCHEMA_VERSION,
    STAGE_HOLDOUT_EVAL,
    STAGE_INFER,
    STAGE_TRAIN_VALIDATION,
    result_table_schema,
)
from dojo.results.writer import ResultWriter

__all__ = [
    "ResultWriter",
    "ResultReader",
    "Provenance",
    "sample_metadata_record",
    "classification_output_record",
    "embedding_record",
    "result_table_schema",
    "RECORD_TYPES",
    "RECORD_TYPE_EMBEDDING",
    "RESULTS_SCHEMA_VERSION",
    "STAGE_INFER",
    "STAGE_HOLDOUT_EVAL",
    "STAGE_TRAIN_VALIDATION",
    "ResultsMetadata",
    "SampleMetadataMeta",
    "ClassificationOutputMeta",
    "ClassificationHeadMeta",
    "CompatibilityMeta",
    "RecordTypesMeta",
    "JsonColumnMeta",
    "write_metadata_json",
]
