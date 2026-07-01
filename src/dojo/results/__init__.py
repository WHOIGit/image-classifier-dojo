"""Canonical tall-Parquet result writing and the ``_metadata.json`` sidecar."""

from dojo.results.metadata import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
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
    sample_metadata_record,
)
from dojo.results.schemas import (
    RECORD_TYPES,
    RESULTS_SCHEMA_VERSION,
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
    "result_table_schema",
    "RECORD_TYPES",
    "RESULTS_SCHEMA_VERSION",
    "STAGE_TRAIN_VALIDATION",
    "ResultsMetadata",
    "SampleMetadataMeta",
    "ClassificationOutputMeta",
    "ClassificationHeadMeta",
    "RecordTypesMeta",
    "JsonColumnMeta",
    "write_metadata_json",
]
