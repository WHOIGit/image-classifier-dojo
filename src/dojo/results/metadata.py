"""The ``_metadata.json`` result sidecar schema and writer.

Top-level ``record_types`` (there is no top-level ``heads`` key), an optional
``compatibility`` block, and per-record-type semantic / provenance metadata.
The compatibility-hash extractors are P2.5, so in the P1 slice the
``compatibility`` block is ``null``.

These Pydantic models are the schema the sidecar validates against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from dojo.results.schemas import RESULTS_SCHEMA_VERSION


class _StrictMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassificationHeadMeta(_StrictMeta):
    target: str
    head_hash: str | None = None
    classes: list[str]
    class_mapping: dict[str, str]


class ClassificationOutputMeta(_StrictMeta):
    heads: dict[str, ClassificationHeadMeta]


class JsonColumnMeta(_StrictMeta):
    columns: list[str]


class SampleMetadataMeta(_StrictMeta):
    description: str | None = None
    source_extra_json: JsonColumnMeta | None = None
    tabular_features_json: JsonColumnMeta | None = None


class RecordTypesMeta(_StrictMeta):
    sample_metadata: SampleMetadataMeta | None = None
    classification_output: ClassificationOutputMeta | None = None


class CompatibilityMeta(_StrictMeta):
    """Compatibility hashes + canonical source sub-blocks.

    All optional for P1: the extractors land in P2.5. When a hash is present
    its paired ``*_source`` object should be too, but the P1 schema does not
    enforce that pairing yet.
    """

    target_schema_hash: str | None = None
    target_schema_source: dict[str, Any] | None = None
    class_mapping_hash: str | None = None
    class_mapping_source: dict[str, Any] | None = None
    model_config_hash: str | None = None
    model_config_source: dict[str, Any] | None = None
    preprocessing_hash: str | None = None
    preprocessing_source: dict[str, Any] | None = None


class ResultsMetadata(_StrictMeta):
    schema_name: str = "dojo.supervised_results"
    schema_version: str = RESULTS_SCHEMA_VERSION
    created_by: str = "dojo"
    run_id: str
    compatibility: CompatibilityMeta | None = None
    record_types: RecordTypesMeta


def write_metadata_json(
    path: str | os.PathLike[str],
    metadata: ResultsMetadata,
) -> Path:
    """Serialize ``metadata`` to ``path`` (``…/results/_metadata.json``)."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        # No sort_keys: insertion order keeps the numeric class_mapping keys
        # ("0","1","2",…) in ascending order. Do not add sort_keys — it would
        # order them as strings ("0","1","10","2",…).
        json.dumps(metadata.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return out
