"""Builders that stamp canonical provenance onto P1 result rows.

These keep call sites (the training loop, later infer/eval) from hand-rolling
the common provenance columns and the per-record-type column set. Every builder
returns a plain ``dict`` ready for :meth:`ResultWriter.write_records`; the
result schema fills any column the builder omits with null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from dojo.results.schemas import (
    RECORD_TYPE_CLASSIFICATION_OUTPUT,
    RECORD_TYPE_EMBEDDING,
    RECORD_TYPE_SAMPLE_METADATA,
    RESULTS_SCHEMA_VERSION,
    STAGE_TRAIN_VALIDATION,
)


@dataclass(frozen=True)
class Provenance:
    """Run-constant provenance shared by every row of a single run."""

    run_id: str
    config_hash: str
    dataset_hash: str
    config_id: str | None = None
    dataset_id: str | None = None
    sweep_id: str | None = None
    sweep_hash: str | None = None
    model_id: str | None = None
    model_hash: str | None = None
    stage: str = STAGE_TRAIN_VALIDATION
    schema_version: str = RESULTS_SCHEMA_VERSION

    def base(self, *, sample_id: str, split: str, record_type: str, uri: str | None) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "uri": uri,
            "split": split,
            "stage": self.stage,
            "record_type": record_type,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "config_hash": self.config_hash,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "sweep_id": self.sweep_id,
            "sweep_hash": self.sweep_hash,
            "schema_version": self.schema_version,
        }


def sample_metadata_record(
    prov: Provenance,
    *,
    sample_id: str,
    split: str,
    uri: str | None = None,
    native_width_px: int | None = None,
    native_height_px: int | None = None,
    resize_width_px: int | None = None,
    resize_height_px: int | None = None,
    aspect_bucket: str | None = None,
    source_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One static ``sample_metadata`` row per evaluated sample."""

    row = prov.base(
        sample_id=sample_id,
        split=split,
        record_type=RECORD_TYPE_SAMPLE_METADATA,
        uri=uri,
    )
    row.update(
        {
            "native_width_px": native_width_px,
            "native_height_px": native_height_px,
            "resize_width_px": resize_width_px,
            "resize_height_px": resize_height_px,
            "aspect_bucket": aspect_bucket,
            "source_extra_json": json.dumps(source_extra, sort_keys=True)
            if source_extra is not None
            else None,
        }
    )
    return row


def classification_output_record(
    prov: Provenance,
    *,
    sample_id: str,
    split: str,
    head_name: str,
    prediction_index: int,
    prediction_label: str | None,
    prediction_confidence: float,
    logits: Sequence[float],
    probabilities: Sequence[float],
    head_hash: str | None = None,
    target_index: int | None = None,
    target_name: str | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    checkpoint_hash: str | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    """One ``classification_output`` row per sample per head."""

    row = prov.base(
        sample_id=sample_id,
        split=split,
        record_type=RECORD_TYPE_CLASSIFICATION_OUTPUT,
        uri=uri,
    )
    row.update(
        {
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint_hash": checkpoint_hash,
            "head_name": head_name,
            "head_hash": head_hash,
            "target_index": target_index,
            "target_name": target_name,
            "prediction_index": prediction_index,
            "prediction_label": prediction_label,
            "prediction_confidence": prediction_confidence,
            "logits": list(logits),
            "probabilities": list(probabilities),
        }
    )
    return row


def embedding_record(
    prov: Provenance,
    *,
    sample_id: str,
    split: str,
    embedding_kind: str,
    embedding: Sequence[float],
    epoch: int | None = None,
    global_step: int | None = None,
    checkpoint_hash: str | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    """One embedding row per sample and embedding kind."""

    row = prov.base(
        sample_id=sample_id,
        split=split,
        record_type=RECORD_TYPE_EMBEDDING,
        uri=uri,
    )
    row.update(
        {
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint_hash": checkpoint_hash,
            "embedding_kind": embedding_kind,
            "embedding": list(embedding),
            "embedding_dim": len(embedding),
        }
    )
    return row
