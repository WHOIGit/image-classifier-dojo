"""Round-trip the canonical result writer through amplify-db-utils.

Result Parquet round-trips through ``amplify-db-utils`` and a reader can filter
by ``stage`` / ``record_type``.
"""

from __future__ import annotations

import pytest

from dojo.results import (
    Provenance,
    ResultWriter,
    classification_output_record,
    sample_metadata_record,
)
from dojo.results.schemas import (
    RECORD_TYPE_CLASSIFICATION_OUTPUT,
    RECORD_TYPE_SAMPLE_METADATA,
    STAGE_TRAIN_VALIDATION,
)


def _provenance() -> Provenance:
    return Provenance(
        run_id="ifcb-green-river",
        config_hash="sha256:cfg",
        dataset_hash="sha256:ds",
        config_id="green-river",
    )


def _seed_rows(prov: Provenance) -> list[dict]:
    rows: list[dict] = []
    for i in range(3):
        rows.append(
            sample_metadata_record(
                prov,
                sample_id=f"s{i}",
                split="val",
                native_width_px=80,
                native_height_px=42,
                resize_width_px=224,
                resize_height_px=224,
                source_extra={"classname": "Skeletonema"},
            )
        )
        rows.append(
            classification_output_record(
                prov,
                sample_id=f"s{i}",
                split="val",
                head_name="species",
                prediction_index=i % 2,
                prediction_label=["A", "B"][i % 2],
                prediction_confidence=0.9,
                logits=[0.1, 0.2],
                probabilities=[0.4, 0.6],
                head_hash="sha256:head",
                target_index=i % 2,
                target_name=["A", "B"][i % 2],
                epoch=2,
                global_step=100 + i,
                checkpoint_hash="sha256:ckpt",
            )
        )
    return rows


def test_roundtrip_and_filter_by_record_type(tmp_path):
    prov = _provenance()
    writer = ResultWriter(tmp_path / "results", partition_by=["record_type"])
    writer.write_records(_seed_rows(prov))

    assert writer.count() == 6
    assert writer.count({"record_type": RECORD_TYPE_SAMPLE_METADATA}) == 3
    assert writer.count({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}) == 3

    cls_rows = list(writer.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))
    assert len(cls_rows) == 3
    sample = cls_rows[0]
    # Vector columns round-trip losslessly.
    assert sample["probabilities"] == pytest.approx([0.4, 0.6])
    assert sample["logits"] == pytest.approx([0.1, 0.2])
    assert sample["head_name"] == "species"
    assert sample["head_hash"] == "sha256:head"
    assert sample["target_index"] in {0, 1}
    assert sample["target_name"] in {"A", "B"}
    assert sample["stage"] == STAGE_TRAIN_VALIDATION
    # Provenance carried through.
    assert sample["run_id"] == "ifcb-green-river"
    assert sample["config_hash"] == "sha256:cfg"


def test_filter_by_stage(tmp_path):
    prov = _provenance()
    writer = ResultWriter(tmp_path / "results", partition_by=["record_type"])
    writer.write_records(_seed_rows(prov))

    assert writer.count({"stage": STAGE_TRAIN_VALIDATION}) == 6
    assert writer.count({"stage": "holdout_eval"}) == 0


def test_partition_directories_written_by_record_type(tmp_path):
    prov = _provenance()
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir, partition_by=["record_type"])
    writer.write_records(_seed_rows(prov))

    names = {p.name for p in results_dir.rglob("record_type=*")}
    assert f"record_type={RECORD_TYPE_SAMPLE_METADATA}" in names
    assert f"record_type={RECORD_TYPE_CLASSIFICATION_OUTPUT}" in names
