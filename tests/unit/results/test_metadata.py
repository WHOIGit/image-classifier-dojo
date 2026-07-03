import json

import pytest
from pydantic import ValidationError

from dojo.results import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
    CompatibilityMeta,
    ResultsMetadata,
    SampleMetadataMeta,
    write_metadata_json,
)
from dojo.results.metadata import JsonColumnMeta, RecordTypesMeta


def _metadata() -> ResultsMetadata:
    return ResultsMetadata(
        run_id="ifcb-green-river",
        compatibility=CompatibilityMeta(
            target_schema_hash="sha256:target",
            target_schema_source={"version": "1", "heads": {}},
            class_mapping_hash="sha256:class",
            class_mapping_source={"version": "1", "heads": {}},
            model_config_hash="sha256:model",
            model_config_source={"version": "1", "model": {}},
            preprocessing_hash="sha256:pre",
            preprocessing_source={"version": "1", "transforms": {}},
        ),
        record_types=RecordTypesMeta(
            sample_metadata=SampleMetadataMeta(
                description="One row per evaluated sample.",
                source_extra_json=JsonColumnMeta(columns=["classname", "original_label"]),
            ),
            classification_output=ClassificationOutputMeta(
                heads={
                    "species": ClassificationHeadMeta(
                        target="species",
                        head_hash="sha256:head",
                        classes=["A", "B", "C"],
                        class_mapping={"0": "A", "1": "B", "2": "C"},
                    )
                }
            ),
        ),
    )


def test_sidecar_writes_and_validates_against_schema(tmp_path):
    path = write_metadata_json(tmp_path / "results" / "_metadata.json", _metadata())
    raw = json.loads(path.read_text())

    # The sidecar validates against its schema.
    reloaded = ResultsMetadata.model_validate(raw)
    assert reloaded.run_id == "ifcb-green-river"
    assert reloaded.schema_name == "dojo.supervised_results"
    assert reloaded.compatibility.target_schema_hash == "sha256:target"
    head = reloaded.record_types.classification_output.heads["species"]
    assert head.classes == ["A", "B", "C"]
    assert head.head_hash == "sha256:head"


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        ResultsMetadata.model_validate(
            {"run_id": "r", "record_types": {}, "bogus": 1}
        )
