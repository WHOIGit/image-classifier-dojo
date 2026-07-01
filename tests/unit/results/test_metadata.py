import json

import pytest
from pydantic import ValidationError

from dojo.results import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
    ResultsMetadata,
    SampleMetadataMeta,
    write_metadata_json,
)
from dojo.results.metadata import JsonColumnMeta, RecordTypesMeta


def _metadata() -> ResultsMetadata:
    return ResultsMetadata(
        run_id="ifcb-green-river",
        record_types=RecordTypesMeta(
            sample_metadata=SampleMetadataMeta(
                description="One row per evaluated sample.",
                source_extra_json=JsonColumnMeta(columns=["classname", "original_label"]),
            ),
            classification_output=ClassificationOutputMeta(
                heads={
                    "species": ClassificationHeadMeta(
                        target="species",
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
    # P1 leaves compatibility null (extractors are P2.5).
    assert reloaded.compatibility is None
    head = reloaded.record_types.classification_output.heads["species"]
    assert head.classes == ["A", "B", "C"]


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        ResultsMetadata.model_validate(
            {"run_id": "r", "record_types": {}, "bogus": 1}
        )
