"""Class-label resolution for data targets: label_index_column / label_name_column.

Covers the three ways a target names its classes against the committed
plankton-toyset fixture, whose rows carry a contiguous ``label`` index and a
readable ``classname`` (e.g. label 0 -> "bubble").
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from dojo.config_schemas import RootConfig
from dojo.config_loader import resolve_runtime_and_paths
from dojo.data import build_datasets
from tests.fixtures.configs import toy_config_dict
from tests.fixtures.synthetic import write_manifest_images_dataset

# Distinct classnames in the toy fixture, in ASCII-sorted order (upper before lower).
TOY_NAMES_ALPHABETICAL = [
    "Copepod_nauplii",
    "Dinophysis_acuminata",
    "Skeletonema",
    "Trichodesmium",
    "bubble",
    "fiber",
]


def _bundle(target: dict):
    cfg_dict = toy_config_dict(canvas=(32, 32))
    cfg_dict["data"]["targets"]["species"] = {
        "type": "multiclass_classification",
        "missing_policy": "error",
        **target,
    }
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(cfg_dict)).config
    return build_datasets(cfg)


def test_index_and_name_columns_pair_into_class_mapping():
    bundle = _bundle({"label_index_column": "label", "label_name_column": "classname"})
    # Names come from the data, keyed by the fixture's own contiguous indices.
    assert bundle.class_mapping[0] == "bubble"
    assert set(bundle.class_mapping.values()) == set(TOY_NAMES_ALPHABETICAL)
    assert sorted(bundle.class_mapping) == list(range(6))


def test_index_only_uses_fixture_schema_metadata_when_available():
    bundle = _bundle({"label_index_column": "label"})
    assert bundle.class_mapping
    assert bundle.class_mapping[0] == "Acanthoica_quattrospina"


def test_index_only_without_schema_metadata_leaves_names_unresolved(tmp_path):
    manifest = write_manifest_images_dataset(
        tmp_path / "csv-labels",
        labels=[0, 1, 0],
        backend="csv_manifest",
    )
    raw = toy_config_dict(canvas=(16, 16))
    raw["data"] = {
        "backend": "csv_manifest",
        "manifest_uri": str(manifest),
        "split_column": "split",
        "sample_id_column": "sample_id",
        "image_uri_column": "image_uri",
        "targets": {
            "species": {
                "label_index_column": "label",
                "type": "multiclass_classification",
                "missing_policy": "error",
            }
        },
    }
    raw["model"]["heads"]["species"]["num_classes"] = 2
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    bundle = build_datasets(cfg)

    assert bundle.class_mapping == {}


def test_name_only_assigns_indices_alphabetically():
    bundle = _bundle({"label_name_column": "classname"})
    assert [bundle.class_mapping[i] for i in range(6)] == TOY_NAMES_ALPHABETICAL
    # The dataset must still yield the matching integer targets for those names.
    train = bundle.datasets["train"]
    class_index_by_name = {name: index for index, name in bundle.class_mapping.items()}
    sample = train[0]
    assert 0 <= sample["target"] < 6
    assert sample["target"] in class_index_by_name.values()


def test_index_only_can_read_class_mapping_from_schema_metadata(tmp_path):
    manifest = write_manifest_images_dataset(
        tmp_path / "metadata-labels",
        labels=[0, 1, 0],
        backend="parquet_manifest",
    )
    table = pq.read_table(manifest)
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "label": {
                            "_type": "ClassLabel",
                            "names": ["bubble", "fiber"],
                        }
                    }
                }
            }
        ).encode("utf-8")
    }
    pq.write_table(table.replace_schema_metadata(metadata), manifest)

    raw = toy_config_dict(canvas=(16, 16))
    raw["data"] = {
        "backend": "parquet_manifest",
        "manifest_uri": str(manifest),
        "split_column": "split",
        "sample_id_column": "sample_id",
        "image_uri_column": "image_uri",
        "targets": {
            "species": {
                "label_index_column": "label",
                "type": "multiclass_classification",
                "missing_policy": "error",
            }
        },
    }
    raw["model"]["heads"]["species"]["num_classes"] = 2
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    bundle = build_datasets(cfg)

    assert bundle.class_mapping == {0: "bubble", 1: "fiber"}


def test_index_and_name_columns_are_supplemented_by_schema_metadata(tmp_path):
    manifest = write_manifest_images_dataset(
        tmp_path / "metadata-plus-observed-labels",
        labels=[1, 1, 1],
        backend="parquet_manifest",
    )
    table = pq.read_table(manifest)
    field = table.schema.field("label").with_metadata(
        {b"dojo:class_names": json.dumps(["zero", "one", "two"]).encode("utf-8")}
    )
    table = table.set_column(table.column_names.index("label"), field, table.column("label"))
    table = table.set_column(
        table.column_names.index("classname"),
        "classname",
        table.column("classname").cast("string"),
    )
    pq.write_table(table, manifest)

    raw = toy_config_dict(canvas=(16, 16))
    raw["data"] = {
        "backend": "parquet_manifest",
        "manifest_uri": str(manifest),
        "split_column": "split",
        "sample_id_column": "sample_id",
        "image_uri_column": "image_uri",
        "targets": {
            "species": {
                "label_index_column": "label",
                "label_name_column": "classname",
                "type": "multiclass_classification",
                "missing_policy": "error",
            }
        },
    }
    raw["model"]["heads"]["species"]["num_classes"] = 3
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    bundle = build_datasets(cfg)

    assert bundle.class_mapping == {0: "zero", 1: "class-1", 2: "two"}


def test_target_requires_a_class_source():
    with pytest.raises(ValidationError, match="label_index_column"):
        RootConfig.model_validate(
            {
                **toy_config_dict(canvas=(32, 32)),
                "data": {
                    **toy_config_dict(canvas=(32, 32))["data"],
                    "targets": {
                        "species": {
                            "type": "multiclass_classification",
                            "missing_policy": "error",
                        }
                    },
                },
            }
        )
