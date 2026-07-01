"""split_from_filename routing: filename globs -> canonical split names."""

from __future__ import annotations

import pytest

from dojo.config_schemas import RootConfig
from dojo.data import DatasetConfigError, build_datasets
from dojo.config_loader import resolve_runtime_and_paths
from tests.fixtures.configs import toy_config_dict
from tests.fixtures.synthetic import write_parquet_images_file


def _config_for(manifest_dir, *, patterns, num_classes=2):
    raw = toy_config_dict()
    raw["data"] = {
        "backend": "parquet_images",
        "manifest_uri": str(manifest_dir),
        "file_pattern": "*.parquet",
        "split_from_filename": patterns,
        "sample_id_column": "sample_id",
        "images": {"column": "image", "bytes_field": "bytes", "path_field": "path"},
        "source_extra_columns": [],
        "targets": {
            "species": {
                "label_index_column": "label",
                "type": "multiclass_classification",
                "missing_policy": "error",
            }
        },
    }
    raw["model"]["heads"]["species"]["num_classes"] = num_classes
    cfg = RootConfig.model_validate(raw)
    return resolve_runtime_and_paths(cfg).config


def test_validation_file_routes_to_val_split(tmp_path):
    data = tmp_path / "data"
    write_parquet_images_file(data / "train-00000.parquet", labels=[0, 1, 0], start_id=0)
    write_parquet_images_file(data / "validation-00000.parquet", labels=[1, 0], start_id=10)

    cfg = _config_for(
        data,
        patterns={"train": "train-*.parquet", "val": "validation-*.parquet"},
    )
    bundle = build_datasets(cfg)

    # "validation-*" normalizes to the canonical split name "val".
    assert set(bundle.datasets) == {"train", "val"}
    assert len(bundle.datasets["train"]) == 3
    assert len(bundle.datasets["val"]) == 2


def test_unmatched_files_are_excluded(tmp_path):
    data = tmp_path / "data"
    write_parquet_images_file(data / "train-00000.parquet", labels=[0, 1])
    write_parquet_images_file(data / "scratch-00000.parquet", labels=[0], start_id=99)

    cfg = _config_for(data, patterns={"train": "train-*.parquet"})
    bundle = build_datasets(cfg)
    assert set(bundle.datasets) == {"train"}
    assert len(bundle.datasets["train"]) == 2


def test_file_matching_multiple_patterns_is_an_error(tmp_path):
    data = tmp_path / "data"
    write_parquet_images_file(data / "train-00000.parquet", labels=[0, 1])

    cfg = _config_for(
        data,
        patterns={"train": "train-*.parquet", "val": "train-*.parquet"},
    )
    with pytest.raises(DatasetConfigError, match="multiple split_from_filename"):
        build_datasets(cfg)
