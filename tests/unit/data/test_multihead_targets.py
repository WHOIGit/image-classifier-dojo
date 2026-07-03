"""True multi-head / multi-target dataset wiring."""

from __future__ import annotations

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.data import build_dataloader, build_datasets, sample_weights_for_dataset
from dojo.data.inspect import inspect_dataset
from tests.fixtures.configs import toy_config_dict
from tests.fixtures.synthetic import write_manifest_images_dataset


def _multihead_config(tmp_path) -> RootConfig:
    manifest = write_manifest_images_dataset(
        tmp_path / "multihead",
        labels=[0, 1, 2, 3, 4, 5],
        coarse_labels=[0, 0, 1, 1, 1, 1],
        coarse_names=[
            "artifact",
            "artifact",
            "organism",
            "organism",
            "organism",
            "organism",
        ],
    )
    raw = toy_config_dict(canvas=(16, 16), batch_size=3)
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
            },
            "coarse": {
                "label_index_column": "coarse_label",
                "label_name_column": "coarse_name",
                "type": "multiclass_classification",
                "missing_policy": "error",
            },
        },
    }
    raw["model"]["heads"]["coarse"] = {
        "type": "multiclass_classification",
        "target": "coarse",
        "num_classes": 2,
        "network": {"type": "linear"},
    }
    raw["objectives"]["coarse"] = {
        "head": "coarse",
        "loss": "cross_entropy",
        "metrics": ["accuracy"],
        "weight": 1.0,
    }
    return resolve_runtime_and_paths(RootConfig.model_validate(raw)).config


def test_multihead_dataset_emits_targets_by_logical_target(tmp_path):
    cfg = _multihead_config(tmp_path)

    bundle = build_datasets(cfg)

    assert bundle.target_names == ("species", "coarse")
    assert bundle.class_mapping_by_target["coarse"] == {0: "artifact", 1: "organism"}
    assert bundle.class_counts_by_target["train"]["coarse"] == {0: 2, 1: 3}
    assert bundle.class_counts["train"] == bundle.class_counts_by_target["train"]["species"]

    sample = bundle.datasets["train"][0]
    assert set(sample["targets"]) == {"species", "coarse"}
    assert sample["target"] == sample["targets"]["species"]

    batch = next(iter(build_dataloader(bundle.datasets["train"], batch_size=3)))
    assert set(batch["targets"]) == {"species", "coarse"}
    assert batch["targets"]["species"].shape == (3,)
    assert batch["targets"]["coarse"].shape == (3,)

    weights = sample_weights_for_dataset(
        bundle.datasets["train"],
        bundle.class_counts_by_target["train"]["coarse"],
        target_name="coarse",
    )
    assert weights[:2] == [1 / 2, 1 / 2]
    assert weights[2:5] == [1 / 3, 1 / 3, 1 / 3]


def test_multihead_inspect_dataset_reports_each_head_mapping(tmp_path):
    cfg = _multihead_config(tmp_path)

    report = inspect_dataset(cfg).report

    assert set(report["targets"]["train"]) == {"species", "coarse"}
    assert report["aspects"]["class_counts"]["per_head"]["coarse"]["total"] == 5
    assert report["aspects"]["class_mapping"]["per_head"]["coarse"] == {
        "0": "artifact",
        "1": "organism",
    }
