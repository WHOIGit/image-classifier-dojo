"""CSV / Parquet manifest backends with external image URIs."""

from __future__ import annotations

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.data import build_datasets
from tests.fixtures.configs import toy_config_dict
from tests.fixtures.synthetic import write_manifest_images_dataset


def _manifest_cfg(tmp_path, backend: str) -> RootConfig:
    manifest = write_manifest_images_dataset(
        tmp_path / backend,
        labels=[0, 1, 0],
        backend=backend,
    )
    raw = toy_config_dict(canvas=(16, 16))
    raw["data"] = {
        "backend": backend,
        "manifest_uri": str(manifest),
        "split_column": "split",
        "sample_id_column": "sample_id",
        "image_uri_column": "image_uri",
        "targets": {
            "species": {
                "label_index_column": "label",
                "label_name_column": "classname",
                "type": "multiclass_classification",
            }
        },
    }
    raw["model"]["heads"]["species"]["num_classes"] = 2
    return resolve_runtime_and_paths(RootConfig.model_validate(raw)).config


def test_parquet_manifest_backend_decodes_external_images(tmp_path):
    bundle = build_datasets(_manifest_cfg(tmp_path, "parquet_manifest"))

    assert set(bundle.datasets) == {"train", "val"}
    sample = bundle.datasets["val"][0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["uri"].endswith(".png")
    assert bundle.class_counts["train"] == {0: 1, 1: 1}


def test_csv_manifest_backend_decodes_external_images(tmp_path):
    bundle = build_datasets(_manifest_cfg(tmp_path, "csv_manifest"))

    assert set(bundle.datasets) == {"train", "val"}
    assert bundle.datasets["train"][0]["image"].shape == (3, 16, 16)
    assert bundle.class_mapping == {0: "class-0", 1: "class-1"}
