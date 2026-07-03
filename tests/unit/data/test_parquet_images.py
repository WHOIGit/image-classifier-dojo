"""parquet_images backend against the committed plankton-toyset fixture."""

from __future__ import annotations

import json

import torch

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.data import build_dataloader, build_datasets
from tests.fixtures.configs import TOY_NUM_CLASSES, toy_config_dict, toy_root_config


def test_splits_decode_and_shape():
    cfg = toy_root_config(canvas=(32, 32))
    bundle = build_datasets(cfg)

    assert set(bundle.datasets) == {"train", "val"}
    assert len(bundle.datasets["train"]) == 49
    assert len(bundle.datasets["val"]) == 13

    sample = bundle.datasets["val"][0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert 0 <= sample["target"] < TOY_NUM_CLASSES
    assert isinstance(sample["sample_id"], str)
    assert sample["split"] == "val"
    assert sample["native_width_px"] > 0 and sample["native_height_px"] > 0
    assert sample["resize_width_px"] == 32 and sample["resize_height_px"] == 32
    assert set(sample["source_extra"]) == {"classname", "original_label"}


def test_parquet_images_dataset_does_not_materialize_image_bytes_column():
    bundle = build_datasets(toy_root_config(canvas=(32, 32)))
    dataset = bundle.datasets["train"]

    assert hasattr(dataset, "_file_paths")
    assert not hasattr(dataset, "_image_bytes")


def test_parquet_images_can_materialize_image_cache(tmp_path):
    raw = toy_config_dict(canvas=(32, 32), output_root=str(tmp_path / "runs"))
    raw["data"]["image_cache"] = {
        "enabled": True,
        "dir": str(tmp_path / "image-cache"),
        "progress": False,
    }
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    bundle = build_datasets(cfg)
    dataset = bundle.datasets["train"]
    sample = dataset[0]

    assert bundle.dataset_content_hash is not None
    assert bundle.dataset_content_hash.startswith("sha256:")
    assert bundle.materialized_image_cache_dir is not None
    assert (bundle.materialized_image_cache_dir / "manifest.json").exists()
    assert dataset._materialized_paths is not None
    assert all(dataset._materialized_paths)
    assert sample["image"].shape == (3, 32, 32)


def test_materialized_image_cache_can_cache_bust_and_clobber(tmp_path):
    raw = toy_config_dict(canvas=(32, 32), output_root=str(tmp_path / "runs"))
    raw["data"]["image_cache"] = {
        "enabled": True,
        "dir": str(tmp_path / "image-cache"),
        "progress": False,
    }
    base_cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config
    base_bundle = build_datasets(base_cfg)

    busted_raw = toy_config_dict(canvas=(32, 32), output_root=str(tmp_path / "runs"))
    busted_raw["data"]["image_cache"] = {
        "enabled": True,
        "dir": str(tmp_path / "image-cache"),
        "progress": False,
        "cache_bust": "trial-1",
    }
    busted_cfg = resolve_runtime_and_paths(RootConfig.model_validate(busted_raw)).config
    busted_bundle = build_datasets(busted_cfg)

    assert base_bundle.dataset_content_hash == busted_bundle.dataset_content_hash
    assert base_bundle.materialized_image_cache_dir != busted_bundle.materialized_image_cache_dir
    assert busted_bundle.materialized_image_cache_dir is not None
    marker = json.loads(
        (busted_bundle.materialized_image_cache_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["cache_bust"] == "trial-1"

    sentinel = busted_bundle.materialized_image_cache_dir / "stale.txt"
    sentinel.write_text("remove me", encoding="utf-8")
    clobber_raw = toy_config_dict(canvas=(32, 32), output_root=str(tmp_path / "runs"))
    clobber_raw["data"]["image_cache"] = {
        "enabled": True,
        "dir": str(tmp_path / "image-cache"),
        "progress": False,
        "cache_bust": "trial-1",
        "clobber": True,
    }
    clobber_cfg = resolve_runtime_and_paths(RootConfig.model_validate(clobber_raw)).config
    clobber_bundle = build_datasets(clobber_cfg)

    assert clobber_bundle.materialized_image_cache_dir == busted_bundle.materialized_image_cache_dir
    assert not sentinel.exists()


def test_class_counts_match_fixture_summary():
    bundle = build_datasets(toy_root_config())
    # Contiguous labels 0..5; per-class train counts from summary.json.
    assert bundle.class_counts["train"] == {0: 19, 1: 13, 2: 8, 3: 5, 4: 3, 5: 1}
    assert bundle.class_counts["val"] == {0: 5, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1}


def test_dataloader_batches_via_shared_collation():
    cfg = toy_root_config(canvas=(32, 32), batch_size=8)
    bundle = build_datasets(cfg)
    loader = build_dataloader(bundle.datasets["val"], batch_size=8)

    batch = next(iter(loader))
    assert batch["image"].shape == (8, 3, 32, 32)
    assert batch["target"].shape == (8,)
    assert batch["target"].dtype == torch.int64
    assert len(batch["sample_id"]) == 8


def test_dataset_hash_is_deterministic_and_uri_size():
    a = build_datasets(toy_root_config())
    b = build_datasets(toy_root_config())
    assert a.dataset_hash == b.dataset_hash
    assert a.dataset_hash.startswith("sha256:")
    assert a.dataset_hash_provenance == "uri_size"
