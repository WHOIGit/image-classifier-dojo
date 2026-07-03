"""Aspect-bucket transform and batch sampler integration."""

from __future__ import annotations

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.data import build_dataloader, build_datasets, sample_weights_for_dataset
from tests.fixtures.configs import toy_config_dict
from tests.fixtures.synthetic import write_parquet_images_file


def test_batch_aspect_buckets_groups_stackable_batches(tmp_path):
    data_dir = tmp_path / "bucketed"
    write_parquet_images_file(
        data_dir / "train-00000.parquet",
        labels=[0, 0, 1, 1],
        sizes=[(40, 10), (30, 10), (10, 40), (10, 30)],
    )
    raw = toy_config_dict(canvas=(16, 16), batch_size=2)
    raw["data"]["manifest_uri"] = str(data_dir)
    raw["data"]["sample_id_column"] = "sample_id"
    raw["data"]["source_extra_columns"] = []
    raw["data"]["split_column"] = None
    raw["data"]["split_from_filename"] = {"train": "train-*.parquet"}
    raw["data"]["targets"]["species"].pop("label_name_column", None)
    raw["model"]["heads"]["species"]["num_classes"] = 2
    raw["transforms"]["pipeline"] = [
        {
            "name": "aspect_bucket",
            "buckets": [
                {"name": "wide", "min_aspect": 1.2, "canvas_size": [16, 32]},
                {"name": "tall", "max_aspect": 0.8, "canvas_size": [32, 16]},
            ],
        }
    ]
    raw["training"]["sampler"] = {"type": "batch_aspect_buckets"}
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config
    bundle = build_datasets(cfg)

    loader = build_dataloader(
        bundle.datasets["train"],
        batch_size=2,
        batch_aspect_buckets=True,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert len(set(batch["aspect_bucket"])) == 1
        assert batch["image"].shape in {(2, 3, 16, 32), (2, 3, 32, 16)}


def test_class_balanced_weights_come_from_train_counts(tmp_path):
    data_dir = tmp_path / "weighted"
    write_parquet_images_file(
        data_dir / "train-00000.parquet",
        labels=[0, 0, 0, 1],
        sizes=[(16, 16)] * 4,
    )
    raw = toy_config_dict(canvas=(16, 16), batch_size=2)
    raw["data"]["manifest_uri"] = str(data_dir)
    raw["data"]["sample_id_column"] = "sample_id"
    raw["data"]["source_extra_columns"] = []
    raw["data"]["split_column"] = None
    raw["data"]["split_from_filename"] = {"train": "train-*.parquet"}
    raw["data"]["targets"]["species"].pop("label_name_column", None)
    raw["model"]["heads"]["species"]["num_classes"] = 2
    raw["training"]["sampler"] = {"type": "class_balanced"}
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config
    bundle = build_datasets(cfg)

    weights = sample_weights_for_dataset(
        bundle.datasets["train"],
        bundle.class_counts["train"],
    )

    assert weights[:3] == [1 / 3, 1 / 3, 1 / 3]
    assert weights[3] == 1.0


def test_class_balanced_sampler_composes_with_aspect_buckets(tmp_path):
    data_dir = tmp_path / "weighted-bucketed"
    write_parquet_images_file(
        data_dir / "train-00000.parquet",
        labels=[0, 0, 1, 1],
        sizes=[(40, 10), (30, 10), (10, 40), (10, 30)],
    )
    raw = toy_config_dict(canvas=(16, 16), batch_size=2)
    raw["data"]["manifest_uri"] = str(data_dir)
    raw["data"]["sample_id_column"] = "sample_id"
    raw["data"]["source_extra_columns"] = []
    raw["data"]["split_column"] = None
    raw["data"]["split_from_filename"] = {"train": "train-*.parquet"}
    raw["data"]["targets"]["species"].pop("label_name_column", None)
    raw["model"]["heads"]["species"]["num_classes"] = 2
    raw["transforms"]["pipeline"] = [
        {
            "name": "aspect_bucket",
            "buckets": [
                {"name": "wide", "min_aspect": 1.2, "canvas_size": [16, 32]},
                {"name": "tall", "max_aspect": 0.8, "canvas_size": [32, 16]},
            ],
        }
    ]
    raw["training"]["sampler"] = {"type": "class_balanced"}
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config
    bundle = build_datasets(cfg)

    loader = build_dataloader(
        bundle.datasets["train"],
        batch_size=2,
        sampler_type=cfg.training.sampler.type,
        class_counts=bundle.class_counts["train"],
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert len(set(batch["aspect_bucket"])) == 1
        assert batch["image"].shape in {(2, 3, 16, 32), (2, 3, 32, 16)}
