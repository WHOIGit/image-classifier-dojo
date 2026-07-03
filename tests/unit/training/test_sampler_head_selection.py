"""Weighted sampler head selection policy."""

from __future__ import annotations

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.data.parquet_images import DataBundle
from dojo.training.run import _select_sampler_head
from tests.fixtures.configs import toy_config_dict


def _bundle(*, species_counts: dict[int, int], coarse_counts: dict[int, int]) -> DataBundle:
    return DataBundle(
        datasets={},
        dataset_hash="sha256:test",
        dataset_hash_provenance="test",
        files=[],
        dataset_content_hash=None,
        materialized_image_cache_dir=None,
        class_counts={"train": species_counts},
        class_counts_by_target={
            "train": {
                "species": species_counts,
                "coarse": coarse_counts,
            }
        },
        target_name="species",
        target_names=("species", "coarse"),
        class_mapping={},
        class_mapping_by_target={"species": {}, "coarse": {}},
    )


def _multihead_cfg(*, sampler_head: str | None = None) -> RootConfig:
    raw = toy_config_dict(canvas=(16, 16), batch_size=2)
    raw["model"]["heads"]["species"]["num_classes"] = 2
    raw["data"]["targets"]["coarse"] = {
        "label_index_column": "coarse_label",
        "label_name_column": "coarse_name",
        "type": "multiclass_classification",
        "missing_policy": "error",
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
    raw["training"]["sampler"] = {"type": "class_balanced"}
    if sampler_head is not None:
        raw["training"]["sampler"]["head"] = sampler_head
    return resolve_runtime_and_paths(RootConfig.model_validate(raw)).config


def test_sampler_head_defaults_to_only_head():
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(toy_config_dict())).config
    bundle = _bundle(species_counts={0: 3, 1: 1}, coarse_counts={0: 1, 1: 1})

    assert _select_sampler_head(cfg, bundle) == "species"


def test_sampler_head_can_be_configured_explicitly():
    cfg = _multihead_cfg(sampler_head="coarse")
    bundle = _bundle(species_counts={0: 9, 1: 1}, coarse_counts={0: 6, 1: 4})

    assert _select_sampler_head(cfg, bundle) == "coarse"


def test_sampler_head_defaults_to_most_imbalanced_classification_head():
    cfg = _multihead_cfg()
    bundle = _bundle(species_counts={0: 9, 1: 1}, coarse_counts={0: 6, 1: 4})

    assert _select_sampler_head(cfg, bundle) == "species"
