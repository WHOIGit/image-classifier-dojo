"""Build a resolved ``RootConfig`` pointed at the committed ``plankton-toyset`` fixture.

Used by data/model/training tests so they don't depend on Hydra composition or
the packaged experiment. The toy fixture carries a canonical ``split`` column
(train/val), so this uses ``split_column`` mode. The backbone uses
``weights.source: none`` to avoid any network fetch in tests, and a small
letterbox canvas keeps decode/transform fast.
"""

from __future__ import annotations

from typing import Any

from dojo.config_schemas import RootConfig
from dojo.config_loader import resolve_runtime_and_paths

TOY_MANIFEST_URI = "./tests/fixtures/plankton-toyset/data.parquet"
TOY_NUM_CLASSES = 6


def toy_config_dict(
    *,
    canvas: tuple[int, int] = (32, 32),
    max_epochs: int = 1,
    batch_size: int = 8,
    output_root: str = "./runs",
) -> dict[str, Any]:
    return {
        "experiment": {"name": "plankton-toyset"},
        "task": {"type": "supervised"},
        # Keep the test suite quiet/fast: no Rich progress bar under pytest capture.
        "runtime": {"progress_bar": False},
        "data": {
            "backend": "parquet_images",
            "manifest_uri": TOY_MANIFEST_URI,
            "split_column": "split",
            "sample_id_column": "ifcb_roi_pid",
            "images": {"column": "image", "bytes_field": "bytes", "path_field": "path"},
            "source_extra_columns": ["classname", "original_label"],
            "targets": {
                "species": {
                    "label_index_column": "label",
                    "label_name_column": "classname",
                    "type": "multiclass_classification",
                    "missing_policy": "error",
                }
            },
        },
        "transforms": {
            "image_mode": "rgb",
            "input_bit_depth": 8,
            "pipeline": [
                {"name": "letterbox", "canvas_size": list(canvas)},
                {"name": "rotate", "mode": "multiples_of_90", "p": 0.5, "train_only": True},
                {"name": "horizontal_flip", "p": 0.5, "train_only": True},
                {
                    "name": "normalize",
                    "mode": "fixed",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ],
        },
        "model": {
            "image_input": {
                "name": "image",
                "backbone": {
                    "architecture": {
                        "source": "torchvision",
                        "name": "efficientnet_b0",
                        "output_dim": "auto",
                        "input_channels": 3,
                    },
                    "weights": {"source": "none"},
                },
            },
            "heads": {
                "species": {
                    "type": "multiclass_classification",
                    "target": "species",
                    "num_classes": TOY_NUM_CLASSES,
                    "network": {"type": "linear"},
                }
            },
        },
        "objectives": {
            "species": {
                "head": "species",
                "loss": "cross_entropy",
                "metrics": ["accuracy", "f1_macro"],
                "weight": 1.0,
            }
        },
        "training": {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "freeze": {"backbone": {"policy": "none"}},
        },
        "optimizer": {"name": "adamw", "lr": 0.0003, "weight_decay": 0.01},
        "checkpointing": {
            "monitor": "val/species/f1_macro",
            "mode": "max",
            "save_top_k": 1,
            "save_last": True,
        },
        "output_root": output_root,
        "training_outputs": {"dir_template": "{experiment.name}/{runtime.run_id}"},
    }


def toy_root_config(**kwargs: Any) -> RootConfig:
    """Return a resolved toy ``RootConfig`` (``inference_pipeline`` populated)."""

    cfg = RootConfig.model_validate(toy_config_dict(**kwargs))
    return resolve_runtime_and_paths(cfg).config
