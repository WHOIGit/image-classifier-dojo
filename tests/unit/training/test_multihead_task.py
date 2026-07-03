"""Multi-head task routing uses each head's logical target."""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.training.task import SupervisedTaskModule
from tests.fixtures.configs import toy_config_dict


class _DummyHead(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes


class _DummyMultiHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.heads = nn.ModuleDict(
            {
                "species": _DummyHead(6),
                "coarse": _DummyHead(2),
            }
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = image.shape[0]
        return {
            "species": self.anchor + torch.zeros(batch, 6, device=image.device),
            "coarse": self.anchor + torch.zeros(batch, 2, device=image.device),
        }


def _multihead_cfg() -> RootConfig:
    raw = toy_config_dict(canvas=(16, 16), batch_size=2)
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
    return resolve_runtime_and_paths(RootConfig.model_validate(raw)).config


def test_training_step_routes_each_objective_to_its_target(monkeypatch):
    monkeypatch.setattr(
        "dojo.training.task.build_supervised_model",
        lambda model_config, *, freeze_cfg: _DummyMultiHeadModel(),
    )
    cfg = _multihead_cfg()
    module = SupervisedTaskModule(
        model_config=cfg.model,
        training_config=cfg.training,
        objectives_config=cfg.objectives,
        optimizer_config=cfg.optimizer,
        class_counts_by_head={
            "species": {0: 1, 5: 1},
            "coarse": {0: 1, 1: 1},
        },
    )
    batch = {
        "image": torch.zeros(2, 3, 16, 16),
        "target": torch.tensor([5, 5]),
        "targets": {
            "species": torch.tensor([5, 5]),
            "coarse": torch.tensor([0, 1]),
        },
        "sample_id": ["a", "b"],
        "uri": [None, None],
        "split": ["train", "train"],
        "native_width_px": [16, 16],
        "native_height_px": [16, 16],
        "resize_width_px": [16, 16],
        "resize_height_px": [16, 16],
        "aspect_bucket": [None, None],
        "source_extra": [None, None],
    }

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
