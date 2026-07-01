"""Model composition: backbones, heads, and the supervised model."""

from dojo.model.backbone import (
    Backbone,
    TorchvisionBackbone,
    apply_freeze_policy,
    build_backbone,
)
from dojo.model.heads import (
    MulticlassClassificationHead,
    build_head,
    build_head_network,
    build_heads,
)
from dojo.model.supervised import SupervisedModel, build_supervised_model

__all__ = [
    "Backbone",
    "TorchvisionBackbone",
    "build_backbone",
    "apply_freeze_policy",
    "MulticlassClassificationHead",
    "build_head",
    "build_heads",
    "build_head_network",
    "SupervisedModel",
    "build_supervised_model",
]
