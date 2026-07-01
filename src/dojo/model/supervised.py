"""Supervised model composition for the P1 thin slice.

Composes a single image backbone with one or more classification heads.
The P1 slice has no tabular input and no embedding adapter, so the
head-input embedding is exactly
the backbone embedding; the composition seam is kept so P2.3 can slot tabular
concatenation and the adapter in without changing call sites.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import FreezeConfig, ModelConfig
from dojo.model.backbone import Backbone, apply_freeze_policy, build_backbone
from dojo.model.heads import build_heads


class SupervisedModel(nn.Module):
    """Backbone + per-target heads. ``forward`` returns logits keyed by head name."""

    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: nn.ModuleDict,
        image_input_name: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.image_input_name = image_input_name
        self.model_input_order = [image_input_name]
        self.embedding_dim = backbone.output_dim

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Head-input embedding for a batch of images."""

        return self.backbone.forward_features(image)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.forward_features(image)
        return {name: head(embedding) for name, head in self.heads.items()}


def build_supervised_model(
    model_cfg: ModelConfig,
    *,
    freeze_cfg: FreezeConfig,
) -> SupervisedModel:
    """Build a :class:`SupervisedModel` from resolved model + freeze configs."""

    if model_cfg.tabular_input.enabled:
        raise NotImplementedError("tabular input is not part of the P1 model layer")
    if model_cfg.embedding_adapter.enabled:
        raise NotImplementedError("embedding adapter is not part of the P1 model layer")

    backbone = build_backbone(model_cfg.image_input.backbone)
    apply_freeze_policy(backbone, freeze_cfg.backbone)

    heads = build_heads(model_cfg.heads, input_dim=backbone.output_dim)

    return SupervisedModel(
        backbone=backbone,
        heads=heads,
        image_input_name=model_cfg.image_input.name,
    )
