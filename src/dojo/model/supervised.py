"""Supervised model composition.

Composes a single image backbone with one or more classification heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import (
    EnabledEmbeddingAdapterConfig,
    FreezeConfig,
    ModelConfig,
)
from dojo.model.backbone import Backbone, apply_freeze_policy, build_backbone
from dojo.model.heads import build_heads


class SupervisedModel(nn.Module):
    """Backbone + per-target heads. ``forward`` returns logits keyed by head name."""

    def __init__(
        self,
        *,
        backbone: Backbone,
        embedding_adapter: nn.Module,
        heads: nn.ModuleDict,
        image_input_name: str,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_adapter = embedding_adapter
        self.heads = heads
        self.image_input_name = image_input_name
        self.model_input_order = [image_input_name]
        self.embedding_dim = embedding_dim

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Head-input embedding for a batch of images."""

        return self.embedding_adapter(self.backbone.forward_features(image))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.forward_features(image)
        return {name: head(embedding) for name, head in self.heads.items()}


_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


def _build_embedding_adapter(
    cfg,
    *,
    input_dim: int,
) -> tuple[nn.Module, int]:
    if not cfg.enabled:
        return nn.Identity(), input_dim

    assert isinstance(cfg, EnabledEmbeddingAdapterConfig)
    if cfg.type == "linear":
        return nn.Linear(input_dim, cfg.output_dim), cfg.output_dim

    activation = _ACTIVATIONS[cfg.activation]
    layers: list[nn.Module] = []
    previous = input_dim
    for width in cfg.hidden_dims:
        layers.append(nn.Linear(previous, width))
        layers.append(activation())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        previous = width
    layers.append(nn.Linear(previous, cfg.output_dim))
    return nn.Sequential(*layers), cfg.output_dim


def build_supervised_model(
    model_cfg: ModelConfig,
    *,
    freeze_cfg: FreezeConfig,
) -> SupervisedModel:
    """Build a :class:`SupervisedModel` from resolved model + freeze configs."""

    if model_cfg.tabular_input.enabled:
        raise NotImplementedError("tabular input is not part of the P1 model layer")

    backbone = build_backbone(model_cfg.image_input.backbone)
    apply_freeze_policy(backbone, freeze_cfg.backbone)

    embedding_adapter, embedding_dim = _build_embedding_adapter(
        model_cfg.embedding_adapter,
        input_dim=backbone.output_dim,
    )
    heads = build_heads(model_cfg.heads, input_dim=embedding_dim)

    return SupervisedModel(
        backbone=backbone,
        embedding_adapter=embedding_adapter,
        heads=heads,
        image_input_name=model_cfg.image_input.name,
        embedding_dim=embedding_dim,
    )
