"""Head builder for the P1 model layer.

Heads define output structure. P1 supports one head type,
``multiclass_classification``, with a ``linear`` or ``mlp`` network before the
class projection. The head type fixes the final output width (``num_classes``);
the network only chooses whether hidden layers precede that projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import (
    HeadConfig,
    LinearNetworkConfig,
    MlpNetworkConfig,
    NetworkConfig,
)

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


def build_head_network(
    network: NetworkConfig,
    *,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    """Build the projection from head-input embedding to ``output_dim`` logits."""

    if isinstance(network, LinearNetworkConfig):
        return nn.Linear(input_dim, output_dim)

    if isinstance(network, MlpNetworkConfig):
        activation = _ACTIVATIONS[network.activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for width in network.hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(activation())
            if network.dropout > 0:
                layers.append(nn.Dropout(network.dropout))
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    raise ValueError(f"unsupported head network type: {type(network).__name__}")


class MulticlassClassificationHead(nn.Module):
    """Emits ``(batch, num_classes)`` logits for one target."""

    def __init__(self, cfg: HeadConfig, *, input_dim: int) -> None:
        super().__init__()
        self.head_type = cfg.type
        self.target = cfg.target
        self.num_classes = cfg.num_classes
        self.network = build_head_network(
            cfg.network, input_dim=input_dim, output_dim=cfg.num_classes
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


def build_head(cfg: HeadConfig, *, input_dim: int) -> nn.Module:
    if cfg.type == "multiclass_classification":
        return MulticlassClassificationHead(cfg, input_dim=input_dim)
    raise ValueError(f"P1 supports only multiclass_classification heads; got {cfg.type!r}")


def build_heads(heads: dict[str, HeadConfig], *, input_dim: int) -> nn.ModuleDict:
    """Build all heads keyed by head name."""

    return nn.ModuleDict(
        {name: build_head(cfg, input_dim=input_dim) for name, cfg in heads.items()}
    )
