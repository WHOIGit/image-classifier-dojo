"""Backbone builder for the P1 model layer.

A :class:`Backbone` exposes ``output_dim`` and ``forward_features(x) -> Tensor``,
returning a ``batch_size x embedding_dim`` tensor. P1 supports the ``torchvision``
``efficientnet_b0`` architecture with ``library`` / ``none`` weights; ``timm``
and ``checkpoint`` initialization are P2.3.

The torchvision classification head (the final ``classifier`` projection) is
replaced with ``nn.Identity`` so the wrapped model emits the pooled feature
embedding; Dojo heads (`heads.py`) own the task-specific projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision as tv

from dojo.config_schemas.root import (
    BackboneArchitectureConfig,
    BackboneConfig,
    BackboneFreezeConfig,
    BackboneWeightsConfig,
)


class Backbone(nn.Module):
    """Feature extractor exposing a fixed-width embedding."""

    output_dim: int

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class TorchvisionBackbone(Backbone):
    """Wraps a torchvision model with its classifier replaced by identity."""

    def __init__(self, model: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.model = model
        self.output_dim = output_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _resolve_torchvision_weights(name: str, weights_cfg: BackboneWeightsConfig):
    if weights_cfg.source == "none":
        return None
    # source == "library"
    enum = tv.models.get_model_weights(name)
    requested = weights_cfg.name or "DEFAULT"
    if requested == "DEFAULT":
        return enum.DEFAULT
    if requested not in enum.__members__:
        raise ValueError(
            f"weights.name {requested!r} is not a valid weight for {name!r}; "
            f"choose one of {sorted(enum.__members__)} or DEFAULT"
        )
    return getattr(enum, requested)


def _strip_classifier(model: nn.Module) -> int:
    """Replace the model's final linear projection with identity, returning its in_features.

    Covers the ``classifier``-as-``Sequential`` family (efficientnet, mobilenet,
    convnext, …), which is all P1 needs.
    """

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        in_features = classifier[-1].in_features
        model.classifier = nn.Identity()
        return in_features
    raise ValueError(
        f"unsupported torchvision head layout on {type(model).__name__}; "
        "P1 supports the classifier-sequential family (e.g. efficientnet_b0)"
    )


def build_backbone(cfg: BackboneConfig) -> Backbone:
    """Build a :class:`Backbone` from a resolved backbone config."""

    arch: BackboneArchitectureConfig = cfg.architecture
    if arch.source != "torchvision":
        raise ValueError(f"P1 supports only torchvision backbones; got {arch.source!r}")
    if arch.input_channels != 3:
        raise ValueError(
            "P1 torchvision backbones require input_channels=3 "
            "(use image_mode grayscale_repeat3 for single-channel sources)"
        )

    weights = _resolve_torchvision_weights(arch.name, cfg.weights)
    model = tv.models.get_model(arch.name, weights=weights)
    embedding_dim = _strip_classifier(model)

    if arch.output_dim != "auto" and arch.output_dim != embedding_dim:
        raise ValueError(
            f"backbone.architecture.output_dim {arch.output_dim} does not match the "
            f"{arch.name!r} embedding width {embedding_dim}; use 'auto'"
        )

    return TorchvisionBackbone(model, embedding_dim)


def apply_freeze_policy(backbone: Backbone, cfg: BackboneFreezeConfig) -> None:
    """Apply the backbone freeze policy. P1 supports ``none`` (fully trainable)."""

    if cfg.policy == "none":
        return
    raise NotImplementedError(f"freeze policy {cfg.policy!r} is not implemented in P1")
