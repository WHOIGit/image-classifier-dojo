"""Backbone builder for supervised image models.

A :class:`Backbone` exposes ``output_dim`` and ``forward_features(x) -> Tensor``,
returning a ``batch_size x embedding_dim`` tensor. P1 supports the ``torchvision``
The torchvision/timm classification head is replaced with ``nn.Identity`` so
the wrapped model emits the pooled feature embedding; Dojo heads (`heads.py`)
own the task-specific projection.
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
    convnext, …) and the ``heads``-as-``Sequential`` family used by the
    torchvision vision transformers (``vit_b_16`` …), whose ``forward`` then
    returns the pooled class-token embedding.
    """

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        in_features = classifier[-1].in_features
        # Replace only the final Linear, preserving any Flatten/norm before it
        # (e.g. convnext's classifier is LayerNorm2d → Flatten → Linear; stripping
        # the whole Sequential would remove the Flatten and produce (N, D, 1, 1)).
        classifier[-1] = nn.Identity()
        return in_features
    heads = getattr(model, "heads", None)
    if isinstance(heads, nn.Sequential) and isinstance(heads[-1], nn.Linear):
        in_features = heads[-1].in_features
        heads[-1] = nn.Identity()
        return in_features
    raise ValueError(
        f"unsupported torchvision head layout on {type(model).__name__}; "
        "supported: the classifier-sequential family (e.g. efficientnet_b0) "
        "and the heads-sequential vision transformers (e.g. vit_b_16)"
    )


def _strip_timm_classifier(model: nn.Module) -> int:
    if not hasattr(model, "num_features"):
        raise ValueError("timm model does not expose num_features")
    output_dim = int(model.num_features)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
        return output_dim
    raise ValueError("timm model does not support reset_classifier")


def _load_checkpoint_initialization(
    backbone: Backbone,
    cfg: BackboneWeightsConfig,
) -> None:
    if cfg.uri is None:
        raise ValueError("checkpoint weights require weights.uri")
    checkpoint = torch.load(cfg.uri, map_location="cpu", weights_only=False)
    state = checkpoint
    if cfg.key is not None:
        for part in cfg.key.split("."):
            state = state[part]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint weights did not resolve to a state_dict mapping")

    candidates = []
    for key, value in state.items():
        if key.startswith("model.backbone."):
            candidates.append((key.removeprefix("model.backbone."), value))
        elif key.startswith("backbone."):
            candidates.append((key.removeprefix("backbone."), value))
        elif key.startswith("model."):
            candidates.append((key.removeprefix("model."), value))
        else:
            candidates.append((key, value))
    backbone.load_state_dict(dict(candidates), strict=cfg.strict)


def _build_timm_backbone(cfg: BackboneConfig) -> Backbone:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "model.image_input.backbone.architecture.source='timm' requires "
            "the image_classifier_dojo[timm] extra"
        ) from exc

    arch = cfg.architecture
    pretrained = cfg.weights.source == "library"
    model = timm.create_model(
        arch.name,
        pretrained=pretrained,
        in_chans=arch.input_channels,
    )
    embedding_dim = _strip_timm_classifier(model)
    if arch.output_dim != "auto" and arch.output_dim != embedding_dim:
        raise ValueError(
            f"backbone.architecture.output_dim {arch.output_dim} does not match the "
            f"{arch.name!r} embedding width {embedding_dim}; use 'auto'"
        )
    backbone = TorchvisionBackbone(model, embedding_dim)
    if cfg.weights.source == "checkpoint":
        _load_checkpoint_initialization(backbone, cfg.weights)
    return backbone


def build_backbone(cfg: BackboneConfig) -> Backbone:
    """Build a :class:`Backbone` from a resolved backbone config."""

    arch: BackboneArchitectureConfig = cfg.architecture
    if arch.source == "timm":
        return _build_timm_backbone(cfg)
    if arch.source != "torchvision":
        raise ValueError(f"unsupported backbone source: {arch.source!r}")
    if arch.input_channels != 3:
        raise ValueError(
            "torchvision backbones require input_channels=3 "
            "(use image_mode grayscale_repeat3 for single-channel sources)"
        )

    weights = None if cfg.weights.source == "checkpoint" else _resolve_torchvision_weights(arch.name, cfg.weights)
    model = tv.models.get_model(arch.name, weights=weights)
    embedding_dim = _strip_classifier(model)

    if arch.output_dim != "auto" and arch.output_dim != embedding_dim:
        raise ValueError(
            f"backbone.architecture.output_dim {arch.output_dim} does not match the "
            f"{arch.name!r} embedding width {embedding_dim}; use 'auto'"
        )

    backbone = TorchvisionBackbone(model, embedding_dim)
    if cfg.weights.source == "checkpoint":
        _load_checkpoint_initialization(backbone, cfg.weights)
    return backbone


def apply_freeze_policy(backbone: Backbone, cfg: BackboneFreezeConfig) -> None:
    """Apply the backbone freeze policy."""

    if cfg.policy == "none":
        return
    if cfg.policy == "frozen":
        for param in backbone.parameters():
            param.requires_grad = False
        return
    raise NotImplementedError(f"freeze policy {cfg.policy!r} is not implemented")
