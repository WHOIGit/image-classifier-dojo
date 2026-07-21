"""Torchvision backbone builder (no network: weights.source = none)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from dojo.config_schemas.root import BackboneConfig
from dojo.model import build_backbone
from dojo.model.backbone import apply_freeze_policy
from dojo.config_schemas.root import BackboneFreezeConfig

EFFICIENTNET_B0_DIM = 1280


def _backbone_cfg(**arch_overrides) -> BackboneConfig:
    arch = {
        "source": "torchvision",
        "name": "efficientnet_b0",
        "output_dim": "auto",
        "input_channels": 3,
    }
    arch.update(arch_overrides)
    return BackboneConfig.model_validate({"architecture": arch, "weights": {"source": "none"}})


def test_builds_and_exposes_embedding_dim():
    backbone = build_backbone(_backbone_cfg())
    assert backbone.output_dim == EFFICIENTNET_B0_DIM
    out = backbone.forward_features(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, EFFICIENTNET_B0_DIM)


def test_explicit_output_dim_mismatch_rejected():
    with pytest.raises(ValueError, match="does not match"):
        build_backbone(_backbone_cfg(output_dim=999))


def test_builds_torchvision_vit_via_heads_strip():
    # torchvision vision transformers expose `.heads` (not `.classifier`);
    # the builder strips it and returns the 768-dim pooled class token.
    backbone = build_backbone(_backbone_cfg(name="vit_b_16"))
    assert backbone.output_dim == 768
    out = backbone.forward_features(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 768)


def test_non_three_input_channels_rejected():
    with pytest.raises(ValueError, match="input_channels=3"):
        build_backbone(_backbone_cfg(input_channels=1))


def test_invalid_library_weight_name_rejected_without_download():
    cfg = BackboneConfig.model_validate(
        {
            "architecture": {"source": "torchvision", "name": "efficientnet_b0"},
            "weights": {"source": "library", "name": "NOT_A_REAL_WEIGHT"},
        }
    )
    with pytest.raises(ValueError, match="not a valid weight"):
        build_backbone(cfg)


@pytest.mark.skipif(
    importlib.util.find_spec("timm") is None,
    reason="timm optional dependency is not installed",
)
def test_builds_timm_backbone_without_download():
    cfg = BackboneConfig.model_validate(
        {
            "architecture": {
                "source": "timm",
                "name": "efficientnet_b0",
                "output_dim": "auto",
                "input_channels": 3,
            },
            "weights": {"source": "none"},
        }
    )

    backbone = build_backbone(cfg)

    assert backbone.output_dim == EFFICIENTNET_B0_DIM
    out = backbone.forward_features(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, EFFICIENTNET_B0_DIM)


def test_freeze_policy_none_is_noop():
    backbone = build_backbone(_backbone_cfg())
    apply_freeze_policy(backbone, BackboneFreezeConfig(policy="none"))
    assert all(p.requires_grad for p in backbone.parameters())


def test_freeze_policy_frozen_disables_backbone_grads():
    backbone = build_backbone(_backbone_cfg())
    apply_freeze_policy(backbone, BackboneFreezeConfig(policy="frozen"))
    assert not any(p.requires_grad for p in backbone.parameters())
