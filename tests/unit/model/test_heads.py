"""Head builder: linear / mlp networks for multiclass classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import HeadConfig
from dojo.model import build_head, build_heads
from dojo.model.heads import MulticlassClassificationHead


def _head_cfg(network) -> HeadConfig:
    return HeadConfig.model_validate(
        {
            "type": "multiclass_classification",
            "target": "species",
            "num_classes": 6,
            "network": network,
        }
    )


def test_linear_head_projects_to_num_classes():
    head = build_head(_head_cfg({"type": "linear"}), input_dim=1280)
    assert isinstance(head, MulticlassClassificationHead)
    assert head.num_classes == 6
    assert head.target == "species"
    assert isinstance(head.network, nn.Linear)
    out = head(torch.randn(4, 1280))
    assert out.shape == (4, 6)


def test_mlp_head_has_hidden_layers_and_dropout():
    head = build_head(
        _head_cfg({"type": "mlp", "hidden_dims": [64, 32], "activation": "relu", "dropout": 0.1}),
        input_dim=1280,
    )
    assert isinstance(head.network, nn.Sequential)
    kinds = [type(m) for m in head.network]
    assert nn.Dropout in kinds and nn.ReLU in kinds
    # Final projection is the last linear, to num_classes.
    linears = [m for m in head.network if isinstance(m, nn.Linear)]
    assert linears[-1].out_features == 6
    out = head(torch.randn(4, 1280))
    assert out.shape == (4, 6)


def test_build_heads_keys_by_name():
    heads = build_heads(
        {"species": _head_cfg({"type": "linear"})},
        input_dim=1280,
    )
    assert set(heads) == {"species"}
    assert heads["species"](torch.randn(2, 1280)).shape == (2, 6)
