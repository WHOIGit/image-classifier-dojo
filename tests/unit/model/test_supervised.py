"""Supervised model composition end to end (backbone + head)."""

from __future__ import annotations

import torch

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.model import build_supervised_model
from tests.fixtures.configs import TOY_NUM_CLASSES, toy_config_dict, toy_root_config


def test_forward_returns_logits_per_head():
    cfg = toy_root_config()
    model = build_supervised_model(cfg.model, freeze_cfg=cfg.training.freeze)

    assert model.embedding_dim == 1280
    assert model.model_input_order == ["image"]

    logits = model(torch.randn(2, 3, 32, 32))
    assert set(logits) == {"species"}
    assert logits["species"].shape == (2, TOY_NUM_CLASSES)


def test_forward_features_is_the_backbone_embedding():
    cfg = toy_root_config()
    model = build_supervised_model(cfg.model, freeze_cfg=cfg.training.freeze)
    emb = model.forward_features(torch.randn(3, 3, 32, 32))
    assert emb.shape == (3, 1280)


def test_linear_embedding_adapter_changes_head_input_dim():
    raw = toy_config_dict()
    raw["model"]["embedding_adapter"] = {
        "enabled": True,
        "type": "linear",
        "output_dim": 64,
    }
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    model = build_supervised_model(cfg.model, freeze_cfg=cfg.training.freeze)

    assert model.embedding_dim == 64
    assert model.forward_features(torch.randn(2, 3, 32, 32)).shape == (2, 64)
    assert model(torch.randn(2, 3, 32, 32))["species"].shape == (2, TOY_NUM_CLASSES)


def test_mlp_embedding_adapter_changes_head_input_dim():
    raw = toy_config_dict()
    raw["model"]["embedding_adapter"] = {
        "enabled": True,
        "type": "mlp",
        "hidden_dims": [128],
        "output_dim": 64,
        "activation": "relu",
    }
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(raw)).config

    model = build_supervised_model(cfg.model, freeze_cfg=cfg.training.freeze)

    assert model.embedding_dim == 64
    assert model.forward_features(torch.randn(2, 3, 32, 32)).shape == (2, 64)


def test_integrates_with_dataset_batch():
    # A real fixture batch flows through the model and yields class logits.
    from dojo.data import build_dataloader, build_datasets

    cfg = toy_root_config(canvas=(32, 32), batch_size=4)
    bundle = build_datasets(cfg)
    model = build_supervised_model(cfg.model, freeze_cfg=cfg.training.freeze)
    model.eval()

    batch = next(iter(build_dataloader(bundle.datasets["val"], batch_size=4)))
    with torch.no_grad():
        logits = model(batch["image"])
    assert logits["species"].shape == (4, TOY_NUM_CLASSES)
