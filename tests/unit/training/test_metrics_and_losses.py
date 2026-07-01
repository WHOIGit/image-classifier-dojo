"""Metric registry + loss builder for the training layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import ObjectiveConfig
from dojo.training.losses import build_loss
from dojo.training.metrics import build_metric_modules, iter_metric_logs


def test_build_loss_cross_entropy_string_and_block():
    assert isinstance(build_loss(ObjectiveConfig(head="species")), nn.CrossEntropyLoss)
    block = ObjectiveConfig(
        head="species",
        loss={"type": "cross_entropy", "params": {"label_smoothing": 0.1}},
    )
    loss = build_loss(block)
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.label_smoothing == 0.1


def test_build_metric_modules_and_scalar_logs():
    metrics = build_metric_modules(["accuracy", "f1_macro"], num_classes=3)
    assert set(metrics) == {"accuracy", "f1_macro"}

    logits = torch.tensor([[2.0, 0.1, 0.0], [0.0, 3.0, 0.1], [0.1, 0.0, 1.0]])
    target = torch.tensor([0, 1, 2])
    metrics["accuracy"].update(logits, target)
    acc = metrics["accuracy"].compute()
    logs = list(iter_metric_logs("accuracy", acc))
    assert logs == [("accuracy", acc)]
    assert acc.item() == 1.0


def test_f1_per_class_expands_to_labeled_scalars():
    metric = build_metric_modules(["f1_per_class"], num_classes=3)["f1_per_class"]
    metric.update(torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 1, 2]))
    value = metric.compute()
    logs = list(iter_metric_logs("f1_per_class", value))
    assert [name for name, _ in logs] == [
        "f1_per_class/0",
        "f1_per_class/1",
        "f1_per_class/2",
    ]
