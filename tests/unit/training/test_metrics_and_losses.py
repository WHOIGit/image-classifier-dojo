"""Metric registry + loss builder for the training layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import ObjectiveConfig
from dojo.training.losses import MulticlassFocalLoss, build_loss
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


def test_build_loss_weighted_cross_entropy_from_class_counts():
    objective = ObjectiveConfig(
        head="species",
        loss={"type": "weighted_cross_entropy", "params": {"scheme": "inverse_frequency"}},
    )

    loss = build_loss(
        objective,
        class_counts={0: 10, 1: 5, 2: 1},
        num_classes=3,
    )

    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.weight is not None
    assert loss.weight[2] > loss.weight[1] > loss.weight[0]
    assert torch.isclose(loss.weight.mean(), torch.tensor(1.0))


def test_weighted_cross_entropy_requires_counts():
    objective = ObjectiveConfig(head="species", loss="weighted_cross_entropy")

    import pytest

    with pytest.raises(ValueError, match="requires train-split class_counts"):
        build_loss(objective)


def test_build_loss_focal_loss_gamma_zero_matches_cross_entropy():
    objective = ObjectiveConfig(
        head="species",
        loss={"type": "focal_loss", "params": {"gamma": 0.0}},
    )
    loss = build_loss(objective, num_classes=3)
    logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    target = torch.tensor([0, 2])

    assert isinstance(loss, MulticlassFocalLoss)
    assert torch.allclose(loss(logits, target), nn.CrossEntropyLoss()(logits, target))


def test_focal_loss_downweights_easy_examples():
    objective = ObjectiveConfig(head="species", loss="focal_loss")
    loss = build_loss(objective, num_classes=2)
    logits = torch.tensor([[8.0, -8.0], [0.1, -0.1]])
    target = torch.tensor([0, 0])
    ce = nn.CrossEntropyLoss(reduction="none")(logits, target)
    focal = loss(logits, target)

    assert isinstance(loss, MulticlassFocalLoss)
    assert focal < ce.mean()


def test_focal_loss_can_use_count_derived_alpha():
    objective = ObjectiveConfig(
        head="species",
        loss={
            "type": "focal_loss",
            "params": {"scheme": "inverse_frequency", "gamma": 2.0},
        },
    )
    loss = build_loss(
        objective,
        class_counts={0: 10, 1: 5, 2: 1},
        num_classes=3,
    )

    assert isinstance(loss, MulticlassFocalLoss)
    assert loss.alpha is not None
    assert loss.alpha[2] > loss.alpha[1] > loss.alpha[0]


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
