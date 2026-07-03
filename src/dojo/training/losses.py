"""Objective loss construction for the supervised training layer.

Supports plain ``cross_entropy`` and train-split class-count weighted
``weighted_cross_entropy`` for imbalanced multiclass targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dojo.config_schemas.root import ObjectiveConfig


def _cross_entropy_kwargs(params: dict) -> dict:
    kwargs = {}
    if "label_smoothing" in params:
        kwargs["label_smoothing"] = float(params["label_smoothing"])
    return kwargs


def class_weight_tensor(
    *,
    class_counts: dict[int, int],
    num_classes: int,
    scheme: str = "inverse_frequency",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Return deterministic class weights normalized to mean one over nonzero classes."""

    counts = torch.tensor(
        [float(class_counts.get(index, 0)) for index in range(num_classes)],
        dtype=torch.float32,
    )
    positive = counts > 0
    weights = torch.zeros_like(counts)

    if scheme == "inverse_frequency":
        weights[positive] = 1.0 / counts[positive]
    elif scheme == "effective_number":
        if not 0.0 < beta < 1.0:
            raise ValueError("weighted_cross_entropy params.beta must be in (0, 1)")
        weights[positive] = (1.0 - beta) / (1.0 - torch.pow(beta, counts[positive]))
    else:
        raise ValueError(
            "weighted_cross_entropy params.scheme must be "
            "'inverse_frequency' or 'effective_number'"
        )

    if positive.any():
        weights[positive] = weights[positive] / weights[positive].mean()
    return weights


def build_loss(
    objective: ObjectiveConfig,
    *,
    class_counts: dict[int, int] | None = None,
    num_classes: int | None = None,
) -> nn.Module:
    loss = objective.loss
    loss_type = loss if isinstance(loss, str) else loss.type
    params = {} if isinstance(loss, str) else dict(loss.params)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**_cross_entropy_kwargs(params))

    if loss_type == "weighted_cross_entropy":
        if class_counts is None or num_classes is None:
            raise ValueError(
                "weighted_cross_entropy requires train-split class_counts and num_classes"
            )
        scheme = str(params.get("scheme", "inverse_frequency"))
        beta = float(params.get("beta", 0.9999))
        weight = class_weight_tensor(
            class_counts=class_counts,
            num_classes=num_classes,
            scheme=scheme,
            beta=beta,
        )
        return nn.CrossEntropyLoss(weight=weight, **_cross_entropy_kwargs(params))

    raise NotImplementedError(f"unsupported loss type: {loss_type!r}")
