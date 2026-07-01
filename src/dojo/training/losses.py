"""Objective loss construction for the P1 training layer.

P1 supports a single classification loss, ``cross_entropy``. The objective's
``loss`` is either the string shorthand or a ``{type, params}`` block; only the
basic ``cross_entropy`` params are honored in the thin slice.
"""

from __future__ import annotations

import torch.nn as nn

from dojo.config_schemas.root import ObjectiveConfig


def build_loss(objective: ObjectiveConfig) -> nn.Module:
    loss = objective.loss
    loss_type = loss if isinstance(loss, str) else loss.type
    if loss_type != "cross_entropy":
        raise NotImplementedError(
            f"P1 supports only cross_entropy loss; got {loss_type!r}"
        )

    params = {} if isinstance(loss, str) else dict(loss.params)
    kwargs = {}
    if "label_smoothing" in params:
        kwargs["label_smoothing"] = float(params["label_smoothing"])
    return nn.CrossEntropyLoss(**kwargs)
