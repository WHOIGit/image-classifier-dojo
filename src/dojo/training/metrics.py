"""Canonical multiclass metric registry for the P1 training layer.

Keyed by the canonical metric names in the schema's ``MetricName``. Each entry
binds a TorchMetrics factory + Dojo default params and a logging output name. Aliases
are not accepted. ``f1_per_class`` emits one labeled scalar per class; the rest
emit a single scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torchmetrics as tm


@dataclass(frozen=True)
class MetricSpec:
    factory: Callable[[int], tm.Metric]
    output: str
    per_class: bool = False


def _multiclass(metric_cls, **params) -> Callable[[int], tm.Metric]:
    def make(num_classes: int) -> tm.Metric:
        return metric_cls(task="multiclass", num_classes=num_classes, **params)

    return make


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(_multiclass(tm.Accuracy, top_k=1), "accuracy"),
    "f1_macro": MetricSpec(_multiclass(tm.F1Score, average="macro"), "f1_macro"),
    "f1_micro": MetricSpec(_multiclass(tm.F1Score, average="micro"), "f1_micro"),
    "f1_per_class": MetricSpec(
        _multiclass(tm.F1Score, average=None), "f1_per_class/{label}", per_class=True
    ),
    "precision_macro": MetricSpec(_multiclass(tm.Precision, average="macro"), "precision_macro"),
    "precision_micro": MetricSpec(_multiclass(tm.Precision, average="micro"), "precision_micro"),
    "recall_macro": MetricSpec(_multiclass(tm.Recall, average="macro"), "recall_macro"),
    "recall_micro": MetricSpec(_multiclass(tm.Recall, average="micro"), "recall_micro"),
}


def build_metric_modules(names: list[str], num_classes: int) -> nn.ModuleDict:
    """Instantiate the named metrics for one objective, keyed by canonical name."""

    return nn.ModuleDict(
        {name: METRIC_REGISTRY[name].factory(num_classes) for name in names}
    )


def iter_metric_logs(name: str, value: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(output_name, scalar)`` pairs for a computed metric value."""

    spec = METRIC_REGISTRY[name]
    if spec.per_class:
        for index, scalar in enumerate(value):
            yield spec.output.format(label=index), scalar
    else:
        yield spec.output, value
