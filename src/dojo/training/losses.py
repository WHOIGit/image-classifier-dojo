"""Objective loss construction for the supervised training layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MulticlassFocalLoss(nn.Module):
    """Focal loss over multiclass logits and integer class targets."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal_loss params.gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("focal_loss params.reduction must be 'mean', 'sum', or 'none'")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 2:
            classes = logits.shape[1]
            logits = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, classes)
            target = target.reshape(-1)

        valid = target != self.ignore_index
        logits = logits[valid]
        target = target[valid]
        if target.numel() == 0:
            empty = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return empty if self.reduction != "none" else logits.new_zeros((0,))

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce = F.nll_loss(
            log_probs,
            target,
            weight=self.alpha,
            reduction="none",
        )
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _explicit_alpha_tensor(alpha: object, *, num_classes: int) -> torch.Tensor | None:
    if alpha is None:
        return None
    if isinstance(alpha, int | float):
        return torch.full((num_classes,), float(alpha), dtype=torch.float32)
    values = list(alpha)  # type: ignore[arg-type]
    if len(values) != num_classes:
        raise ValueError(
            "focal_loss params.alpha must have length matching head.num_classes"
        )
    return torch.tensor([float(value) for value in values], dtype=torch.float32)


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

    if loss_type == "focal_loss":
        gamma = float(params.get("gamma", 2.0))
        reduction = str(params.get("reduction", "mean"))
        ignore_index = int(params.get("ignore_index", -100))
        alpha = None
        if "alpha" in params:
            if num_classes is None:
                raise ValueError("focal_loss params.alpha requires num_classes")
            alpha = _explicit_alpha_tensor(params["alpha"], num_classes=num_classes)
        elif "scheme" in params:
            if class_counts is None or num_classes is None:
                raise ValueError(
                    "focal_loss params.scheme requires train-split class_counts and num_classes"
                )
            alpha = class_weight_tensor(
                class_counts=class_counts,
                num_classes=num_classes,
                scheme=str(params.get("scheme", "inverse_frequency")),
                beta=float(params.get("beta", 0.9999)),
            )
        return MulticlassFocalLoss(
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            ignore_index=ignore_index,
        )

    raise NotImplementedError(f"unsupported loss type: {loss_type!r}")
