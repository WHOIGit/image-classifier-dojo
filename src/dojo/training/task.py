"""The supervised LightningModule for the P1 training layer.

Owns forward orchestration, per-objective loss, the weighted total loss, and
metric updates from objective definitions. It does **not** own dataset path logic,
artifact layout, or result serialization — those live in the data, storage, and
results layers and the ``dojo train`` orchestration.

Metrics are logged as ``{stage}/{objective}/{output}`` and losses as
``{stage}/{objective}/loss`` plus the weighted ``{stage}/loss`` total, so a
checkpoint monitor like ``val/species/f1_macro`` or ``val/loss`` resolves.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn

from dojo.config_schemas.root import (
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrainingConfig,
)
from dojo.data.contract import SampleBatch
from dojo.model import build_supervised_model
from dojo.training.losses import build_loss
from dojo.training.metrics import build_metric_modules, iter_metric_logs


@dataclass(frozen=True)
class _Objective:
    name: str
    head_name: str
    weight: float


class SupervisedTaskModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        objectives_config: dict[str, ObjectiveConfig],
        optimizer_config: OptimizerConfig,
        class_counts_by_head: dict[str, dict[int, int]] | None = None,
        inference_contract: dict | None = None,
    ) -> None:
        super().__init__()
        # Pydantic configs are picklable; keep them out of the metrics logger.
        self.save_hyperparameters(logger=False)

        self.model = build_supervised_model(
            model_config, freeze_cfg=training_config.freeze
        )
        self._optimizer_config = optimizer_config
        self._inference_contract = inference_contract

        objectives: list[_Objective] = []
        losses: dict[str, nn.Module] = {}
        train_metrics: dict[str, nn.ModuleDict] = {}
        val_metrics: dict[str, nn.ModuleDict] = {}
        for name, objective in objectives_config.items():
            if not objective.enabled:
                continue
            head_name = objective.head or name
            objectives.append(_Objective(name, head_name, float(objective.weight)))
            num_classes = self.model.heads[head_name].num_classes
            losses[name] = build_loss(
                objective,
                class_counts=(class_counts_by_head or {}).get(head_name),
                num_classes=num_classes,
            )
            train_metrics[name] = build_metric_modules(objective.metrics, num_classes)
            val_metrics[name] = build_metric_modules(objective.metrics, num_classes)

        self._objectives = objectives
        self._losses = nn.ModuleDict(losses)
        # Keyed separately rather than by stage: nn.ModuleDict rejects the key
        # "train" because it collides with Module.train().
        self._train_metrics = nn.ModuleDict(train_metrics)
        self._val_metrics = nn.ModuleDict(val_metrics)

    def _stage_metrics(self, stage: str) -> nn.ModuleDict:
        return self._train_metrics if stage == "train" else self._val_metrics

    def configure_optimizers(self):
        cfg = self._optimizer_config
        return torch.optim.AdamW(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def _step(self, batch: SampleBatch, stage: str) -> torch.Tensor:
        images = batch["image"]
        targets = batch["target"]
        logits = self.model(images)
        batch_size = images.shape[0]

        total = images.new_zeros(())
        for objective in self._objectives:
            head_logits = logits[objective.head_name]
            loss = self._losses[objective.name](head_logits, targets)
            total = total + objective.weight * loss
            self.log(
                f"{stage}/{objective.name}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            for metric in self._stage_metrics(stage)[objective.name].values():
                metric.update(head_logits, targets)

        # Epoch-only: the CSV logger emits one row per epoch. prog_bar still
        # updates live during the epoch via the running mean.
        self.log(
            f"{stage}/loss",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return total

    def training_step(self, batch: SampleBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: SampleBatch, batch_idx: int) -> None:
        self._step(batch, "val")

    def _log_epoch_metrics(self, stage: str) -> None:
        for objective in self._objectives:
            metrics = self._stage_metrics(stage)[objective.name]
            for metric_name, metric in metrics.items():
                value = metric.compute()
                for output_name, scalar in iter_metric_logs(metric_name, value):
                    self.log(f"{stage}/{objective.name}/{output_name}", scalar)
                metric.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._inference_contract is not None:
            checkpoint["dojo_inference_contract"] = self._inference_contract
