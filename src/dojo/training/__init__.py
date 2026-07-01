"""Supervised training: LightningModule, losses, metrics, checkpointing, trainer."""

from dojo.training.checkpoint import checkpoint_filename, checkpoint_hash
from dojo.training.losses import build_loss
from dojo.training.metrics import (
    METRIC_REGISTRY,
    build_metric_modules,
    iter_metric_logs,
)
from dojo.training.task import SupervisedTaskModule
from dojo.training.trainer import build_callbacks, build_logger, build_trainer

# Imported last: run.py pulls from the training submodules above, so their names
# must already be bound when this package import reaches it.
from dojo.training.run import TrainResult, execute_train

__all__ = [
    "SupervisedTaskModule",
    "build_loss",
    "METRIC_REGISTRY",
    "build_metric_modules",
    "iter_metric_logs",
    "checkpoint_hash",
    "checkpoint_filename",
    "build_callbacks",
    "build_logger",
    "build_trainer",
    "execute_train",
    "TrainResult",
]
