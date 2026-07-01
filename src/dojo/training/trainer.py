"""Trainer / callback / logger construction for the P1 training layer.

Builds the best-k + last checkpoint callback keyed on ``checkpointing.monitor``,
the optional early-stopping callback, the ``local`` metrics sink (a Lightning
``CSVLogger`` writing under the run's ``metrics/`` directory), and the Lightning
``Trainer``. The ``dojo train`` orchestration (next step) resolves the run-dir
paths and wires these together.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from dojo.config_schemas.root import CheckpointingConfig, RootConfig, TrainingConfig


def build_callbacks(
    checkpointing: CheckpointingConfig,
    training: TrainingConfig,
    checkpoint_dir: str | os.PathLike[str],
) -> list[Callback]:
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=checkpointing.monitor,
            mode=checkpointing.mode,
            save_top_k=checkpointing.save_top_k,
            save_last=checkpointing.save_last,
        )
    ]
    early = training.early_stopping
    if early is not None and early.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=early.monitor or checkpointing.monitor,
                mode=early.mode,
                patience=early.patience,
            )
        )
    return callbacks


def build_logger(run_dir: str | os.PathLike[str]) -> CSVLogger:
    """The ``local`` metrics sink: ``<run_dir>/metrics/metrics.csv``."""

    return CSVLogger(save_dir=str(Path(run_dir)), name="metrics", version="")


def build_trainer(
    cfg: RootConfig,
    *,
    callbacks: list[Callback],
    logger: CSVLogger | bool,
    **overrides,
) -> L.Trainer:
    # fp32 matmul precision/throughput tradeoff on Tensor-Core GPUs (no-op on CPU
    # / non-Tensor-Core hardware). "medium"/"high" use TF32/bf16 to silence
    # Lightning's perf hint; "highest" keeps true fp32.
    torch.set_float32_matmul_precision(cfg.runtime.float32_matmul_precision)

    # Rich progress bar (rich is a base dependency, consistent with the CLI's
    # output). Toggled by runtime.progress_bar; an explicit `enable_progress_bar`
    # override still wins.
    progress_bar = cfg.runtime.progress_bar
    trainer_callbacks = list(callbacks)
    if progress_bar and "enable_progress_bar" not in overrides:
        trainer_callbacks.append(RichProgressBar())

    params: dict = {
        "max_epochs": cfg.training.max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "precision": cfg.runtime.precision,
        "logger": logger,
        # The task logs epoch-only, so this step interval never gates our metrics.
        # It exists only to satisfy Lightning's batches-vs-interval check, which
        # warns on small datasets when the default (50) exceeds the batch count.
        "log_every_n_steps": 1,
        "callbacks": trainer_callbacks,
        "enable_progress_bar": progress_bar,
        "fast_dev_run": cfg.runtime.fast_dev_run,
    }
    params.update(overrides)
    return L.Trainer(**params)
