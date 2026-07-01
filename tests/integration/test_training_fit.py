"""A short real fit on the plankton-toyset fixture exercises the whole training layer."""

from __future__ import annotations

import pytest

from dojo.data import build_dataloader, build_datasets
from dojo.training import (
    SupervisedTaskModule,
    build_callbacks,
    build_logger,
    build_trainer,
    checkpoint_hash,
)

pytestmark = pytest.mark.expensive


def test_short_fit_produces_checkpoint_and_metrics(tmp_path):
    cfg = build_cfg(tmp_path)
    bundle = build_datasets(cfg)

    module = SupervisedTaskModule(
        model_config=cfg.model,
        training_config=cfg.training,
        objectives_config=cfg.objectives,
        optimizer_config=cfg.optimizer,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    callbacks = build_callbacks(cfg.checkpointing, cfg.training, checkpoint_dir)
    logger = build_logger(tmp_path / "run")
    trainer = build_trainer(
        cfg,
        callbacks=callbacks,
        logger=logger,
        accelerator="cpu",
        precision="32-true",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        module,
        build_dataloader(bundle.datasets["train"], batch_size=8, shuffle=True),
        build_dataloader(bundle.datasets["val"], batch_size=8),
    )

    # The monitored metric was logged, so checkpointing had something to track.
    assert "val/species/f1_macro" in trainer.callback_metrics
    assert "val/loss" in trainer.callback_metrics

    # A best checkpoint and last.ckpt were written, and hash deterministically.
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    assert checkpoints
    assert (checkpoint_dir / "last.ckpt").exists()
    h = checkpoint_hash(checkpoints[0])
    assert h.startswith("sha256:")
    assert checkpoint_hash(checkpoints[0]) == h

    # The local metrics sink wrote a CSV under metrics/.
    assert (tmp_path / "run" / "metrics" / "metrics.csv").exists()


def build_cfg(tmp_path):
    from tests.fixtures.configs import toy_root_config

    return toy_root_config(canvas=(32, 32), max_epochs=1, batch_size=8, output_root=str(tmp_path))
