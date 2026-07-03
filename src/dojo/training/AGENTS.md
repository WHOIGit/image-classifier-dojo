# src/dojo/training — task, trainer, run orchestration

## Purpose

The P1 supervised training layer built on Lightning.

- `run.py` — `execute_train`: takes a resolved `RootConfig`, wires data → model
  → fit → hash best checkpoint → write canonical results. The one place every
  boundary meets.
- `task.py` — the supervised `LightningModule` (forward, per-objective loss,
  weighted total, metric updates).
- `trainer.py` — Trainer/callback/logger construction (best-k + last checkpoint,
  optional early stopping, `local` CSV metrics sink).
- `losses.py`, `metrics.py`, `checkpoint.py` — objective losses, metrics,
  checkpoint hashing helpers.

## Ownership

Owns training orchestration and objective→loss/metric binding. Does not own
dataset path logic, artifact layout, or result serialization (data/storage/
results own those).

## Local Contracts

- `execute_train` consumes an already-resolved config; it does not compose or
  validate.
- Checkpoint callback keys on `checkpointing.monitor`.
- The best-checkpoint hash feeds result provenance — keep it deterministic.

## Verification

- `tests/unit/training/`; `tests/integration/test_training_fit.py` (expensive).
</content>
