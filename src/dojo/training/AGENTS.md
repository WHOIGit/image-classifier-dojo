# src/dojo/training — task, trainer, run orchestration

## Purpose

The supervised training layer built on Lightning.

- `run.py` — `execute_train`: takes a resolved `RootConfig`, runs preflight,
  wires data → model → fit → hash best checkpoint → write canonical results /
  metrics / figures. The one place every training boundary meets.
- `task.py` — the supervised `LightningModule` (forward, per-objective target
  routing, loss, weighted total, metric updates, checkpoint inference
  contract).
- `trainer.py` — Trainer/callback/logger construction (best-k + last checkpoint,
  optional early stopping, `local` CSV metrics sink).
- `losses.py`, `metrics.py`, `checkpoint.py`, `figures.py`,
  `inference_contract.py` — objective losses, metrics, checkpoint hashing,
  standalone HTML figures, and checkpoint contract helpers.

## Ownership

Owns training orchestration and objective→loss/metric binding. Does not own
dataset path logic, artifact layout, or result serialization (data/storage/
results own those).

## Local Contracts

- `execute_train` consumes an already-resolved config; it does not compose or
  validate.
- `execute_train` may emit phase messages through an optional
  `status_callback`; it must not write terminal output directly.
- Checkpoint callback keys on `checkpointing.monitor`.
- The best-checkpoint hash feeds result provenance — keep it deterministic.
- Weighted/class-balanced objective behavior must use train-split class counts
  from `data/`, not ad hoc retallies inside the task.
- `losses.py` owns active multiclass losses: `cross_entropy`,
  `weighted_cross_entropy`, and `focal_loss`.
- Weighted/class-balanced DataLoader sampling uses `training.sampler.head`
  when configured; otherwise it selects the only classification head, or the
  most imbalanced classification head in multi-head configs.
- Multi-head objectives route labels by the objective head's configured
  `target`; use `SampleBatch["targets"]` for true multi-target datasets and
  keep `SampleBatch["target"]` only as the primary-target fallback.
- Objectives skip samples whose routed target is the shared missing-target ignore
  index, enabling sparse multihead `mask_objective` datasets.
- `f1_macro` and `f1_micro` use the standard TorchMetrics multiclass F1
  implementations after masked target rows are removed; do not add custom
  absent-class handling unless TorchMetrics semantics change.
- Result scoring and checkpoint inference contracts must use per-head class
  mappings and class counts, not the primary target mapping unless the head
  actually uses that target.
- Result scoring must stream writes batch by batch; do not accumulate a full
  validation split's logits/probabilities in memory.
- Metrics CSV cleanup keeps one merged row per epoch when Lightning emits
  train and validation metrics separately.
- Figure output keeps cumulative train/validation loss, normalized loss, and
  naive mean validation F1 curves at the top level; per-head loss, normalized
  loss, validation F1, confusion matrix, and per-class metrics live under
  `figures/<head_name>/`.

## Verification

- `tests/unit/training/`; `tests/integration/test_training_fit.py` (expensive).
</content>
