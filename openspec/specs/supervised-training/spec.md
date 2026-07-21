# Supervised Training Specification

## Purpose

The supervised Lightning task module: losses, metrics, checkpointing,
per-epoch metrics CSV, and training figures. Source design:
`DESIGN-DOC/05-models-training-and-heads.md`,
`DESIGN-DOC/06-results-artifacts-and-metadata.md` (workplan P1, P2.3,
P2.5b).

## Requirements

### Requirement: Supervised training run
`dojo train` SHALL run a supervised LightningModule with AdamW and
best-k checkpointing, producing a resolved run directory with
`config/`, `checkpoints/`, `results/`, and `metrics/`.

#### Scenario: Fixture end-to-end run
- **WHEN** `dojo train` runs on the `parquet_images` fixture
- **THEN** the run directory contains the four artifact subdirectories
  and result Parquet round-trips through `amplify-db-utils`

### Requirement: Objective losses
The strict-schema objective loss types SHALL include `cross_entropy`,
`weighted_cross_entropy`, and `focal_loss`. `weighted_cross_entropy`
derives weights from train-split class counts via `inverse_frequency`
(default) or `effective_number` (configurable `beta`), normalized to
mean one over non-empty classes. `focal_loss` supports `gamma` (default
2.0), `alpha` (scalar, per-class, or count-derived via `scheme`/`beta`),
`reduction`, and `ignore_index`. Label smoothing is available on
`cross_entropy` / `weighted_cross_entropy` via `params.label_smoothing`.

#### Scenario: Per-head class counts for weighted losses
- **WHEN** a weighted loss is configured on a head
- **THEN** class counts are selected for that head's configured target

### Requirement: Per-epoch metrics CSV
Post-fit metrics cleanup SHALL merge Lightning's split train/val epoch
rows into one row per epoch in `metrics/metrics.csv`; logging is
epoch-only (no per-batch step rows).

#### Scenario: Cleaned CSV
- **WHEN** a fit completes
- **THEN** `metrics.csv` has one merged row per epoch

### Requirement: Training figures
When `training_outputs.figures.enabled: true`, the system SHALL write
standalone HTML figures under `training_outputs.figures.dir`:
`loss_curves.html`, `loss_curves_normalized.html`,
`val_f1_curves.html`, `confusion_matrix.html`, and
`per_class_metrics.html`, implemented as embedded JSON plus Plotly.js
from the CDN (no Python Plotly dependency). Line plots read
`metrics/metrics.csv`; confusion matrix and per-class metrics read
canonical `classification_output` rows.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled: false`
- **THEN** no figure files are written

### Requirement: Streaming result writing
Training result scoring and inference/eval output writing SHALL flush
records per batch rather than accumulating a full split in memory.

#### Scenario: Large validation split
- **WHEN** validation scoring runs over a large split with logits and
  probabilities enabled
- **THEN** rows are flushed per batch
