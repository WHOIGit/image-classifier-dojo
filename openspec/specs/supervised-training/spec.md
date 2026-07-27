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
standalone interactive HTML figures under `training_outputs.figures.dir`,
implemented as embedded precomputed JSON plus Plotly.js from the CDN and a
small shared client-side control layer (no Python Plotly dependency). Line
plots read `metrics/metrics.csv`; the confusion matrix, per-class metrics, and
misclassification explorer read canonical `classification_output` rows. All
figures SHALL be computable purely from `metrics/metrics.csv`,
`classification_output` rows, and `_metadata.json`.

The figure set SHALL be:

- `loss_curves.html` — training and validation loss, with a control below the
  title switching between raw loss and loss normalized to the first epoch, the
  legend inside the plot area (top-right), and the y-axis floored at `0.0`.
- `val_f1_curves.html` — validation F1 lines, with the y-axis maximum pinned at
  `1.0` and the minimum auto-scaled.
- `confusion_matrix.html` — a square heatmap with blank, non-hoverable zero
  cells; controls below the title switch between counts and row-normalized
  values and symmetrically reorder rows and columns by class-list or per-class
  metric order. Hovering a non-empty cell shows actual/predicted labels, row and
  column totals, count, and normalized value.
- `per_class_metrics.html` — horizontal per-class bars with independent metric
  and order-by selectors below the title.
- `misclassification_explorer.html` — stacked horizontal FP/FN bars for a
  selected class, with class and order selectors below the title. FP + FN order
  sorts the displayed classes and y-axis largest-first, and the figure height
  adapts to the displayed rows.

For single-head models the figures SHALL be written directly under
`training_outputs.figures.dir` with no per-head subdirectory, and the averaged
loss and F1 figures SHALL be omitted. For multi-head models each head's figures
SHALL be written under a per-head subdirectory and top-level averaged loss and
F1 line figures SHALL also be written; those averaged line figures SHALL offer
an optional, default-off min–max envelope band spanning the per-head minimum
and maximum at each epoch.

The system SHALL also expose a command that regenerates this same figure set
from a completed run's top-level directory — its persisted
`metrics/metrics.csv`, `classification_output` rows, `_metadata.json`, and
`config/resolved.yaml` — without retraining, producing figures identical to the
training-time output for the same inputs. The command SHALL overwrite the run's
`figures/` in place.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled: false`
- **THEN** no figure files are written during training

#### Scenario: Loss curve normalization toggle
- **WHEN** a user opens `loss_curves.html` and activates the normalization
  control below the title
- **THEN** the same document switches between raw loss and first-epoch
  normalized loss without loading another file, with the y-axis floored at
  `0.0`

#### Scenario: Confusion matrix normalize and reorder
- **WHEN** a user changes the row-normalization or ordering control in
  `confusion_matrix.html`
- **THEN** the cell values and the same row/column permutation update, while
  hover continues to report actual/predicted labels, row and column totals,
  count, and normalized value

#### Scenario: Per-class metric and order-by are independent
- **WHEN** a user changes the metric selector and the order-by selector in
  `per_class_metrics.html`
- **THEN** the displayed metric and row ordering update independently, and the
  automatic order follows the selected metric

#### Scenario: Misclassification explorer by class
- **WHEN** a user changes the class or FP + FN ordering selector in
  `misclassification_explorer.html`
- **THEN** the figure updates its FP/FN bars, y-axis classes, and height for
  that selection, with the largest FP + FN total at the top

#### Scenario: Single-head layout
- **WHEN** a run has exactly one classification head
- **THEN** figures are written directly under `training_outputs.figures.dir`
  with no per-head subdirectory and no averaged loss/F1 figures

#### Scenario: Multi-head averaged envelope
- **WHEN** a run has more than one head and a user enables the envelope control
  on a top-level averaged loss or F1 figure
- **THEN** a min–max band spanning the per-head minimum and maximum at each
  epoch is drawn around the average line, and it is hidden by default

#### Scenario: Regenerate figures from a completed run
- **WHEN** `dojo render run <run_dir>` is invoked against a finished run
  directory that contains `metrics/metrics.csv`, `classification_output` result
  rows, and `config/resolved.yaml`
- **THEN** the full HTML figure set is rewritten into `<run_dir>/figures/` from
  those artifacts without retraining, matching the training-time output for the
  same inputs

#### Scenario: Back up existing figures before overwrite
- **WHEN** `dojo render run <run_dir> --backup` is invoked and `figures/`
  already exists
- **THEN** the existing `figures/` is copied to the next numeric-incremented
  sibling directory before the fresh figure set overwrites `figures/`

### Requirement: Streaming result writing
Training result scoring and inference/eval output writing SHALL flush
records per batch rather than accumulating a full split in memory.

#### Scenario: Large validation split
- **WHEN** validation scoring runs over a large split with logits and
  probabilities enabled
- **THEN** rows are flushed per batch
