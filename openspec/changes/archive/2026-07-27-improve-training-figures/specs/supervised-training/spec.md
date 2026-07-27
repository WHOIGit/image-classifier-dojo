## MODIFIED Requirements

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

- `loss_curves.html` — training and validation loss, with a toggle below the
  title switching between raw loss and loss normalized to the first epoch, the
  legend placed inside the plot area (top-right), and the y-axis floored at
  `0.0`.
- `val_f1_curves.html` — validation F1 lines, with the y-axis maximum pinned at
  `1.0` and the minimum auto-scaled.
- `confusion_matrix.html` — a heatmap whose plot area (inside the axes) is
  square; cells with zero count are rendered as background (blank) with no
  hover; a toggle under the title switches between raw counts and
  row-normalized (recall) values; an order selector beside the toggle reorders
  rows and columns by the same permutation (symmetric), offering class-list
  order and each per-class metric (including validation counts); rows are
  Actual and columns are Predicted; hovering a non-empty cell always shows
  Actual, Predicted, Actual count sum, Predicted count sum, the cell Count, and
  the Normalized value regardless of toggle state.
- `per_class_metrics.html` — horizontal per-class bars with two independent
  controls under the title: a metric selector (f1, precision, recall,
  validation counts) and an order-by selector (`auto:<selected-metric>`,
  class-list order, then each metric including counts).
- `misclassification_explorer.html` — a per-class selector that shows, for the
  chosen class, its false positives (classes mistaken as it) and false
  negatives (classes it was mistaken for) as stacked horizontal bars.

For single-head models the figures SHALL be written directly under
`training_outputs.figures.dir` with no per-head subdirectory, and the averaged
loss and F1 figures SHALL be omitted. For multi-head models each head's figures
SHALL be written under a per-head subdirectory and top-level averaged loss and
F1 line figures SHALL also be written; those averaged line figures SHALL offer
an optional, default-off min–max envelope band spanning the per-head minimum
and maximum at each epoch.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled: false`
- **THEN** no figure files are written

#### Scenario: Loss curve normalization toggle
- **WHEN** a user opens `loss_curves.html` and activates the normalization
  toggle below the title
- **THEN** the same document switches between raw loss and first-epoch
  normalized loss without loading another file, with the y-axis floored at
  `0.0`

#### Scenario: Confusion matrix square area and blank zero cells
- **WHEN** `confusion_matrix.html` is rendered
- **THEN** the area inside the axes is square and cells with a zero count are
  drawn as background with no hover

#### Scenario: Confusion matrix normalize and reorder
- **WHEN** a user toggles row-normalization and selects an order in
  `confusion_matrix.html`
- **THEN** the cell values switch between raw counts and recall, the same
  reorder permutation is applied to both rows and columns, and hover still
  reports Actual, Predicted, Actual sum, Predicted sum, Count, and Normalized

#### Scenario: Per-class metric and order-by are independent
- **WHEN** a user changes the metric selector and the order-by selector in
  `per_class_metrics.html`
- **THEN** the displayed metric and the row ordering update independently, and
  `auto:<selected-metric>` orders by whichever metric is currently selected

#### Scenario: Misclassification explorer by class
- **WHEN** a user selects a class in `misclassification_explorer.html`
- **THEN** the figure shows that class's false positives and false negatives as
  stacked horizontal bars

#### Scenario: Single-head layout
- **WHEN** a run has exactly one classification head
- **THEN** figures are written directly under `training_outputs.figures.dir`
  with no per-head subdirectory and no averaged loss/F1 figures

#### Scenario: Multi-head averaged envelope
- **WHEN** a run has more than one head and a user enables the envelope toggle
  on a top-level averaged loss or F1 figure
- **THEN** a min–max band spanning the per-head minimum and maximum at each
  epoch is drawn around the average line, and it is hidden by default
