## MODIFIED Requirements

### Requirement: Training figures
When `training_outputs.figures.enabled: true`, the system SHALL write
standalone HTML figures under `training_outputs.figures.dir`:
`loss_curves.html`, `loss_curves_normalized.html`, `val_f1_curves.html`,
`confusion_matrix.html`, and `per_class_metrics.html`, implemented as embedded
JSON plus Plotly.js from the CDN (no Python Plotly dependency). Line plots read
`metrics/metrics.csv`; confusion matrix and per-class metrics read canonical
`classification_output` rows. The system SHALL also expose a command that
regenerates this same figure set from a completed run's persisted
`metrics/metrics.csv` and `classification_output` rows without retraining,
producing figures identical to the training-time output for the same inputs.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled: false`
- **THEN** no figure files are written during training

#### Scenario: Regenerate figures from a completed run
- **WHEN** the figure-regeneration command is run against a finished run
  directory that contains `metrics/metrics.csv` and `classification_output`
  result rows
- **THEN** the full HTML figure set is rewritten from those artifacts without
  retraining
