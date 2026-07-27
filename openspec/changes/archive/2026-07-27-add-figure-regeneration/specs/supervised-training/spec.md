## MODIFIED Requirements

### Requirement: Training figures
When `training_outputs.figures.enabled: true`, the system SHALL write
standalone HTML figures under `training_outputs.figures.dir`:
`loss_curves.html`, `val_f1_curves.html`, `confusion_matrix.html`,
`per_class_metrics.html`, and `misclassification_explorer.html`, implemented as
embedded JSON plus Plotly.js from the CDN and a small shared client-side control
layer (no Python Plotly dependency). Line plots read `metrics/metrics.csv`;
confusion matrix, per-class metrics, and misclassification explorer read
canonical `classification_output` rows. The system SHALL also expose a command
that regenerates this same figure set from a completed run's top-level
directory — its persisted `metrics/metrics.csv`, `classification_output` rows,
`_metadata.json`, and `config/resolved.yaml` — without retraining, producing
figures identical to the training-time output for the same inputs. The command
SHALL overwrite the run's `figures/` in place.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled: false`
- **THEN** no figure files are written during training

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
