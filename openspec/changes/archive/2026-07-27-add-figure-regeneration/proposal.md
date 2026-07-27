## Why

Figures are currently emitted only as a side effect of a training run
(`Training figures` requirement in `supervised-training`). Iterating on figure
quality — new plot types, restyling, bug fixes — forces a full retrain even
though every input already persists in the finished run: line plots read
`metrics/metrics.csv` and the confusion-matrix / per-class plots read canonical
`classification_output` result rows.

## What Changes

- Add a new top-level `dojo render` command group and a `dojo render run
  <run_dir>` command that regenerates the standalone HTML figure set from a
  completed run's top-level directory, reading the persisted `metrics.csv`,
  `classification_output` rows, and `config/resolved.yaml`, without retraining.
- Reuse the exact figure builders used during training so live and regenerated
  figures are byte-identical for the same inputs; `objective_to_head` is
  reconstructed from the run's snapshot config (identity fallback for older
  runs lacking it).
- Overwrite the run's `figures/` in place by default; `--backup` first copies
  the existing `figures/` to a numeric-incremented sibling, then overwrites.

## Capabilities

### Modified Capabilities

- `supervised-training`: figures may be (re)generated from a completed run's
  persisted metrics and result rows via a dedicated command, in addition to
  being written during training.
