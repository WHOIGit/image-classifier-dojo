## Why

Figures are currently emitted only as a side effect of a training run
(`Training figures` requirement in `supervised-training`). Iterating on figure
quality — new plot types, restyling, bug fixes — forces a full retrain even
though every input already persists in the finished run: line plots read
`metrics/metrics.csv` and the confusion-matrix / per-class plots read canonical
`classification_output` result rows.

## What Changes

- Add a `dojo eval training-figures` command that regenerates the standalone
  HTML figure set from a completed run directory (or its resolved config),
  reading the persisted `metrics.csv` and `classification_output` rows, without
  retraining.
- Reuse the exact figure builders used during training so live and regenerated
  figures are byte-identical for the same inputs.
- Write to `training_outputs.figures.dir` (or an explicit output override),
  honoring `--clobber` semantics consistent with other artifact writers.

## Capabilities

### Modified Capabilities

- `supervised-training`: figures may be (re)generated from a completed run's
  persisted metrics and result rows via a dedicated command, in addition to
  being written during training.
