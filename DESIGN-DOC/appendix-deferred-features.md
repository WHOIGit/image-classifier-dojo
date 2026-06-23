
# Appendix — Deferred Features

## Purpose

Lists features that are intentionally out of scope for the initial
implementation but preserved as backlog items. Each entry names the
stub-test obligation it owns per the testing policy in
`12-validation-testing-and-preflight.md`.

## Stub-test policy recap

Each deferred runtime path has exactly one test that:

1. constructs a config that exercises the deferred feature;
2. invokes the runtime path;
3. asserts a `NotImplementedError`;
4. asserts the error message names the deferred feature and points at
   this backlog.

Schema-only tests, inspect-output enumeration tests, and scaffolded
runtime tests are **not** added for deferred features. When a feature is
unstubbed, the stub-assertion test is replaced with real functional
tests.

## Deferred features

### Aim logger sink

- Config slot: `training_outputs.logging.sinks[].type: aim`.
- Schema is present. Runtime is stubbed.
- Rationale: for the foreseeable future the project records metrics and
  figures **locally only** (the `local` sink). Aim was previously
  planned as functional but is deferred; the `aim` dependency also has
  no installable build on current Python (`aimrocks` ships no 3.13/3.14
  wheel), which reinforces deferring the runtime.
- Stub-test obligation: configure a training run with an Aim sink;
  invoke training entry point; assert `NotImplementedError` naming
  "Aim logger sink".
- Extra: `aim` (commented out in `pyproject.toml` while undeliverable on
  current Python).

### MLflow logger sink

- Config slot: `training_outputs.logging.sinks[].type: mlflow`.
- Schema is present. Runtime is stubbed.
- Stub-test obligation: configure a training run with an MLflow sink;
  invoke training entry point; assert `NotImplementedError` naming
  "MLflow logger sink".
- Extra: `mlflow` — **not currently declared** in `pyproject.toml`;
  re-add the extra when the runtime is unstubbed.

### Non-DINOv2 SSL methods

- Config slots: `ssl.method: simclr | vicreg | pmsn | dino`.
- Stub-test obligation: one test per method asserting
  `NotImplementedError` naming the SSL method.
- The functional method is `dino_v2` via Lightly.

### Weight-space ensembles

- Model soup, greedy soup, uniform soup, SWA, EMA.
- These produce a single exported model artifact (not a prediction-space
  combine). Out of scope for the initial implementation.
- Stub-test obligation: one test per selection / averaging strategy
  asserting `NotImplementedError`.

### Weighted ensemble combine modes

- Weighted variants: weighted_logits_mean, weighted_probabilities_mean,
  weighted_vote, weighted_prediction_mean, weighted_ordinal_logits_mean,
  weighted_ordinal_probabilities_mean.
- Stub-test obligation: configure a weighted combine mode; invoke
  ensemble path; assert `NotImplementedError` naming
  "weighted combine modes".

### `prediction_trimmed_mean`

- Sort member predictions, drop configured low / high extremes, average
  the rest.
- Stub-test obligation: configure
  `regression: prediction_trimmed_mean`; invoke ensemble path; assert
  `NotImplementedError`.

### Registry-based ensemble candidate discovery

- Broad automatic registry-based cross-run candidate discovery.
- Note: cross-run ensembling itself is **functional** via explicit
  candidate sources (`explicit`, `run_dir_glob`, `checkpoint_glob`,
  `result_uri_glob`, `manifest`). Only the registry-driven discovery is
  deferred. The term "cross-run ensemble" is no longer used as a
  distinct mode (see `08-ensembles.md`).
- Stub-test obligation: configure a registry-type candidate source;
  invoke discovery; assert `NotImplementedError` naming "registry-based
  candidate discovery".

### WebDataset backend

- Config slot: `data.backend: webdataset`.
- Stub-test obligation: configure `data.backend: webdataset`; invoke
  any train / eval / infer path; assert `NotImplementedError`.

### Bayesian / AutoML HPO

- Initial sweeps are Hydra multirun (grid / explicit value lists). See
  `09-sweeps-and-batch-runs.md`.
- Config slot: `sweep.mode: bayesian` with `sweep.bayesian`.
- Stub-test obligation: configure a Bayesian / Optuna-style sweep;
  invoke; assert `NotImplementedError`.

### HDF / HDF5 derived result exports

- `.h5` metrics rollups, `results.h5`. Includes the corresponding
  `hdf` extra and `h5py` / `tables` dependencies (which are dropped).
- Stub-test obligation: configure an HDF export under a
  `*_outputs.export` block; assert `NotImplementedError` naming
  "HDF result exports".

## Cross-References

- `01-goals-and-scope.md` — scope and non-goals.
- `08-ensembles.md` — explicit candidate sources for the
  not-deferred parts of cross-run ensembling; deferred selection /
  combine modes.
- `09-sweeps-and-batch-runs.md` — Hydra multirun vs. Bayesian HPO.
- `11-dependencies.md` — extras list and dropped dependencies.
- `12-validation-testing-and-preflight.md` — stub-test policy in full.
- `glossary.md` — terminology.
