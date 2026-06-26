
# Appendix — Deferred Features

## Purpose

Lists features that are intentionally out of scope for the initial
implementation but preserved as backlog items. Runtime entries name the
stub-test obligation they own per the testing policy in
`12-validation-testing-and-preflight.md`; cleanup milestones name their
verification obligation instead.

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

### P4.1 Bayesian sweeps

- Initial sweeps are Hydra multirun (grid / explicit value lists). See
  `09-sweeps-and-batch-runs.md`.
- Config slot: `sweep.mode: bayesian` with `sweep.bayesian`.
- Stub-test obligation: configure a Bayesian / Optuna-style sweep;
  invoke; assert `NotImplementedError` naming "Bayesian sweeps".

### P4.2 Multilabel support

- Config slot: `model.heads.<name>.type: multilabel_classification`.
- Meaning: one classifier head emits several binary labels from a single
  target/output vector. This is different from multi-head multiclass,
  where each target has its own `multiclass_classification` head.
- Current status: the head type name is reserved in the schema narrative,
  but runtime support is deferred. Multi-head multiclass is functional and
  should not depend on multilabel support.
- Stub-test obligation: configure a `multilabel_classification` head;
  invoke training or inference; assert `NotImplementedError` naming
  "multilabel support".

### P4.3 Aim logger sink

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

### P4.4 MLflow logger sink

- Config slot: `training_outputs.logging.sinks[].type: mlflow`.
- Schema is present. Runtime is stubbed.
- Stub-test obligation: configure a training run with an MLflow sink;
  invoke training entry point; assert `NotImplementedError` naming
  "MLflow logger sink".
- Extra: `mlflow` — **not currently declared** in `pyproject.toml`;
  re-add the extra when the runtime is unstubbed.

### P4.5 Non-DINOv2 SSL methods

- Config slots: `ssl.method: simclr | vicreg | pmsn | dino`.
- Stub-test obligation: one test per method asserting
  `NotImplementedError` naming the SSL method.
- The functional method is `dino_v2` via Lightly.

### P4.6a Weight-space ensembles

- Model soup, greedy soup, uniform soup, SWA, EMA.
- These produce a single exported model artifact (not a prediction-space
  combine). Out of scope for the initial implementation.
- Stub-test obligation: one test per selection / averaging strategy
  asserting `NotImplementedError`.

### P4.6b Weighted ensemble combine modes

- Weighted variants: weighted_logits_mean, weighted_probabilities_mean,
  weighted_vote, weighted_prediction_mean, weighted_ordinal_logits_mean,
  weighted_ordinal_probabilities_mean.
- Stub-test obligation: configure a weighted combine mode; invoke
  ensemble path; assert `NotImplementedError` naming
  "weighted combine modes".

### P4.6c `prediction_trimmed_mean`

- Sort member predictions, drop configured low / high extremes, average
  the rest.
- Stub-test obligation: configure
  `regression: prediction_trimmed_mean`; invoke ensemble path; assert
  `NotImplementedError`.

### P4.7 HDF / HDF5 derived result exports

- `.h5` metrics rollups, `results.h5`. Includes the corresponding
  `hdf` extra and `h5py` / `tables` dependencies (which are dropped).
- Stub-test obligation: configure an HDF export under a
  `*_outputs.export` block; assert `NotImplementedError` naming
  "HDF result exports".

### P4.8 Deprecated package removal

- Cleanup milestone, not a runtime feature or stub.
- Remove `src/dojo_deprecated/` once the new `src/dojo` implementation
  covers everything through P4.7 and nothing imports from it, per
  `13-workplan.md`.
- Verification obligation: no imports from `src/dojo_deprecated/`, no
  tests depending on it, and any remaining useful behavior has either
  been reimplemented in `src/dojo` or intentionally dropped.

### P4.9 Distributional and count regression heads

- Config slots: `model.heads.<name>.type: distributional_regression`
  (emits distribution parameters, e.g. `gaussian`, `negative_binomial`)
  and `count_regression` (emits count / rate parameters).
- Deferred together because neither has a result record type in the
  initial implementation: `06-results-artifacts-and-metadata.md` defines
  `regression_output` (single value + uncertainty) but no
  `distributional_output` / `count_output` for multi-parameter
  distribution or count / rate heads. Their dedicated losses
  (`gaussian_nll`, `negative_binomial_nll`, `poisson_nll`) are deferred
  with them; plain `regression` (`mse`, `mae`, `huber`, `smooth_l1`,
  `quantile`) is the only functional regression head type.
- Stub-test obligation: one test per head type — configure a
  `distributional_regression` / `count_regression` head; invoke training
  or inference; assert `NotImplementedError` naming the head type.

### P4.10 WebDataset backend

- Config slot: `data.backend: webdataset`.
- Stub-test obligation: configure `data.backend: webdataset`; invoke
  any train / eval / infer path; assert `NotImplementedError` naming
  "WebDataset backend".

### P4.11 Registry-based ensemble candidate discovery

- Broad automatic registry-based cross-run candidate discovery.
- Note: cross-run ensembling itself is **functional** via explicit
  candidate sources (`explicit`, `run_dir_glob`, `checkpoint_glob`,
  `result_uri_glob`, `manifest`). Only the registry-driven discovery is
  deferred. The term "cross-run ensemble" is no longer used as a
  distinct mode (see `08-ensembles.md`).
- Stub-test obligation: configure a registry-type candidate source;
  invoke discovery; assert `NotImplementedError` naming "registry-based
  candidate discovery".

## Cross-References

- `01-goals-and-scope.md` — scope and non-goals.
- `08-ensembles.md` — explicit candidate sources for the
  not-deferred parts of cross-run ensembling; deferred selection /
  combine modes.
- `09-sweeps-and-batch-runs.md` — Hydra multirun vs. Bayesian HPO.
- `11-dependencies.md` — extras list and dropped dependencies.
- `12-validation-testing-and-preflight.md` — stub-test policy in full.
- `glossary.md` — terminology.
