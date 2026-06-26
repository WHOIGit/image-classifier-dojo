
# 01. Goals and Scope

## Purpose

Defines the scope of the refactor: what the initial implementation must
deliver, what is out of scope, and what is stubbed / deferred. Other files
explain how each goal is satisfied; this file just names them.

## Core goals

- **Config-first, Hydra + Pydantic architecture.** Configs compose via
  Hydra and validate via Pydantic. Pydantic is the runtime contract.
- **Canonical CLI command families.** `dojo train` / `dojo infer` /
  `dojo eval` / `dojo inspect` / `dojo ensemble` / `dojo export`. Task
  paradigm selected via `task.type`. See `02-cli-and-task-types.md`.
- **Modular model composition.** Backbone + optional tabular encoder +
  optional tabular fusion + optional embedding adapter + one-or-more heads.
- **Multi-head, multi-objective supervised training.** Heads define
  output structure; objectives bind heads to losses, weights, and metrics.
- **Dataset backends:** `csv_manifest`, `parquet_manifest`,
  `parquet_images`, `ifcb_bins`. `class_folder` is an inspect-only source.
- **First-class embeddings.** Extractable from supervised models, SSL
  encoders, and transfer-learning checkpoints via `dojo infer embeddings`.
- **Canonical result schemas.** Tall Parquet via `amplify-db-utils`, with
  `_metadata.json` sidecar. CSV / wide exports are configurable derived
  outputs.
- **Layered validation and preflight.** Pydantic → `dojo inspect config`
  → `dojo inspect dataset` / training preflight → runtime validation.
- **Representation evaluation.** Both training-integrated and standalone
  (`dojo eval representation`). Usable against supervised or SSL encoders.
- **SSL via Lightly.** DINOv2 functional; other SSL methods are deferred.
- **Ensembling.** Prediction-space ensembles with explicit candidate
  discovery, candidate manifests, supported selection strategies, and
  combine modes. Snapshot ensembles via `task.type: snapshot_ensemble`.
- **Export.** TorchScript and ONNX via `*_outputs.export` blocks or
  `dojo export`.
- **Lightweight base install plus optional extras.** `train`, `timm`,
  `ssl`, `ifcb`, `repr_eval`, `onnx`, `all`, `dev`. (`aim` is commented
  out and `mlflow` / `s3` extras are not currently declared — Aim/MLflow
  logging is deferred; S3 rides on the base `amplify-storage-utils`
  dependency.)

## Non-goals for the initial implementation

- Maintaining backward compatibility with old listfile dataset formats.
- Porting the old `multilabel` module (it was multi-head multiclass, not
  multilabel).
- Porting `torchensemble`'s Bagging / Boosting / Fusion / Adversarial /
  FastGeometric strategies.
- Cross-run ensembling as a distinct algorithm type (it is just
  `dojo ensemble` with explicit candidate sources).

## Deferred-feature backlog

The following are out of scope for the initial implementation but are
intentionally preserved as deferred backlog items. Each has exactly one
test that asserts a `NotImplementedError` at the relevant runtime path.
See `appendix-deferred-features.md` and
`12-validation-testing-and-preflight.md`.

- Aim logger sink (schema present; runtime stubbed). Only the `local`
  sink is functional; metrics and figures are recorded locally for the
  foreseeable future.
- MLflow logger sink (schema present; runtime stubbed).
- Non-`dino_v2` SSL methods (SimCLR, VICReg, PMSN, original DINO).
- Weight-space ensembles (model soup, greedy soup, uniform soup, SWA,
  EMA).
- Weighted ensemble combine modes.
- `prediction_trimmed_mean`.
- WebDataset dataset backend.
- Bayesian / AutoML hyperparameter search.
- Registry-based ensemble candidate discovery (the **discovery
  mechanism**; cross-run ensembling itself is functional via explicit
  candidate sources).
- HDF / HDF5 derived result exports.

## Workplan framing

Use **"initial implementation"**, **"deferred-feature backlog"**, and
**"workplan"** language consistently. Do not use "phase 1", "first
refactor phase", or "post-Phase-8".

## Cross-References

- `02-cli-and-task-types.md` — canonical CLI surface.
- `03-configuration.md` — canonical config tree.
- `11-dependencies.md` — base + extras layout.
- `13-workplan.md` — priority order for the new `src/dojo`
  implementation.
- `appendix-deferred-features.md` — deferred features with their
  stub-test obligations.
- `glossary.md` — terminology.
