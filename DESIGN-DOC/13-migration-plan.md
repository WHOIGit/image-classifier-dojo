# 13. Migration Plan

## Purpose

Describes how the refactor lands incrementally without breaking the
existing codebase mid-flight. Uses "initial implementation",
"deferred-feature backlog", and "migration plan" language. Avoids "phase"
numbering except when explicitly removing it from older docs.

**Guiding rule:** prioritize porting extant capabilities before adding
new similar ones. Anything in `src/dojo_deprecated/` that has a working
analogue gets ported first; new capabilities are stubbed in the
meantime per the testing policy
(`12-validation-testing-and-preflight.md`).

## Step 1 — Deprecation move

Status: COMPLETE

1. `git mv src/dojo src/dojo_deprecated`.
2. Create an empty `src/dojo/` package skeleton (`__init__.py`,
   `cli/__init__.py`, `config_schemas/__init__.py`).
3. Update `pyproject.toml`:
   - bump `version` to mark the refactor cycle;
   - point console-scripts entry at the new `src/dojo/cli/main.py`
     (which initially just prints a friendly "refactor in progress"
     message).
4. `src/dojo_deprecated/` remains importable for reference during the
   refactor and is deleted in the final step once nothing depends on it.

The deprecated tree is the source of porting cues:
`dojo_deprecated/multiclass/`, `dojo_deprecated/multilabel/`,
`dojo_deprecated/selfsupervised/`,
`dojo_deprecated/tools/dataset_lists_from_folder.py`,
`dojo_deprecated/schemas/core.py`, and
`dojo_deprecated/multiclass/callbacks.py` are all useful references.

The old `multilabel` module is actually multi-head multiclass; do not
port it directly. Reserve `multilabel_classification` for true multi-hot
multilabel problems. Use the new multi-head paradigm for current and
future head types.

## Step 2 — Config, CLI, and storage foundation

1. Add Hydra entrypoint and `configs/` skeleton (group dirs aligned with
   `03-configuration.md`).
2. Add `config_schemas` (Pydantic) — the validated contract per
   `03-configuration.md`.
3. Add Hydra ↔ `output_root` / `*_outputs` reconciliation defaults per
   `09-sweeps-and-batch-runs.md`.
4. Add `dojo inspect config` (replaces `dojo validate-config`).
5. Add local / amplify storage resolver.
6. Add result config schemas (per
   `06-results-artifacts-and-metadata.md`).
7. Add logger abstraction (local sink functional; Aim and MLflow stubbed
   per `appendix-deferred-features.md`).
8. Add `dojo inspect dataset` — built early because it is the cheapest
   gate against bad manifests. Folds in the old `dojo tools
   make-manifest` behavior.

## Step 3 — Supervised refactor

1. Implement the shared dataset record contract.
2. Implement `csv_manifest`, `parquet_manifest`, and `parquet_images`
   datamodules.
3. Add backbone registry — `torchvision`, `timm`, and `checkpoint`
   sources all functional (timm gated by the `timm` extra).
4. Add head registry with required `target` validation per
   `05-models-training-and-heads.md`.
5. Add objectives.
6. Add the supervised model compositor.
7. Add the supervised LightningModule (`task.type: supervised`).
8. Add the canonical results writer.
9. Port and fix the `count_perclass` / `parse_targets_file` logic from
   `src/dojo_deprecated/multiclass/datasets.py` into the new inspector
   and datamodules. The old `multilabel/datasets.py` bug (undefined
   `target` should be `label`) is fixed as part of this porting.

## Step 4 — Transforms

1. Add transform builder.
2. Add letterbox, aspect buckets, size buckets, foreground-aware crop,
   grayscale, normalization (per `05-models-training-and-heads.md`).

## Step 5 — Logging sinks

1. Wire up the local logger — metrics and figures recorded locally. This
   is the only functional sink for the foreseeable future.
2. Add the composite logger.
3. Aim logger: **stubbed** per `12-validation-testing-and-preflight.md` —
   config schema present, runtime raises `NotImplementedError`. Aim is
   deferred (see `appendix-deferred-features.md`); its dependency also
   has no installable build on current Python (`aimrocks` ships no
   3.13/3.14 wheel). The `AimLoggerConfig.artifacts_location` URI
   handling in `src/dojo_deprecated/schemas/core.py` (including
   `file:///absolute/path` normalization) is the porting cue when Aim is
   later unstubbed.
4. MLflow logger: **stubbed** per
   `12-validation-testing-and-preflight.md` — config schema present,
   runtime raises `NotImplementedError`.
5. Ensure result / artifact config works across the local sink.

## Step 6 — Export

1. Add TorchScript single-model export.
2. Add TorchScript snapshot / ensemble export.
3. Add ONNX single-model export.
4. Add ONNX metadata embedding.
5. Add bucket-aware ONNX export support.

## Step 7 — IFCB bins

1. Add `ifcbkit` dependency. Remove `pyifcb` dependency.
2. Implement the IFCB-bins dataset.
3. Implement the IFCB-bins datamodule.
4. Add bin manifest support with `bin_id_column` and `bin_uri_column`.
5. Add runtime ROI expansion.
6. Add tests for variable ROI counts per bin.

## Step 8 — SSL with Lightly (DINOv2 only)

1. Add the `dojo[ssl]` extra.
2. Add the Lightly DINOv2-style task.
3. Add SSL transforms.
4. Add representation evaluation per
   `07-ssl-and-representation-eval.md` (labeled probes; unlabeled
   diagnostics / retrieval / clustering / projections).
5. Add encoder export.
6. Non-DINOv2 SSL methods (SimCLR, VICReg, PMSN, original DINO) remain
   **stubbed** — see `appendix-deferred-features.md`.

## Step 9 — Ensembles and cleanup

1. Add the shared candidate discovery / compatibility pipeline (explicit
   sources only).
2. Add prediction-space combine modes (averaging and voting).
3. Add selection strategies (`all`, `best_candidate`, `top_k`,
   `greedy_forward_selection`, `cycle_end_snapshots`) usable against
   both live inference and cached canonical results.
4. Add `task.type: snapshot_ensemble`: integrate the snapshot-cycle
   scheduler in the supervised LightningModule and feed snapshot
   checkpoints into the ensemble pipeline through the shared run
   directory.
5. Add `dojo ensemble` — accepts user-supplied checkpoints, exported
   model files, run directories, URIs, and cached-result directories.
6. Add `dojo ensemble candidates` for explicit candidate-source
   discovery and manifest writing.
7. Add the ensemble bundle artifact format.
8. Add ensemble result writing.
9. **Drop `torchensemble`** from dependencies. Confirm nothing imports
   from `src/dojo_deprecated/homogenous_ensembles/`.
10. **Delete `src/dojo_deprecated/`** once a green CI run on Tier 1
    fixtures (`12-validation-testing-and-preflight.md`) confirms no
    remaining dependencies.
11. Verify all stubbed paths (Aim logger, MLflow, non-DINOv2 SSL,
    weight-space ensembles, weighted combine modes, registry-based
    candidate discovery, WebDataset) raise clear `NotImplementedError`
    messages that name the feature and point at the backlog.
12. Commit fixture Parquet under `tests/fixtures/parquet/` via git LFS
    — pending explicit owner approval.

## Deferred-feature backlog

Carried as separate work items, in no particular order:

- Aim logger runtime (deferred; also blocked by `aimrocks` lacking a
  current-Python wheel);
- MLflow logger runtime;
- non-DINOv2 SSL methods (SimCLR, VICReg, PMSN, original DINO);
- weight-space ensembles: model soup / greedy soup, SWA, EMA;
- weighted ensemble combine modes;
- `prediction_trimmed_mean`;
- registry-based ensemble candidate discovery;
- WebDataset backend;
- Bayesian / AutoML hyperparameter search;
- HDF / HDF5 derived result exports;
- `improv` export integration;
- Inception-style auxiliary-logit handling.

See `appendix-deferred-features.md` for stub-test obligations per item.

## Term replacements applied during the migration

The following stale terms are removed from new code and documentation
(intentional mentions are allowed only in migration notes and
replacement notes):

- `ssl_eval` → `representation_eval`;
- `dojo validate-config` → `dojo inspect config`;
- `dojo tools make-manifest` → `dojo inspect dataset` behavior;
- `dojo eval embeddings` → `dojo infer embeddings`;
- `dojo train supervised` / `dojo train ssl` /
  `dojo train-snapshot-ensemble` → `dojo train` with `task.type`;
- `model.fusion` → `model.tabular.fusion`;
- `checkpoint_id` → `checkpoint_hash` plus filename convention;
- `train.skip` / `runtime.skip_training` → removed; historical-run
  ensembling uses `dojo ensemble` with `run_checkpoints` candidate
  sources;
- `type: pt` → removed; use `torchscript` or `onnx`;
- `ordinal_regression` → `ordinal_classification`;
- `model.backbone.source: lightly` → removed; SSL configs use
  `ssl.framework: lightly` while backbone source is `timm`,
  `torchvision`, or `checkpoint`.

## Cross-References

- `01-goals-and-scope.md` — scope of the migration.
- `02-cli-and-task-types.md` — CLI replacements.
- `03-configuration.md` — new top-level config tree.
- `06-results-artifacts-and-metadata.md` — `checkpoint_hash`,
  partitioning, sidecar `_metadata.json`.
- `07-ssl-and-representation-eval.md` — `representation_eval` renaming.
- `08-ensembles.md` — ensemble migration.
- `11-dependencies.md` — dropped dependencies (`torchensemble`,
  `pyifcb`, `h5py` / `tables`).
- `12-validation-testing-and-preflight.md` — stub-test obligations.
- `appendix-deferred-features.md` — deferred backlog.
