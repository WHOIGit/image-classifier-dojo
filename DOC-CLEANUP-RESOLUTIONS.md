# Design Doc Cleanup Resolutions

This file captures decisions made while reviewing `REFACTOR-DESIGN-DOC.md`.
These resolutions are intended to be applied to the design doc after the review
pass is complete.

## Config Structure

- Use a top-level `outputs` group for run paths, logging, results, and export.
- Rename `ssl_eval` to `representation_eval`.
- Keep `optimizer`, `scheduler`, and `checkpointing` as top-level config groups.
- Keep `storage` top-level, separate from `runtime`.
- Move `seed` under `runtime`.
- Add top-level `runtime` for run/process behavior.
- Remove `train.skip` / `runtime.skip_training`.
- Group `backbone`, `tabular`, `embedding_adapter`, and `heads` under `model`.
- Use `model.tabular.fusion`, not top-level `model.fusion`.
- Use top-level `sweep_outputs` for multirun/model-comparison outputs.

Canonical root shape:

```yaml
experiment:
task:
runtime:
storage:

data:
transforms:

model:
  backbone:
  tabular:
    fusion:
  embedding_adapter:
  heads:

objectives:

ssl:
representation_eval:

training:
optimizer:
scheduler:
checkpointing:

ensemble:

outputs:
  run_root:
  run_dir_template:
  sweep_dir_template:
  logging:
  results:
  export:

sweep_outputs:
```

## Output Paths, Run IDs, and Config Artifacts

- Split output path configuration into `outputs.run_root` and
  `outputs.run_dir_template`.
- Add `outputs.sweep_dir_template`.
- Use Dojo-owned Python-style template syntax for paths, for example:
  - `{experiment.name}`
  - `{runtime.run_id}`
  - `{model.backbone.name:slug}`
  - `{training.batch_size:03}`
- Prefer this syntax over OmegaConf-style `${...}` for output paths.
- Render Dojo path templates after config composition, validation, and runtime
  value generation.
- Allow `runtime.run_id` to be static or generated, including generated
  coolname-style values.
- Include generated runtime values in resolved config artifacts.
- Warn or error on accidental output overwrite according to an explicit policy. 

Run config artifacts should include:

```text
config/composed.yaml      # after config composition and CLI overrides
config/resolved.yaml      # fully resolved with generated values and defaults
config/resolved.json
config/cli.txt            # invoked command and overrides
config/overrides.txt
config/sweep_values.txt   # only for sweep members
```

## CLI Structure

- Canonical command families are `dojo train`,  `dojo infer` and `dojo eval`.
- Keep config-first commands; subcommands are shorthands that constrain the
  target output.
- Use:
  - `dojo infer`
  - `dojo infer predictions`
  - `dojo infer embeddings`
  - `dojo eval`
  - `dojo eval holdout`
  - `dojo eval representation`
- Rename the old `dojo eval embeddings` concept to `dojo infer embeddings`.
- Remove `dojo eval knn` as a primary command for now.
- Remove `dojo eval linear-probe` as a primary command for now.
- Remove `dojo tools make-manifest`.
- Fold class-folder manifest generation into `dojo inspect dataset`.
- Replace `dojo validate-config` with `dojo inspect config`.

`dojo inspect config` should:

- compose configs;
- apply overrides;
- run schema validation;
- render output path templates;
- show expected output folder structure;
- warn about run/sweep directory collisions;
- show enabled outputs and deferred/stubbed features;
- optionally emit machine-readable JSON for CI/tests.

## Dataset Architecture

- Abandon old listfile dataset formats instead of maintaining compatibility.
- Treat `class_folder` as an `inspect dataset` source, not a train/eval/infer
  backend.
- Supported train/eval/infer dataset backends:
  - `csv_manifest`
  - `parquet_manifest`
  - `parquet_images`
  - `ifcb_bins`
- `dojo inspect dataset` is read-only by default.
- `dojo inspect dataset` may write canonical CSV/Parquet manifests when an
  output path is explicitly configured.
- Inspect outputs are not normal run outputs and should not require a run
  directory.
- `dojo inspect dataset` should report missing targets and summarize how many
  samples will be dropped, skipped, or fail validation.
- Default missing-target policy is `error` for all heads/head counts.

IFCB behavior to preserve or port if supported by `ifcbkit`:
- blacklist/exclude filtering;
- old schema handling;
- shuffle buffer;
- stable ROI IDs;
- optional estimated or cached length.

## Model, Heads, and Objectives

- The old `multilabel` module is actually multi-head multiclass and should not
  be directly ported.
- Move the old multi-head multiclass module to `dojo_deprecated` for reference.
- Move all current `src/dojo` code to `src/dojo_depricated`, kept for reference but will ultimately be removed. 
- Reserve `multilabel_classification` for true multi-hot multilabel problems.
- Use the new multi-head paradigm for current and future head types.
- Heads reference logical data targets, not raw manifest columns.
- Objectives bind heads to losses, metrics, and weights.

Reference chain:

```text
objective -> head -> data target -> physical column
```

Example:

```yaml
data:
  targets:
    species:
      column: species_idx
      type: multiclass_classification

model:
  heads:
    species:
      type: multiclass_classification
      target: species

objectives:
  species:
    head: species
```

## Dependency Plan

- Use a lightweight base install plus optional extras.
- Base install should support config validation, inspection, storage/result
  access, schema handling, and artifact introspection without requiring Torch.
- `amplify-db-utils` is a core dependency.
- `amplify-storage-utils` remains a core dependency.
- Training, SSL, Aim, MLflow, ONNX, and IFCB support should live behind extras.

Suggested dependency shape:

```toml
dependencies = [
  "pydantic",
  "pydantic-settings",
  "hydra-core",
  "omegaconf",
  "pyarrow",
  "duckdb",
  "amplify-storage-utils",
  "amplify-db-utils",
  "typer",
  "rich",
  "coolname",
  "humanize",
  "tqdm",
]

[project.optional-dependencies]
train = [
  "torch",
  "torchvision",
  "lightning",
  "torchmetrics",
  "scikit-learn",
  "numpy",
  "pandas",
  "pillow",
]

ssl = [
  "lightly",
  "umap-learn",
  "hdbscan",
  "ifcbkit",
]

timm = ["timm"]
aim = ["aim"]
mlflow = ["mlflow"]
onnx = ["onnx", "onnxruntime"]
```

## Results Backend and Schemas

- Dojo owns canonical result schemas and `_metadata.json`.
- Use `amplify-db-utils` for partitioned DuckDB/Parquet-backed result writing,
  schema registration/checking, filtered reads, bulk reads, local paths, and
  object-store-compatible paths.
- Use explicit `pyarrow.Schema` definitions for Dojo result tables, especially
  for Arrow list/vector columns.
- Writers should handle Arrow/Parquet dictionary encoding for low-cardinality
  or repeated columns.
- Keep `improv` export/integration deferred to appendix.

Columns that benefit from dictionary encoding include:

- `split`
- `stage`
- `record_type`
- `head_name`
- `embedding_kind`
- `prediction_label`
- `diagnostic_scope`
- `resize_width_px`
- `resize_height_px`

Use Arrow list-columns for vector values:

```text
embedding
logits
probabilities
ordinal_logits
```

Only per-sample records belong in result Parquet files. Per-class metrics,
run-level metrics, and plots belong in `metrics/` and `figures/`.

## Result Metadata

- Keep `_metadata.json`.
- Use top-level `record_types` keys, not a top-level `heads` key.
- Store semantic/provenance metadata there, not low-level Parquet encoding
  details.
- Per-record-type metadata should contain class mappings, target transforms,
  head mappings, and relevant schema semantics.
- Include `schema_version` as a column and also summarize schema metadata in
  `_metadata.json`.
- `source_extra_json` is allowed only when configured via
  `source_extra_columns`.
- `source_extra_json` should be a single JSON string column and only appear on
  `record_type=sample_metadata`.
- Tabular features should use a single `tabular_features_json` column on
  sample metadata records.
- First-class sample metadata columns include:
  - `native_width_px`
  - `native_height_px`
  - `resize_width_px`
  - `resize_height_px`

Regression result columns should include:

```text
target
prediction_value
target_transformed
prediction_value_transformed
prediction_uncertainty
```

## Artifact Layout

- Make `figures/` a first-class output directory.
- Do not add a generic training-run `data/` folder; input dataset information is
  covered by configs and resolved config artifacts.
- Use `ensemble_manifests/` for ensemble candidate manifests.
- Use JSON for ensemble manifests, not Parquet.

Recommended run layout:

```text
runs/{run_id}/
  config/
  checkpoints/
  exports/
  metrics/
  figures/
  results/
  ensemble_results/
  ensemble_metrics/
  ensemble_figures/
  ensemble_manifests/
  ensemble_members/
```

`ensemble_members/` is optional and only used when `dojo ensemble` materializes
member artifacts locally.

## Result Partitioning

- Support configurable Parquet partitioning via `outputs.results.partition_by`.
- Partition fields may include values like:
  - `stage`
  - `epoch`
  - `ensemble_member_id`
  - `record_type`
- Avoid awkward null-only ensemble member partitions when possible; existing
  `checkpoint_id` and `model_id` usually identify source model/checkpoint.
- Include `sweep_id` when a row is produced as part of a Hydra sweep.
- Distinguish:
  - `split`: source dataset split, such as `train`, `val`, `test`, `unlabeled`
  - `stage`: process that produced the row, such as `train_validation`,
    `holdout_eval`, `infer`, `representation_eval`, `ensemble_eval`

## Runtime Controls

Keep these runtime controls:

- `fast_dev_run`
- `precision`
- `num_workers`
- `autobatch`
- early stopping

ONNX export does not need to be a runtime config item. It should be handled by
the config-driven task orchestration/export system outside the training loop.

## Logging and Diagnostics

- Keep Aim as a functional logging sink.
- Keep Aim diagnostic figures.
- MLflow config schema may exist, but runtime implementation will remain a clear
  stub for the first refactor phase. As a defered feature, it gets put in the design-doc Defered appendix

## Validation and Preflight

Use layered validation:

1. Pydantic static schema validation.
2. `dojo inspect config` for composition, validation, rendered paths, and output
   tree preview.
3. `dojo inspect dataset` / training preflight for manifest and target checks.
4. Runtime validation for checkpoints, models, schemas, tensor shapes, and
   exports.

Use `runtime.preflight` for preflight controls.

Dataset checks should include:

- `empty_train_classes`
- `empty_eval_classes`
- `non_contiguous_class_indices`
- `imbalance_ratio_gt`

`imbalance_ratio_gt` means:

```text
max_class_count / min_nonzero_class_count > configured_threshold
```

This should default to a warning, not an error.

## SSL and Representation Evaluation

- Use:

```yaml
ssl:
  method: DINOv2
  framework: lightly
```

- DINOv2 through Lightly is functional in the refactor.
- SimCLR, VICReg, PMSN, and original DINO are intentionally removed from the
  first refactor runtime. Defered to appendix.
- Preserve config stubs for deferred SSL methods with clear errors.
- Rename SSL evaluation to representation evaluation.
- `representation_eval` should support both training-integrated evaluation and
  standalone `dojo eval representation`.
- UMAP and t-SNE are optional-extra features but not deferred.
- HDBSCAN is optional-extra-backed but not deferred.
- Regression and ordinal probes are included and not deferred.
- Supervised fine-tuning from an SSL pretrained backbone belongs under
  supervised transfer learning.
- Supervised probes stay in SSL/representation-evaluation design.

## Ensemble Architecture

- Ensemble means prediction-space multi-model output processing.
- Do not use separate `cross_run_ensemble` or `snapshot_ensemble` algorithm types.
- Cross-run and snapshot are candidate-source patterns, not separate ensemble
  algorithms.
- Defer weight/checkpoint averaging methods such as SWA, EMA, model soup,
  greedy soup, and uniform soup.
- Drop `torchensemble`.
- Do not port Bagging, Boosting, Fusion, Adversarial, or FastGeometric
  strategies.

Supported selection strategies:

- `all`
- `best_candidate`
- `top_k`
- `greedy_forward_selection`
- `cycle_end_snapshots`

`best_candidate` means selecting the single best candidate from the candidate
set by the configured metric. It is useful as a baseline/control.

Supported combine modes:

- classification:
  - `probabilities_mean`
  - `logits_mean`
  - `majority_vote`
  - `soft_vote`
- regression:
  - `prediction_mean`
  - `prediction_median`
- ordinal:
  - `ordinal_probabilities_mean`
  - `ordinal_logits_mean`

Defer:

- weighted combine modes;
- `prediction_trimmed_mean`.

`prediction_trimmed_mean` means sorting member predictions, dropping configured
low/high extremes, and averaging the rest.

## Ensemble Inputs, Manifests, and Commands

- Use `dojo ensemble candidates` for discovery, compatibility inspection, cache
  assessment, and manifest creation.
- Use `dojo ensemble` for actual ensemble evaluation/inference.
- A manifest is the normalized output of candidate-source discovery unless a
  manifest is provided directly.
- Manifests may be written outside run directories for reusable/shared use.
- Compatibility and cache reports may also be written outside run directories.
- Assess compatibility by default using a cascading metadata policy:
  1. `config/resolved.json`
  2. result metadata
  3. checkpoint metadata
  4. exported model metadata
- Support explicit metadata-source policies and drift checks across multiple
  metadata targets.
- Ensemble commands should default to using existing result files as inputs when
  input dataset and output target match.
- Use a source policy to handle mismatches:
  - `strict_no_inference`
  - `inference_as_needed`
  - `force_inference`
- Allow selecting target split, such as validation or holdout.

## Ensemble Outputs

- Use ensemble-specific output folders:
  - `ensemble_results/`
  - `ensemble_metrics/`
  - `ensemble_figures/`
  - `ensemble_manifests/`
- This allows ensemble artifacts to coexist with training artifacts in workflows
  like train-snapshot-ensemble.
- Optional `ensemble_members/` materialization is allowed only for
  `dojo ensemble` runs.
- Local member files may be symlinked.
- Remote member files may be cached through `storage` config and then symlinked.
- Ensemble metrics and figures should compare member best/final metrics against
  the created ensemble model.
- Ensemble-specific output controls may live under `ensemble.outputs`.

## Hydra Sweeps and Sweep Outputs

- Use Hydra sweeps to explore ensemble strategy/combine-mode combinations.
- Add top-level `sweep_outputs` for final model comparison metrics across sweep
  jobs.
- Use `outputs.sweep_dir_template` to control sweep output location.
- `sweep_outputs` controls aggregation content.

Example sweep output needs:

- compare best/final F1 across best/final epoch for output model trainings;
- compare per-class F1 across best/final epoch for model trainings;
- write comparison metrics CSV/json/HDF;
- write comparison figures.

## Export

- ONNX export remains supported but is outside runtime training config.
- Export should be explicit through `outputs.export`, `dojo export`, or
  task-orchestration config.

## Deferred Features Appendix

Move deferred features out of the main design flow and into an appendix.

Deferred features include:

- SimCLR, VICReg, PMSN, and original DINO runtime implementations;
- generic timm backbone support, subject to later timm/DINOv2 clarification;
- MLflow runtime implementation;
- model soup, greedy soup, uniform soup;
- SWA and EMA workflows if not implemented in the first phase;
- weighted ensemble combine modes;
- `prediction_trimmed_mean`;
- generic multimodal/multibranch fusion outside `model.tabular.fusion`;
- `improv` export/integration;
- Prefect flows;
- WebDataset;
- Bayesian/AutoML HPO;
- broad automatic registry-based cross-run discovery if not implemented in the
  first phase.
