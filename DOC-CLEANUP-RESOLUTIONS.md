# Design Doc Cleanup Resolutions

This file captures decisions made while reviewing `REFACTOR-DESIGN-DOC.md`.
These resolutions are intended to be applied to the design doc after the review
pass is complete.

## Config Structure

- Use top-level `output_root` (a single filepath string) as the base
  for `training_outputs`, `ensemble_outputs`, and `sweep_outputs`
  directories when those directories are rendered from `dir_template`
  or supplied as bare relative `dir` values.
- Use top-level `training_outputs`, `ensemble_outputs`, and
  `sweep_outputs` as **peer** output-config blocks, one per output
  family. Each has its own `dir_template` (relative to `output_root`)
  and its own `results`, `export`, `metrics`, and `figures`
  sub-blocks as applicable (sweep has no `results`).
- The old top-level `outputs:` group is replaced by
  `training_outputs:`. There is no longer a single `outputs:` block.
- Rename `ssl_eval` to `representation_eval`.
- Keep `optimizer`, `scheduler`, and `checkpointing` as top-level config groups.
- Keep `storage` top-level, separate from `runtime`.
- Move `seed` under `runtime`.
- Add top-level `runtime` for run/process behavior.
- Remove `train.skip` / `runtime.skip_training`.
- Group `backbone`, `tabular`, `embedding_adapter`, and `heads` under `model`.
- Use `model.tabular.fusion`, not top-level `model.fusion`.
- `logging` lives under `training_outputs.logging` and applies only to
  model-training runs. Standalone `dojo ensemble` and
  `dojo ensemble candidates` invocations do not initialize experiment
  logging.

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

output_root:

training_outputs:
  dir:
  dir_template:
  logging:
  results:
  export:
  metrics:
  figures:

ensemble_outputs:
  dir:
  dir_template:
  results:
  export:
  metrics:
  figures:
  members:
  manifests:

sweep_outputs:
  dir:
  dir_template:
  export:
  metrics:
  figures:
```

For a `task.type: snapshot_ensemble` run, both `training_outputs.dir_template`
and `ensemble_outputs.dir_template` typically resolve to the same
directory so training checkpoints and ensemble artifacts share one
run directory. Defaulting `ensemble_outputs.dir_template` to match
`training_outputs.dir_template` is the recommended pattern.

Minimal supervised config example:

```yaml
experiment:
  name: ifcb_species_baseline

task:
  type: supervised

runtime:
  seed: 123
  run_id: "{coolname}"
  precision: bf16-mixed
  num_workers: 8
  fast_dev_run: false

storage:
  local_cache_dir: ./.cache/dojo

data:
  backend: parquet_manifest
  manifest_uri: s3://datasets/ifcb/species_manifest.parquet
  image_uri_column: image_uri
  split_column: split
  sample_id_column: roi_id
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      class_names: s3://datasets/ifcb/species_classes.json
      missing_policy: error

model:
  backbone:
    source: torchvision
    name: resnet50
    weights: IMAGENET1K_V2
  tabular:
    enabled: false
  embedding_adapter:
    enabled: false
  heads:
    species:
      type: multiclass_classification
      target: species
      num_classes: 42

objectives:
  species:
    head: species
    loss: cross_entropy
    metrics: [accuracy, macro_f1, per_class_f1]
    weight: 1.0

training:
  max_epochs: 50
  batch_size: 64

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_epochs: 3

checkpointing:
  monitor: val/species/macro_f1
  mode: max
  save_top_k: 3

output_root: ./runs

training_outputs:
  dir_template: "{experiment.name}/{runtime.run_id}"
```

## Output Paths, Run IDs, and Config Artifacts

- Top-level `output_root` is a single filepath string used as the base
  for `training_outputs.dir_template`, `ensemble_outputs.dir_template`,
  and `sweep_outputs.dir_template` when direct `*_outputs.dir` values
  are not supplied. It is also the base for bare relative top-level
  `*_outputs.dir` values.
- Each `*_outputs` block carries its own concrete `dir` or
  `dir_template`:
  - `training_outputs.dir`
  - `training_outputs.dir_template`
  - `ensemble_outputs.dir`
  - `ensemble_outputs.dir_template`
  - `sweep_outputs.dir`
  - `sweep_outputs.dir_template`
- When `dir_template` is used, the rendered template is joined under
  `output_root` to produce the resolved `*_outputs.dir` (the actual
  filesystem path written to). Tooling commands may set
  `*_outputs.dir` directly when they need a concrete shared output
  location.
- Design note for all `dir` keys:
  - absolute paths are used as provided, for example `dir: /folder`;
  - relative paths starting with `./` are resolved relative to the
    process current working directory, for example `dir: ./folder`;
  - bare relative paths are resolved relative to the parent object's
    resolved `dir`; for top-level `*_outputs.dir` values, the parent
    base is `output_root`; for sub-block values, such as
    `results.dir: folder` under `training_outputs`, the parent base is
    `training_outputs.dir`.
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
- Warn or error on accidental output overwrite according to an explicit
  policy. The supported `existing_run_dir` values are:
  - `error` — refuse to start when the resolved run directory already
    exists. **Default.**
  - `overwrite` — delete all extant content of the resolved run
    directory before starting.
- When multiple output blocks resolve to the same physical directory
  (for example `training_outputs.dir` and `ensemble_outputs.dir` in a
  `task.type: snapshot_ensemble` run), the overwrite policy is evaluated
  once per resolved physical directory at command startup. Later phases
  of the same command must not re-apply `overwrite` and delete artifacts
  produced by earlier phases.

Run config artifacts should include:

```text
config/composed.yaml      # after config composition and CLI overrides
config/resolved.yaml      # fully resolved with generated values and defaults
config/resolved.json
config/cli.txt            # invoked command and overrides
config/overrides.txt
config/sweep_values.txt   # only for sweep members
```

Output template example:

```yaml
runtime:
  run_id: "{coolname}"
  sweep_id: "{coolname}"

output_root: ./runs

model:
  backbone:
    source: torchvision
    name: efficientnet_b0,resnet50
  ...

training:
  batch_size: 32,64

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.backbone.name:slug}/bs{training.batch_size:03}/
  existing_run_dir: error

sweep_outputs:
  dir_template: "{experiment.name}/sweep_results/{runtime.sweep_id}"
```

## CLI Structure

- Canonical command families are `dojo train`, `dojo infer`, `dojo eval`,
  `dojo inspect`, `dojo ensemble`, and `dojo export`.
- Keep config-first commands; subcommands are shorthands that constrain the
  target output.
- Use:
  - `dojo train` (task type selected via `task.type`)
  - `dojo infer`
  - `dojo infer predictions`
  - `dojo infer embeddings`
  - `dojo eval`
  - `dojo eval holdout`
  - `dojo eval representation`
  - `dojo inspect`
  - `dojo inspect config`
  - `dojo inspect dataset`
  - `dojo inspect backbone`
  - `dojo inspect checkpoint`
  - `dojo ensemble`
  - `dojo ensemble candidates`
  - `dojo export`
- `dojo train` dispatches on `task.type` (see Task Types section below):
  `supervised`, `ssl`, `snapshot_ensemble`. There are no
  `dojo train supervised` / `dojo train ssl` / `dojo train-snapshot-ensemble`
  subcommands.
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

Remote-artifact validation strength:

- Default behavior is **schema-and-local feasibility only**: resolve
  paths, validate format of any locally-accessible artifacts, do not
  require network access to remote URIs. CI-safe and offline-friendly.
- Opt-in `--check-remote` flag (or `inspect.check_remote: true`)
  performs HEAD requests / etag checks against remote URIs (S3
  manifests, checkpoints, etc.) and reports availability and size.
- Without `--check-remote`, missing-remote-artifact errors surface at
  runtime, not at inspect time. This is intentional.

CLI examples:

```bash
dojo inspect config experiment=ifcb/species_baseline training.batch_size=64
dojo inspect dataset data=ifcb/species_manifest output=./inspect_outputs/species_manifest.parquet
dojo train experiment=ifcb/species_baseline
dojo infer embeddings experiment=ifcb/species_baseline checkpoint=./runs/baseline/checkpoints/best.ckpt
dojo eval representation experiment=ifcb/dinov2_repr_eval
dojo ensemble candidates experiment=ifcb/ensemble_candidates ensemble_outputs.manifests.dir=./shared_manifests
dojo ensemble experiment=ifcb/ensemble_search ensemble.candidates.manifest_uri=./shared_manifests/ifcb_candidates.json
```

## Task Types

- `task.type` is the axis that determines what `dojo train` does. Supported
  task types in the initial implementation:
  - `supervised`
  - `ssl`
  - `snapshot_ensemble`
- `task.type: snapshot_ensemble` is a single-command convenience that
  internally orchestrates two steps against one shared run directory:
  1. supervised training with a snapshot-cycle scheduler (cosine warm
     restarts producing one checkpoint per cycle);
  2. ensembling the just-finished snapshot checkpoints using the
     ensemble pipeline.
- The combined `task.type: snapshot_ensemble` run shares the run
  directory with both steps. Snapshot checkpoints land in
  `<training_outputs.dir>/checkpoints/`; ensemble artifacts land under
  the configured `ensemble_outputs` subdirectories. The default
  ensemble result and figure directories are `ensemble_results/` and
  `ensemble_figures/`, controlled by `ensemble_outputs.results.dir`
  and `ensemble_outputs.figures.dir`.
- The old top-level command `dojo train-snapshot-ensemble` is **dropped**.
  The canonical invocation is:

  ```bash
  dojo train experiment=ifcb/snapshot_experiment
  ```

  where the experiment config sets `task.type: snapshot_ensemble`.
- "Re-run ensembling against a historical training run" is **not** a
  `task.type: snapshot_ensemble` invocation. It is a regular
  `dojo ensemble` invocation with a `run_checkpoints` candidate source
  pointing at the historical run directory.
- A `task.type: snapshot_ensemble` config must include both a
  `training:` block (with a snapshot-cycle scheduler) and an
  `ensemble:` block (selection strategy + combine modes). Pydantic
  validation should enforce both blocks are present and that the
  scheduler is one capable of producing snapshot candidates.

Example:

```yaml
task:
  type: snapshot_ensemble

training:
  max_epochs: 300

scheduler:
  name: cosine_warm_restarts
  first_cycle_epochs: 50
  cycle_mult: 1.0
  max_lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_epochs: 5

checkpointing:
  save_cycle_snapshots:
    enabled: true
    at_cycle_end: true

ensemble:
  selection:
    strategy: cycle_end_snapshots
    use_all_cycles: true
  inference:
    combine:
      classification: probabilities_mean
```

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

CSV/Parquet manifest example:

```yaml
data:
  backend: parquet_manifest
  manifest_uri: s3://datasets/ifcb/species_manifest.parquet
  sample_id_column: roi_id
  image_uri_column: image_uri
  split_column: split
  source_extra_columns: [cruise_id, cast_id, instrument_id]
  tabular_feature_columns:
    - depth_m
    - temperature_c
    - salinity_psu
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      missing_policy: error
    biovolume:
      column: biovolume_um3
      type: regression
      missing_policy: drop_sample
```

IFCB bins example:

```yaml
data:
  backend: ifcb_bins
  manifest_uri: s3://datasets/ifcb/bin_manifest.parquet
  bin_id_column: bin_id
  bin_uri_column: bin_uri
  split_column: split
  exclude_patterns: [bad, skip, beads, temp, data_temp]
  shuffle_buffer_size: 1000
  length:
    mode: cached
    cache_uri: s3://datasets/ifcb/cache/bin_lengths.parquet
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      missing_policy: error
```

## Model, Heads, and Objectives

- The old `multilabel` module is actually multi-head multiclass and should not
  be directly ported.
- Move the old multi-head multiclass module to `dojo_deprecated` for reference.
- Move all current `src/dojo` code to `src/dojo_deprecated`, kept for reference but will ultimately be removed. 
- Reserve `multilabel_classification` for true multi-hot multilabel problems.
- Use the new multi-head paradigm for current and future head types.
- Heads reference logical data targets, not raw manifest columns.
- Objectives bind heads to losses, metrics, and weights.

Ordinal naming:

- Head type: `ordinal_classification`. (Not `ordinal_regression` — the
  head predicts a discrete ordered bin, not a continuous value.)
- Supported ordinal losses: `coral`, `corn`, `ordinal_cross_entropy`.
- Result `record_type` values: `ordinal_output` for native ordinal
  heads; `ordinal_probe_prediction` for ordinal probes.
- Result columns: keep both `ordinal_logits` (raw cumulative logits
  for CORAL/CORN) and `probabilities` (per-bin probabilities). Both
  populated; downstream tools can use whichever is meaningful for the
  loss/decode rule. For CORAL/CORN, `probabilities` are derived by
  differencing the cumulative probabilities decoded from
  `ordinal_logits`; the writer is responsible for this derivation so
  consumers always see per-bin probabilities in the `probabilities`
  column regardless of loss family.

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

Multi-head example:

```yaml
data:
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      missing_policy: error
    life_stage:
      column: life_stage_idx
      type: ordinal_classification
      missing_policy: error
    biovolume:
      column: biovolume_um3
      type: regression
      transform: log1p
      missing_policy: drop_sample

model:
  tabular:
    enabled: true
    columns: [depth_m, temperature_c, salinity_psu]
    encoder:
      type: mlp
      hidden_dims: [64, 64]
    fusion:
      type: concat_mlp
      output_dim: 512
  heads:
    species:
      type: multiclass_classification
      target: species
      num_classes: 42
    life_stage:
      type: ordinal_classification
      target: life_stage
      num_classes: 5
    biovolume:
      type: regression
      target: biovolume

objectives:
  species:
    head: species
    loss: cross_entropy
    metrics: [macro_f1, per_class_f1]
    weight: 1.0
  life_stage:
    head: life_stage
    loss: ordinal_cross_entropy
    metrics: [mae, quadratic_weighted_kappa]
    weight: 0.5
  biovolume:
    head: biovolume
    loss: smooth_l1
    metrics: [mae, rmse, r2]
    weight: 0.25
```

## Backbones

- `model.backbone.source: timm` is **functional** in the initial
  implementation (not deferred).
- Supported user-facing backbone sources in the initial implementation:
  - `torchvision`
  - `timm`
  - `checkpoint`
- timm support is gated by the `timm` optional extra (`pip install
  dojo[timm]`). The schema accepts `source: timm` regardless of install;
  the runtime raises a clear error if `timm` is not installed.
- DINOv2 (Lightly) is allowed to use timm ViT backbones internally and
  through the public `source: timm` selector. There is no separate
  "DINOv2-only timm path."
- Inception-style auxiliary-logit handling remains out of scope for the
  generic backbone path in the initial implementation (unchanged from
  the existing design doc).
- `model.backbone.source: lightly` is not introduced as a public
  backbone source. Lightly remains an SSL framework implementation
  detail; SSL configs select it via `ssl.framework: lightly`. The
  backbone for an SSL run is still selected via
  `model.backbone.source: timm|torchvision|checkpoint` as for
  supervised training.
- Supervised transfer learning from an SSL-pretrained encoder is not
  a new task type. It is plain `task.type: supervised` with
  `model.backbone.source: checkpoint` and `checkpoint_uri` pointing
  at the SSL encoder export. Freeze policy, embedding adapter, and
  head are configured exactly as for any supervised training run.

## Dependency Plan

- Use a lightweight base install plus optional extras.
- Base install should support config validation, inspection, storage/result
  access, schema handling, and artifact introspection without requiring Torch.
- [amplify-db-utils](https://github.com/WHOIGit/amplify-db-utils) is a core dependency.
- [amplify-storage-utils](https://github.com/WHOIGit/amplify-storage-utils) remains a core dependency.
- Training, SSL, Aim, MLflow, ONNX, and IFCB support should live behind extras.

Final optional-extra layout:

- `train` — Torch stack required to run training/inference.
- `timm` — first-class `model.backbone.source: timm` support.
- `ssl` — Lightly only. SSL DINOv2 ViT backbones come from the `timm`
  extra; install both for SSL.
- `ifcb` — `ifcbkit` for the `ifcb_bins` dataset backend. Usable
  outside SSL (supervised IFCB training, holdout eval, inspect).
  Pulls `ifcbkit[s3]` so IFCB-via-S3 works when `s3` is also
  installed.
- `repr_eval` — UMAP, HDBSCAN, and scikit-learn for representation
  evaluation (projections, clustering, linear/ridge probes, baseline
  metrics). Usable against supervised encoders too, not SSL-only.
- `aim` — Aim logger sink.
- `mlflow` — MLflow logger sink (schema present; runtime stubbed in
  the initial implementation).
- `onnx` — ONNX export and runtime.
- `s3` — S3 capability for storage (`amplify-storage-utils[s3]`).
- `all` — convenience meta-extra that pulls every functional extra
  above.
- `dev` — testing and developer tooling (pytest, pytest-cov, ruff,
  mypy, pre-commit).

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
  "numpy",
  "pandas",
  "pillow",
]

timm = ["timm"]

ssl = ["lightly"]

ifcb = ["ifcbkit[s3]"]

repr_eval = [
  "umap-learn",
  "hdbscan",
  "scikit-learn",
]

aim = ["aim"]
mlflow = ["mlflow"]
onnx = ["onnx", "onnxruntime"]

s3 = ["amplify-storage-utils[s3]"]

all = [
  # union of functional extras: train, timm, ssl, ifcb, repr_eval, aim, onnx, s3
  "image_classifier_dojo[train,timm,ssl,ifcb,repr_eval,aim,onnx,s3]",
]

dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]
```

Common install recipes:

- Schema / config inspection / result reading only:
  `pip install image_classifier_dojo`
- Supervised training: `pip install image_classifier_dojo[train]`
- Supervised + timm backbones:
  `pip install image_classifier_dojo[train,timm]`
- SSL DINOv2 with representation evaluation:
  `pip install image_classifier_dojo[train,timm,ssl,repr_eval]`
- IFCB bins over S3:
  `pip install image_classifier_dojo[train,ifcb,s3]`
- All functional extras: `pip install image_classifier_dojo[all]`
- Local development:
  `pip install -e .[all,dev]`

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

Result writer config example:

```yaml
training_outputs:
  results:
    enabled: true
    backend: amplify_db_utils
    dir: results  # relative to training_outputs.dir, which is in turn resolved from training_outputs.dir_template
    format: parquet
    partition_by: [stage, record_type, epoch]
    dictionary_encode:
      enabled: true
      columns:
        - split
        - stage
        - record_type
        - head_name
        - embedding_kind
        - prediction_label
        - resize_width_px
        - resize_height_px
    write_metadata_json: true
```

The result write path defaults to `<training_outputs.dir>/results/`
under the resolved training-output directory; no explicit `uri:` is
required. The same `results:` sub-block exists under
`ensemble_outputs:` for ensemble result writing, defaulting to
`<ensemble_outputs.dir>/ensemble_results/` unless overridden by
`ensemble_outputs.results.dir`.

## IDs and Hashes

Identity and content-equality are separate concerns and are represented
separately:

- A `*_hash` is derived deterministically from the object's content. Two
  objects with equal hashable content produce the same hash. Hashes are
  what compatibility checks compare on.
- A `*_id` is either manually set (human-chosen) or generated as a
  human-readable name. When generated, the id is a **seedname**: a
  coolname produced by seeding `random.Random` with the corresponding
  `*_hash`. Same hash → same seedname, so seednames are reproducible
  identifiers, not random ones.
- Explicit exceptions:
  - `run_id` identifies one invocation and may be generated from a
    fresh, unseeded template token such as `{coolname}`;
  - `dataset_id` is only a dataset self-name when one exists; it does
    not fall back to a seedname;
  - `sweep_id` may be generated like `run_id` for a concrete sweep
    invocation, but falls back to a seedname from `sweep_hash` when no
    explicit or template value is configured.

Some objects have both a hash and an id; some have only one. See the
table below.

### Storage and truncation

- Record **full hashes** by default (e.g. full SHA-256 hex). Truncation
  is per-use-site and explicitly documented; the only standard
  truncation is the **first 6 hex characters** embedded in checkpoint
  filenames.
- Result-row provenance columns carry both the `*_id` and the `*_hash`
  where both exist, so downstream consumers can join either way.

### Canonicalizer

- Lives in `dojo.utils.artifact_hashing`.
- One JSON normalization recipe used by every hash function:
  - canonical UTF-8 JSON;
  - sorted object keys;
  - no insignificant whitespace;
  - lists preserved in source order;
  - floats rounded to **12 significant decimal digits** before
    serialization (well within float64 precision of ~15.95, absorbs
    roundtrip noise);
  - `NaN`, `+Infinity`, and `-Infinity` serialize as the literal
    strings `"NaN"`, `"Infinity"`, `"-Infinity"` (JSON-encoder-agnostic);
  - bytes/binary inputs hashed directly without JSON wrapping.
- Each `*_hash` function selects a specific subset of fields from the
  source object before canonicalization. The key-selection rules per
  hash type are listed below; the **exact field lists are TBD** and
  must be settled before implementation (see action items).

### Per-field table

| Field | Kind | Derivation | Default when not set |
| --- | --- | --- | --- |
| `run_id` | id only | manual override, else a template-string render. The `{coolname}` template token expands to a fresh (unseeded) coolname per run — distinct from the **seedname** mechanism used for `*_id` fields paired with a `*_hash`. If `run_id` is unset or empty, it falls back to a coolname | template default like `{experiment.name}-{timestamp}-{job_num}` or `{coolname}`; template is configurable and may use post-resolve config values, timestamps, and helpers like `coolname` |
| `config_id` | id (paired with `config_hash`) | manual override, else seedname from `config_hash` | seedname |
| `config_hash` | hash | canonical hash of the resolved config, **excluding** runtime-resolved values, output paths, `output_root`, and all `*_outputs` blocks | always derived |
| `dataset_id` | id only | the dataset's self-name when the manifest provides one | null when the dataset does not self-name (no seedname fallback) |
| `dataset_hash` | hash | URI + size + etag/last-modified (or full content hash when locally accessible and cheap), plus the data backend type. When size/etag are unavailable (e.g. some `class_folder` inspect sources), fall back to URI-only hashing and set `dataset_hash_provenance: uri_only` in the metadata sidecar | always derived |
| `checkpoint_hash` | hash | SHA-256 of the `.ckpt` file bytes | always derived |
| `model_id` | id (paired with `model_hash`) | manual override on the export command, else seedname from `model_hash` | seedname |
| `model_hash` | hash | SHA-256 of the exported `.pt` / `.onnx` file bytes | always derived |
| `ensemble_id` | id (paired with `ensemble_hash`) | manual override on the ensemble command, else seedname from `ensemble_hash` | seedname |
| `ensemble_hash` | hash | canonical hash of the ensemble manifest JSON (selected members + combine config + selection config) | always derived |
| `sweep_id` | id (paired with `sweep_hash`) | manual override, else a template-string render with the same semantics as `run_id` (for example `{coolname}`). When unset, falls back to seedname from `sweep_hash` | seedname or `{coolname}` |
| `sweep_hash` | hash | canonical hash of the sweep definition (base config + sweep axes + value lists), **excluding** runtime-resolved values and output paths | always derived |
| `ensemble_member_id` | union column | for member-level result rows and partitioning. Value is the member's `checkpoint_hash` when the member is a checkpoint, or its `model_id` when the member is an exported model artifact. Always populated for member-level rows | derived per-row from the underlying member artifact |

There is no `run_hash`. Runs are by construction per-invocation and are
identified by `run_id` alone. Two runs with identical configs share
`config_id` / `config_hash` but never share `run_id`.

There is no `checkpoint_id`. Checkpoints are identified by
`checkpoint_hash` plus their filename, which embeds the first 6 hex
characters of the hash.

### Checkpoint filename convention

Checkpoint filenames embed the first 6 hex characters of
`checkpoint_hash` between the descriptive stem and the extension:

```text
{stem}.{first6_hex}.{ext}
```

Examples:

```text
loss-1.23_epoch-003_f1score-88.7ff91a.ckpt
last.a1b2c3.ckpt
snapshot_cycle-02_epoch-100.0f0f0f.ckpt
```

The first-6 prefix is a disambiguation aid for humans inspecting a
directory and is not a substitute for the full hash recorded in
metadata.

### Compatibility hashes (target_schema_hash, class_mapping_hash, etc.)

For compatibility-check hashes used by ensembling and result-table
joins, both the hash **and** the source sub-block are stored in the
metadata sidecar. Fast path: compare hashes for equality. Slow path:
when hashes differ, diff the source sub-blocks and present a
human-readable mismatch report.

Key-selection rules per hash (exact field lists TBD):

- `target_schema_hash`: head names, head types, `num_classes`, ordinal
  encoding/decoding rules, regression `output_dim`. **Excludes** loss
  type, loss params, objective weights, and metrics — those are
  training-time choices that do not affect output compatibility.
- `class_mapping_hash`: per-classification-head ordered list of
  `(index, label)` pairs.
- `model_config_hash`: backbone source/name/weights, freeze policy,
  embedding adapter shape, tabular encoder shape, fusion config, and
  head shapes. **Excludes** optimizer, scheduler, training, logging,
  `output_root`, and `*_outputs` blocks.
- `preprocessing_hash`: transform pipeline ordering and parameters,
  image mode, normalization mean/std, resize/bucket definitions, and
  tabular feature normalization stats. **Excludes** training-only
  augmentation toggles unless they alter the inference-time preprocess
  contract.

### Open work before implementation

- Freeze the exact field lists each compatibility hash consumes. The
  narrative rules above ("includes head type, num_classes, …;
  excludes loss type, weight, metrics") are the binding intent;
  implementers mark the contributing fields in the Pydantic schemas
  (e.g. via a `compatibility_hash_includes=True` field flag or an
  equivalent registry in `dojo.utils.artifact_hashing`). Keeping the
  field list in code next to the schema definitions is preferred
  over freezing it in static documentation.

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
target_internal
prediction_value_internal
prediction_uncertainty
```

Target/prediction naming convention (applies to regression, ordinal, and any
head that uses a `target_transform`):

- `target` and `prediction_value` are the **external** values, in the
  original physical units the user thinks in. `prediction_value` is the
  model output after inverse-transforming back from model-space.
- `target_internal` and `prediction_value_internal` are the **model-space**
  values: the transformed target that was actually fed into the loss, and
  the raw pre-inverse-transform model output.
- When no `target_transform` is configured, `target_internal` equals
  `target` and `prediction_value_internal` equals `prediction_value`.
  Writers may either populate both columns identically or leave the
  `_internal` columns null in that case; the chosen behavior must be
  documented in the metadata sidecar.
- `prediction_uncertainty` is reported in the same space as
  `prediction_value` (external units) unless the head explicitly defines
  uncertainty in transformed space, in which case a parallel
  `prediction_uncertainty_internal` column may be added.
- The sidecar `_metadata.json` per-record-type block names the
  `target_transform` so downstream tools can re-derive the relationship
  between external and internal columns.

Metadata sidecar sketch:

```json
{
  "schema_name": "dojo.supervised_results",
  "schema_version": "1.0.0",
  "created_by": "dojo",
  "run_id": "ifcb-green-river",
  "record_types": {
    "sample_metadata": {
      "description": "One row per evaluated sample.",
      "source_extra_json": {
        "columns": ["cruise_id", "cast_id", "instrument_id"]
      },
      "tabular_features_json": {
        "columns": ["depth_m", "temperature_c", "salinity_psu"]
      }
    },
    "classification_output": {
      "heads": {
        "species": {
          "target": "species",
          "labels": ["A", "B", "C"],
          "label_mappings": {"0": "A", "1": "B", "2": "C"}
        }
      }
    },
    "regression_output": {
      "heads": {
        "biovolume": {
          "target": "biovolume",
          "target_transform": "log1p"
        }
      }
    }
  }
}
```

## Artifact Layout

- Each `*_outputs` block owns a set of sub-directories under its
  resolved `dir`. The sub-directory names are stable; whether each is
  written depends on the corresponding sub-block being enabled in
  config.
- `metrics/` holds numeric aggregates only: per-split metric JSON
  and tabular confusion-matrix data (JSON or CSV). No rendered
  images.
- `figures/` is a **configurable output block** (not just a
  directory). Config specifies which kinds of plots to render
  (training curves, confusion-matrix heatmaps, UMAP/t-SNE/PCA
  scatter, retrieval panels, calibration, ensemble comparison
  charts) and plot styling. Output files are PNG / SVG / HTML.
- The metrics-vs-figures split is the same across `training_outputs`,
  `ensemble_outputs`, and `sweep_outputs`.
- Do not add a generic training-run `data/` folder; input dataset information is
  covered by configs and resolved config artifacts.
- Use `ensemble_outputs.manifests.dir` for ensemble candidate
  manifests; it defaults to
  `{ensemble_outputs.dir}/ensemble_manifests`. Shared candidate
  manifests written outside a normal run directory are handled by
  setting `output_root`, `ensemble_outputs.dir`, or
  `ensemble_outputs.manifests.dir` for the `dojo ensemble candidates`
  invocation.
- Use JSON for ensemble manifests, not Parquet.

Per-output-block contents under each `*_outputs.dir`:

```text
training_outputs.dir/
  config/
  checkpoints/
  exports/
  metrics/
  figures/
  results/

ensemble_outputs.dir/
  config/
  exports/
  metrics/
  ensemble_figures/
  ensemble_results/
  ensemble_manifests/
  ensemble_members/

sweep_outputs.dir/
  config/
  exports/
  metrics/
  figures/
```

For a `task.type: snapshot_ensemble` run with
`ensemble_outputs.dir_template` defaulted to the training value, both
`training_outputs.dir` and `ensemble_outputs.dir` resolve to the
same path and the directory carries the union of both layouts:

```text
<training_outputs.dir>/  # same path as <ensemble_outputs.dir> for snapshot_ensemble
  config/
  checkpoints/        # training_outputs
  exports/            # union; both blocks write here
  metrics/            # union; namespaced by block when needed
  figures/            # union; namespaced by block when needed
  results/            # union; namespaced by block when needed
  ensemble_results/   # default ensemble_outputs.results.dir
  ensemble_figures/   # default ensemble_outputs.figures.dir
  ensemble_manifests/ # default ensemble manifest dir
  ensemble_members/   # default ensemble member dir (optional)
```

These `ensemble_*` subdirectory names are defaults, not hard-coded
paths. They are controlled by the corresponding `ensemble_outputs`
sub-blocks, such as `ensemble_outputs.results.dir` and
`ensemble_outputs.figures.dir`.

`ensemble_members/` is optional and only used when `dojo ensemble`
materializes member artifacts locally.

When training and ensemble outputs share a directory, writers
namespace their files within the shared sub-directories
(e.g. result files include `stage=train_validation` vs
`stage=ensemble_eval` partitions; metrics files include the producing
block in their filenames).

## Result Partitioning

- Support configurable Parquet partitioning via
  `training_outputs.results.partition_by` and
  `ensemble_outputs.results.partition_by`.
- Partition fields may include values like:
  - `stage`
  - `epoch`
  - `ensemble_member_id`
  - `record_type`
- `ensemble_member_id` is a union column whose value is the member's
  `checkpoint_hash` (when the member is a checkpoint) or `model_id`
  (when the member is an exported model). Keeping one partition
  column with mixed-source values avoids null-only partitions and
  keeps `partition_by` clean.
- Include `sweep_id` when a row is produced as part of a Hydra sweep.
- Distinguish:
  - `split`: source dataset split, such as `train`, `val`, `test`, `unlabeled`
  - `stage`: process that produced the row, such as `train_validation`,
    `holdout_eval`, `infer`, `representation_eval`, `ensemble_eval`

Partitioning examples:

```yaml
training_outputs:
  results:
    partition_by: [stage, record_type]
```

```yaml
training_outputs:
  results:
    partition_by: [stage, epoch, record_type]
```

```yaml
ensemble_outputs:
  results:
    partition_by: [stage, record_type, ensemble_member_id]
```

## Runtime Controls

Keep these runtime controls:
- `seed`
- `fast_dev_run`
- `precision`
- `num_workers`
- `autobatch`

Early stopping is a training-loop behavior and belongs under `training`,
not `runtime`.

Training early-stopping example:

```yaml
training:
  early_stopping:
    enabled: true
    monitor: val/loss
    mode: min
    patience: 10
```

ONNX export does not need to be a runtime config item. It should be handled by
the config-driven task orchestration/export system outside the training loop.

Runtime/preflight example:

```yaml
runtime:
  seed: 123
  precision: bf16-mixed
  num_workers: 8
  fast_dev_run: false
  autobatch:
    enabled: true
    mode: binsearch
  preflight:
    enabled: true
    checks:
      empty_train_classes: error
      empty_eval_classes: warn
      non_contiguous_class_indices: error
      imbalance_ratio_gt:
        severity: warn
        threshold: 20.0
```

## Testing Policy for Deferred Features

For features that are stubbed in the initial implementation, tests
assert the stub behavior and nothing else. The stub itself is the
contract.

- Each stubbed runtime path has exactly one test that:
  1. constructs a config that exercises the deferred feature;
  2. invokes the runtime path;
  3. asserts a `NotImplementedError` is raised;
  4. asserts the error message names the deferred feature and points at
     the deferred-feature backlog.
- Stubbed features get no schema-only tests, no inspect-output
  enumeration tests, and no scaffolded runtime tests.
- When a deferred feature is later unstubbed, the stub-assertion test
  is deleted and replaced with real functional tests.

Stubbed features in the initial implementation:

- MLflow logger sink (`training_outputs.logging.sinks[].type: mlflow`);
- non-`dino_v2` SSL methods (`ssl.method: simclr | vicreg | pmsn | dino`);
- weight-space ensembles (model soup, greedy soup, uniform soup, SWA,
  EMA);
- weighted ensemble types;
- broad automatic registry-based cross-run discovery;
- `prediction_trimmed_mean` and weighted combine modes;
- WebDataset backend.

Functional features that **do** get full functional tests when the
relevant extra is installed:

- `model.backbone.source: timm` (functional per the Backbones section);
- `model.backbone.source: torchvision` and `checkpoint`;
- Aim logger sink;
- `dino_v2` SSL via Lightly;
- `ifcb_bins` dataset backend (with `[ifcb]` extra);
- UMAP, t-SNE, HDBSCAN, regression/ordinal/classification probes (with
  `[repr_eval]` extra);
- ONNX export (with `[onnx]` extra);
- S3 storage (with `[s3]` extra);
- snapshot ensembles and prediction-space ensembles (selection
  strategies `all`, `best_candidate`, `top_k`,
  `greedy_forward_selection`, `cycle_end_snapshots`; combine modes per
  Ensemble Architecture section).

## Logging and Diagnostics

- Keep Aim as a functional logging sink.
- Keep Aim diagnostic figures.
- MLflow config schema may exist, but runtime implementation will remain a clear
  stub for the initial implementation. As a deferred feature, it gets
  put in the design-doc Deferred Features appendix.
- Experiment logging applies only to model training. `dojo ensemble
  candidates` does not initialize experiment logging.
- Multi-sink composition is supported in the initial implementation:
  `local`, `aim`, and `local + aim` are all functional via a
  `CompositeExperimentLogger`.
  No artificial cap on sink count, but three-or-more-sink configs are
  not specifically exercised in tests.
- Multi-sink behavior involving MLflow inherits MLflow's stubbed
  runtime per the Testing Policy for Deferred Features section.

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
  method: dino_v2
  framework: lightly
```

- `dino_v2` through Lightly is functional in the refactor.
- see https://docs.lightly.ai/self-supervised-learning/examples/dinov2.html for example code. 
- SimCLR, VICReg, PMSN, and original DINO are intentionally removed from the
  initial implementation runtime. Deferred to appendix.
- Preserve config stubs for deferred SSL methods with clear errors.
- Rename SSL evaluation to representation evaluation.
- `representation_eval` should support both training-integrated evaluation and
  standalone `dojo eval representation`.
- `representation_eval` is **not SSL-only**. It is a top-level config
  group usable with any task that produces image embeddings,
  including `task.type: supervised` and `task.type: ssl`. A
  supervised run may schedule representation evaluation against its
  own encoder during training, and the standalone
  `dojo eval representation` command works against checkpoints from
  either task type.
- UMAP and t-SNE are optional-extra features but not deferred.
- HDBSCAN is optional-extra-backed but not deferred.
- Regression and ordinal probes are included and not deferred.
- Supervised fine-tuning from an SSL pretrained backbone belongs under
  supervised transfer learning.
- Supervised probes stay in SSL/representation-evaluation design.

SSL training example:

```yaml
task:
  type: ssl

ssl:
  method: dino_v2
  framework: lightly
  image_size: 224
  projection_dim: 65536

model:
  backbone:
    source: timm
    name: vit_small_patch14_dinov2

representation_eval:
  schedule:
    every_n_epochs: 5
    on_fit_end: true
  embeddings:
    enabled: true
    split: val
  projections:
    methods: [umap, tsne]
  clustering:
    methods: [hdbscan]
  probes:
    classification:
      enabled: true
      heads: [species]
    regression:
      enabled: true
      targets: [biovolume]
    ordinal:
      enabled: true
      targets: [quality_grade]
```

Standalone representation evaluation example:

```bash
dojo eval representation \
  experiment=ifcb/dinov2_repr_eval \
  model.backbone.checkpoint_uri=./runs/dinov2/checkpoints/best.ckpt \
  representation_eval.embeddings.split=holdout
```

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

- Candidate discovery in the initial implementation is **limited to explicit sources**:
  explicit artifact lists, run-directory globs, result-URI globs, and
  pre-built candidate manifests. Broad automatic registry-based
  cross-run discovery is deferred (see Deferred Features Appendix).
- Use `dojo ensemble candidates` for discovery, compatibility inspection, cache
  assessment, and manifest creation.
- `dojo ensemble candidates` is an artifact-inspection/manifest-writing
  command and does not initialize experiment logging.
- `dojo ensemble candidates` writes candidate manifests under
  `ensemble_outputs.dir/ensemble_manifests/` by default. To write a
  shared manifest outside a normal run directory, set the manifest
  directory directly:

  ```bash
  dojo ensemble candidates experiment=ifcb/candidate_search \
    ensemble_outputs.manifests.dir=./shared_manifests
  ```

  The manifest filename is produced by the candidate-manifest writer
  under that directory unless the relevant `ensemble_outputs.manifests`
  config overrides it.
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

Candidate discovery example:

```yaml
ensemble:
  candidates:
    sources:
      - type: run_dir_glob
        uri_glob: s3://dojo-runs/ifcb_species/*/
      - type: explicit
        artifacts:
          - run_dir: ./runs/resnet50_a
          - result_uri: ./runs/convnext_b/results
    metadata_resolution:
      policy: cascade
      order:
        - resolved_config
        - result_metadata
        - checkpoint
        - exported_model
      drift_check:
        enabled: true
        compare: [resolved_config, checkpoint, result_metadata]
  target:
    split: val
    dataset_id: ifcb_species_v4   # self-named dataset; use dataset_hash instead when the dataset does not self-name
  source_policy: inference_as_needed
```

Ensemble run example:

```yaml
ensemble:
  candidates:
    manifest_uri: ./shared_manifests/ifcb_candidates.json
  target:
    split: holdout
  source_policy: strict_no_inference
  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_members: 8
  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_median
      ordinal: ordinal_probabilities_mean

ensemble_outputs:
   ...
```

## Ensemble Outputs

- All ensemble output configuration lives under the top-level
  `ensemble_outputs:` block, peer to `training_outputs:` and
  `sweep_outputs:` (see Config Structure section).
- `ensemble_outputs.dir_template` controls where ensemble artifacts
  land. For a `task.type: snapshot_ensemble` run the recommended
  default is to match `training_outputs.dir_template` so training
  and ensemble outputs share one run directory.
- Sub-blocks: `results`, `export`, `metrics`, `figures`,
  `manifests`, and `members`. Their on-disk sub-directory names
  follow the per-block layout in the Artifact Layout section
  (`ensemble_results/`, `exports/`, `metrics/`, `ensemble_figures/`,
  `ensemble_manifests/`, and optional `ensemble_members/` by default).
- Optional member materialization to `ensemble_members/` is allowed only for
  `dojo ensemble` runs.
- Local member files may be symlinked.
- Remote member files may be cached through `storage` config and then symlinked.
- Ensemble metrics and figures should compare member best/final metrics against
  the created ensemble model.
- When training and ensemble outputs share a run directory, writers
  namespace per-row data by `stage` (`train_validation` vs
  `ensemble_eval`) so canonical result Parquet files coexist
  without collision.

## Hydra Sweeps and Sweep Outputs

- Use Hydra sweeps to explore ensemble strategy/combine-mode combinations.
- Top-level `sweep_outputs:` is the peer output block for sweep-level
  aggregation. It has `dir_template`, `export`, `metrics`, and
  `figures` sub-blocks but **no** `results` sub-block (per-row data
  comes from the underlying per-job `training_outputs` /
  `ensemble_outputs`).
- `sweep_outputs.dir_template` controls sweep aggregation output
  location. Per-job training/ensemble outputs continue to land under
  their own `*_outputs.dir_template`.
- `sweep_outputs.metrics` and `sweep_outputs.figures` control which
  aggregations are produced.

Example sweep output needs:

- compare best/final F1 across best/final epoch for output model trainings;
- compare per-class F1 across best/final epoch for model trainings;
- write comparison metrics CSV/JSON/Parquet;
- write comparison figures.

Sweep output example:

```yaml
runtime:
  sweep_id: "{coolname}"

model:
  backbone:
    source: torchvision
    name: resnet50  # swept: resnet50, efficientnet_b0, convnext_tiny

training:
  batch_size: 32  # swept: 32, 64

optimizer:
  lr: 0.0003  # swept: 0.0003, 0.0001

output_root: ./runs

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.backbone.name:slug}/bs{training.batch_size:03}/lr{optimizer.lr:slug}/

sweep_outputs:
  dir_template: >-
    {experiment.name}/sweep_results/{runtime.sweep_id}
  enabled: true
  collect:
    - metric: val/species/macro_f1
      source: best
      mode: max
    - metric: val/species/per_class_f1
      source: best
      mode: max
  metrics:
    summary_csv: true
    summary_json: true
    summary_parquet: true
  figures:
    metric_rankings: true
    per_class_heatmap: true
```

Hydra ensemble sweep example:

```bash
dojo ensemble -m \
  experiment=ifcb/ensemble_search \
  ensemble.selection.strategy=top_k,greedy_forward_selection \
  ensemble.inference.combine.classification=probabilities_mean,logits_mean
```

## Export

- ONNX export remains supported but is outside runtime training config.
- Export is explicit through `training_outputs.export`,
  `ensemble_outputs.export`, `sweep_outputs.export`, the
  `dojo export` command, or task-orchestration config.
- Export artifact `type` enum values in config: `torchscript`, `onnx`.
  There is no `pt` type. `.pt` is a file extension that TorchScript
  artifacts use by default; the `type` field names the artifact
  format, not its filename suffix.
- State-dict-only export is not an initial-implementation artifact type. Pickled
  state dicts are training-internal artifacts produced by Lightning
  checkpointing (`.ckpt`); portable exports go through `torchscript`
  or `onnx`.

Export config example:

```yaml
training_outputs:
  export:
    enabled: true
    artifacts:
      - type: torchscript
        name: model.pt
        source: best_checkpoint
      - type: onnx
        name: model.onnx
        source: best_checkpoint
        opset: 18
        dynamic_axes: true
```

The same `export:` sub-block shape applies under `ensemble_outputs:`
and `sweep_outputs:`. Each writes into its own `exports/`
sub-directory under the resolved `*_outputs.dir`.

## Deferred Features Appendix

Move deferred features out of the main design flow and into an appendix.

Deferred features include:

- SimCLR, VICReg, PMSN, and original DINO runtime implementations;
- HDF / HDF5 derived result exports (`.h5` metrics rollups,
  `results.h5`, and the corresponding `hdf` extra and `h5py`/`tables`
  dependencies);
- MLflow runtime implementation;
- model soup, greedy soup, uniform soup;
- SWA and EMA workflows;
- weighted ensemble combine modes;
- `prediction_trimmed_mean`;
- generic multimodal/multibranch fusion outside `model.tabular.fusion`;
- `improv` export/integration;
- Prefect flows;
- WebDataset;
- Bayesian/AutoML HPO;
- broad automatic registry-based cross-run discovery.

## Remaining Revision Action Items

- Update `REFACTOR-DESIGN-DOC.md` root config examples to match the canonical
  top-level structure in this file.
- Replace every `ssl_eval` reference with `representation_eval`.
- Replace old command names and examples with the canonical `dojo train`,
  `dojo infer`, `dojo eval`, `dojo inspect`, and `dojo ensemble` families.
- Remove `dojo validate-config` and document `dojo inspect config`.
- Remove `dojo tools make-manifest` and fold its behavior into
  `dojo inspect dataset`.
- Remove old listfile compatibility from dataset sections.
- Rewrite dataset backend sections around `csv_manifest`, `parquet_manifest`,
  `parquet_images`, and `ifcb_bins`.
- Add explicit target/head/objective reference-chain documentation.
- Replace generic `model.fusion` examples with `model.tabular.fusion`.
- Rewrite dependency plan around lightweight core plus extras.
- Add `amplify-db-utils` to core dependency and results backend sections.
- Move `improv` integration/export to the deferred appendix.
- Rewrite canonical result schemas around Arrow list columns, sidecar
  `_metadata.json`, and configurable Parquet partitions.
- Add first-class configurable `figures` blocks and per-output-block
  directory layouts to run layout sections.
- Remove training-loop ONNX export behavior from runtime config.
- Add layered validation and `runtime.preflight` semantics.
- Rename SSL evaluation sections to representation evaluation.
- Update SSL section to functional Lightly DINOv2 and deferred stubs for
  SimCLR, VICReg, PMSN, and original DINO.
- Add UMAP, t-SNE, HDBSCAN, regression probes, and ordinal probes to
  representation evaluation.
- Rewrite ensemble architecture around candidate manifests, prediction-space
  combine modes, and selection strategies.
- Remove or defer weight-space ensembling, model soups, SWA, EMA, weighted
  combine modes, and `prediction_trimmed_mean`.
- Add `dojo ensemble candidates` documentation, including compatibility/cache
  reports and metadata resolution policy.
- Add Hydra path-template and sweep-output documentation.
- Move deferred/stubbed items out of the main design flow and into a deferred
  features appendix.
- Remove phase terminology from `REFACTOR-DESIGN-DOC.md`; use
  "initial implementation", "deferred-feature backlog", and
  "migration plan" language instead.
- Update §1.1 core goals to drop or qualify items that are now deferred or
  reframed (snapshot-bundling phrasing, HDF result export, MLflow as a
  runtime sink). timm remains a first-class backbone source per the
  Backbones section above.
- Replace the §2.2 CLI command list with the new canonical command families:
  `dojo train`, `dojo infer [predictions|embeddings]`,
  `dojo eval [holdout|representation]`,
  `dojo inspect [config|dataset|backbone|checkpoint]`,
  `dojo ensemble [candidates]`, and `dojo export`. Remove
  `dojo train supervised`, `dojo train ssl`, and
  `dojo train-snapshot-ensemble`; training paradigm is selected via
  `task.type` (see Task Types section).
- Rewrite §13.12 "Snapshot ensemble command" to reflect that snapshot
  ensembling is invoked as `dojo train` with
  `task.type: snapshot_ensemble`. The internal orchestration steps
  (train with snapshot scheduler, then ensemble against the
  just-finished checkpoints in the same run directory) are unchanged;
  only the entry point changes. Remove `train.skip=true` and move the
  "re-run ensembling against a historical run" example into the
  `dojo ensemble` documentation with a `run_checkpoints` candidate
  source.
- Rewrite §3 repo structure to match the new architecture: drop `tools/`,
  drop separate `ensemble/soup.py` / `ensemble/swa.py` modules (deferred),
  rename `tasks/eval/` and related modules to representation-evaluation,
  align `config_schemas/` files with the new top-level groups
  (`runtime`, `storage`, `output_root`, `training_outputs`,
  `ensemble_outputs`, `sweep_outputs`, `representation_eval`, etc.), and
  fold `dojo tools make-manifest` behavior into `cli/inspect.py`.
- Update §10.3 "Tabular metadata as model input" so `tabular`, `fusion`, and
  `embedding_adapter` are shown nested under `model:` rather than as
  top-level groups.
- Rewrite §20 migration plan to use new command and concept names:
  `dojo inspect config` (not `dojo validate-config`), no
  `dojo tools make-manifest`, `representation_eval` (not `ssl_eval`),
  drop listfile-porting language, and refer to the new `output_root`,
  `*_outputs`, and `storage` config structure.
- Correct the `dojo_deprecated` spelling everywhere it is referenced in
  the design doc (the doc already uses the correct form; flag this so any
  new edits do not regress).
- Apply the IDs and Hashes section against the design doc. Every
  reference to `config_id`, `dataset_id`, `model_id`, `checkpoint_id`,
  `ensemble_id`, `target_schema_hash`, `class_mapping_hash`,
  `model_config_hash`, and `preprocessing_hash` must align with the
  new identity-vs-hash split. Affected sections include §10.5
  (canonical result columns), §13.4 (candidate compatibility), §13.5
  (candidate manifest columns), §13.11 (snapshot metadata), §13.13
  (ensemble artifact format), §13.15 (ensemble result records), §15.3
  (export metadata), and §18 (Hydra sweep / sweep_id usage). Replace
  any uses of `checkpoint_id` with `checkpoint_hash` plus the
  filename convention. Add `sweep_id` / `sweep_hash` to the result
  schema and provenance columns where a row is produced as part of
  a sweep.
- Scrub the design doc (§8.2, §11, any SSL example configs) for any
  reference to `model.backbone.source: lightly`. Lightly is not a
  public backbone source; replace with `source: timm` (or
  `torchvision` / `checkpoint`) as appropriate.
- Update the §4 dependency plan to match the final optional-extra
  layout (Dependency Plan section above): split `ifcb` out of `ssl`,
  add `repr_eval`, fold `scikit-learn` from `train` into `repr_eval`,
  remove `pyifcb`, simplify `s3` to `amplify-storage-utils[s3]`, and
  add `all` and `dev` meta-extras. Update the README install
  examples accordingly.
- Apply the Testing Policy for Deferred Features section against
  §19. Rewrite §19.5 and §19.11 (and any other subsection that
  mentions testing a stubbed feature as functional) so each deferred
  feature has only a stub-assertion test, no schema-only or
  scaffolded runtime tests. Move full functional tests for timm,
  Aim, DINOv2, ifcb_bins, repr_eval, ONNX, S3, and supported
  ensemble strategies to their respective extras-gated test files.
- Rename §9.5 "Ordinal regression head" to "Ordinal classification
  head" and update the head `type` enum to `ordinal_classification`
  in §9.1 and every example. Confirm result columns retain both
  `ordinal_logits` and `probabilities` (per Model/Heads/Objectives
  section); update §10.5 if needed.
- Note in §8.6 (Checkpoint transfer learning) that transfer learning
  from an SSL-pretrained encoder is plain `task.type: supervised`
  with `model.backbone.source: checkpoint`. No new task type.
- Update §17 to state that multi-sink composition (`local + aim`) is
  functional in the initial implementation; three-or-more-sink configs
  are allowed but not specifically tested. MLflow-involving multi-sink
  configs inherit the MLflow stub.
- Update §15 Export so artifact `type` values are limited to
  `torchscript` and `onnx`. Remove `pt` as a config type
  everywhere; `.pt` remains the default TorchScript filename
  extension.
- Remove the HDF result export from the canonical results pipeline
  in §10.5 and §11.6. Remove the `hdf` optional extra from §4
  dependencies. Remove `metrics.h5` and `results.h5` from run-layout
  examples. Update §1.1 core goals to drop the HDF mention.
- Update §6.7 / §2 documentation of `dojo inspect config` so the
  default is offline schema-and-local validation, with an opt-in
  `--check-remote` flag for remote-availability checks.
- In §13 Ensemble Architecture, scope candidate discovery to
  explicit sources only (explicit artifact lists, run-directory
  globs, result-URI globs, pre-built manifests). Move any broad
  registry-based cross-run discovery language to the deferred
  appendix.
- Remove all "timm deferred / stubbed" language from the design doc.
  Affected sections include §1.2 "Implemented-as-stub", §8.4 timm
  backbones, §11 (any cross-reference), §17 / §19 testing language, and
  the §20 migration plan. timm should be presented as a
  functional first-class backbone source gated by the `timm` optional
  extra, with a clear runtime error when the extra is missing. Also
  remove timm from the deferred-feature backlog list.
