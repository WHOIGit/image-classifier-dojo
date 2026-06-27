
# Glossary

## Purpose

Single source of truth for terminology used across the design doc. Defines
config keys, identifiers, hashes, record taxonomies, and architectural concepts
referenced by every other file. When in doubt, the terms below win.

## Identifiers and hashes

- **`run_id`** — Identifier for one `dojo train` / `dojo eval` / `dojo ensemble`
  invocation. Either manually set, rendered from a template, or generated as a
  fresh (unseeded) `{coolname}`. There is no `run_hash`.
- **`config_id` / `config_hash`** — `config_hash` is a deterministic hash of
  the resolved config excluding runtime-resolved values, output paths,
  `output_root`, and the `*_outputs` blocks. `config_id` is either manually
  set or generated as a **seedname** from `config_hash` (a coolname produced
  by seeding `random.Random` with `config_hash`).
- **`dataset_id` / `dataset_hash`** — `dataset_hash` is a cheap, always-available
  identity that never reads image pixels: manifest content or URI plus
  size/etag/last-modified plus backend type; falls back to URI-only hashing
  with `dataset_hash_provenance: uri_only` when size/etag are unavailable.
  `dataset_id` is only present when the manifest provides a self-name; there
  is no seedname fallback.
- **`dataset_content_hash`** — a separate, optional true hash over all image
  bytes, recorded only when `dojo inspect dataset` runs a full pass
  (`--content-hash` / `--normalization`). It is integrity / drift verification
  and is never folded into `dataset_hash`, which must stay stable regardless
  of whether a full pass ran.
- **`checkpoint_hash`** — SHA-256 of the `.ckpt` file bytes. The first 6 hex
  characters appear in the checkpoint filename
  (`{stem}.{first6_hex}.{ext}`). There is no `checkpoint_id`.
- **`model_id` / `model_hash`** — Applies to exported portable model
  artifacts (`.pt` / `.onnx`). `model_hash` is SHA-256 of the exported file
  bytes; `model_id` is either manual or seedname.
- **`ensemble_id` / `ensemble_hash`** — `ensemble_hash` is canonical hash of
  the selected-ensemble identity block (selected members + selection +
  combine config), excluding candidate-audit metadata; `ensemble_id` is
  either manual or seedname.
- **`sweep_id` / `sweep_hash`** — `sweep_hash` is canonical hash of the
  sweep definition (base config + axes + value lists), excluding
  runtime-resolved values and output paths. `sweep_id` may be a template
  render (e.g. `{coolname}`) or seedname fallback from `sweep_hash`.
- **`ensemble_member_id`** — Union column for ensemble member-level
  result rows. Equals the member's `checkpoint_hash` (for checkpoint
  members) or `model_id` (for exported-model members). Populated only when
  `ensemble_result_scope=member`.
- **`ensemble_result_scope`** — Ensemble result-row scope. `member` means a
  retained selected-member prediction row; `ensemble` means the combined
  ensemble prediction row.
- **Compatibility hashes** — `target_schema_hash`, `class_mapping_hash`,
  `model_config_hash`, `preprocessing_hash`. Both the hash and the source
  sub-block are stored in `_metadata.json` so consumers can fast-compare on
  the hash and slow-compare via human-readable diff on mismatch. The
  canonical source field lists live in `06-results-artifacts-and-metadata.md`.

### Seednames vs. fresh coolnames

`{coolname}` as a template token expands to a **fresh** unseeded coolname per
invocation. A **seedname** is a coolname produced by seeding
`random.Random(*_hash)`, so the same hash always produces the same human-readable
name. Most `*_id` fields paired with a `*_hash` fall back to seednames;
`run_id` and `sweep_id` are exceptions that may use fresh `{coolname}` tokens.

### Storage and truncation

Full hashes are recorded by default (e.g. full SHA-256 hex). The only
standard truncation is the **first 6 hex characters** embedded in checkpoint
filenames. Result-row provenance columns carry both `*_id` and `*_hash` when
both exist.

## Top-level config groups

- **`experiment`** — Experiment-level metadata (name, etc.).
- **`task`** — Carries `task.type` which selects training behavior:
  `supervised`, `ssl`, or `snapshot_ensemble`.
- **`runtime`** — Process behavior: `seed`, `precision`, `num_workers`,
  `fast_dev_run`, `autobatch`, `preflight`, and the `run_id` / `sweep_id`
  templates.
- **`storage`** — Storage resolver config (local cache dir, etc.). Peer to
  `runtime`.
- **`data`** — Dataset backend + targets.
- **`transforms`** — Image transform pipeline.
- **`model`** — `image_input`, `tabular_input`, `embedding_adapter`,
  `heads`. `model.image_input.backbone` holds the image backbone config.
  `backbone.architecture` describes the module shape; `backbone.weights`
  describes initialization.
  Image and tabular embeddings concatenate implicitly when both inputs are
  enabled.
- **`objectives`** — Bind heads to losses, weights, metrics.
- **`ssl`** — SSL framework selector (functional method: `dino_v2` via
  Lightly).
- **`representation_eval`** — Replaces the old `ssl_eval`; applicable to any
  task that produces image embeddings, supervised or SSL.
- **`training`**, **`optimizer`**, **`scheduler`**, **`checkpointing`** —
  Top-level training config blocks.
- **`ensemble`** — Ensemble selection/combine/candidate config for the
  `dojo ensemble` family and `task.type: snapshot_ensemble`.
- **`sweep`** — Sweep-generation config. `sweep.mode: grid` is
  functional; `sweep.mode: bayesian` is a deferred schema slot. Sweep
  axes normalize into this block before concrete runs are expanded.
- **`output_root`** — Single filepath string used as the base for rendered
  `*_outputs.dir_template` values and bare-relative `*_outputs.dir` values.
- **`training_outputs`**, **`ensemble_outputs`**, **`sweep_outputs`** — Peer
  output-config blocks (see `03-configuration.md`).

## Sweep terminology

- **Grid sweep** — A sweep with `sweep.mode: grid`, where each entry in
  `sweep.grid` maps a target config path to a YAML list of values. The
  concrete runs are the cartesian product of those lists.
- **Batch run** — A batch-run-style grid sweep where the only intentional
  run-varying parameter is `runtime.seed`. Used to measure sensitivity
  to random initialization, data order, and other seeded behavior while
  holding model / training settings fixed.
- **Bayesian sweep** — A deferred `sweep.mode: bayesian` schema slot for
  search engines such as Optuna. Runtime support is stubbed in the
  initial implementation.
- **`sweep.active_run`** — Generated resolved-config metadata for one
  concrete run in a sweep. It records the realized sweep-axis values for
  that run and is not written in source configs.

## Path resolution

For any `dir` key:

- Absolute paths (`/folder`) are used as provided.
- `./folder` is resolved relative to the process CWD.
- Bare relative paths (`folder`) are resolved against the parent object's
  resolved `dir`. For top-level `*_outputs.dir`, the parent base is
  `output_root`. For sub-block values (e.g. `results.dir` under
  `training_outputs`), the parent base is `training_outputs.dir`.

`dir_template` uses Dojo-owned `{...}` template syntax with an **open**
token set: any dotted resolved-config path (e.g. `{experiment.name}`,
`{runtime.run_id}`, `{training.batch_size}`) plus a fixed set of special
tokens (`{coolname}`, `{timestamp}`, `{job_num}`, `{ensemble_id}`).
Tokens may carry a `:spec` suffix — the custom `:slug` filesystem-safe
formatter or any standard Python format spec (e.g.
`{model.image_input.backbone.architecture.name:slug}`,
`{training.batch_size:03}`). Resolved after config composition, validation,
and runtime-value generation. The full rules live in
`03-configuration.md`.

## Result taxonomy

- **`split`** — Source dataset split: `train`, `val`, `test`, `unlabeled`,
  `holdout`.
- **`stage`** — Process that produced a result row:
  `train_validation`, `holdout_eval`, `infer`, `representation_eval`,
  `ensemble_eval`.
- **`record_type`** — What columns are populated on a row:
  `sample_metadata`, `embedding`, `classification_output`,
  `regression_output`, `ordinal_output`, `nearest_neighbor`,
  `knn_prediction`, `classification_probe_prediction`,
  `regression_probe_prediction`, `ordinal_probe_prediction`,
  `cluster_assignment`, `projection`, `outlier_score`, `diagnostic`.
- **`head_name`** — Logical head identifier on output rows.
- **`embedding_kind`** — `image_embedding`, `tabular_embedding`,
  `fused_input_embedding`, `head_input_embedding`.

### External vs. internal target/prediction columns

- `target` / `prediction_value` are **external** (original units).
- `target_internal` / `prediction_value_internal` are **model-space**
  (post `target_transform`).
- When no `target_transform` is configured, writers may either populate both
  identically or leave the `_internal` columns null; the behavior is
  recorded in `_metadata.json`.

## Task types (`task.type`)

- **`supervised`** — Standard supervised training, single- or multi-head.
- **`ssl`** — Self-supervised training. Functional method: `dino_v2` via
  Lightly.
- **`snapshot_ensemble`** — Convenience type that runs supervised training
  with a snapshot-cycle scheduler and then ensembles the snapshot
  checkpoints, sharing one run directory.

## Head types

- `multiclass_classification`
- `binary_classification`
- `multilabel_classification` (reserved for true multi-hot multilabel)
- `regression`
- `ordinal_classification` (not `ordinal_regression`; the head predicts a
  discrete ordered bin)
- `distributional_regression` (deferred — see appendix P4.9)
- `count_regression` (deferred — see appendix P4.9)

## Backbone sources and weights

`model.image_input.backbone.architecture.source`: `torchvision`, `timm`.
`timm` is functional, gated by the `timm` extra. Checkpoint loading is
`model.image_input.backbone.weights.source: checkpoint`, not an
architecture source. Lightly is **not** a public backbone source; SSL
configs use `ssl.framework: lightly` but still select a backbone
architecture through one of the public architecture sources.

## Cross-References

- `03-configuration.md` — canonical config shape and path-resolution
  semantics.
- `06-results-artifacts-and-metadata.md` — full IDs/Hashes table and
  result schemas.
- `02-cli-and-task-types.md` — CLI surface and `task.type` semantics.
