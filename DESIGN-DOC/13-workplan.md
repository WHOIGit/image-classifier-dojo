
# 13. Workplan

## Purpose

Describes the order of work for the new `src/dojo` implementation. This
is a clean-break refactor rather than a line-by-line port from
`src/dojo_deprecated/`; the deprecated package is reference material,
not the implementation plan.

The work is ordered to prove shared contracts early, then layer broader
capabilities on top once config, storage, results, and identity hashing
are stable.

## Priority 1 — Minimal viable end-to-end thin slice

**Goal:** one supervised single-head training run, configured through
Hydra / Pydantic, reading a Parquet train / val dataset, writing
canonical tall-Parquet results with a `_metadata.json` sidecar.

This slice is deliberately the thinnest path that touches every
architectural boundary exactly once. It proves the contracts that
everything else depends on, and surfaces integration problems with
`amplify-db-utils` / `amplify-storage-utils` while the blast radius is
still small.

| Layer | Minimal inclusion |
|---|---|
| Config | Hydra composition + Pydantic root schema for a minimal supervised run (`experiment`, `task.type: supervised`, `runtime`, `storage`, `data`, `model`, `objectives`, `training`, `output_root`, `training_outputs`) |
| CLI | `dojo train` and `dojo inspect config` only |
| Data | `parquet_manifest` backend + shared sample contract (`sample_id`, `uri`, `split`, one target); `parquet_images` fixture to avoid path-resolution combinatorics |
| Model | `torchvision` backbone + single `multiclass_classification` head + one `cross_entropy` objective; no tabular, adapter, or multi-head path |
| Training | Supervised LightningModule, AdamW, best-k checkpointing, `local` logger sink only |
| Storage | `amplify-storage-utils` resolver behind the Dojo storage interface; `output_root` / `dir_template` resolution |
| Results | Canonical tall-Parquet writer via `amplify-db-utils`: `sample_metadata` + `classification_output` record types, provenance columns, `config_hash` / `dataset_hash` / `checkpoint_hash`, `_metadata.json` sidecar |

Explicitly excluded from the slice: SSL, ensembling, sweeps, export,
multi-head, tabular input, representation evaluation, `ifcb_bins`,
`timm`, ONNX, and all non-local logging.

Acceptance criteria:

- `dojo inspect config` composes, validates, renders output paths, and
  reports enabled outputs.
- `dojo train` runs on the `parquet_images` fixture and produces
  a resolved run directory with `config/`, `checkpoints/`, `results/`,
  and `metrics/`.
- Result Parquet round-trips through `amplify-db-utils`, and a reader can
  filter by `stage` / `record_type`; the `_metadata.json` sidecar
  validates against the schema.
- Hashes are deterministic and reproducible across two runs of the same
  config.
- One integration test (`test_train_supervised.py`) plus unit tests for
  the config schema, result writer, and hashing canonicalizer are green
  in CI.

Why this gates the rest: if the `amplify-db-utils` result contract, the
storage interface, or the hash / identity scheme need rework, it is far
cheaper to find out here than after heads, objectives, ensembling, and
sweeps are layered on top.

## Priority 2 — Core supervised platform

Sections keep stable `P2.x` identifiers (referenced elsewhere in the
design doc) but are listed here in **recommended build order**, not
numeric order. The order is dependency-driven:

- **Foundations** consumed at config resolution come first: the full
  config tree (P2.1) is the shape every other section extends, and the
  dataset stats cache (P2.2) freezes the class-count / normalization /
  dimension / bit-depth / bin-length values that model and transform
  resolution read.
- **Core producers** come next: model composition (P2.3) and transforms
  (P2.4) are near-independent of each other but both depend on P2.2's
  frozen values.
- **Consumers / consolidation** come last: the config-derived result
  hashes (P2.5b) must be cut against the settled resolved model and
  preprocessing shapes, and inference / holdout evaluation (P2.6)
  consumes the whole stack.

P2.5 is **split**. Its schema / taxonomy half (**P2.5a**) is pulled
ahead of the producers so multi-head result rows are written into a
stable schema from the start rather than migrated later. Its
config-derived hash extractors (**P2.5b**) stay after model and
transforms, since `model_config_hash` / `preprocessing_hash` cannot be
finalized before those resolved shapes exist.

**Build order: P2.1 → P2.2 → P2.5a → P2.3 → P2.4 → P2.5b → P2.6.**

### P2.1 Config / CLI / storage foundation

- Full config tree and Pydantic contract.
- `dojo init` for materializing packaged configs and optional fixture data
  into a local editable project.
- `dojo inspect dataset`, folding in old manifest-making behavior and
  acting as the cheapest gate against bad data.
- `dojo inspect backbone` and `dojo inspect checkpoint`.
- Composite logger abstraction: `local` functional; Aim and MLflow
  deferred (not registered sink types).

### P2.2 Dataset backends

- `csv_manifest`, `parquet_manifest`, and `parquet_images`.
- `parquet_images` materialized image cache for embedded image bytes, keyed by
  source Parquet relative file names and bytes, with rebuild / clobber /
  cache-bust controls.
- Shared record contract.
- Preflight checks: `empty_train_classes`,
  `non_contiguous_class_indices`, `imbalance_ratio_gt`.
- `dojo inspect dataset` aspect flags (tiered manifest / header / decode
  passes, terminal chart rendering) and the `dataset_hash`-keyed stats cache
  producing the frozen normalization / tabular / target / class-count /
  bit-depth / dimension / bin-length values consumed at config resolution,
  plus the separate `dataset_content_hash` on full passes.
- Class-name derivation from dataset-level feature metadata, as a third source
  behind the P1 `label_index_column` / `label_name_column` (`04-data-and-storage.md`).
  When neither target column carries readable names, read the index → label
  mapping from the dataset's own schema metadata — e.g. a Hugging Face
  `ClassLabel` feature (`features["label"].int2str(...)`), or equivalent
  sidecar / Arrow field metadata — and fold the resolved mapping into the
  frozen class mapping in the stats cache. Not all inputs expose this (plain
  Parquet, CSV manifests), so it stays an optional enrichment, never required.

### P2.5a Results schema and taxonomy

Sequenced ahead of the producers so P2.3's multi-head training writes
into a stable result schema instead of a later migration.

- Full record-type taxonomy.
- Ground-truth columns on `classification_output` rows (the sample's true
  `target_index` / `target_name` alongside the `prediction_*` columns), so a
  row is self-scoring without joining back to `sample_metadata` / the source
  manifest. This is the ground-truth (`y_true`) half of the `target` vs
  `prediction` pairing; P1 writes predictions only and drops the old target
  identifier column (rows are scoped by `head_name`, and the head → target link
  lives in `_metadata.json`). Settle the readable-suffix asymmetry then
  (`prediction_label` vs `target_name`).
- Per-head `head_hash` for join-free cross-run querying: a content hash over a
  head's resolved config (target, `num_classes`, `class_mapping`, network),
  finer-grained than the whole-schema `target_schema_hash`. Lets rows from the
  same head configuration be grouped across runs without a denormalized name
  column. Source-field list pinned in `06-results-artifacts-and-metadata.md`.
- Partitioning.
- External / internal column convention.

### P2.3 Model composition

- Backbone registry: `torchvision` architecture functional, `timm`
  architecture gated by the `timm` extra; `weights.source` `none` /
  `library` / `checkpoint` initialization all functional.
- Head registry with target validation.
- Objectives binding heads to losses, metrics, and weights.
- Multi-head / multi-objective normalization.
- Embedding adapter.
- Supervised transfer learning from a checkpoint.

### P2.4 Transforms

- Transform builder.
- Resize, letterbox, aspect buckets, foreground crop, grayscale, `normalize`.
- Per-step `train_only` flag and the derived, resolved-only
  `inference_pipeline` consumed by non-train stages, export, and
  `preprocessing_hash`.
- `aspect_bucket` assignment as a working-manifest `aspect_bucket` column
  (derived from cached dimensions × scheme) and the `batch_aspect_buckets`
  bucket-aware sampler yielding size-homogeneous batches.
- Sampler factory: `class_balanced` and `weighted` samplers reading frozen
  class counts, composing with `batch_aspect_buckets` (bucket grouping
  outer, class weighting within bucket), and selecting class counts by
  configured sampler head or the most imbalanced classification head.
- Record scale metadata columns.

### P2.5b Results hash extractors and metrics cleanup

The config-derived half of results hardening, sequenced after model and
transforms because the hashes are content hashes over their resolved
shapes. (The schema / taxonomy half is P2.5a, ahead of the producers.)

- Clean per-epoch metrics CSV: one merged row per epoch instead of separate
  train and validation rows at the same step (a Lightning `CSVLogger` quirk —
  the `on_train_epoch_end` / `on_validation_epoch_end` hooks flush as separate
  `log_metrics` calls). P1 logs epoch-only (no per-batch step rows) but leaves
  the train/val row split as-is.
- Compatibility-hash extractors implementing the pinned
  `target_schema_hash`, `class_mapping_hash`, `model_config_hash`, and
  `preprocessing_hash` source field lists from
  `06-results-artifacts-and-metadata.md`.
- Inference-contract writer: the task module's `on_save_checkpoint` hook
  embeds the portable inference contract into
  `checkpoint["dojo_inference_contract"]` (buildable `model_config`, resolved
  `inference_pipeline` with frozen preprocessing stats, per-head class maps,
  target schema, resolved `objective_summary`, and the four compatibility
  hashes; see `05-models-training-and-heads.md` / `06-...md`). Depends on the
  hash extractors above; this is the write side that P2.6 reads.

### P2.6 Inference and holdout evaluation

- `dojo infer predictions` and `dojo infer embeddings`.
- `dojo eval holdout`.
- `eval_outputs` directory handling, canonical result writing, and the
  always-written `eval_manifest.json`.
- Checkpoint / export loading through the embedded portable inference
  contract, without depending on the producing run's `resolved.yaml`.
- Holdout scorer reconstruction from `objective_summary` and
  `target_schema`.
- Result rows with `stage=infer` / `stage=holdout_eval`, including
  canonical provenance, embedding, and prediction columns.

## Priority 3 — Major capability layers

### P3.1 Export

- TorchScript and ONNX.
- Export metadata.
- Bucket-aware ONNX.

Needed for downstream serving and for ensembling exported models.

### P3.2 Prediction-space ensembling

- Candidate discovery with explicit sources only.
- Compatibility checking.
- Selection strategies.
- Combine modes.
- Cached-result path.
- `task.type: snapshot_ensemble`.

### P3.3 Sweeps

- Dojo-owned sweep expansion over the Hydra Compose API (no `@hydra.main`,
  no `-m`).
- Top-level `sweep:` block: grid sweeps and batch-run-style seed sweeps.
- `dojo sweep prepare`, manual per-job training, `dojo sweep status`, and
  `dojo sweep report`.
- `sweep_outputs` reporting.

### P3.4 SSL and representation evaluation

- DINOv2 via Lightly.
- `representation_eval`: probes, projections, clustering, diagnostics.
- Standalone and training-integrated execution.

### P3.5 IFCB bins

- `ifcbkit` integration for the `ifcb_bins` backend across train / eval /
  infer paths, including SSL where applicable.
- Custom dataset and dataloaders for IFCB bins.

### P3.6 Tabular input

- `model.tabular_input`: selected logical feature columns, input-stream name
  (`name`, default `tabular`), and tabular encoder. Excluded from the P1 slice
  and P2.3 model composition; layered in here.
- `model.image_input.name` names the image input stream and defaults to
  `image`; `model.image_input.backbone.architecture.name` remains the
  architecture selector.
- When image and tabular inputs are both
  enabled, the supervised compositor concatenates embeddings implicitly in
  canonical order: image first, tabular second. Learned post-concat
  capacity belongs in `embedding_adapter`.
- `transforms.tabular`: numeric normalization specs / statistics,
  missing-value imputation (per-column strategy, frozen train-split fill
  values, optional numeric missing indicators), one-hot categorical encoding
  with frozen train-split vocabularies and explicit unknown / missing tokens,
  and train-only augmentations such as `random_missing`.
- Resolved tabular preprocessing state persisted in the config artifact,
  exported with portable models (`10-export.md`), and exercised by the
  `preprocessing_hash` / `model_config_hash` extractors from P2.5
  (`model.tabular_input.enabled: false` leaves the tabular-input
  sub-block disabled).

## Priority 4 — Deferred backlog

These features are lower priority. Deferred features are **absent from
the strict schema** until promoted into active work, so configuring one
fails generic validation — there are no `NotImplementedError` stubs or
reserved slots. Cleanup and enhancement items name their verification
obligation instead. See `appendix-deferred-features.md`.

- P4.1 Bayesian sweeps.
- P4.2 Multilabel support: one classifier head can emit several
  outputs from the list of possible outputs.
- P4.3 Aim runtime logging.
- P4.4 MLflow runtime logging.
- P4.5 Non-DINOv2 SSL: SimCLR, VICReg, PMSN, original DINO.
- P4.6 Weight-space ensembles (P4.6a), weighted combine modes (P4.6b),
  and `prediction_trimmed_mean` (P4.6c).
- P4.7 HDF / HDF5 result exports.
- P4.8 Deprecated package `src/dojo_deprecated/` removal.
- P4.9 Distributional and count regression heads
  (`distributional_regression`, `count_regression`).
- P4.10 WebDataset.
- P4.11 Registry-based ensemble candidate discovery.
- P4.12 `majority_vote` probability-mass tie-break: when members carry
  `probabilities`, break vote ties by highest summed member probability
  across the tied classes (falling back to lowest class index). An
  enhancement to the functional lowest-index tie-break, not a deferred
  config token, so it carries no obligation and has no
  `appendix-deferred-features.md` entry.
- P4.13 Tabular-only model schema.
- P4.14 Expanded tabular encoder families beyond `identity`, `linear`,
  and `mlp` (`tab_transformer`, `ft_transformer`, `tabnet`,
  `embedding_bag`, `wide_and_deep`).
- P4.15 `dojo init --wizard` interactive config/project questionnaire.
- P4.16 Automated sweep execution runners: local sequential execution and
  Slurm / HPC queue submission. Initial execution mode is `manual` via
  `sweep.execution.mode`.
- P4.17 Optional rename of the local shadow dir `./configs` to
  `./config_defaults`. The packaged Hydra YAML tree now lives at
  `src/dojo/config_defaults/`; the local shadow remains `./configs` for current
  user-editable project overrides.

## Historical Notes

`src/dojo_deprecated/` has already been moved out of the new package
path. Treat it as reference material for behavior and edge cases, not as
the structure to port directly.

Likely useful reference areas:

- `dojo_deprecated/multiclass/` for supervised training cues,
  per-class counting, and callback behavior.
- `dojo_deprecated/multilabel/` as a historical example of multi-head
  multiclass behavior; do not port it as multilabel support.
- `dojo_deprecated/selfsupervised/` for SSL implementation cues.
- `dojo_deprecated/tools/dataset_lists_from_folder.py` for manifest
  inspection behavior that now belongs under `dojo inspect dataset`.
- `dojo_deprecated/schemas/core.py` for old config-field intent, while
  the new Pydantic schemas remain the contract.
- `dojo_deprecated/multiclass/callbacks.py` for checkpointing and
  training-loop behavior worth preserving.

Deprecated code should not receive new features. Once the new `src/dojo`
implementation covers everything through P4.7 and nothing (code or test)
imports from `src/dojo_deprecated/`, the deprecated package can be
deleted (the P4.8 cleanup milestone).

## Cross-References

- `01-goals-and-scope.md` — scope and deferred backlog.
- `02-cli-and-task-types.md` — CLI surface.
- `03-configuration.md` — top-level config tree.
- `06-results-artifacts-and-metadata.md` — result schemas, sidecar
  `_metadata.json`, hashes, and artifact layout.
- `07-ssl-and-representation-eval.md` — SSL and representation
  evaluation.
- `08-ensembles.md` — prediction-space ensembling.
- `09-sweeps-and-batch-runs.md` — grid sweeps, batch-run-style seed
  sweeps, and sweep outputs.
- `11-dependencies.md` — base install and extras.
- `12-validation-testing-and-preflight.md` — validation and strict-schema
  deferral policy.
- `appendix-deferred-features.md` — deferred-feature backlog (absent from
  the strict schema) and the P4.8 cleanup milestone.
