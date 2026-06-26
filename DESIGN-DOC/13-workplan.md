
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
| Training | Supervised LightningModule, AdamW, cosine scheduler, best-k checkpointing, `local` logger sink only |
| Storage | `amplify-storage-utils` resolver behind the Dojo storage interface; `output_root` / `dir_template` resolution |
| Results | Canonical tall-Parquet writer via `amplify-db-utils`: `sample_metadata` + `classification_output` record types, provenance columns, `config_hash` / `dataset_hash` / `checkpoint_hash`, `_metadata.json` sidecar |

Explicitly excluded from the slice: SSL, ensembling, sweeps, export,
multi-head, tabular fusion, representation evaluation, `ifcb_bins`,
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

### P2.1 Config / CLI / storage foundation

- Full config tree and Pydantic contract.
- `dojo inspect dataset`, folding in old manifest-making behavior and
  acting as the cheapest gate against bad data.
- `dojo inspect backbone` and `dojo inspect checkpoint`.
- Composite logger abstraction: `local` functional; Aim and MLflow
  stubbed.

### P2.2 Dataset backends

- `csv_manifest`, `parquet_manifest`, and `parquet_images`.
- Shared record contract.
- Preflight checks: `empty_train_classes`,
  `non_contiguous_class_indices`, `imbalance_ratio_gt`.
- `dojo inspect dataset` aspect flags (tiered manifest / header / decode
  passes) and the `dataset_hash`-keyed stats cache producing the frozen
  normalization / tabular / target / class-count / bit-depth / bin-length
  values consumed at config resolution.

### P2.3 Model composition

- Backbone registry: `torchvision` and `checkpoint` functional; `timm`
  gated by extra.
- Head registry with target validation.
- Objectives binding heads to losses, metrics, and weights.
- Multi-head / multi-objective normalization.
- Embedding adapter.
- Supervised transfer learning from a checkpoint.

### P2.4 Transforms

- Transform builder.
- Letterbox, aspect / size buckets, foreground crop, grayscale,
  normalization.
- Per-step `train_only` flag and the derived, resolved-only
  `inference_pipeline` consumed by non-train stages, export, and
  `preprocessing_hash`.
- Record scale metadata columns.

### P2.5 Results hardening

- Full record-type taxonomy.
- Partitioning.
- External / internal column convention.
- Compatibility-hash extractors implementing the pinned
  `target_schema_hash`, `class_mapping_hash`, `model_config_hash`, and
  `preprocessing_hash` source field lists from
  `06-results-artifacts-and-metadata.md`.

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

Resolve the flagged open item on member-vs-ensemble row partitioning
before finalizing.

### P3.3 Sweeps

- Hydra multirun.
- Top-level `sweep:` block: grid sweeps and batch-run-style seed sweeps.
- `sweep_outputs` aggregation.

### P3.4 SSL and representation evaluation

- DINOv2 via Lightly.
- `representation_eval`: probes, projections, clustering, diagnostics.
- Standalone and training-integrated execution.

### P3.5 IFCB bins

- `ifcbkit` integration into SSL.
- Custom dataset and dataloaders for IFCB bins.

### P3.6 Tabular features and fusion

- `model.tabular`: feature columns, encoder, and image / tabular fusion
  (`concat` / `concat_mlp`). Excluded from the P1 slice and P2.3 model
  composition; layered in here.
- Tabular preprocessing: categorical encodings, normalization statistics,
  and missing-value imputation (per-column strategy, frozen train-split
  fill values, optional missing indicators).
- Resolved tabular preprocessing state persisted in the config artifact,
  exported with portable models (`10-export.md`), and exercised by the
  `preprocessing_hash` / `model_config_hash` extractors from P2.5 (the
  tabular sub-blocks are empty when `model.tabular.enabled` is false).

## Priority 4 — Deferred runtime stubs

These features are lower priority. They are represented as explicit
runtime stubs with `NotImplementedError` contracts until they are
promoted into active work. See `appendix-deferred-features.md` for the
stub-test obligations.

- P4.1 Bayesian sweeps.
- P4.2 Multilabel support: one classifier head can emit several
  outputs from the list of possible outputs.
- P4.3 Aim runtime logging.
- P4.4 MLflow runtime logging.
- P4.5 Non-DINOv2 SSL: SimCLR, VICReg, PMSN, original DINO.
- P4.6 Weight-space ensembles, weighted combine modes, and
  `prediction_trimmed_mean`.
- P4.7 HDF / HDF5 result exports.
- P4.8 Deprecated package removal after the new `src/dojo`
  implementation covers up to and including P4.8.
- P4.9 WebDataset.
- P4.10 Registry-based ensemble candidate discovery.

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
implementation covers up to and including P4.8, `src/dojo_deprecated/`
can be deleted.

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
- `12-validation-testing-and-preflight.md` — validation and stub-test
  policy.
- `appendix-deferred-features.md` — deferred runtime stubs.
