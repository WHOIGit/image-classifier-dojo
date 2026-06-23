
# 12. Validation, Testing, and Preflight

## Purpose

Defines layered validation (Pydantic, `dojo inspect config`, dataset
preflight, runtime validation), `runtime.preflight` controls, the
testing policy for deferred features (one `NotImplementedError` test
per stubbed runtime path), and the functional-feature testing scope.

## Layered validation

1. **Pydantic static schema validation.** Catches structural and type
   errors at config load.
2. **`dojo inspect config`.** Composition, validation, rendered paths,
   output-tree preview, run / sweep directory collision warnings,
   enabled-output / deferred-feature reporting. Default: offline
   schema-and-local feasibility. Opt-in `--check-remote` flag for
   remote-availability HEAD / etag checks (see
   `02-cli-and-task-types.md`).
3. **`dojo inspect dataset`** / training preflight. Manifest and target
   checks, missing-target reporting, sample-drop summarization. Default
   missing-target policy is `error` for all heads.
4. **Runtime validation.** Checkpoints, models, schemas, tensor shapes,
   exports.

## `runtime.preflight`

Preflight controls live under `runtime.preflight`. Dataset checks
include:

- `empty_train_classes`
- `empty_eval_classes`
- `non_contiguous_class_indices`
- `imbalance_ratio_gt`

`imbalance_ratio_gt` means:

```text
max_class_count / min_nonzero_class_count > configured_threshold
```

Defaults to a warning, not an error.

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

### Runtime controls

Supported `runtime` keys: `seed`, `fast_dev_run`, `precision`,
`num_workers`, `autobatch`, `preflight`, `run_id`, `sweep_id`.

Early stopping is a training-loop behavior under `training:`, not
`runtime:` (see `05-models-training-and-heads.md`).

## Testing policy for deferred features

For features that are stubbed in the initial implementation, tests
assert the stub behavior and nothing else. **The stub itself is the
contract.**

Each stubbed runtime path has **exactly one** test that:

1. constructs a config that exercises the deferred feature;
2. invokes the runtime path;
3. asserts a `NotImplementedError` is raised;
4. asserts the error message names the deferred feature and points at
   the deferred-feature backlog.

Stubbed features get **no** schema-only tests, **no** inspect-output
enumeration tests, and **no** scaffolded runtime tests.

When a deferred feature is later unstubbed, the stub-assertion test is
deleted and replaced with real functional tests.

### Stubbed features in the initial implementation

- Aim logger sink (`training_outputs.logging.sinks[].type: aim`). Only
  the `local` sink is functional; metrics and figures are recorded
  locally for the foreseeable future.
- MLflow logger sink (`training_outputs.logging.sinks[].type: mlflow`).
- Non-`dino_v2` SSL methods (`ssl.method: simclr | vicreg | pmsn |
  dino`).
- Weight-space ensembles (model soup, greedy soup, uniform soup, SWA,
  EMA).
- Weighted ensemble combine modes (weighted logits / probabilities /
  vote / mean).
- Broad automatic registry-based cross-run candidate discovery.
- `prediction_trimmed_mean`.
- WebDataset dataset backend.
- Bayesian / AutoML HPO (`sweep.mode: bayesian`).
- HDF / HDF5 result exports.

See `appendix-deferred-features.md` for the full list with stub-test
obligations.

## Functional features get full functional tests

When the relevant extra is installed:

- `model.backbone.source: timm` (functional);
- `model.backbone.source: torchvision` and `checkpoint`;
- `local` logger sink (the only functional sink);
- `dino_v2` SSL via Lightly;
- `ifcb_bins` dataset backend (with `[ifcb]`);
- UMAP, t-SNE, HDBSCAN, regression / ordinal / classification probes
  (with `[repr_eval]`);
- ONNX export (with `[onnx]`);
- S3 storage (via the base `amplify-storage-utils` dependency);
- snapshot ensembles and prediction-space ensembles (selection
  strategies `all`, `best_candidate`, `top_k`,
  `greedy_forward_selection`, `cycle_end_snapshots`; combine modes per
  `08-ensembles.md`).

The `local` sink is the only functional logging sink, so functional
logging tests exercise `local` alone. Multi-sink composition is
supported structurally, but any config involving `aim` or `mlflow`
inherits their stubbed runtime.

## Test scope by area

- **Config tests** — Hydra composition; Pydantic validation success and
  failure; CLI override validation; invalid head / loss combinations;
  invalid objective references; invalid dataset columns; invalid logger
  sink configs; invalid ensemble configs.
- **Dataset tests** — CSV / Parquet / `parquet_images` / `ifcb_bins`
  backends; multi-head targets; tabular feature extraction;
  `sample_id` / `uri` / `bin_id` / `bin_uri` propagation; mocked
  storage resolver for S3-style paths.
- **Storage tests** — local amplify-backed storage; cache resolver
  behavior; S3 optional-import behavior;
  `open_bytes` / `localize` / `write_bytes` / `exists`.
- **Transform tests** — letterbox, aspect / size buckets,
  foreground-aware crop, grayscale repeat-to-3, normalization, DINOv2
  multi-view transform, extreme aspect ratios, small native resolutions.
- **Model tests** — torchvision, timm (with `[timm]`), and checkpoint
  backbones; freeze policies; head construction; multi-head model
  composition; embedding adapter; tabular fusion.
- **Objective / loss tests** — loss / metric compatibility per head
  type; objective shorthand; weighted total loss.
- **Supervised training smoke tests** — minimal-fixture train +
  validate + canonical result writing.
- **SSL training smoke tests** — DINOv2 functional smoke test.
- **Representation evaluation tests** — embeddings, projections,
  clustering, probes (with `[repr_eval]`).
- **Ensemble tests** — candidate discovery; manifest writing;
  `dojo ensemble candidates`; supported selection strategies and
  combine modes; cached-result and live-inference paths.
- **Export tests** — TorchScript and ONNX exports; metadata embedding.
- **Logging tests** — `local` functional; Aim and MLflow
  stub-assertions.

## Test fixtures

Three tiers:

- **Tier 1 — committed fixtures (`tests/fixtures/`).** Minimal,
  deterministic data used by unit and integration tests. Parquet
  manifests with **image bytes inlined** (`parquet_images` mode).
  Tracked via git LFS. Fixture parquet content is added separately,
  by the developer; this design doc only specifies the format
  and pipeline.
- **Tier 2 — local development fixture.** Not committed. Lives at
  `/home/sbatchelder/Projects/ifcbNN/datasets/miniset/` or similar;
  pointed at via env var (e.g. `DOJO_DEV_DATASET_ROOT`).
- **Tier 3 — real example dataset.** Not committed; fetched at
  runtime. Hugging Face dataset
  `sbatchelder/NES-plankton-classifier-2022-dataset`.

Integration tests consume Tier 1 fixtures exclusively.

## Cross-References

- `02-cli-and-task-types.md` — `dojo inspect config`, `dojo inspect
  dataset`, remote-validation flag.
- `03-configuration.md` — `runtime:` block placement.
- `04-data-and-storage.md` — dataset preflight checks.
- `06-results-artifacts-and-metadata.md` — logging-sink behavior.
- `appendix-deferred-features.md` — every deferred feature has a
  stub-assertion test obligation defined here.
- `11-dependencies.md` — extras gate which functional tests run.
- `glossary.md` — preflight / runtime vocabulary.
