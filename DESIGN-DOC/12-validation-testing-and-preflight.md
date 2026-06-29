
# 12. Validation, Testing, and Preflight

## Purpose

Defines layered validation (Pydantic, `dojo inspect config`, dataset
preflight, runtime validation), `runtime.preflight` controls, the
strict-schema policy for deferred features (they are absent from the
schema and fail generic validation), and the functional-feature testing
scope.

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
   checks, missing-target reporting, sample-drop summarization, and
   required frozen-stats cache availability. Default missing-target policy
   is `error` for all heads. Preflight consumes frozen stats; it does not
   compute or repair missing cache aspects.
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

`autobatch` runs a pre-training batch-size search to size the batch to the
current device/GPU before training proper begins — a thin wrapper over
Lightning's `Tuner.scale_batch_size`. It is off unless
`autobatch.enabled: true`; `mode` is `binsearch` (default) or `power`. The
found batch size feeds `training.batch_size` for the run.

Early stopping is a training-loop behavior under `training:`, not
`runtime:` (see `05-models-training-and-heads.md`).

## Policy for deferred features

Deferred features are **absent from the schema**, not stubbed. The
Pydantic config models are strict (`extra="forbid"`), and enum /
discriminated-union fields list only implemented values, so configuring
a deferred feature fails generic validation at config load — exactly as
a typo or nonsense key would. There are no reserved-but-inert config
slots and no runtime `NotImplementedError` stubs.

Deferred features therefore carry **no per-feature test obligation**. The
whole class is covered by a small number of generic strict-schema tests:

1. an unknown key anywhere in the tree is rejected (`extra="forbid"`);
2. an out-of-enum value / unknown discriminated-union tag is rejected,
   and the error lists the implemented values.

The generic validation error names the offending key or value only; it
does **not** advertise the feature as planned. The deferred roadmap lives
in `appendix-deferred-features.md` and `13-workplan.md`, never in the
running schema. When a deferred feature is built, its value is added to
the schema and gains real functional tests; nothing feature-specific
needs to be deleted first.

The one validation message that stays curated is for **invalid
combinations of implemented features** (incompatible head / loss pairs,
bad objective references, invalid dataset columns); those are real
current contract and keep their specific messages.

Two deferred entries are not config tokens and so are handled in kind:
deprecated-package removal (P4.8) is a cleanup milestone with a
no-imports verification obligation, and the `majority_vote`
probability-mass tie-break (P4.12) is an enhancement to functional
behavior. Neither is a stub.

See `appendix-deferred-features.md` for the full deferred backlog.

## Functional features get full functional tests

When the relevant extra is installed:

- `model.image_input.backbone.architecture.source: timm` (functional);
- `model.image_input.backbone.architecture.source: torchvision`;
- `model.image_input.backbone.weights.source: checkpoint`;
- `local` logger sink (the only functional sink);
- `dino_v2` SSL via Lightly;
- `ifcb_bins` dataset backend (with `[ifcb]`);
- UMAP, t-SNE, HDBSCAN, regression / ordinal / classification probes
  (with `[repr_eval]`);
- ONNX export (with `[onnx]`);
- S3 storage (via the base `amplify-storage-utils` dependency);
- snapshot ensembles and prediction-space ensembles (selection
  strategies `all`, `best_candidate`, `top_k`,
  `greedy_forward_selection`; combine modes per `08-ensembles.md`;
  snapshot ensembles select over the run's implicit cycle-snapshot
  source).

The `local` sink is the only functional logging sink, so functional
logging tests exercise `local` alone. Multi-sink composition is supported
structurally (`CompositeExperimentLogger`), but `aim` and `mlflow` are
not registered sink types — a config naming either fails schema
validation at load.

## Test scope by area

- **Config tests** — Hydra composition; Pydantic validation success and
  failure; CLI override validation; invalid head / loss combinations;
  invalid objective references; invalid dataset columns; invalid logger
  sink configs; invalid ensemble configs; `dojo init` materialization,
  dependency-closure copying, fixture-data copying, and non-clobber /
  clobber behavior.
- **Dataset tests** — CSV / Parquet / `parquet_images` / `ifcb_bins`
  backends; multi-head targets; tabular feature extraction;
  `sample_id` / `uri` / `bin_id` / `bin_uri` propagation; mocked
  storage resolver for S3-style paths.
- **Storage tests** — local amplify-backed storage; cache resolver
  behavior; S3 optional-import behavior;
  `open_bytes` / `localize` / `write_bytes` / `exists`.
- **Transform tests** — letterbox, aspect / size buckets,
  foreground-aware crop, grayscale repeat-to-3, `normalize`, `rotate`, flips, DINOv2 multi-view transform, extreme aspect ratios, small native resolutions.
- **Model tests** — torchvision, timm (with `[timm]`), checkpoint inspection
  gated by `[train]`, and checkpoint backbones; freeze policies; head
  construction; multi-head model composition; embedding adapter; tabular input
  concatenation.
- **Objective / loss tests** — loss / metric compatibility per head
  type; metric registry canonical names / params / output names; objective
  shorthand; weighted total loss; resolved `objective_summary`
  serialization.
- **Supervised training smoke tests** — minimal-fixture train +
  validate + canonical result writing.
- **SSL training smoke tests** — DINOv2 functional smoke test.
- **Representation evaluation tests** — embeddings, projections,
  clustering, probes (with `[repr_eval]`).
- **Ensemble tests** — candidate discovery; manifest writing;
  `dojo ensemble candidates`; supported selection strategies and
  combine modes; cached-result and live-inference paths.
- **Export tests** — TorchScript and ONNX exports; metadata embedding;
  holdout-eval scorer reconstruction from the embedded inference contract.
- **Logging tests** — `local` functional; configs naming `aim` or
  `mlflow` sinks rejected at schema validation.

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
- `appendix-deferred-features.md` — deferred-feature backlog (absent
  from the schema) and the P4.8 cleanup milestone.
- `11-dependencies.md` — extras gate which functional tests run.
- `glossary.md` — preflight / runtime vocabulary.
