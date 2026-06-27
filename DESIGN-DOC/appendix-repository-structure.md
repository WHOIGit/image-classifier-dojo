
# Appendix — Proposed Repository Structure

## Purpose

Concrete proposed target directory layout. The tree is derived from the
canonical decisions in `03-configuration.md`,
`04-data-and-storage.md`, `05-models-training-and-heads.md`,
`06-results-artifacts-and-metadata.md`, `07-ssl-and-representation-eval.md`,
`08-ensembles.md`, `09-sweeps-and-batch-runs.md`, `10-export.md`,
`11-dependencies.md`, `12-validation-testing-and-preflight.md`, and
`13-workplan.md`. It is populated according to the priority order in the
workplan; modules shown here may start as stubs or remain unimplemented
until their priority is reached.

This is intent, not contract. Implementers may collapse, split, or
rename leaf modules to fit emergent code shape; the *areas of
responsibility* shown here are what must be preserved.

## Top-level layout

```text
image-classifier-dojo/
  pyproject.toml
  README.md
  Dockerfile
  LICENSE

  configs/
  src/dojo/
  src/dojo_deprecated/       # reference during refactor; removal governed by 13-workplan.md. No new code to be developed here. Exists temporarily for historical reference only.
  tests/
  DESIGN-DOC/
```

## `configs/`

The repository-level `configs/` tree is the editable development copy of
the packaged config resources. The installable package also carries a
read-only copy under `src/dojo/configs/`; `dojo init` materializes selected
packaged configs into a user's local `./configs` tree.

```text
configs/
  config.yaml                # top-level Hydra entrypoint

  experiment/                # Hydra experiment group; selected via experiment=<name>
    ifcb/
      baseline_resnet50.yaml
      experimentA.yaml
      dino_v2_ssl.yaml
      convnext_snapshot.yaml
      transfer_from_ssl.yaml
      ensemble_from_runs.yaml
      ensemble_from_cached_results.yaml
      representation_eval_standalone.yaml
    sweeps/                  # experiment configs that define a sweep: block
      grid_lr_wd.yaml
      batch_seed_5.yaml
      ensemble_axis_sweep.yaml
      bayesian_optuna_stub.yaml
      
  data/
    csv_local.yaml
    csv_s3.yaml
    parquet_manifest.yaml
    parquet_images.yaml
    ifcb_bins.yaml

  transforms/
    supervised_default.yaml
    letterbox_square.yaml
    aspect_bucket.yaml
    ssl_dino_v2_plankton.yaml

  backbone/
    torchvision/
      resnet50.yaml
      efficientnet_b0.yaml
      inception_v3.yaml
      vit_b_16.yaml
    timm/                    # functional; gated by [timm] extra
      convnext_tiny.yaml
      vit_small_patch16_224.yaml
      efficientnet_b0.yaml
    checkpoint/
      from_local_checkpoint.yaml
      from_s3_checkpoint.yaml

  head/
    single_classification.yaml
    multihead_species_quality.yaml
    classification_regression_ordinal.yaml

  ssl/
    dino_v2.yaml

  representation_eval/
    knn.yaml
    linear_probe.yaml
    diagnostics_default.yaml
    standalone_default.yaml

  optimizer/
    adamw.yaml
    sgd.yaml

  scheduler/
    cosine.yaml
    cosine_warm_restarts.yaml  # snapshot-cycle scheduler
    step.yaml
    none.yaml

  loss/
    cross_entropy.yaml
    weighted_cross_entropy.yaml
    class_balanced_effective_number.yaml
    focal.yaml
    label_smoothing.yaml
    regression_huber.yaml
    ordinal_coral.yaml
    ordinal_corn.yaml

  runtime/
    default.yaml
    fast_dev_run.yaml
    preflight_strict.yaml

  storage/
    local_only.yaml
    s3.yaml

  sweep/
    disabled.yaml
    grid.yaml
    batch_seed_5.yaml              # grid sweep over runtime.seed only
    bayesian_optuna_stub.yaml      # schema present; runtime stubbed

  training_outputs/
    default.yaml
    sweep_aware.yaml

  ensemble_outputs/
    default.yaml
    snapshot_shared_with_training.yaml

  sweep_outputs/
    default.yaml

  ensemble/
    disabled.yaml
    cycle_end_snapshots.yaml
    top_k.yaml
    greedy_forward_selection.yaml
    cached_results_only.yaml   # source_policy: strict_no_inference

  export/
    torchscript_model.yaml
    torchscript_ensemble_model.yaml
    onnx_model.yaml

  logging/
    local.yaml                 # only functional sink
    # aim.yaml is provided but stubbed at runtime
    # mlflow.yaml is provided but stubbed at runtime
```

No `task/` config group — `task.type` is set in the experiment config.
No `configs/hydra/` group — Dojo uses the Hydra Compose API, not
`@hydra.main`, so there is no Hydra run / sweep / `chdir` config to set.

## `src/dojo/`

```text
src/dojo/
  __init__.py

  cli/
    __init__.py
    main.py                    # Typer app entrypoint; composes config, dispatches
    init.py                    # dojo init: materialize configs + optional fixture data
    train.py                   # all task.type values: supervised, ssl, snapshot_ensemble
    eval.py                    # dojo eval, dojo eval holdout, dojo eval representation
    infer.py                   # dojo infer, dojo infer predictions, dojo infer embeddings
    ensemble.py                # dojo ensemble, dojo ensemble candidates
    sweep.py                   # dojo sweep prepare|train|status|report
    export.py                  # dojo export
    inspect.py                 # dojo inspect config|dataset|backbone|checkpoint

  configs/                     # packaged read-only config resources for search fallback / dojo init
    config.yaml
    experiment/
    data/
    transforms/
    backbone/
    training/
    sweep/

  example_data/                # tiny packaged fixture dataset for dojo init --data

  config_schemas/              # Pydantic; single source of truth
    __init__.py
    root.py
    experiment.py
    task.py                    # task.type union
    runtime.py                 # runtime + preflight
    storage.py
    data.py
    transforms.py
    model.py                   # image_input, tabular_input, embedding_adapter, heads
    backbones.py
    heads.py
    objectives.py
    losses.py
    optimizers.py
    schedulers.py
    checkpointing.py
    ssl.py
    representation_eval.py
    ensemble.py
    sweep.py
    outputs.py                 # training_outputs, ensemble_outputs, sweep_outputs
    logging.py
    export.py
    validation.py              # cross-cutting validators
    hashing.py                 # canonicalizer + per-field hash rules

  data/
    __init__.py

    datamodules/
      __init__.py
      base.py
      csv.py
      parquet.py
      ifcb_bins.py

    datasets/
      __init__.py
      csv_image_dataset.py
      parquet_image_dataset.py
      ifcb_bins_dataset.py

    record_schemas/            # Pydantic for persisted records
      __init__.py
      sample.py
      target.py
      result.py
      embedding.py
      batch.py                 # hot-path batches stay as dicts; this is the schema spec
      ensemble_member.py

    transforms/
      __init__.py
      builder.py
      letterbox.py
      aspect_bucket.py
      foreground_crop.py
      grayscale.py
      normalization.py
      crop.py
      blur.py
      noise.py
      rotation.py

    samplers/
      __init__.py
      factory.py
      class_balanced.py
      batch_aspect_buckets.py
      weighted.py

    manifests/                  # dataset manifest IO + dojo inspect dataset
      __init__.py
      io.py
      directory_scan.py
      summary.py                # used by dojo inspect dataset
      validation.py

  storage/
    __init__.py
    resolver.py                  # output_root + dir_template + path resolution
    amplify.py                   # wrapper around amplify-storage-utils
    cache.py                     # local read-through cache (when configured)

  models/
    __init__.py

    backbones/
      __init__.py
      base.py
      registry.py
      torchvision.py
      timm.py                   # gated by [timm] extra
      weights.py                # none/library/checkpoint initialization
      feature_extractor.py
      freeze.py                 # apply training.freeze.backbone policies

    heads/
      __init__.py
      base.py
      classification.py
      regression.py
      ordinal.py                # ordinal_classification semantics
      multihead.py
      projection.py              # SSL projection / prediction heads

    tabular_input/
      __init__.py
      encoders.py
      normalization.py

    embedding_adapter/
      __init__.py
      identity.py
      linear.py
      mlp.py

    compositors/
      __init__.py
      supervised.py              # backbone + tabular_input + implicit concat + adapter + head(s)
      ssl.py                     # SSL composition (encoder + projection)
      snapshot_ensemble.py       # composition for task.type: snapshot_ensemble

  tasks/
    __init__.py

    supervised/
      __init__.py
      module.py                  # LightningModule
      objectives.py
      metrics.py
      step_outputs.py

    ssl/
      __init__.py
      dino_v2.py                 # functional
      lightly_backbones.py
      lightly_heads.py
      lightly_losses.py
      eval_callbacks.py          # schedules representation_eval during SSL training

    snapshot_ensemble/
      __init__.py
      module.py                  # supervised module + cycle-end snapshots
      cycle_scheduler.py
      cycle_checkpointing.py

    representation_eval/         # was eval/; supports SSL + supervised checkpoints
      __init__.py
      runner.py                  # dojo eval representation entrypoint
      knn.py
      linear_probe.py
      embeddings.py
      diagnostics.py             # augmentation consistency, retrieval, etc.
      clustering.py
      projections.py
      outlier.py
      scheduling.py              # during-training scheduling helpers

  losses/
    __init__.py
    classification.py
    regression.py
    ordinal.py                   # CORAL, CORN
    class_balanced.py
    focal.py
    factory.py

  metrics/
    __init__.py
    classification.py
    regression.py
    ordinal.py
    multihead.py
    calibration.py
    confusion.py
    representation_eval.py       # knn / linear-probe / diagnostic metrics

  training/
    __init__.py
    trainer_factory.py
    callbacks.py
    checkpointing.py
    resume.py
    early_stopping.py

  ensemble/
    __init__.py
    runner.py                    # dojo ensemble entrypoint
    candidates.py                # dojo ensemble candidates entrypoint
    discovery.py                 # explicit, manifest, run_dir_glob, checkpoint_glob, result_uri_glob
    compatibility.py             # hashes + cascading metadata policy + drift check
    selection.py                 # all, best_candidate, top_k, greedy_forward, cycle_end_snapshots
    combine.py                   # logits_mean, probabilities_mean, majority_vote, prediction_mean, prediction_median
    cached_results.py            # offline ensembling from result Parquet
    manifest.py                  # JSON manifest IO
    bundle.py                    # ensemble artifact bundling
    preprocessing.py             # member-specific + shared fast path

  inference/
    __init__.py
    inferencer.py
    outputs.py
    preprocessing.py
    batch_writer.py

  export/
    __init__.py
    torchscript.py
    onnx.py
    metadata.py
    bucketing.py                 # ONNX + aspect_bucket shapes

  results/
    __init__.py
    schemas.py                   # canonical tall-Parquet schema
    writers.py                   # multi-sink result writer via amplify-db-utils
    parquet.py                   # primary functional backend via amplify-db-utils
    partitioning.py              # split / stage / epoch keys
    metadata.py                  # _metadata.json sidecar
    confusion.py
    # No csv.py as primary — Parquet is canonical; CSV optional

  artifacts/
    __init__.py
    paths.py                     # path resolution against output_root + dir_template
    manifest.py                  # run-level artifact manifest
    metrics.py
    checkpoints.py
    hashing.py                   # checkpoint_hash + filename convention

  logging/
    __init__.py
    base.py
    factory.py                   # multi-sink composition
    local.py                     # functional (only functional sink)
    aim.py                       # stubbed; raises NotImplementedError at runtime
    mlflow.py                    # stubbed; raises NotImplementedError at runtime

  runtime/
    __init__.py
    preflight.py                 # runtime.preflight checks
    controls.py                  # runtime.* behavior knobs
    seeding.py
    device.py
    autobatch.py

  hydra/
    __init__.py
    compose.py                   # Hydra Compose API wrapper (no @hydra.main)
    resolvers.py                 # custom OmegaConf resolvers
    sweep.py                     # Dojo sweep preparation + sweep_id / manifest wiring

  utils/
    __init__.py
    import_utils.py
    random.py
    distributed.py
    torch_utils.py
    serialization.py
    canonicalizer.py             # canonical JSON for hashing

  patches/
    __init__.py
    model_summary_with_grad.py
```

## `src/dojo_deprecated/`

The pre-refactor package, already moved out of the new `src/dojo` path.
Importable for reference during the clean-break refactor, deleted once
the new `src/dojo` implementation covers everything through P4.7 and
nothing imports from it, per `13-workplan.md`. No code added here.

## `tests/`

```text
tests/
  fixtures/
    configs/
    images/
    manifests/
    checkpoints/
    parquet/
    ifcb_bins/
    cached_results/              # for offline ensemble tests

  unit/
    config_schemas/
    runtime/
    data/
    transforms/
    models/
    heads/
    objectives/
    losses/
    metrics/
    logging/
    results/
    export/
    ensemble/
    representation_eval/
    hashing/

  integration/
    test_train_supervised.py
    test_train_ssl_dino_v2.py
    test_train_snapshot_ensemble.py
    test_representation_eval_standalone.py
    test_representation_eval_during_ssl.py
    test_supervised_holdout_eval.py
    test_inspect_config.py
    test_inspect_dataset.py
    test_inspect_backbone.py
    test_inspect_checkpoint.py
    test_infer_predictions.py
    test_infer_embeddings.py
    test_ensemble_from_run_dirs.py
    test_ensemble_from_cached_results.py
    test_ensemble_candidates_manifest.py
    test_export_torchscript.py
    test_export_onnx.py
    test_sweep_expansion_config.py
    test_sweep_outputs.py

  deferred_stubs/                # NotImplementedError contract tests
    test_bayesian_hpo_stub.py
    test_multilabel_stub.py
    test_aim_logger_stub.py
    test_mlflow_logger_stub.py
    test_non_dinov2_ssl_stubs.py
    test_weight_space_ensemble_stubs.py
    test_weighted_combine_stubs.py
    test_prediction_trimmed_mean_stub.py
    test_hdf_export_stub.py
    test_webdataset_backend_stub.py
    test_registry_discovery_stub.py
```

## Notes on key directories

### `config_schemas/`

Pydantic is the single source of truth for the config tree
(`03-configuration.md`). The CLI loads Hydra-composed configs and
validates them through `config_schemas.root`. Hashing rules and the
canonicalizer (`utils/canonicalizer.py`,
`config_schemas/hashing.py`) live next to the schemas they consume so
field-level hash inclusion rules stay co-located with the field
definitions.

### `data/record_schemas/`

Pydantic schemas for persisted records (samples, targets, results,
embeddings, ensemble-member rows). Hot-path DataLoader batches stay as
dicts/dataclasses internally; record_schemas describe the on-disk shape
written by `results/writers.py`.

### `storage/`

Thin wrapper around `amplify-storage-utils`. The wrapper exists so
output-path resolution (`output_root`, `dir_template`, `dir`,
`existing_run_dir` policy) is a project-level concern not pushed into
the library. It is top-level because storage is shared by data loading,
result writing, checkpointing, export, and artifact inspection.

### `models/compositors/`

Compositors assemble model parts:

```text
image_input backbone + optional tabular_input encoder + implicit input concatenation + optional embedding adapter + head(s)
```

Tasks (`tasks/supervised`, `tasks/ssl`, `tasks/snapshot_ensemble`) train
composited models. The compositor for `task.type: snapshot_ensemble` is
a supervised compositor plus the cycle-end snapshot machinery.

### `ensemble/`

The ensemble package is intentionally *not* organized around ensemble
algorithm types. There is one prediction-space pipeline; candidate
source, selection strategy, and combine mode are the axes that vary.
`cached_results.py` is the offline-from-Parquet path described in
`08-ensembles.md`; it does not duplicate logic from `combine.py`.

### `runtime/`

Process-level concerns separated from training-loop concerns: preflight
checks, seeding, device selection, autobatch, fast-dev-run wiring.
`preflight.py` is invoked from CLI entrypoints before any heavy work,
per `12-validation-testing-and-preflight.md`.

### `tasks/representation_eval/`

`representation_eval` is the package name — `ssl_eval` does not exist
in the V2 tree. The runner supports both standalone use
(`dojo eval representation`) and during-training scheduling (via
`eval_callbacks.py` in the SSL task and `scheduling.py` helpers callable
from the supervised task).


## Cross-References

- `03-configuration.md` — config tree the `config_schemas/` package
  validates; `outputs/` resolution semantics.
- `04-data-and-storage.md` — dataset backends plus top-level
  `amplify-storage-utils` integration in `storage/`.
- `05-models-training-and-heads.md` — what lives under `models/`,
  `losses/`, `metrics/`, `training/`, and `tasks/supervised/`.
- `06-results-artifacts-and-metadata.md` — schema implemented by
  `results/`, `artifacts/`, and `data/record_schemas/`.
- `07-ssl-and-representation-eval.md` — what lives under `tasks/ssl/`
  and `tasks/representation_eval/`.
- `08-ensembles.md` — package layout under `src/dojo/ensemble/`.
- `09-sweeps-and-batch-runs.md` — Compose API + Dojo sweep expansion
  under `hydra/` and `sweep_outputs/`.
- `10-export.md` — `export/` package contents.
- `11-dependencies.md` — extras that gate `models/backbones/timm.py`,
  `tasks/ssl/`, IFCB, etc.
- `12-validation-testing-and-preflight.md` — `runtime/preflight.py` and
  the `tests/` tree.
- `13-workplan.md` — priority order for populating this tree.
- `appendix-deferred-features.md` — runtime paths deliberately stubbed
  until promoted into active work.
