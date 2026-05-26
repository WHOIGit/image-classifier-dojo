# Image Classifier Dojo Refactor Design Doc

## Status

Draft architectural design.

## Target Repository

This design is intended for a major breaking refactor of `WHOIGit/image-classifier-dojo`.

The refactor prioritizes:

- supervised image modeling with one or more heads
- self-supervised learning using Lightly, initially focused on DINOv2-style workflows
- strong Pydantic schemas as the source of truth
- Hydra-based config composition and CLI overrides
- configurable experiment logging and artifact output
- snapshot/checkpoint ensembling workflows
- local/S3-capable storage using `amplify-storage-utils`
- columnar, improv-compatible result output
- future Prefect orchestration without adding Prefect to the core project

---

# 1. Goals

## 1.1 Core goals

The refactored project should support:

1. Supervised image classification, regression, and ordinal regression.
2. Multi-head supervised models with one or more output heads.
3. Transfer learning from:
   - torchvision pretrained backbones
   - timm pretrained backbones
   - non-pretrained torchvision/timm backbones
   - local or remote checkpoints
4. Self-supervised learning using Lightly, initially focused on DINOv2-style training.
5. SSL evaluation during training using labeled and/or unlabeled evaluation datasets.
6. Supervised model holdout evaluation for one or more checkpoints/models.
7. Snapshot and checkpoint ensembling workflows.
8. Bundling snapshot/checkpoint ensembles into a single `.pt` artifact containing multiple snapshot `state_dict`s.
9. Optional export to ONNX, including preprocessing and result metadata.
10. Dataset loading from:
    - CSV manifests
    - Parquet manifests
    - Parquet image datasets
    - IFCB bins dataset or equivalent current Dojo support
11. S3 path support for CSV/Parquet-defined image paths.
12. Use of `amplify-storage-utils` for local storage, object storage abstraction, caching, and optional S3 support.
13. Hyperparameter search through Hydra multirun.
14. Experiment logging through:
    - local files
    - Aim
    - MLflow
    - optionally more than one sink if configured
15. Configurable result outputs:
    - canonical tall Parquet
    - wide CSV summaries
    - embeddings CSV
    - confusion matrix CSV
    - HDF/HDF5 export

## 1.2 Non-goals for this refactor phase

The following are intentionally deferred:

1. Full WebDataset support.
2. Full production model serving framework.
3. Prefect flows inside the Dojo package.
4. Full AutoML or Bayesian optimization.
5. Direct dependency on Meta DINOv2 repositories.
6. Exhaustive support for every possible timm architecture edge case.
7. Specialized Inception auxiliary-logit handling in the initial generic path.

---

# 2. Key architectural decisions

## 2.1 Configuration

Use:

```text
Hydra + Pydantic
```

Hydra is responsible for:

- config group composition
- experiment config selection
- CLI overrides
- multirun sweeps

Pydantic is responsible for:

- schema structure
- validation
- defaults
- field constraints
- runtime config objects
- Prefect-compatible config contracts

Flow:

```text
Hydra config groups
        ↓
experiment config
        ↓
CLI overrides
        ↓
Hydra-composed OmegaConf
        ↓
plain dict
        ↓
Pydantic validation
        ↓
validated ExperimentConfig
        ↓
training / evaluation / export code
```

Core training code should receive Pydantic objects, not raw Hydra `DictConfig`s.

```python
def train_supervised(cfg: ExperimentConfig) -> RunResult:
    ...
```

## 2.2 CLI style

Use one package CLI with subcommands.

Primary commands:

```bash
dojo train supervised
dojo train ssl

dojo eval holdout
dojo eval knn
dojo eval linear-probe
dojo eval embeddings

dojo infer

dojo ensemble
dojo ensemble snapshot

dojo export pt
dojo export onnx

dojo inspect backbone
dojo inspect checkpoint

dojo validate-config
```

### Command intent

#### `dojo train supervised`

Train supervised single-head or multi-head models.

#### `dojo train ssl`

Train self-supervised models using Lightly.

#### `dojo eval holdout`

Evaluate one or more supervised models/checkpoints against a holdout dataset.

This should accept multiple checkpoints/models so users can compare:

```text
best-k checkpoints
last checkpoint
snapshot checkpoints
exported models
candidate soups/ensembles
```

#### `dojo eval knn`

Run k-NN representation evaluation over checkpoint-derived or precomputed embeddings.

#### `dojo eval linear-probe`

Train/evaluate a frozen-backbone linear probe.

#### `dojo eval embeddings`

Extract and persist embeddings from a model/encoder checkpoint.

#### `dojo infer`

Run inference and write configured result artifacts.

This replaces `dojo predict`.

#### `dojo ensemble`

Evaluate many possible checkpoint/model combinations and select ensemble candidates based on validation performance and computational tradeoff.

This is the generic ensemble search/selection command.

#### `dojo ensemble snapshot`

Specialized subcommand for snapshot ensembles from a training run or snapshot checkpoint collection.

#### `dojo export pt`

Serialize a selected model or ensemble into a portable `.pt` artifact.

#### `dojo export onnx`

Serialize a selected model or ensemble wrapper into ONNX, when supported.

#### `dojo inspect backbone`

List module names, parameter counts, output dimensions, and freeze-policy effects for a backbone.

#### `dojo inspect checkpoint`

Inspect model/checkpoint structure and available module names.

### Config-first usage

Hydra usage should be config-file based.

Most real usage should look like:

```bash
dojo train supervised experiment=ifcb/experimentA
```

or:

```bash
dojo train ssl experiment=ifcb/dino_v2_ssl
```

Where:

```text
configs/example_experiments/ifcb/experimentA.yaml
```

composes lower-level config groups.

CLI overrides are for small modifications, debugging, and sweeps:

```bash
dojo train supervised \
  experiment=ifcb/experimentA \
  optimizer.lr=3e-4
```

Hydra multirun:

```bash
dojo train supervised -m \
  experiment=ifcb/experimentA \
  backbone=torchvision/resnet50,timm/convnext_tiny \
  optimizer.lr=1e-4,3e-4
```

## 2.3 Logging and artifacts

Logging should support:

```text
local only
Aim only
MLflow only
local + Aim
local + MLflow
local + Aim + MLflow, if not burdensome
```

Local artifacts should generally be produced for reproducibility, but the logger config should not prohibit tracker-only workflows if a user explicitly configures them.

Recommended default:

```text
local artifacts enabled
one experiment tracker optional
multiple trackers allowed if implementation remains simple
```

Logging config should be explicit:

```yaml
logging:
  sinks:
    - type: local
      run_root: ./runs

    - type: mlflow
      tracking_uri: http://localhost:5000
      experiment_name: ifcb

    # optionally:
    # - type: aim
    #   repo: ./aim
    #   experiment_name: ifcb
```

---

# 3. Proposed repository structure

```text
image-classifier-dojo/
  pyproject.toml
  README.md

  configs/
    config.yaml

    example_experiments/
      ifcb/
        baseline_resnet50.yaml
        experimentA.yaml
        dino_v2_ssl.yaml
        convnext_snapshot.yaml
        transfer_from_ssl.yaml

    task/
      supervised.yaml
      ssl.yaml
      linear_probe.yaml
      knn_eval.yaml
      embedding_export.yaml
      inference.yaml
      holdout_eval.yaml
      export.yaml

    data/
      csv_local.yaml
      csv_s3.yaml
      parquet_manifest.yaml
      parquet_images.yaml
      ifcb_bins.yaml

    backbone/
      torchvision/
        resnet50.yaml
        efficientnet_b0.yaml
        inception_v3.yaml
        vit_b_16.yaml

      timm/
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

    transforms/
      supervised_default.yaml
      microscopy_letterbox.yaml
      bucketed_aspect_size.yaml
      ssl_dino_v2_plankton.yaml

    optimizer/
      adamw.yaml
      sgd.yaml

    scheduler/
      cosine.yaml
      cosine_restarts.yaml
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

    results/
      canonical_parquet.yaml
      csv_exports.yaml
      hdf_exports.yaml

    logging/
      local.yaml
      aim.yaml
      mlflow.yaml
      local_and_mlflow.yaml
      local_and_aim.yaml

    ensemble/
      disabled.yaml
      cosine_snapshots.yaml
      greedy_soup.yaml
      top_k.yaml

    export/
      pt_single_model.yaml
      pt_snapshot_ensemble.yaml
      onnx_single_model.yaml
      onnx_snapshot_ensemble.yaml

  src/
    dojo/
      __init__.py

      cli/
        __init__.py
        main.py
        train.py
        eval.py
        infer.py
        export.py
        ensemble.py
        inspect.py

      config_schemas/
        __init__.py
        root.py
        data.py
        backbones.py
        heads.py
        objectives.py
        losses.py
        optimizers.py
        schedulers.py
        transforms.py
        ssl.py
        logging.py
        ensemble.py
        results.py
        export.py
        validation.py
        hydra.py

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

        record_schemas/
          __init__.py
          sample.py
          target.py
          result.py
          embedding.py
          batch.py

        storage/
          __init__.py
          resolver.py
          local.py
          s3.py
          cache.py
          amplify.py

        transforms/
          __init__.py
          builder.py
          primitives/
            __init__.py
            letterbox.py
            aspect_bucket.py
            size_bucket.py
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
          aspect_bucket.py
          weighted.py

      models/
        __init__.py

        backbones/
          __init__.py
          base.py
          registry.py
          torchvision.py
          timm.py
          checkpoint.py
          feature_extractor.py

        heads/
          __init__.py
          base.py
          classification.py
          regression.py
          ordinal.py
          multihead.py
          projection.py

        tabular/
          __init__.py
          encoders.py
          normalization.py

        compositors/
          __init__.py
          supervised.py
          ssl.py
          snapshot_ensemble.py

      tasks/
        __init__.py

        supervised/
          __init__.py
          module.py
          objectives.py
          metrics.py
          step_outputs.py

        ssl/
          __init__.py
          dino_v2.py
          lightly_backbones.py
          lightly_heads.py
          lightly_losses.py
          eval_callbacks.py

        eval/
          __init__.py
          holdout.py
          knn.py
          linear_probe.py
          embeddings.py
          diagnostics.py
          clustering.py
          projections.py

      losses/
        __init__.py
        classification.py
        regression.py
        ordinal.py
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

      training/
        __init__.py
        supervised.py
        ssl.py
        trainer_factory.py
        callbacks.py
        checkpointing.py
        snapshot.py
        resume.py

      ensemble/
        __init__.py
        snapshot.py
        soup.py
        predict.py
        bundle.py
        artifact.py
        selection.py

      inference/
        __init__.py
        inferencer.py
        outputs.py
        preprocessing.py
        batch_writer.py

      export/
        __init__.py
        pt.py
        onnx.py
        metadata.py

      results/
        __init__.py
        schemas.py
        writers.py
        parquet.py
        csv.py
        hdf.py
        confusion.py
        improv.py

      artifacts/
        __init__.py
        paths.py
        manifest.py
        metrics.py
        checkpoints.py

      logging/
        __init__.py
        base.py
        local.py
        aim.py
        mlflow.py
        factory.py

      utils/
        __init__.py
        import_utils.py
        random.py
        distributed.py
        torch_utils.py
        serialization.py

  tests/
    fixtures/
      configs/
      images/
      manifests/
      checkpoints/
      parquet/
      ifcb_bins/

    unit/
      config_schemas/
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

    integration/
      test_train_supervised.py
      test_train_ssl_dino_v2.py
      test_ssl_eval_callbacks.py
      test_supervised_holdout_eval.py
      test_snapshot_ensemble.py
      test_export_pt.py
      test_export_onnx.py
      test_hydra_multirun_config.py
```

## 3.1 Config folders vs experiment configs

Most files under `configs/` are reusable config groups.

Users should usually create or modify files under:

```text
configs/example_experiments/
```

A runnable experiment config composes reusable config groups.

Example:

```yaml
# configs/example_experiments/ifcb/experimentA.yaml
defaults:
  - /task: supervised
  - /data: csv_s3
  - /transforms: bucketed_aspect_size
  - /backbone: torchvision/resnet50
  - /optimizer: adamw
  - /scheduler: cosine
  - /logging: local
  - /results: canonical_parquet
  - /ensemble: disabled
  - _self_

experiment:
  name: ifcb_experimentA

data:
  manifest_uri: s3://bucket/ifcb/train.csv
  image_uri_column: filename
  sample_id_column: sample_id
  split_column: split

  targets:
    species:
      column: class_idx
      type: multiclass
```

Then run:

```bash
dojo train supervised experiment=ifcb/experimentA
```

---

# 4. Dependency plan

## 4.1 Core dependencies

Recommended core dependencies:

```text
torch
torchvision
lightning
torchmetrics
pydantic
hydra-core
omegaconf
pandas
pyarrow
numpy
pillow
scikit-learn
amplify-storage-utils
```

`amplify-storage-utils` should be a core dependency because it provides local/object-store abstractions beyond S3.

## 4.2 Optional extras

```toml
[project.optional-dependencies]

ssl = [
  "lightly",
]

timm = [
  "timm",
]

onnx = [
  "onnx",
  "onnxruntime",
]

aim = [
  "aim",
]

mlflow = [
  "mlflow",
]

s3 = [
  "amplify-storage-utils[s3]",
]

hdf = [
  "h5py",
  "tables",
]

dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]
```

Notes:

- `ssl` contains Lightly.
- `timm` keeps timm optional.
- `onnx` keeps export dependencies optional.
- `aim` and `mlflow` are optional logger dependencies.
- `s3` enables S3-compatible storage through `amplify-storage-utils`.
- `hdf` enables HDF/HDF5 result exports.

---

# 5. Config architecture

## 5.1 Pydantic is the schema source of truth

Hydra config files are inputs.

Pydantic models define the valid contract.

Core modules receive validated Pydantic configs.

```python
def train_supervised(cfg: ExperimentConfig) -> RunResult:
    ...
```

## 5.2 Root config shape

Illustrative schema:

```python
class ExperimentConfig(BaseModel):
    experiment: ExperimentMetadataConfig
    task: TaskConfig
    data: DataConfig
    transforms: TransformConfig
    backbone: BackboneConfig
    embedding_adapter: EmbeddingAdapterConfig | None = None
    heads: dict[str, HeadConfig] | None = None
    objectives: dict[str, ObjectiveConfig] | None = None
    ssl: SSLConfig | None = None
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None
    training: TrainingConfig
    logging: LoggingConfig
    artifacts: ArtifactConfig
    results: ResultsConfig
    ensemble: EnsembleConfig | None = None
    export: ExportConfig | None = None
    seed: int = 13
```

## 5.3 Experiment config composition

Example root experiment config:

```yaml
defaults:
  - /task: supervised
  - /data: csv_s3
  - /transforms: bucketed_aspect_size
  - /backbone: torchvision/resnet50
  - /optimizer: adamw
  - /scheduler: cosine
  - /logging: local
  - /results: canonical_parquet
  - /ensemble: disabled
  - _self_

experiment:
  name: ifcb_resnet50_baseline
  tags:
    - ifcb
    - supervised

seed: 13
```

## 5.4 CLI overrides

Examples should use `experiment=...` style.

```bash
dojo train supervised experiment=ifcb/experimentA
```

```bash
dojo train supervised \
  experiment=ifcb/experimentA \
  optimizer.lr=3e-4
```

```bash
dojo train ssl \
  experiment=ifcb/dino_v2_ssl \
  ssl_eval.unlabeled.embedding_diagnostics.schedule.every_fractional_epoch=0.10
```

## 5.5 Resolved config artifact

Every run should save the fully resolved and validated config:

```text
runs/{run_id}/config/resolved.yaml
runs/{run_id}/config/resolved.json
```

This config should be sufficient to reproduce the run.

---

# 6. Dataset architecture

## 6.1 Dataset backends

Initial supported dataset backends:

```text
csv
parquet_manifest
parquet_images
ifcb_bins
```

Future backend:

```text
webdataset
```

## 6.2 Shared sample contract

All datasets should return a common record shape.

```python
class SampleRecord(BaseModel):
    sample_id: str
    uri: str | None = None
    image: Any
    targets: dict[str, Any] = {}
    tabular: dict[str, Any] = {}
    source_extra: dict[str, Any] = {}
```

Hot-path DataLoader batches may use lightweight dictionaries/dataclasses rather than Pydantic validation on every batch.

## 6.3 CSV datasets

CSV manifests should support:

```text
filename
sample_id
one or more target columns
optional tabular feature columns
optional source/context columns
```

Example:

```csv
sample_id,filename,species_idx,quality_idx,biomass,equivalent_diameter_um,split
abc123,s3://bucket/images/abc123.png,42,0,1.25,18.2,train
abc124,/data/images/abc124.png,7,1,0.80,12.9,train
```

Config example:

```yaml
data:
  backend: csv
  manifest_uri: s3://bucket/manifests/train.csv
  image_uri_column: filename
  sample_id_column: sample_id
  split_column: split

  targets:
    species:
      column: species_idx
      type: multiclass

    quality:
      column: quality_idx
      type: multiclass

    biomass:
      column: biomass
      type: regression

  tabular_features:
    numeric:
      - equivalent_diameter_um
```

## 6.4 Parquet datasets

Two Parquet modes should be supported.

### Mode A: Parquet manifest

Parquet contains image paths and metadata.

```text
sample_id
image_uri
species_idx
quality_idx
tabular features...
```

### Mode B: Parquet images

Parquet contains encoded image bytes or arrays.

```text
sample_id
image_bytes
image_format
species_idx
quality_idx
tabular features...
```

## 6.5 IFCB bins dataset

The refactor should preserve current Dojo IFCB bins functionality through a new dataset/datamodule equivalent:

```text
src/dojo/data/datasets/ifcb_bins_dataset.py
src/dojo/data/datamodules/ifcb_bins.py
```

The dataset should conform to the shared sample contract.

Example:

```yaml
data:
  backend: ifcb_bins
  root_uri: s3://bucket/ifcb_bins/
  manifest_uri: s3://bucket/ifcb_bins/manifest.csv
  image_uri_column: roi_uri
  sample_id_column: roi_id

  targets:
    species:
      column: class_idx
      type: multiclass
```

## 6.6 Storage using `amplify-storage-utils`

Use `amplify-storage-utils` as a core dependency for local storage abstraction.

S3 support should be optional through `[s3]`.

Dojo should still define a Dojo-specific storage-facing interface:

```python
class StorageResolver:
    def open_bytes(self, uri: str) -> bytes: ...
    def localize(self, uri: str) -> Path: ...
    def write_bytes(self, uri: str, data: bytes) -> None: ...
    def exists(self, uri: str) -> bool: ...
```

Implementation relationship:

```text
Dojo datasets / training / export code
        ↓
Dojo StorageResolver
        ↓
amplify-storage-utils ObjectStore
        ↓
filesystem / cache / zip / sqlite / optional S3
```

Config example:

```yaml
storage:
  backend: amplify

  cache:
    enabled: true
    location: /tmp/dojo-cache

  stores:
    default:
      type: filesystem
      root: ./data

    remote:
      type: s3
      bucket: whoi-bucket
```

Datasets should depend only on `StorageResolver`, not directly on `amplify-storage-utils`.

---

# 7. Transform and preprocessing architecture

## 7.1 Transform philosophy

Avoid hard-coding domain-specific transform modules where YAML composition can express the behavior.

Python should provide reusable transform primitives.

YAML should compose those primitives into experiment-specific transform pipelines.

```text
Python primitives = reusable operations
YAML configs       = experiment/domain-specific recipes
```

## 7.2 Transform primitives

Initial primitives:

```text
letterbox
aspect_bucket
size_bucket
foreground_crop
grayscale_repeat
normalization
random_crop
random_rotation
random_flip
brightness_contrast
gaussian_blur
gaussian_noise
to_tensor
```

## 7.3 Transform pipeline example

```yaml
transforms:
  image_mode: grayscale_repeat3

  pipeline:
    - name: foreground_crop
      enabled: true
      method: threshold_bbox
      expand_margin_fraction: 0.15

    - name: bucketed_resize
      bucket_by:
        - aspect_ratio
        - native_long_side
      buckets:
        - name: small_square
          min_aspect: 0.75
          max_aspect: 1.33
          max_native_long_side: 96
          canvas_size: [96, 96]

        - name: standard_square
          min_aspect: 0.75
          max_aspect: 1.33
          min_native_long_side: 97
          canvas_size: [224, 224]

        - name: wide
          min_aspect: 1.33
          max_aspect: 3.0
          canvas_size: [224, 448]

    - name: random_rotation
      mode: multiples_of_90
      p: 0.5

    - name: horizontal_flip
      p: 0.5

    - name: normalize
      mode: dataset
      mean: [0.42, 0.42, 0.42]
      std: [0.18, 0.18, 0.18]
```

## 7.4 Aspect and size buckets

Bucketing should support both:

```text
aspect-ratio buckets
size-aware buckets
```

Rationale:

- aspect buckets preserve morphology for long/thin organisms
- size buckets help avoid artificially resizing very small plankton to appear as large as larger organisms

Config should support:

```yaml
transforms:
  resize:
    policy: bucketed
    bucket_by:
      - aspect_ratio
      - native_long_side
```

Important scale-related columns should be recordable as explicit result columns when relevant:

```text
native_width_px
native_height_px
input_width_px
input_height_px
resize_bucket
microns_per_pixel
```

## 7.5 Tabular metadata as model input

Some size/shape/acquisition features may be useful model inputs.

Examples:

```text
equivalent_diameter_um
major_axis_um
minor_axis_um
bbox_area_px
microns_per_pixel
native_width_px
native_height_px
```

These should be provided as tabular features to heads/compositors.

Config example:

```yaml
heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    inputs:
      image_embedding: true
      tabular_features:
        enabled: true
        numeric:
          - equivalent_diameter_um
          - major_axis_um
          - minor_axis_um
        encoder:
          type: mlp
          hidden_dims: [32]
          output_dim: 32
          normalization: standard
```

Model flow:

```text
image → backbone → image embedding
tabular columns → tabular encoder → tabular embedding
image embedding + tabular embedding → fused embedding → head
```

Exported model artifacts must include tabular feature names, ordering, encodings, and normalization statistics.

---

# 8. Backbone architecture

## 8.1 Backbone contract

All backbones should expose a consistent feature-extraction interface.

```python
class Backbone(nn.Module):
    output_dim: int

    def forward_features(self, x: Tensor) -> Tensor:
        ...
```

The returned tensor should usually be:

```text
batch_size x embedding_dim
```

## 8.2 Supported backbone sources

```text
torchvision
timm
checkpoint
custom registered class
```

## 8.3 Torchvision backbones

Examples:

```text
resnet50
efficientnet_b0
inception_v3
vit_b_16
convnext_tiny
```

Config:

```yaml
backbone:
  source: torchvision
  name: resnet50
  pretrained: true
  weights: DEFAULT
  output_dim: auto

  freeze:
    policy: none
```

## 8.4 timm backbones

Config:

```yaml
backbone:
  source: timm
  name: vit_small_patch16_224
  pretrained: true
  output_dim: auto

  freeze:
    policy: last_n_blocks_trainable
    n: 2
```

## 8.5 `output_dim: auto`

`backbone.output_dim: auto` should be the default and should usually not be manually changed.

It means Dojo infers the backbone’s native feature dimension from the selected architecture.

Examples:

```text
resnet50          → 2048
resnet18          → 512
efficientnet_b0   → 1280
vit_b_16          → 768
vit_s             → 384
convnext_tiny     → 768
```

If users want a different downstream embedding size, that should be modeled as an explicit embedding adapter/projection layer after the backbone, not as an override of the backbone’s native output dimension.

```yaml
backbone:
  source: timm
  name: vit_small_patch16_224
  pretrained: true
  output_dim: auto

embedding_adapter:
  enabled: true
  type: mlp
  hidden_dims: [512]
  output_dim: 256
  activation: gelu
  dropout: 0.1
```

## 8.6 Checkpoint transfer learning

Config:

```yaml
backbone:
  source: checkpoint
  architecture:
    source: timm
    name: vit_small_patch16_224
    pretrained: false

  checkpoint_uri: s3://bucket/runs/ssl_dino_v2/exports/encoder.pt
  checkpoint_key: encoder_state_dict
  strict: false

  freeze:
    policy: last_n_blocks_trainable
    n: 4
```

## 8.7 Freeze policies

Keep `freeze` as the operative config section.

Policies should be named in terms of what remains trainable.

Recommended policies:

```text
none
all
last_n_blocks_trainable
named_modules_trainable
named_modules_frozen
after_module_trainable
before_module_trainable
```

Examples:

```yaml
backbone:
  freeze:
    policy: none
```

Train all backbone parameters.

```yaml
backbone:
  freeze:
    policy: all
```

Freeze all backbone parameters.

```yaml
backbone:
  freeze:
    policy: last_n_blocks_trainable
    n: 2
```

Freeze all except the final 2 blocks.

```yaml
backbone:
  freeze:
    policy: named_modules_trainable
    names:
      - layer4
      - blocks.10
      - blocks.11
```

Freeze everything except listed modules.

```yaml
backbone:
  freeze:
    policy: after_module_trainable
    module: layer3
    inclusive: true
```

Freeze modules before `layer3`; train `layer3` and everything after it.

```yaml
backbone:
  freeze:
    policy: before_module_trainable
    module: layer3
    inclusive: false
```

Train everything before `layer3`; freeze `layer3` and everything after it.

## 8.8 Inspect command

Add:

```bash
dojo inspect backbone backbone=torchvision/resnet50
```

```bash
dojo inspect backbone \
  backbone=torchvision/resnet50 \
  backbone.freeze.policy=after_module_trainable \
  backbone.freeze.module=layer3 \
  backbone.freeze.inclusive=true
```

Output should include:

```text
module name
module type
parameter count
trainable/frozen status
suggested block/stage names
output embedding dim
```

Example output:

```text
Backbone: torchvision/resnet50

stem                       frozen       params=9,536
layer1                     frozen       params=215,808
layer2                     frozen       params=1,219,584
layer3                     trainable    params=7,098,368
layer4                     trainable    params=14,964,736
avgpool                    n/a
embedding_dim              2048
```

## 8.9 Inception and special models

Inception is a special case because of auxiliary logits and historical training conventions.

Initial refactor should not overfit the generic path around Inception-specific aux-logit behavior.

Support basic feature extraction if practical, but it is acceptable to skip special auxiliary-logit handling in the first implementation to keep the main backbone/head path clean.

---

# 9. Head, objective, and loss architecture

## 9.1 Head contract

Heads define output structure and semantics.

They should not own loss configuration.

Supported head task types:

```text
multiclass_classification
binary_classification
multilabel_classification
regression
ordinal_regression
distributional_regression
count_regression
```

## 9.2 Multi-head model

A supervised model should have one backbone and one or more heads.

```text
image
  ↓
backbone
  ↓
embedding
  ↓
├── species head
├── quality head
├── biomass regression head
└── life-stage ordinal head
```

## 9.3 Classification head

```yaml
heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    network:
      type: linear
```

Compatible losses include:

```text
cross_entropy
weighted_cross_entropy
class_balanced_effective_number
focal
label_smoothing_cross_entropy
```

## 9.4 Regression head

```yaml
heads:
  biomass:
    type: regression
    target_column: biomass
    output_dim: 1
    network:
      type: linear
```

Regression should be modeled as:

```text
regression head
+ target transform
+ output activation, if needed
+ loss
```

rather than separate head classes for every regression scale.

Common target transforms:

```text
identity
standardize
log1p
log1p_standardize
power / Box-Cox / Yeo-Johnson
```

Common losses:

```text
mse
mae
huber
smooth_l1
gaussian_nll
poisson_nll
negative_binomial_nll
quantile
```

Examples:

```yaml
objectives:
  biomass:
    head: biomass
    target_transform:
      type: log1p_standardize
    loss:
      type: huber
      delta: 1.0
    weight: 0.10
```

For positive-only values, prefer log transforms first. Bounded output activations are optional when the target has a true physical range.

## 9.5 Ordinal regression head

```yaml
heads:
  life_stage:
    type: ordinal_regression
    target_column: stage_idx
    num_classes: 5
    network:
      type: linear
```

Compatible losses include:

```text
coral
corn
ordinal_cross_entropy
```

## 9.6 Objectives bind heads to losses, weights, and metrics

Use an explicit `objectives` layer.

Heads define output structure.

Objectives define training intent.

```yaml
heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120

  quality:
    type: multiclass_classification
    target_column: quality_idx
    num_classes: 4

  biomass:
    type: regression
    target_column: biomass
    output_dim: 1

  life_stage:
    type: ordinal_regression
    target_column: stage_idx
    num_classes: 5

objectives:
  species:
    head: species
    loss:
      type: class_balanced_effective_number
      beta: 0.999
    weight: 1.0
    metrics:
      - macro_f1
      - weighted_f1
      - per_class_f1

  quality:
    head: quality
    loss:
      type: cross_entropy
    weight: 0.25
    metrics:
      - accuracy
      - macro_f1

  biomass:
    head: biomass
    target_transform:
      type: log1p_standardize
    loss:
      type: huber
      delta: 1.0
    weight: 0.10
    metrics:
      - mae
      - rmse

  life_stage:
    head: life_stage
    loss:
      type: coral
    weight: 0.25
    metrics:
      - ordinal_mae
      - accuracy
```

Total loss is implicitly a weighted sum of objective losses:

```text
total_loss =
  1.0  * species_loss
+ 0.25 * quality_loss
+ 0.10 * biomass_loss
+ 0.25 * life_stage_loss
```

Do not add a separate `loss_aggregation_config` initially. Weighted sum is the default and only phase-1 behavior.

## 9.7 Objective shorthand

For simple configs, allow objective name to imply head name.

```yaml
objectives:
  species:
    loss:
      type: focal
      gamma: 2.0
    weight: 1.0
```

Resolved internally as:

```yaml
objectives:
  species:
    head: species
    loss:
      type: focal
      gamma: 2.0
    weight: 1.0
```

## 9.8 Pydantic validation

Validation should enforce:

```text
every objective references an existing head
loss is compatible with referenced head type
metrics are compatible with referenced head type
target transforms are compatible with objective/head type
objective weights are non-negative
at least one objective is enabled for supervised training
```

Invalid example:

```yaml
objectives:
  bad_objective:
    head: species
    loss:
      type: huber
```

This should fail because Huber is not valid for a multiclass classification head.

## 9.9 Single-head vs multi-head implementation

Do not create separate `OneHeadSupervisedModel` or `SimpleSupervisedTaskModule` classes.

A single-head model is a special case of the generic supervised model with exactly one configured head.

Use one internal representation:

```yaml
heads:
  species: ...

objectives:
  species:
    head: species
    loss: ...
```

A user-facing shorthand config may be allowed, but it should normalize to the canonical multi-head/objective structure.

---

# 10. Supervised training architecture

## 10.1 LightningModule

The supervised LightningModule should own:

```text
forward pass orchestration
training_step
validation_step
test_step
optimizer/scheduler creation
metric updates
objective loss computation
weighted objective loss sum
```

It should not own:

```text
dataset-specific path logic
experiment tracker-specific logic
artifact layout decisions
snapshot bundling implementation
export implementation
result file serialization
```

Suggested class:

```python
class SupervisedTaskModule(L.LightningModule):
    def __init__(
        self,
        model: SupervisedModel,
        objectives: ObjectiveCollection,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
        metric_collection: ObjectiveMetricCollection | None = None,
    ):
        ...
```

Where:

```text
model
  = backbone + optional embedding adapter + optional tabular encoder + heads

objectives
  = binds head → target → target transform → loss → weight → metrics

optimizer_config / scheduler_config
  = consumed in configure_optimizers()
```

Snapshot ensembles, EMA/SWA, checkpointing, logging, artifact writing, result writing, and export should remain outside the module as callbacks, trainer setup, writers, or post-training commands.

## 10.2 Model composition

```python
backbone = build_backbone(cfg.backbone)
adapter = build_embedding_adapter(cfg.embedding_adapter, backbone.output_dim)
heads = build_heads(cfg.heads, input_dim=effective_embedding_dim)
model = SupervisedModel(backbone=backbone, adapter=adapter, heads=heads)
```

If tabular features are configured:

```python
tabular_encoder = build_tabular_encoder(cfg.data.tabular_features, cfg.heads)
model = SupervisedModel(
    backbone=backbone,
    adapter=adapter,
    tabular_encoder=tabular_encoder,
    heads=heads,
)
```

## 10.3 Training command

Use experiment config style.

```bash
dojo train supervised experiment=ifcb/experimentA
```

With overrides:

```bash
dojo train supervised \
  experiment=ifcb/experimentA \
  optimizer.lr=3e-4
```

## 10.4 Supervised output artifacts and results

Training outputs should distinguish between:

```text
checkpoints  = training/resume artifacts
exports      = portable model artifacts
metrics      = aggregate metrics
results      = sample-level and head-level outputs
```

### Checkpoints

Do not assume only `last.ckpt` and `best.ckpt`.

Checkpoint outputs are determined by checkpoint callback configuration and may include:

```text
best-k checkpoints
last checkpoint
epoch checkpoints
step checkpoints
snapshot checkpoints
EMA/SWA checkpoints
manual checkpoints
```

Portable `model_best.pt` / `model_final.pt` files should not be automatic training checkpoints. These belong under `exports/` and should be produced by explicit export configuration or `dojo export`.

### Recommended run layout

```text
runs/{run_id}/
  config/
    resolved.yaml
    resolved.json

  checkpoints/
    # Lightning/training checkpoints according to checkpoint callback config
    last.ckpt
    epoch=004-val_macro_f1=0.842.ckpt
    epoch=009-val_macro_f1=0.861.ckpt
    snapshot_001.ckpt
    snapshot_002.ckpt

  exports/
    # Created only by explicit export command/callback
    model.pt
    model.onnx
    snapshot_ensemble.pt
    metadata.json

  metrics/
    train_metrics.json
    val_metrics.json
    test_metrics.json
    per_class_metrics.parquet
    confusion_matrix.parquet

  results/
    # Canonical tall Parquet
    val_results.parquet
    test_results.parquet
    infer_results.parquet
    val_results_by_epoch.parquet

    # Optional convenience exports
    val_results_wide.csv
    test_results_wide.csv
    embeddings.csv
    confusion_matrix.csv
    results.h5
```

### Canonical result format

Canonical result format should be tall Parquet.

It should be compatible with improv-style schemas/provenance concepts.

A single sample may appear multiple times, for example:

```text
one row for embedding output
one row for species head output
one row for quality head output
one row for biomass head output
one row for nearest-neighbor output
```

Core columns should be explicit.

Avoid a generic `metadata` column in canonical results.

Recommended canonical columns:

```text
sample_id
uri
split
epoch
global_step
checkpoint_id
model_artifact_id
record_type
head_name
target
prediction_index
prediction_label
prediction_value
logits
scores
confidence
embedding
distance
neighbor_sample_id
neighbor_rank
cluster_id
projection_x
projection_y
resize_bucket
native_width_px
native_height_px
input_width_px
input_height_px
source_extra_json
```

`source_extra_json` should be optional and used only for dataset-specific passthrough fields that are not part of the core Dojo schema.

Important context used by Dojo should be promoted to explicit columns.

### Result record types

Potential `record_type` values:

```text
embedding
classification_output
regression_output
ordinal_output
target
nearest_neighbor
cluster_assignment
projection
outlier_score
diagnostic
```

### Convenience outputs

In addition to canonical tall Parquet, Dojo should optionally produce:

```text
wide CSV summary
embeddings CSV
confusion matrix CSV
HDF/HDF5 export
```

Wide CSV should be one sample per row and should generally exclude embeddings.

Example wide columns:

```text
sample_id
uri
split
species_target
species_prediction_index
species_prediction_label
species_confidence
quality_prediction_index
quality_prediction_label
biomass_prediction_value
```

Embeddings CSV should be one sample per row.

Confusion matrix CSV should be available for classification heads.

HDF/HDF5 should be treated as an optional export format derived from canonical results.

### Epoch-level validation results

It should be possible to output results for:

```text
final exported model
selected checkpoint
every validation epoch
every N epochs
every N batches
```

`val_results_by_epoch.parquet` should include:

```text
epoch
global_step
checkpoint_id
sample_id
record_type
head_name
...
```

This supports:

```text
learning dynamics
tail-class stability analysis
checkpoint selection
snapshot selection
ensemble selection
```

Because this may become large, it should be configurable.

### Result output config

Result output behavior should be fully configurable in YAML.

Example:

```yaml
results:
  canonical:
    enabled: true
    format: parquet
    layout: tall
    improv_compatible: true
    include:
      embeddings: true
      logits: true
      scores: true
      predictions: true
      targets: true
      input_shape_context: true

  validation_by_epoch:
    enabled: true
    every_n_epochs: 1
    include_embeddings: false
    include_logits: true
    include_scores: true

  exports:
    wide_csv:
      enabled: true
      exclude_embeddings: true

    embeddings_csv:
      enabled: true

    confusion_matrix_csv:
      enabled: true

    hdf:
      enabled: false
```

---

# 11. Self-supervised training architecture

## 11.1 SSL framework

Use Lightly only for SSL method implementations.

Do not depend directly on Meta DINOv2 repositories.

The initial SSL task should focus on DINOv2-style training using Lightly components.

## 11.2 SSL model structure

```text
image views
  ↓
backbone encoder
  ↓
projection head
  ↓
DINOv2-style SSL loss
```

After SSL training:

```text
encoder is used for downstream embeddings/transfer
projection head may be exported for reproducibility/debugging
teacher/student states may be preserved in ssl_model.pt if configured
```

## 11.3 SSL task module

Suggested class:

```python
class DinoV2SSLTaskModule(L.LightningModule):
    def __init__(
        self,
        ssl_model: SSLModel,
        ssl_objective: SSLObjective,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None,
        eval_config: SSLEvaluationConfig,
    ):
        ...
```

## 11.4 SSL config

```yaml
ssl:
  method: dino_v2
  framework: lightly

  teacher:
    momentum_start: 0.996
    momentum_end: 1.0

  student:
    projection_dim: 65536
    hidden_dim: 2048
    bottleneck_dim: 256

  loss:
    type: dino_v2
    teacher_temperature: 0.04
    student_temperature: 0.1
    center_momentum: 0.9

  output:
    save_encoder_export: true
    save_ssl_model_export: false
```

## 11.5 SSL training command

Use experiment config style.

```bash
dojo train ssl experiment=ifcb/dino_v2_ssl
```

With override:

```bash
dojo train ssl \
  experiment=ifcb/dino_v2_ssl \
  training.max_epochs=300
```

## 11.6 SSL artifacts

SSL training outputs should follow the same conventions as supervised training:

```text
checkpoints  = training/resume artifacts
exports      = portable encoder/model artifacts
metrics      = aggregate metrics
results      = canonical evaluation/embedding outputs
```

### Recommended SSL run layout

```text
runs/{run_id}/
  config/
    resolved.yaml
    resolved.json

  checkpoints/
    # Lightning/training checkpoints according to checkpoint callback config
    last.ckpt
    epoch=049-ssl_loss=1.84.ckpt
    epoch=099-knn_macro_f1=0.63.ckpt

  exports/
    # Created only by explicit export command/callback
    encoder.pt
    encoder.onnx
    ssl_model.pt
    metadata.json

  metrics/
    ssl_train_metrics.json
    ssl_eval_knn_metrics.json
    ssl_eval_linear_probe_metrics.json
    ssl_embedding_diagnostics.json

  results/
    # Canonical tall-format Parquet outputs
    ssl_eval_results.parquet
    ssl_eval_results_by_epoch.parquet
    embedding_results.parquet

    # Optional convenience outputs
    ssl_eval_wide.csv
    embedding_wide.csv
    results.h5
```

### `.ckpt` vs `encoder.pt` vs `ssl_model.pt`

A Lightning checkpoint is for resuming training and may include:

```text
student encoder weights
teacher encoder weights
projection heads
optimizer state
scheduler state
epoch
global step
Lightning loop state
callback state
mixed precision scaler
```

`encoder.pt` is a portable artifact for downstream use:

```text
embedding extraction
transfer learning
attention visualization
fine-tuning
```

`ssl_model.pt` is a portable SSL model artifact that may include:

```text
encoder state
projection head state
teacher encoder state
student encoder state
SSL method config
preprocessing metadata
```

Use `ssl_model.pt` when SSL-specific internals are needed.

Use `encoder.pt` for most downstream classification, embedding extraction, and attention visualization.

### SSL result records

SSL canonical results should use the same tall Parquet result model as supervised outputs.

SSL-specific record types may include:

```text
embedding
nearest_neighbor
knn_prediction
linear_probe_prediction
cluster_assignment
projection
outlier_score
embedding_diagnostic
```

Recommended columns include:

```text
sample_id
uri
split
epoch
global_step
checkpoint_id
model_artifact_id
record_type
evaluation_name
head_name
embedding
target
prediction_index
prediction_label
scores
confidence
distance
neighbor_sample_id
neighbor_rank
cluster_id
projection_x
projection_y
resize_bucket
native_width_px
native_height_px
input_width_px
input_height_px
source_extra_json
```

Avoid generic `metadata` in canonical schemas. Use explicit columns where Dojo understands the field.

---

# 12. SSL evaluation

## 12.1 Evaluation categories

SSL evaluation should support three categories:

```text
label-required evaluation
label-free evaluation
visual/diagnostic evaluation
```

## 12.2 Label-required SSL evaluation

When a labeled evaluation dataset is available, support:

```text
k-NN classification
linear probe
supervised fine-tuning evaluation
macro/per-class metrics
confusion matrix
```

Config example:

```yaml
ssl_eval:
  labeled:
    enabled: true
    dataset:
      backend: csv
      manifest_uri: s3://bucket/manifests/val_labeled.csv
      image_uri_column: filename
      sample_id_column: sample_id
      targets:
        species:
          column: species_idx
          type: multiclass

    knn:
      enabled: true
      k: 20
      distance: cosine
      schedule:
        every_fractional_epoch: 0.10

    linear_probe:
      enabled: true
      max_epochs: 10
      schedule:
        every_n_epochs: 10
```

## 12.3 Label-free SSL evaluation

When no labeled dataset is available, support:

```text
embedding extraction
embedding collapse checks
embedding variance/covariance diagnostics
nearest-neighbor retrieval
augmentation consistency
clustering
dimensionality reduction
density/outlier scoring
```

These do not produce classification F1, but they help detect whether representation learning is useful or collapsed.

### Embedding diagnostics

Useful during training:

```text
embedding_norm_mean
embedding_norm_std
per_dimension_std
effective_rank
covariance_condition
pairwise_cosine_mean
pairwise_cosine_std
```

### Augmentation consistency

Compare embeddings from multiple valid augmented views of the same image.

```text
cosine_similarity(z_view1, z_view2)
```

### Retrieval panels

Nearest-neighbor retrieval is useful for visual inspection.

Records:

```text
query_sample_id
neighbor_sample_id
neighbor_rank
distance
epoch
checkpoint_id
```

Optional figure outputs:

```text
figures/retrieval_panels/epoch_010/*.png
```

### Clustering

Support:

```text
kmeans
mini_batch_kmeans
hdbscan, optional
agglomerative, optional
```

Outputs:

```text
sample_id
cluster_id
cluster_distance
epoch
checkpoint_id
```

If labels later become available, NMI/ARI can be computed retrospectively.

### Dimensionality reduction

Support:

```text
PCA
UMAP
t-SNE
```

Outputs:

```text
sample_id
projection_x
projection_y
projection_method
epoch
checkpoint_id
```

Example files:

```text
results/embedding_projection_umap.parquet
results/embedding_projection_pca.parquet
figures/umap_epoch_050.png
```

## 12.4 Unlabeled SSL eval config

```yaml
ssl_eval:
  unlabeled:
    enabled: true
    dataset:
      backend: parquet_images
      uri: s3://bucket/plankton/unlabeled/*.parquet

    embedding_diagnostics:
      enabled: true
      schedule:
        every_fractional_epoch: 0.10

    retrieval:
      enabled: true
      num_queries: 64
      k: 12
      distance: cosine
      schedule:
        every_n_epochs: 5

    clustering:
      enabled: true
      methods:
        - type: mini_batch_kmeans
          n_clusters: 100
      schedule:
        every_n_epochs: 10

    dimensionality_reduction:
      enabled: true
      methods:
        - type: pca
          n_components: 2
        - type: umap
          n_components: 2
      max_samples: 50000
      schedule:
        every_n_epochs: 10
```

## 12.5 Evaluation scheduling

Evaluations should be schedulable by:

```text
every N epochs
every N train batches
every fractional epoch
end of epoch
end of training
```

Example:

```yaml
ssl_eval:
  labeled:
    knn:
      schedule:
        every_fractional_epoch: 0.10

    linear_probe:
      schedule:
        every_n_epochs: 10

  unlabeled:
    embedding_diagnostics:
      schedule:
        every_fractional_epoch: 0.10

    retrieval:
      schedule:
        every_n_epochs: 5
```

`every_fractional_epoch: 0.10` means approximately every 10% of an epoch.

Expensive evaluations should support subsampling:

```yaml
ssl_eval:
  labeled:
    knn:
      max_reference_samples: 50000
      max_query_samples: 10000
```

## 12.6 Standalone vs training-integrated evaluation

Evaluation should exist as reusable evaluator modules with two execution modes:

```text
training-integrated callbacks
standalone CLI wrappers
```

The same underlying evaluator code should be used by:

```bash
dojo eval knn
dojo eval linear-probe
dojo eval embeddings
```

and by:

```yaml
ssl_eval:
  labeled:
    knn:
      enabled: true
```

---

# 13. Ensemble and snapshot architecture

## 13.1 Concept

Snapshot/checkpoint ensembling is a supervised model-selection and inference workflow.

It should be implemented as:

```text
checkpoint discovery
checkpoint/model selection
ensemble construction
ensemble evaluation
artifact bundling
```

not as a separate training model architecture.

## 13.2 Generic ensemble command

```bash
dojo ensemble experiment=ifcb/ensemble_search
```

Purpose:

```text
given many checkpoints/models
find which combination produces best validation result
under configured computational constraints
```

Inputs may include:

```text
checkpoints from one run
snapshots from one run
best-k checkpoints
checkpoints from different runs
exported .pt models
candidate soups
```

Selection strategies:

```text
top_k
greedy_forward_selection
greedy_soup
snapshot_cycle_selection
diversity_aware_selection
budget_constrained_selection
```

Budget constraints:

```text
max_models
max_latency_ms
max_file_size_mb
max_memory_mb
```

## 13.3 Snapshot-specific command

```bash
dojo ensemble snapshot experiment=ifcb/snapshot_ensemble
```

Specialized for snapshots from one training run.

## 13.4 Snapshot ensemble config

```yaml
ensemble:
  enabled: true
  type: snapshot

  discovery:
    run_uri: s3://bucket/runs/run123
    checkpoint_glob: checkpoints/snapshot_*.ckpt

  selection:
    strategy: top_k
    k: 5
    metric: val/species/macro_f1
    mode: max

  artifact:
    bundle_single_pt: true
    bundle_filename: snapshot_ensemble.pt

  inference:
    combine: logits_mean
```

## 13.5 Snapshot artifact format

The bundled `.pt` file should contain:

```python
{
    "artifact_type": "snapshot_ensemble",
    "format_version": "1.0",

    "model_config": {...},
    "preprocessing": {...},
    "heads": {...},
    "class_mappings": {...},

    "snapshots": [
        {
            "name": "snapshot_001",
            "epoch": 20,
            "global_step": 12345,
            "metric": 0.842,
            "state_dict": {...},
        },
        {
            "name": "snapshot_002",
            "epoch": 40,
            "global_step": 24690,
            "metric": 0.858,
            "state_dict": {...},
        },
    ],

    "combine": "logits_mean",
    "created_at": "...",
    "dojo_version": "...",
    "source_run_id": "...",
}
```

## 13.6 Ensemble inference

For classification:

```text
ensemble_logits = mean(snapshot_logits)
probabilities = softmax(ensemble_logits)
prediction = argmax(probabilities)
```

For regression:

```text
ensemble_prediction = mean(snapshot_prediction)
```

For ordinal regression:

```text
average ordinal logits
decode ordinal prediction
```

## 13.7 Ensemble outputs

```text
runs/{run_id}/
  ensemble/
    selected_checkpoints.json
    ensemble_manifest.json
    ensemble_metrics.json

  exports/
    snapshot_ensemble.pt

  results/
    val_results.parquet
    test_results.parquet
```

---

# 14. Model soups, SWA, and EMA

The architecture should leave room for model averaging methods.

Potential modules:

```text
src/dojo/ensemble/soup.py
src/dojo/ensemble/swa.py
src/dojo/training/ema.py
```

## 14.1 Greedy soup

Greedy soup can reuse saved checkpoints and evaluate averaged weights.

This belongs in the ensemble workflow.

```bash
dojo ensemble experiment=ifcb/greedy_soup
```

## 14.2 SWA

SWA should be treated as a training callback or end-of-training phase.

## 14.3 EMA

EMA should be a training callback that maintains shadow weights.

## 14.4 Relationship to imbalance

For long-tail data, ensemble selection should support metrics such as:

```text
macro F1
tail-class recall
per-class F1
balanced accuracy
```

Selection should not default to plain accuracy for imbalanced datasets.

---

# 15. Export architecture

## 15.1 Export targets

Supported export formats:

```text
.pt single model
.pt snapshot/checkpoint ensemble
.onnx single model
.onnx ensemble wrapper, optional
```

## 15.2 `dojo export pt`

`dojo export pt` converts checkpoint(s) or ensemble definitions into portable `.pt` artifacts.

It does not decide which checkpoints belong in an ensemble unless explicitly configured.

That selection belongs to `dojo ensemble`.

Example:

```bash
dojo export pt \
  checkpoint=s3://bucket/runs/run123/checkpoints/best.ckpt \
  output=s3://bucket/runs/run123/exports/model.pt
```

For ensemble export:

```bash
dojo export pt \
  ensemble_manifest=s3://bucket/runs/run123/ensemble/ensemble_manifest.json \
  output=s3://bucket/runs/run123/exports/snapshot_ensemble.pt
```

## 15.3 Export metadata

Exported artifacts should include:

```text
model architecture
backbone source/name
head definitions
objective summary
class names
class index mappings
normalization mean/std
image mode
resize policy
bucket definitions
tabular feature names/order/stats
input shape
training config hash
source checkpoint URI
Dojo version
```

## 15.4 ONNX export

ONNX export should write:

```text
model.onnx
metadata.json
```

Metadata should also be embedded into ONNX metadata properties when possible.

For aspect/size buckets:

- CNN/ConvNeXt/ResNet may support dynamic H/W.
- ViT exports are likely more robust as one ONNX file per bucket shape.

Example metadata:

```yaml
preprocessing:
  resize_policy: bucketed
  buckets:
    - name: square
      size: [224, 224]
      min_aspect: 0.75
      max_aspect: 1.33
      onnx_model: model_224x224.onnx

    - name: wide
      size: [224, 448]
      min_aspect: 1.33
      max_aspect: 3.0
      onnx_model: model_224x448.onnx
```

---

# 16. Inference and embedding extraction

## 16.1 Inference command

Use:

```bash
dojo infer experiment=ifcb/infer_model
```

or:

```bash
dojo infer \
  model=s3://bucket/runs/run123/exports/model.pt \
  data=s3://bucket/manifests/holdout.csv
```

## 16.2 Inference outputs

Inference should optionally output:

```text
head predictions
raw logits
scores/probabilities
confidence
embeddings
distance/novelty scores
input-shape context
resize bucket
sample identifiers
```

All outputs should use the canonical results system.

## 16.3 Embedding extraction

Embedding extraction should work with:

```text
supervised checkpoints
SSL encoder exports
SSL training checkpoints
snapshot ensemble members
```

Command:

```bash
dojo eval embeddings \
  checkpoint=s3://bucket/runs/ssl/exports/encoder.pt \
  data=csv_s3 \
  results.output_uri=s3://bucket/results/embedding_results.parquet
```

---

# 17. Experiment logging

## 17.1 Logger abstraction

Create a logger interface:

```python
class ExperimentLogger:
    def log_config(self, cfg: ExperimentConfig) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_artifact(self, local_path: Path, artifact_path: str) -> None: ...
    def close(self) -> None: ...
```

Implementations:

```text
LocalExperimentLogger
AimExperimentLogger
MLflowExperimentLogger
CompositeExperimentLogger
```

## 17.2 Runtime logger selection

```yaml
logging:
  sinks:
    - type: local
      run_root: ./runs

    - type: aim
      repo: ./aim
      experiment_name: ifcb
```

or:

```yaml
logging:
  sinks:
    - type: local
      run_root: ./runs

    - type: mlflow
      tracking_uri: http://localhost:5000
      experiment_name: ifcb
```

Multiple sinks may be supported if simple.

## 17.3 Artifact policy

Artifacts should be created according to result/export/checkpoint configuration.

Logger sinks may register or upload artifacts after local creation.

---

# 18. Hyperparameter search

Use Hydra multirun only.

Example:

```bash
dojo train supervised -m \
  experiment=ifcb/experimentA \
  backbone=torchvision/resnet50,timm/convnext_tiny \
  optimizer.lr=1e-4,3e-4 \
  data.batch_size=32,64
```

Each Hydra job should:

1. Compose config.
2. Apply CLI overrides.
3. Validate with Pydantic.
4. Create run directory.
5. Save resolved config.
6. Run training/evaluation.
7. Save configured artifacts/results.
8. Log to configured sinks.

---

# 19. Testing strategy

## 19.1 Config tests

Test:

```text
Hydra config composition
Pydantic validation success
Pydantic validation failures
CLI override validation
invalid head/loss combinations
invalid objective references
invalid dataset columns
invalid logger sink configs
invalid ensemble configs
```

## 19.2 Dataset tests

Test:

```text
CSV local images
CSV S3-style paths through mocked storage resolver
Parquet manifest
Parquet image bytes
IFCB bins dataset
multi-head targets
tabular feature extraction
sample_id propagation
uri propagation
```

## 19.3 Storage tests

Test:

```text
local amplify-backed storage
cache resolver behavior
S3 optional import behavior
StorageResolver open_bytes/localize/write_bytes/exists
```

## 19.4 Transform tests

Test:

```text
letterbox preserves aspect ratio
aspect bucket assignment
size bucket assignment
foreground-aware crop
grayscale repeat-to-3
normalization
DINOv2 multi-view transform
extreme aspect ratio images
small native resolution images
```

## 19.5 Model tests

Test:

```text
torchvision backbone construction
timm backbone construction when extra installed
checkpoint backbone loading
embedding adapter
freeze policies
dojo inspect output
classification head
regression head
ordinal head
multi-head forward
tabular feature fusion
```

## 19.6 Objective/loss tests

Test:

```text
objective references existing head
loss compatibility validation
weighted objective sum
classification losses
regression losses
log target transform
ordinal losses
```

## 19.7 Supervised training smoke tests

Test:

```text
single-head supervised 1 epoch
multi-head supervised 1 epoch
tabular + image supervised 1 epoch
results written as canonical Parquet
wide CSV export
confusion matrix CSV export
checkpoint callback outputs
```

## 19.8 SSL training smoke tests

Behind `ssl` extra.

Test:

```text
DINOv2-style Lightly task for a few batches
multi-crop transform
online embedding diagnostics
online k-NN when labels are available
unlabeled retrieval output
encoder export
```

## 19.9 Ensemble tests

Test:

```text
snapshot discovery
top-k selection
greedy ensemble selection
bundle .pt creation
ensemble inference
ensemble result output
```

## 19.10 Export tests

Behind `onnx` extra where needed.

Test:

```text
single .pt export
snapshot ensemble .pt export
ONNX single-model export
ONNX metadata sidecar
bucketed ONNX metadata
```

## 19.11 Logging tests

Test:

```text
local logger
Aim logger when extra installed
MLflow logger when extra installed
multiple sinks if supported
artifact registration
```

---

# 20. Migration plan

## Phase 1: Config and CLI foundation

1. Add Hydra entrypoint.
2. Add `config_schemas`.
3. Add `dojo validate-config`.
4. Add local/amplify storage resolver.
5. Add result config schemas.
6. Add logger abstraction.

## Phase 2: Supervised refactor

1. Implement shared dataset record contract.
2. Implement CSV datamodule.
3. Implement Parquet datamodule.
4. Port IFCB bins dataset.
5. Add backbone registry.
6. Add head registry.
7. Add objectives.
8. Add supervised LightningModule.
9. Add canonical results writer.

## Phase 3: Transform refactor

1. Add primitive transform builder.
2. Add letterbox.
3. Add aspect buckets.
4. Add size buckets.
5. Add foreground-aware crop.
6. Add grayscale/normalization primitives.

## Phase 4: SSL with Lightly

1. Add `dojo[ssl]`.
2. Add Lightly DINOv2-style task.
3. Add SSL transforms.
4. Add labeled SSL eval.
5. Add unlabeled diagnostics/retrieval/clustering/projections.
6. Add encoder export.

## Phase 5: Ensemble workflows

1. Add `dojo ensemble`.
2. Add `dojo ensemble snapshot`.
3. Add checkpoint selection.
4. Add snapshot bundle artifact.
5. Add ensemble result writing.

## Phase 6: Export

1. Add `.pt` single-model export.
2. Add `.pt` snapshot ensemble export.
3. Add ONNX single-model export.
4. Add ONNX metadata.
5. Add bucket-aware ONNX export support.

## Phase 7: Logging sinks

1. Add Aim logger.
2. Add MLflow logger.
3. Add composite logger if needed.
4. Ensure result/artifact config works across sinks.

---

# 21. Design principles

## 21.1 Pydantic schemas are the contract

Hydra composes configs.

Pydantic validates and defines the runtime contract.

## 21.2 Keep task logic separate from model composition

```text
models/backbones     = feature extractors
models/heads         = output modules
models/compositors   = assemble PyTorch models
tasks                = Lightning training/evaluation objectives
```

## 21.3 Prefer canonical internal representations

Single-head supervised models should normalize into the same head/objective structure as multi-head models.

## 21.4 Results are first-class artifacts

Canonical results are tall Parquet.

CSV/HDF are configurable derived exports.

## 21.5 Avoid generic metadata junk drawers

Use explicit columns for fields Dojo understands.

Use `source_extra_json` only for optional dataset-specific passthrough fields.

## 21.6 Keep storage behind a Dojo interface

Use `amplify-storage-utils` underneath, but do not leak its API throughout datasets/training/export code.

## 21.7 Make embeddings first-class

Embeddings should be extractable and persistable from:

```text
supervised models
SSL encoders
snapshot ensemble members
transfer-learning checkpoints
```

## 21.8 Optimize for external orchestration

No Prefect dependency should be added to core.

Functions should accept validated Pydantic configs and return structured result objects.

Example:

```python
result = train_supervised(cfg)
```

where `result` includes:

```text
run_id
run_dir
checkpoint_uris
export_uris
metric_uris
result_uris
```
