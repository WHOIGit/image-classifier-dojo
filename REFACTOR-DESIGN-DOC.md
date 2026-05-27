# Image Classifier Dojo Refactor Design Doc

## Status

Draft architectural design.

## Target Repository

This design is intended for a major breaking refactor of [`WHOIGit/image-classifier-dojo`](https://github.com/WHOIGit/image-classifier-dojo).

The refactor prioritizes:

- supervised image modeling with one or more output heads
- self-supervised learning (SSL) using Lightly, initially focused on DINOv2-style workflows
- strong Pydantic schemas as the source of truth
- Hydra-based config composition and CLI overrides
- configurable experiment logging and artifact output
- snapshot/checkpoint ensembling workflows
- local/S3-capable storage using [`amplify-storage-utils`](https://github.com/WHOIGit/amplify-storage-utils)
- IFCB bin support (for SSL) using [`ifcbkit`](https://github.com/WHOIGit/ifcbkit)
- columnar, [`improv`](https://github.com/WHOIGit/improv)-compatible result output
- future Prefect orchestration without adding Prefect to the core project

---

# 1. Goals

## 1.1 Core goals

The refactored project should support:

1. Supervised image classification, regression, count regression, distributional regression, and ordinal regression.
2. Supervised models with one or more output heads.
3. Transfer learning from:
   - torchvision pretrained backbones
   - timm pretrained backbones
   - non-pretrained torchvision/timm backbones
   - local or remote checkpoints
   - SSL encoder exports
4. Self-supervised learning using Lightly, initially focused on DINOv2-style training.
5. SSL evaluation during training using labeled and/or unlabeled evaluation datasets.
6. Supervised holdout evaluation for one or more checkpoints/models.
7. Snapshot and checkpoint ensembling workflows.
8. Bundling selected checkpoint/snapshot ensembles into a single `.pt` artifact containing multiple `state_dict`s.
9. Optional export to ONNX, including preprocessing and model metadata.
10. Dataset loading from:
    - CSV manifests
    - Parquet manifests
    - Parquet image datasets
    - IFCB bin manifests
11. S3 path support for CSV/Parquet-defined image paths.
12. IFCB raw-bin support through [`ifcbkit`](https://github.com/WHOIGit/ifcbkit).
13. Use of [`amplify-storage-utils`](https://github.com/WHOIGit/amplify-storage-utils) for local storage, object-store abstraction, caching, and optional S3 support.
14. Hyperparameter search through Hydra multirun.
15. Experiment logging through:
    - local files
    - Aim
    - MLflow
    - optionally more than one sink, if configured
16. Configurable result outputs:
    - canonical tall Parquet
    - wide CSV summaries
    - embeddings CSV
    - confusion matrix CSV
    - HDF/HDF5 export

## 1.2 Non-goals for this refactor phase

The following are intentionally deferred:

1. Full WebDataset support.
2. Production model serving framework.
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

Core training/evaluation code should receive Pydantic objects, not raw Hydra `DictConfig`s.

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
exported .pt models
candidate ensembles
candidate soups
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

Evaluate possible checkpoint/model combinations and select ensemble candidates based on validation performance and computational tradeoff.

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
```

Recommended default:

```text
local artifacts & metrics enabled
experiment tracker optional
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
    - type: aim
      repo: ./aim
      experiment_name: ifcb
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
      letterbox_square.yaml
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

      patches/
        __init__.py
        model_summary_with_grad.py

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

## 3.1 Notes on structure

### `config_schemas/`

Pydantic experiment/config schemas live in:

```text
src/dojo/config_schemas/
```

### `data/record_schemas/`

Pydantic schemas for persisted records and result row formats live in:

```text
src/dojo/data/record_schemas/
```

Hot-path DataLoader batches however should still use lighter dicts/dataclasses internally.

### `models/compositors/`

Compositors assemble PyTorch model parts.

```text
backbone + optional tabular encoder + optional adapter/fusion + head(s)
```

Tasks train composited models.

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
  "ifcbkit",
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
  "ifcbkit[s3]",
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
- `s3` enables S3-compatible storage.
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
    tabular: TabularInputConfig | None = None
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

Example experiment config:

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

Hot-path DataLoader batches should use lightweight dictionaries/dataclasses rather than Pydantic validation on every batch.

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

The IFCB bins dataset is primarily for SSL/unlabeled training over raw IFCB bins.

IFCB bins are a data format containing IFCB ROIs. A manifest does not necessarily enumerate every ROI. Instead, a row may identify a bin, and the dataset expands that bin into a variable number of ROI images at runtime.

This backend should use [`ifcbkit`](https://github.com/WHOIGit/ifcbkit), which provides IFCB raw-bin parsing and image access.

IFCB raw data is organized around `.hdr`, `.adc`, and `.roi` triplets. `ifcbkit` can read all ROI images from a bin and can also read a single image by ROI ID.

Suggested modules:

```text
src/dojo/data/datasets/ifcb_bins_dataset.py
src/dojo/data/datamodules/ifcb_bins.py
```

See extant (not yet refactored) src/dojo/selfsupervised/datasets.py for fuctional example.

### Manifest shape

Preferred manifest fields:

```text
bin_id
bin_uri
optional source/context columns
```

Use:

```text
bin_id_column
bin_uri_column
```

rather than image/ROI-specific fields.

Example config for unlabeled SSL training:

```yaml
data:
  backend: ifcb_bins
  manifest_uri: s3://bucket/ifcb_bins/unlabeled_bins.parquet
  bin_id_column: bin_id
  bin_uri_column: bin_uri
```

For labeled evaluation over selected ROIs, a separate CSV/Parquet ROI manifest may still be used.

### IFCB bin expansion

The dataset should:

1. Read the bin manifest.
2. Use `bin_id_column` and `bin_uri_column` to locate each bin.
3. Use `ifcbkit` to access `.hdr` / `.adc` / `.roi` content.
4. Expand each bin into ROI image samples.
5. Produce stable ROI `sample_id`s from IFCB ROI identifiers.
6. Preserve bin-level information as explicit fields where relevant.

Recommended sample/result fields:

```text
sample_id          # ROI ID
bin_id
bin_uri
roi_number
uri                # optional ROI-level pseudo-URI
native_width_px
native_height_px
```

---

# 6.6 Storage using `amplify-storage-utils`

Use [`amplify-storage-utils`](https://github.com/WHOIGit/amplify-storage-utils) as a core dependency for local storage abstraction.

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

Datasets should depend only on `StorageResolver`, not directly on `amplify-storage-utils`.

---

# 7. Transform and preprocessing architecture

## 7.1 Transform philosophy

Avoid hard-coding domain-specific transform modules where YAML composition can express the behavior.

Python should provide reusable transform primitives.

YAML should compose those primitives into experiment-specific transform pipelines.

```text
Python transform modules = reusable operations
YAML configs             = experiment/domain-specific recipes
```

## 7.2 Transform modules

Initial transform modules:

```text
letterbox
aspect_bucket
size_bucket
foreground_crop
grayscale
normalization
crop
blur
noise
rotation
```

These live directly under:

```text
src/dojo/data/transforms/
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

Important scale-related fields should be recordable as explicit result columns when relevant:

```text
native_width_px
native_height_px
input_width_px
input_height_px
resize_bucket
microns_per_pixel
```

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

### What `strict: false` means

`strict` should map to the PyTorch `load_state_dict(..., strict=...)` behavior.

Use:

```yaml
strict: true
```

when the checkpoint must match the architecture exactly.

Use:

```yaml
strict: false
```

when transfer learning may involve missing or unexpected keys, such as:

```text
loading an encoder without its old classifier head
loading SSL encoder weights into a supervised model
loading a backbone while replacing projection/classification heads
```

When `strict: false`, Dojo should log missing and unexpected keys clearly.

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
    policy: named_modules_frozen
    names:
      - stem
      - layer1
```

Freeze only listed modules.

```yaml
backbone:
  freeze:
    policy: after_module_trainable
    module: layer3
    inclusive: true
```

Freeze modules before `layer3`; train `layer3` and everything after it.


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

The current Dojo patch [`src/dojo/patches/model_summary_patch.py`](https://github.com/WHOIGit/image-classifier-dojo/blob/main/src/dojo/patches/model_summary_patch.py) is relevant here. It extends Lightning’s model summary to include `requires_grad` state. The refactor should either preserve this functionality or turn it into the basis for `dojo inspect`.

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
optional tabular fusion / embedding adapter
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
objective loss computation
weighted objective loss sum
metric updates derived from objective definitions
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
        model_config: SupervisedModelConfig,
        objectives_config: ObjectiveCollectionConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_supervised_model(model_config)
        self.objectives = build_objectives(objectives_config)
        self.metrics = build_metrics(objectives_config)
```

Te `SupervisedTaskModule` constructor should not receive an already-instantiated `SupervisedModel`. Constructor arguments should be serializable Pydantic configs so Lightning checkpoints, hyperparameter logging, and reproducibility are clean. Lightning prefers constructor inputs that are serializable because save_hyperparameters() can store them in the checkpoint.

`objectives` contains loss definitions, objective weights, and metric definitions per head/objective. Therefore a separate `metric_collection` constructor argument is probably redundant. The module can build metrics from the validated objective definitions.

Snapshot ensembles, EMA/SWA, checkpointing, logging, artifact writing, result writing, and export should remain outside the module as callbacks, trainer setup, writers, or post-training commands.

## 10.2 Model composition

The model compositor should assemble the image backbone, optional tabular encoder (described in next section), optional embedding adapter/fusion layer, and heads.

Suggested construction:

```python
backbone = build_backbone(cfg.backbone)

tabular_encoder = build_tabular_encoder(cfg.tabular) if cfg.tabular else None

embedding_adapter = build_embedding_adapter(
    cfg.embedding_adapter,
    image_embedding_dim=backbone.output_dim,
    tabular_embedding_dim=tabular_encoder.output_dim if tabular_encoder else 0,
) if cfg.embedding_adapter else None

heads = build_heads(
    cfg.heads,
    input_dim=embedding_adapter.output_dim if embedding_adapter else backbone.output_dim,
)

model = SupervisedModel(
    backbone=backbone,
    tabular_encoder=tabular_encoder,
    embedding_adapter=embedding_adapter,
    heads=heads,
)
```

The exact effective head input dimension should be derived from the compositor:

```text
image embedding dim
+ optional tabular fusion-embedding dim
→ optional adapter output dim
→ head input dim
```

## 10.3 Tabular metadata as model input

Tabular features may be useful model inputs.

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

Tabular/image fusion should be configured as part of model input/composition.

Example:

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

  tabular_features:
    numeric:
      - equivalent_diameter_um
      - major_axis_um
      - minor_axis_um
      - native_area_px

backbone:
  source: timm
  name: convnext_tiny
  pretrained: true
  output_dim: auto

  freeze:
    policy: last_n_blocks_trainable
    n: 2

tabular:
  enabled: true
  numeric:
    - equivalent_diameter_um
    - major_axis_um
    - minor_axis_um
    - native_area_px

  encoder:
    type: mlp
    hidden_dims: [32]
    output_dim: 32
    activation: relu
    normalization: standard

fusion:
  enabled: true
  type: concat
  inputs:
    - image_embedding
    - tabular_embedding

embedding_adapter:
  enabled: true
  type: mlp
  hidden_dims: [512]
  output_dim: 256
  activation: gelu
  dropout: 0.1

heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    network:
      type: linear

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
```

Model flow:

```text
image → backbone → image_embedding (dim 768)
tabular features → tabular_encoder → tabular_embedding (dim 32)

image_embedding + tabular_embedding 
  ↓
fusion layer (concat) → fused_embedding (dim 800)
  ↓
embedding_adapter (optional) → head_input_embedding (dim 256)
  ↓
head(s)
```

The supervised compositor is responsible for assembling:

```text
backbone
optional tabular encoder and fusion layer
optional embedding adapter
head(s)
```

Exported model artifacts must include tabular feature names, ordering, encodings, and normalization statistics.

Fusion should initially support `concat` only. More complex fusion strategies can be added later.

`embedding_adapter` is optional. It is a projection/MLP applied after the image/tabular fusion step, or directly after the image embedding when no tabular features are used. It can reduce dimensionality, standardize embedding sizes across backbones, add task-specific capacity, or reshape fused embeddings before heads. It should be disabled by default and omitted for true linear-probe evaluation.

The tabular encoder is trained jointly through the supervised objective. Gradients flow from the objective loss through the head, adapter, fusion layer, and into both the tabular encoder and the image backbone if the backbone is trainable. To avoid tabular overfitting or shortcut learning, keep the tabular encoder small, support parameter-group learning rates, regularize the tabular branch, and compare image-only, tabular-only, and image+tabular ablations. Another training option is to use a frozen image-embedding from a previous training to see if addition of tabular data and adapter improves scores.



## 10.4 Training command

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

## 10.5 Supervised output artifacts and results

Training outputs should distinguish between:

```text
checkpoints  = training/resume artifacts
exports      = portable model artifacts
metrics      = aggregate training metrics and plots
results      = sample-level, embedding, and diagnostic outputs for validation epochs (best or final or all, as configured)
```

### Checkpoints

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
    confusion_matrix.json
    metrics.h5

  results/
    # Canonical tall Parquet
    val_results/
      epoch=000/
        part-00000.parquet
      epoch=001/
        part-00000.parquet
      best -> epoch=001

    test_results/
      part-00000.parquet

    infer_results/
      part-00000.parquet

    # Optional convenience exports
    val_results_wide.csv
    test_results_wide.csv
    embeddings.csv
    confusion_matrix.csv
    results.h5
```

### Metrics format

Aggregate metrics should generally be JSON and/or HDF5.

Metrics may include:

```text
train_metrics.json
val_metrics.json
test_metrics.json
confusion_matrix.json
metrics.h5
```

Parquet should be used for result rows and large tabular outputs, but JSON/HDF are sufficient for aggregate metrics.

### Canonical result format

Canonical result format should be tall Parquet.

It should be compatible with [`improv`](https://github.com/WHOIGit/improv)-style schema/provenance concepts.

A single sample may appear multiple times, for example:

```text
one sample_metadata record
one embedding record
one species head output record
one quality head output record
one biomass head output record
one nearest-neighbor output record
```

Do not require static sample metadata to be repeated for every epoch/head output record. Static metadata can be written once and joined later by `sample_id`, `config_id`, and/or `dataset_id`.

### Canonical process/sample headers

Use one common column block per major output family, then add only record-specific columns under each `record_type`.

## Supervised result columns

### Supervised common columns

All supervised sample/output records should share this common provenance/header block.

```text
sample_id
uri
split
record_type
run_id
config_id
dataset_id
epoch
global_step
checkpoint_id
model_id
```

Notes:

- `epoch` and `global_step` may be null for static sample metadata records.
- `checkpoint_id` identifies the checkpoint that produced a result row.
- `model_id` identifies a portable exported model artifact when results come from `exports/model.pt` or another exported artifact.
- `record_type = ...` hints at what other columns will have data for a given record

### Supervised `record_type` values

Recommended supervised `record_type` values:

```text
sample_metadata
embedding
classification_output
regression_output
ordinal_output
```

### `sample_metadata` record-specific columns

These are mostly static per sample/config and need only be included once. 
Not all columns will always be present. For instance: 
- `resize_bucket` for instance ought only be present if training using varying input bucket sizes configuration.
- `tabular_data` is optional if included in training and can be a JSON string.


```text
record_type = sample_metadata
bin_id
bin_uri
roi_number
native_width_px
native_height_px
input_width_px
input_height_px
resize_bucket
microns_per_pixel
tabular_data
source_extra_json
```

`source_extra_json` should be optional and used only for dataset-specific passthrough fields that are not part of the core schema.

### `embedding` record-specific columns

```text
record_type = embedding
embedding_kind = {image_embedding | tabular_embedding | fused_embedding | head_input_embedding}
embedding
embedding_dim
```

Default embedding kind for supervised learning is `head_input_embedding`.

`tabular_embedding` and `fused_embedding` are only relevant when using tabular input data.

`image_embedding` is only different from `head_input_embedding` in supervised learning when an embedding adaptor is used.

Default embedding kind for self-supervised learning is `image_embedding`.

### Supervised-column heads

Classification output:

```text
record_type = classification_output
head_name
target
prediction_index
prediction_label
prediction_confidence
logits
probabilities
```

Regression output:

```text
record_type = regression_output
head_name
target
prediction_value
prediction_uncertainty
```

Ordinal output:

```text
record_type = ordinal_output
head_name
target
prediction_index
prediction_label
prediction_confidence
ordinal_logits
probabilities
```

## Representation-evaluation result columns

Representation-evaluation outputs can be produced from embeddings generated by supervised or self-supervised model families.

Some record types require labeled data; others can be produced from unlabeled data. A few can be used in either context.

### Label availability by record type

| `record_type`             | Requires labels for eval? | Notes                                                                  |
| ------------------------- | ------------------------: | ---------------------------------------------------------------------- |
| `embedding`               |                        No | Can be produced for any image.                                         |
| `nearest_neighbor`        |                        No | Used for unlabeled retrieval or as support records for labeled k-NN.   |
| `knn_prediction`          |                       Yes | Requires neighbor labels to produce predicted class/label.             |
| `classification_probe_prediction` |               Yes | Requires labeled train/eval data. Columns mirror `classification_output`. |
| `regression_probe_prediction`     |               Yes | Requires labeled train/eval data. Columns mirror `regression_output`. |
| `ordinal_probe_prediction`        |               Yes | Requires labeled train/eval data. Columns mirror `ordinal_output`. |
| `cluster_assignment`      |                        No | Labels optional; labels can be used later for ARI/NMI/purity.          |
| `projection`              |                        No | PCA/UMAP/t-SNE coordinates; labels optional for coloring plots.        |
| `outlier_score`           |                        No | Can be computed from embedding density/distance without labels.        |
| `diagnostic`              |                        No | Collapse checks, embedding variance, augmentation consistency, etc.    |

### Representation-evaluation common columns

All canonical self-supervised and representation-evaluation result records should share this common provenance/header block.

```text
sample_id
uri
split
record_type
run_id
config_id
dataset_id
epoch
global_step
checkpoint_id
model_id
evaluation_name
```

Notes:

- `record_type = ...` may still be repeated in record-specific column sections for clarity, even though it is part of the common columns.
- `evaluation_name` identifies the evaluation job, probe, retrieval run, clustering run, projection run, diagnostic pass, or other representation-evaluation process.

### Representation-evaluation `record_type` values

Recommended self-supervised and representation-evaluation `record_type` values:

Representation-evaluation records are model-training-method agnostic. They may be produced from embeddings generated by supervised, self-supervised, semi-supervised, or unsupervised models. This allows direct comparison of embedding spaces across training regimes using the same evaluation methods.

```text
embedding
nearest_neighbor
knn_prediction
linear_probe_prediction
cluster_assignment
projection
outlier_score
diagnostic
```

### `embedding` record-specific columns

Used with labeled or unlabeled datasets.

```text
record_type = embedding
embedding_kind = {image_embedding | tabular_embedding | fused_embedding | head_input_embedding}
embedding
embedding_dim
embedding_model_name
```

`embedding_kind` should identify which representation was saved. For self-supervised learning, this is typically `image_embedding`.

### `nearest_neighbor` record-specific columns

Used with labeled or unlabeled datasets.

For unlabeled evaluation, this is a retrieval/similarity record.

For labeled k-NN evaluation, this can be used as supporting provenance for a `knn_prediction` record.

```text
record_type = nearest_neighbor
query_sample_id
neighbor_sample_id
neighbor_rank
distance
distance_metric
neighbor_label
```

`neighbor_label` is optional and should be null when labels are unavailable.

### `knn_prediction` record-specific columns

Requires labeled data.

This record summarizes the k-NN classification result for a query sample.

```text
record_type = knn_prediction
head_name
target
prediction_index
prediction_label
prediction_confidence
probabilities
k
distance_metric
```

Optional supporting fields:

```text
neighbor_sample_ids
neighbor_labels
neighbor_distances
```

If storing each neighbor separately, prefer separate `nearest_neighbor` records instead of list-valued neighbor columns.

### Probe prediction record-specific columns

Requires labeled data.

Probe predictions are produced by training a small downstream model on frozen embeddings. 
They are used to evaluate representation quality, not to report the native output of the original supervised model.

Use the supervised head column templates in the [Supervised-column heads](#supervised-column-heads) section according to the probe task type:

- For classification probes, use the `classification_output` column template.
- For regression probes, use the `regression_output` column template.
- For ordinal probes, use the `ordinal_output` column template.

Possible probe `record_type` values:

```text
classification_probe_prediction
regression_probe_prediction
ordinal_probe_prediction
```

These probe-specific record_type values make it clear that the row came from a downstream representation-evaluation probe rather than from the model's native supervised head.

Adl. notes:
- head_name should identify the probe, for example linear_probe_species, linear_probe_depth, or ridge_probe_biomass.
- A ridge probe is a linear regression probe trained on frozen embeddings with L2 regularization. It is useful for continuous targets and should generally use `record_type = regression_probe_prediction` with `probe_model_type = ridge`.
- For strictly linear probes, the downstream model should be linear with respect to the saved embedding.
- For non-linear downstream probes, prefer explicit record types or metadata such as `probe_model_type = mlp_probe`, `probe_model_type = random_forest_probe`, or `probe_model_type = kernel_svm_probe`. These need not be implemented at this stage.
- Avoid using `record_type = classification_output`, `record_type = regression_output`, or `record_type = ordinal_output` for probe results unless downstream tools require the native supervised-output schemas. Those record types are best reserved for outputs from the model's own supervised heads.

### `cluster_assignment` record-specific columns

Does not require labels.

Labels may be joined later for cluster purity, ARI, NMI, or review workflows.

```text
record_type = cluster_assignment
cluster_method
cluster_id
cluster_distance
cluster_probability
```

`cluster_probability` is optional and only applies to clustering methods that provide soft assignments or membership confidence.

### `projection` record-specific columns

Does not require labels.

Used for dimensionality reduction outputs such as PCA, UMAP, or t-SNE.

```text
record_type = projection
projection_method
projection_x
projection_y
projection_z
```

`projection_z` is optional and should be null for 2D projections.

Labels can be joined later for visualization coloring.

### `outlier_score` record-specific columns

Does not require labels.

Used for novelty, anomaly, or density-based scoring.

```text
record_type = outlier_score
outlier_method
outlier_score
outlier_rank
is_outlier
```

`is_outlier` should be optional because thresholding may be configured separately from scoring.

### `diagnostic` record-specific columns

Does not require labels.

Used for representation diagnostics such as collapse checks, embedding variance, effective rank, augmentation consistency, or covariance statistics.

Diagnostics may be sample-level or aggregate. If the diagnostic is aggregate, `sample_id` should be null.

```text
record_type = diagnostic
diagnostic_name
diagnostic_value
diagnostic_scope
```

Suggested `diagnostic_scope` values:

```text
sample
batch
epoch
dataset
```

Example `diagnostic_name` values:

```text
embedding_norm_mean
embedding_norm_std
per_dimension_std
effective_rank
covariance_condition
pairwise_cosine_mean
pairwise_cosine_std
augmentation_consistency_cosine
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
species_prediction_confidence
quality_prediction_index
quality_prediction_label
biomass_prediction_value
...
```

Embeddings CSV should be one sample per row.

Confusion matrix CSV should be available for classification heads.

HDF/HDF5 should be treated as an optional export format derived from canonical results.

### Epoch-level validation results

Validation results across epochs should use partitioned Parquet.

Example:

```text
results/val_results/
  epoch=000/
    part-00000.parquet
  epoch=001/
    part-00000.parquet
  epoch=002/
    part-00000.parquet
  best -> epoch=001
```

A symlink or small manifest can point to the best epoch for usability without duplicating data.

Example manifest:

```json
{
  "best_epoch": 1,
  "best_checkpoint_id": "epoch=001-val_macro_f1=0.861",
  "best_path": "results/val_results/epoch=001/"
}
```

This should be YAML configurable.

### Result output config

Canonical results are tall Parquet. 

Example:

```yaml
results:
  canonical:
    enabled: true
    include:
      embeddings: true
      logits: true
      probabilities: true
      predictions: true
      targets: true
      input_shape_context: true

  validation_by_epoch:
    enabled: true
    every_n_epochs: 1
    partition_by:
      - epoch
    best_pointer:
      enabled: true
      mode: symlink_or_manifest

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

The schema should be designed to be compatible with [`improv`](https://github.com/WHOIGit/improv) concepts by construction.

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
    metrics.h5

  results/
    # Canonical tall-format Parquet outputs
    ssl_eval_results/
      epoch=000/
        part-00000.parquet
      epoch=010/
        part-00000.parquet
      best -> epoch=010

    embedding_results/
      part-00000.parquet

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

SSL canonical results should use the same tall Parquet result model as supervised and representation-evaluation outputs.

The SSL section should not define a separate SSL-only result schema. Instead, SSL evaluations should emit the canonical [Representation-evaluation record types](#representation-evaluation-record-types). These record types are model-training-method agnostic and may also be produced from supervised model embeddings, semi-supervised model embeddings, or exported foundation-model embeddings.

This allows direct comparison of embedding spaces across training regimes, for example:

```text
supervised encoder embeddings      -> k-NN / probes / clustering / projection / diagnostics
self-supervised encoder embeddings -> k-NN / probes / clustering / projection / diagnostics

SSL-specific `record_type`s may include:

```text
embedding
nearest_neighbor
knn_prediction
classification_probe_prediction
regression_probe_prediction
ordinal_probe_prediction
cluster_assignment
projection
outlier_score
diagnostic
```

These are not SSL-specific record types persay. They describe the evaluation result that was produced from an embedding space.

Again see [Representation-evaluation record types](#representation-evaluation-record-types) section for common and record_type specific columns.

Partitioned Parquet should be supported for epoch-level SSL evaluation outputs too.

---

# 12. SSL evaluation

## 12.1 Evaluation categories

SSL evaluation should support three categories:

```text
label-required representation evaluation
label-free representation evaluation
visual/diagnostic representation evaluation
```

Although this section focuses on SSL-trained encoders, the same representation-evaluation methods should also be usable for supervised encoders. This allows embeddings from supervised and self-supervised models to be compared with the same evaluation protocols.

## 12.2 Label-required SSL evaluation

When a labeled evaluation dataset is available, support:

```text
k-NN classification
linear probe
supervised fine-tuning evaluation
macro/per-class metrics
confusion matrix
```

Probe evaluations train small downstream models on frozen embeddings.

Fine-tuning evaluations update some or all model weights and should be treated as supervised training/evaluation, not as probe evaluation.

```text
k-NN classification              -> record_type = knn_prediction
classification probe             -> record_type = classification_probe_prediction
regression probe                 -> record_type = regression_probe_prediction
ordinal probe                    -> record_type = ordinal_probe_prediction
supervised fine-tuning classifier -> record_type = classification_output
supervised fine-tuning regressor  -> record_type = regression_output
supervised fine-tuning ordinal    -> record_type = ordinal_output
```

Config example:

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

    probes:
      enabled: true
      tasks:
        species:
          type: classification
          probe_model_type: linear_classifier
          loss: cross_entropy
    
        biomass:
          type: regression
          probe_model_type: ridge
          loss: mean_squared_error
    
        life_stage:
          type: ordinal
          probe_model_type: ordinal_logistic_regression
          schedule:

        every_n_epochs: 10
```

For classification probes, `probe_model_type = linear_classifier` refers to a single linear layer or equivalent multinomial logistic regression model trained on frozen embeddings. It produces logits and probabilities using the same column schema as `classification_output`.

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

```
record_type = diagnostic

# fields
diagnostic_name
diagnostic_value
diagnostic_scope
```

diagnostic_names for assessing during training.

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

Reccommended record form:
```
record_type = diagnostic
diagnostic_name = augmentation_consistency_cosine
diagnostic_value
diagnostic_scope = sample
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
record_type = cluster_assignment
sample_id
cluster_method
cluster_id
cluster_distance
cluster_probability
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
record_type = projection
sample_id
projection_method
projection_x
projection_y
projection_z
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
      backend: ifcb_bins
      manifest_uri: s3://bucket/ifcb_bins/unlabeled_bins.parquet
      bin_id_column: bin_id
      bin_uri_column: bin_uri

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

# 13. Generic model ensembling architecture

## 13.1 Concept

Model ensembling is a model-selection, artifact-construction, and inference workflow over compatible candidate models.

It should be implemented as:

```text
candidate discovery
candidate compatibility validation
candidate scoring
ensemble selection
ensemble construction
ensemble evaluation
artifact bundling
canonical result export
```

not as a separate training model architecture.

An ensemble may combine:

```text
checkpoints from one training run
snapshots from one training run
best-k checkpoints from one run
checkpoints from different runs
exported model artifacts
fine-tuned models
probe models
candidate model soups
SWA exports
EMA exports
```

The core requirement is that ensemble members produce compatible outputs for the same task schema.

## 13.2 Ensemble types

The framework should support multiple ensemble families.

```text
prediction_space_ensemble
snapshot_ensemble
checkpoint_ensemble
cross_run_ensemble
weighted_ensemble
model_soup
swa_model
ema_model
```

### Prediction-space ensemble

A prediction-space ensemble keeps multiple member models and combines their respective output heads at inference time.

Examples:

```text
average classification probabilities
average classification logits
average regression predictions
average ordinal logits or ordinal probabilities
majority vote
weighted average by validation score
```

Prediction-space ensembles can combine models with different architectures if their inputs and outputs are compatible.

### Snapshot ensemble

A snapshot ensemble is a prediction-space ensemble whose members are checkpoints from the same training run.

Snapshot ensembles are useful when training produces multiple good checkpoints at different epochs or schedule cycles.

### Cross-run ensemble

A cross-run ensemble combines compatible models from different runs.

This is useful for combining models trained with different seeds, augmentations, folds, architectures, or training schedules.

### Weight-space ensemble

Weight-space ensemble methods combine model parameters and produce a single exported model.

Examples:

```text
model soup
greedy soup
SWA
EMA
```

Weight-space methods require stricter compatibility than prediction-space ensembles.

They require compatible:

```text
architecture
parameter names
tensor shapes
head definitions
target schema
class mappings
preprocessing
```

At inference time, model soups, SWA, and EMA behave like single models, not multi-member prediction ensembles.

## 13.3 Generic ensemble command

```bash
dojo ensemble experiment=ifcb/ensemble_search
```

Purpose:

```text
given many compatible candidate checkpoints or exported models
select an ensemble that optimizes the configured validation objective
under configured compute, latency, memory, and artifact-size constraints
```

The command should be able to run against:

```text
local run directories
remote run directories
explicit checkpoint manifests
explicit model manifests
experiment search results
model registry entries
```

Example:

```bash
dojo ensemble \
  experiment=ifcb/ensemble_search \
  ensemble.discovery.run_uri=s3://bucket/runs/run123 \
  ensemble.selection.strategy=greedy_forward_selection
```

## 13.4 Candidate compatibility requirements

Prediction-space ensembles require compatible output semantics, not necessarily identical input preprocessing.

Each ensemble member may define and apply its own preprocessing pipeline, image size, resize policy, normalization, and input transforms, as long as the ensemble runner can provide the required raw sample fields and each member produces compatible outputs.

Hard compatibility requirements:

```text
sample identity semantics
required input fields are available
target schema
head names
head task types
class mappings
ordinal encoding rules
ordinal decoding rules
regression target units
regression target scaling
output tensor shapes
output tensor meanings
```

For classification, all members included for a given head must use the same class order.

For regression, all members included for a given head must predict the same target in the same units and scaling.

For ordinal heads, all members included for a given head must use the same ordinal encoding and decoding rules.

For multi-head models, compatibility should be checked per head.

A candidate may be excluded from one head and included for another if the framework supports partial-head ensembling.

### Member-specific preprocessing

Candidate models may have different preprocessing requirements.

Examples:

```text
different image sizes
different resize policies
different normalization constants
different crop strategies
different tabular feature transforms
different required input modalities
```

These differences are allowed if each member artifact stores its own preprocessing metadata and the ensemble runner can apply preprocessing separately per member.

This enables heterogeneous ensembles such as:

```text
EfficientNet model with 384 px inputs
ViT model with 224 px inputs
ConvNeXt model with 320 px inputs
image-only model
image + tabular model
```

The ensemble artifact should preserve member-specific preprocessing metadata.

Recommended member metadata:

```text
ensemble_member_id
model_id
checkpoint_id
preprocessing_config
preprocessing_hash
required_input_fields
input_modalities
target_schema_hash
class_mapping_hash
model_config_hash
```

### Shared-preprocessing fast path

The framework may optionally support a shared-preprocessing fast path.

This path preprocesses each sample once and sends the same tensor batch to all ensemble members.

Shared preprocessing requires compatible:

```text
input modality
image size
resize policy
normalization
tensor layout
tabular feature transforms
```

This is an optimization, not a general ensemble requirement.

If shared preprocessing is not possible, the ensemble runner should fall back to member-specific preprocessing.

### Weight-space compatibility

Weight-space methods have stricter requirements than prediction-space ensembles.

Methods such as model soups, SWA, and EMA require compatible:

```text
architecture
parameter names
tensor shapes
head definitions
target schema
class mappings
preprocessing assumptions
```

Unlike prediction-space ensembles, weight-space methods produce a single model artifact, so the resulting averaged model must have one coherent preprocessing contract.

## 13.5 Candidate discovery

Candidate discovery should produce a normalized candidate manifest.

Inputs may include:

```text
checkpoints from one run
snapshots from one run
best-k checkpoints
checkpoints from different runs
exported .pt models
exported ONNX models
candidate soups
EMA exports
SWA exports
```

Example discovery config:

```yaml
ensemble:
  discovery:
    sources:
      - type: run_checkpoints
        run_uri: s3://bucket/runs/run123
        checkpoint_glob: checkpoints/*.ckpt

      - type: exported_models
        uri_glob: s3://bucket/runs/*/exports/model.pt

      - type: manifest
        manifest_uri: s3://bucket/ensembles/candidates.parquet
```

Candidate manifest columns:

```text
candidate_id
candidate_uri
candidate_type
source_run_id
source_config_id
checkpoint_id
model_id
epoch
global_step
metric_name
metric_value
split
created_at
preprocessing_hash
target_schema_hash
class_mapping_hash
model_config_hash
```

Candidate types:

```text
checkpoint
exported_model
snapshot_checkpoint
soup_model
swa_model
ema_model
external_model
```

## 13.6 Selection strategies

Supported selection strategies:

```text
top_k
best_single
greedy_forward_selection
greedy_backward_elimination
weighted_greedy_forward_selection
diversity_aware_selection
snapshot_cycle_selection
budget_constrained_selection
greedy_soup
uniform_soup
```

### top_k

Select the top K candidates by a validation metric.

Useful as a simple baseline.

```yaml
ensemble:
  selection:
    strategy: top_k
    k: 5
    metric: val/species/macro_f1
    mode: max
```

### greedy_forward_selection

Start with the best single model, then iteratively add the candidate that most improves ensemble validation performance.

```yaml
ensemble:
  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_models: 8
    stop_if_no_improvement: true
```

### diversity_aware_selection

Select candidates using both validation quality and diversity.

Diversity signals may include:

```text
prediction disagreement
error disagreement
low correlation between logits
low correlation between probabilities
different random seeds
different architectures
different augmentations
different training folds
different checkpoint epochs
```

Example:

```yaml
ensemble:
  selection:
    strategy: diversity_aware_selection
    metric: val/species/macro_f1
    mode: max
    diversity_metric: prediction_disagreement
    diversity_weight: 0.20
    max_models: 8
```

### budget_constrained_selection

Select the best ensemble under deployment constraints.

Budget constraints:

```text
max_models
max_latency_ms
max_file_size_mb
max_memory_mb
max_parameters
max_flops
```

Example:

```yaml
ensemble:
  selection:
    strategy: budget_constrained_selection
    metric: val/species/macro_f1
    mode: max
    constraints:
      max_models: 4
      max_latency_ms: 100
      max_file_size_mb: 2048
      max_memory_mb: 4096
```

## 13.X Offline ensemble assessment from canonical results

Prediction-space ensembles can be assessed from previously exported canonical result files, without loading model checkpoints, when all candidate results were produced on the same evaluation dataset and compatible output heads.

This enables fast ensemble search over cached predictions.

Required alignment fields:

"""text
sample_id
split
head_name
target
record_type
"""

Recommended provenance fields:

"""text
run_id
config_id
checkpoint_id
model_id
epoch
global_step
"""

Required compatibility:

"""text
same evaluation dataset or exactly aligned sample_id set
same target semantics
same head task type
same class mapping for classification
same ordinal encoding and decoding rules for ordinal outputs
same regression units and target scaling for regression outputs
"""

Candidate preprocessing, image size, model architecture, and training method do not need to match because the ensemble is combining canonical outputs, not raw model inputs.

### Cached classification ensembling

Classification ensemble assessment should use cached `classification_output` records.

Recommended columns:

"""text
sample_id
split
record_type = classification_output
head_name
target
prediction_index
prediction_label
prediction_confidence
logits
probabilities
run_id
checkpoint_id
model_id
"""

Supported cached-output combine modes:

"""text
probabilities_mean
weighted_probabilities_mean
logits_mean
weighted_logits_mean
majority_vote
weighted_vote
"""

If `probabilities` are available, prefer `probabilities_mean` for heterogeneous models.

If only `prediction_label` is available, only vote-based ensemble assessment is possible.

### Cached regression ensembling

Regression ensemble assessment should use cached `regression_output` records.

Recommended columns:

"""text
sample_id
split
record_type = regression_output
head_name
target
prediction_value
prediction_uncertainty
run_id
checkpoint_id
model_id
"""

Supported cached-output combine modes:

"""text
prediction_mean
prediction_median
weighted_prediction_mean
prediction_trimmed_mean
"""

For basic regression ensembles:

"""text
ensemble_prediction_value = mean(member_prediction_values)
ensemble_prediction_uncertainty = std(member_prediction_values)
"""

### Cached ordinal ensembling

Ordinal ensemble assessment should use cached `ordinal_output` records.

Recommended columns:

"""text
sample_id
split
record_type = ordinal_output
head_name
target
prediction_index
prediction_label
prediction_confidence
ordinal_logits
probabilities
run_id
checkpoint_id
model_id
"""

Supported cached-output combine modes:

"""text
ordinal_logits_mean
ordinal_probabilities_mean
expected_rank_mean
"""

All candidate outputs must use the same ordinal encoding and decoding rules.

### Offline ensemble selection

Cached canonical results can be used for:

"""text
best single model selection
top-k ensemble assessment
greedy forward ensemble selection
weighted ensemble search
diversity/disagreement analysis
calibration analysis
budget-aware selection, if candidate metadata includes latency or size
"""

Example command:

"""bash
dojo ensemble from-results experiment=ifcb/offline_ensemble_search
"""

Example config:

"""yaml
ensemble:
  type: cached_prediction_ensemble

  discovery:
    sources:
      - type: canonical_results
        uri: s3://bucket/runs/run123/results/val_results/
      - type: canonical_results
        uri: s3://bucket/runs/run456/results/val_results/
      - type: canonical_results
        uri: s3://bucket/runs/run789/results/val_results/

  compatibility:
    require_same_sample_ids: true
    require_same_target_schema: true
    require_same_class_mappings: true

  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_models: 5

  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_mean
      ordinal: ordinal_probabilities_mean
"""

### Limitations

Cached canonical results are sufficient to assess ensemble performance on an existing dataset, but they are not sufficient to deploy the ensemble on new data.

Deployment still requires access to the selected member model artifacts or a bundled ensemble artifact.

Cached results also cannot measure true ensemble latency or memory use unless candidate metadata includes those measurements.

The validation set may be used for ensemble selection. A separate test set should be used for final unbiased reporting.

## 13.7 Ensemble inference combine modes

Combine modes should be task-specific.

Example config:

```yaml
ensemble:
  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_mean
      ordinal: ordinal_probabilities_mean
```

Supported classification combine modes:

```text
logits_mean
probabilities_mean
weighted_logits_mean
weighted_probabilities_mean
majority_vote
weighted_vote
```

Supported regression combine modes:

```text
prediction_mean
prediction_median
weighted_prediction_mean
prediction_trimmed_mean
```

Supported ordinal combine modes:

```text
ordinal_logits_mean
ordinal_probabilities_mean
expected_rank_mean
weighted_ordinal_logits_mean
weighted_ordinal_probabilities_mean
```

Recommended defaults:

```text
snapshot ensemble from one run: logits_mean or probabilities_mean
cross-run homogeneous ensemble: probabilities_mean
cross-run heterogeneous ensemble: probabilities_mean
regression ensemble: prediction_mean
ordinal ensemble: ordinal_probabilities_mean
```

### Classification inference

With `logits_mean`:

```text
member_logits = [logits_1, logits_2, ...]
ensemble_logits = mean(member_logits)
probabilities = softmax(ensemble_logits)
prediction_index = argmax(probabilities)
prediction_confidence = max(probabilities)
```

With `probabilities_mean`:

```text
member_probabilities = [softmax(logits_1), softmax(logits_2), ...]
probabilities = mean(member_probabilities)
prediction_index = argmax(probabilities)
prediction_confidence = max(probabilities)
```

`probabilities_mean` is usually safer when combining heterogeneous models because it reduces sensitivity to different logit scales.

### Regression inference

```text
member_predictions = [prediction_1, prediction_2, ...]
prediction_value = mean(member_predictions)
prediction_uncertainty = std(member_predictions)
```

If members also produce aleatoric uncertainty, total uncertainty may combine:

```text
within_member_uncertainty
between_member_disagreement
```

### Ordinal inference

Ordinal ensembling must respect the ordinal head implementation.

Common approaches:

```text
average ordinal logits, then decode
average ordinal probabilities, then decode
average expected rank values
```

The ensemble artifact should store the ordinal decoding rule used for each ordinal head.

## 13.8 Ensemble config

Generic config:

```yaml
ensemble:
  enabled: true
  type: prediction_space_ensemble

  discovery:
    sources:
      - type: run_checkpoints
        run_uri: s3://bucket/runs/run123
        checkpoint_glob: checkpoints/*.ckpt

  compatibility:
    require_same_preprocessing: false
    require_same_target_schema: true
    require_same_class_mappings: true
    allow_partial_head_compatibility: false

  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    split: val
    max_models: 5
    stop_if_no_improvement: true

  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_mean
      ordinal: ordinal_probabilities_mean

  artifact:
    bundle_single_pt: true
    bundle_filename: ensemble.pt
    include_member_state_dicts: true
    include_member_uris: true
    include_member_hashes: true

  output:
    save_member_predictions: false
    save_ensemble_predictions: true
    save_disagreement_metrics: true
```

## 13.9 Final-output snapshot ensemble

A final-output snapshot ensemble is a prediction-space ensemble over checkpoints from the same supervised or fine-tuned training run.

Each snapshot contains the full model needed to produce final task outputs:

```text
backbone / encoder
neck or adapter, if used
task heads
class mappings
target transforms
preprocessing metadata
```

The ensemble combines final task outputs, not embeddings.

Typical use cases:

```text
classification snapshot ensemble
multi-head supervised snapshot ensemble
fine-tuned SSL model snapshot ensemble
```

## 13.10 Training setup for final-output snapshot ensembles

A final-output snapshot ensemble requires a training run that intentionally produces multiple useful checkpoints from one optimization trajectory.

The recommended approach is snapshot-cycle training:

```text
train one model
use a cyclic or warm-restart learning-rate schedule
save a snapshot near the end of each cycle
combine saved snapshots later as a prediction-space ensemble
```

This follows the classic snapshot-ensemble pattern:

```text
high learning rate at cycle restart
  -> model moves to a new region of weight space

cosine decay within the cycle
  -> model settles into a useful local solution

cycle end
  -> save checkpoint snapshot

repeat
  -> produce multiple diverse-but-compatible snapshots
```

Snapshot-cycle training is most useful for final-output ensembles where each saved snapshot contains the full model needed for prediction:

```text
backbone / encoder
neck or adapter, if used
task heads
class mappings
target transforms
preprocessing metadata
```

The ensemble combines final task outputs, not embeddings.

Typical use cases:

```text
classification snapshot ensemble
regression snapshot ensemble
ordinal regression snapshot ensemble
multi-head supervised snapshot ensemble
fine-tuned SSL model snapshot ensemble
```

### Snapshot-cycle scheduler

The training config should support cosine cycles or cosine warm restarts.

Example:

```yaml
training:
  max_epochs: 300

  scheduler:
    type: cosine_warm_restarts

    # Number of epochs in the first cycle.
    first_cycle_epochs: 50

    # Keep all cycles the same length.
    cycle_mult: 1.0

    # Learning-rate range within each cycle.
    max_lr: 1.0e-4
    min_lr: 1.0e-6

    # Optional warmup before the first cycle.
    warmup_epochs: 5
```

This produces cycle-end snapshot candidates such as:

```text
epoch 050
epoch 100
epoch 150
epoch 200
epoch 250
epoch 300
```

The exact cycle schedule may be expressed in epochs or iterations, but the checkpointing logic should know when a cycle ends.

### Cycle-end snapshot checkpointing

Snapshot checkpoints should be saved at or near the end of each scheduler cycle.

Example:

```yaml
training:
  checkpointing:
    enabled: true

    save_last: true

    save_cycle_snapshots:
      enabled: true
      at_cycle_end: true
      filename_template: snapshot_cycle={cycle:02d}_epoch={epoch:03d}_step={global_step}

    save_top_k:
      enabled: true
      k: 5
      monitor: val/species/macro_f1
      mode: max
      filename_template: best_epoch={epoch:03d}_macro_f1={metric:.4f}
```

Cycle-end snapshots and best-k checkpoints can both be included as ensemble candidates.

Recommended candidate sources:

```text
cycle-end snapshots
best-k validation checkpoints
last checkpoint
late-training checkpoints
EMA checkpoint, if enabled
SWA checkpoint, if enabled
```

### Validation during snapshot-cycle training

Validation should run often enough to score snapshot candidates.

Example:

```yaml
training:
  validation:
    run_every_n_epochs: 1

  metrics:
    primary_metric: val/species/macro_f1
    mode: max
```

Each saved snapshot should record the metrics available at that point in training.

Recommended snapshot metadata:

```text
run_id
config_id
checkpoint_id
epoch
global_step
cycle_index
cycle_start_epoch
cycle_end_epoch
scheduler_type
learning_rate_at_save
selection_metric_name
selection_metric_value
model_config_hash
target_schema_hash
class_mapping_hash
preprocessing_hash
```

### Alternative snapshot sources

Snapshot-cycle training is the preferred setup when snapshot ensembling is planned from the beginning.

The framework should also support simpler snapshot sources:

```text
regular epoch checkpoints
late-training checkpoints
best-k validation checkpoints
manual checkpoint manifests
```

Example late-training snapshot setup:

```yaml
training:
  max_epochs: 300

  checkpointing:
    enabled: true
    save_last: true

    save_epoch_snapshots:
      enabled: true
      every_n_epochs: 5
      start_epoch: 200
      filename_template: late_snapshot_epoch={epoch:03d}
```

This does not intentionally create cycle diversity, but it may still produce useful ensemble candidates.

### Relationship to model soups, SWA, and EMA

Snapshot-cycle training produces candidates for a prediction-space ensemble.

It should not be confused with weight-averaging methods.

```text
snapshot ensemble:
  save multiple checkpoints
  keep multiple member models
  combine outputs at inference

model soup:
  average compatible checkpoint weights
  export one model

SWA:
  average weights during or after training
  export one model

EMA:
  maintain shadow weights during training
  export one model
```

The same training run may produce snapshot candidates, EMA weights, and SWA weights, but these should be represented as different candidate types.

## 13.11 Snapshot ensemble command

Snapshot ensembling should be exposed as a specialized mode of the generic ensemble workflow.

Specialized command:

```bash
dojo ensemble snapshot experiment=ifcb/snapshot_ensemble
```

Equivalent generic command:

```bash
dojo ensemble \
  experiment=ifcb/snapshot_ensemble \
  ensemble.type=snapshot_ensemble \
  ensemble.discovery.sources.0.type=run_checkpoints
```

The snapshot command should:

```text
discover snapshot checkpoints from one run
validate candidate compatibility
score candidates on a validation split
select a subset of snapshots
construct a prediction-space ensemble
evaluate the selected ensemble
export an ensemble artifact
write canonical result records
```

### Snapshot-cycle ensemble config

Example config for snapshots produced by cosine warm-restart training:

```yaml
ensemble:
  enabled: true
  type: snapshot_ensemble

  discovery:
    sources:
      - type: run_checkpoints
        run_uri: s3://bucket/runs/run123
        checkpoint_glob: checkpoints/snapshot_cycle=*.ckpt
        candidate_type: snapshot_checkpoint

  compatibility:
    require_same_target_schema: true
    require_same_class_mappings: true
    allow_member_specific_preprocessing: true
    allow_partial_head_compatibility: false

  selection:
    strategy: cycle_end_snapshots

    # Optional: use all discovered cycle-end snapshots.
    use_all_cycles: true

    # Optional: cap number of selected snapshots.
    max_models: 6

    metric: val/species/macro_f1
    mode: max
    split: val

  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_mean
      ordinal: ordinal_probabilities_mean

  artifact:
    bundle_single_pt: true
    bundle_filename: snapshot_ensemble.pt
    include_member_state_dicts: true
    include_member_uris: true
    include_member_hashes: true

  output:
    save_ensemble_predictions: true
    save_member_predictions: false
    save_disagreement_metrics: true
```

### Snapshot selection strategies

The simplest snapshot ensemble uses all cycle-end snapshots.

```yaml
ensemble:
  selection:
    strategy: cycle_end_snapshots
    use_all_cycles: true
```

A validation-filtered snapshot ensemble selects the best K cycle-end snapshots.

```yaml
ensemble:
  selection:
    strategy: top_k
    candidate_filter:
      candidate_type: snapshot_checkpoint
    k: 5
    metric: val/species/macro_f1
    mode: max
```

A greedy snapshot ensemble selects snapshots that improve ensemble validation performance.

```yaml
ensemble:
  selection:
    strategy: greedy_forward_selection
    candidate_filter:
      candidate_type: snapshot_checkpoint
    metric: val/species/macro_f1
    mode: max
    max_models: 5
    stop_if_no_improvement: true
```

The framework should allow both classic uniform snapshot ensembling and validation-based snapshot selection.

### Cached-result snapshot ensemble search

If canonical validation results have already been written for each snapshot, the snapshot ensemble can be assessed without re-running model inference.

Example:

```bash
dojo ensemble snapshot from-results experiment=ifcb/snapshot_ensemble_from_results
```

Example config:

```yaml
ensemble:
  enabled: true
  type: cached_snapshot_ensemble

  discovery:
    sources:
      - type: canonical_results
        uri: s3://bucket/runs/run123/results/snapshot_val_results/
        candidate_type: snapshot_checkpoint

  compatibility:
    require_same_sample_ids: true
    require_same_target_schema: true
    require_same_class_mappings: true

  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_models: 5

  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_mean
      ordinal: ordinal_probabilities_mean
```

Cached-result search is useful for fast ensemble selection, but deployment still requires access to the selected member checkpoints or a bundled ensemble artifact.

### Snapshot ensemble outputs

Recommended output layout:

```text
runs/{run_id}/
  ensemble/
    candidate_manifest.json
    compatibility_report.json
    selected_candidates.json
    ensemble_manifest.json
    ensemble_metrics.json
    member_metrics.json
    disagreement_metrics.json

  exports/
    snapshot_ensemble.pt

  results/
    val_results/
      part-00000.parquet

    test_results/
      part-00000.parquet

    # Optional member-level outputs
    val_member_results/
      member=snapshot_cycle_01/
        part-00000.parquet
      member=snapshot_cycle_02/
        part-00000.parquet
```

Ensemble prediction rows should use the normal canonical supervised result types:

```text
classification ensemble -> record_type = classification_output
regression ensemble     -> record_type = regression_output
ordinal ensemble        -> record_type = ordinal_output
```

The ensemble artifact should be identified with provenance fields such as:

```text
model_id
ensemble_id
ensemble_method
ensemble_member_count
ensemble_combine_method
```

For snapshot ensembles, `ensemble_method` should identify the method used to generate or select members, for example:

```text
snapshot_cycle_uniform
snapshot_cycle_top_k
snapshot_cycle_greedy_forward
late_checkpoint_top_k
```

## 13.12 Ensemble artifact format

The exported ensemble artifact should contain enough information for reproducible inference.

Example bundled artifact:

```python
{
    "artifact_type": "prediction_space_ensemble",
    "ensemble_type": "snapshot_ensemble",
    "format_version": "1.0",

    "ensemble_id": "...",
    "model_id": "...",

    "model_config": {...},
    "preprocessing": {...},
    "target_schema": {...},
    "heads": {...},
    "class_mappings": {...},

    "members": [
        {
            "ensemble_member_id": "snapshot_001",
            "candidate_id": "...",
            "name": "snapshot_001",
            "candidate_type": "snapshot_checkpoint",
            "source_run_id": "run123",
            "source_config_id": "...",
            "checkpoint_id": "...",
            "checkpoint_uri": "s3://bucket/runs/run123/checkpoints/snapshot_001.ckpt",
            "checkpoint_sha256": "...",
            "epoch": 20,
            "global_step": 12345,
            "selection_metric_name": "val/species/macro_f1",
            "selection_metric_value": 0.842,
            "weight": 1.0,
            "state_dict": {...},
        },
        {
            "ensemble_member_id": "snapshot_002",
            "candidate_id": "...",
            "name": "snapshot_002",
            "candidate_type": "snapshot_checkpoint",
            "source_run_id": "run123",
            "source_config_id": "...",
            "checkpoint_id": "...",
            "checkpoint_uri": "s3://bucket/runs/run123/checkpoints/snapshot_002.ckpt",
            "checkpoint_sha256": "...",
            "epoch": 40,
            "global_step": 24690,
            "selection_metric_name": "val/species/macro_f1",
            "selection_metric_value": 0.858,
            "weight": 1.0,
            "state_dict": {...},
        },
    ],

    "combine": {
        "classification": "logits_mean",
        "regression": "prediction_mean",
        "ordinal": "ordinal_logits_mean",
    },

    "selection_config": {...},
    "selection_results": {...},
    "compatibility_report": {...},

    "created_at": "...",
    "dojo_version": "...",
}
```

The artifact may either bundle member weights directly or store immutable references to member artifacts.

If member weights are not bundled, the artifact must include:

```text
member checkpoint/model URI
content hash
model config hash
preprocessing hash
target schema hash
class mapping hash
```

## 13.13 Ensemble run layout

Recommended output layout:

```text
runs/{run_id}/
  config/
    resolved.yaml
    resolved.json

  ensemble/
    candidate_manifest.json
    compatibility_report.json
    selected_candidates.json
    ensemble_manifest.json
    ensemble_metrics.json
    member_metrics.json
    disagreement_metrics.json

  exports/
    ensemble.pt
    snapshot_ensemble.pt

  results/
    val_results/
      part-00000.parquet

    test_results/
      part-00000.parquet

    # Optional debugging/provenance outputs
    val_member_results/
      member=snapshot_001/
        part-00000.parquet
      member=snapshot_002/
        part-00000.parquet

    test_member_results/
      member=snapshot_001/
        part-00000.parquet
      member=snapshot_002/
        part-00000.parquet
```

## 13.14 Ensemble result records

Ensemble prediction outputs should use the same canonical result schemas as ordinary supervised outputs.

Recommended record types:

```text
classification ensemble -> record_type = classification_output
regression ensemble     -> record_type = regression_output
ordinal ensemble        -> record_type = ordinal_output
```

The row represents the ensemble prediction, not an individual member prediction.

Recommended additional provenance columns:

```text
model_id
ensemble_id
ensemble_method
ensemble_member_count
ensemble_combine_method
```

For ensemble outputs, `model_id` should identify the exported ensemble artifact, such as `ensemble.pt` or `snapshot_ensemble.pt`.

`checkpoint_id` may be null for bundled ensembles because no single checkpoint produced the prediction.

Member checkpoint provenance should be stored in the ensemble manifest rather than repeated in every prediction row.

Optional member-level prediction rows may be saved for debugging, calibration analysis, and disagreement analysis.

Recommended member-level provenance columns:

```text
ensemble_id
ensemble_member_id
member_checkpoint_id
member_model_id
member_weight
```

Member-level outputs should also use the canonical supervised result schemas.

## 13.15 Ensemble metrics

The ensemble workflow should report:

```text
ensemble validation metrics
ensemble test metrics, when available
single best candidate metrics
selected member metrics
member disagreement metrics
calibration metrics
latency and memory metrics
artifact size
```

For classification:

```text
accuracy
balanced_accuracy
macro_f1
micro_f1
weighted_f1
per_class_f1
per_class_recall
per_class_precision
macro_auroc
macro_average_precision
negative_log_likelihood
expected_calibration_error
confusion_matrix
```

For regression:

```text
mae
mse
rmse
r2
spearman_correlation
pearson_correlation
prediction_uncertainty_summary
```

For ordinal regression:

```text
accuracy
balanced_accuracy
macro_f1
mean_absolute_rank_error
quadratic_weighted_kappa
ordinal_calibration_error
```

For long-tail or imbalanced datasets, selection should not default to plain accuracy.

Preferred selection metrics include:

```text
macro_f1
balanced_accuracy
tail_class_recall
per_class_f1
macro_average_precision
macro_auroc
negative_log_likelihood
calibration_error
```

Selection should support a primary metric and optional tie-breakers.

Example:

```yaml
ensemble:
  selection:
    metric: val/species/macro_f1
    mode: max
    tie_breakers:
      - metric: val/species/tail_recall
        mode: max
      - metric: val/species/expected_calibration_error
        mode: min
```

## 13.16 Relationship to model soups, SWA, and EMA

Model soups, SWA, and EMA are related to ensembling but should be represented differently from prediction-space ensembles.

```text
prediction-space ensemble:
  keeps multiple member models
  combines outputs at inference
  exports an ensemble artifact

model soup:
  averages compatible model weights
  exports one ordinary model artifact

SWA:
  averages weights during or after training
  exports one ordinary model artifact

EMA:
  maintains shadow weights during training
  exports one ordinary model artifact
```

Prediction-space ensembles should use:

```text
artifact_type = prediction_space_ensemble
```

Model soups, SWA, and EMA should usually export ordinary model artifacts with provenance indicating how the weights were produced.

Example:

```text
artifact_type = model
weight_source = greedy_soup
```

```text
artifact_type = model
weight_source = swa
```

```text
artifact_type = model
weight_source = ema
```

# 14. Model soups, SWA, and EMA 

The architecture should leave room for model averaging methods. Potential modules:

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
probabilities
prediction_confidence
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
supervised .pt model exports that expose their embeddings
SSL encoder exports
SSL training checkpoints
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

These should play nice with Lightning's logging system.

See extant src/dojo/multiclass/callbacks.py for AimLogger metrics examples

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

Use Hydra multirun.
Baysian hyperparameter search may be added in the future.

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
IFCB bin manifest
IFCB bin expansion into ROI samples
multi-head targets
tabular feature extraction
sample_id propagation
uri propagation
bin_id/bin_uri propagation
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
strict true/false checkpoint loading behavior
embedding adapter
tabular encoder
fusion/compositor output dimensions
freeze policies
dojo inspect output
classification head
regression head
ordinal head
multi-head forward
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
metrics built from objective definitions
```

## 19.7 Supervised training smoke tests

Test:

```text
single-head supervised 1 epoch
multi-head supervised 1 epoch
tabular + image supervised 1 epoch
canonical tall Parquet results
partitioned validation results by epoch
best pointer manifest/symlink
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
IFCB bins unlabeled SSL path
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
4. Add backbone registry.
5. Add head registry.
6. Add objectives.
7. Add supervised model compositor.
8. Add supervised LightningModule.
9. Add canonical results writer.

## Phase 3: IFCB bins

1. Add `ifcbkit` dependency. Remove `pyifcb` dependancy. 
2. Implement IFCB bin dataset.
3. Implement IFCB bin datamodule.
4. Add bin manifest support with `bin_id_column` and `bin_uri_column`.
5. Add runtime ROI expansion.
6. Add tests for variable ROI counts per bin.

## Phase 4: Transform refactor

1. Add transform builder.
2. Add letterbox.
3. Add aspect buckets.
4. Add size buckets.
5. Add foreground-aware crop.
6. Add grayscale/normalization transforms.

## Phase 5: SSL with Lightly

1. Add `dojo[ssl]`.
2. Add Lightly DINOv2-style task.
3. Add SSL transforms.
4. Add labeled SSL eval.
5. Add unlabeled diagnostics/retrieval/clustering/projections.
6. Add encoder export.

## Phase 6: Ensemble workflows

1. Add `dojo ensemble`.
2. Add `dojo ensemble snapshot`.
3. Add checkpoint selection.
4. Add snapshot bundle artifact.
5. Add ensemble result writing.

## Phase 7: Export

1. Add `.pt` single-model export.
2. Add `.pt` snapshot ensemble export.
3. Add ONNX single-model export.
4. Add ONNX metadata.
5. Add bucket-aware ONNX export support.

## Phase 8: Logging sinks

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

Use [`amplify-storage-utils`](https://github.com/WHOIGit/amplify-storage-utils) underneath, but do not leak its API throughout datasets/training/export code.

## 21.7 Use `ifcbkit` for IFCB raw-bin handling

Use [`ifcbkit`](https://github.com/WHOIGit/ifcbkit) for IFCB bin discovery, parsing, ROI image extraction, and identifier handling.

Dojo should not reimplement IFCB raw-bin parsing.

## 21.8 Make embeddings first-class

Embeddings should be extractable and persistable from:

```text
supervised models
SSL encoders
transfer-learning checkpoints
```

## 21.9 Optimize for external orchestration

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
