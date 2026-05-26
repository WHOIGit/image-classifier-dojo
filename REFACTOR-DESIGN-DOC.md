# Image Classifier Dojo Refactor Design Doc

## Status

Draft architectural design.

## Target Repository

This design is intended for a major breaking refactor of `WHOIGit/image-classifier-dojo`.

The refactor prioritizes:

- a clean supervised + self-supervised training architecture
- strong Pydantic config validation
- Hydra-based config composition and CLI overrides
- single-head and multi-head supervised learning
- Lightly-based self-supervised learning, starting with DINOv2-style training
- snapshot ensemble training and bundling
- local-first artifact persistence with optional Aim or MLflow logging

---

# 1. Goals

## 1.1 Core goals

The refactored project should support:

1. Supervised image classification, regression, and ordinal regression.
2. Multi-head models with one or more output heads.
3. Transfer learning from:
   - torchvision pretrained backbones
   - timm pretrained backbones
   - non-pretrained torchvision/timm backbones
   - local or remote checkpoints
4. Self-supervised learning using Lightly, initially focused on DINOv2-style workflows.
5. SSL evaluation during training using supervised-style labeled datasets.
6. Snapshot ensemble training for supervised models.
7. Bundling snapshot ensembles into a single `.pt` artifact containing multiple snapshot `state_dict`s.
8. Optional export to ONNX, including preprocessing metadata.
9. Dataset loading from:
   - CSV manifests
   - Parquet manifests or Parquet image datasets
   - IFCB bins dataset or equivalent current Dojo dataset support
10. S3 path support for CSV-defined image paths.
11. Potential use of `amplify-storage-utils` for S3 access and local caching.
12. Hyperparameter search through Hydra multirun.
13. Experiment logging through:
   - local files
   - Aim
   - MLflow

## 1.2 Non-goals for this refactor phase

The following are intentionally deferred:

1. Full WebDataset support.
2. Full streaming training from remote object storage.
3. Prefect flows inside the Dojo package.
4. Support for multiple experiment trackers simultaneously.
5. Production model serving framework.
6. Full AutoML or Bayesian optimization.
7. Direct dependency on Meta DINOv2 repositories.

---

# 2. Key architectural decisions

## 2.1 Configuration

Use:

```text
Hydra + Pydantic
```

Hydra is responsible for config composition and CLI overrides.

Pydantic is the source of truth for:

- schema structure
- validation
- default values
- field constraints
- runtime config objects

The flow is:

```text
Hydra config groups
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

Core training code should depend on Pydantic config objects, not Hydra objects.

## 2.2 CLI style

Use one package CLI with subcommands:

```bash
dojo train supervised
dojo train ssl
dojo eval knn
dojo eval linear-probe
dojo eval embeddings
dojo infer
dojo export pt
dojo export onnx
dojo bundle snapshot-ensemble
```

All commands should accept Hydra overrides.

Examples:

```bash
dojo train supervised \
  task=supervised \
  backbone=resnet50 \
  data=csv_local \
  training.max_epochs=50
```

```bash
dojo train ssl \
  task=ssl \
  ssl=dino_v2 \
  backbone=vit_small \
  data=parquet_plankton \
  ssl_eval.knn.enabled=true
```

```bash
dojo train supervised -m \
  backbone=resnet50,efficientnet_b0,vit_b_16 \
  optimizer.lr=1e-4,3e-4 \
  data.batch_size=32,64
```

## 2.3 Local-first artifacts

All metrics and artifacts should always be saved locally.

Experiment trackers are optional sinks.

```text
training run
   ↓
local run directory
   ↓
optional Aim OR MLflow logging
```

The logger choice is mutually exclusive:

```yaml
logging:
  tracker: local_only  # local_only | aim | mlflow
```

## 2.4 Backward compatibility

Major breaking changes are allowed.

The refactor should preserve useful ideas from the current project but should not be constrained by current schemas, module names, or command signatures.

---

# 3. Proposed repository structure

```text
image-classifier-dojo/
  pyproject.toml
  README.md

  configs/
    config.yaml

    task/
      supervised.yaml
      ssl.yaml
      linear_probe.yaml
      knn_eval.yaml
      embedding_export.yaml
      predict.yaml
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
      microscopy_bucketed.yaml
      ssl_dino_v2_microscopy.yaml

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

    logging/
      local_only.yaml
      aim.yaml
      mlflow.yaml

    snapshot_ensemble/
      disabled.yaml
      cosine_snapshots.yaml

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
        predict.py
        export.py
        bundle.py

      config/
        __init__.py
        root.py
        data.py
        backbones.py
        heads.py
        losses.py
        optimizers.py
        schedulers.py
        transforms.py
        ssl.py
        logging.py
        snapshot.py
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

        records/
          __init__.py
          sample.py
          target.py
          prediction.py

        storage/
          __init__.py
          resolver.py
          local.py
          s3.py
          cache.py
          amplify.py

        transforms/
          __init__.py
          build.py
          base.py
          microscopy.py
          letterbox.py
          aspect_bucket.py
          foreground_crop.py
          normalization.py
          ssl_dino.py

        samplers/
          __init__.py
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

        wrappers/
          __init__.py
          supervised.py
          ssl.py
          snapshot_ensemble.py

      tasks/
        __init__.py

        supervised/
          __init__.py
          module.py
          loss.py
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
          knn.py
          linear_probe.py
          embeddings.py
          predictions.py

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

      inference/
        __init__.py
        predictor.py
        outputs.py
        preprocessing.py
        batch_writer.py

      export/
        __init__.py
        pt.py
        onnx.py
        metadata.py

      artifacts/
        __init__.py
        paths.py
        manifest.py
        metrics.py
        predictions.py
        embeddings.py
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
      config/
      data/
      transforms/
      models/
      heads/
      losses/
      metrics/
      logging/
      export/

    integration/
      test_train_supervised.py
      test_train_ssl_dino_v2.py
      test_ssl_eval_callbacks.py
      test_snapshot_ensemble.py
      test_export_pt.py
      test_export_onnx.py
      test_hydra_multirun_config.py
```

---

# 4. Dependency plan

## 4.1 Core dependencies

Core dependencies should support supervised training, configuration, CSV/Parquet datasets, local artifact writing, and PyTorch Lightning training.

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
```

## 4.2 Optional extras

The proposed optional extras are reasonable.

Recommended extras:

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

dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]

storage = [
  "amplify-storage-utils",
]
```

Notes:

- `ssl` contains Lightly.
- `timm` keeps timm optional.
- `onnx` keeps export dependencies optional.
- `aim` and `mlflow` are mutually exclusive at runtime but can both be installed.
- `storage` is recommended in addition to the requested extras because S3/caching is a distinct capability.

---

# 5. Config architecture

## 5.1 Pydantic is the schema source of truth

Hydra config files should be treated as inputs.

Pydantic models define the valid contract.

Core code should receive Pydantic objects.

```python
def train_supervised(cfg: ExperimentConfig) -> RunResult:
    ...
```

not:

```python
def train_supervised(cfg: DictConfig) -> RunResult:
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
    heads: dict[str, HeadConfig] | None = None
    ssl: SSLConfig | None = None
    losses: dict[str, LossConfig] | None = None
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None
    training: TrainingConfig
    logging: LoggingConfig
    artifacts: ArtifactConfig
    snapshot_ensemble: SnapshotEnsembleConfig | None = None
    export: ExportConfig | None = None
    seed: int = 13
```

## 5.3 Hydra composition

Example root config:

```yaml
defaults:
  - task: supervised
  - data: csv_local
  - transforms: microscopy_letterbox
  - backbone: torchvision/resnet50
  - head: single_classification
  - optimizer: adamw
  - scheduler: cosine
  - logging: local_only
  - snapshot_ensemble: disabled
  - _self_

experiment:
  name: plankton_resnet50_baseline
  tags:
    - plankton
    - supervised

seed: 13
```

## 5.4 CLI overrides

Hydra CLI overrides should be the standard override mechanism.

Examples:

```bash
dojo train supervised backbone=torchvision/efficientnet_b0
```

```bash
dojo train supervised \
  data.manifest_uri=s3://bucket/train.csv \
  training.max_epochs=100 \
  optimizer.lr=3e-4
```

```bash
dojo train ssl \
  ssl=dino_v2 \
  ssl_eval.knn.every_fractional_epoch=0.1
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
class SampleRecord(TypedDict):
    image: Tensor
    targets: dict[str, Tensor]
    sample_id: str
    uri: str | None
    metadata: dict[str, Any]
```

This allows the same training modules to work across CSV, Parquet, and IFCB datasets.

## 6.3 CSV datasets

CSV manifests should support:

```text
filename
sample_id
one or more target columns
optional metadata columns
```

Example:

```csv
sample_id,filename,species_idx,quality_idx,biomass,split
abc123,s3://bucket/images/abc123.png,42,0,1.25,train
abc124,/data/images/abc124.png,7,1,0.80,train
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

  storage:
    allow_s3: true
    cache:
      enabled: true
      cache_dir: /tmp/dojo-cache
      backend: amplify_storage_utils
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
metadata...
```

### Mode B: Parquet images

Parquet contains encoded image bytes or arrays.

```text
sample_id
image_bytes
image_format
species_idx
quality_idx
metadata...
```

Config example:

```yaml
data:
  backend: parquet_images
  uri: s3://bucket/datasets/plankton_v1/train/*.parquet
  image_bytes_column: image_bytes
  image_format_column: image_format
  sample_id_column: sample_id
```

## 6.5 IFCB bins dataset

The refactor should preserve the current Dojo IFCB bins functionality through a new dataset/datamodule equivalent.

Suggested modules:

```text
src/dojo/data/datasets/ifcb_bins_dataset.py
src/dojo/data/datamodules/ifcb_bins.py
```

The dataset should conform to the common `SampleRecord` contract.

Config example:

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

## 6.6 S3 and caching

S3 access should be abstracted behind a storage resolver.

Suggested API:

```python
class StorageResolver:
    def open(self, uri: str) -> BinaryIO: ...
    def localize(self, uri: str) -> Path: ...
    def exists(self, uri: str) -> bool: ...
```

Implementations:

```text
LocalStorageResolver
S3StorageResolver
CachedStorageResolver
AmplifyStorageResolver
```

`amplify-storage-utils` should be used through an adapter, not imported throughout the codebase.

```text
data/storage/amplify.py
```

This keeps S3 and cache behavior isolated from datasets.

---

# 7. Transform and preprocessing architecture

## 7.1 Transform goals

Transforms must support both generic image classification and microscopy-specific preprocessing.

Required policies:

```text
resize
center crop
random crop
letterbox / pad
aspect-ratio buckets
foreground-aware crop
grayscale repeat-to-3-channel
domain mean/std normalization
ImageNet-style normalization
DINOv2-style multi-crop transforms
```

## 7.2 Transform stages

Transforms should be organized by stage:

```text
decode
base preprocessing
augmentation
tensor conversion
normalization
```

Example:

```text
load image
  ↓
decode grayscale/RGB
  ↓
foreground-aware crop or full frame
  ↓
resize / letterbox / aspect bucket
  ↓
augment
  ↓
to tensor
  ↓
normalize
```

## 7.3 Letterbox config

```yaml
transforms:
  preset: microscopy_letterbox
  image_mode: grayscale_repeat3

  resize:
    policy: letterbox
    canvas_size: [224, 224]
    preserve_aspect_ratio: true
    pad_mode: background_median
    pad_value: null

  normalization:
    mode: dataset
    mean: [0.42, 0.42, 0.42]
    std: [0.18, 0.18, 0.18]
```

## 7.4 Aspect bucket config

```yaml
transforms:
  preset: microscopy_bucketed
  image_mode: grayscale_repeat3

  resize:
    policy: aspect_bucket
    buckets:
      - size: [224, 224]
        min_aspect: 0.75
        max_aspect: 1.33

      - size: [224, 448]
        min_aspect: 1.33
        max_aspect: 3.0

      - size: [224, 672]
        min_aspect: 3.0
        max_aspect: 99.0

      - size: [448, 224]
        min_aspect: 0.33
        max_aspect: 0.75

      - size: [672, 224]
        min_aspect: 0.0
        max_aspect: 0.33
```

## 7.5 Foreground-aware crop

For centered plankton imagery, foreground-aware crops should prevent DINO local crops from becoming mostly padding/background.

```yaml
transforms:
  foreground_crop:
    enabled: true
    method: threshold_bbox
    threshold_mode: otsu
    expand_margin_fraction: 0.15
    min_foreground_fraction: 0.10
    max_resample_attempts: 10
```

## 7.6 SSL DINOv2 microscopy transform

```yaml
transforms:
  preset: ssl_dino_v2_microscopy

  image_mode: grayscale_repeat3

  dino:
    global_crops:
      count: 2
      size: [224, 224]
      scale: [0.75, 1.0]

    local_crops:
      count: 4
      size: [96, 96]
      scale: [0.20, 0.60]

  augment:
    horizontal_flip: true
    vertical_flip: true
    rotation:
      mode: multiples_of_90
      enabled: true

    brightness:
      enabled: true
      max_delta: 0.20

    contrast:
      enabled: true
      max_delta: 0.20

    gaussian_blur:
      enabled: true
      p: 0.25

    gaussian_noise:
      enabled: true
      std: 0.01
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

The returned tensor should be a 2D embedding:

```text
batch_size x embedding_dim
```

Backbone adapters are responsible for converting model-specific outputs into this contract.

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
  features_only: false
  output_dim: auto
  freeze:
    policy: last_n_blocks
    trainable_blocks: 2
```

## 8.5 Checkpoint transfer learning

Transfer learning should allow loading a checkpoint from local or remote storage.

Config:

```yaml
backbone:
  source: checkpoint
  architecture:
    source: timm
    name: vit_small_patch16_224
    pretrained: false

  checkpoint_uri: s3://bucket/runs/ssl_dino_v2/model/encoder.pt
  checkpoint_key: encoder_state_dict
  strict: false

  freeze:
    policy: none
```

Freeze policies:

```text
none
all
last_n_blocks
except_head
until_epoch
```

## 8.6 Inception and special models

Some torchvision models, such as Inception, may return auxiliary logits or model-specific structures.

Backbone adapters must normalize this.

Rules:

1. Supervised task modules should not contain model-specific logic.
2. Backbone adapters should disable or ignore auxiliary classifier outputs unless explicitly configured.
3. Feature extraction should always return a single embedding tensor.

---

# 9. Head architecture

## 9.1 Head contract

Heads map embeddings to task-specific outputs.

```python
class Head(nn.Module):
    name: str
    task_type: HeadTaskType

    def forward(self, embedding: Tensor) -> Tensor | dict[str, Tensor]:
        ...
```

Supported head task types:

```text
multiclass_classification
binary_classification
multilabel_classification
regression
ordinal_regression
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

Forward output:

```python
{
    "embedding": Tensor,
    "heads": {
        "species": {
            "logits": Tensor,
        },
        "quality": {
            "logits": Tensor,
        },
        "biomass": {
            "prediction": Tensor,
        },
        "life_stage": {
            "ordinal_logits": Tensor,
        },
    },
}
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
      dropout: 0.0
```

Loss options:

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

Loss options:

```text
mse
mae
huber
smooth_l1
gaussian_nll
```

## 9.5 Ordinal regression head

```yaml
heads:
  life_stage:
    type: ordinal_regression
    target_column: life_stage_idx
    num_classes: 5
    network:
      type: linear
```

Loss options:

```text
coral
corn
ordinal_cross_entropy
```

## 9.6 Multi-head loss aggregation

Losses are computed per head and combined with weights.

```yaml
losses:
  species:
    type: class_balanced_effective_number
    beta: 0.999
    weight: 1.0

  quality:
    type: cross_entropy
    weight: 0.25

  biomass:
    type: huber
    delta: 1.0
    weight: 0.10

  life_stage:
    type: coral
    weight: 0.25
```

Total loss:

```text
total_loss =
    species_weight * species_loss
  + quality_weight * quality_loss
  + biomass_weight * biomass_loss
  + ordinal_weight * ordinal_loss
```

---

# 10. Supervised training architecture

## 10.1 LightningModule

The supervised LightningModule should own:

```text
forward pass
training_step
validation_step
test_step
optimizer/scheduler creation
metric updates
loss aggregation
```

It should not own:

```text
dataset-specific path logic
experiment tracker-specific logic
artifact layout decisions
snapshot bundling implementation
```

Suggested class:

```python
class SupervisedTaskModule(L.LightningModule):
    def __init__(
        self,
        model: SupervisedModel,
        losses: MultiHeadLoss,
        metrics: MultiHeadMetricCollection,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None,
    ):
        ...
```

## 10.2 Model composition

```python
backbone = build_backbone(cfg.backbone)
heads = build_heads(cfg.heads, input_dim=backbone.output_dim)
model = SupervisedModel(backbone=backbone, heads=heads)
```

## 10.3 Training command

```bash
dojo train supervised \
  task=supervised \
  data=csv_s3 \
  backbone=torchvision/resnet50 \
  head=multihead_species_quality \
  logging=mlflow
```

## 10.4 Supervised output artifacts

Every supervised run should save:

```text
runs/{run_id}/
  config/
    resolved.yaml
    resolved.json

  checkpoints/
    last.ckpt
    best.ckpt

  model/
    model_final.pt
    model_best.pt

  metrics/
    train_metrics.json
    val_metrics.json
    test_metrics.json
    per_class_metrics.parquet
    confusion_matrix.parquet

  predictions/
    val_predictions.parquet
    test_predictions.parquet

  embeddings/
    val_embeddings.parquet
    test_embeddings.parquet

  logs/
    events.jsonl
```

Predictions should include:

```text
sample_id
uri
split
head_name
target
prediction
logits
probabilities
embedding_uri or embedding row reference
metadata
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
keep encoder
optionally keep projection head for reproducibility
discard projection head for supervised transfer unless explicitly requested
```

## 11.3 SSL task module

Suggested class:

```python
class DinoV2SSLTaskModule(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        ssl_loss: nn.Module,
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
    save_encoder: true
    save_projection_head: true
```

## 11.5 SSL training command

```bash
dojo train ssl \
  task=ssl \
  ssl=dino_v2 \
  data=parquet_images \
  backbone=timm/vit_small_patch16_224 \
  transforms=ssl_dino_v2_microscopy
```

## 11.6 SSL artifacts

```text
runs/{run_id}/
  config/
    resolved.yaml
    resolved.json

  checkpoints/
    last.ckpt
    best_ssl.ckpt

  model/
    encoder_final.pt
    encoder_best.pt
    projection_head_final.pt

  metrics/
    ssl_loss.json
    knn_metrics.json
    linear_probe_metrics.json

  embeddings/
    train_embeddings.parquet
    val_embeddings.parquet
```

---

# 12. SSL evaluation during training

## 12.1 Evaluation methods

The SSL task should support:

```text
online k-NN evaluation
periodic linear probe
embedding export
nearest-neighbor validation metrics
```

These should be configurable independently.

## 12.2 Evaluation dataset

SSL evaluation may use a supervised-style labeled dataset.

Example:

```yaml
ssl_eval:
  labeled_data:
    backend: csv
    manifest_uri: s3://bucket/manifests/supervised_val.csv
    image_uri_column: filename
    sample_id_column: sample_id
    label_column: species_idx

  knn:
    enabled: true
    k: 20
    distance: cosine

  linear_probe:
    enabled: true
    max_epochs: 10
    train_backbone: false

  embedding_export:
    enabled: true
```

## 12.3 Evaluation scheduling

Evaluations should be schedulable by:

```text
every N epochs
every N train batches
every fractional epoch
end of epoch
end of training
```

Config example:

```yaml
ssl_eval:
  knn:
    enabled: true
    schedule:
      every_n_epochs: 1
      every_n_train_batches: null
      every_fractional_epoch: 0.10

  linear_probe:
    enabled: true
    schedule:
      every_n_epochs: 10

  embedding_export:
    enabled: true
    schedule:
      every_n_epochs: 10
```

Implementation detail:

- `every_fractional_epoch: 0.10` means approximately every 10% of an epoch.
- The callback should compute this from `estimated_train_batches`.
- Expensive evaluations should support subsampling.

```yaml
ssl_eval:
  knn:
    max_reference_samples: 50000
    max_query_samples: 10000
```

## 12.4 SSL evaluation outputs

```text
metrics/ssl_eval_knn.json
metrics/ssl_eval_linear_probe.json
embeddings/ssl_eval_epoch_010.parquet
```

---

# 13. Snapshot ensemble training

## 13.1 Concept

Snapshot ensemble training is a supervised training mode that periodically saves model snapshots during a single run.

It should be implemented as:

```text
training strategy + callback + artifact bundler
```

not as a separate model architecture.

The supervised model remains:

```text
backbone → heads
```

## 13.2 Snapshot ensemble config

```yaml
snapshot_ensemble:
  enabled: true

  schedule:
    type: cosine_restarts
    num_snapshots: 5
    cycle_epochs: 20
    save_at: cycle_end

  selection:
    metric: val/species/macro_f1
    mode: max

  artifact:
    save_individual_checkpoints: true
    bundle_single_pt: true
    bundle_filename: snapshot_ensemble.pt

  inference:
    combine: logits_mean
```

## 13.3 Training flow

```text
epoch 1-20
  ↓
save snapshot_001.ckpt

epoch 21-40
  ↓
save snapshot_002.ckpt

epoch 41-60
  ↓
save snapshot_003.ckpt

epoch 61-80
  ↓
save snapshot_004.ckpt

epoch 81-100
  ↓
save snapshot_005.ckpt

bundle snapshots
  ↓
snapshot_ensemble.pt
```

## 13.4 Snapshot artifact format

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

## 13.5 Snapshot ensemble inference

A snapshot ensemble predictor should:

1. Load the bundled `.pt`.
2. Rebuild the model architecture.
3. Load each snapshot state dict one at a time or materialize multiple copies.
4. Run inference for each snapshot.
5. Average logits by head.
6. Apply head-specific post-processing.

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
then decode ordinal prediction
```

## 13.6 Snapshot ensemble artifacts

```text
runs/{run_id}/
  checkpoints/
    snapshot_001.ckpt
    snapshot_002.ckpt
    snapshot_003.ckpt

  ensemble/
    snapshot_ensemble.pt
    snapshot_manifest.json
    snapshot_metrics.json

  predictions/
    val_snapshot_ensemble.parquet
    test_snapshot_ensemble.parquet
```

---

# 14. Model soups, SWA, and EMA

This architecture should leave room for model averaging methods, but the initial design should prioritize snapshot ensembles.

Potential future modules:

```text
src/dojo/ensemble/soup.py
src/dojo/ensemble/swa.py
src/dojo/training/ema.py
```

## 14.1 Greedy soup

Greedy soup can reuse saved checkpoints and evaluate averaged weights.

This should be an evaluation/bundling operation, not a training task.

```bash
dojo bundle soup \
  checkpoints=s3://bucket/runs/run123/checkpoints/*.ckpt \
  selection.metric=val/species/macro_f1
```

## 14.2 SWA

SWA should be treated as a training callback or end-of-training phase.

## 14.3 EMA

EMA should be a training callback that maintains shadow weights.

These methods should not be required for the initial snapshot ensemble path.

---

# 15. Export architecture

## 15.1 Export targets

Supported export formats:

```text
.pt single model
.pt snapshot ensemble
.onnx single model
.onnx snapshot ensemble wrapper
```

The initial priority should be:

1. `.pt` single model
2. `.pt` snapshot ensemble containing multiple snapshot `state_dict`s
3. `.onnx` single model
4. `.onnx` snapshot ensemble wrapper

## 15.2 Export metadata

Exported artifacts should include preprocessing metadata.

Metadata should include:

```text
model architecture
backbone source/name
head definitions
class names
class index mappings
normalization mean/std
image mode
resize policy
letterbox settings
aspect bucket settings
input shape
training config hash
source checkpoint URI
Dojo version
```

## 15.3 `.pt` export

Single model `.pt`:

```python
{
    "artifact_type": "supervised_model",
    "format_version": "1.0",
    "state_dict": {...},
    "model_config": {...},
    "preprocessing": {...},
    "heads": {...},
    "class_mappings": {...},
}
```

Snapshot ensemble `.pt`:

```python
{
    "artifact_type": "snapshot_ensemble",
    "snapshots": [...],
    "model_config": {...},
    "preprocessing": {...},
}
```

## 15.4 ONNX export

ONNX export should write:

```text
model.onnx
model.metadata.json
```

Additionally, metadata should be embedded into ONNX metadata properties when possible.

Config:

```yaml
export:
  format: onnx
  mode: single_model
  checkpoint_uri: runs/run123/checkpoints/best.ckpt
  output_uri: runs/run123/export/model.onnx

  input:
    name: image
    shape: [1, 3, 224, 224]
    dynamic_batch: true

  metadata:
    include_preprocessing: true
    include_class_mappings: true
```

## 15.5 Snapshot ensemble ONNX export

For true ensemble ONNX export, the project can build a wrapper module:

```text
input image
  ↓
snapshot model 1
snapshot model 2
snapshot model 3
  ↓
average logits
  ↓
output ensemble logits
```

This may produce large ONNX files.

Config should allow:

```yaml
export:
  format: onnx
  mode: snapshot_ensemble
  max_snapshots: 5
  combine: logits_mean
```

If ONNX ensemble export becomes too complex, the `.pt` snapshot ensemble artifact remains the primary ensemble deployment format.

---

# 16. Inference and embedding extraction

## 16.1 Inference outputs

Inference should optionally output:

```text
head predictions
raw logits
probabilities
embeddings
novelty scores
metadata
```

Command:

```bash
dojo predict \
  checkpoint=s3://bucket/runs/run123/model/model_best.pt \
  data=parquet_manifest \
  predict.output_embeddings=true \
  predict.output_logits=true
```

## 16.2 Output schema

Predictions Parquet:

```text
sample_id
uri
split
embedding
head_name
target
prediction
logits
probabilities
confidence
entropy
top1_top2_margin
metadata
```

For multi-head predictions:

```text
sample_id
uri
head_name
output_type
target
prediction
raw_output
metadata
```

## 16.3 Embedding export

Embedding export should work with:

```text
supervised checkpoints
SSL encoder checkpoints
snapshot ensemble members
```

Command:

```bash
dojo eval embeddings \
  checkpoint=s3://bucket/runs/ssl/model/encoder_final.pt \
  data=csv_s3 \
  output.uri=s3://bucket/embeddings/plankton_v1.parquet
```

Embedding Parquet schema:

```text
sample_id
uri
split
embedding_model
embedding_dim
embedding
label columns
metadata columns
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
```

## 17.2 Runtime logger selection

```yaml
logging:
  tracker: mlflow  # local_only | aim | mlflow

  local:
    run_root: ./runs

  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: plankton

  aim:
    repo: ./aim
    experiment_name: plankton
```

Only one tracker should be active per run.

Local artifact writing is always active.

## 17.3 Artifact policy

All artifacts are written locally first.

Then logger-specific implementations may upload or register them.

```text
artifact produced
   ↓
save to local run directory
   ↓
optionally log to Aim or MLflow
```

---

# 18. Hyperparameter search

Use Hydra multirun only.

No Optuna or Ray Tune in the initial refactor.

Example:

```bash
dojo train supervised -m \
  backbone=torchvision/resnet50,timm/convnext_tiny \
  optimizer.lr=1e-4,3e-4 \
  data.batch_size=32,64 \
  losses.species.type=cross_entropy,class_balanced_effective_number
```

Each Hydra job should:

1. Compose config.
2. Convert to dict.
3. Validate with Pydantic.
4. Create independent run directory.
5. Save resolved config.
6. Run training/evaluation.
7. Save local artifacts.
8. Optionally log to Aim or MLflow.

---

# 19. CLI design

## 19.1 Top-level command

```bash
dojo --help
```

Subcommands:

```bash
dojo train supervised
dojo train ssl

dojo eval knn
dojo eval linear-probe
dojo eval embeddings

dojo predict

dojo bundle snapshot-ensemble
dojo bundle soup

dojo export pt
dojo export onnx

dojo validate-config
```

## 19.2 Command responsibilities

### `dojo train supervised`

Trains supervised single-head or multi-head models.

### `dojo train ssl`

Trains SSL models using Lightly.

### `dojo eval knn`

Runs k-NN evaluation over embeddings or a checkpoint.

### `dojo eval linear-probe`

Trains a frozen-backbone linear probe.

### `dojo eval embeddings`

Exports embeddings from a checkpoint.

### `dojo predict`

Runs inference and writes predictions.

### `dojo bundle snapshot-ensemble`

Bundles saved snapshots into one `.pt`.

### `dojo export pt`

Exports a model or snapshot ensemble to `.pt`.

### `dojo export onnx`

Exports a single model or ensemble wrapper to ONNX.

### `dojo validate-config`

Composes and validates a config without running training.

---

# 20. Example configs

## 20.1 Supervised single-head classifier

```yaml
defaults:
  - task: supervised
  - data: csv_s3
  - transforms: microscopy_letterbox
  - backbone: torchvision/resnet50
  - optimizer: adamw
  - scheduler: cosine
  - logging: local_only
  - snapshot_ensemble: disabled
  - _self_

experiment:
  name: plankton_resnet50_species

heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    network:
      type: linear

losses:
  species:
    type: class_balanced_effective_number
    beta: 0.999
    weight: 1.0

training:
  max_epochs: 50
  batch_size: 64
  precision: 16-mixed
```

## 20.2 Multi-head supervised model

```yaml
defaults:
  - task: supervised
  - data: parquet_manifest
  - transforms: microscopy_bucketed
  - backbone: timm/convnext_tiny
  - optimizer: adamw
  - scheduler: cosine
  - logging: mlflow
  - _self_

experiment:
  name: plankton_multihead_convnext

heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    network:
      type: linear

  quality:
    type: multiclass_classification
    target_column: quality_idx
    num_classes: 4
    network:
      type: linear

  biomass:
    type: regression
    target_column: biomass
    output_dim: 1
    network:
      type: linear

  life_stage:
    type: ordinal_regression
    target_column: stage_idx
    num_classes: 5
    network:
      type: linear

losses:
  species:
    type: focal
    gamma: 2.0
    weight: 1.0

  quality:
    type: cross_entropy
    weight: 0.25

  biomass:
    type: huber
    delta: 1.0
    weight: 0.10

  life_stage:
    type: coral
    weight: 0.25
```

## 20.3 SSL DINOv2-style training with Lightly

```yaml
defaults:
  - task: ssl
  - data: parquet_images
  - transforms: ssl_dino_v2_microscopy
  - backbone: timm/vit_small_patch16_224
  - ssl: dino_v2
  - optimizer: adamw
  - scheduler: cosine
  - logging: aim
  - _self_

experiment:
  name: plankton_dino_v2_ssl

training:
  max_epochs: 300
  batch_size: 128
  precision: 16-mixed

ssl_eval:
  labeled_data:
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
    schedule:
      every_n_epochs: 10
    max_epochs: 10

  embedding_export:
    enabled: true
    schedule:
      every_n_epochs: 10
```

## 20.4 Snapshot ensemble supervised training

```yaml
defaults:
  - task: supervised
  - data: csv_s3
  - transforms: microscopy_letterbox
  - backbone: torchvision/efficientnet_b0
  - optimizer: adamw
  - scheduler: cosine_restarts
  - logging: mlflow
  - snapshot_ensemble: cosine_snapshots
  - _self_

experiment:
  name: plankton_efficientnet_snapshot_ensemble

training:
  max_epochs: 100
  batch_size: 64

scheduler:
  type: cosine_restarts
  cycle_epochs: 20

snapshot_ensemble:
  enabled: true
  schedule:
    type: cosine_restarts
    num_snapshots: 5
    cycle_epochs: 20
    save_at: cycle_end

  artifact:
    save_individual_checkpoints: true
    bundle_single_pt: true
    bundle_filename: snapshot_ensemble.pt

  inference:
    combine: logits_mean
```

## 20.5 Transfer learning from SSL checkpoint

```yaml
defaults:
  - task: supervised
  - data: csv_s3
  - transforms: microscopy_bucketed
  - optimizer: adamw
  - scheduler: cosine
  - logging: local_only
  - _self_

experiment:
  name: plankton_transfer_from_ssl_dino

backbone:
  source: checkpoint
  architecture:
    source: timm
    name: vit_small_patch16_224
    pretrained: false

  checkpoint_uri: s3://bucket/runs/plankton_dino_v2_ssl/model/encoder_final.pt
  checkpoint_key: encoder_state_dict
  strict: false

  freeze:
    policy: last_n_blocks
    trainable_blocks: 4

heads:
  species:
    type: multiclass_classification
    target_column: species_idx
    num_classes: 120
    network:
      type: linear
```

---

# 21. Workflow summaries

## 21.1 Supervised training workflow

```text
Hydra compose config
  ↓
Pydantic validate config
  ↓
create run directory
  ↓
build datamodule
  ↓
build transforms
  ↓
build backbone
  ↓
build heads
  ↓
build losses and metrics
  ↓
train LightningModule
  ↓
save checkpoints
  ↓
save metrics
  ↓
save predictions and optional embeddings
  ↓
save final .pt artifact
  ↓
optionally log to Aim or MLflow
```

## 21.2 SSL training workflow

```text
Hydra compose config
  ↓
Pydantic validate config
  ↓
build unlabeled datamodule
  ↓
build DINOv2-style Lightly transforms
  ↓
build encoder backbone
  ↓
build Lightly SSL head/loss
  ↓
train SSL LightningModule
  ↓
periodically evaluate with k-NN / linear probe / embeddings
  ↓
save encoder checkpoint
  ↓
save metrics and embeddings
  ↓
optionally log to Aim or MLflow
```

## 21.3 Transfer learning workflow

```text
load pretrained torchvision/timm/checkpoint backbone
  ↓
replace or attach supervised heads
  ↓
freeze according to policy
  ↓
train heads or fine-tune backbone
  ↓
evaluate
  ↓
export model
```

## 21.4 Snapshot ensemble workflow

```text
train supervised model with restart schedule
  ↓
save snapshots at configured cycle boundaries
  ↓
evaluate individual snapshots
  ↓
bundle snapshots into single .pt
  ↓
run ensemble prediction
  ↓
save ensemble metrics
  ↓
optionally export ONNX wrapper
```

---

# 22. Testing strategy

## 22.1 Test fixture structure

```text
tests/fixtures/
  images/
    grayscale/
    rgb/
    extreme_aspect_ratio/
    corrupt/

  manifests/
    train.csv
    val.csv
    multihead.csv
    s3_paths.csv

  parquet/
    manifest.parquet
    images.parquet

  configs/
    supervised_minimal.yaml
    supervised_multihead.yaml
    ssl_dino_v2_minimal.yaml
    snapshot_ensemble.yaml
    export_onnx.yaml

  checkpoints/
    tiny_resnet.ckpt
    tiny_ssl_encoder.pt

  ifcb_bins/
    sample_manifest.csv
    sample_images/
```

## 22.2 Config tests

Test:

```text
Hydra config composition
Pydantic validation success
Pydantic validation failure for invalid configs
CLI override validation
missing required fields
invalid head/loss combinations
invalid dataset columns
invalid tracker choice
invalid snapshot ensemble config
```

Examples:

```text
classification head requires num_classes
regression head requires output_dim
ordinal head requires num_classes > 1
focal loss only valid for classification heads
CORAL loss only valid for ordinal heads
snapshot ensemble requires supervised task
```

## 22.3 Dataset tests

Test:

```text
CSV dataset loads local images
CSV dataset accepts S3-style paths through mocked storage resolver
Parquet manifest loads image paths
Parquet image dataset decodes image bytes
IFCB bins dataset conforms to SampleRecord contract
multi-head targets are correctly parsed
missing target handling
bad image handling
sample_id propagation
metadata propagation
```

## 22.4 Transform tests

Test:

```text
letterbox preserves aspect ratio
letterbox returns configured canvas size
aspect bucket assigns expected bucket
foreground-aware crop avoids mostly background crops
grayscale repeat-to-3 returns 3 channels
normalization applies expected shape and dtype
DINOv2 SSL transform returns expected number of views
extreme aspect ratio image does not get squashed
```

## 22.5 Backbone tests

Test:

```text
torchvision ResNet feature extractor returns [B, D]
torchvision EfficientNet feature extractor returns [B, D]
torchvision Inception adapter handles aux outputs
torchvision ViT feature extractor returns [B, D]
timm backbone works when timm extra installed
checkpoint backbone loads state_dict
freeze policies correctly set requires_grad
```

## 22.6 Head tests

Test:

```text
classification head output shape
regression head output shape
ordinal head output shape
multi-head forward output dict
invalid target/head mismatch raises validation error
```

## 22.7 Loss tests

Test:

```text
cross entropy
weighted cross entropy
effective-number class-balanced loss
focal loss
label smoothing
MSE
MAE
Huber
CORAL/CORN ordinal loss
multi-head weighted aggregation
```

## 22.8 Supervised training smoke tests

Use tiny synthetic datasets and tiny models.

Test:

```text
single-head supervised training runs for 1 epoch
multi-head supervised training runs for 1 epoch
metrics are written locally
checkpoints are written locally
predictions are exported
embeddings are optionally exported
```

## 22.9 SSL training smoke tests

Behind `ssl` extra.

Test:

```text
DINOv2-style Lightly task runs for a few batches
multi-crop transform returns valid views
online k-NN callback runs
linear probe callback can be invoked
encoder artifact is saved
```

## 22.10 Snapshot ensemble tests

Test:

```text
snapshot callback saves snapshots
bundle command creates one .pt file
bundle contains multiple state_dicts
snapshot ensemble predictor averages logits
snapshot ensemble metrics are written
snapshot ensemble works with multi-head model
```

## 22.11 Export tests

Behind `onnx` extra.

Test:

```text
single .pt export contains metadata
snapshot ensemble .pt export contains snapshots
ONNX single-model export creates model.onnx
ONNX metadata JSON includes preprocessing
ONNX runtime can load exported model
```

## 22.12 Logging tests

Test:

```text
local logger writes metrics/artifacts
Aim logger adapter can be constructed when aim extra installed
MLflow logger adapter can be constructed when mlflow extra installed
config prevents aim and mlflow simultaneously
local artifacts are created even when tracker is enabled
```

---

# 23. Migration plan

## Phase 1: Config and CLI foundation

1. Add Hydra entrypoint.
2. Add Pydantic schema package.
3. Add `dojo validate-config`.
4. Implement local artifact run directory.
5. Add logger abstraction with local-only implementation.

## Phase 2: Supervised refactor

1. Introduce common dataset record contract.
2. Implement CSV datamodule.
3. Implement Parquet datamodule.
4. Port IFCB bins dataset to new dataset contract.
5. Add backbone registry.
6. Add head registry.
7. Add supervised LightningModule.
8. Add multi-head loss and metrics.
9. Add basic supervised training command.

## Phase 3: Transform refactor

1. Add microscopy transform presets.
2. Add letterbox.
3. Add aspect buckets.
4. Add foreground-aware crop.
5. Add normalization config.
6. Add transform tests.

## Phase 4: SSL with Lightly

1. Add `dojo[ssl]` extra.
2. Add DINOv2-style Lightly task module.
3. Add SSL transform preset.
4. Add online k-NN eval callback.
5. Add linear probe evaluation command.
6. Add SSL embedding export.

## Phase 5: Snapshot ensemble

1. Add snapshot training callback.
2. Add restart scheduler config.
3. Add snapshot bundle artifact.
4. Add ensemble predictor.
5. Add ensemble metrics/export.

## Phase 6: Export

1. Add `.pt` single-model export.
2. Add `.pt` snapshot ensemble export.
3. Add ONNX single-model export.
4. Add ONNX metadata sidecar and embedded metadata.
5. Add optional ONNX snapshot ensemble wrapper.

## Phase 7: Experiment tracking

1. Add Aim logger adapter.
2. Add MLflow logger adapter.
3. Ensure local-first artifact policy.
4. Add tracking tests.

---

# 24. Design principles

## 24.1 Keep task logic separate from model structure

A backbone/head model should not know whether it is used for:

```text
supervised training
linear probing
snapshot ensemble
embedding extraction
prediction
```

## 24.2 Keep Hydra out of core logic

Hydra should be limited to CLI/config composition.

Core modules should use Pydantic configs.

## 24.3 Keep storage concerns out of datasets

Datasets should request bytes or localized paths through a storage resolver.

They should not contain S3/cache implementation details.

## 24.4 Save everything needed for reproducibility

Every run should save:

```text
resolved config
code/package version if available
dataset references
model architecture config
preprocessing metadata
checkpoint provenance
metrics
predictions
artifacts
```

## 24.5 Make embeddings first-class

Embeddings should be extractable from:

```text
supervised models
SSL encoders
snapshot ensemble members
transfer-learning checkpoints
```


---

# 25. Open implementation details for later design docs

This architectural doc intentionally does not fully specify:

1. Exact Pydantic class definitions.
2. Exact Lightly DINOv2 implementation details.
3. Exact ONNX wrapper implementation for snapshot ensembles.
4. Exact `amplify-storage-utils` adapter API.
5. Exact metric naming conventions.
6. Exact Parquet schema serialization for vectors/logits.
7. Exact migration of current code modules line-by-line.

Recommended follow-up docs:

```text
01_config_schema_design.md
02_supervised_module_design.md
03_ssl_dino_v2_design.md
04_dataset_storage_design.md
05_snapshot_ensemble_design.md
06_export_and_inference_design.md
07_testing_plan.md
```
