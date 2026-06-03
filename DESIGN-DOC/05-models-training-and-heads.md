
# 05. Models, Training, and Heads

## Purpose

Defines the model composition pipeline (transforms → backbone → optional
tabular fusion → optional embedding adapter → heads), the head /
objective contract, the supervised training LightningModule, optimizer /
scheduler / checkpointing config, and supervised transfer learning. This
file absorbs what the original draft split between transforms, backbones,
heads, objectives, and supervised training.

## Transforms and preprocessing

Avoid hard-coding domain-specific transform modules where YAML
composition can express the behavior.

```text
Python transform modules = reusable operations
YAML configs             = experiment/domain-specific recipes
```

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

Pipeline example:

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

Aspect / size bucketing should support both `aspect_ratio` buckets
(preserve morphology for elongated organisms) and `size`-aware buckets
(avoid artificially upscaling tiny ROIs). Scale-related fields are
recorded as `sample_metadata` columns (`native_width_px`,
`native_height_px`, `resize_width_px`, `resize_height_px`,
`resize_bucket`, `microns_per_pixel`) — see
`06-results-artifacts-and-metadata.md`.

## Backbones

### Contract

```python
class Backbone(nn.Module):
    output_dim: int

    def forward_features(self, x: Tensor) -> Tensor:
        ...
```

Returned tensor is usually `batch_size x embedding_dim`.

### Supported sources

`model.backbone.source`:

- `torchvision`
- `timm`
- `checkpoint`

`timm` is **functional** in the initial implementation, gated by the
`timm` optional extra. The schema accepts `source: timm` regardless of
install; the runtime raises a clear error if `timm` is not installed.

`source: lightly` is **not** introduced. Lightly remains an SSL framework
implementation detail; SSL configs select it via `ssl.framework: lightly`
while the backbone is still selected through `torchvision`, `timm`, or
`checkpoint`. DINOv2 (Lightly) is allowed to use timm ViT backbones
internally and through the public `source: timm` selector.

Inception-style auxiliary-logit handling is out of scope for the generic
backbone path — see `appendix-deferred-features.md`.

### Examples

Torchvision:

```yaml
model:
  backbone:
    source: torchvision
    name: resnet50
    pretrained: true
    weights: DEFAULT
    output_dim: auto
    freeze:
      policy: none
```

timm:

```yaml
model:
  backbone:
    source: timm
    name: vit_small_patch16_224
    pretrained: true
    output_dim: auto
    freeze:
      policy: last_n_blocks_trainable
      n: 2
```

Checkpoint:

```yaml
model:
  backbone:
    source: checkpoint
    architecture:
      source: timm
      name: vit_small_patch14_dinov2
      pretrained: false
    checkpoint_uri: s3://bucket/runs/ssl_dino_v2/exports/encoder.pt
    checkpoint_key: encoder_state_dict
    strict: false
    freeze:
      policy: last_n_blocks_trainable
      n: 4
```

`backbone.output_dim: auto` is the default and should usually not be
changed manually.

### Freeze policies

Named in terms of what remains trainable:

```text
none
all
last_n_blocks_trainable
named_modules_trainable
named_modules_frozen
after_module_trainable
```

`dojo inspect backbone` reports per-module shape, parameter count, and
trainable / frozen status with a freeze policy applied. Use it to pick
module names. See `02-cli-and-task-types.md`.

### Supervised transfer learning from SSL

Transfer learning from an SSL-pretrained encoder is **not a new task
type**. It is plain `task.type: supervised` with
`model.backbone.source: checkpoint` and `checkpoint_uri` pointing at the
SSL encoder export. Freeze policy, embedding adapter, and heads are
configured exactly as for any supervised run.

`strict: false` maps to PyTorch's `load_state_dict(..., strict=False)`
and governs **backbone** keys only. Heads in the new run come from the
new `heads:` block; nothing from the source checkpoint's head ever
appears in the new model.

When `strict: false`, Dojo logs missing and unexpected keys clearly.

Frozen feature extractor recipe:

```yaml
model:
  backbone:
    source: checkpoint
    architecture:
      source: torchvision
      name: resnet50
      pretrained: false
    checkpoint_uri: s3://bucket/runs/ssl_dino_v2/exports/encoder.pt
    checkpoint_key: encoder_state_dict
    strict: false
    freeze:
      policy: all

  embedding_adapter:
    enabled: true
    type: mlp
    hidden_dims: [512]
    output_dim: 256
    activation: gelu
    dropout: 0.1
```

For a true linear-probe evaluation, leave `embedding_adapter` disabled
and let the head's `network: linear` consume the raw backbone embedding.

## Tabular features and fusion

Tabular features may be useful model inputs (size descriptors, depth,
temperature, etc.). Tabular / image fusion is configured under `model:`:

```yaml
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

  embedding_adapter:
    enabled: true
    type: mlp
    hidden_dims: [512]
    output_dim: 256
    activation: gelu
    dropout: 0.1
```

`model.tabular.fusion` is the nested key — there is no top-level
`model.fusion`. Initial fusion type: `concat` / `concat_mlp`. More
complex fusion lives in the deferred backlog.

Model flow:

```text
image → backbone → image_embedding
tabular features → tabular_encoder → tabular_embedding
  ↓ fusion (concat) → fused_embedding
  ↓ optional embedding_adapter → head_input_embedding
  ↓ head(s)
```

Exported model artifacts must include tabular feature names, ordering,
encodings, and normalization statistics (see `10-export.md`).

## Heads, objectives, and the reference chain

```text
objective -> head -> data target -> physical column
```

Heads define **output structure** (type, num_classes, network spec, the
logical `target` they read). Objectives define **training intent** (loss,
metrics, weight). Heads do **not** own loss config.

### Head types

```text
multiclass_classification
binary_classification
multilabel_classification          # reserved for true multi-hot multilabel
regression
ordinal_classification             # head predicts a discrete ordered bin
distributional_regression
count_regression
```

`multilabel_classification` is reserved for true multi-hot multilabel
problems; the old `multilabel` module (which was actually multi-head
multiclass) is not ported. Multi-head multiclass uses one
`multiclass_classification` head per target. The old module is preserved
under `dojo_deprecated` for reference.

### Required head fields

```text
type               # one of the head types above
target             # logical data target key from data.targets
network            # head sub-network spec (e.g. {type: linear})
```

Type-specific required fields:

```text
multiclass_classification / binary_classification / multilabel_classification:
  num_classes

regression:
  output_dim     (default: 1)

ordinal_classification:
  num_classes

distributional_regression:
  distribution   (e.g. gaussian, negative_binomial)

count_regression:
  output_dim     (default: 1)
```

Pydantic validation ensures the resolved `target` exists in
`data.targets` and has a compatible dtype.

### Ordinal naming and result columns

- Head type: `ordinal_classification` (the head predicts a discrete
  ordered bin, not a continuous value).
- Supported ordinal losses: `coral`, `corn`, `ordinal_cross_entropy`.
- Result `record_type` values: `ordinal_output` for native ordinal heads;
  `ordinal_probe_prediction` for ordinal probes.
- Both `ordinal_logits` (raw cumulative logits for CORAL/CORN) and
  `probabilities` (per-bin probabilities) are populated; for CORAL/CORN
  the writer derives `probabilities` by differencing cumulative
  probabilities decoded from `ordinal_logits`.

See `06-results-artifacts-and-metadata.md`.

### Single- vs. multi-head

There are no separate single-head / multi-head model classes. A
single-head model is a special case with exactly one head and one
objective. Internally, configs normalize to the canonical
multi-head / multi-objective shape.

### Reference-chain example

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
      network:
        type: linear

objectives:
  species:
    head: species
    loss: cross_entropy
```

### Multi-head example

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
      network:
        type: linear
    life_stage:
      type: ordinal_classification
      target: life_stage
      num_classes: 5
      network:
        type: linear
    biovolume:
      type: regression
      target: biovolume
      output_dim: 1
      network:
        type: linear

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
    loss:
      type: huber
      delta: 1.0
    target_transform:
      type: log1p_standardize
    metrics: [mae, rmse, r2]
    weight: 0.25
```

### Compatible losses

Classification: `cross_entropy`, `weighted_cross_entropy`,
`class_balanced_effective_number`, `focal`,
`label_smoothing_cross_entropy`.

Regression: `mse`, `mae`, `huber`, `smooth_l1`, `gaussian_nll`,
`poisson_nll`, `negative_binomial_nll`, `quantile`.

Ordinal: `coral`, `corn`, `ordinal_cross_entropy`.

### Target transforms

Common transforms (regression / ordinal heads): `identity`, `standardize`,
`log1p`, `log1p_standardize`, `power` / Box-Cox / Yeo-Johnson. See
`06-results-artifacts-and-metadata.md` for the external vs. internal
column convention.

### Total loss

The total loss is the weighted sum of objective losses:

```text
total_loss = sum( objective_i.weight * objective_i.loss )
```

### Objective shorthand

For simple configs, an objective name may imply the head name:

```yaml
objectives:
  species:
    loss:
      type: focal
      gamma: 2.0
    weight: 1.0
```

resolves to:

```yaml
objectives:
  species:
    head: species
    loss:
      type: focal
      gamma: 2.0
    weight: 1.0
```

### Pydantic validation

Validation enforces:

- every objective references an existing head;
- loss is compatible with the referenced head type;
- metrics are compatible with the referenced head type;
- target transforms are compatible with the head type;
- objective weights are non-negative;
- at least one objective is enabled for supervised training.

## Supervised training

### LightningModule responsibilities

Owns:

- forward-pass orchestration;
- `training_step`, `validation_step`, `test_step`;
- optimizer / scheduler creation;
- per-objective loss computation;
- weighted total-loss sum;
- metric updates derived from objective definitions.

Does **not** own dataset path logic, experiment-tracker specifics,
artifact layout, snapshot bundling, export, or result-file
serialization.

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

Constructor arguments are serializable Pydantic configs so Lightning
hyperparameter checkpoints stay clean.

### Model composition

```python
backbone = build_backbone(cfg.model.backbone)

tabular_encoder = (
    build_tabular_encoder(cfg.model.tabular) if cfg.model.tabular.enabled else None
)

embedding_adapter = (
    build_embedding_adapter(
        cfg.model.embedding_adapter,
        image_embedding_dim=backbone.output_dim,
        tabular_embedding_dim=tabular_encoder.output_dim if tabular_encoder else 0,
    )
    if cfg.model.embedding_adapter.enabled
    else None
)

heads = build_heads(
    cfg.model.heads,
    input_dim=embedding_adapter.output_dim if embedding_adapter else backbone.output_dim,
)

model = SupervisedModel(
    backbone=backbone,
    tabular_encoder=tabular_encoder,
    embedding_adapter=embedding_adapter,
    heads=heads,
)
```

### Optimizer, scheduler, checkpointing

`optimizer`, `scheduler`, and `checkpointing` are top-level config groups
peer to `training`. Checkpointing supports best-k, last, epoch, step,
and snapshot checkpoints. Snapshot-cycle checkpointing for
`task.type: snapshot_ensemble`:

```yaml
checkpointing:
  monitor: val/species/macro_f1
  mode: max
  save_top_k: 3
  save_last: true
  save_cycle_snapshots:
    enabled: true
    at_cycle_end: true
```

The cosine-warm-restarts scheduler is the recommended snapshot-cycle
scheduler. See `02-cli-and-task-types.md` for the
`task.type: snapshot_ensemble` flow and `08-ensembles.md` for the
candidate-selection step.

Portable `.pt` / `.onnx` files are **not** automatic training
artifacts — they live under `exports/` and are produced by explicit
export configuration (`training_outputs.export`) or `dojo export`. See
`10-export.md`.

### Early stopping

Early stopping is a training-loop behavior under `training:`, not
`runtime:`:

```yaml
training:
  early_stopping:
    enabled: true
    monitor: val/loss
    mode: min
    patience: 10
```

### Representation evaluation scheduling

A supervised run may schedule representation evaluation against its own
encoder during training. The same `representation_eval` config is used
standalone via `dojo eval representation`. See
`07-ssl-and-representation-eval.md`.

## Cross-References

- `02-cli-and-task-types.md` — `task.type: supervised`,
  `task.type: snapshot_ensemble`, `dojo inspect backbone`.
- `03-configuration.md` — placement of `model:`, `transforms:`,
  `training:`, `optimizer:`, `scheduler:`, `checkpointing:`,
  `objectives:`.
- `04-data-and-storage.md` — `data.targets` referenced by heads.
- `06-results-artifacts-and-metadata.md` — record types
  (`classification_output`, `regression_output`, `ordinal_output`,
  `sample_metadata`), external vs. internal target/prediction columns,
  embedding kinds.
- `07-ssl-and-representation-eval.md` — supervised encoders fed into
  representation eval; SSL transfer-learning recipe.
- `08-ensembles.md` — snapshot-cycle scheduler feeds snapshot ensembles.
- `10-export.md` — exports/ artifacts and their metadata.
- `11-dependencies.md` — `train` and `timm` optional extras.
- `glossary.md` — head-type and backbone-source vocabulary.
