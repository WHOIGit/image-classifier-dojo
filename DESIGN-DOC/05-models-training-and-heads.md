
# 05. Models, Training, and Heads

## Purpose

Defines the model composition pipeline (transforms → backbone → optional
tabular input encoder → implicit input concatenation → optional embedding
adapter → heads), the head / objective contract, the supervised training
LightningModule, optimizer / scheduler / checkpointing config, and
supervised transfer learning. This file absorbs what the original draft
split between transforms, backbones, heads, objectives, and supervised
training.

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

    - name: aspect_bucket
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
      train_only: true

    - name: horizontal_flip
      p: 0.5
      train_only: true

    - name: normalize
      mode: dataset
      mean: [0.42, 0.42, 0.42]
      std: [0.18, 0.18, 0.18]
```

The `aspect_bucket` transform buckets by aspect ratio and/or native size
(`bucket_by`), supporting both aspect-ratio buckets (preserve morphology for
elongated organisms) and size-aware buckets (avoid artificially upscaling
tiny ROIs). Scale-related fields are recorded as `sample_metadata` columns
(`native_width_px`, `native_height_px`, `resize_width_px`,
`resize_height_px`, `aspect_bucket`, `microns_per_pixel`) — see
`06-results-artifacts-and-metadata.md`.

### Bucketed batching (`batch_aspect_buckets`)

`aspect_bucket` produces variable canvas sizes *across* buckets but a fixed
size *within* a bucket. Tensors in a batch must stack to identical H × W, so
a bucketed run batches **within** a single bucket rather than across —
otherwise the whole point of bucketing (avoiding letterbox padding waste) is
lost.

The `batch_aspect_buckets` sampler is the consumer that makes this work: it
groups samples by their `aspect_bucket` assignment and emits
size-homogeneous batches. That assignment is a deterministic function of
each sample's native dimensions and the resolved bucket scheme, materialized
as a working-manifest column at run setup from cached dimensions — so the
sampler is a column lookup with **no per-batch image I/O**, and changing the
bucket scheme only re-derives the column (see `04-data-and-storage.md`). The
chain reads: the `aspect_bucket` transform assigns each sample an
`aspect_bucket` column value, and the `batch_aspect_buckets` sampler groups
batches by it.

### Class-balanced sampling

The `class_balanced` sampler weights sampling toward under-represented
classes using the frozen per-class counts from the dataset stats cache
(`dojo inspect dataset --class-counts`, `04-data-and-storage.md`) — the same
counts the weighted losses consume — so it never re-tallies the dataset at
run start. Sampler selection (`class_balanced`, `batch_aspect_buckets`,
`weighted`, or unsampled) is a data-loading concern. Class-balancing and
bucketed batching compose with a precedence rule: a batch must stay within
one `aspect_bucket`, so bucket grouping is the outer constraint and class
weighting applies within each bucket.

### Train-only vs. always-on steps

`transforms.pipeline` is a single ordered list so interleaving is explicit
(deterministic ops and augmentation can alternate, e.g. crop → augment →
resize → augment → normalize). Each step may set:

- `enabled` — global on/off (default `true`).
- `train_only` — when `true`, the step runs during `train` only and is
  dropped for every non-train stage (`val`, `predict`, export). Default
  `false` (always-on). Stochastic augmentation (random rotation, flip,
  blur, noise) sets `train_only: true`; deterministic preprocessing
  (foreground crop, bucketed resize, normalize) leaves it `false`.

A step is augmentation by **stochastic intent**, marked per step — not by
module identity. A `crop` may be a deterministic center crop (always-on)
or a random crop (`train_only`), and `rotation` may be a fixed or a random
rotation; the flag, not the module, draws the line.

`image_mode` is a load-time channel-layout policy, not a pipeline step: it
is singular, always-on, and defines the channel **contract** (count /
layout) the pipeline and backbone assume. Values: `rgb` (3-channel color),
`grayscale` (1-channel — requires a backbone that accepts `in_chans=1`,
e.g. timm; torchvision ImageNet models expect 3 and need
`grayscale_repeat3` instead), `grayscale_repeat3` (decode 1-channel,
broadcast to 3 channels for an RGB-pretrained backbone). It stays a
top-level `transforms` field.

The `grayscale` pipeline module is **orthogonal** to `image_mode` and is
not a duplicate of it. `image_mode` sets the channel container; the
`grayscale` module operates on pixel **content** — desaturating color to
luminance — while leaving the channel count to `image_mode`. They compose:
`image_mode: rgb` plus a `grayscale` step (always-on) gives deterministic
luminance carried in 3 channels, and `image_mode: rgb` plus a `grayscale`
step with `train_only: true` and a probability is random-grayscale
augmentation. For inherently single-channel sources (e.g. IFCB),
`image_mode: grayscale_repeat3` already yields desaturated content and the
`grayscale` module is unnecessary.

Config compilation materializes a derived **`inference_pipeline`**: the
ordered subset of `pipeline` where each step is `enabled` and not
`train_only`, with parameters baked in. It is the pipeline used by every
non-train stage and by exported models, and it is the sole input the
`preprocessing_hash` extractor reads for transform steps
(`06-results-artifacts-and-metadata.md`). `inference_pipeline` is
**resolved-only**: validation rejects it in authored configs and it must
not be hand-edited. Resolved configs therefore carry both the full
training `pipeline` and the derived `inference_pipeline`; SSL multi-view
augmentation is configured separately under `ssl:`
(`07-ssl-and-representation-eval.md`) and is not part of this pipeline.

### Pixel value range and bit depth

The transform builder follows a fixed value-range convention, so range is
never an independent config knob: **decode → scale to `[0.0, 1.0]` →
`normalize`**. Raw integer pixels are scaled to floats in `[0, 1]`, then the
`normalize` step applies `mean` / `std`. The model-facing range is therefore
an emergent property of `normalize`, not a separate setting — `mean: 0.5,
std: 0.5` yields `[-1, 1]`, ImageNet stats yield roughly `[-2, 2.6]`, and so
on. Normalization must match the backbone's pretraining (ImageNet stats for
torchvision / timm ImageNet weights, DINOv2's expected stats for DINOv2);
there is deliberately no `pixel_range` enum. When `normalize: {mode:
dataset}`, the mean / std are produced once by `dojo inspect dataset
--stats --normalization` (the decode-tier pass; bare `--stats` does not
read pixels), frozen into the resolved config, and read from the dataset
stats cache at resolution (`04-data-and-storage.md`).

The scale-to-`[0, 1]` divisor depends on source bit depth, set with
`transforms.input_bit_depth`:

```yaml
transforms:
  image_mode: grayscale_repeat3
  input_bit_depth: auto   # auto | 8 | 12 | 16 → divide by 255 / 4095 / 65535
```

Default `auto`, which resolves at config-compile time from the storage
dtype (`uint8` → 8, `uint16` → 16) plus format metadata when present (e.g.
TIFF `BitsPerSample`, which catches 12-bit data). The resolved value is
**materialized as a concrete integer** in the resolved config — decided
once, frozen, and never a per-image runtime decision, so the same raw
sample always scales identically. `dojo inspect dataset --bit-depth`
performs this resolution and flags heterogeneous depths
(`04-data-and-storage.md`).

`auto` cannot disambiguate the one genuinely ambiguous case: a 12- (or 10-,
14-) bit image stored in a 16-bit container with no bit-depth metadata
reads as `uint16` (max 65535) though its true maximum is 4095, and dividing
by the wrong number silently rescales every pixel. High-bit-depth IFCB /
microscopy sources of that kind must set `input_bit_depth` explicitly;
content-based guessing (per-image or dataset-wide max scans) is rejected
because it makes scaling data-dependent and breaks on unseen inference
images.

`input_bit_depth` is a load-time decode policy — singular and always-on,
like `image_mode` — and is part of the input contract, so it contributes to
`preprocessing_hash` as its **resolved integer**, never as the literal
`auto`.

## Backbones

### Contract

```python
class Backbone(nn.Module):
    output_dim: int

    def forward_features(self, x: Tensor) -> Tensor:
        ...
```

Returned tensor is usually `batch_size x embedding_dim`.

`model.image_input.name` names the image input stream and defaults to
`image`. `model.image_input.backbone` holds the image backbone config.
`model.image_input.backbone.architecture` describes the module shape;
`model.image_input.backbone.weights` describes how that module is
initialized.

This keeps the input-stream name separate from
`model.image_input.backbone.architecture.name`, which is the backbone
architecture selector (`resnet50`, `vit_small_patch16_224`, etc.) and is
used by config templates such as
`{model.image_input.backbone.architecture.name:slug}`.

### Architecture and weights sources

`model.image_input.backbone.architecture.source`:

- `torchvision`
- `timm`

`model.image_input.backbone.weights.source`:

- `none` — initialize from the architecture provider's default random
  initialization.
- `library` — initialize from provider-native pretrained weights
  (`weights.name` for torchvision, provider default for timm unless a
  specific name is supported).
- `checkpoint` — initialize from a Dojo checkpoint / exported encoder
  using `weights.uri`, `weights.key`, and `weights.strict`.

Authored configs may use `weights.name: DEFAULT` as a Dojo convenience
alias for provider-native library weights. For torchvision, Dojo maps it to
torchvision's native `DEFAULT` weight enum for the selected architecture,
then stores the concrete resolved enum/name. For timm, Dojo maps it to
timm's default pretrained configuration for the selected architecture
(`pretrained=True` behavior), then stores the resolved pretrained config
identity (for example the resolved tag / config name / HF Hub id when
available). Resolved configs, saved config artifacts, and provenance never
store the moving `DEFAULT` alias.

`timm` is **functional** in the initial implementation, gated by the
`timm` optional extra. The schema accepts `architecture.source: timm`
regardless of install; the runtime raises a clear error if `timm` is not
installed.

`architecture.source: lightly` is **not** introduced. Lightly remains an
SSL framework implementation detail; SSL configs select it via
`ssl.framework: lightly` while the backbone architecture is still selected
through `torchvision` or `timm`. DINOv2 (Lightly) is allowed to use timm
ViT backbones internally and through the public `architecture.source: timm`
selector.

### Examples

Torchvision:

```yaml
model:
  image_input:
    name: image
    backbone:
      architecture:
        source: torchvision
        name: resnet50
        output_dim: auto
      weights:
        source: library
        name: DEFAULT        # resolved config stores the concrete name

training:
  freeze:
    backbone:
      policy: none
```

timm:

```yaml
model:
  image_input:
    name: image
    backbone:
      architecture:
        source: timm
        name: vit_small_patch16_224
        output_dim: auto
      weights:
        source: library
        name: default

training:
  freeze:
    backbone:
      policy: last_n_blocks_trainable
      n: 2
```

Checkpoint:

```yaml
model:
  image_input:
    name: image
    backbone:
      architecture:
        source: timm
        name: vit_small_patch14_dinov2
        output_dim: auto
      weights:
        source: checkpoint
        uri: s3://bucket/runs/ssl_dino_v2/exports/encoder.pt
        key: encoder_state_dict
        strict: false

training:
  freeze:
    backbone:
      policy: last_n_blocks_trainable
      n: 4
```

`backbone.architecture.output_dim: auto` is the default and should usually
not be changed manually.

### Freeze policies

Freeze policies live under `training.freeze.backbone`, because they control
trainability and optimizer membership rather than model architecture. They
are accepted for SSL and supervised training.

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
`model.image_input.backbone.weights.source: checkpoint` and `weights.uri`
pointing at the SSL encoder export. Freeze policy, embedding adapter, and
heads are configured exactly as for any supervised run.

`weights.strict: false` maps to PyTorch's
`load_state_dict(..., strict=False)` and governs **backbone** keys only.
Heads in the new run come from the new `heads:` block; nothing from the
source checkpoint's head ever appears in the new model.

When `weights.strict: false`, Dojo logs missing and unexpected keys
clearly.

Frozen feature extractor recipe:

```yaml
model:
  image_input:
    name: image
    backbone:
      architecture:
        source: torchvision
        name: resnet50
        output_dim: auto
      weights:
        source: checkpoint
        uri: s3://bucket/runs/ssl_dino_v2/exports/encoder.pt
        key: encoder_state_dict
        strict: false

  embedding_adapter:
    enabled: true
    type: mlp
    hidden_dims: [512]
    output_dim: 256
    activation: gelu
    dropout: 0.1

training:
  freeze:
    backbone:
      policy: all
```

For a true linear-probe evaluation, leave `embedding_adapter` disabled
and let the head's `network: linear` consume the raw backbone embedding.

## Tabular input and implicit concatenation

Tabular features may be useful model inputs (size descriptors, depth,
temperature, etc.). Tabular preprocessing and encoding are configured
under `model.tabular_input`. There is no `model.fusion` config block:
when both image and tabular inputs are enabled, Dojo concatenates their
embeddings implicitly in canonical input order, image first and tabular
second.

```yaml
model:
  image_input:
    name: image         # default input-stream name
    backbone:
      architecture:
        source: torchvision
        name: resnet50
      weights:
        source: library

  tabular_input:
    enabled: true
    name: tabular       # default input-stream name
    columns: [depth_m, temperature_c, salinity_psu]
    imputation:
      default:
        strategy: median              # train-split statistic, frozen at fit time
      per_column:
        salinity_psu:
          strategy: constant
          fill_value: 35.0
      add_missing_indicator: false
    encoder:
      type: mlp
      hidden_dims: [64, 64]
      output_dim: 64

  embedding_adapter:
    enabled: true
    type: mlp
    hidden_dims: [512]
    output_dim: 256
    activation: gelu
    dropout: 0.1
```

`model.tabular_input.name` names the tabular input stream and defaults to
`tabular`. The initial implementation has at most two model inputs:
`model.image_input.name` (`image` by default) and
`model.tabular_input.name` (`tabular` by default). Image input is required
and tabular input is optional; tabular-only model schema is deferred to
P4.13.

If only image input is enabled, the backbone embedding flows directly to
the optional `embedding_adapter`. If image and tabular inputs are enabled,
Dojo concatenates the two embeddings along the feature dimension in fixed
order: image embedding first, tabular embedding second. This concatenation
is identity-like and non-parametric: it learns no weights and has no
authored config. Shared learned capacity after concatenation belongs in
`embedding_adapter`.

Model flow:

```text
image → backbone → image_embedding
tabular features → tabular_encoder → tabular_embedding
image_embedding + tabular_embedding → implicit concat → fused_input_embedding
  ↓ optional embedding_adapter → head_input_embedding
  ↓ head(s)
```

Exported model artifacts must include tabular feature names, ordering,
input-stream names, encodings, normalization statistics, imputation fill
values, and implicit concatenation order (see `10-export.md`).

### Tabular missing values

Tabular feature columns may be missing per sample (sensor dropout,
unresolved joins, ROIs without co-located measurements). Because the
tabular encoder needs a finite numeric tensor, missing feature values are
**imputed**, not dropped. This is distinct from
`data.targets.<t>.missing_policy` (`04-data-and-storage.md`), which governs
missing **labels**: a missing label may drop a sample, but a missing
feature is filled so the sample can still produce a prediction at inference
time.

`model.tabular_input.imputation` configures the fill:

- `default.strategy` — rule applied to every column without an override:
  `mean`, `median`, or `most_frequent` (computed on the train split and
  frozen), or `constant` with an explicit `fill_value`.
- `per_column.<col>` — per-column override of `strategy` / `fill_value`.
- `add_missing_indicator` — when `true`, append one synthetic binary
  feature per configured column marking whether the original value was
  present (`0`) or missing-and-imputed (`1`), letting the model use
  missingness as a signal. The indicator set is fixed by config (all
  configured columns), not by which columns happen to contain nulls in a
  given split, so the tabular input width stays reproducible across
  datasets. Because it widens the tabular encoder input, it is part of the
  model architecture contract (`model_config_hash`) as well as the input
  contract.

Statistic-based fill values are computed on the train split only and
frozen, exactly like normalization mean/std. The frozen fill values,
per-column strategy, categorical encodings, and normalization statistics
are resolved preprocessing state: persisted in the config artifact,
exported with portable models (`10-export.md`), and contributing to
`preprocessing_hash` by value (`06-results-artifacts-and-metadata.md`).

The initial tabular input targets **numeric** features (normalization +
imputation). The categorical-encoding schema (one-hot / learned embedding /
ordinal) is not yet specified; it lands with the P3.6 tabular work
(`13-workplan.md`), so the `encodings` slot in `preprocessing_hash` and
export metadata is a reserved placeholder until then.

### Simple network specs

The same small set of simple network specs appears in several model
sub-blocks. Keep the names consistent, but do not add a `network:` wrapper
outside heads:

- `model.tabular_input.encoder.type`
- `model.embedding_adapter.type`
- `model.heads.<head>.network.type`

Initial values:

| Type | Meaning | Initial locations |
| --- | --- | --- |
| `identity` | No learned module; output is the input feature vector unchanged. | `model.tabular_input.encoder.type` |
| `linear` | One learned affine projection. No hidden layers. For heads, this means the head-specific final projection only. | `model.tabular_input.encoder.type`, `model.embedding_adapter.type`, `model.heads.<head>.network.type` |
| `mlp` | One or more hidden layers before the output projection; supports nonlinear feature interactions. | `model.tabular_input.encoder.type`, `model.embedding_adapter.type`, `model.heads.<head>.network.type` |

For `tabular_input.encoder` and `embedding_adapter`, `linear` and `mlp`
require an explicit `output_dim`. For head networks, the head type
determines the final output shape, so `network` does not set `output_dim`;
it only chooses whether hidden layers exist before the head-specific
projection.

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
multilabel_classification          # reserved; runtime-stubbed in initial implementation
regression
ordinal_classification             # head predicts a discrete ordered bin
distributional_regression          # deferred — see appendix P4.9
count_regression                   # deferred — see appendix P4.9
```

`multilabel_classification` is reserved for true multi-hot multilabel
problems. The schema slot is present, but runtime support is deferred and
raises `NotImplementedError` in the initial implementation (see
`appendix-deferred-features.md` P4.2). The old `multilabel` module (which
was actually multi-head multiclass) is not ported. Multi-head multiclass
uses one `multiclass_classification` head per target. The old module is
preserved under `dojo_deprecated` for reference.

`distributional_regression` and `count_regression` are **deferred
schema-backlog** head types in the initial implementation: the initial
schema rejects them with a clear validation error (they are not accepted
and then stubbed at runtime). Neither has a result record type yet
(`06-results-artifacts-and-metadata.md` defines no `distributional_output`
/ `count_output`), and their dedicated losses (`gaussian_nll`,
`negative_binomial_nll`, `poisson_nll`) are deferred with them. Plain
`regression` is the only functional regression head type. See
`appendix-deferred-features.md` P4.9.

### Required head fields and network defaults

Authored configs must provide:

```text
type               # one of the head types above
target             # logical data target key from data.targets
```

`network` is optional in authored configs. If omitted, config
compilation injects the head type's default network spec. Resolved
configs, saved config artifacts, and runtime objects always include
`network`.

Type-specific required fields:

```text
multiclass_classification / binary_classification / multilabel_classification:
  num_classes

regression:
  output_dim     (default: 1)

ordinal_classification:
  num_classes
  ordinal        # optional; encoding / decoding, defaulted (see "Ordinal encoding and decoding")

distributional_regression:
  distribution   (e.g. gaussian, negative_binomial)

count_regression:
  output_dim     (default: 1)
```

Initial supported head network types:

| Head type | Allowed `network.type` | Default `network.type` |
| --- | --- | --- |
| `multiclass_classification` | `linear`, `mlp` | `linear` |
| `binary_classification` | `linear`, `mlp` | `linear` |
| `multilabel_classification` (runtime-stubbed, P4.2) | `linear`, `mlp` | `linear` |
| `regression` | `linear`, `mlp` | `linear` |
| `ordinal_classification` | `linear`, `mlp` | `linear` |
| `distributional_regression` (deferred, P4.9) | `linear`, `mlp` | `linear` |
| `count_regression` (deferred, P4.9) | `linear`, `mlp` | `linear` |

`network.type: linear` means there are no hidden layers between the
head input embedding and the final head-specific projection. The head
type still determines output shape and interpretation: classification
heads emit logits, regression heads emit continuous values,
ordinal heads emit ordinal logits / bins, distributional regression
heads emit distribution parameters, and count regression heads emit
count/rate parameters.

`network.type: mlp` adds "MultiLayer Perceptron" hidden layers before the same head-specific
final projection. Initial MLP head-network config:

```yaml
network:
  type: mlp
  hidden_dims: [512]      # required; one or more hidden layer widths
  activation: gelu        # default: gelu
  dropout: 0.0            # default: 0.0
```

Head MLP networks do not set `output_dim`; the head type and its
type-specific fields determine the final projection size.

Pydantic validation ensures the resolved `target` exists in
`data.targets` and has a compatible dtype.

### Ordinal naming and result columns

- Head type: `ordinal_classification` (the head predicts a discrete
  ordered bin, not a continuous value).
- Supported ordinal losses: `coral`, `corn`, `ordinal_cross_entropy`.
- Result `record_type` values: `ordinal_output` for native ordinal heads;
  `ordinal_probe_prediction` for ordinal probes.
- `ordinal_logits` (the `num_classes - 1` cumulative threshold logits) is
  populated only for cumulative encodings (`coral`, `corn`) and is null for
  `ordinal_cross_entropy`. `probabilities` (per-bin) is always populated:
  by differencing decoded cumulative probabilities for `coral` / `corn`, or
  directly by softmax over per-bin logits for `ordinal_cross_entropy`.

See `06-results-artifacts-and-metadata.md`.

### Ordinal encoding and decoding

Order semantics are carried by the head (`type: ordinal_classification`,
`num_classes`, and the ordered class labels). How those ordered classes
are turned into output units, and how outputs map back to a class, is
configured on the head under `ordinal`:

```yaml
model:
  heads:
    size_category:
      type: ordinal_classification
      target: size_category
      num_classes: 4
      ordinal:
        encoding: coral        # coral | corn | ordinal_cross_entropy
        decoding: threshold    # threshold | expected_rank | argmax
      network:
        type: linear
```

- `encoding` determines the output tensor. `coral` and `corn` emit
  `num_classes - 1` cumulative threshold logits ("is the class beyond bin
  k?"); `ordinal_cross_entropy` emits `num_classes` per-bin logits.
- `decoding` maps outputs to a predicted class: `threshold` counts how
  many cumulative thresholds clear 0.5 (CORAL / CORN); `expected_rank`
  takes the probability-weighted rank; `argmax` takes the most probable
  bin (per-bin form).

`ordinal` is optional in authored configs. Config compilation injects a
default: `encoding` follows the configured ordinal loss when an objective
is present (else `coral`), and `decoding` defaults per encoding
(`coral` / `corn` → `threshold`, `ordinal_cross_entropy` → `argmax`).
Resolved configs, saved config artifacts, and runtime objects always carry
`ordinal` explicitly.

Encoding / decoding live on the head, not on the loss, because they define
output structure and interpretation that must survive without an
`objectives` block — exported portable models and cached-result ensembles
still have to interpret `ordinal_logits` at inference time. They therefore
contribute to `target_schema_hash` and are written into export metadata
(`10-export.md`). When an objective is present, config validation checks
that its ordinal loss is consistent with the head's `ordinal.encoding`;
this is a cross-field check, not the head owning loss config.

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
      num_classes: 42
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
      transform: log1p_standardize
      missing_policy: drop_sample

model:
  tabular_input:
    enabled: true
    name: tabular
    columns: [depth_m, temperature_c, salinity_psu]
    encoder:
      type: mlp
      hidden_dims: [64, 64]
      output_dim: 64
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
      ordinal:
        encoding: ordinal_cross_entropy
        decoding: argmax
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
    metrics: [mae, rmse, r2]
    weight: 0.25
```

### Compatible losses

Classification: `cross_entropy`, `weighted_cross_entropy`,
`class_balanced_effective_number`, `focal`,
`label_smoothing_cross_entropy`.

Regression: `mse`, `mae`, `huber`, `smooth_l1`, `quantile`.
(`gaussian_nll`, `poisson_nll`, and `negative_binomial_nll` are deferred
with the `distributional_regression` / `count_regression` heads — see
`appendix-deferred-features.md` P4.9.)

Ordinal: `coral`, `corn`, `ordinal_cross_entropy`.

Count-dependent losses (`weighted_cross_entropy`,
`class_balanced_effective_number`) derive their per-class weights from the
frozen train-split class counts in the dataset stats cache (`dojo inspect
dataset --class-counts`, `04-data-and-storage.md`). The resolved weights are
frozen into config, not recomputed per run.

### Target transforms

Common transforms (regression / ordinal heads): `identity`, `standardize`,
`log1p`, `log1p_standardize`, `power` / Box-Cox / Yeo-Johnson. See
`06-results-artifacts-and-metadata.md` for the external vs. internal
column convention.

Target transforms are configured once, on the data target
(`data.targets.<t>.transform`) — not on the objective. The functional form
is authored; config compilation freezes any fitted statistics (standardize
mean / std, Box-Cox λ) into the resolved transform — the same
authored-vs-resolved pattern used for normalization and imputation stats.
The inverse is applied to predictions at inference to recover external
units, so the resolved transform must survive without an `objectives`
block: it is carried in export metadata and contributes to
`target_schema_hash` by value.

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
- at least one objective is enabled for supervised training;
- for classification / ordinal heads, the head's `num_classes` matches the
  resolved class map for the referenced `data.targets.<target>` (the
  ordered labels from `class_names`, or the `--class-counts` class count);
  a mismatch is a config error.

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
        training_config: TrainingConfig,
        objectives_config: ObjectiveCollectionConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_supervised_model(
            model_config,
            freeze_config=training_config.freeze,
        )
        self.objectives = build_objectives(objectives_config)
        self.metrics = build_metrics(objectives_config)
```

Constructor arguments are serializable Pydantic configs so Lightning
hyperparameter checkpoints stay clean.

### Model composition

```python
backbone = build_backbone(cfg.model.image_input.backbone.architecture)
initialize_backbone_weights(backbone, cfg.model.image_input.backbone.weights)
apply_freeze_policy(backbone, cfg.training.freeze.backbone)

tabular_encoder = (
    build_tabular_encoder(cfg.model.tabular_input)
    if cfg.model.tabular_input.enabled
    else None
)

model_input_order = [cfg.model.image_input.name]
model_embedding_dim = backbone.output_dim
if tabular_encoder:
    model_input_order.append(cfg.model.tabular_input.name)
    model_embedding_dim += tabular_encoder.output_dim

# SupervisedModel concatenates enabled input embeddings in model_input_order
# before applying the optional embedding adapter.
embedding_adapter = (
    build_embedding_adapter(
        cfg.model.embedding_adapter,
        input_dim=model_embedding_dim,
    )
    if cfg.model.embedding_adapter.enabled
    else None
)

heads = build_heads(
    cfg.model.heads,
    input_dim=embedding_adapter.output_dim if embedding_adapter else model_embedding_dim,
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

### Inference-contract embedding

The task module's `on_save_checkpoint` hook writes a portable **inference
contract** into `checkpoint["dojo_inference_contract"]`: the buildable
`model_config`, the resolved `inference_pipeline` with frozen preprocessing
stats, per-head class maps, the target schema with frozen target transforms,
the resolved `objective_summary` used for holdout scoring, and the four
compatibility hashes. It is kept **out** of Lightning
`hyper_parameters` (so the constructor configs stay clean) and makes a
`.ckpt` as self-describing as an export: `dojo infer` / `dojo eval` rebuild
the model and its input pipeline from the artifact alone, with no dependency
on the producing run's `resolved.yaml`. See
`06-results-artifacts-and-metadata.md`.

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
