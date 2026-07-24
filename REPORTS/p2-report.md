# P2 Training Findings

## 2026-07-23 15:43 EDT

Analyzed completed training artifacts under <code>runs/&#8203;p2_*</code>. This report groups
experiments by the source layout under `configs/experiment/p2/`, so the report
order mirrors the experiment config order.

Most P2 experiments are variations on the `00_baseline/nes_baseline.yaml`
configuration. That baseline uses the NES plankton Hugging Face-style
parquet-image dataset at `./datasets/nes-hf/data`, species-only multiclass
classification with 155 classes, torchvision EfficientNet-B0 with library
`DEFAULT` weights, 224x224 RGB resize, horizontal and vertical train-only
flips, ImageNet normalization, AdamW with `lr=0.0003` and
`weight_decay=0.01`, batch size 64, no backbone freezing, early stopping on
`val/loss`, and checkpoint selection on `val/species/f1_macro`.

The primary comparison value is the best validation macro-F1 observed during
training. For species-only runs this is `val/species/f1_macro`; for Type-only
runs it is `val/Type/f1_macro`; for multihead runs it is the mean of available
`val/<head>/f1_macro` columns at each epoch.

## Experiment Overview

| Config group | Dojo feature exercised | Experiments |
| --- | --- | --- |
| [`00_baseline`](#00_baseline-results) | Full default P2 supervised pipeline; `nes_simple` removes augmentation as a control | `nes_baseline`, `nes_simple` |
| [`01_architecture`](#01_architecture-results) | `backbone.architecture.name` swept with provider held to torchvision | `EfficientNet-B0`, `ConvNeXt Tiny`, `ViT-B/16` |
| [`02_provider`](#02_provider-results) | `backbone.architecture.source: timm` with `output_dim: auto`; architectures mirror group 01 | `TIMM EfficientNet-B0`, `TIMM ConvNeXt Tiny`, `TIMM ViT Base` |
| [`03_pretraining-and-normalization`](#03_pretraining-and-normalization-results) | `backbone.weights.source` (library vs none) × `normalize` statistics (ImageNet vs NES dataset) | `pretrained/ImageNet norm`, `pretrained/dataset norm`, `scratch/ImageNet norm`, `scratch/dataset norm` |
| [`04_imbalance`](#04_imbalance-results) | `objectives.species.loss.type` [focal, label-smoothed CE, inverse-frequency weighted CE] and `training.sampler.type` [inverse-frequency weighted] | `focal loss`, `label smoothing`, `weighted CE`, `weighted sampler` |
| [`05_input-fit`](#05_input-fit-results) | `name: letterbox` (pad-to-square) and `name: aspect_bucket` + `sampler.type: batch_aspect_buckets` | `aspect buckets`, `letterbox` |
| [`06_augmentation`](#06_augmentation-results) | Transform pipeline ablation: `name: foreground_crop` prepended; flips-only baseline; no augmentation | `foreground crop`, `flips`, `no augmentation` |
| [`07a_transfer-miniset`](#07a_transfer-miniset-results) | `backbone.weights.source: checkpoint` (strict=false) onto `plankton-miniset`; varies `freeze.backbone.policy` × `embedding_adapter.type` | `frozen`, `frozen linear adapter`, `frozen MLP adapter`, `unfrozen` |
| [`07b_transfer-denticle-type`](#07b_transfer-denticle-type-results) | Same checkpoint transfer as 07a onto `ichthyoliths_denticle_type` (cross-domain); single `Type` head | `frozen/unfrozen backbone with linear or MLP adapter` |
| [`08_multihead`](#08_multihead-results) | 46 parallel heads with per-head sparse-label masking; `data.split_column`; `objectives.Type` addition and `weight` up-weighting; single-head `type_only` control | `split heads`, `commons split`, `Type head`, `weighted Type`, `Type-only` |

All results below come from models trained with early stopping on validation
loss with patience 10 epochs. The selected model for each run is the checkpoint
with the best validation macro-F1 score. Metric columns report values from that
selected checkpoint; `Epoch` is shown as `n/m`, where `n` is the selected epoch
and `m` is the final logged epoch.

## 00_baseline Results

The baseline group establishes the reference configuration and its minimal control.
`nes_baseline` exercises the full default P2 supervised pipeline end-to-end: NES HF
parquet dataset, torchvision EfficientNet-B0 with ImageNet pretrained weights, a
512-dim linear embedding adapter, a single 155-class species head, plain CE loss,
train-only horizontal and vertical flip augmentation, fixed ImageNet normalization,
AdamW (lr=0.0003, weight_decay=0.01), batch 64, and early stopping / checkpoint
selection on `val/loss` and `val/species/f1_macro` respectively.
`nes_simple` removes the flip steps so the transform pipeline is resize + normalize
only, isolating the flip contribution from all other factors.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_nes_baseline` | <code>runs/&#8203;p2_nes_baseline/&#8203;2026-07-21_17-47-48_&#8203;venomous-panda</code> | 15/17 | 0.9222 | 0.9439 |
| `p2_nes_simple` | <code>runs/&#8203;p2_nes_simple/&#8203;2026-07-21_18-18-04_&#8203;screeching-chimera</code> | 8/13 | 0.9180 | 0.9374 |

The no-augmentation control trails the baseline slightly, which is consistent
with the horizontal and vertical flips being useful but not dominant.

## 01_architecture Results

This group exercises `model.image_input.backbone.architecture.name` with the provider
held to torchvision (the default). Each experiment overrides only the architecture name
(`efficientnet_b0`, `convnext_tiny`, or `vit_b_16`) and inherits all other config from
the baseline. Comparing against group 02 isolates provider from architecture effects.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_arch_tv_convnext_tiny` | <code>runs/&#8203;p2_arch_tv_convnext_tiny/&#8203;2026-07-22_20-09-50_&#8203;rigorous-bumblebee</code> | 32/32 | 0.9296 | 0.9480 |
| `p2_arch_tv_effb0` | <code>runs/&#8203;p2_arch_tv_effb0/&#8203;2026-07-21_18-44-05_&#8203;crouching-crayfish</code> | 15/17 | 0.9222 | 0.9439 |
| `p2_arch_tv_vit_b16` | <code>runs/&#8203;p2_arch_tv_vit_b16/&#8203;2026-07-21_19-13-31_&#8203;khaki-wapiti</code> | 21/22 | 0.8895 | 0.9225 |

Torchvision ConvNeXt Tiny was the best architecture in this group.
EfficientNet-B0 remained competitive. ViT-B/16 underperformed the convolutional
baselines in this run set.

## 02_provider Results

This group exercises the TIMM provider path: `backbone.architecture.source: timm`,
`weights.source: library`, and `output_dim: auto` (dojo queries the model for its
output feature dimension at construction time). Architecture names match group 01
(`efficientnet_b0`, `convnext_tiny`, `vit_base_patch16_224`) so the provider/API
effect can be read directly off the cross-group difference.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_provider_timm_effb0` | <code>runs/&#8203;p2_provider_timm_effb0/&#8203;2026-07-21_21-15-42_&#8203;golden-pudu</code> | 19/19 | 0.9217 | 0.9384 |
| `p2_provider_timm_convnext_tiny` | <code>runs/&#8203;p2_provider_timm_convnext_tiny/&#8203;2026-07-21_20-31-04_&#8203;cherubic-phoenix</code> | 15/21 | 0.9155 | 0.9360 |
| `p2_provider_timm_vit_base` | <code>runs/&#8203;p2_provider_timm_vit_base/&#8203;2026-07-21_21-50-27_&#8203;helpful-copperhead</code> | 34/36 | 0.8790 | 0.9158 |

TIMM EfficientNet-B0 was close to the torchvision EfficientNet-B0 result. The
TIMM ConvNeXt and ViT variants did not improve over the corresponding
torchvision-side architecture experiments.

## 03_pretraining-and-normalization Results

This group crosses two binary knobs: `backbone.weights.source` (`library` = ImageNet
pretrained vs `none` = random scratch init) and the `normalize` step mean/std
(ImageNet statistics [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] vs NES dataset
statistics [0.643, 0.643, 0.643] / [0.191, 0.191, 0.191] — grayscale, replicated
across channels). All other config is the baseline.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_pretrained_dataset_norm` | <code>runs/&#8203;p2_pretrained_dataset_norm/&#8203;2026-07-21_23-36-27_&#8203;elegant-dogfish</code> | 14/16 | 0.9227 | 0.9446 |
| `p2_pretrained_imagenet_norm` | <code>runs/&#8203;p2_pretrained_imagenet_norm/&#8203;2026-07-22_00-06-32_&#8203;vehement-zebra</code> | 15/17 | 0.9222 | 0.9439 |
| `p2_scratch_dataset_norm` | <code>runs/&#8203;p2_scratch_dataset_norm/&#8203;2026-07-22_00-38-14_&#8203;camouflaged-wolf</code> | 34/34 | 0.9045 | 0.9325 |
| `p2_scratch_imagenet_norm` | <code>runs/&#8203;p2_scratch_imagenet_norm/&#8203;2026-07-22_01-33-56_&#8203;onyx-chachalaca</code> | 24/25 | 0.8973 | 0.9260 |

Pretraining mattered more than normalization choice. Dataset normalization
slightly edged ImageNet normalization when pretrained, but the margin was small.
Scratch training was consistently worse.

## 04_imbalance Results

This group exercises the dojo's long-tail surface. Loss variants set
`objectives.species.loss` to: `focal_loss` with `params.gamma: 2.0`;
`cross_entropy` with `params.label_smoothing: 0.1`; `weighted_cross_entropy` with
`params.scheme: inverse_frequency`. The sampler variant sets
`training.sampler.type: weighted` with `class_weight_scheme: inverse_frequency`
and `head: species`.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_imbalance_label_smoothing` | <code>runs/&#8203;p2_imbalance_label_smoothing/&#8203;2026-07-22_03-21-16_&#8203;gay-turaco</code> | 35/37 | 0.9402 | 0.9549 |
| `p2_imbalance_weighted_ce` | <code>runs/&#8203;p2_imbalance_weighted_ce/&#8203;2026-07-22_04-21-50_&#8203;sceptical-jaguar</code> | 26/26 | 0.9287 | 0.9455 |
| `p2_imbalance_weighted_sampler` | <code>runs/&#8203;p2_imbalance_weighted_sampler/&#8203;2026-07-22_05-06-23_&#8203;proficient-boar</code> | 18/19 | 0.9191 | 0.9316 |
| `p2_imbalance_focal` | <code>runs/&#8203;p2_imbalance_focal/&#8203;2026-07-22_02-52-11_&#8203;infrared-beetle</code> | 5/15 | 0.9158 | 0.9382 |

Label smoothing was the strongest full NES species run in the batch. Weighted
cross-entropy was the best ordinary imbalance treatment after label smoothing.

## 05_input-fit Results

This group exercises the two aspect-ratio-preserving input-fit transforms. `letterbox`
replaces the resize step with `name: letterbox, canvas_size: [224, 224]`, which
pads the shorter axis to square while keeping the original aspect ratio.
`aspect_buckets` uses `name: aspect_bucket` with four named buckets (tall ≤0.8,
square 0.8–1.25, wide 1.25–2.5, ultrawide ≥2.5) each with its own canvas size,
combined with `training.sampler.type: batch_aspect_buckets` so each mini-batch
draws from a single bucket (reducing within-batch padding to zero). Batch size is
reduced to 48 from the 64 baseline to accommodate the variable canvas dimensions.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_input_letterbox` | <code>runs/&#8203;p2_input_letterbox/&#8203;2026-07-22_06-32-33_&#8203;massive-turtle</code> | 9/17 | 0.9241 | 0.9414 |
| `p2_input_aspect_buckets` | <code>runs/&#8203;p2_input_aspect_buckets/&#8203;2026-07-22_05-40-59_&#8203;pragmatic-hare</code> | 18/18 | 0.8700 | 0.9034 |

Letterbox was modestly above the EfficientNet baseline. Aspect buckets were a
clear regression in this run set.

## 06_augmentation Results

This group is a transform pipeline ablation. `aug_flips` inherits the baseline
unchanged (H+V flip only, named for the ablation context). `aug_fgcrop` prepends
`name: foreground_crop, threshold: 0.0, train_only: true` before the resize step —
this crops the input image to the bounding box of pixels above the background
threshold before resizing, exercising the dojo's background-removal transform.
`aug_none` omits both flip steps, reducing to resize + normalize only (equivalent
to `nes_simple` in group 00).

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_aug_fgcrop` | <code>runs/&#8203;p2_aug_fgcrop/&#8203;2026-07-22_07-02-56_&#8203;marvellous-binturong</code> | 15/17 | 0.9222 | 0.9439 |
| `p2_aug_flips` | <code>runs/&#8203;p2_aug_flips/&#8203;2026-07-22_07-34-24_&#8203;messy-bug</code> | 15/17 | 0.9222 | 0.9439 |
| `p2_aug_none` | <code>runs/&#8203;p2_aug_none/&#8203;2026-07-22_08-05-55_&#8203;tangible-starfish</code> | 8/13 | 0.9180 | 0.9374 |

Foreground crop and flips matched the baseline nearly exactly. No augmentation
was slightly lower.

**Note:** `p2_aug_fgcrop` is misconfigured for this dataset. The `foreground_crop`
step uses `threshold: 0.0`, which crops pixels with max channel value exactly
equal to 0.0. NES plankton images have a light background (observed min pixel
≈ 0.16–0.36 across the dataset); across 30,000 training images, zero had an
all-zero border row or column. The crop was a no-op on every image, making
`p2_aug_fgcrop` functionally identical to `p2_nes_baseline` — confirmed by
byte-for-byte identical metrics CSVs and zero per-sample prediction disagreements
in the parquet output. To exercise foreground cropping on this dataset, the
threshold should be set to a value that reflects the actual background level
(e.g. `threshold: 0.15`), or the logic should be inverted to crop toward pixels
below a high-value ceiling if the intent is to isolate organisms from a light
background.

## 07a_transfer-miniset Results

This group exercises `backbone.weights.source: checkpoint` pointing to a saved NES
`.ckpt` file (`strict: false` allows the head and adapter weights to be skipped).
The retarget dataset is `plankton-miniset` (30 species classes, in-domain), so this
is same-domain transfer. The four experiments vary two orthogonal knobs:
`freeze.backbone.policy` (`frozen` = backbone weights held fixed, `none` = full
fine-tuning) and `embedding_adapter` (`disabled` = no adapter between backbone
and head, `type: linear` = 512-dim projection, `type: mlp` = two-layer 512-dim
MLP with GELU activation and 0.1 dropout). These runs are not directly comparable
to full NES runs because they use a smaller 30-class dataset. For groups with
multiple successful versions, this table keeps the later run whose backbone
comes from the later baseline model.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_transfer_miniset_frozen` | <code>runs/&#8203;p2_transfer_miniset_frozen/&#8203;2026-07-22_21-52-44_&#8203;unbiased-toucan</code> | 17/97 | 0.9864 | 0.9881 |
| `p2_transfer_miniset_frozen_linear` | <code>runs/&#8203;p2_transfer_miniset_frozen_linear/&#8203;2026-07-22_21-28-44_&#8203;fierce-ocelot</code> | 6/59 | 0.9864 | 0.9881 |
| `p2_transfer_miniset_frozen_mlp` | <code>runs/&#8203;p2_transfer_miniset_frozen_mlp/&#8203;2026-07-22_21-42-59_&#8203;fiery-fulmar</code> | 12/31 | 0.9864 | 0.9881 |
| `p2_transfer_miniset_unfrozen` | <code>runs/&#8203;p2_transfer_miniset_unfrozen/&#8203;2026-07-22_22-18-26_&#8203;astonishing-shark</code> | 18/34 | 0.9831 | 0.9821 |

All four transfer variants performed strongly within the miniset context.

## 07b_transfer-denticle-type Results

This group uses the same `backbone.weights.source: checkpoint` mechanism as 07a but
targets the `ichthyoliths_denticle_type` dataset — a cross-domain shift from
plankton imagery to ichthyolith ROI images. The head is a single `Type` multiclass
classifier with 48 configured classes, `checkpointing.monitor: val/Type/f1_macro`,
and `batch_size: 16` (smaller due to dataset size). All four freeze × adapter
combinations from 07a are replicated here to separate the backbone freeze and
adapter capacity effects from the domain-shift effect.

| Experiment | Run | Epoch | F1 Macro | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `p2_transfer_type_unfrozen_linear` | <code>runs/&#8203;p2_transfer_type_unfrozen_linear/&#8203;2026-07-22_21-19-14_&#8203;jovial-baboon</code> | 12/12 | 0.1475 | 0.2778 |
| `p2_transfer_type_unfrozen_mlp` | <code>runs/&#8203;p2_transfer_type_unfrozen_mlp/&#8203;2026-07-22_21-20-46_&#8203;violet-ammonite</code> | 10/15 | 0.1110 | 0.2222 |
| `p2_transfer_type_frozen_linear` | <code>runs/&#8203;p2_transfer_type_frozen_linear/&#8203;2026-07-22_21-16-16_&#8203;gracious-chachalaca</code> | 11/14 | 0.1027 | 0.1944 |
| `p2_transfer_type_frozen_mlp` | <code>runs/&#8203;p2_transfer_type_frozen_mlp/&#8203;2026-07-22_21-17-46_&#8203;spicy-hoatzin</code> | 13/13 | 0.0747 | 0.1944 |

Unfrozen transfer did best within this group, but all Type-transfer scores were
low.

## 08_multihead Results

This group exercises the P2 multi-head classification surface using the ichthyolith
denticle morphology dataset. The base config (`_base.yaml`) defines 46 named heads
(A1–O10) each as a `multiclass_classification` head with its own CE objective and
equal loss weight; the dojo masks each head's loss for rows where the corresponding
target column is blank, enabling sparse-label training. The backbone is timm
EfficientNet-B1 (ImageNet pretrained) resized to 240×240 with a 512-dim linear
adapter. `multihead_split` uses the default `split` column; `multihead_split_commons`
sets `data.split_column: split-commons` to use a commons-weighted split. `multihead_type`
adds a 47th `Type` head by extending `model.heads` and `objectives.Type` at
weight=1.0. `multihead_type_weighted` raises `objectives.Type.weight: 3.0` to
up-weight the Type objective relative to the morphology heads. `type_only` is a
single-head control: it uses `ichthyoliths_denticle_type` (rows with blank Type are
dropped rather than masked), timm EfficientNet-B1, and `checkpointing.monitor:
val/Type/f1_macro`, providing an upper-bound reference for the Type task in isolation.

For split-head runs, `F1 Macro` is the mean across split-head macro-F1 values.
For Type-inclusive multihead runs, it is the mean across split-head and Type
macro-F1 values. For Type-only, it is Type macro-F1.

| Experiment | Run | Epoch | F1 Macro |
| --- | --- | ---: | ---: |
| `p2_multihead_split_commons` | <code>runs/&#8203;p2_multihead_split_commons/&#8203;2026-07-22_09-19-01_&#8203;hospitable-tiger</code> | 11/16 | 0.5146 |
| `p2_multihead_split` | <code>runs/&#8203;p2_multihead_split/&#8203;2026-07-22_09-21-11_&#8203;noisy-chimpanzee</code> | 14/18 | 0.4075 |
| `p2_multihead_type` | <code>runs/&#8203;p2_multihead_type/&#8203;2026-07-22_21-24-51_&#8203;rare-ferret</code> | 15/18 | 0.3850 |
| `p2_multihead_type_weighted` | <code>runs/&#8203;p2_multihead_type_weighted/&#8203;2026-07-22_21-22-23_&#8203;accomplished-squid</code> | 18/18 | 0.3838 |
| `p2_multihead_type_only` | <code>runs/&#8203;p2_multihead_type_only/&#8203;2026-07-22_21-27-05_&#8203;dancing-kiwi</code> | 11/15 | 0.1230 |

The Type target has 48 configured classes. The manifest contains only 36
validation examples with non-empty Type labels across 24 observed Type classes.
This makes Type macro-F1 difficult, sparse, and noisy. The low Type scores are
still meaningful as a sign that the current setup is not learning the Type
target reliably, but the runs do exercise the intended multihead and
Type-training features.

## Conclusions

- `p2_imbalance_label_smoothing` is the strongest completed full NES species
  run in this batch.
- ConvNeXt Tiny and label smoothing are the most promising full NES variations
  to combine in a follow-up run.
- EfficientNet-B0 remains a useful stable baseline because it is competitive
  and cheaper than the larger alternatives.
- ViT variants and aspect buckets did not look promising in these runs.
- Type and multihead work should be evaluated with a larger stratified
  validation set or grouped cross-validation before drawing model-quality
  conclusions.

These experiments should be read primarily as indicative exercises of P2 Dojo
training features, not as a thorough attempt to produce the best possible model.
