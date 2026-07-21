# P2 Training Plans

Experiment plan for the P2 training campaign. Two dataset families:

- **NES plankton** (`data=nes-hf`) — 155-species single-head classification on
  the HuggingFace NES-plankton-classifier-2022 parquet shards under
  `./datasets/nes-hf/data`. Splits from filename (`train-*` / `validation-*`).
- **Ichthyolith denticles** (`data=ichthyoliths_denticle_*`) — small (217-row)
  locally annotated ROI set under
  `./datasets/ichthyoliths/local_annotated_denticle_rois_split.csv`, used for
  transfer and multi-head morphology experiments. Two hand-built splits:
  `split` (coverage-optimized: forces ≥1 of every learnable A1–O10 value into
  val) and `split-commons` (proportional: commons ~20% val, rares → train).

All experiment configs live under `configs/experiment/p2/<NN_group>/` and are
selected with `experiment=p2/<NN_group>/<name>`. Each group's configs inherit a
shared base via the Hydra defaults list so only the varied axis differs.

Normalization constants:

- ImageNet: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` — the
  default everywhere, since the baseline uses ImageNet-pretrained weights.
- NES (dataset-derived, via `dojo inspect dataset --normalization` over the NES
  train split): mean `[0.6425, 0.6425, 0.6425]`, std `[0.1915, 0.1915, 0.1915]`
  — used **only** in `03_pretraining-and-normalization`, which is where the
  normalization choice is deliberately examined.

## Groups

### 00_baseline — anchor

`nes_baseline` is the reference all other NES configs inherit: torchvision
EfficientNet-B0 (ImageNet-pretrained), 224×224 resize, horizontal + vertical
flip, **ImageNet** normalization, linear embedding adapter (512), AdamW, plain
cross-entropy, `val/species/f1_macro` monitor. `nes_simple` strips augmentation
(resize + normalize only) as a minimal control.

### 01_architecture — backbone family (torchvision)

Provider held to **torchvision**: EfficientNet-B0 (`nes_effb0`) vs ConvNeXt-Tiny
(`nes_convnext_tiny`) vs ViT-B/16 (`nes_vit_b16`, `vit_b_16` — torchvision's
smallest vision transformer). Everything else held to the baseline; ViT keeps
the 224×224 resize (fixed patch grid). Compare each entry against the matching
architecture in `02_provider` to read off the provider effect.

> Torchvision vision transformers expose a `.heads` classifier (not the
> `.classifier` sequential of the CNN family); the backbone builder strips both,
> returning the pooled class-token embedding.

### 02_provider — same architectures via timm

Provider held to **timm**: EfficientNet-B0, ConvNeXt-Tiny, and ViT-Base/16
(`vit_base_patch16_224`, matching torchvision's `vit_b_16`). Pairs one-to-one
with `01_architecture` so `01/x` vs `02/x` isolates timm-vs-torchvision weight
and preprocessing differences.

### 03_pretraining-and-normalization — 2×2 matrix

Crosses initialization × normalization, the only place dataset-derived
normalization is used:

- `pretrained_imagenet-norm` — ImageNet weights + ImageNet norm (= baseline).
- `pretrained_dataset-norm` — ImageNet weights + NES norm.
- `scratch_imagenet-norm` — `weights.source: none` + ImageNet norm (30 epochs).
- `scratch_dataset-norm` — `weights.source: none` + NES norm (30 epochs).

Tests both the value of ImageNet initialization and whether normalization
statistics should follow the pretraining corpus or the target dataset.

### 04_imbalance — long-tail handling

Baseline already covers plain CE, so this group holds the backbone fixed and
varies the rebalancing method:

- `nes_label_smoothing` — cross-entropy with `label_smoothing: 0.1`.
- `nes_focal` — `focal_loss` (gamma 2.0).
- `nes_weighted_ce` — inverse-frequency `weighted_cross_entropy`.
- `nes_class_balanced_sampler` — `class_balanced` sampler.
- `nes_weighted_sampler` — inverse-frequency `weighted` sampler.

Isolates loss-based (smoothing/focal/weighted-CE) vs sampler-based rebalancing.

### 05_input-fit — canvas strategy

`nes_letterbox` (pad to square, preserve aspect) vs `nes_aspect_buckets`
(bucketed variable canvases + `batch_aspect_buckets` sampler). Plankton ROIs
vary widely in aspect ratio.

### 06_augmentation — augmentation ablation

`nes_noaug` (resize + normalize only) vs `nes_flips` (H+V flip) vs `nes_fgcrop`
(foreground crop + flips). Measures augmentation payoff on NES.

### 07a_transfer-miniset — freeze policy on in-domain transfer

Transfer a NES-trained Dojo checkpoint onto `plankton-miniset` (same domain,
30 classes). Four configs: `unfrozen`, `frozen`, `frozen_linear` (frozen
backbone + linear adapter head), `frozen_mlp` (frozen backbone + MLP adapter).
Exercises the `freeze.backbone.policy: frozen` support and the embedding adapter
as the trainable transfer surface.

> **Checkpoint dependency.** These configs point
> `backbone.weights.source: checkpoint` at a NES run checkpoint. Set the `uri`
> to a concrete `runs/.../checkpoints/*.ckpt` before running (placeholder left
> in each file).

### 07b_transfer-denticle-type — cross-domain transfer

Transfer the NES-trained backbone onto the ichthyolith denticle **Type** head
(strong domain shift: plankton → microfossil ROIs). Type only, `drop_sample`
for blanks. Four configs crossing {unfrozen, frozen} × {linear, mlp} adapter.
Same checkpoint dependency as `07a`.

### 08_multihead — morphology multi-head

A1–O10 morphological heads (46) with `mask_objective` so sparse blank cells
mask the per-head loss instead of dropping rows. timm EfficientNet-B1, 240×240,
ImageNet norm. Configs:

- `multihead_split` — 46 heads, `split` (coverage) split.
- `multihead_split_commons` — same, `split-commons` split.
- `multihead_type` — 46 heads + a coarse `Type` head.
- `multihead_type_weighted` — as above, Type objective `weight: 3.0`.
- `type_only` — single `Type` head (upper-bound control for the Type task).

`val/loss` monitor (min) throughout, since no single head is the objective.

## Running

```bash
# NES baseline
dojo train experiment=p2/00_baseline/nes_baseline

# architecture sweep (torchvision) and its timm counterpart
dojo train experiment=p2/01_architecture/nes_vit_b16
dojo train experiment=p2/02_provider/nes_vit_base

# pretraining × normalization matrix
dojo train experiment=p2/03_pretraining-and-normalization/scratch_dataset-norm

# multihead with the commons split
dojo train experiment=p2/08_multihead/multihead_split_commons
```

Transfer groups (`07a`, `07b`) require a trained NES checkpoint URI to be filled
in first. Inspect any config before running with:

```bash
dojo inspect config --config configs/experiment/p2/00_baseline/nes_baseline.yaml
```

## Utilities Demo

Beyond `dojo train`, the P1/P2 CLI ships inspection, inference, and eval
utilities. They take the same Hydra selectors/overrides as `train`
(`experiment=…`, `data=…`, `--config PATH`), so any command below can be pointed
at a p2 config.

### 1. Bootstrap a local project — `dojo init`

Materialize the packaged config defaults into an editable `./configs` tree.

```bash
dojo init --supervised --data --dry-run   # preview actions
dojo init --all                           # copy every default + example fixture
```

### 2. Compose & inspect a config — `dojo inspect config`

Resolve the full config, see output-path resolution and the P1 result contract;
emit JSON, or diff two configs by hash-source section.

```bash
dojo inspect config --config configs/experiment/p2/00_baseline/nes_baseline.yaml --format json
dojo inspect config-compare --config-a <a>.yaml --config-b <b>.yaml
```

`config-compare` is the "why did these two runs differ" tool — it compares by
the sections that feed the dataset/model/inference hashes.

### 3. Dataset stats & normalization — `dojo inspect dataset`

How the NES normalization constants above were derived, and how the
deterministic stats cache is written:

```bash
dojo inspect dataset --config .../nes_baseline.yaml --normalization   # train-split mean/std
dojo inspect dataset --config .../nes_baseline.yaml --stats           # write stats cache
dojo inspect dataset --config .../nes_baseline.yaml --dimensions      # per-sample native sizes
```

Also surfaces the preflight checks (empty train/eval classes, imbalance ratio)
used by the multihead configs.

### 4. Backbone & freeze inspection — `dojo inspect backbone`

Shows the resolved backbone and its **freeze policy** — the quickest way to
confirm `policy: frozen` took effect for the `07a`/`07b` transfer configs:

```bash
dojo inspect backbone --config configs/experiment/p2/07a_transfer-miniset/frozen.yaml
```

### 5. Checkpoint inspection — `dojo inspect checkpoint`

Dump a trained checkpoint's embedded inference contract (class maps,
preprocessing state) — useful before wiring a checkpoint URI into a transfer
config:

```bash
dojo inspect checkpoint runs/p2_nes_baseline/<ts>/checkpoints/last.ckpt
```

### 6. Inference — `dojo infer`

Write tall-Parquet `stage=infer` rows (predictions or embeddings) from a
checkpoint, composing data from the same selectors:

```bash
dojo infer predictions --checkpoint <ckpt> experiment=p2/00_baseline/nes_baseline
dojo infer embeddings  --checkpoint <ckpt> data=nes-hf     # feature vectors
```

### 7. Holdout evaluation — `dojo eval holdout`

Score a checkpoint against a holdout split, writing `stage=holdout_eval` rows:

```bash
dojo eval holdout --checkpoint <ckpt> data=ichthyoliths_denticle_type
```

### 8. Artifacts produced

A run writes, under `runs/<experiment.name>/<ts>_<run_id>/`:
`metrics/metrics.csv` (per-epoch), tall-Parquet result rows (`sample_metadata` +
`classification_output`, with head/target/dataset hashes), and standalone Plotly
`figures/*.html` (train/val loss, first-epoch-normalized losses, val F1
macro/micro, confusion matrix, per-class dropdown). All `infer`/`eval` rows share
that result schema, so they load with the same results reader.

> **Figure regeneration is not yet a command.** Figures are currently emitted
> only during a training run. Re-plotting from a completed run's `metrics.csv` /
> result Parquet without retraining (e.g. `dojo eval training-figures`) is
> proposed as the `add-figure-regeneration` OpenSpec change — see
> `openspec/changes/add-figure-regeneration/`. Useful once we start iterating on
> figure output quality.
