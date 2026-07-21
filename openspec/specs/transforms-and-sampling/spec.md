# Transforms and Sampling Specification

## Purpose

The image transform builder, aspect buckets, and the DataLoader sampler
factory. Source design: `DESIGN-DOC/05-models-training-and-heads.md` and
`DESIGN-DOC/04-data-and-storage.md` (workplan P2.4).

## Requirements

### Requirement: Transform pipeline builder
`transforms.pipeline` SHALL support `resize` (direct to
`(height, width)`), `letterbox` (aspect-preserving padded),
`aspect_bucket`, `foreground_crop`, `grayscale`, flips, and `normalize`,
with a per-step `train_only` flag. A resolved-only `inference_pipeline`
(train-only steps stripped) SHALL be derived for non-train stages,
export, and `preprocessing_hash`.

#### Scenario: train_only steps excluded from inference
- **WHEN** a pipeline contains train-only augmentations (e.g. flips)
- **THEN** the derived `inference_pipeline` omits them

### Requirement: Aspect buckets
`aspect_bucket` SHALL deterministically choose a configured canvas from
native image dimensions and resize directly into that canvas (no
letterboxing), record the assignment as an `aspect_bucket` column on the
working manifest, the shared sample/batch contract, and
`sample_metadata` result rows.

#### Scenario: Stackable variable-size batches
- **WHEN** `training.sampler.type: batch_aspect_buckets` is configured
- **THEN** DataLoader batches are grouped by bucket assignment so
  variable canvas sizes remain tensor-stackable

### Requirement: Class-balanced and weighted samplers
`training.sampler.type` SHALL support `class_balanced` and `weighted`,
deriving per-sample weights from frozen train-split class counts, and
SHALL compose with aspect buckets by sampling within each bucket
(bucket grouping outer, class weighting within bucket).

#### Scenario: Sampler head selection
- **WHEN** `training.sampler.head` is omitted
- **THEN** the single classification head is used, or with multiple
  classification heads the one with the largest train-split imbalance
  ratio; weights follow that head's configured target while losses and
  metrics remain per-head

### Requirement: No resampling outside training
Validation, inference, and holdout-eval datasets SHALL NOT be resampled;
non-training paths choose bucket grouping from the dataset's resolved
aspect-bucket state only.

#### Scenario: Holdout eval with bucketed dataset
- **WHEN** holdout eval runs over a bucketed dataset
- **THEN** batches are bucket-grouped but sample order is not
  reweighted
