## Why

The P2 augmentation ablation (`p2_aug_fgcrop`) was intended to test spatial
augmentation beyond flips, but the `foreground_crop` step is a no-op on NES
plankton imagery because the dataset has a light background (no all-zero border
rows or columns at `threshold: 0.0`). Adding `random_erasing` (cutout-style
occlusion) gives the 06_augmentation group a meaningful train-only augmentation
that works on any background type and is a well-supported technique for
improving generalization on imbalanced, texture-rich datasets.

## What Changes

- New pipeline step `random_erasing` added to `TransformStep` union in
  `src/dojo/config_schemas/root.py`.
- Implementation in `src/dojo/data/transforms.py` dispatched inside
  `_CompiledImageTransform.__call__`, consistent with existing step pattern.
- `transforms-and-sampling` spec updated to list `random_erasing` as a
  supported step.
- `configs/experiment/p2/06_augmentation/nes_cutouts.yaml` placeholder filled
  in with a concrete `random_erasing` pipeline step.

## Capabilities

### New Capabilities

- `random-erasing-step`: A train-only pipeline step that randomly masks one or
  more rectangular patches in the image tensor with a constant or random fill
  value, parameterised by patch count, area scale range, aspect ratio range,
  fill value, and application probability.

### Modified Capabilities

- `transforms-and-sampling`: Requirement list extended to include
  `random_erasing` as a supported named step alongside the existing steps.

## Impact

- `src/dojo/config_schemas/root.py` — new `RandomErasingStep` model, added to
  `TransformStep` union.
- `src/dojo/data/transforms.py` — new `_random_erasing` helper and dispatch
  branch in `_CompiledImageTransform.__call__`.
- `configs/experiment/p2/06_augmentation/nes_cutouts.yaml` — filled in from
  placeholder.
- `openspec/specs/transforms-and-sampling/spec.md` — delta spec adds the new
  requirement.
- No changes to inference pipeline derivation, DataLoader, or result schemas;
  `train_only: true` default means the step is automatically excluded from
  `inference_pipeline`.
