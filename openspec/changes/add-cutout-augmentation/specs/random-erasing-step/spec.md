## ADDED Requirements

### Requirement: Random erasing pipeline step
The transform pipeline SHALL support a `random_erasing` step that randomly
masks one or more rectangular patches in the CHW float tensor with a
constant or random fill value. The step SHALL be `train_only: true` by
default and SHALL have no effect on the `inference_pipeline`.

Parameters:
- `num_patches` (int, default 1): number of patches to erase per image.
- `scale` (tuple[float, float], default [0.02, 0.33]): min/max fraction of
  total image area that a single patch may cover.
- `ratio` (tuple[float, float], default [0.3, 3.3]): min/max aspect ratio
  (width/height) of the erased patch.
- `value` (float | "random", default 0.0): fill value; `"random"` fills with
  per-patch uniform noise in [0, 1].
- `p` (float in [0, 1], default 0.5): probability of applying any erasing
  for a given image.
- `enabled` (bool, default true): global on/off, consistent with other steps.
- `train_only` (bool, default true): when true, excluded from `inference_pipeline`.

#### Scenario: Step skipped at inference probability gate
- **WHEN** `random_erasing` appears in `transforms.pipeline` with `train_only: true`
- **THEN** the resolved `inference_pipeline` omits it, so validation and
  holdout-eval images are not erased

#### Scenario: No erasing when probability gate not met
- **WHEN** a random draw exceeds `p`
- **THEN** the tensor is returned unchanged with no patches applied

#### Scenario: Single black patch erasing
- **WHEN** `num_patches: 1`, `value: 0.0`, and the probability gate is met
- **THEN** exactly one rectangular region is set to 0.0 in the output tensor

#### Scenario: Multiple random-fill patches
- **WHEN** `num_patches: 3`, `value: "random"`, and the probability gate is met
- **THEN** three rectangular regions are filled with independent uniform
  random noise in [0, 1]

#### Scenario: Invalid box rejected gracefully
- **WHEN** rejection sampling cannot find a box satisfying `scale` and `ratio`
  within the attempt limit (10 tries per patch)
- **THEN** that patch is skipped silently and remaining patches are still applied
