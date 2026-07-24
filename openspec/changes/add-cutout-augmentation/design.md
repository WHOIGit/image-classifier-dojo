## Context

The transform pipeline in `src/dojo/data/transforms.py` dispatches over a
`Sequence[TransformStep]` inside `_CompiledImageTransform.__call__`. Each step
is a discriminated-union Pydantic model declared in `src/dojo/config_schemas/root.py`
and keyed on the `name` literal. The `train_only` flag on each step is handled
upstream by the config resolver (`_derive_inference_pipeline`), which strips
`train_only: true` steps before building the `inference_pipeline` list; the
compiled transform itself just applies whatever list it receives.

The p2 augmentation ablation currently has a `foreground_crop` placeholder that
is a no-op on light-background plankton imagery. This change replaces it with a
`random_erasing` step that is dataset-agnostic.

## Goals / Non-Goals

**Goals:**
- Add `RandomErasingStep` Pydantic model with parameters: `num_patches`, `scale`,
  `ratio`, `value`, `p`, and the standard `enabled`/`train_only` flags.
- Implement `_random_erasing` as a pure function operating on a CHW float tensor.
- Dispatch the new step in `_CompiledImageTransform.__call__`.
- Fill in `nes_cutouts.yaml` with a concrete `random_erasing` pipeline step.
- Add the step to the `transforms-and-sampling` spec.

**Non-Goals:**
- GPU-side augmentation or torchvision `v2.RandomErasing` transform object
  (we operate as a step function on tensors, not a transform composition).
- Multi-scale or saliency-guided erasing strategies.
- Changing `foreground_crop` behaviour or removing it from the codebase.

## Decisions

### Use a manual patch loop rather than `torchvision.transforms.v2.functional`

`torchvision.transforms.v2.functional` does not expose a functional
`random_erasing` that operates on a plain CHW float tensor without wrapping
it in a `tv_tensors.Image`. Every other step in `transforms.py` calls
`F.<op>(tensor, ...)` directly without wrapping. To stay consistent we
implement a small manual loop: sample a bounding box using the same
scale/ratio rejection-sampling strategy as torchvision's implementation,
then fill with `step.value` (float) or uniform random noise per-patch.

Tradeoff: slightly more code than delegating to torchvision, but avoids
introducing a dependency on the `tv_tensors` wrapping convention that the
rest of the pipeline doesn't use.

### `value` as `float | Literal["random"]`

A scalar float (default `0.0`) fills patches with a constant; `"random"`
fills with `torch.rand_like` per patch. This covers the two most common
use cases (black cutout and random noise) without pulling in colour jitter
or mean-fill complexity.

### `num_patches` default of 1, `scale` default `[0.02, 0.33]`, `ratio` default `[0.3, 3.3]`

These match torchvision `RandomErasing` defaults, which are well-studied for
ImageNet-scale classification. The user can override all of them per-experiment.

### `p` applied once per call, not per patch

A single probability gate before the patch loop is simpler and means
`p=0.5` gives a 50 % chance of any erasing at all, consistent with how
other stochastic steps (`horizontal_flip`, `vertical_flip`) apply `p`.

## Risks / Trade-offs

- [Risk] Rejection sampling for patch bounding box may loop indefinitely for
  extreme `scale`/`ratio` combinations → Mitigation: cap attempts at 10
  (same as torchvision) and skip erasing if no valid box is found.
- [Risk] `value="random"` fills with `[0, 1]` uniform noise, which after
  ImageNet normalization may produce out-of-distribution values → Acceptable
  for augmentation purposes; users who care can use `value: 0.0`.
