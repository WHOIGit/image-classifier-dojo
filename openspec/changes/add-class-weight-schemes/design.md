## Context

Class weights are derived in two independent places:

- **Sampler side** — `class_weight_by_index(class_counts, scheme, beta)` in
  `src/dojo/data/samplers.py` returns `dict[int, float]` (class index → weight),
  consumed by `sample_weights_for_dataset` to produce a per-sample weight list.
- **Loss side** — `class_weight_tensor(class_counts, num_classes, scheme, beta)`
  in `src/dojo/training/losses.py` returns a normalized `torch.Tensor` of length
  `num_classes`, consumed by `weighted_cross_entropy` and optionally by
  `focal_loss` with `scheme`.

Both functions currently accept only `inverse_frequency` and `effective_number`
and raise `ValueError` for anything else. They share the same conceptual
operation — mapping class counts to relative weights — but differ in output type
(`dict` vs `Tensor`) and normalization (loss side normalizes to mean=1; sampler
side does not normalize, relying on `random.choices` to handle unnormalized
weights).

## Goals / Non-Goals

**Goals:**
- Add `sqrt_inverse_frequency`, `log_inverse_frequency`,
  `capped_inverse_frequency`, and `custom` to both weight functions.
- Extend `SamplerConfig` schema with optional `max_weight_ratio` (float, default
  `10.0`) and `class_weights` (dict[str, float]) fields; loss params dict is
  untyped so no schema change needed there.
- Forward new params through `data/__init__.py` → `sample_weights_for_dataset`
  and through `training/run.py` → loss builder.
- Validate `custom` scheme: unknown class names raise `ValueError` at
  weight-derivation time (not at config parse time, since class mapping is not
  available in the schema layer).

**Non-Goals:**
- Unifying the sampler and loss weight functions into a single shared module
  (would require a cross-package dependency; keep them parallel for now).
- Learnable or schedule-adjusted class weights.
- Changing how `effective_number` or `beta` work.

## Decisions

### New scheme formulas

| Scheme | Formula per class c | Notes |
|---|---|---|
| `sqrt_inverse_frequency` | `1 / sqrt(count_c)` | Softer than 1/count; no new params |
| `log_inverse_frequency` | `1 / log(1 + count_c)` | Softest built-in; good for 3+ OOM count range |
| `capped_inverse_frequency` | `min(1/count_c, max_weight_ratio / count_mode)` | `count_mode` = count of the most-common class; cap expressed as a ratio so it scales with dataset size |
| `custom` | user-supplied float per class name | Weights used as-is before normalization |

For `capped_inverse_frequency`, the cap is `max_weight_ratio × (1 / count_mode)`
so a ratio of 10 means no class is sampled or weighted more than 10× the
most-common class. This is more interpretable than an absolute weight ceiling.

### `custom` scheme: name→index resolution

The user configures `class_weights: {SpeciesA: 2.0, SpeciesB: 0.5}`. The weight
functions receive `class_counts: dict[int, int]` (index-keyed), so they also
need the reverse mapping. Solution: add an optional
`class_name_to_index: dict[str, int] | None` parameter to both
`class_weight_by_index` and `class_weight_tensor`; required (non-None) only
when `scheme == "custom"`. Callers already have this mapping available
(`class_index_by_name` in the dataset bundle / `run.py`). Unknown names in
`class_weights` that are absent from `class_name_to_index` raise `ValueError`.
Classes absent from `class_weights` get weight `0.0` (effectively excluded),
which is intentional and documented.

### No normalization change on sampler side

The sampler side does not normalize weights because `random.choices` treats them
as unnormalized probabilities. Keeping this consistent with the existing
behaviour avoids changing the effective sampling distribution for existing
schemes.

### SamplerConfig schema additions

```python
max_weight_ratio: float = Field(default=10.0, gt=0.0)
class_weights: dict[str, float] = Field(default_factory=dict)
```

Both are always present in the schema (with harmless defaults) rather than being
`Optional` conditionally required by scheme, which would require a validator.
Users do not need to set them unless they choose the corresponding scheme.

## Risks / Trade-offs

- [Risk] `custom` with `class_weights` that don't cover all classes silently
  zero-weights the missing classes, which could starve rare-class training →
  Mitigation: document clearly; add a warning log when >0 classes receive zero
  weight from a `custom` config.
- [Risk] `log_inverse_frequency` produces weight `1/log(2) ≈ 1.44` for a class
  with count=1, which is very close to `1/log(2) ≈ 1.44` for count=1 and
  `1/log(11) ≈ 0.43` for count=10 — the contrast is much lower than
  inverse_frequency → Mitigation: document expected behaviour; this is the
  intended use case for very wide count ranges.
- [Risk] `capped_inverse_frequency` cap calculation depends on `max(class_counts)`
  which could be unstable if class counts change between runs → counts are frozen
  at dataset-hash time (same as today), so this is not a new risk.
