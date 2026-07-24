## 1. Schema

- [ ] 1.1 Extend `SamplerConfig.class_weight_scheme` Literal in
  `src/dojo/config_schemas/root.py` to include `sqrt_inverse_frequency`,
  `log_inverse_frequency`, `capped_inverse_frequency`, and `custom`
- [ ] 1.2 Add `max_weight_ratio: float = Field(default=10.0, gt=0.0)` to
  `SamplerConfig`
- [ ] 1.3 Add `class_weights: dict[str, float] = Field(default_factory=dict)`
  to `SamplerConfig`
- [ ] 1.4 Verify: `openspec validate --all` passes; round-trip a config with
  each new scheme value through Pydantic without error

## 2. Sampler weight function

- [ ] 2.1 Add `sqrt_inverse_frequency` branch to `class_weight_by_index` in
  `src/dojo/data/samplers.py`: `w = 1.0 / sqrt(count)`
- [ ] 2.2 Add `log_inverse_frequency` branch: `w = 1.0 / log(1.0 + count)`
- [ ] 2.3 Add `capped_inverse_frequency` branch: compute `1/count` for all
  classes, then cap at `max_weight_ratio / max(counts)`;  add
  `max_weight_ratio: float = 10.0` kwarg to `class_weight_by_index` and
  `sample_weights_for_dataset`
- [ ] 2.4 Add `custom` branch: add `class_name_to_index: dict[str, int] | None`
  and `class_weights: dict[str, float]` kwargs; raise `ValueError` for unknown
  names; emit `warnings.warn` listing classes that receive weight `0.0`
- [ ] 2.5 Update the `ValueError` message in the `else` branch to enumerate all
  six valid scheme names
- [ ] 2.6 Verify: `tests/` unit tests covering all four new branches in
  `class_weight_by_index`; assert capped weights, sqrt/log values, custom
  lookup, unknown-name error, and zero-weight warning

## 3. Loss weight function

- [ ] 3.1 Add `sqrt_inverse_frequency` branch to `class_weight_tensor` in
  `src/dojo/training/losses.py`: `weights[positive] = 1.0 / counts[positive].sqrt()`
- [ ] 3.2 Add `log_inverse_frequency` branch:
  `weights[positive] = 1.0 / (1.0 + counts[positive]).log()`
- [ ] 3.3 Add `capped_inverse_frequency` branch: inverse-frequency then
  `torch.clamp(weights, max=max_weight_ratio / counts[positive].max())`;
  add `max_weight_ratio: float = 10.0` kwarg to `class_weight_tensor`
- [ ] 3.4 Add `custom` branch: add `class_name_to_index: dict[str, int] | None`
  and `class_weights: dict[str, float]` kwargs; same validation as sampler side;
  emit `warnings.warn` for zero-weight classes
- [ ] 3.5 Update the `ValueError` message to enumerate all six valid scheme names
- [ ] 3.6 Verify: unit tests covering all four new branches in
  `class_weight_tensor`; assert normalization still holds (mean=1 over nonzero)
  for all schemes

## 4. Wiring

- [ ] 4.1 Add `max_weight_ratio` and `class_weights` kwargs to
  `build_dataloader` (or equivalent) in `src/dojo/data/__init__.py`; forward
  from `class_weight_scheme` call site at line 74
- [ ] 4.2 Pass `cfg.training.sampler.max_weight_ratio` and
  `cfg.training.sampler.class_weights` through from `src/dojo/training/run.py`
  (alongside the existing `class_weight_scheme` and `class_weight_beta` at
  lines 371–372)
- [ ] 4.3 Pass `class_name_to_index` (from the head's class mapping) to
  `sample_weights_for_dataset` when `scheme == "custom"`; same for the loss
  builder in `run.py` when constructing `class_weight_tensor`
- [ ] 4.4 Verify: integration test with `custom` scheme confirming the
  correct per-sample weights reach `WeightedBatchSampler`

## 5. Spec sync

- [ ] 5.1 Merge the `class-weight-schemes` delta spec into
  `openspec/specs/transforms-and-sampling/spec.md` sampler requirement
- [ ] 5.2 Merge the `supervised-training` delta spec loss requirement into
  `openspec/specs/supervised-training/spec.md`
- [ ] 5.3 Verify: `openspec validate --all` passes
