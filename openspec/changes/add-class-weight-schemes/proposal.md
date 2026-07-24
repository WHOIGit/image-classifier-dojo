## Why

The current `class_weight_scheme` parameter supports only `inverse_frequency`
and `effective_number` in both the weighted sampler and the class-weighted loss
functions. On long-tail datasets like NES plankton (155 species with counts
spanning several orders of magnitude), plain inverse-frequency weighting can
produce extreme per-class weights that destabilise training or cause a handful
of very rare classes to dominate every batch. Adding softer curves and a
hard-cap scheme gives practitioners a menu of calibrated alternatives without
requiring the beta hyperparameter of `effective_number`. A `custom` scheme
covers the domain-knowledge case where statistical frequency alone should not
dictate importance.

## What Changes

- Four new `class_weight_scheme` values: `sqrt_inverse_frequency`,
  `log_inverse_frequency`, `capped_inverse_frequency`, and `custom`, accepted
  by both the sampler weight function (`src/dojo/data/samplers.py`) and the
  loss weight function (`src/dojo/training/losses.py`).
- `capped_inverse_frequency` adds an optional `max_weight_ratio` parameter
  (default `10.0`) to `SamplerConfig` and to loss `params`.
- `custom` adds an optional `class_weights` parameter (map from class name to
  float) to `SamplerConfig` and to loss `params`; unknown class names raise a
  validation error at weight-derivation time.
- Schema `Literal` for `class_weight_scheme` extended in `SamplerConfig`; loss
  side is untyped `params` dict so no schema change needed there beyond
  documentation.
- Delta specs for both `transforms-and-sampling` (sampler requirement) and
  `supervised-training` (loss requirement).

## Capabilities

### New Capabilities

- `class-weight-schemes`: The four new weighting schemes and their parameters,
  shared between sampler and loss weight derivation.

### Modified Capabilities

- `transforms-and-sampling`: Sampler requirement extended to enumerate all
  supported `class_weight_scheme` values and the new `max_weight_ratio` /
  `class_weights` parameters.
- `supervised-training`: Weighted-loss requirement extended to enumerate the
  same scheme values and parameters for `weighted_cross_entropy` and
  `focal_loss` with `scheme`.

## Impact

- `src/dojo/data/samplers.py` — extend `class_weight_by_index` with four new
  branches; `sample_weights_for_dataset` gains `max_weight_ratio` and
  `class_weights` kwargs forwarded from callers.
- `src/dojo/training/losses.py` — extend `class_weight_tensor` with the same
  four branches and the same extra kwargs.
- `src/dojo/config_schemas/root.py` — extend `SamplerConfig.class_weight_scheme`
  Literal and add `max_weight_ratio: float` and `class_weights: dict[str, float]`
  optional fields.
- `src/dojo/data/__init__.py` — forward new sampler config fields to
  `sample_weights_for_dataset`.
- `src/dojo/training/run.py` — forward new sampler config fields when building
  the dataloader.
- No changes to result schemas, CLI, or Parquet outputs.
