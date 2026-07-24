## ADDED Requirements

### Requirement: Extended class weight schemes
The class weight derivation functions SHALL support six named schemes:
`inverse_frequency`, `effective_number`, `sqrt_inverse_frequency`,
`log_inverse_frequency`, `capped_inverse_frequency`, and `custom`. All six
SHALL be available identically in both the sampler weight path
(`class_weight_by_index` in `src/dojo/data/samplers.py`) and the loss weight
path (`class_weight_tensor` in `src/dojo/training/losses.py`).

Scheme formulas (c = class, n_c = count of class c, n_max = count of
most-common class):

- `inverse_frequency`: `w_c = 1 / n_c`
- `effective_number`: `w_c = (1 - β) / (1 - β^n_c)`, requires `beta ∈ (0,1)`
- `sqrt_inverse_frequency`: `w_c = 1 / sqrt(n_c)`
- `log_inverse_frequency`: `w_c = 1 / log(1 + n_c)`
- `capped_inverse_frequency`: `w_c = min(1/n_c, max_weight_ratio / n_max)`,
  requires `max_weight_ratio > 0` (default `10.0`)
- `custom`: `w_c = class_weights[name_c]` for each class; classes absent from
  `class_weights` receive weight `0.0`; unknown names raise `ValueError`

#### Scenario: Softer weighting with sqrt scheme
- **WHEN** `class_weight_scheme: sqrt_inverse_frequency` is configured
- **THEN** weights follow `1/sqrt(count)`, producing a shallower slope than
  `inverse_frequency` across the class count range

#### Scenario: Log weighting for wide count ranges
- **WHEN** `class_weight_scheme: log_inverse_frequency` is configured
- **THEN** weights follow `1/log(1 + count)`, compressing the weight range
  for datasets where counts span several orders of magnitude

#### Scenario: Capped weighting limits rare-class dominance
- **WHEN** `class_weight_scheme: capped_inverse_frequency` and
  `max_weight_ratio: 10.0` are configured
- **THEN** no class receives a weight more than 10× that of the most-common
  class

#### Scenario: Custom per-class weights
- **WHEN** `class_weight_scheme: custom` and `class_weights: {SpeciesA: 2.0}`
  are configured
- **THEN** `SpeciesA` receives weight `2.0`; all other classes receive `0.0`

#### Scenario: Unknown class name in custom weights
- **WHEN** `class_weight_scheme: custom` references a class name not present
  in the dataset's class mapping
- **THEN** weight derivation raises `ValueError` identifying the unknown name

#### Scenario: Zero-weight classes logged as warning
- **WHEN** `class_weight_scheme: custom` leaves one or more classes with
  weight `0.0` (absent from `class_weights`)
- **THEN** a warning is emitted listing the zero-weight class names so the
  user can confirm the omission is intentional
