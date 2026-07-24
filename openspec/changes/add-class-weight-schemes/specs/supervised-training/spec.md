## MODIFIED Requirements

### Requirement: Objective losses
The strict-schema objective loss types SHALL include `cross_entropy`,
`weighted_cross_entropy`, and `focal_loss`. `weighted_cross_entropy` derives
weights from train-split class counts using any scheme from the
[[class-weight-schemes]] capability: `inverse_frequency` (default),
`effective_number` (configurable `beta`), `sqrt_inverse_frequency`,
`log_inverse_frequency`, `capped_inverse_frequency` (configurable
`max_weight_ratio`, default `10.0`), or `custom` (configurable
`class_weights: dict[str, float]`). All schemes are normalized to mean one
over non-empty classes before being passed to the loss function. `focal_loss`
supports the same `scheme` / `beta` / `max_weight_ratio` / `class_weights`
parameters when `alpha` is count-derived. Label smoothing is available on
`cross_entropy` / `weighted_cross_entropy` via `params.label_smoothing`.

#### Scenario: Per-head class counts for weighted losses
- **WHEN** a weighted loss is configured on a head
- **THEN** class counts are selected for that head's configured target

#### Scenario: Capped loss weight
- **WHEN** `loss: weighted_cross_entropy` and
  `params.scheme: capped_inverse_frequency` with `params.max_weight_ratio: 10.0`
- **THEN** no class loss weight exceeds 10× the weight of the most-common class
  before normalization
