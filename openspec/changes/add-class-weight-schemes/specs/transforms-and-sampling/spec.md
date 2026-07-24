## MODIFIED Requirements

### Requirement: Class-balanced and weighted samplers
`training.sampler.type` SHALL support `class_balanced` and `weighted`,
deriving per-sample weights from frozen train-split class counts using any
scheme from the [[class-weight-schemes]] capability: `inverse_frequency`,
`effective_number`, `sqrt_inverse_frequency`, `log_inverse_frequency`,
`capped_inverse_frequency`, or `custom`. The `SamplerConfig` SHALL expose
`max_weight_ratio: float` (default `10.0`, only used with
`capped_inverse_frequency`) and `class_weights: dict[str, float]` (default
empty, only used with `custom`). Sampling SHALL compose with aspect buckets
by applying class weighting within each bucket.

#### Scenario: Sampler head selection
- **WHEN** `training.sampler.head` is omitted
- **THEN** the single classification head is used, or with multiple
  classification heads the one with the largest train-split imbalance
  ratio; weights follow that head's configured target while losses and
  metrics remain per-head

#### Scenario: Capped sampler weight
- **WHEN** `training.sampler.type: weighted` and
  `class_weight_scheme: capped_inverse_frequency` with `max_weight_ratio: 5.0`
- **THEN** no sample is assigned a weight more than 5× that of the
  most-common class's samples
