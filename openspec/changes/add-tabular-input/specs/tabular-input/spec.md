## ADDED Requirements

### Requirement: Tabular input stream
`model.tabular_input` SHALL select logical feature columns into a named
input stream (default `tabular`) encoded by `identity`, `linear`, or
`mlp`; when image and tabular inputs are both enabled the compositor
concatenates embeddings in canonical order (image first, tabular
second), with learned post-concat capacity via `embedding_adapter`.

#### Scenario: Dual-stream supervised model
- **WHEN** image and tabular inputs are both enabled
- **THEN** heads receive the canonical-order concatenated embedding

### Requirement: Tabular transforms with frozen state
`transforms.tabular` SHALL support numeric normalization, per-column
imputation with frozen train-split fill values and optional missing
indicators, one-hot encoding with frozen vocabularies and explicit
unknown/missing tokens, and train-only augmentations such as
`random_missing`; resolved state SHALL persist in the config artifact,
export with portable models, and feed `preprocessing_hash` /
`model_config_hash`.

#### Scenario: Unknown category at inference
- **WHEN** inference encounters a category absent from the frozen
  vocabulary
- **THEN** it maps to the explicit unknown token rather than failing
