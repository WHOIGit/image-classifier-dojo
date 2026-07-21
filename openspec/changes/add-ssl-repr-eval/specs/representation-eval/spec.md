## ADDED Requirements

### Requirement: Representation evaluation
`representation_eval` SHALL run probes, projections, clustering, and
diagnostics over embeddings (gated by the `repr_eval` extra), both
standalone and training-integrated, writing outputs in the canonical
result taxonomy.

#### Scenario: Standalone evaluation of an embedding run
- **WHEN** representation eval runs over `record_type=embedding` rows
- **THEN** probe/projection/clustering outputs are written with
  canonical provenance columns
