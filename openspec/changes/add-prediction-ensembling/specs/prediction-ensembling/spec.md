## ADDED Requirements

### Requirement: Prediction-space ensembling
The system SHALL combine predictions from explicitly listed candidate
sources (checkpoints, exported models, or cached canonical results),
verifying member compatibility via `target_schema_hash`,
`class_mapping_hash`, and per-head `head_hash` before combining, with
configurable selection strategies and combine modes.

#### Scenario: Incompatible member rejected
- **WHEN** a candidate's class-mapping hash differs from the ensemble's
- **THEN** the run fails compatibility checking with a clear report

#### Scenario: Cached-result member
- **WHEN** a member is given as an existing canonical result Parquet
- **THEN** its predictions are reused without a forward pass

### Requirement: Snapshot ensembles
`task.type: snapshot_ensemble` SHALL ensemble checkpoints produced by a
single training run.

#### Scenario: Snapshot ensemble run
- **WHEN** a snapshot-ensemble task runs over a run's best-k checkpoints
- **THEN** combined predictions are written as canonical result rows
