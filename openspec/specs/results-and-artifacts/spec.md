# Results and Artifacts Specification

## Purpose

Canonical tall-Parquet results, the `_metadata.json` sidecar, identity
and compatibility hashes, and the embedded portable inference contract.
Source design: `DESIGN-DOC/06-results-artifacts-and-metadata.md`
(workplan P1, P2.5a, P2.5b).

## Requirements

### Requirement: Canonical tall-Parquet results
Results SHALL be written as tall Parquet via `amplify-db-utils` with
provenance columns and a `_metadata.json` sidecar validating against its
schema. The record-type taxonomy SHALL include `sample_metadata`,
`classification_output`, and `embedding`; readers can filter by `stage`
and `record_type`.

#### Scenario: Round-trip and filter
- **WHEN** a reader loads a run's result Parquet
- **THEN** filtering by `stage` / `record_type` returns the expected row
  subsets

### Requirement: Self-scoring classification rows
`classification_output` rows SHALL carry ground-truth `target_index` /
`target_name` alongside the `prediction_*` columns, scoped by
`head_name`, so a row is self-scoring without joining back to
`sample_metadata`; the head → target link lives in `_metadata.json`.

#### Scenario: Score without joins
- **WHEN** a consumer computes accuracy from `classification_output`
  rows alone
- **THEN** all needed truth and prediction columns are present

### Requirement: Identity and compatibility hashes
The system SHALL compute deterministic `config_hash`, `dataset_hash`,
and `checkpoint_hash`, plus compatibility hashes `target_schema_hash`,
`class_mapping_hash`, `model_config_hash`, and `preprocessing_hash`
implementing the pinned source-field lists, and a per-head `head_hash`
(content hash over the head's resolved target, `num_classes`,
`class_mapping`, network) for join-free cross-run grouping.
`_metadata.json` SHALL carry the compatibility block.

#### Scenario: Reproducible hashes
- **WHEN** the same config runs twice
- **THEN** all hashes are identical across the two runs

### Requirement: Embedded portable inference contract
`SupervisedTaskModule.on_save_checkpoint` SHALL embed
`checkpoint["dojo_inference_contract"]` containing a buildable
`model_config`, the resolved `inference_pipeline` with frozen
preprocessing stats, per-head class maps, target schema, resolved
`objective_summary`, and the four compatibility hashes.

#### Scenario: Consumer independence from resolved.yaml
- **WHEN** a checkpoint is loaded for inference or holdout eval
- **THEN** the model and preprocessing are reconstructed from the
  embedded contract without the producing run's `resolved.yaml`
