## Why

Workplan P3.2 (`DESIGN-DOC/13-workplan.md`, `DESIGN-DOC/08-ensembles.md`):
prediction-space ensembling over compatible runs. Not started.

## What Changes

- Candidate discovery with explicit sources only (registry-based
  discovery stays deferred, P4.11).
- Compatibility checking via the compatibility hashes
  (`target_schema_hash`, `class_mapping_hash`, per-head `head_hash`).
- Selection strategies and combine modes (weighted combine modes and
  `prediction_trimmed_mean` stay deferred, P4.6b/c).
- Cached-result path reusing canonical result Parquet.
- `task.type: snapshot_ensemble`.

## Capabilities

### New Capabilities

- `prediction-ensembling`: candidate discovery, compatibility checks,
  selection, combine modes, cached-result path, snapshot ensembles.

### Modified Capabilities

- `config-and-cli`: new `task.type: snapshot_ensemble` in the strict
  schema.

## Impact

- New ensembling module; depends on `results-and-artifacts` hashes and
  optionally `model-export` (P3.1) for exported-model members.
