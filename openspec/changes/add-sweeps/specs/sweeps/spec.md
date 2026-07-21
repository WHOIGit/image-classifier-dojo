## ADDED Requirements

### Requirement: Dojo-owned sweep expansion
The system SHALL expand a top-level `sweep:` block (grid sweeps and
batch-run-style seed sweeps) into per-job configs via the Hydra Compose
API, without `@hydra.main` or `-m`.

#### Scenario: Grid expansion
- **WHEN** `dojo sweep prepare` runs on a grid sweep
- **THEN** one resolved job config per grid point is materialized with a
  deterministic job identity

### Requirement: Sweep lifecycle commands
The system SHALL provide `dojo sweep prepare`, manual per-job training
(`sweep.execution.mode: manual`), `dojo sweep status`, and
`dojo sweep report` writing `sweep_outputs` reporting.

#### Scenario: Status over a partial sweep
- **WHEN** some prepared jobs have completed runs
- **THEN** `dojo sweep status` reports per-job completion and
  `dojo sweep report` aggregates completed results
