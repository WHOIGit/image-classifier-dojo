## Why

Workplan P3.3 (`DESIGN-DOC/13-workplan.md`,
`DESIGN-DOC/09-sweeps-and-batch-runs.md`): grid sweeps and
batch-run-style seed sweeps. Not started.

## What Changes

- Dojo-owned sweep expansion over the Hydra Compose API (no
  `@hydra.main`, no `-m`).
- Top-level `sweep:` block in the strict schema: grid sweeps and seed
  sweeps.
- `dojo sweep prepare`, manual per-job training
  (`sweep.execution.mode: manual`; automated runners stay deferred,
  P4.16), `dojo sweep status`, `dojo sweep report`.
- `sweep_outputs` reporting.

## Capabilities

### New Capabilities

- `sweeps`: sweep expansion, prepare/status/report commands, manual
  execution, sweep outputs.

### Modified Capabilities

- `config-and-cli`: top-level `sweep:` block joins the strict schema.

## Impact

- New sweep module and `dojo sweep` CLI group; consumes config hashing
  for job identity.
