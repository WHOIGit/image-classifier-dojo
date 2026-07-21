## Why

Workplan P3.5 (`DESIGN-DOC/13-workplan.md`,
`DESIGN-DOC/04-data-and-storage.md`): first-class IFCB bin datasets via
`ifcbkit`. Not started; the `ifcb` extra exists but nothing imports it.

## What Changes

- `ifcb_bins` dataset backend across train / eval / infer paths
  (including SSL where applicable), gated by the `ifcb` extra.
- Custom dataset and dataloaders for IFCB bins under the shared sample
  contract.

## Capabilities

### New Capabilities

- `ifcb-bins-backend`: `ifcb_bins` backend, bin-aware dataset and
  dataloaders.

### Modified Capabilities

- `dataset-backends`: `ifcb_bins` joins the backend registry and the
  stats-cache / inspect-dataset paths (bin-length values already have a
  reserved slot in the frozen stats).

## Impact

- New backend in `src/dojo/data/`; `ifcbkit` git dependency exercised
  for the first time.
