## Why

Workplan P3.4 (`DESIGN-DOC/13-workplan.md`,
`DESIGN-DOC/07-ssl-and-representation-eval.md`): self-supervised
pretraining and representation evaluation. Not started; the `ssl` and
`repr_eval` extras exist in `pyproject.toml` but nothing imports them.

## What Changes

- DINOv2 pretraining via Lightly (other SSL methods stay deferred,
  P4.5).
- `representation_eval`: probes, projections, clustering, diagnostics —
  gated by the `repr_eval` extra.
- Standalone and training-integrated execution.

## Capabilities

### New Capabilities

- `ssl-pretraining`: DINOv2 task type, SSL transforms, checkpoints
  usable as backbone `weights.source: checkpoint`.
- `representation-eval`: probe/projection/clustering/diagnostic outputs
  in the canonical result taxonomy.

### Modified Capabilities

- `config-and-cli`: new task type(s) in the strict schema.
- `results-and-artifacts`: representation-eval record types.

## Impact

- New SSL/repr-eval modules; heaviest new dependency surface (lightly,
  umap-learn, hdbscan, scikit-learn).
