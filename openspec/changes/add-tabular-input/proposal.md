## Why

Workplan P3.6 (`DESIGN-DOC/13-workplan.md`): tabular feature input
alongside the image stream. Excluded from P1 and P2.3 by design; the
embedding adapter and multi-head plumbing it composes with are now in
place.

## What Changes

- `model.tabular_input`: selected logical feature columns, input-stream
  name (default `tabular`), and tabular encoder (`identity`, `linear`,
  `mlp`; expanded families stay deferred, P4.14).
- `model.image_input.name` names the image stream (default `image`).
- Implicit embedding concatenation in canonical order (image first,
  tabular second); learned post-concat capacity via the existing
  `embedding_adapter`.
- `transforms.tabular`: numeric normalization, per-column imputation
  with frozen train-split fill values and optional missing indicators,
  one-hot encoding with frozen vocabularies and explicit
  unknown/missing tokens, train-only augmentations (`random_missing`).
- Resolved tabular preprocessing state persisted in the config artifact,
  exported with portable models, and folded into `preprocessing_hash` /
  `model_config_hash`.

## Capabilities

### New Capabilities

- `tabular-input`: tabular encoder, stream naming, tabular transforms
  and frozen preprocessing state.

### Modified Capabilities

- `model-composition`: dual-stream composition with canonical concat
  order.
- `results-and-artifacts`: tabular state joins the hash extractors and
  the portable inference contract.
- `dataset-backends`: tabular feature columns flow through the sample
  contract and stats cache.

## Impact

- `src/dojo/model/`, `src/dojo/data/transforms.py`, config schemas,
  hash extractors, inference contract. Tabular-only models stay
  deferred (P4.13).
