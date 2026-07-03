# src/dojo — the `dojo` package

## Purpose

The importable `dojo` package: the completed Priority-1 end-to-end thin slice
(config → data → model → training → storage → results) plus the implemented
Priority-2 core supervised-platform surface. Base install is Torch-free for
config/init/storage/result operations and lightweight inspect paths; dataset
inspection, model inspection, training, inference, and eval need the vision
stack.

## Ownership

Owns all runtime code under `src/dojo/`. The sibling `src/dojo_deprecated/` is
legacy `dojo` code kept for reference only — do not extend it, and treat it as
outside the DOX contract. `amplify-db-utils` and `amplify-storage-utils` are
external dependencies, not owned here.

## Local Contracts

- Architecture flows one direction: `cli` → `config_loader`/`config_schemas` →
  `data` + `model` → `training` → `storage` + `results`. Do not create back-edges
  (e.g. schemas must not import training).
- Pydantic schemas in `config_schemas/` are the runtime contract; strict schema
  (`extra="forbid"`) — deferred features are absent, not stubbed.
- Entry points return structured results for external orchestration; avoid
  `print`-driven control flow outside `cli/`.
- Design-doc sections in `DESIGN-DOC/` are the authoritative spec; when code and
  spec diverge, reconcile explicitly rather than silently drifting.

## Work Guidance

- Keep implemented features functional end to end: schema, runtime behavior,
  result provenance, and tests should land together. Leave P3+ values absent
  from the strict schema until wired through.

## Verification

- `pytest` (fast loop); `pytest --run-expensive` for real fits. Run from repo root.

## Child DOX Index

- [cli/](cli/AGENTS.md) — Typer front-end over the Hydra Compose API.
- [config_loader/](config_loader/AGENTS.md) — compose → validate → resolve.
- [config_schemas/](config_schemas/AGENTS.md) — strict Pydantic contract + hashing.
- [configs/](configs/AGENTS.md) — packaged Hydra config groups.
- [data/](data/AGENTS.md) — dataset backends, sample contract, transforms.
- [inference/](inference/AGENTS.md) — checkpoint-backed inference and holdout eval.
- [model/](model/AGENTS.md) — backbone + heads + supervised composition.
- [training/](training/AGENTS.md) — Lightning task, trainer, run orchestration.
- [storage/](storage/AGENTS.md) — URI-based storage seam.
- [results/](results/AGENTS.md) — canonical tall-Parquet writer/reader + sidecar.
</content>
