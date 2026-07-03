# src/dojo/config_loader — compose → validate → resolve

## Purpose

Turns CLI selectors/overrides into a validated, path-resolved `RootConfig`.

- `compositor.py` — Hydra Compose API (no `@hydra.main`, no launcher).
- `resolver.py` — renders `run_id`, resolves output paths, derives the
  inference pipeline.
- `conductor.py` — sequences compose → validate → resolve into one call.

## Ownership

Owns config composition and path/runtime resolution of `output_root` /
`dir_template`. Does not own the schema (that is `config_schemas/`) or the
config-group YAML (that is `configs/`).

## Local Contracts

- Packaged groups under `src/dojo/configs/` are shadowed and extended by a
  project-root `./configs/` when present.
- Validation is via the `config_schemas` Pydantic models; resolution happens
  after validation.
- `run_id` and the `*_hash` columns must stay deterministic and reproducible
  for a given config.

## Verification

- `tests/integration/test_config_fallthrough.py`.
</content>
