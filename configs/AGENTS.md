# configs — project-root config shadow

## Purpose

Optional project-root Hydra config groups that **shadow and extend** the
packaged groups in `src/dojo/configs/`, so datasets/experiments can be
overridden or added without copying the whole packaged tree.

## Ownership

Owns local, uncommitted-friendly config overrides (currently the
`plankton-miniset` / NES data selectors and the local `p1/` and `p2/`
experiment roots). Packaged defaults are owned by `src/dojo/configs/`.

## Local Contracts

- Group names here override same-named packaged groups; keep the schema shape
  identical (must validate against `config_schemas`).
- Mirror the packaged directory layout (`data/`, `experiment/`, …).
</content>
