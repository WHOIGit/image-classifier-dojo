# src/dojo/configs — packaged Hydra config groups

## Purpose

The config groups that ship inside the package: `runtime`, `storage`, `data`,
`transforms`, `backbone`, `optimizer`, `training_outputs`, and `experiment`
roots (P1 experiments under `experiment/p1/`).

## Ownership

Owns the packaged defaults. The project-root `./configs/` directory shadows and
extends these without copying the whole tree — keep the two consistent in shape.

## Local Contracts

- `data=<name>` selects a config **group by name**, not a file path.
- An experiment root is a small file whose `defaults:` list pulls groups
  together plus the `model`, `objectives`, and `training` blocks.
- Every group must validate against `config_schemas`; adding a key requires a
  matching schema field (strict schema forbids extras).
</content>
