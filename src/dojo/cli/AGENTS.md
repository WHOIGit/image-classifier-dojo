# src/dojo/cli — command surface

## Purpose

Typer front-end over the Hydra Compose API. P1 commands: `dojo inspect config`
and `dojo train`. `main.py` wires the Typer app; `inspect.py` and `train.py`
hold the per-command logic.

## Ownership

Owns argument parsing, option/override splitting, and human-facing output
(rich tables, JSON). Delegates all composition/validation/resolution to
`config_loader/` and all execution to `training/`.

## Local Contracts

- Dash-prefixed tokens are command options; dash-free `key=value` tokens are
  Hydra config overrides. Preserve this split.
- `--config`, `--resolved-config`, and `--format` are the stable option seam.
- CLI is the only layer that owns terminal I/O; business logic returns
  structured results.

## Verification

- `tests/integration/test_inspect_config.py`, `test_train_supervised.py`.
</content>
