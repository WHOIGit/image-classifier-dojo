# src/dojo/cli — command surface

## Purpose

Typer front-end over the Hydra Compose API and checkpoint-backed inference.
Implemented commands: `dojo init`, `dojo inspect config`, `dojo inspect
config-compare`, `dojo inspect dataset`, `dojo inspect backbone`, `dojo inspect
checkpoint`, `dojo train`, `dojo infer predictions`, `dojo infer embeddings`,
and `dojo eval holdout`.
`main.py` wires the Typer app; command modules hold per-command logic.

## Ownership

Owns argument parsing, option/override splitting, and human-facing output
(rich tables, JSON). Delegates composition/validation/resolution to
`config_loader/` and execution to `data/`, `training/`, or `inference/`.

## Local Contracts

- Dash-prefixed tokens are command options; dash-free `key=value` tokens are
  Hydra config overrides. Preserve this split.
- `--config`, `--resolved-config`, and `--format` are the stable option seam
  for compose-backed commands.
- CLI is the only layer that owns terminal I/O; business logic returns
  structured results.
- `dojo train` displays phase status messages from the training
  `status_callback`, especially around post-fit checkpoint, results, metrics,
  and figure work.
- `dojo init` is additive by default: create missing files, skip existing
  files, overwrite only with `--clobber`, and write nothing with `--dry-run`.
- `dojo inspect dataset --stats` writes `data.stats_cache_uri` only when that
  config field is set; otherwise it is display-only.

## Verification

- `tests/integration/test_init.py`, `test_inspect_config.py`,
  `test_inspect_dataset.py`, `test_train_supervised.py`, plus
  `tests/unit/inference/` for infer/eval execution behavior.
</content>
