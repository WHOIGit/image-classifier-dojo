## Context

`DESIGN-DOC/03-configuration.md` ("Run config artifacts") specifies five files under
a run's `config/`; only `resolved.yaml` is written today, by
`_write_resolved_config` (`src/dojo/training/run.py:73-78`), called from
`execute_train`.

The three unwritten provenance files need information the run never sees.
`compose_config` (`src/dojo/config_loader/compositor.py:155`) returns a
`ComposedConfig` carrying `config`, `config_name`, `config_dirs`, and `overrides`,
but `compose_and_resolve` (`src/dojo/config_loader/conductor.py:37-57`) uses it as a
local and returns a `ResolutionResult` holding only `config` + `warnings`
(`src/dojo/config_loader/resolver.py:22-25`). By the time `execute_train` runs, the
composed tree and the overrides are gone.

Constraints:
- `execute_train` deliberately has no Typer dependency and accepts an
  already-resolved `RootConfig`; tests and external orchestrators call it directly
  (`tests/integration/test_train_supervised.py:41,153,199`) and must keep working.
- `dojo train --resolved-config` starts from an artifact with no composition
  history, so provenance is genuinely absent in a supported path — not just in tests.
- `eval_outputs.dir` carries the same `config/` layout
  (`DESIGN-DOC/06-results-artifacts-and-metadata.md:945-951`), and `dojo eval` /
  `dojo infer` already call `compose_and_resolve`.

## Goals / Non-Goals

**Goals:**
- Write `composed.yaml`, `resolved.json`, `cli.txt`, and `overrides.txt` next to the
  existing `resolved.yaml` for CLI-invoked runs.
- Carry composition provenance from the loader to whatever writes the run directory.
- Keep the writer reusable by `dojo eval` / `dojo infer` without moving run-directory
  ownership into the CLI.
- Degrade honestly when provenance is absent.

**Non-Goals:**
- `config/sweep_values.txt` — sweep members only; belongs to `add-sweeps`.
- Wiring the writer into `dojo eval` / `dojo infer`. This change makes it reusable
  and leaves adoption to the eval-outputs work.
- Changing `resolved.yaml`'s name, location, or content, or any config schema change.
- Storage-interface routing for config artifacts; these stay direct local writes, as
  `_write_resolved_config` is today.

## Decisions

### Provenance rides on `ResolutionResult`, not on `RootConfig`

`ResolutionResult` gains the composed config, the overrides, and the invoked command.
`compose_and_resolve` already builds the `ComposedConfig`; it stops discarding it.

Alternative rejected — put the artifacts on `RootConfig`: the schema is strict and
describes the run's configuration, not how it was invoked. Composition history is
metadata about the call, and adding it would pollute `config_hash` inputs.

Alternative rejected — have the run recompose from `config_name` + `config_dirs`:
recomposition can diverge from what actually ran and costs a second Hydra pass.

### `execute_train` keeps owning `config/`, taking provenance as an optional argument

`execute_train` gains an optional provenance parameter, defaulting to `None`. The
run directory stays assembled in one place, so no caller can forget part of it.

Alternative rejected — write the artifacts in the CLI layer: it splits run-directory
construction across two modules and makes every future command re-implement it.

### The invoked command is captured at the CLI boundary, not reconstructed

`dojo train` records the actual invocation (from `sys.argv`) and passes it in.
Reconstructing a plausible command line inside the loader would produce a string
that was never run — the same fabrication problem as a synthesized `composed.yaml`.
`cli.txt` is therefore CLI-only by construction, which matches its purpose as an
audit record.

### Absent provenance means absent files, never fabricated ones

With no provenance, only `resolved.yaml` and `resolved.json` are written. A
`composed.yaml` reconstructed by stripping resolved values would be a guess at the
pre-resolution input and would silently mislead anyone branching a run from it
(`DESIGN-DOC/03-configuration.md:390`). A missing file is unambiguous.

### A shared writer module, not a training-private helper

`_write_resolved_config` is replaced by a writer that takes a target `config/`
directory, a resolved `RootConfig`, and optional provenance, and writes whichever
subset it can. It lives outside `dojo.training` so `dojo eval` / `dojo infer` can
adopt it for `eval_outputs.dir` without importing the training package.

### Serialization mirrors the existing resolved-YAML path

`resolved.json` is produced from the same `cfg.model_dump(mode="json",
exclude_none=True)` container that already backs `resolved.yaml`, so the two cannot
drift. `composed.yaml` is written from the composed `DictConfig` via OmegaConf,
preserving it as composed rather than round-tripping through the Pydantic schema.
`overrides.txt` is one override per line; both text files end with a trailing
newline.

## Risks / Trade-offs

- **`composed.yaml` is absent for `--resolved-config` runs, so a run directory's
  contents vary by invocation path** → Specified explicitly as a scenario, and the
  variance is honest: those runs had no composition step. Consumers must treat the
  three provenance files as optional; `resolved.yaml` / `resolved.json` remain the
  guaranteed pair that ensembling and eval read.
- **`cli.txt` may capture secrets passed as inline overrides** → Overrides are config
  keys, and the project's configs carry paths and hyperparameters, not credentials.
  Worth revisiting if remote-storage credentials ever become overridable.
- **Adding fields to `ResolutionResult` touches every `compose_and_resolve` caller's
  expectations** → The new fields are additive with defaults; the four other callers
  (`cli/eval.py`, `cli/infer.py`, `cli/inspect.py`, `cli/train.py`) read
  `.config` / `.warnings` and are unaffected.
- **Two serializations of the resolved config could drift** → Both derive from one
  `model_dump` call, and a test asserts they parse equal.
