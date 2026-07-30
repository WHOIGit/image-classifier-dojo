## Why

`DESIGN-DOC/03-configuration.md` ("Run config artifacts") specifies that every run
writes five files into `config/` under its resolved output directory, but training
runs today emit only `resolved.yaml` (`src/dojo/training/run.py:73-78`). The missing
`resolved.json` is a documented consumer input for ensembling
(`DESIGN-DOC/08-ensembles.md:216,253`), and the missing `composed.yaml` is the
documented way to branch a run (`DESIGN-DOC/03-configuration.md:390`) — so the gap
blocks the P3 ensembling work and leaves finished runs without the provenance needed
to reproduce or audit them. This closes a P2.1 (config / CLI / storage foundation)
gap that no spec currently covers.

## What Changes

- Write `config/composed.yaml` — the config after Hydra composition and CLI
  overrides, before runtime/path resolution.
- Write `config/resolved.json` — the JSON form of the resolved config, alongside the
  existing `resolved.yaml`.
- Write `config/cli.txt` — the invoked command line.
- Write `config/overrides.txt` — the override list applied, one per line.
- Thread composition provenance from the config loader to the run: `ResolutionResult`
  gains the composed config, the overrides, and the invoked command;
  `compose_and_resolve` stops discarding the `ComposedConfig` it already builds.
- `execute_train` takes the provenance as an optional argument. Callers that pass a
  bare `RootConfig` (tests, external orchestrators) still get `resolved.yaml` and
  `resolved.json`; the three provenance-derived files are skipped rather than
  fabricated from the resolved config.
- Factor the writer into a shared helper so `dojo eval` / `dojo infer` — whose
  `eval_outputs.dir` carries the same `config/` layout
  (`DESIGN-DOC/06-results-artifacts-and-metadata.md:945-951`) — can reuse it.
- Not breaking: `resolved.yaml` keeps its name, location, and content.

Out of scope: `config/sweep_values.txt`, which is written for sweep members only and
belongs to the `add-sweeps` change.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `config-and-cli`: add a requirement that a run's `config/` directory contains the
  documented set of config artifacts, and that the provenance-derived ones are
  omitted rather than reconstructed when a caller supplies an already-resolved
  config with no composition history.

## Impact

- `src/dojo/config_loader/resolver.py` — `ResolutionResult` gains provenance fields.
- `src/dojo/config_loader/conductor.py` — `compose_and_resolve` threads
  `ComposedConfig` through instead of dropping it.
- `src/dojo/training/run.py` — `_write_resolved_config` replaced by the shared
  writer; `execute_train` accepts optional provenance.
- `src/dojo/cli/train.py` — passes the invoked command and provenance through.
- New shared config-artifact writer module, reusable by `dojo eval` / `dojo infer`.
- `src/dojo/cli/render.py:72` already reads `config/resolved.yaml`; unaffected.
- Tests: unit coverage for the writer and for the no-provenance path; integration
  assertion that a fixture run's `config/` holds all five files.
- No new dependencies; no config schema changes.
