# Config and CLI Specification

## Purpose

Hydra-composed, Pydantic-validated configuration and the `dojo` CLI
surface for the supervised platform. Source design: `DESIGN-DOC/02-cli-and-task-types.md`,
`DESIGN-DOC/03-configuration.md` (workplan P1, P2.1).

## Requirements

### Requirement: Composed and validated config tree
The system SHALL compose run configuration through the Hydra Compose API
from packaged defaults in `src/dojo/config_defaults/` with a project-root
`./configs` shadow layer for local overrides, and SHALL validate the
composed config against a strict Pydantic root schema covering
`experiment`, `task`, `runtime`, `storage`, `data`, `model`,
`objectives`, `transforms`, `training`, `output_root`, and
`training_outputs`.

#### Scenario: Invalid or deferred config keys fail validation
- **WHEN** a config sets a key outside the strict schema (including any
  deferred-feature token not yet promoted into active work)
- **THEN** validation fails with a Pydantic error; there are no
  `NotImplementedError` stubs or reserved slots

#### Scenario: Packaged defaults with local shadow
- **WHEN** a selector resolves in both the packaged tree and `./configs`
- **THEN** the project-root `./configs` entry wins

### Requirement: dojo inspect config
`dojo inspect config` SHALL compose and validate the config, render the
resolved output paths, and report enabled outputs, in text and JSON
formats.

#### Scenario: Valid experiment config
- **WHEN** `dojo inspect config --format json` runs on a valid experiment
- **THEN** it exits zero and emits resolved paths and enabled outputs

### Requirement: dojo inspect config-compare
`dojo inspect config-compare --config-a --config-b` SHALL diff two
authored or resolved config YAMLs, organizing the report by hash-source
inputs (`config_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, per-head `head_hash`) plus an
`other` section, in text and JSON formats.

#### Scenario: Authored inputs
- **WHEN** authored (unresolved) YAMLs are compared
- **THEN** they are composed and validated without rendering runtime
  output paths, and class-mapping-dependent sections use index-string
  fallback labels while reporting that assumption

### Requirement: dojo init
`dojo init` SHALL materialize packaged configs and optional fixture data
into a local editable project.

#### Scenario: Fresh project
- **WHEN** `dojo init` runs in an empty directory
- **THEN** an editable `./configs` tree is created from the packaged
  defaults

### Requirement: CLI command groups
The `dojo` CLI SHALL expose `train`, `infer` (`predictions`,
`embeddings`), `eval` (`holdout`), `init`, and `inspect` (`config`,
`config-compare`, `dataset`, `backbone`, `checkpoint`) command groups via
Typer.

#### Scenario: Inspect backbone and checkpoint
- **WHEN** `dojo inspect backbone` or `dojo inspect checkpoint` runs
- **THEN** the backbone/checkpoint metadata is reported without starting
  a training run

### Requirement: Local logger sink
The composite logger abstraction SHALL support the `local` sink; Aim and
MLflow are deferred and SHALL NOT be registered sink types.

#### Scenario: Deferred sink requested
- **WHEN** a config requests an `aim` or `mlflow` sink
- **THEN** config validation fails

### Requirement: Run config artifacts
Every run SHALL write its configuration artifacts into `config/` under its
resolved output directory. The full set is `composed.yaml` (the config after
Hydra composition and CLI overrides, before runtime and path resolution),
`resolved.yaml` and `resolved.json` (the fully resolved config, including
generated runtime values such as `run_id`, in YAML and JSON), `cli.txt` (the
invoked command), and `overrides.txt` (the applied overrides, one per line).

`resolved.yaml` and `resolved.json` SHALL always be written. `composed.yaml`,
`cli.txt`, and `overrides.txt` derive from composition provenance and SHALL be
written only when the caller supplies it; they SHALL NOT be reconstructed from
the resolved config.

#### Scenario: Run invoked through the CLI
- **WHEN** `dojo train` composes a config and completes a run
- **THEN** the run's `config/` directory contains `composed.yaml`,
  `resolved.yaml`, `resolved.json`, `cli.txt`, and `overrides.txt`

#### Scenario: Run invoked without composition provenance
- **WHEN** a caller runs training from an already-built `RootConfig` rather
  than through config composition
- **THEN** `config/resolved.yaml` and `config/resolved.json` are written and
  `composed.yaml`, `cli.txt`, and `overrides.txt` are absent

#### Scenario: Overrides recorded verbatim
- **WHEN** a run is invoked with dotted overrides such as
  `training.batch_size=8`
- **THEN** `overrides.txt` lists each override as given, and `composed.yaml`
  reflects them while omitting resolver-generated runtime values

#### Scenario: Resolved artifacts agree
- **WHEN** a run writes `resolved.yaml` and `resolved.json`
- **THEN** both parse to the same configuration, and re-running via
  `dojo train --resolved-config` against either reproduces the same config

### Requirement: Composition provenance carried through resolution
Config composition SHALL surface the composed config, the applied overrides,
and the invoked command alongside the resolved config, so that callers can
persist run config artifacts without recomposing.

#### Scenario: Resolution result exposes provenance
- **WHEN** a config is composed and resolved in one call
- **THEN** the result carries the resolved config plus the composed config and
  the overrides that produced it
