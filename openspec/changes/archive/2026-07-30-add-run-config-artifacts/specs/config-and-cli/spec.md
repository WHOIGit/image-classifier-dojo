## ADDED Requirements

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
