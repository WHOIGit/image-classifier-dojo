# Storage and Outputs Specification

## Purpose

The storage interface and output-path resolution. Source design:
`DESIGN-DOC/04-data-and-storage.md`,
`DESIGN-DOC/06-results-artifacts-and-metadata.md` (workplan P1).

## Requirements

### Requirement: Storage resolver
Storage access SHALL go through the Dojo storage interface backed by
`amplify-storage-utils`.

#### Scenario: Local filesystem run
- **WHEN** a run uses local paths
- **THEN** reads and writes resolve through the storage interface

### Requirement: Output path resolution
`output_root` and `dir_template` SHALL resolve to a deterministic run
directory; `dojo inspect config` renders the resolved paths without
side effects.

#### Scenario: Rendered paths match run
- **WHEN** `dojo inspect config` renders paths and `dojo train` then
  runs the same config
- **THEN** the run writes into the rendered directory layout
