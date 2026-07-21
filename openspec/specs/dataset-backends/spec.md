# Dataset Backends Specification

## Purpose

Dataset backends, the shared sample contract, preflight checks, the
stats cache, the materialized image cache, and `dojo inspect dataset`.
Source design: `DESIGN-DOC/04-data-and-storage.md` (workplan P2.2).

## Requirements

### Requirement: Manifest backends and shared sample contract
The system SHALL support `csv_manifest`, `parquet_manifest`, and
`parquet_images` dataset backends, all producing the shared sample
contract (`sample_id`, `uri`, `split`, targets).

#### Scenario: Multi-target samples
- **WHEN** a dataset declares multiple targets (e.g. `label` and
  `coarse_label`)
- **THEN** `DecodedSample` / `SampleBatch` expose legacy `target` for the
  primary target and a `targets` map from logical target name to
  tensor/index for all configured targets

### Requirement: Lazy parquet_images access
`parquet_images` SHALL NOT materialize embedded image bytes into split
tables; split tables carry hidden row references (`__dojo_file_path`,
`__dojo_row_group`, `__dojo_row_in_group`) and `__getitem__` reads bytes
lazily through a one-row-group reader cache.

#### Scenario: Memory-bounded dataset construction
- **WHEN** split tables are built over a large embedded-image Parquet
  dataset
- **THEN** the embedded image column is excluded from the split tables
  and no `_image_bytes` list is materialized

### Requirement: Materialized image cache
`data.image_cache` for `parquet_images` SHALL materialize embedded image
bytes to image files keyed by a deterministic `dataset_content_hash`
over source Parquet relative file names and bytes, with
`<cache_base>/<hash>/manifest.json` as the completion marker, and SHALL
support `enabled`, `dir`, `progress`, `force_rebuild` (rebuild in
place), `cache_bust` (distinct cache dir under the same identity), and
`clobber` (delete and rebuild) controls.

#### Scenario: Cache reuse
- **WHEN** a complete manifest with the same content hash and file list
  exists
- **THEN** the cache is reused and DataLoader workers open materialized
  files directly instead of reading Parquet row groups

### Requirement: Train-time preflight checks
Train-time dataset preflight SHALL evaluate `empty_train_classes`,
`non_contiguous_class_indices`, and `imbalance_ratio_gt` against the
deterministic pre-training dataset bundle, honoring per-check severity.

#### Scenario: Severity handling
- **WHEN** a check trips with severity `error` / `warn` / `ignore`
- **THEN** `execute_train` stops / a Python warning is emitted / the
  issue is suppressed, respectively

### Requirement: dojo inspect dataset and stats cache
`dojo inspect dataset` SHALL run tiered manifest / header / decode
passes with aspect flags, and SHALL produce a `dataset_hash`-keyed stats
cache freezing normalization, target, class-count, bit-depth, dimension,
and bin-length values consumed at config resolution. Per-sample
dimensions SHALL be written to a Parquet sidecar rather than embedded in
the JSON cache; full image passes SHALL also record a separate
`dataset_content_hash`.

#### Scenario: Deterministic normalization source
- **WHEN** `dojo inspect dataset --stats --normalization` runs over a
  train split
- **THEN** deterministic train-split mean/std are reported for
  hard-coding into experiment configs

#### Scenario: Class mapping visibility
- **WHEN** text output is rendered
- **THEN** per-head class count totals, class index, readable label, and
  train-split count per class are shown (mirrored in JSON under
  `aspects.class_mapping.per_head` / `aspects.class_counts.per_head`)

### Requirement: Class-name derivation from schema metadata
When neither target column carries readable names, the system SHALL
resolve readable class mappings from recognized dataset schema metadata
(e.g. HuggingFace `ClassLabel` `names`) as an optional enrichment,
falling back to index-string labels when absent.

#### Scenario: Index-only dataset without metadata
- **WHEN** an index-only dataset has no recognized schema metadata
- **THEN** downstream consumers use index-string labels
