## ADDED Requirements

### Requirement: HuggingFace Hub dataset URIs
`data.manifest_uri` SHALL accept HuggingFace Hub dataset references — the
`hf://datasets/<org>/<name>[@revision][/subpath]` scheme and the bare
`<org>/<name>` repo-id form — in addition to local filesystem paths.
Remote references SHALL be resolved to a locally materialized path
(downloaded and cached via `huggingface_hub`) before backend file
discovery, and the resolved local path SHALL feed the existing
`csv_manifest` / `parquet_manifest` / `parquet_images` loaders unchanged.
Local paths SHALL continue to resolve exactly as before.

#### Scenario: parquet_images from a Hub dataset
- **WHEN** a `parquet_images` config sets
  `manifest_uri: hf://datasets/<org>/<name>` and the extra is installed
- **THEN** the dataset parquet is downloaded to the Hub cache, discovered
  as if local, and training/inspection proceed normally

#### Scenario: Cached reuse and offline runs
- **WHEN** a Hub dataset has already been resolved into the local cache
- **THEN** a subsequent run reuses the cached files without re-downloading
  and succeeds without network access

#### Scenario: Missing Hub dependency
- **WHEN** a Hub URI is configured but `huggingface_hub` is not installed
- **THEN** resolution fails with a clear error naming the missing
  dependency

### Requirement: Class names from Hub schema metadata
Class-name derivation from schema metadata SHALL work on Hub-resolved
datasets: the embedded Parquet/Arrow `huggingface` / `features`
`ClassLabel` block SHALL be preserved through resolution so an
index-only target (no `label_name_column`) resolves readable class names
from it.

#### Scenario: Index-only Hub target
- **WHEN** an index-only target points at a Hub dataset whose parquet
  carries a `ClassLabel` feature for the index column
- **THEN** `dojo inspect dataset` reports readable class names derived
  from that metadata

### Requirement: Reproducible Hub revision pinning
When a Hub URI supplies an explicit `@revision`, resolution SHALL pin to
that revision; the resolved revision SHALL be recorded so `dataset_hash`
remains reproducible across runs.

#### Scenario: Pinned revision
- **WHEN** two runs use the same `hf://datasets/<org>/<name>@<rev>` URI
- **THEN** both resolve the same files and produce the same `dataset_hash`
