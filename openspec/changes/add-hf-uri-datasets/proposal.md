## Why

Relates to `DESIGN-DOC/04-data-and-storage.md` (workplan P2.2 dataset
backends); net-new enhancement, not a numbered workplan item.

Today `data.manifest_uri` is resolved by `LocalStorage.localize()`, which
wraps the string as a local filesystem `Path`; the loader then does
`root.exists()` / `root.glob(...)` (`src/dojo/data/parquet_images.py`).
A remote reference such as `hf://datasets/<org>/<name>` or a HuggingFace
repo id fails with `data.manifest_uri does not exist`. Users must
manually download datasets first.

The storage interface already anticipates this: `get_storage()` and the
`Storage.localize()` seam are documented as the slot for "the s3 /
cache-aware backend" (`src/dojo/storage/io.py`). Native Hub resolution
lets configs reference published datasets directly and makes the
class-name-from-schema-metadata path (HuggingFace `ClassLabel`) trivially
demonstrable, since Hub parquet retains its embedded `huggingface`
metadata.

## What Changes

- Resolve `hf://datasets/<org>/<name>[@revision][/subpath]` (and the
  bare `<org>/<name>` repo-id form) HuggingFace Hub dataset URIs to a
  local materialized path behind `Storage.localize()`, downloading via
  the `huggingface_hub` cache and reusing it on subsequent runs.
- Preserve embedded Parquet/Arrow key-value metadata (notably the
  `huggingface` / `features` `ClassLabel` block) through resolution so
  class-name derivation from schema metadata works on Hub datasets.
- Pin resolution to an explicit `@revision` when supplied; record the
  resolved revision so `dataset_hash` stays reproducible.
- Keep local paths unchanged; remote schemes are opt-in by URI form.
- `huggingface_hub` gated behind the base `datasets` dependency (already
  present) or a small extra; a clear error names it if absent.

## Capabilities

### Modified Capabilities

- `dataset-backends`: `data.manifest_uri` accepts HuggingFace Hub URIs in
  addition to local paths.
- `storage-and-outputs`: the storage resolver localizes remote dataset
  URIs to a cached local path.

## Impact

- `src/dojo/storage/io.py` (`localize` / a Hub-aware resolver),
  `src/dojo/data/parquet_images.py` and the manifest backends (no change
  expected once localization returns a local path), config docs.
- New optional network dependency at dataset-resolution time; offline
  runs against already-cached datasets must still work.
