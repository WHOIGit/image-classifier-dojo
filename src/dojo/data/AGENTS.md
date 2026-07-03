# src/dojo/data — datasets, sample contract, transforms

## Purpose

Reads CSV/Parquet manifest-backed image datasets into decoded samples and
batches for training, inspection, inference, and holdout eval.

- `contract.py` — `DecodedSample` / `SampleBatch` TypedDicts: the shared sample
  contract (image tensor, primary target index, all target indices, URI, and
  provenance the result writer needs).
- `parquet_images.py` — manifest discovery/split routing for `csv_manifest`,
  `parquet_manifest`, and `parquet_images`.
- `inspect.py` — `dojo inspect dataset` report and stats-cache writer.
- `dataset.py`, `identity.py`, `transforms.py` — dataset assembly, sample
  identity/hashing, and transform application.
- `preflight.py`, `samplers.py` — train-time dataset checks and default /
  bucketed / class-balanced batch sampling.

## Ownership

Owns decoding, splitting, target-label mapping, and collation. Does not own
transform *schema* (in `config_schemas`) nor result columns (in `results`), but
must emit exactly the provenance fields the sample contract declares.

## Local Contracts

- `DecodedSample` / `SampleBatch` are the boundary contract with `training/`,
  `inference/`, and `results/`. Keep them in sync with the `sample_metadata`
  columns in `06-results-artifacts-and-metadata.md`.
- Samples expose `target` as the first configured data target for legacy
  single-target consumers and `targets` as the complete logical target-name to
  class-index mapping for true multi-head / multi-target training and eval.
- Split is data-driven; the split column must be present in the dataset.
- External image manifest URIs are resolved relative to the manifest file/dir
  unless absolute or scheme-qualified.
- `parquet_images` split tables are metadata-only. When no image cache is
  enabled, embedded image bytes must stay in source Parquet files and be read
  lazily in `Dataset.__getitem__` through lightweight row references.
- `data.image_cache.enabled` materializes embedded Parquet images into a
  content-addressed disk cache keyed by `dataset_content_hash`; cached datasets
  must load images from filesystem paths in workers, not from Parquet row
  groups.
- Materialized image cache identity intentionally hashes source Parquet relative
  file names and bytes. `data.image_cache.cache_bust` creates a distinct cache
  directory under the same source identity; `data.image_cache.clobber` deletes
  and rebuilds the selected cache directory.
- Stats-cache writing is explicit: `inspect.py` writes JSON to
  `data.stats_cache_uri` only when requested by the caller.
- Stats caches keep large per-sample arrays out of JSON. Dimension rows are
  written as Parquet sidecars referenced from the JSON cache, and full image
  inspect passes record a separate `dataset_content_hash`.
- Target class mappings and class counts are target-specific. `DataBundle`
  keeps legacy `class_mapping` / `class_counts` for the primary target, while
  new head-aware code must use `class_mapping_by_target` /
  `class_counts_by_target`.
- Target class mappings come from `label_index_column` + `label_name_column`,
  name-only targets, or recognized dataset schema metadata such as HuggingFace
  `ClassLabel` names. Index-only datasets without metadata fall back to index
  strings downstream.
- `aspect_bucket` transforms can produce variable tensor shapes across
  samples; non-training and weighted training loaders must batch within buckets
  when a dataset exposes aspect buckets.
- `resize` directly resizes to the configured `(height, width)`, `letterbox`
  preserves aspect ratio with padding, and `aspect_bucket` selects a bucket
  canvas from native dimensions then directly resizes to that canvas.
- Class-balanced and weighted samplers use train-split class counts for a
  selected head/target; they must not resample validation, inference, or
  holdout-eval datasets.

## Verification

- `tests/unit/data/` (manifest backends, parquet images, split-from-filename,
  target labels, transforms builder, preflight, samplers, identity).
</content>
