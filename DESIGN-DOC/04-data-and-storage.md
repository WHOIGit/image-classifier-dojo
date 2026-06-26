
# 04. Data and Storage

## Purpose

Defines supported dataset backends, the shared sample contract, manifest
schemas, IFCB-specific behavior, the storage resolver, and the
`dojo inspect dataset` command. The `data:` and `storage:` config blocks
live at the top of the config tree (see `03-configuration.md`).

## Supported dataset backends

For train / eval / infer:

- `csv_manifest`
- `parquet_manifest`
- `parquet_images` (Parquet with inlined image bytes)
- `ifcb_bins`

`class_folder` is an **`dojo inspect dataset` source only** — it is not a
train / eval / infer backend. Use `dojo inspect dataset` to materialize a
canonical CSV/Parquet manifest from a class-folder layout.

Old listfile dataset formats are not maintained or ported.

WebDataset is deferred — see `appendix-deferred-features.md`.

## Shared sample contract

Every sample carries:

- `sample_id` (stable per-row identifier)
- `uri` (resolved image URI)
- `split` (`train`, `val`, `test`, `unlabeled`, `holdout`)
- optional `bin_id` / `bin_uri` for IFCB-derived samples
- target columns (one or more) per the head/objective configuration
- optional tabular feature columns
- optional source-extra passthrough columns

## CSV / Parquet manifests

Example:

```yaml
data:
  backend: parquet_manifest
  manifest_uri: s3://datasets/ifcb/species_manifest.parquet
  sample_id_column: roi_id
  image_uri_column: image_uri
  split_column: split
  source_extra_columns: [cruise_id, cast_id, instrument_id]
  tabular_feature_columns:
    - depth_m
    - temperature_c
    - salinity_psu
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      missing_policy: error
    biovolume:
      column: biovolume_um3
      type: regression
      missing_policy: drop_sample
```

`csv_manifest` mirrors this shape against a CSV file. `parquet_images`
inlines image bytes in the same Parquet file as the manifest — preferred
for test fixtures because it avoids path-resolution combinatorics.

### Targets

`data.targets` declares logical data targets (named keys). Heads
reference these targets by name; objectives bind heads to losses,
metrics, and weights. See `05-models-training-and-heads.md` for the
target → head → objective reference chain. Per-target fields:

- `column` — physical manifest column.
- `type` — target type (`multiclass_classification`, `regression`,
  `ordinal_classification`, etc.).
- `class_names` — optional URI to a class-label JSON.
- `missing_policy` — `error` (default), `drop_sample`, etc.
- `transform` — optional target transform for regression / ordinal targets
  (e.g. `log1p`, `log1p_standardize`). Authored as the functional form;
  resolved configs add any fitted statistics (e.g. standardize mean / std)
  frozen at fit time. This is the single home for target transforms —
  objectives carry none.

## IFCB bins

```yaml
data:
  backend: ifcb_bins
  manifest_uri: s3://datasets/ifcb/bin_manifest.parquet
  bin_id_column: bin_id
  bin_uri_column: bin_uri
  split_column: split
  exclude_patterns: [bad, skip, beads, temp, data_temp]
  shuffle_buffer_size: 1000
  length:
    mode: cached
    cache_uri: s3://datasets/ifcb/cache/bin_lengths.parquet
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      missing_policy: error
```

IFCB-specific behavior to preserve or port through `ifcbkit`:

- blacklist / exclude filtering;
- old-schema handling;
- shuffle buffer;
- stable ROI IDs;
- optional estimated or cached length.

Dojo does not reimplement IFCB raw-bin parsing — `ifcbkit` is the
canonical interface.

## Storage

Top-level `storage:` configures the storage resolver (peer to `runtime`,
**not** under `runtime`). Storage uses `amplify-storage-utils` underneath.
Dataset / training / export code should not leak `amplify-storage-utils`
APIs directly — go through the Dojo storage interface.

Minimal example:

```yaml
storage:
  local_cache_dir: ./.cache/dojo
```

S3 capability comes through the base `amplify-storage-utils` dependency
(installed as a git reference). There is no separate `s3` optional extra
in the current `pyproject.toml`; see `11-dependencies.md`.

## `dojo inspect dataset`

Read-only by default. May write canonical CSV / Parquet manifests when an
output path is explicitly configured. Inspect outputs do not require a
run directory.

Behavior:

- Resolves the configured backend.
- For `class_folder` source: scans the directory tree and produces a
  canonical manifest.
- Reports missing targets and summarizes how many samples would be
  dropped, skipped, or fail validation.
- Default missing-target policy is `error` for all heads / head counts.
- Optionally writes a CSV or Parquet manifest:

```bash
dojo inspect dataset \
  data=ifcb/species_manifest \
  output=./inspect_outputs/species_manifest.parquet
```

The full aspect-flag surface (`--stats`, `--normalization`,
`--bucket-histogram`, `--bit-depth`, …) is documented in
`02-cli-and-task-types.md`.

### Dataset stats cache

Several resolved values are dataset-derived statistics computed once and
frozen: image normalization mean / std (`normalize: {mode: dataset}`),
tabular normalization stats and imputation fill values, fitted target
transform statistics, per-class counts, the resolved class map, the
resolved `input_bit_depth`, and `ifcb_bins` bin lengths. `dojo inspect
dataset --stats[=URI]` is the **producer**: it computes the fit statistics
on the `train` split (structural properties such as bit depth and bin
lengths across all splits) and writes them to a stats cache.

The cache is keyed by `dataset_hash` so it auto-invalidates when the
underlying data changes. Config resolution **consumes** it: `normalize:
{mode: dataset}`, tabular imputation / normalization, target transforms, and
`length: {mode: cached}` read their frozen values from the cache instead of
recomputing per run. The `length: {mode: cached, cache_uri}` block under
`ifcb_bins` above is the first instance of this pattern; the stats cache
generalizes it to every dataset-derived frozen value.

Resolved values are materialized into the resolved config and hashed by
content (`06-results-artifacts-and-metadata.md`); the cache is a production
and reuse mechanism, not the hash input. A stale or missing cache is a
performance concern, never a correctness one — resolution recomputes when
the cache is absent.

### Preflight in `dojo train`

Training runs the dataset preflight checks listed in
`12-validation-testing-and-preflight.md` (e.g. `empty_train_classes`,
`non_contiguous_class_indices`, `imbalance_ratio_gt`).

## Cross-References

- `02-cli-and-task-types.md` — `dojo inspect dataset` and related
  commands.
- `03-configuration.md` — placement of `data:` and `storage:` blocks.
- `05-models-training-and-heads.md` — target / head / objective
  reference chain, tabular fusion.
- `06-results-artifacts-and-metadata.md` — `source_extra_json` and
  `tabular_features_json` on `sample_metadata` rows.
- `11-dependencies.md` — `ifcb` optional extra; S3 via the base
  `amplify-storage-utils` dependency.
- `12-validation-testing-and-preflight.md` — preflight checks.
- `appendix-deferred-features.md` — WebDataset.
- `glossary.md` — `split`, `sample_id`, dataset-backend vocabulary.
