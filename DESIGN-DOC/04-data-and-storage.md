
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
  stats_cache_uri: s3://datasets/ifcb/species_manifest.stats.json  # colocated with the manifest; omit for display-only
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

`parquet_images` uses the same `dataset_hash` / `dataset_content_hash`
contract as every other backend. Embedded image bytes are not folded into the
cheap `dataset_hash`; they belong to `dataset_content_hash`. For tabular data,
the available logical tabular column names / physical bindings are part of
`dataset_hash`, while the per-row tabular values are part of
`dataset_content_hash`.

### Tabular features

`data.tabular_feature_columns` declares the available logical tabular
features in the dataset. The simple list form is shorthand for numeric
features where each entry is both the logical feature name and the physical
manifest column:

```yaml
data:
  tabular_feature_columns:
    - depth_m
    - temperature_c
```

P3.6 adds the explicit object form for categorical features. In object form,
the mapping key is the logical feature name, `column` binds it to the physical
manifest column, and `type` is `numeric` or `categorical`:

```yaml
data:
  tabular_feature_columns:
    depth_m:
      column: depth_m
      type: numeric
    instrument:
      column: instrument_id
      type: categorical
```

String-list entries resolve to object entries with `type: numeric`.
Categorical feature values are canonicalized to strings before vocabulary
fitting and inference-time lookup.

This block is dataset schema, not model input order and not preprocessing.
`transforms.tabular` owns tabular preprocessing such as imputation,
normalization, train-only augmentation, categorical encoding, and future
value transforms.
`model.tabular_input.columns` selects the logical tabular features consumed by
a model and fixes tensor order; every selected feature must exist in
`data.tabular_feature_columns`. If `model.tabular_input.columns` is omitted
while tabular input is enabled, it resolves to all declared tabular features in
data order.

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
- optional estimated bin length (exact ROI counts come from the dataset
  stats cache via `--bin-lengths`).

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
  --output ./inspect_outputs/species_manifest.parquet
```

The full aspect-flag surface (`--stats`, `--normalization`,
`--bucket-histogram`, `--bit-depth`, …) is documented in
`02-cli-and-task-types.md`.

### Dataset stats cache

Several resolved values are dataset-derived statistics computed once and
frozen: image normalization mean / std (`normalize: {mode: dataset}`),
numeric tabular normalization stats and imputation fill values, categorical
tabular vocabularies, fitted target transform statistics, per-class counts, the
resolved class map, the
resolved `input_bit_depth`, per-sample native dimensions, and `ifcb_bins`
bin lengths. `dojo inspect dataset --stats` is the **producer**. To
stay cheap it computes only the manifest- and header-level frozen aspects —
per-class counts and class map, fitted target / tabular stats, imputation
fill values, categorical vocabularies, per-sample dimensions,
`input_bit_depth`, and `ifcb_bins` bin lengths — fitting on the `train` split
(structural properties such as bit
depth and bin lengths across all splits) and writing them to a stats cache.
Image normalization mean / std is a **decode-tier** frozen aspect: it is not
part of a bare `--stats` and is produced by `--stats --normalization` (or
`--all`), which reads every image pixel.

The cache location is `data.stats_cache_uri`, configured in the data config
and conventionally **colocated with the manifest**. `dojo inspect dataset`
always **displays** the computed aspects to the terminal (per `--format`)
and additionally writes the frozen aspects to `data.stats_cache_uri` when
that field is set; with no `stats_cache_uri` configured, `--stats` is
display-only and writes nothing.

The cache is keyed by `dataset_hash`. When `dataset_hash_provenance` is
strong (`manifest_content` or `uri_etag`), this auto-invalidates stale
cache entries when the manifest identity changes. When provenance falls
back to `uri_only`, Dojo records that weakness and cannot promise automatic
stale-cache detection; production configs should use content-derived or
etag-derived dataset identity for frozen stats.

Config resolution **consumes** the stats cache; it does not produce or
repair it. `normalize: {mode: dataset}`, tabular imputation /
normalization / categorical vocabularies, target transforms, cached
`ifcb_bins` bin lengths, the count-dependent losses (`weighted_cross_entropy`,
`class_balanced_effective_number`), and the `class_balanced` sampler read
their frozen values from the cache instead of recomputing per run. Every
dataset-derived frozen value — including `ifcb_bins` bin lengths — lives in
this one cache; there is no separate per-aspect cache file.

Resolved values are materialized into the resolved config and hashed by
content (`06-results-artifacts-and-metadata.md`); the cache is a production
and reuse mechanism, not the hash input. Missing required cache aspects,
dataset-hash mismatches, estimated values in non-sampled runs, and weak
`uri_only` provenance where strict freshness is required are configuration
errors with an actionable fix: run `dojo inspect dataset` with the needed
frozen aspect flags, or `--stats`, to write an updated cache before
training / evaluation / inference.

`dataset_hash` itself stays cheap and usually available without reading image
pixels or tabular cell payloads — manifest identity / canonical manifest
projection, or URI + size + etag / last-modified when available — so it
remains a stable identity for the cache key, result rows, and ensemble
compatibility. Because a `--normalization` / `--content-hash` pass already
reads sample content, `dojo inspect dataset` additionally records a separate
**`dataset_content_hash`** (a true hash over image bytes and, when present,
tabular feature values) for integrity / drift verification. It is recorded
alongside, never folded into, `dataset_hash`: the cheap identity must not
change value depending on whether a full pass happened to run.
`dataset_content_hash` catches in-place pixel or tabular-value mutation that
manifest-level identity cannot see.

### Stats cache format

The stats cache is a JSON document per dataset, written by `dojo inspect
dataset --stats` to `data.stats_cache_uri`. Small aggregates live inline; per-sample arrays
(dimensions, bin lengths) are too large for JSON and are written as Parquet
sidecars referenced by URI.

```json
{
  "schema_version": "1.0.0",
  "dataset_hash": "sha256:1a2b…",
  "dataset_hash_provenance": "manifest_content",
  "dataset_content_hash": "sha256:9f8e…",
  "created_by": "dojo inspect dataset",
  "dojo_version": "0.1.0",
  "splits_present": ["train", "val", "test"],
  "aspects": {
    "normalization": {
      "split": "train", "estimated": false,
      "image_mode": "grayscale_repeat3",
      "mean": [0.42, 0.42, 0.42], "std": [0.18, 0.18, 0.18]
    },
    "bit_depth": { "split": "all", "value": 8, "heterogeneous": false },
    "class_counts": {
      "split": "train", "estimated": false,
      "per_head": {
        "species": { "num_classes": 42, "total": 50000,
                     "counts": { "0": 1203, "1": 88, "2": 940 } }
      }
    },
    "target_stats": {
      "split": "train",
      "per_target": {
        "biovolume": { "transform": "log1p_standardize",
                       "fit": { "mean": 2.31, "std": 0.74 } }
      }
    },
    "tabular_stats": {
      "split": "train",
      "per_column": {
        "depth_m": {
          "type": "numeric",
          "mean": 48.2,
          "std": 31.7,
          "impute": 45.0
        },
        "instrument": {
          "type": "categorical",
          "encoding": "one_hot",
          "observed_categories": ["IFCB101", "IFCB102"],
          "vocabulary": ["IFCB101", "IFCB102", "__MISSING__", "__UNKNOWN__"],
          "missing_token": "__MISSING__",
          "unknown_token": "__UNKNOWN__"
        }
      }
    },
    "dimensions": {
      "split": "all",
      "parquet_uri": "stats/dimensions.parquet",
      "columns": ["sample_id", "native_width_px", "native_height_px"]
    },
    "bin_lengths": {
      "split": "all",
      "parquet_uri": "stats/bin_lengths.parquet",
      "columns": ["bin_id", "roi_count"]
    }
  }
}
```

Rules:

- Every aspect records the `split` it was computed on and an `estimated`
  flag (`true` when produced under `--sample`); resolution refuses to freeze
  an `estimated` value into a non-sampled run.
- `dataset_hash` is the cache key. Resolution compares it against the current
  dataset and refuses the cache on mismatch. It does not recompute inside
  config resolution; run `dojo inspect dataset` to refresh the cache.
- Aspects are independent: a cache may hold only the aspects that have been
  run. If a required aspect is missing, resolution fails with a message
  naming the missing aspect and the inspect command that produces it.
- Inline aggregates (`normalization`, `bit_depth`, `class_counts`,
  `target_stats`, `tabular_stats`) are the values hashed by content into the
  compatibility hashes; `tabular_stats` includes both numeric tabular
  statistics and categorical vocabularies. The per-sample Parquet sidecars (`dimensions`,
  `bin_lengths`) are derivation inputs (bucket assignment, sampler lengths),
  not hash inputs.

### Aspect-bucket assignment

When `aspect_bucket` is enabled, every sample's bucket is a deterministic
function of its native dimensions and the resolved bucket scheme. Per-sample
dimensions are the cached primitive (`dojo inspect dataset --dimensions`),
so candidate bucket schemes are evaluated from cache without re-reading
images, and at run setup the resolved `aspect_bucket` assignment is
materialized as a working-manifest column — computed from cached dimensions
× the scheme, with no image I/O at run time. Changing the bucket scheme
re-derives the column from the same cached dimensions; it never requires a
rescan.

Batches must be tensor-stackable (identical H × W), so bucketing requires
batching **within** a bucket. The `batch_aspect_buckets` sampler groups
samples by the `aspect_bucket` column to yield size-homogeneous batches (see
`05-models-training-and-heads.md`). The bucket scheme is part of
`preprocessing_hash` via `inference_pipeline`, so assignments are
reproducible.

### Preflight in `dojo train`

Training runs the dataset preflight checks listed in
`12-validation-testing-and-preflight.md` (e.g. `empty_train_classes`,
`non_contiguous_class_indices`, `imbalance_ratio_gt`).

## Cross-References

- `02-cli-and-task-types.md` — `dojo inspect dataset` and related
  commands.
- `03-configuration.md` — placement of `data:` and `storage:` blocks.
- `05-models-training-and-heads.md` — target / head / objective
  reference chain, tabular input composition.
- `06-results-artifacts-and-metadata.md` — `source_extra_json` and
  `tabular_features_json` on `sample_metadata` rows.
- `11-dependencies.md` — `ifcb` optional extra; S3 via the base
  `amplify-storage-utils` dependency.
- `12-validation-testing-and-preflight.md` — preflight checks.
- `appendix-deferred-features.md` — WebDataset.
- `glossary.md` — `split`, `sample_id`, dataset-backend vocabulary.
