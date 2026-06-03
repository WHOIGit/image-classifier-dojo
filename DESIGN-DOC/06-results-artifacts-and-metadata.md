
# 06. Results, Artifacts, and Metadata

## Purpose

Defines the canonical result schemas, identifiers and hashes, sidecar
`_metadata.json` shape, on-disk artifact layout under each `*_outputs.dir`,
result partitioning, and logging/diagnostics behavior. This file is the
authoritative reference for everything that gets written to disk other
than checkpoints.

## Results backend

- Dojo owns canonical result schemas and `_metadata.json`.
- Result writing uses `amplify-db-utils` for partitioned DuckDB / Parquet
  output, schema registration, filtered reads, bulk reads, and
  object-store-compatible paths.
- Explicit `pyarrow.Schema` definitions back Dojo result tables,
  especially for Arrow list / vector columns.
- Only per-sample records belong in result Parquet. Per-class metrics,
  run-level metrics, and plots belong in `metrics/` and `figures/`.

### Result writer config

```yaml
training_outputs:
  results:
    enabled: true
    backend: amplify_db_utils
    dir: results            # relative to training_outputs.dir
    format: parquet
    partition_by: [stage, record_type, epoch]
    dictionary_encode:
      enabled: true
      columns:
        - split
        - stage
        - record_type
        - head_name
        - embedding_kind
        - prediction_label
        - resize_width_px
        - resize_height_px
    write_metadata_json: true
```

Defaults: `training_outputs.results.dir` → `<training_outputs.dir>/results/`.
`ensemble_outputs.results.dir` → `<ensemble_outputs.dir>/ensemble_results/`.

### Arrow list columns

Vector columns:

```text
embedding
logits
probabilities
ordinal_logits
```

### Dictionary-encoded columns

```text
split
stage
record_type
head_name
embedding_kind
prediction_label
resize_width_px
resize_height_px
```

## IDs and hashes

Identity and content-equality are tracked separately:

- A **`*_hash`** is derived deterministically from object content.
- A **`*_id`** is either manually set or generated as a **seedname**: a
  coolname produced by seeding `random.Random(*_hash)`. Same hash → same
  seedname.

Explicit exceptions:

- `run_id` may be generated from a fresh, unseeded `{coolname}` template
  token. There is no `run_hash`.
- `dataset_id` is only set when the dataset self-names; no seedname
  fallback.
- `sweep_id` may be a template render like `run_id`, or fall back to
  seedname from `sweep_hash` when neither is configured.

Full hashes are recorded by default. The only standard truncation is the
first 6 hex characters embedded in checkpoint filenames.

### Canonicalizer

Lives in `dojo.utils.artifact_hashing`. One JSON normalization recipe
used by every hash function:

- canonical UTF-8 JSON;
- sorted object keys;
- no insignificant whitespace;
- lists preserved in source order;
- floats rounded to **12 significant decimal digits** before
  serialization (well within float64 precision of ~15.95, absorbs
  roundtrip noise);
- `NaN`, `+Infinity`, `-Infinity` serialize as the literal strings
  `"NaN"`, `"Infinity"`, `"-Infinity"`;
- bytes/binary inputs hashed directly without JSON wrapping.

Each `*_hash` function selects a specific subset of fields from the
source object before canonicalization.

### Per-field table

| Field | Kind | Derivation | Default when not set |
| --- | --- | --- | --- |
| `run_id` | id only | manual, else template render. `{coolname}` expands to fresh unseeded coolname; falls back to coolname when unset | template like `{experiment.name}-{timestamp}-{job_num}` or `{coolname}` |
| `config_id` | id (paired with `config_hash`) | manual, else seedname from `config_hash` | seedname |
| `config_hash` | hash | canonical hash of resolved config, **excluding** runtime-resolved values, output paths, `output_root`, and all `*_outputs` blocks | always derived |
| `dataset_id` | id only | the dataset's self-name when the manifest provides one | null (no seedname fallback) |
| `dataset_hash` | hash | URI + size + etag/last-modified (or full content hash when locally cheap), plus backend type. When size/etag unavailable, falls back to URI-only with `dataset_hash_provenance: uri_only` in metadata | always derived |
| `checkpoint_hash` | hash | SHA-256 of the `.ckpt` file bytes | always derived |
| `model_id` | id (paired with `model_hash`) | manual on export, else seedname from `model_hash` | seedname |
| `model_hash` | hash | SHA-256 of the exported `.pt` / `.onnx` file bytes | always derived |
| `ensemble_id` | id (paired with `ensemble_hash`) | manual on ensemble, else seedname from `ensemble_hash` | seedname |
| `ensemble_hash` | hash | canonical hash of ensemble manifest JSON (members + combine + selection) | always derived |
| `sweep_id` | id (paired with `sweep_hash`) | manual, else template render (e.g. `{coolname}`); falls back to seedname from `sweep_hash` when unset | seedname or `{coolname}` |
| `sweep_hash` | hash | canonical hash of sweep definition (base config + sweep axes + value lists), excluding runtime-resolved values and output paths | always derived |
| `ensemble_member_id` | union column | for member-level rows / partitioning; member's `checkpoint_hash` (checkpoint member) or `model_id` (exported model member) | derived per-row |

There is no `run_hash`. There is no `checkpoint_id`; checkpoints are
identified by `checkpoint_hash` plus their filename.

### Checkpoint filename convention

```text
{stem}.{first6_hex}.{ext}
```

Examples:

```text
loss-1.23_epoch-003_f1score-88.7ff91a.ckpt
last.a1b2c3.ckpt
snapshot_cycle-02_epoch-100.0f0f0f.ckpt
```

### Compatibility hashes

For `target_schema_hash`, `class_mapping_hash`, `model_config_hash`, and
`preprocessing_hash`, both the hash and the source sub-block are stored
in `_metadata.json`. Fast path: compare hashes. Slow path: when hashes
differ, diff source sub-blocks and present a human-readable mismatch
report.

Key-selection narrative rules (binding intent; **exact field lists TBD**
and tracked next to the Pydantic schemas via something like a
`compatibility_hash_includes=True` field flag in
`dojo.utils.artifact_hashing`):

- `target_schema_hash`: head names, head types, `num_classes`, ordinal
  encoding/decoding rules, regression `output_dim`. **Excludes** loss
  type/params, objective weights, metrics.
- `class_mapping_hash`: per-classification-head ordered list of
  `(index, label)` pairs.
- `model_config_hash`: backbone source/name/weights, freeze policy,
  embedding adapter shape, tabular encoder shape, fusion config, head
  shapes. **Excludes** optimizer, scheduler, training, logging,
  `output_root`, `*_outputs` blocks.
- `preprocessing_hash`: transform pipeline ordering and parameters,
  image mode, normalization mean/std, resize/bucket definitions, tabular
  feature normalization stats. **Excludes** training-only augmentation
  toggles unless they alter the inference-time preprocess contract.

> **Open item.** Freezing the exact field lists is deferred until just
> before implementation. The narrative rules above are binding intent;
> implementers mark contributing fields in the Pydantic schemas. Keeping
> the field list next to schema definitions is preferred over freezing it
> in static documentation.

## Result rows

### Common provenance columns

All supervised / representation-evaluation result records share:

```text
sample_id
uri
split
stage
record_type
run_id
config_id
config_hash
dataset_id
dataset_hash
epoch
global_step
checkpoint_hash
model_id
model_hash
sweep_id            # when produced as part of a sweep
sweep_hash
schema_version
```

`epoch` and `global_step` may be null for static sample metadata.
`checkpoint_hash` identifies the producing checkpoint. `model_id` /
`model_hash` are populated when the row was produced by an exported model
artifact.

For representation-evaluation rows, add `evaluation_name` to identify the
specific probe / retrieval / clustering / projection / diagnostic pass.

### `split` vs. `stage`

- `split` — source dataset split (`train`, `val`, `test`,
  `unlabeled`, `holdout`).
- `stage` — process that produced the row
  (`train_validation`, `holdout_eval`, `infer`,
  `representation_eval`, `ensemble_eval`).

### `record_type` values

Supervised:

```text
sample_metadata
embedding
classification_output
regression_output
ordinal_output
```

Representation-evaluation:

```text
embedding
nearest_neighbor
knn_prediction
classification_probe_prediction
regression_probe_prediction
ordinal_probe_prediction
cluster_assignment
projection
outlier_score
diagnostic
```

### `sample_metadata` columns

First-class sample metadata columns:

```text
native_width_px
native_height_px
resize_width_px
resize_height_px
microns_per_pixel
resize_bucket
bin_id
bin_uri
roi_number
```

Plus optional single-column JSON blobs:

- `source_extra_json` — present only when `data.source_extra_columns` is
  configured; only appears on `record_type=sample_metadata`.
- `tabular_features_json` — single JSON-string column for configured
  tabular features.

### `embedding` columns

```text
embedding_kind = image_embedding | tabular_embedding | fused_embedding | head_input_embedding
embedding
embedding_dim
embedding_model_name        # representation-eval only
```

Default `embedding_kind`:

- supervised → `head_input_embedding`
- SSL → `image_embedding`

### Classification output

```text
head_name
target
prediction_index
prediction_label
prediction_confidence
logits
probabilities
```

### Regression output

External vs. internal columns:

- `target` and `prediction_value` are external (original units).
  `prediction_value` is the post-inverse-transform model output.
- `target_internal` and `prediction_value_internal` are model-space
  values (post `target_transform`).
- When no `target_transform` is configured, writers may either populate
  both columns identically or leave the `_internal` columns null; the
  behavior is recorded in `_metadata.json`.
- `prediction_uncertainty` is in external units by default; if the head
  defines uncertainty in transformed space, a parallel
  `prediction_uncertainty_internal` column may be added.

Columns:

```text
head_name
target
prediction_value
target_internal
prediction_value_internal
prediction_uncertainty
```

### Ordinal output

```text
head_name
target
prediction_index
prediction_label
prediction_confidence
ordinal_logits         # raw cumulative logits for CORAL/CORN
probabilities          # per-bin probabilities, derived if needed
```

For CORAL/CORN, `probabilities` are derived by differencing cumulative
probabilities decoded from `ordinal_logits`. The writer is responsible for
this derivation so consumers always see per-bin probabilities in the
`probabilities` column regardless of loss family.

Native ordinal heads write `record_type=ordinal_output`. Ordinal probes
write `record_type=ordinal_probe_prediction`.

### Representation-evaluation record specifics

- `nearest_neighbor`: `query_sample_id`, `neighbor_sample_id`,
  `neighbor_rank`, `distance`, `distance_metric`, `neighbor_label`
  (nullable when unlabeled).
- `knn_prediction`: classification-style head columns plus `k` and
  `distance_metric`.
- Probe predictions: same column shape as the matching native output
  record. `head_name` should identify the probe (e.g. `linear_probe_species`).
  `probe_model_type` documents the probe class (e.g. `ridge`,
  `linear_classifier`, `ordinal_logistic_regression`).
- `cluster_assignment`: `cluster_method`, `cluster_id`,
  `cluster_distance`, `cluster_probability` (optional, soft assignments
  only).
- `projection`: `projection_method`, `projection_x`, `projection_y`,
  `projection_z` (null for 2D).
- `outlier_score`: `outlier_method`, `outlier_score`, `outlier_rank`,
  `is_outlier` (optional).
- `diagnostic`: `diagnostic_name`, `diagnostic_value`,
  `diagnostic_scope` (`sample`, `batch`, `epoch`, `dataset`).
  Aggregate diagnostics may leave `sample_id` null.

## `_metadata.json` sidecar

Structure: top-level `record_types` keys (not a top-level `heads` key).
Stores semantic / provenance metadata; **not** low-level Parquet encoding
details. Per-record-type sections name `target_transform`, class mappings,
and head mappings.

```json
{
  "schema_name": "dojo.supervised_results",
  "schema_version": "1.0.0",
  "created_by": "dojo",
  "run_id": "ifcb-green-river",
  "record_types": {
    "sample_metadata": {
      "description": "One row per evaluated sample.",
      "source_extra_json": {
        "columns": ["cruise_id", "cast_id", "instrument_id"]
      },
      "tabular_features_json": {
        "columns": ["depth_m", "temperature_c", "salinity_psu"]
      }
    },
    "classification_output": {
      "heads": {
        "species": {
          "target": "species",
          "labels": ["A", "B", "C"],
          "label_mappings": {"0": "A", "1": "B", "2": "C"}
        }
      }
    },
    "regression_output": {
      "heads": {
        "biovolume": {
          "target": "biovolume",
          "target_transform": "log1p"
        }
      }
    }
  }
}
```

The sidecar should also record compatibility-hash sub-blocks so consumers
can diff source content on mismatch.

## Result partitioning

`training_outputs.results.partition_by` and
`ensemble_outputs.results.partition_by` are configurable lists. Partition
field options include:

```text
stage
epoch
ensemble_member_id
record_type
sweep_id
```

Include `sweep_id` automatically when a row is produced as part of a Hydra
sweep. `ensemble_member_id` is a union column whose value is the
member's `checkpoint_hash` (checkpoint members) or `model_id` (exported
model members), so a single partition column has no nulls.

Examples:

```yaml
training_outputs:
  results:
    partition_by: [stage, record_type]
```

```yaml
training_outputs:
  results:
    partition_by: [stage, epoch, record_type]
```

```yaml
ensemble_outputs:
  results:
    partition_by: [stage, record_type, ensemble_member_id]
```

> **Open item.** Result partitioning interaction with ensemble result rows
> (which rows are `stage=ensemble_eval` vs. `stage=train_validation` when
> training and ensemble outputs share a directory) is under-specified. The
> intent is that writers namespace by `stage`, but the exact partitioning
> of member-level vs. ensemble-level rows is not yet pinned down. See
> `08-ensembles.md`.

## Artifact layout

Each `*_outputs` block owns a stable set of sub-directories under its
resolved `dir`. Whether each is written depends on the corresponding
sub-block being enabled in config.

- `metrics/` — numeric aggregates only: per-split metric JSON and
  confusion-matrix data (JSON or CSV). No rendered images.
- `figures/` — **configurable output block**, not just a directory.
  Config specifies which kinds of plots to render (training curves,
  confusion-matrix heatmaps, UMAP/t-SNE/PCA scatter, retrieval panels,
  calibration, ensemble comparison) and plot styling. Output files are
  PNG / SVG / HTML.
- There is no generic training-run `data/` folder; input-dataset
  information lives in configs and resolved-config artifacts.
- Ensemble candidate manifests live under
  `ensemble_outputs.manifests.dir`, default
  `{ensemble_outputs.dir}/ensemble_manifests`. Manifests are JSON, not
  Parquet.

Per-block layouts:

```text
training_outputs.dir/
  config/
  checkpoints/
  exports/
  metrics/
  figures/
  results/

ensemble_outputs.dir/
  config/
  exports/
  metrics/
  ensemble_figures/
  ensemble_results/
  ensemble_manifests/
  ensemble_members/         # optional; only when ensemble materializes members locally

sweep_outputs.dir/
  config/
  exports/
  metrics/
  figures/
```

For `task.type: snapshot_ensemble` with
`ensemble_outputs.dir_template` defaulted to the training value, both
blocks resolve to the same path and the directory holds the union of both
layouts. Writers namespace files within shared sub-directories — for
example, result rows include `stage=train_validation` vs.
`stage=ensemble_eval` partitions, and metrics files include the
producing block in their filenames.

`ensemble_members/` is only used when `dojo ensemble` materializes
member artifacts locally. Local member files may be symlinked; remote
member files may be cached through `storage` config and then symlinked.

## Logging and diagnostics

Experiment logging applies only to model-training runs. `dojo ensemble`
and `dojo ensemble candidates` do not initialize experiment logging.

Sinks:

- `local` — functional. **The only functional sink for the foreseeable
  future**; metrics and figures are recorded locally.
- `aim` — schema present; **runtime stubbed**. See
  `appendix-deferred-features.md`.
- `mlflow` — schema present; **runtime stubbed** in the initial
  implementation. See `appendix-deferred-features.md`.

Multi-sink composition is supported via `CompositeExperimentLogger`, but
`local` is the only functional sink, so functional configurations use
`local` alone. There is no artificial cap on sink count. Any
configuration involving `aim` or `mlflow` inherits their stubbed runtime
per `12-validation-testing-and-preflight.md`.

`logging` lives under `training_outputs.logging`:

```yaml
training_outputs:
  logging:
    sinks:
      - type: local
```

The logger abstraction:

```python
class ExperimentLogger:
    def log_config(self, cfg) -> None: ...
    def log_metrics(self, metrics, step=None) -> None: ...
    def log_artifact(self, local_path, artifact_path) -> None: ...
    def close(self) -> None: ...
```

Artifacts are produced according to result / export / checkpoint
configuration; logger sinks may register or upload them after local
creation. Logger sinks should play nicely with Lightning's logging
system.

## Cross-References

- `03-configuration.md` — `output_root` and `*_outputs` resolution,
  `results:` / `metrics:` / `figures:` / `export:` sub-block placement.
- `02-cli-and-task-types.md` — which commands write what kinds of
  results.
- `04-data-and-storage.md` — `source_extra_columns` and tabular feature
  configuration drive the sidecar contents.
- `05-models-training-and-heads.md` — heads / objectives / target
  transforms inform record types and sidecar `record_types` blocks.
- `07-ssl-and-representation-eval.md` — representation-evaluation
  record types and `evaluation_name`.
- `08-ensembles.md` — `ensemble_member_id`, `stage=ensemble_eval`
  rows, namespacing in shared directories, manifest JSON files.
- `09-sweeps-and-batch-runs.md` — `sweep_id` / `sweep_hash` provenance
  columns and `sweep_outputs/` layout.
- `10-export.md` — `exports/` sub-directory and export metadata.
- `12-validation-testing-and-preflight.md` — testing policy for the
  stubbed Aim and MLflow sinks.
- `appendix-deferred-features.md` — Aim and MLflow runtime stubs; HDF /
  `.h5` result exports.
- `glossary.md` — `split`, `stage`, `record_type`, `embedding_kind`,
  `head_name`, identifier / hash vocabulary.
