
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
        - ensemble_result_scope
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
ensemble_result_scope
head_name
embedding_kind
prediction_label
resize_width_px
resize_height_px
```

## IDs and hashes

Identity and content-equality are tracked separately:

- A **`*_hash`** is derived deterministically from object content.
- A **`*_id`** is either manually set or generated from an explicit coolname
  template. Hash-paired IDs use hash-seeded coolnames by default.

Coolname template forms:

| Form | Seed behavior | Deterministic? | Intended use |
| --- | --- | --- | --- |
| `{coolname:<source>}` | Seed from the named resolved value, usually a hash such as `config_hash`, `model_hash`, `ensemble_hash`, or `sweep_hash` | yes | hash-derived IDs |
| `{coolname}` | Seed from resolved `runtime.seed` | yes | rare/debug; collision-prone for run IDs |
| `{coolname:noseed}` | Fresh unseeded coolname | no | invocation IDs such as `run_id` |

The seed material includes the source label as well as the source value
(for example `model_hash:<hash>`), so identical raw hash strings in different
domains do not imply identical names. A hash-seeded coolname can only be
rendered after that hash has been computed; generated IDs are excluded from
the source hash and do not feed back into it.

Explicit exceptions:

- `run_id` may be generated from a fresh `{coolname:noseed}` template
  token. There is no `run_hash`.
- `dataset_id` is only set when the dataset self-names; no generated
  fallback.
- `sweep_id` may be manually set, rendered from a coolname template, or fall
  back to `{coolname:sweep_hash}` when unset.

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
| `run_id` | id only | manual, else template render — e.g. `{coolname:noseed}` (fresh unseeded) or a composed pattern like `{experiment.name}-{timestamp}-{job_num}` | `{coolname:noseed}` |
| `config_id` | id (paired with `config_hash`) | manual, else `{coolname:config_hash}` | `{coolname:config_hash}` |
| `config_hash` | hash | canonical hash of resolved config, **excluding** runtime-resolved values, output paths, `output_root`, and all `*_outputs` blocks. When a loaded config carries blocks the active command does not use, the hash covers only that command's active block-set (see `02-cli-and-task-types.md`). | always derived |
| `dataset_id` | id only | the dataset's self-name when the manifest provides one | null (no generated fallback) |
| `dataset_hash` | hash | cheap identity (never reads image pixels or tabular cell payloads): manifest identity / canonical manifest projection or URI + size + etag/last-modified, plus backend type and declared logical tabular column names / bindings. Basis recorded in `dataset_hash_provenance` (`manifest_content` / `uri_etag` / `uri_only` fallback). `uri_only` is weak identity and does not reliably auto-invalidate stale stats caches. | always derived |
| `dataset_content_hash` | hash | true hash over sample content: all image bytes plus tabular feature values for declared tabular columns when present; recorded separately when a content pass runs (`dojo inspect dataset --content-hash` / `--normalization`). Integrity / drift verification only — **not** the cache key, identity, or a compatibility hash | derived when a content pass runs, else null |
| `checkpoint_hash` | hash | SHA-256 of the `.ckpt` file bytes | always derived |
| `model_id` | id (paired with `model_hash`) | manual on export, else `{coolname:model_hash}` after export bytes are hashed | `{coolname:model_hash}` |
| `model_hash` | hash | SHA-256 of the exported `.pt` / `.onnx` file bytes | always derived |
| `ensemble_id` | id (paired with `ensemble_hash`) | manual on ensemble, else `{coolname:ensemble_hash}` after selected-ensemble identity is hashed | `{coolname:ensemble_hash}` |
| `ensemble_hash` | hash | canonical hash of the selected-ensemble identity block (selected members + combine + selection). Candidate-audit metadata in the manifest is excluded. | always derived |
| `sweep_id` | id (paired with `sweep_hash`) | manual, else template render; falls back to `{coolname:sweep_hash}` when unset | `{coolname:sweep_hash}` |
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
`preprocessing_hash`, both the hash and the exact canonical source
sub-block are stored in `_metadata.json`. Fast path: compare hashes. Slow
path: when hashes differ, diff source sub-blocks and present a
human-readable mismatch report.

Compatibility hashes are computed from the **resolved config** after
defaults and generated schema values are injected, but before run-local
paths and runtime identifiers matter. The extractor must build a minimal
canonical JSON object containing only the fields listed below. Do not hash
whole config branches by reference.

For prediction-space ensembles, not every compatibility hash is a
cross-member equality requirement. `target_schema_hash` and
`class_mapping_hash` usually must agree for a shared head, while
`model_config_hash` and `preprocessing_hash` primarily validate a member's
own checkpoints, exports, cached rows, and metadata drift. See
`08-ensembles.md`.

What each hash validates:

- `target_schema_hash` validates output-task structure: head names, head
  types, target types, output dimensions, `num_classes`, ordinal
  encoding / decoding, distributional head shape, and target transforms.
  It answers whether result rows, checkpoints, or models are comparable
  for the same prediction-task shape.
- `class_mapping_hash` validates label-index semantics for discrete
  heads. Two models may both output 42 logits, but they are incompatible
  if index `7` names different classes. It answers whether each logit /
  probability position means the same label across artifacts.
- `model_config_hash` validates the architecture contract governing
  checkpoint loadability and the inference-time forward function:
  image-input backbone architecture, tabular-input encoder shape,
  embedding adapter shape, and head network shapes and activations. It
  answers whether checkpoints and exports load under the same model
  definition and compute the same function. It excludes initialization
  (library or checkpoint weights), trainability (freeze policy), and training-only
  regularization (dropout) — none change tensor shapes or inference
  outputs.
- `preprocessing_hash` validates the input contract: image mode,
  bit-depth scaling, resize / bucketing, normalization, foreground crop,
  grayscale handling,
  inference pipeline, tabular feature ordering, encodings, and
  normalization stats. It answers whether the same raw sample would be
  transformed into the same model input tensor.

Common exclusions for all compatibility hashes:

```text
experiment
task
runtime
storage
training
optimizer
scheduler
checkpointing
objectives
ensemble
sweep
ssl
representation_eval
output_root
training_outputs
ensemble_outputs
sweep_outputs
eval_outputs
logging
metrics
figures
export destinations
run_id / sweep_id / config_id
local cache paths
```

`target_schema_hash` source fields:

```text
version: "1"
heads:
  <head_name>:
    type
    target
    target_type                       # from data.targets[<target>].type
    target_transform                  # resolved data.targets[<target>].transform, by value (type + frozen stats), null if absent
    output_dim                        # regression/count/distributional heads
    num_classes                       # classification/ordinal heads
    distribution                      # distributional_regression only (deferred head; see appendix P4.9)
    ordinal:                          # ordinal_classification heads only; from model.heads[<head>].ordinal
      encoding                        # coral | corn | ordinal_cross_entropy
      decoding                        # threshold | expected_rank | argmax
```

The `heads` object is keyed by resolved head name and sorted by key during
canonicalization. `target_schema_hash` excludes loss type, loss params,
objective weights, metric selections, optimizer settings, and checkpoint
monitor fields. Ordinal `encoding` / `decoding` are read from the resolved
head (`model.heads.<head>.ordinal`), not from the objective loss, so the
loss exclusion holds even though the encoding determines output structure.
The target transform is read only from `data.targets.<target>.transform`
(its single home; objectives carry no target transform) and is hashed by
value, including any resolved fit statistics.

`class_mapping_hash` source fields:

```text
version: "1"
heads:
  <head_name>:
    target
    ordered_labels:
      - index: 0
        label: <string>
      - index: 1
        label: <string>
```

Include only heads with discrete classes:
`multiclass_classification`, `binary_classification`,
`multilabel_classification`, and `ordinal_classification` when ordinal
bins have configured class names. `classes` is the resolved ordered class
names for the target — from its `label_name_column`, or upstream dataset
class metadata — recorded in `_metadata.json` alongside `class_mapping`. The
resolved ordered class content is the hash input, not any source URI.

`model_config_hash` source fields:

```text
version: "1"
model:
  image_input:
    backbone:
      architecture:
        source                       # torchvision / timm
        name
        output_dim
        params                       # architecture params that change module shape / forward behavior
  tabular_input:
    enabled
    columns                         # selected logical features in tensor order
    encoder                         # type, input_dim, output_dim, hidden_dims, activation
  embedding_adapter:
    enabled
    type
    hidden_dims
    output_dim
    activation
  heads:
    <head_name>:
      type
      target
      network                        # type, hidden_dims, activation (excludes dropout)
      num_classes
      output_dim
      distribution
```

For checkpoint-initialized models, include only the effective backbone
architecture (the module the checkpoint instantiates), not the
checkpoint-loading plumbing. `weights.source`, `weights.uri`,
`weights.key`, and `weights.strict` govern which upstream weights
initialize the backbone at build time, not the resulting architecture or
inference function; that initialization provenance lives in `config_hash`,
the trained bytes in `checkpoint_hash`, and export artifacts in
`model_hash`.
`model_config_hash` excludes optimizer, scheduler, training loop settings,
objective loss/metric choices, checkpoint save policy, output paths, and
runtime identifiers. It also excludes fields that change neither tensor
shapes nor the inference forward function: `model.image_input.backbone.weights`
(initialization), `training.freeze` (trainability), and `dropout`
(training-only regularization). `activation` is retained because it changes
inference outputs even though it is stateless.

`model.tabular_input.columns` is retained because it maps tabular encoder
input positions to feature semantics; changing the selected features or their
order changes the inference function even when the tensor width is unchanged.
Available-but-unused tabular features from `data.tabular_feature_columns` are
not part of `model_config_hash`.

`preprocessing_hash` source fields:

```text
version: "1"
transforms:
  image_mode
  input_bit_depth                   # resolved integer (auto resolves to 8/12/16); [0,1] scaling divisor
  inference_pipeline                # resolved inference steps, ordered, with parameters
tabular_preprocessing:
  selected_columns                  # resolved model.tabular_input.columns after default expansion
  encodings                         # per selected categorical feature: one_hot policy, frozen vocabulary, tokens, expanded positions
  imputation                        # per-column strategy, frozen fill values, missing-indicator set
  normalization                     # per-column type plus resolved mean/std for mean_std
```

Only inference-time preprocessing belongs in `preprocessing_hash`. Manifest
column-name bindings (`image_uri_column`, `sample_id_column`,
`split_column`) and the storage `backend` are excluded: they locate a
sample, they do not transform it, so they must not make otherwise-identical
input contracts hash differently. Image transform steps come solely from the
resolved `inference_pipeline` (`05-models-training-and-heads.md`), which
excludes `train_only` augmentation by construction; the full training
`pipeline` is not hashed, and normalization, resize / bucket, and
foreground-crop parameters are the parameters of their steps inside
`inference_pipeline`, not separate fields. The `inference_pipeline` uses the
same canonical transform step names accepted in authored configs.

Tabular preprocessing comes from resolved `transforms.tabular` for the
selected model-input features only, excluding train-only
`transforms.tabular.augmentations`. `data.tabular_feature_columns` is dataset
schema: adding an unused available feature does not change the
`preprocessing_hash`. Tabular augmentations affect training behavior and
therefore `config_hash`, but they are not part of the export inference
contract or `preprocessing_hash`. Resolved dataset statistics such as
normalization mean/std, numeric tabular normalization stats, frozen tabular
imputation fill values, and categorical vocabularies are included by value,
not by the URI from which they were loaded. These resolved statistics are
produced by `dojo inspect dataset` and cached (`04-data-and-storage.md`) —
`--stats` for the manifest/header-level stats, `--stats --normalization` for
dataset normalization mean/std — but the hash always uses their resolved
content, never the cache location. The
`imputation` entry captures the per-column fill strategy, the frozen fill
values, and the missing-indicator set. The `normalization` entry captures the
per-column normalization type (`identity` or `mean_std`) and the frozen
train-split `mean` / `std` values for `mean_std`; `identity` carries no fitted
statistics. When `add_missing_indicator` is enabled the resulting encoder
input width is additionally reflected in `model_config_hash` via the resolved
`model.tabular_input.encoder.input_dim` and any downstream resolved
concatenated / adapter dimensions (see `05-models-training-and-heads.md`).
For categorical features, the `encodings` entry captures the resolved
`one_hot` encoding contract by value: canonical feature name, source type,
resolved vocabulary in tensor order, `missing_policy`, `missing_token`,
`unknown_policy`, `unknown_token`, and the expanded tabular input positions.
This ensures exported models and cached-result consumers agree on how raw
category values become numeric model inputs. Adding, removing, or reordering a
category changes `preprocessing_hash`; the resulting input width also changes
`model_config_hash` through the resolved tabular encoder input dimension.

The Pydantic schema should keep these field lists close to the relevant
models, for example with compatibility-hash extractor methods or field
metadata. The documentation above is the execution contract those
extractors must satisfy.

## Portable inference contract

`dojo infer` and `dojo eval` rebuild a model and its exact input pipeline
from the model artifact alone — they do **not** require the producing run's
`resolved.yaml`. `dojo inspect checkpoint` also reads this contract from
`.ckpt` files, but `.ckpt` deserialization requires the Torch stack
(`11-dependencies.md`). Everything these consumers need is bundled as a single
**inference contract** object, defined once and serialized identically by both
artifact kinds:

- **Checkpoints** embed it under a dedicated
  `checkpoint["dojo_inference_contract"]` key, written by the task module's
  `on_save_checkpoint` hook. It is **not** folded into Lightning
  `hyper_parameters`, which stay limited to the constructor configs (see
  `05-models-training-and-heads.md`).
- **Exports** write it as their export metadata (`10-export.md`); the export
  metadata *is* the inference contract.

Contract fields:

```text
schema_version
model_config            # buildable resolved model: backbone architecture, tabular encoder, embedding adapter, head networks
inference_pipeline      # resolved, ordered inference transforms with frozen parameters
preprocessing_stats     # frozen normalization mean/std, resolved input_bit_depth, bucket scheme, tabular normalization/imputation
class_maps              # ordered index -> label per discrete head (resolved label content, not a URI)
target_schema           # head types, targets, num_classes/output_dim, ordinal encoding/decoding, target transforms with frozen fit stats
objective_summary       # objective/head loss and metric metadata needed for holdout eval scoring
compatibility           # target_schema_hash / class_mapping_hash / model_config_hash / preprocessing_hash plus their canonical source sub-blocks
provenance              # config_hash, source run_id, dojo_version
```

It deliberately **excludes** training-dataset identity (`manifest_uri`,
`dataset_hash`, per-class counts): those describe the data a model was
trained on, not what is needed to run it on new data. `dojo inspect
checkpoint` reads this contract to report embedded hashes and head /
preprocessing configuration. `objective_summary` is descriptive and
metric-driving metadata; it is not a compatibility-hash input.

### `objective_summary`

`objective_summary` is the resolved scoring contract for supervised
artifacts. It lets `dojo eval holdout` compute the same loss / metric family
the model was trained and validated with, without requiring an `objectives:`
block in the eval config. It is not used to rebuild the model.

Shape:

```yaml
objective_summary:
  schema_version: 1
  total_loss:
    reduction: weighted_sum
  objectives:
    species:
      head: species
      target: species
      weight: 1.0
      loss:
        type: cross_entropy
        params: {}
      metrics:
        - name: accuracy
          params:
            top_k: 1
          output: accuracy
        - name: f1_macro
          params:
            average: "macro"
          output: f1_macro
        - name: f1_per_class
          params:
            average: null
          output: "f1_per_class/{label}"
```

Rules:

- `objectives` contains the enabled resolved objectives, keyed by objective
  name. Disabled authored objectives are omitted from the portable contract.
- `head` must name a head in `target_schema`; `target` must match that head's
  target. This duplication is a validation guard and makes scorer setup
  straightforward, but `target_schema` remains authoritative for head type,
  output dimensions, ordinal encoding / decoding, and target transforms.
- `loss` is the canonical resolved loss spec: string shorthand is expanded to
  `{type, params}` and any resolved class / sample weighting values needed to
  reproduce evaluation loss are stored by value, not by dataset-stat URI.
- `metrics` is the ordered list of canonical resolved metric specs from the
  metric registry (`05-models-training-and-heads.md`). Aliases are not
  accepted. Registry defaults such as averaging mode, top-k values,
  thresholds, and per-class behavior are materialized under `params`; output /
  logging names are materialized under `output`.
- `weight` is included so holdout eval can report both per-objective losses
  and the weighted total loss using the same weighted-sum rule as training.
- Classification and ordinal label names are read from `class_maps`; target
  inverse transforms and internal-vs-external unit conventions are read from
  `target_schema`, not duplicated here.
- Pure embedding artifacts with no supervised objectives serialize
  `objectives: {}` and `total_loss: null`; `dojo eval holdout` raises a
  validation error if no objective covers the requested labeled target.

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

Ensemble result records additionally carry:

```text
ensemble_id
ensemble_hash
ensemble_result_scope    # member | ensemble
ensemble_member_id       # populated only for member rows
```

Rows written by the ensemble pipeline use `stage=ensemble_eval`. Cached
source rows keep their original stage in their source result dataset; if
they are transcribed into `ensemble_outputs.results.dir`, they are rewritten
as ensemble-produced rows with `stage=ensemble_eval`.

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
aspect_bucket
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
embedding_kind = image_embedding | tabular_embedding | fused_input_embedding | head_input_embedding
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
prediction_index
prediction_label
prediction_confidence
logits
probabilities
```

The row is scoped by `head_name`; the head → target link lives in
`_metadata.json` (each head's `target`). Ground-truth columns
(`target_index` / `target_name`) are reserved for P2 (see workplan P2.5); P1
writes predictions only.

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
ordinal_logits         # cumulative threshold logits; null for non-cumulative encodings
probabilities          # per-bin probabilities, always populated
```

`ordinal_logits` holds the `num_classes - 1` cumulative threshold logits
and is populated only for cumulative encodings (`coral`, `corn`); it is
null for `ordinal_cross_entropy`, which has no cumulative parameterization.
`probabilities` always holds the `num_classes` per-bin probabilities: for
`coral` / `corn` the writer derives them by differencing cumulative
probabilities decoded from `ordinal_logits`; for `ordinal_cross_entropy`
they are the softmax over the per-bin logits directly. Consumers therefore
always see per-bin probabilities regardless of `ordinal.encoding`. Because
`ordinal_logits` is encoding-specific, the `ordinal_logits_mean` ensemble
combine mode applies only to cumulative-encoding members; per-bin members
combine via `ordinal_probabilities_mean` (see `08-ensembles.md`).

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
  "compatibility": {
    "target_schema_hash": "sha256:...",
    "target_schema_source": {
      "version": "1",
      "heads": {}
    },
    "class_mapping_hash": "sha256:...",
    "class_mapping_source": {
      "version": "1",
      "heads": {}
    },
    "model_config_hash": "sha256:...",
    "model_config_source": {
      "version": "1",
      "model": {}
    },
    "preprocessing_hash": "sha256:...",
    "preprocessing_source": {
      "version": "1",
      "transforms": {},
      "tabular_preprocessing": {}
    }
  },
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
          "classes": ["A", "B", "C"],
          "class_mapping": {"0": "A", "1": "B", "2": "C"}
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

The `compatibility` block is required for model-produced result sidecars,
checkpoint-adjacent metadata, and exported-model metadata. A value may be
`null` only when the artifact genuinely cannot supply that compatibility
dimension, such as cached sample metadata without a producing model.
Consumers compare `*_hash` values first and diff the paired `*_source`
objects when hashes differ.

## Result partitioning

`training_outputs.results.partition_by`,
`ensemble_outputs.results.partition_by`, and
`eval_outputs.results.partition_by` are configurable lists. Partition
field options include:

```text
stage
epoch
ensemble_result_scope
ensemble_member_id
record_type
sweep_id
```

Include `sweep_id` automatically when a row is produced as part of a
sweep. `ensemble_result_scope` distinguishes combined ensemble rows
(`ensemble`) from retained member-level rows (`member`).
`ensemble_member_id` is a union column whose value is the member's
`checkpoint_hash` (checkpoint members) or `model_id` (exported model
members). It is populated only for `ensemble_result_scope=member`, so it is
not a default partition key for mixed member / ensemble result datasets.

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
    partition_by: [stage, record_type, ensemble_result_scope]
```

`ensemble_member_id` may be added as an opt-in partition key for member-only
analysis outputs, but mixed ensemble/member outputs should partition by
`ensemble_result_scope` first or leave `ensemble_member_id` as a filter
column.

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
    composed.yaml
    resolved.yaml
    resolved.json
    sweep_manifest.json
    cli.txt
    overrides.txt
  exports/
  metrics/
  figures/

eval_outputs.dir/
  config/
  results/                  # stage=infer | holdout_eval | representation_eval rows
  metrics/
  figures/
  eval_manifest.json        # always written: model source + dataset identity + metric summary
```

`eval_outputs/` is written by `dojo infer`, `dojo eval`, and standalone
`dojo eval representation`. `eval_manifest.json` is always written and
records the model source (artifact URI + `checkpoint_hash` / `model_hash`
and the inference-contract compatibility hashes), the evaluation dataset
(`dataset_id` / `dataset_hash` + split), a metric summary, and the
result-rows location — mirroring the ensemble manifest so a later
`dojo ensemble` or report step can consume eval runs. Like `dojo ensemble`,
an `infer` / `eval` run does **not** initialize experiment logging.

For `task.type: snapshot_ensemble` with
`ensemble_outputs.dir_template` defaulted to the training value, both
blocks resolve to the same path and the directory holds the union of both
layouts. Training rows are written under `results/`; ensemble rows are
written under `ensemble_results/`. Row `stage` still records provenance
(`train_validation` vs. `ensemble_eval`), but separate result directories
are the primary collision-avoidance boundary. Metrics files include the
producing block in their filenames when namespacing is needed.

`ensemble_members/` is only used when `dojo ensemble` materializes
member artifacts locally. Local member files may be symlinked; remote
member files may be cached through `storage` config and then symlinked.

## Logging and diagnostics

Experiment logging applies only to model-training runs. `dojo ensemble`
and `dojo ensemble candidates` do not initialize experiment logging.

Sinks:

- `local` — functional. **The only functional sink for the foreseeable
  future**; metrics and figures are recorded locally.
- `aim` — deferred; **not a registered sink type**. See
  `appendix-deferred-features.md`.
- `mlflow` — deferred; **not a registered sink type**. See
  `appendix-deferred-features.md`.

Multi-sink composition is supported via `CompositeExperimentLogger`, but
`local` is the only registered sink, so functional configurations use
`local` alone. There is no artificial cap on sink count. A configuration
naming `aim` or `mlflow` fails schema validation at load (unknown sink
type) per `12-validation-testing-and-preflight.md`.

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
- `08-ensembles.md` — `ensemble_result_scope`, `ensemble_member_id`,
  `stage=ensemble_eval` rows, namespacing in shared directories, manifest
  JSON files.
- `09-sweeps-and-batch-runs.md` — `sweep_id` / `sweep_hash` provenance
  columns and `sweep_outputs/` layout.
- `10-export.md` — `exports/` sub-directory and export metadata.
- `12-validation-testing-and-preflight.md` — strict-schema deferral
  policy covering the Aim and MLflow sinks.
- `appendix-deferred-features.md` — deferred Aim and MLflow sinks; HDF /
  `.h5` result exports.
- `glossary.md` — `split`, `stage`, `record_type`, `embedding_kind`,
  `head_name`, identifier / hash vocabulary.
