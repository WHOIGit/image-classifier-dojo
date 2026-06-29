
# 07. SSL and Representation Evaluation

## Purpose

Defines self-supervised training (functional: DINOv2 via Lightly) and the
`representation_eval` config group, which replaces the old `ssl_eval`.
Representation evaluation is **not SSL-only** — it works against any task
that produces image embeddings, including supervised models.

## SSL

### Framework

Use Lightly only for SSL method implementations. Do not depend directly
on Meta DINOv2 repositories. The initial SSL task focuses on DINOv2-style
training using Lightly components.

`ssl.framework: lightly` selects the Lightly framework. The backbone
architecture for an SSL run is still selected via
`model.image_input.backbone.architecture.source: timm | torchvision`.
Checkpoint initialization uses
`model.image_input.backbone.weights.source: checkpoint`. There is no
`model.image_input.backbone.architecture.source: lightly`.

DINOv2 typically uses a timm ViT backbone. See the Lightly DINOv2 example
at <https://docs.lightly.ai/self-supervised-learning/examples/dinov2.html>.

### Functional method

```yaml
ssl:
  method: dino_v2
  framework: lightly
```

`dino_v2` through Lightly is functional in the initial implementation.

### Deferred SSL methods

SimCLR, VICReg, PMSN, and the original DINO method are intentionally out
of scope for the initial implementation. They are **not** `ssl.method`
schema values, so authoring `ssl.method: simclr | vicreg | pmsn | dino`
fails generic validation as an out-of-enum value. See
`appendix-deferred-features.md`.

### Model structure

```text
image views
  ↓
backbone encoder (timm/torchvision/checkpoint)
  ↓
projection head
  ↓
DINOv2-style SSL loss
```

The SSL task module wraps the encoder + projection head, applies the
SSL-specific multi-view transform, and produces canonical results
(embeddings, SSL diagnostics) through the same result writer as
supervised runs.

### SSL training example

```yaml
task:
  type: ssl

ssl:
  method: dino_v2
  framework: lightly
  image_size: 224
  projection_dim: 65536

model:
  image_input:
    backbone:
      architecture:
        source: timm
        name: vit_small_patch14_dinov2
      weights:
        source: none

representation_eval:
  enabled: true
  name: ssl_epoch_eval
  schedule:
    mode: every_n_epochs
    every_n_epochs: 5
    include_fit_end: true
  dataset:
    splits:
      reference: train
      query: val
    seed: 123
  embeddings:
    enabled: true
    kinds: [image_embedding]
    cache: true
  projections:
    enabled: true
    methods:
      - type: umap
      - type: tsne
  clustering:
    enabled: true
    methods:
      - type: hdbscan
  probes:
    classification:
      enabled: true
      targets: [species]
    regression:
      enabled: true
      targets: [biovolume]
    ordinal:
      enabled: true
      targets: [quality_grade]
```

### SSL encoder export

SSL training produces standard `.ckpt` training artifacts and may export
a portable encoder via `training_outputs.export` or `dojo export`. The
exported encoder is the standard input for supervised transfer learning
(see `05-models-training-and-heads.md`).

## Representation evaluation

`representation_eval` is a top-level config group usable with any task
that produces image embeddings, including `task.type: supervised` and
`task.type: ssl`. A supervised run may schedule representation
evaluation against its own encoder during training. The standalone
`dojo eval representation` command works against checkpoints from
either task type.

### Canonical config shape

`representation_eval` is one schema with optional sub-blocks. Disabled
sub-blocks are ignored; enabled sub-blocks validate their required fields
and write canonical result rows.

```yaml
representation_eval:
  enabled: true
  name: default_repr_eval

  schedule:
    mode: fit_end        # disabled | fit_end | every_n_epochs | every_n_steps | fractional_epoch
    every_n_epochs: null
    every_n_steps: null
    fractional_epoch: null
    include_fit_end: true

  dataset:
    splits:
      reference: train
      query: val
    max_reference_samples: null
    max_query_samples: null
    include_reference_outputs: false
    require_labels: auto
    seed: 123

  embeddings:
    enabled: true
    kinds: [image_embedding]
    batch_size: null
    cache: true

  diagnostics:
    enabled: true
    metrics: [embedding_norm, per_dimension_std, effective_rank]

  nearest_neighbors:
    enabled: false
    k: [1, 5, 10]
    distance: cosine
    write_neighbors: true
    write_knn_predictions: true

  probes:
    classification:
      enabled: false
      targets: []
      model: linear_classifier
    regression:
      enabled: false
      targets: []
      model: ridge
    ordinal:
      enabled: false
      targets: []
      model: ordinal_logistic_regression

  projections:
    enabled: false
    methods:
      - type: pca
        n_components: 2
      - type: umap
        n_components: 2
      - type: tsne
        n_components: 2

  clustering:
    enabled: false
    methods:
      - type: kmeans
        n_clusters: auto
      - type: mini_batch_kmeans
        n_clusters: auto
      - type: hdbscan
      - type: agglomerative
        n_clusters: auto

  outliers:
    enabled: false
    methods:
      - type: local_outlier_factor
      - type: isolation_forest
```

`representation_eval.name` becomes `evaluation_name` on result rows and
disambiguates multiple scheduled / standalone evaluations in the same run.
`dataset.seed` controls deterministic subsampling and any representation-eval
algorithm with a random state.

### Supported components

- Embedding extraction.
- Embedding diagnostics (collapse checks, variance, effective rank,
  augmentation consistency, etc.).
- Dimensionality reduction: PCA, UMAP, t-SNE.
- Clustering: kmeans, mini_batch_kmeans, HDBSCAN, agglomerative
  (optional).
- Nearest-neighbor retrieval (labeled or unlabeled).
- Probes:
  - classification (linear, multinomial-logistic);
  - regression (ridge);
  - ordinal (ordinal logistic regression).
- Outlier / novelty scoring.

UMAP, t-SNE, and HDBSCAN are gated by the `repr_eval` optional extra but
are **not deferred** — they are functional when the extra is installed.
Regression and ordinal probes are functional. Supervised fine-tuning from
an SSL pretrained backbone is supervised transfer learning, not a
representation-evaluation mode — see `05-models-training-and-heads.md`.

### Reference/query split semantics

Representation evaluation has two logical sample sets:

- **reference** — fit set for learned evaluation artifacts: probe models,
  k-NN reference indexes, projection fit steps when the method supports
  transform, clustering fit steps, and outlier/density reference models.
- **query** — scored / predicted / assigned set that produces the primary
  output rows.

Defaults: `reference=train`, `query=val`. Learned evaluation artifacts fit
on the reference split only, then score the query split. Set
`include_reference_outputs: true` to also write predictions, projections,
cluster assignments, or outlier scores for the reference samples. Methods
that do not support a clean fit/transform split, such as t-SNE and HDBSCAN,
fit on the union of reference + query embeddings for visualization /
clustering, but query rows remain the primary reported outputs unless
`include_reference_outputs` is enabled.

Subsampling is deterministic per split from `representation_eval.dataset.seed`.
When subsampling is enabled, written rows and sidecar metadata record the
sample counts, requested limits, and seed.

### Label-required vs. label-free evaluation

| record_type | Requires labels | Notes |
| --- | --- | --- |
| `embedding` | No | Any image. |
| `nearest_neighbor` | No | Retrieval / supporting record for `knn_prediction`. |
| `knn_prediction` | Yes | k-NN classification. |
| `classification_probe_prediction` | Yes | Linear / multinomial probes. |
| `regression_probe_prediction` | Yes | Ridge / similar probes. |
| `ordinal_probe_prediction` | Yes | Ordinal probes. |
| `cluster_assignment` | No | Labels optional for ARI / NMI / purity later. |
| `projection` | No | PCA / UMAP / t-SNE coords; labels optional for coloring. |
| `outlier_score` | No | Density / distance / novelty. |
| `diagnostic` | No | Collapse / variance / augmentation consistency. |

Probe `head_name` should identify the probe (e.g.
`linear_probe_species`, `ridge_probe_biovolume`). `probe_model_type`
documents the probe class (e.g. `ridge`, `linear_classifier`,
`ordinal_logistic_regression`). Probe predictions reuse the matching
native head record column shape but with the probe-specific
`record_type`. See `06-results-artifacts-and-metadata.md`.

### Validation contract

Pydantic validation enforces:

- `enabled: false` disables the whole block; enabled sub-blocks validate
  independently.
- `schedule.mode: disabled` is allowed only for standalone configs or when
  the block is present but intentionally inactive during training.
- `schedule.mode: every_n_epochs` requires `every_n_epochs`; `every_n_steps`
  requires `every_n_steps`; `fractional_epoch` requires
  `fractional_epoch` in `(0, 1]`.
- `dataset.splits.reference` and `dataset.splits.query` must be valid dataset
  splits for the active command.
- `max_reference_samples` and `max_query_samples`, when set, must be positive
  integers.
- Probe `targets` must reference `data.targets`, and the probe family must
  match the target type: classification probes for classification targets,
  regression probes for regression targets, and ordinal probes for
  `ordinal_classification` targets.
- `nearest_neighbors.write_knn_predictions: true` requires classification
  labels for the reference and query samples selected by the active splits.
- `require_labels: auto` derives the requirement from enabled components;
  `true` errors if any selected sample lacks required labels; `false` allows
  label-free components only and rejects enabled probes / k-NN predictions.
- Standalone `dojo eval representation` must configure an encoder through
  `model.image_input.backbone.weights.source: checkpoint` or another
  buildable checkpoint-backed model config. Training-integrated evaluation
  uses the current run encoder.
- UMAP, t-SNE, HDBSCAN, sklearn probes, clustering, and outlier methods are
  gated by the `repr_eval` optional extra. Missing extras raise a clear
  runtime dependency error, not a deferred-feature error.

### Scheduling

Training-integrated evaluations are schedulable by:

```text
disabled
fit_end
every_n_epochs
every_n_steps
fractional_epoch
```

`fractional_epoch: 0.10` means approximately every 10% of an epoch.
`include_fit_end: true` adds one final evaluation at training end even when
the main schedule is epoch-, step-, or fractional-epoch-based.

Expensive evaluations support subsampling:

```yaml
representation_eval:
  dataset:
    max_reference_samples: 50000
    max_query_samples: 10000
```

### Execution and result mapping

Both standalone and training-integrated paths use the same evaluator:

```text
build / load encoder
extract embeddings for reference and query splits
optionally cache embeddings for downstream components
run diagnostics
build nearest-neighbor index and optional k-NN predictions
fit probes on reference embeddings and predict query embeddings
fit / transform projections
fit / assign clusters
fit / score outliers
write canonical result rows and metrics / figures
```

### Standalone vs. training-integrated

Same evaluator code, two entry points:

- training-integrated callbacks (scheduled inside a `dojo train` run),
  writing into that run's `training_outputs`;
- standalone CLI: `dojo eval representation`, writing rows / metrics and an
  `eval_manifest.json` into `eval_outputs` (`03-configuration.md`).

Standalone example:

```bash
dojo eval representation \
  experiment=ifcb/dinov2_repr_eval \
  model.image_input.backbone.weights.source=checkpoint \
  model.image_input.backbone.weights.uri=./runs/dinov2/checkpoints/best.ckpt \
  representation_eval.dataset.splits.query=holdout
```

Result mapping:

| Config block | Result rows |
| --- | --- |
| `embeddings` | `record_type=embedding` |
| `diagnostics` | `record_type=diagnostic` |
| `nearest_neighbors.write_neighbors` | `record_type=nearest_neighbor` |
| `nearest_neighbors.write_knn_predictions` | `record_type=knn_prediction` |
| `probes.classification` | `record_type=classification_probe_prediction` |
| `probes.regression` | `record_type=regression_probe_prediction` |
| `probes.ordinal` | `record_type=ordinal_probe_prediction` |
| `projections` | `record_type=projection` |
| `clustering` | `record_type=cluster_assignment` |
| `outliers` | `record_type=outlier_score` |

Training-integrated callbacks write rows, metrics, and figures under the
current `training_outputs` directory. Standalone `dojo eval representation`
writes rows / metrics / figures and `eval_manifest.json` under
`eval_outputs`.

### Diagnostics

Useful `diagnostic_name` values for the `diagnostic` record type:

```text
embedding_norm_mean
embedding_norm_std
per_dimension_std
effective_rank
covariance_condition
pairwise_cosine_mean
pairwise_cosine_std
augmentation_consistency_cosine
```

`diagnostic_scope` values: `sample`, `batch`, `epoch`, `dataset`.

## Cross-References

- `02-cli-and-task-types.md` — `task.type: ssl` and `dojo eval
  representation`.
- `03-configuration.md` — placement of `ssl:` and `representation_eval:`.
- `05-models-training-and-heads.md` — backbone sources used by SSL,
  supervised transfer learning from SSL exports, supervised
  representation-eval scheduling.
- `06-results-artifacts-and-metadata.md` — representation-evaluation
  record types and `evaluation_name`.
- `11-dependencies.md` — `ssl` and `repr_eval` optional extras.
- `appendix-deferred-features.md` — non-DINOv2 SSL methods.
- `glossary.md` — record-type vocabulary.
