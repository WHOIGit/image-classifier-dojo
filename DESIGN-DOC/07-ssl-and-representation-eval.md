
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

SimCLR, VICReg, PMSN, and the original DINO method are intentionally
removed from the initial-implementation runtime. Their config slots
exist (`ssl.method: simclr | vicreg | pmsn | dino`); the runtime raises
`NotImplementedError`. See `appendix-deferred-features.md`.

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
  schedule:
    every_n_epochs: 5
    on_fit_end: true
  embeddings:
    enabled: true
    split: val
  projections:
    methods: [umap, tsne]
  clustering:
    methods: [hdbscan]
  probes:
    classification:
      enabled: true
      heads: [species]
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

### Scheduling

Evaluations are schedulable by:

```text
every N epochs
every N train batches
every fractional epoch
end of epoch
end of training
```

`every_fractional_epoch: 0.10` means approximately every 10% of an
epoch.

Expensive evaluations support subsampling:

```yaml
representation_eval:
  knn:
    max_reference_samples: 50000
    max_query_samples: 10000
```

### Standalone vs. training-integrated

Same evaluator code, two entry points:

- training-integrated callbacks (scheduled inside a `dojo train` run);
- standalone CLI: `dojo eval representation`.

Standalone example:

```bash
dojo eval representation \
  experiment=ifcb/dinov2_repr_eval \
  model.image_input.backbone.weights.source=checkpoint \
  model.image_input.backbone.weights.uri=./runs/dinov2/checkpoints/best.ckpt \
  representation_eval.embeddings.split=holdout
```

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
