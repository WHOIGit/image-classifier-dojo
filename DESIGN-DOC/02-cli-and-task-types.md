
# 02. CLI and Task Types

## Purpose

Defines the canonical CLI command surface, how `task.type` selects training
behavior, and the relationship between subcommands and config-first usage.
This file is the single source of truth for command names; other files link
here rather than re-listing commands.

## Command families

Top-level groups:

- `dojo train`
- `dojo infer`
- `dojo eval`
- `dojo inspect`
- `dojo ensemble`
- `dojo export`

Subcommands are config-first shorthands that constrain the target output.

```text
dojo train                                # task type comes from task.type
dojo infer
dojo infer predictions
dojo infer embeddings
dojo eval
dojo eval holdout
dojo eval representation
dojo inspect
dojo inspect config
dojo inspect dataset
dojo inspect backbone
dojo inspect checkpoint
dojo ensemble
dojo ensemble candidates
dojo export
```

There are **no** `dojo train supervised`, `dojo train ssl`, or
`dojo train-snapshot-ensemble` subcommands. Training paradigm is selected
via `task.type` (see below).

## Task types

`task.type` is the axis that determines what `dojo train` does. Supported
types in the initial implementation:

- `supervised`
- `ssl`
- `snapshot_ensemble`

### `supervised`

Standard supervised training, single- or multi-head, with optional tabular
fusion and embedding adapter. See `05-models-training-and-heads.md`.

### `ssl`

Self-supervised training. The functional method is `dino_v2` via Lightly.
Other SSL methods (SimCLR, VICReg, PMSN, original DINO) have config-schema
slots but stubbed runtime paths — see `appendix-deferred-features.md`.

### `snapshot_ensemble`

Single-command convenience that orchestrates two steps against one shared
run directory:

1. Supervised training with a snapshot-cycle scheduler (cosine warm
   restarts producing one checkpoint per cycle).
2. Ensembling the just-finished snapshot checkpoints through the ensemble
   pipeline.

Snapshot checkpoints land in `<training_outputs.dir>/checkpoints/`.
Ensemble artifacts land under `ensemble_outputs/` sub-directories (default
`ensemble_results/` and `ensemble_figures/`, controlled by
`ensemble_outputs.results.dir` and `ensemble_outputs.figures.dir`).

A `task.type: snapshot_ensemble` config must include both a `training:`
block (with a snapshot-capable scheduler) and an `ensemble:` block
(selection strategy + combine modes). Pydantic validation enforces both.

Canonical invocation:

```bash
dojo train experiment=ifcb/snapshot_experiment
```

where the experiment config sets `task.type: snapshot_ensemble`.

**Re-running ensembling against a historical training run is not a
`task.type: snapshot_ensemble` invocation.** It is a regular
`dojo ensemble` invocation with a `run_dir` or `run_dir_glob` candidate
source pointing at the historical run directory.

Example `task.type: snapshot_ensemble` config:

```yaml
task:
  type: snapshot_ensemble

training:
  max_epochs: 300

scheduler:
  name: cosine_warm_restarts
  first_cycle_epochs: 50
  cycle_mult: 1.0
  max_lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_epochs: 5

checkpointing:
  save_cycle_snapshots:
    enabled: true
    at_cycle_end: true

ensemble:
  selection:
    strategy: cycle_end_snapshots
    use_all_cycles: true
  inference:
    combine:
      classification: probabilities_mean
```

## `dojo inspect config`

Replaces the old `dojo validate-config`. Behavior:

- compose configs;
- apply CLI overrides;
- run schema validation;
- render output-path templates;
- show expected output folder structure;
- warn about run/sweep directory collisions;
- show enabled outputs and deferred/stubbed features;
- optionally emit machine-readable JSON for CI/tests.

### Remote-artifact validation strength

Default: **schema-and-local feasibility only**. Resolve paths, validate
format of locally-accessible artifacts; do **not** require network access
to remote URIs. CI-safe and offline-friendly.

Opt-in `--check-remote` flag (or `inspect.check_remote: true`) performs
HEAD requests / etag checks against remote URIs (S3 manifests, checkpoints,
etc.) and reports availability and size. Without `--check-remote`,
missing-remote-artifact errors surface at runtime, not at inspect time.
This is intentional.

## `dojo inspect dataset`

Replaces the old `dojo tools make-manifest`. Read-only by default. May
write canonical CSV / Parquet manifests when an output path is explicitly
configured. Class-folder manifest generation is folded in: `class_folder`
is an inspect-dataset source, **not** a train / eval / infer backend.

Inspect outputs are not normal run outputs and do not require a run
directory.

With no aspect flag it reports missing targets and summarizes how many
samples will be dropped, skipped, or fail validation (default
missing-target policy `error` for all heads).

Aspect flags scope **both what is computed and the I/O cost paid**, so a
header-level histogram never triggers a full pixel decode. Each aspect is
either a **frozen** value (deterministic, written to the dataset stats
cache, consumed at config resolution, and hashed) or an **advisory** report
(human-facing, may be sampled, never cached or hashed). Frozen fit
statistics are computed on the `train` split; structural properties are
computed across all splits. The stats cache is defined in
`04-data-and-storage.md`.

Tier 0 — manifest only (no image I/O):

| Flag | Computes | Kind |
|---|---|---|
| `--targets` | missing-target / drop / skip counts (default) | advisory |
| `--class-counts` | per-class counts + frequencies | frozen |
| `--class-map` | resolved index ↔ label mapping | frozen |
| `--target-stats` | regression / ordinal target distribution + fitted transform stats (standardize mean / std, Box-Cox λ) | frozen |
| `--tabular-stats` | tabular normalization stats + imputation fill values | frozen |
| `--imbalance` | imbalance ratio, empty-class, non-contiguous-index checks | advisory |

Tier 1 — image headers only (`imagesize` / lazy open, no pixel decode):

| Flag | Computes | Kind |
|---|---|---|
| `--bucket-histogram` | aspect-ratio + long-side distribution; suggested bucket boundaries | advisory |
| `--bit-depth` | resolve `input_bit_depth`; flag heterogeneous depths | frozen value + advisory warning |

Tier 2 — full pixel decode (opt-in, expensive):

| Flag | Computes | Kind |
|---|---|---|
| `--normalization` | per-channel mean / std (streaming) for `normalize: {mode: dataset}` | frozen |

Backend-specific:

| Flag | Computes | Kind |
|---|---|---|
| `--bin-lengths` | ROI counts per bin → `bin_lengths` cache (`ifcb_bins` only) | frozen |

Compound and control flags:

- `--stats[=URI]` — run all frozen aspects and write / update the dataset
  stats cache (the producer for `normalize: {mode: dataset}`, tabular and
  target stats, imputation, `class-map` / `class-counts`, `bit-depth`, and
  `bin-lengths`).
- `--report` — run all advisory aspects, print / write a human report,
  write no cache.
- `--all` — every aspect (warns: incurs a full-decode pass).
- `--split train|val|all` — override split selection; otherwise fit
  statistics default to `train` and structural properties to `all`.
- `--sample N|FRACTION` — estimate from a subsample. Permitted for advisory
  aspects; for frozen aspects it marks the result `estimated: true` so a
  sampled statistic is never silently frozen into the contract.
- `output=PATH` — write a canonical manifest (incl. `class_folder` scan).

When tiers stack (e.g. `--normalization --bit-depth --bucket-histogram`),
images are opened once and all requested aspects are collected in that
single pass; the decode tier subsumes the header tier.

Examples:

```bash
# fast, header-only bucket advisory (sub-second with --sample)
dojo inspect dataset data=ifcb/species_manifest --bucket-histogram

# produce the frozen stats cache consumed at training time
dojo inspect dataset data=ifcb/species_manifest \
  --stats=s3://datasets/ifcb/cache/species_stats.parquet

# canonical manifest from a class-folder source
dojo inspect dataset data=ifcb/species_manifest \
  output=./inspect_outputs/species_manifest.parquet
```

## `dojo inspect backbone`

Reports per-module shape, parameter count, trainable/frozen status, and
output embedding dim for the configured backbone with freeze policy
applied. Useful for picking `named_modules_trainable` / freeze module
names. Example:

```bash
dojo inspect backbone backbone=torchvision/resnet50
```

```bash
dojo inspect backbone \
  backbone=torchvision/resnet50 \
  model.backbone.freeze.policy=after_module_trainable \
  model.backbone.freeze.module=layer3 \
  model.backbone.freeze.inclusive=true
```

## `dojo inspect checkpoint`

Reports checkpoint metadata, embedded compatibility hashes, head /
preprocessing configuration, and checkpoint-hash filename match.

## `dojo infer`

`dojo infer predictions` writes per-sample prediction rows (canonical
result schema). `dojo infer embeddings` extracts embeddings from
supervised checkpoints, supervised `.pt` model exports, SSL training
checkpoints, or SSL encoder exports.

`dojo infer embeddings` is the canonical embedding-extraction command;
the legacy `dojo eval embeddings` name is removed.

## `dojo eval`

- `dojo eval holdout` — evaluate a trained model against a holdout split.
- `dojo eval representation` — standalone representation evaluation. See
  `07-ssl-and-representation-eval.md`.

`dojo eval knn` and `dojo eval linear-probe` are not primary commands in
the initial implementation; that functionality lives inside
`representation_eval` config sub-blocks accessed through
`dojo eval representation` or scheduled during training.

## `dojo ensemble`

- `dojo ensemble` — ensemble evaluation / inference against a candidate
  manifest or candidate-discovery sources.
- `dojo ensemble candidates` — artifact-inspection / manifest-writing
  command. Does not initialize experiment logging.

Cross-run ensembling is not a distinct mode — it works via `dojo
ensemble` with explicit candidate sources such as `run_dir` and
`run_dir_glob`. See `08-ensembles.md`.

## `dojo export`

Explicit export from a checkpoint, manifest, or already-existing run
output. Artifact types: `torchscript` and `onnx`. See `10-export.md`.

## Config-first usage

Every command accepts Hydra-style overrides. Examples:

```bash
dojo inspect config experiment=ifcb/species_baseline training.batch_size=64
dojo inspect dataset data=ifcb/species_manifest output=./inspect_outputs/species_manifest.parquet
dojo train experiment=ifcb/species_baseline
dojo infer embeddings experiment=ifcb/species_baseline checkpoint=./runs/baseline/checkpoints/best.ckpt
dojo eval representation experiment=ifcb/dinov2_repr_eval
dojo ensemble candidates experiment=ifcb/ensemble_candidates ensemble_outputs.manifests.dir=./shared_manifests
dojo ensemble \
  experiment=ifcb/ensemble_search \
  ensemble.candidates.sources.0.type=manifest \
  ensemble.candidates.sources.0.manifest_uri=./shared_manifests/ifcb_candidates.json
```

## Cross-References

- `03-configuration.md` — config-key surface every command consumes.
- `04-data-and-storage.md` — dataset backends and `dojo inspect dataset`
  behavior.
- `05-models-training-and-heads.md` — `task.type: supervised` and
  `task.type: snapshot_ensemble` training internals.
- `07-ssl-and-representation-eval.md` — `task.type: ssl`, `dojo eval
  representation`.
- `08-ensembles.md` — `dojo ensemble` / `dojo ensemble candidates` and
  the snapshot-ensemble orchestration step.
- `10-export.md` — `dojo export` and `*_outputs.export` blocks.
- `12-validation-testing-and-preflight.md` — `dojo inspect config`
  validation tiers and preflight checks.
- `glossary.md` — task type and CLI vocabulary.
