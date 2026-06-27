
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

## CLI architecture and execution model

Dojo's CLI is a **Typer** application (git-style subcommands, `--options`,
rich help) that composes configs through the **Hydra Compose API**
(`hydra.compose`). It does **not** use `@hydra.main`, and it does not use
Hydra's launcher / sweeper plugins. Sweep expansion, runtime-ID
generation, output-directory resolution, and existing-directory policy are
all owned by Dojo (see `09-sweeps-and-batch-runs.md`).

### Config overrides vs. command options

Every command line carries two distinct kinds of token, separated by a
simple lexical rule — **every Hydra override is dash-free, every Typer
option starts with `-`**:

- **Config overrides** — dash-free `key=value` and `group=option` tokens,
  passed through to the Compose API; they land in the composed config
  tree. Examples: `experiment=ifcb/species_baseline`,
  `training.batch_size=64`,
  `model.image_input.backbone.architecture.name=resnet50`,
  `data=ifcb/species_manifest`.
- **Command options** — Typer `--flags` and positional arguments. They
  control the command itself and are **not** part of the config tree.
  Examples: `--check-remote`, `--stats`, `--format json`, `--output`,
  `--type`, `--checkpoint`.

A command mixes both freely:

```bash
dojo inspect dataset data=ifcb/species_manifest --stats --format json
#                    └─ config override ─┘       └─ command options ─┘
```

### Command I/O is options, not config

Per-invocation inputs and outputs are command options, never config keys:
`--checkpoint`, `--output`, `--type`, `--ensemble-manifest`. They do not
appear in the canonical root shape (`03-configuration.md`).

### Two ways a checkpoint enters a command

- **Building a model for the run** → config. A checkpoint that initializes
  the model (transfer learning, representation eval against an encoder) is
  `model.image_input.backbone.weights.source: checkpoint` +
  `model.image_input.backbone.weights.uri` — composed, validated, and
  recorded in the run's config provenance. See
  `05-models-training-and-heads.md`.
- **Consuming a checkpoint / model artifact** → command option
  `--checkpoint`. `dojo export`, `dojo inspect checkpoint`, and
  `dojo infer` / `dojo eval` loading a complete trained model (whose
  architecture self-describes from the checkpoint's stored
  hyperparameters) take the artifact as a `--checkpoint` option.

### Sweeps are config-defined

There is no `-m` / `--multirun` flag. A run is a sweep when the composed
config's `sweep:` block defines axes (`sweep.mode: grid` with a non-empty
`sweep.grid`). Dojo expands the cartesian product itself. See
`09-sweeps-and-batch-runs.md` for the sweep block and the authored →
resolved pipeline.

## Task types

`task.type` is the axis that determines what `dojo train` does. Supported
types in the initial implementation:

- `supervised`
- `ssl`
- `snapshot_ensemble`

### `supervised`

Standard supervised training, single- or multi-head, with optional tabular
input encoder, implicit input concatenation, and embedding adapter. See
`05-models-training-and-heads.md`.

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

Opt-in `--check-remote` command option performs HEAD requests / etag
checks against remote URIs (S3 manifests, checkpoints, etc.) and reports
availability and size. It is a command option only — there is no
`inspect.check_remote` config key. Without `--check-remote`,
missing-remote-artifact errors surface at runtime, not at inspect time.
This is intentional.

`dojo inspect config` is not a dataset-stats producer. It never computes
missing frozen dataset statistics and never updates the stats cache. When a
config requires frozen stats (`normalize: {mode: dataset}`, tabular
imputation / normalization, target transforms, cached lengths, or
count-dependent weighting), inspect config validates the configured cache
shape and any locally available cache metadata. Without `--check-remote`,
remote cache contents are not fetched; missing or stale remote cache
contents surface when run resolution / preflight actually consumes them.
Use `dojo inspect dataset` with the needed aspect flags, or `--stats`, to
create or refresh those values.

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
| `--dimensions` | per-sample native width / height (→ aspect ratio, long side), cached as the primitive for bucket design and assignment | frozen (structural) |
| `--bucket-histogram` | aspect-ratio + long-side distribution; suggested bucket boundaries — derived from cached `--dimensions` (reads headers only if absent) | advisory |
| `--bit-depth` | resolve `input_bit_depth`; flag heterogeneous depths | frozen value + advisory warning |

Tier 2 — full pixel decode / full byte read (opt-in, expensive):

| Flag | Computes | Kind |
|---|---|---|
| `--normalization` | per-channel mean / std (streaming) for `normalize: {mode: dataset}` | frozen |
| `--content-hash` | true `dataset_content_hash` over all image bytes; folds into any full pass for free (see `04-data-and-storage.md`) | frozen (verification) |

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
- `--format text|chart|json` — advisory output rendering. `chart` (default
  in an interactive terminal) draws in-terminal bar charts / histograms for
  distributions (`--bucket-histogram`, `--class-counts`, `--imbalance`,
  target / tabular distributions); `text` is plain tabular; `json` is
  machine-readable. Frozen values are always written to the cache
  regardless of `--format`.
- `--output PATH` — write a canonical manifest (incl. `class_folder`
  scan). A command option, not a config key.

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
  --output ./inspect_outputs/species_manifest.parquet
```

## `dojo inspect backbone`

Reports per-module shape, parameter count, trainable/frozen status, and
output embedding dim for the configured backbone with freeze policy
applied. Useful for picking `named_modules_trainable` / freeze module
names. Example:

```bash
dojo inspect backbone experiment=ifcb/baseline
```

This composes the experiment config, then inspects
`model.image_input.backbone` with `training.freeze.backbone` applied.

```bash
dojo inspect backbone \
  model.image_input.backbone.architecture.source=torchvision \
  model.image_input.backbone.architecture.name=resnet50
```

```bash
dojo inspect backbone \
  model.image_input.backbone.architecture.source=torchvision \
  model.image_input.backbone.architecture.name=resnet50 \
  training.freeze.backbone.policy=after_module_trainable \
  training.freeze.backbone.module=layer3 \
  training.freeze.backbone.inclusive=true
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

Every command accepts dash-free Hydra config overrides; command I/O is
expressed as `--options` (see "CLI architecture and execution model"
above). Examples:

```bash
dojo inspect config experiment=ifcb/species_baseline training.batch_size=64
dojo inspect dataset data=ifcb/species_manifest --output ./inspect_outputs/species_manifest.parquet
dojo train experiment=ifcb/species_baseline
dojo infer embeddings experiment=ifcb/species_baseline --checkpoint ./runs/baseline/checkpoints/best.ckpt
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
