
# 02. CLI and Task Types

## Purpose

Defines the canonical CLI command surface, how `task.type` selects training
behavior, and the relationship between subcommands and config-first usage.
This file is the single source of truth for command names; other files link
here rather than re-listing commands.

## Command families

Top-level groups:

- `dojo init`
- `dojo train`
- `dojo infer`
- `dojo eval`
- `dojo inspect`
- `dojo ensemble`
- `dojo sweep`
- `dojo export`

Subcommands are config-first shorthands that constrain the target output.

```text
dojo init
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
dojo sweep
dojo sweep prepare
dojo sweep train
dojo sweep status
dojo sweep report
dojo export
```

There are **no** `dojo train supervised`, `dojo train ssl`, or
`dojo train-snapshot-ensemble` subcommands. Training paradigm is selected
via `task.type` (see below).

Invoking `dojo infer`, `dojo eval`, `dojo inspect`, or `dojo sweep` with no
subcommand prints that group's help and exits — they have no default
subcommand. `dojo ensemble` is the exception: bare `dojo ensemble` runs an
ensemble, with `dojo ensemble candidates` as its only subcommand.
`dojo init`, `dojo train`, and `dojo export` are leaf commands.

## CLI architecture and execution model

Dojo's CLI is a **Typer** application (git-style subcommands, `--options`,
rich help). `dojo init` bootstraps local project files. All other
config-aware commands compose configs through the **Hydra Compose API**
(`hydra.compose`). Dojo does **not** use `@hydra.main`, and it does not
use Hydra's launcher / sweeper plugins. Sweep expansion, runtime-ID
generation, output-directory resolution, and run-directory collision
handling are all owned by Dojo (see `09-sweeps-and-batch-runs.md`).

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

### Config sources and replay modes

Config-aware commands can start from authored, composed, or resolved config
inputs:

- **Config group selectors** — dash-free Hydra selectors such as
  `experiment=ifcb/species_baseline`. These resolve from the active config
  search path: explicit `--config-dir` entries, then local `./configs` when
  present, then packaged read-only Dojo configs.
- **Authored root file** — `--config FILE` loads one human-written root YAML
  file and then applies CLI overrides. This is useful for project-local
  configs that are not arranged as Hydra groups.
- **Composed config artifact** — `--config RUN_DIR/config/composed.yaml`
  replays composed authored intent and still performs fresh runtime
  resolution: new generated values, output directories, and frozen-stat
  checks as appropriate for the command.
- **Resolved config artifact** — `--resolved-config RUN_DIR/config/resolved.yaml`
  loads a fully resolved run artifact. Read-only inspection commands may
  consume it directly. Mutating commands such as `dojo train` may execute
  it directly when the referenced run directory is absent, empty, or only
  contains prepared config artifacts. If the rest of the run folder is not
  empty, the command requires an explicit override: `--resume` to continue
  the same run context, or `--clobber` to delete the existing directory
  contents and run the resolved config in place. To branch instead, copy
  the composed / resolved config to a new location, edit its output `dir` /
  `dir_template`, and run that as a fresh config. When output blocks share
  a physical directory (e.g. the snapshot-ensemble case), `--clobber`
  clears each resolved physical directory once at startup, so later phases
  of the same command do not re-clear it.

`--config`, `--resolved-config`, and Hydra group selectors are mutually
exclusive as root config sources, though ordinary value overrides may still
be applied where the command permits them.

### Command I/O is options, not config

Per-invocation inputs and outputs are command options, never config keys:
`--checkpoint`, `--model`, `--output`, `--type`, `--ensemble-manifest`. They
do not appear in the canonical root shape (`03-configuration.md`).

### Commands select which config blocks apply

Dojo uses **one** root config schema for every command. The **command**, not
`task.type`, decides which blocks are active: `dojo train` consumes
`model` / `training` / `optimizer` / `scheduler` / `objectives` /
`checkpointing` (with `task.type` selecting the training paradigm among
`supervised` / `ssl` / `snapshot_ensemble`); `dojo infer` and `dojo eval holdout`
consume `data` / `runtime` / `storage` / `eval_outputs` and take the model
and its preprocessing from the artifact's embedded inference contract
(`06-results-artifacts-and-metadata.md`); `dojo eval representation`
additionally activates `model` / `transforms` / `representation_eval`
because it builds an encoder and configured probes; `dojo ensemble` consumes
`ensemble` / `ensemble_outputs` and, when member inference is possible or
required (`inference_as_needed` / `force_inference`), also consumes `data` /
`runtime` / `storage` for the target dataset and execution context;
`dojo sweep` consumes `sweep` /
`sweep_outputs`. `inference` and `eval` are **not** `task.type` values —
they are commands, exactly like `ensemble` and `export`.

Per-operation configs are the documented norm, but a single "loaded"
lifecycle config carrying blocks for several commands is valid: each command
consumes its slice and **warns** about blocks it does not use rather than
failing. When a config carries more than the active command needs,
`config_hash` is computed over that command's active block-set only, so
editing an unused block never changes the job's identity
(`06-results-artifacts-and-metadata.md`). A command's **active block-set**
is the blocks it consumes (enumerated above) minus the global `config_hash`
exclusions — runtime-resolved values, output paths, `output_root`, and the
`*_outputs` blocks. So an `infer` / `eval holdout` job hashes `data`; a
`train` job hashes its full model / training block-set; an
`eval representation` job additionally includes `model` / `transforms` /
`representation_eval`.

### Two ways a checkpoint enters a command

- **Building a model for the run** → config. A checkpoint that initializes
  the model (transfer learning, representation eval against an encoder) is
  `model.image_input.backbone.weights.source: checkpoint` +
  `model.image_input.backbone.weights.uri` — composed, validated, and
  recorded in the run's config provenance. See
  `05-models-training-and-heads.md`.
- **Consuming a complete model artifact** → command option. `dojo export`,
  `dojo inspect checkpoint`, `dojo infer`, and `dojo eval` take a finished
  model as `--checkpoint` (a `.ckpt`) or `--model` (an exported `.pt` /
  `.onnx`, with `--export-config` for an ONNX metadata sidecar). The model
  and its input pipeline self-describe from the artifact's embedded
  inference contract (`06-results-artifacts-and-metadata.md`), so no
  `resolved.yaml` is required.

### Sweeps are prepared by `dojo sweep`

There is no `-m` / `--multirun` flag. `dojo train` executes one concrete
training run and rejects authored / composed configs with an active
`sweep.grid`. Use `dojo sweep prepare` to compose a config whose `sweep:`
block defines axes (`sweep.mode: grid` with a non-empty `sweep.grid`) and
write the sweep directory, manifest, and per-run resolved configs.

```bash
dojo sweep prepare experiment=ifcb/baseline \
  'sweep.grid.model.image_input.backbone.architecture.name=[efficientnet_b0,efficientnet_b1]'
```

Initial manual sweep workflow:

```bash
dojo sweep prepare experiment=ifcb/baseline \
  'sweep.grid.optimizer.lr=[1e-4,3e-4]'
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --index 0
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --run-id RUN_ID
dojo sweep status ./runs/ifcb/sweep_results/SWEEP_ID
dojo sweep status ./runs/ifcb/sweep_results/SWEEP_ID --index 0
dojo sweep report ./runs/ifcb/sweep_results/SWEEP_ID
```

`dojo sweep train` is a convenience wrapper for training jobs in a prepared
sweep. It reads the sweep manifest, selects one job by `--index` or
`--run-id`, and runs the same internal training path as
`dojo train --resolved-config JOB_DIR/config/resolved.yaml`. The lower-level
`dojo train --resolved-config ...` path remains valid.

`dojo sweep status` reads the sweep manifest and per-run status files. With
no selector, it lists every sweep index and `run_id`; with `--index` or
`--run-id`, it reports one job. `dojo sweep report` reads the same manifest
and writes sweep-level metrics, figures, and promoted artifacts according to
`sweep_outputs`.

`SWEEP_DIR` is the canonical target for `dojo sweep train`, `status`, and
`report`. For config-parity with other commands, they may also accept
`--resolved-config SWEEP_DIR/config/resolved.yaml`. The standalone
`sweep_manifest.json` path is an internal artifact, not a top-level CLI
input in the initial contract. See `09-sweeps-and-batch-runs.md` for the
sweep block, manifest, and authored -> resolved pipeline.

## `dojo init`

`dojo init` bootstraps an editable local project from packaged Dojo config
templates. It is additive by default: missing files are created and
existing files are skipped. It writes nothing when `--dry-run` is used, and
overwrites existing files only with `--clobber`.

```bash
dojo init
dojo init ./my-dojo-project --minimal --supervised --data
dojo init --ssl --sweep
dojo init --all
dojo init experiment=ifcb/species_baseline
dojo init --config ./my_experiment.yaml
```

The optional positional argument is the project directory, default `.`.
Configs are written under `<project>/configs`. `--data` materializes a
small packaged fixture dataset under `<project>/example-data` and writes
local data config entries that point at the materialized files, so the
starter experiment can run without separate dataset setup.

Initial options:

- `--minimal` — copy only selected starter roots and their referenced config
  dependency closure.
- `--supervised` — include supervised training templates and starter
  experiments.
- `--ssl` — include SSL / DINOv2 templates and starter experiments.
- `--sweep` — include grid-sweep templates and starter experiments.
- `--data` — materialize the small packaged fixture dataset and matching
  local data config.
- `--all` — copy the whole packaged config tree and materialize the
  fixture dataset (it includes `--data`).
- `--config FILE` — use a concrete authored YAML file as a materialization
  root, copying any referenced packaged configs needed by that file.
- `--dry-run` — report create / skip / overwrite actions without writing.
- `--clobber` — overwrite existing files.

Scope flags are additive. If no scope flag or explicit materialization root
is provided, `dojo init` defaults to `--minimal --supervised`. `--all` is
mutually exclusive with `--minimal`, scope flags, and explicit
materialization roots. Dash-free config selectors such as
`experiment=ifcb/species_baseline` are materialization roots for
`dojo init`: Dojo copies that selected packaged config and any packaged
configs it references into the local project.

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
Other SSL methods (SimCLR, VICReg, PMSN, original DINO) are deferred and
not schema values; authoring them fails validation — see
`appendix-deferred-features.md`.

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

A `task.type: snapshot_ensemble` config must include a `training:` block, a
snapshot-capable `scheduler:` block (a top-level peer of `training:`, e.g.
`cosine_warm_restarts`), and an `ensemble:` block (selection strategy +
combine modes). Pydantic validation enforces all three.

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
    strategy: all          # candidates are the run's cycle-end snapshots (implicit source)
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
- show enabled outputs;
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
| `--tabular-stats` | numeric tabular normalization stats + imputation fill values; categorical vocabularies | frozen |
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
| `--content-hash` | true `dataset_content_hash` over image bytes plus declared tabular feature values when present; folds into any full pass for free (see `04-data-and-storage.md`) | frozen (verification) |

Backend-specific:

| Flag | Computes | Kind |
|---|---|---|
| `--bin-lengths` | ROI counts per bin → `bin_lengths` cache (`ifcb_bins` only) | frozen |

Compound and control flags:

- `--stats` — compute the **cheap** frozen aspects (Tier 0 manifest +
  Tier 1 header level): tabular and target stats, imputation, categorical
  vocabularies, `class-map` / `class-counts`, `dimensions`, `bit-depth`, and (`ifcb_bins`)
  `bin-lengths`. Results are displayed (per `--format`) and, when
  `data.stats_cache_uri` is configured, written to that cache; with no cache
  configured it is display-only. The decode-tier frozen aspects are **not**
  run by `--stats` alone because they need a full pixel pass: add
  `--normalization` (the producer for `normalize: {mode: dataset}`) or use
  `--all`. A bare `--stats` reads no pixels and never writes
  `dataset_content_hash`; that hash is recorded only when a full pass
  actually runs (`--normalization` / `--content-hash` / `--all`, see
  `04-data-and-storage.md`).
- `--report` — run all advisory aspects, print / write a human report,
  write no cache.
- `--all` — every aspect, including the decode tier (warns: incurs a
  full-decode pass; this is what also records `dataset_content_hash`).
- `--split train|val|all` — override split selection; otherwise fit
  statistics default to `train` and structural properties to `all`.
- `--sample N|FRACTION` — estimate from a subsample. Permitted for advisory
  aspects; for frozen aspects it marks the result `estimated: true` so a
  sampled statistic is never silently frozen into the contract.
- `--format text|chart|json` — advisory output rendering. `chart` (default
  in an interactive terminal) draws in-terminal bar charts / histograms for
  distributions (`--bucket-histogram`, `--class-counts`, `--imbalance`,
  target / tabular distributions); `text` is plain tabular; `json` is
  machine-readable. Frozen values are displayed too, and — when
  `data.stats_cache_uri` is configured — also written to the cache
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

# write the frozen stats cache (data.stats_cache_uri) consumed at training time
# (add --normalization for dataset normalization mean/std; bare --stats reads no pixels)
dojo inspect dataset data=ifcb/species_manifest --normalization --stats

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

Reports the checkpoint's embedded inference contract — compatibility hashes,
head / preprocessing configuration (see "Portable inference contract",
`06-results-artifacts-and-metadata.md`) — and the checkpoint-hash filename
match. This command requires the Torch stack (`image_classifier_dojo[train]`)
for `.ckpt` deserialization; base installs do not inspect Lightning
checkpoints.

## `dojo infer`

`dojo infer predictions` writes per-sample prediction rows (canonical result
schema, `stage=infer`); `dojo infer embeddings` extracts embeddings. Both
take the model as a `--checkpoint` (`.ckpt`) or `--model` (`.pt` / `.onnx`)
artifact and rebuild the model + input pipeline from its embedded inference
contract; they never re-author `transforms:` (the artifact's
`inference_pipeline` is authoritative). The dataset to run on is the only
required addition: a `data=<group>` selector, an authored `data:` block, or
the `--input PATH` shorthand (backend inferred from the file, default
columns, `split=all`). `data.targets` are optional for inference.

Per-invocation output selection is command options, not config: `--output`,
`--format parquet|csv`, `--heads`, `--embeddings <kind>`. Durable output
configuration (directory, partitioning) lives in `eval_outputs`
(`03-configuration.md`).

`dojo infer embeddings` is the canonical embedding-extraction command;
the legacy `dojo eval embeddings` name is removed.

## `dojo eval`

`dojo eval` scores a finished model against a **labeled** dataset, outside
any training run — e.g. when fresh labeled data arrives and you want metrics
without retraining.

- `dojo eval holdout` — evaluate against a holdout split
  (`stage=holdout_eval`). Takes the model as `--checkpoint` / `--model` and
  rebuilds everything from the inference contract; it **requires**
  `data.targets` and computes metrics from the artifact's
  `objective_summary` plus `target_schema` (no `objectives:` block needed).
- `dojo eval representation` — standalone representation evaluation
  (`stage=representation_eval`). This **builds an encoder** from
  `model.image_input.backbone.weights.source: checkpoint` and attaches the
  probes / projections / clustering configured under `representation_eval:`,
  so it activates `model` / `transforms` / `representation_eval` rather than
  consuming a finished-model artifact. See
  `07-ssl-and-representation-eval.md`.

Results, metrics, and an always-written **eval manifest** (model source +
hashes, dataset identity + split, metric summary) land in `eval_outputs`. By
default an eval run is standalone with its own `run_id`; it can instead
co-locate beside the source run by templating `eval_outputs.dir_template`
with `{source_run_dir}` (`03-configuration.md`). The manifest lets a later
`dojo ensemble` or report step consume eval runs as a candidate / result
source.

`dojo eval knn` and `dojo eval linear-probe` are not primary commands in
the initial implementation; that functionality lives inside
`representation_eval` config sub-blocks accessed through
`dojo eval representation` or scheduled during training.

## `dojo ensemble`

- `dojo ensemble` — ensemble evaluation / inference against a candidate
  manifest or candidate-discovery sources.
- `dojo ensemble candidates` — artifact-inspection / manifest-writing
  command. Does not initialize experiment logging.

Cross-run ensembling is not a distinct mode. It works via `dojo ensemble`
with explicit candidate sources such as `run_dir` and `run_dir_glob`. See
`08-ensembles.md`.

## `dojo sweep`

- `dojo sweep prepare` — compose an authored sweep config, expand
  `sweep.grid`, write the sweep directory, immutable manifest, and per-run
  resolved configs.
- `dojo sweep train` — execute one prepared training job selected by
  `--index` or `--run-id`.
- `dojo sweep status` — inspect all prepared jobs, or one job selected by
  `--index` / `--run-id`, from the manifest plus per-run status files.
- `dojo sweep report` — after required jobs are done, write sweep-level
  metrics, figures, and exports according to `sweep_outputs`.

Automated local sequential and Slurm execution are deferred; the initial
`sweep.execution.mode` is `manual`.

## `dojo export`

Explicit export from a checkpoint, manifest, or already-existing run
output. Artifact types: `torchscript` and `onnx`. See `10-export.md`.

## Config-first usage

Config-aware execution and inspection commands accept dash-free Hydra config
overrides; command I/O is expressed as `--options` (see "CLI architecture
and execution model" above). Examples:

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
dojo sweep prepare experiment=ifcb/baseline \
  'sweep.grid.optimizer.lr=[1e-4,3e-4]'
```

## Cross-References

- `03-configuration.md` — config-key surface every command consumes.
- `04-data-and-storage.md` — dataset backends and `dojo inspect dataset`
  behavior.
- `05-models-training-and-heads.md` — `task.type: supervised` and
  `task.type: snapshot_ensemble` training internals.
- `07-ssl-and-representation-eval.md` — `task.type: ssl`,
  `dojo eval representation`.
- `08-ensembles.md` — `dojo ensemble` / `dojo ensemble candidates` and
  the snapshot-ensemble orchestration step.
- `09-sweeps-and-batch-runs.md` — `dojo sweep prepare`,
  `dojo sweep train`, `dojo sweep status`, and `dojo sweep report`.
- `10-export.md` — `dojo export`, training / ensemble export blocks, and
  sweep-level artifact promotion.
- `12-validation-testing-and-preflight.md` — `dojo inspect config`
  validation tiers and preflight checks.
- `glossary.md` — task type and CLI vocabulary.
