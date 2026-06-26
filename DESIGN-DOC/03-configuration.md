
# 03. Configuration

## Purpose

Defines the canonical config tree: top-level groups, the peer
`*_outputs` blocks, path-resolution semantics, run-config artifacts, and
overwrite policy. This file is the canonical schema source referenced by
every other file that mentions a config key.

## Schema source of truth

Hydra composes configs from `configs/` groups. Pydantic validates the
composed config and is the runtime contract. CLI overrides apply before
Pydantic validation.

## Canonical root shape

```yaml
experiment:
task:
runtime:
storage:

data:
transforms:

model:
  backbone:
  tabular:
  fusion:
  embedding_adapter:
  heads:

objectives:

ssl:
representation_eval:

training:
optimizer:
scheduler:
checkpointing:

ensemble:

sweep:

output_root:

training_outputs:
  dir:
  dir_template:
  logging:
  results:
  export:
  metrics:
  figures:

ensemble_outputs:
  dir:
  dir_template:
  results:
    member_results:
      mode:
  export:
  metrics:
  figures:
  members:
  manifests:

sweep_outputs:
  dir:
  dir_template:
  enabled:
  collect:
  export:
  metrics:
  figures:
```

Notes:

- `runtime` holds process behavior (`seed`, `precision`, `num_workers`,
  `fast_dev_run`, `autobatch`, `preflight`, `run_id`, `sweep_id`).
  `seed` lives under `runtime`, not at top level.
- `storage` is top-level, peer to `runtime`. It does **not** live under
  `runtime`.
- `optimizer`, `scheduler`, `checkpointing` are top-level peers of
  `training`.
- `model.fusion` is authored-optional and resolved into a concrete block.
  With one enabled model input, fusion is disabled and not included in the
  graph. With more than one enabled input, the initial active fusion type is
  non-parametric `concat`, and `input_order` records concatenation order.
  Authored `input_order`, when present, must be exactly the enabled model
  inputs. In the initial implementation, image input is required and
  tabular input is optional; tabular-only schema is deferred.
- The old top-level `outputs:` block is gone. `training_outputs` is its
  replacement.
- `logging` lives under `training_outputs.logging`. Only model-training
  runs initialize experiment logging; `dojo ensemble` and
  `dojo ensemble candidates` do not.
- `ensemble` is the algorithmic block (selection / combine / candidate
  config); `ensemble_outputs` is the corresponding output block.
- `ensemble_outputs.results.member_results.mode` controls whether selected
  member prediction rows are materialized into `ensemble_results/`; the
  ensemble manifest is written regardless.
- `sweep` is the algorithmic block for sweep generation / search
  (`mode: grid` or deferred `mode: bayesian`); `sweep_outputs` is the
  corresponding sweep-level aggregation output block.

## Minimal supervised example

```yaml
experiment:
  name: ifcb_species_baseline

task:
  type: supervised

runtime:
  seed: 123
  run_id: "{coolname}"
  precision: bf16-mixed
  num_workers: 8
  fast_dev_run: false

storage:
  local_cache_dir: ./.cache/dojo

data:
  backend: parquet_manifest
  manifest_uri: s3://datasets/ifcb/species_manifest.parquet
  image_uri_column: image_uri
  split_column: split
  sample_id_column: roi_id
  targets:
    species:
      column: species_idx
      type: multiclass_classification
      class_names: s3://datasets/ifcb/species_classes.json
      missing_policy: error

model:
  backbone:
    source: torchvision
    name: resnet50
    weights: IMAGENET1K_V2
  tabular:
    enabled: false
  embedding_adapter:
    enabled: false
  heads:
    species:
      type: multiclass_classification
      target: species
      num_classes: 42
      network:
        type: linear

objectives:
  species:
    head: species
    loss: cross_entropy
    metrics: [accuracy, macro_f1, per_class_f1]
    weight: 1.0

training:
  max_epochs: 50
  batch_size: 64

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_epochs: 3

checkpointing:
  monitor: val/species/macro_f1
  mode: max
  save_top_k: 3

output_root: ./runs

training_outputs:
  dir_template: "{experiment.name}/{runtime.run_id}"
```

## Output paths and directory resolution

`output_root` is a single filepath string used as the base for rendered
`training_outputs.dir_template`, `ensemble_outputs.dir_template`, and
`sweep_outputs.dir_template`, and for bare-relative top-level
`*_outputs.dir` values.

Each `*_outputs` block has its own concrete `dir` or `dir_template`. When
`dir_template` is used, the template resolves to the corresponding
`*_outputs.dir` value under `output_root`.

For any `dir` key (top-level or sub-block):

- Absolute paths (`/folder`) are used as provided.
- `./folder` is resolved relative to process CWD.
- Bare-relative paths are resolved against the parent object's resolved
  `dir`. For top-level `*_outputs.dir`, the parent base is `output_root`.
  For sub-block values such as `results.dir: folder` under
  `training_outputs`, the parent base is `training_outputs.dir`.

### Path template syntax

`dir_template` values use Dojo-owned `{...}` tokens, resolved by Dojo
(not OmegaConf `${...}`). The token set is **open**: a token is either a
dotted path into the resolved config or one of a small fixed set of
special tokens.

- **Config-path tokens** — any dotted path into the resolved config; the
  token renders the resolved value at that path. Examples:
  `{experiment.name}`, `{runtime.run_id}`, `{runtime.sweep_id}`,
  `{training.batch_size}`, `{optimizer.lr}`, `{model.backbone.name}`.
- **Special tokens** (not config paths):
  - `{coolname}` — a fresh, unseeded coolname generated per render (see
    `06-results-artifacts-and-metadata.md`).
  - `{timestamp}` — the run's start time as a filesystem-safe string
    (e.g. `2026-06-26_14-30-05`).
  - `{job_num}` — Hydra `hydra.job.num`, the per-job index within a
    sweep.
  - `{ensemble_id}` — the generated selected-ensemble id, available after
    candidate discovery and selection (see `08-ensembles.md`).

#### Format specifiers

A token may carry a `:spec` suffix applied to its resolved value:

- `:slug` — a Dojo **custom** formatter that produces a filesystem-safe
  token: characters in `[a-zA-Z0-9-_.]` are kept as-is (so
  `{optimizer.lr:slug}` renders `0.0003` unchanged and
  `{model.backbone.name:slug}` keeps `resnet50`), spaces become `-`, and
  every other character becomes `_`. Uses the `python-slugify` internally.
- Any standard Python format spec works normally, e.g.
  `{training.batch_size:03}` → `032` and `{optimizer.lr:.0e}` → `3e-04`.

Templates resolve **after** config composition, validation, and
runtime-value generation (so generated `run_id` / `sweep_id` /
`ensemble_id` values are available). Use this syntax over OmegaConf
`${...}` for string composition in configs, typically output paths. See `09-sweeps-and-batch-runs.md` for the
unified runtime ID and output resolution order.

For ensemble runs that use `{ensemble_id}`,
`ensemble_outputs.dir_template` resolves after candidate discovery,
compatibility validation, selection, and selected-ensemble identity
generation. The discovered candidate audit and selected member list are
execution artifacts recorded in the ensemble manifest, not authored-config
entries.

### Snapshot ensemble directory sharing

For `task.type: snapshot_ensemble`, both `training_outputs.dir_template`
and `ensemble_outputs.dir_template` typically resolve to the same
directory. The recommended pattern is to default
`ensemble_outputs.dir_template` to match `training_outputs.dir_template`.

## Existing-run-dir policy

`existing_run_dir` controls behavior when a resolved run directory already
exists:

- `error` — refuse to start. **Default.**
- `overwrite` — delete all extant content of the resolved directory before
  starting.

When multiple output blocks resolve to the same physical directory (e.g.
the snapshot-ensemble case), the policy is evaluated once per resolved
physical directory at command startup. Later phases of the same command
must not re-apply `overwrite` and delete artifacts the earlier phases just
wrote.

## Sweep output aggregation

`sweep_outputs.enabled` controls sweep-level aggregation only. Default:
`true`. When `false`, Dojo still expands and executes the sweep runs, but
skips sweep-level collection, summary metrics, aggregate figures, and
sweep exports.

`sweep_outputs.collect` is a list of metric / artifact collection specs
used by the sweep aggregator. Each item names the metric or artifact to
collect from every concrete run and the per-run source to read. For
metric specs, optional `mode` is `min` or `max` and controls sweep-level
ranking / best-run selection for that collected metric:

```yaml
sweep_outputs:
  enabled: true
  collect:
    - metric: val/species/macro_f1
      source: best
      mode: max
```

Initial `source` values:

- `best` — collect the value associated with the run's best checkpoint.
- `last` — collect the final recorded value for the run.
- `all` — collect all recorded values for that metric across epochs /
  steps.

Sweep aggregation uses the persisted sweep manifest written during sweep
expansion as its run index. The manifest records each concrete run's
resolved output directories and resolved config artifact paths. The
aggregator reads those per-run resolved configs and metric artifacts from
the recorded locations rather than discovering runs by scanning
directories.

## Run config artifacts

Every run writes its resolved configuration into `config/` under the
resolved output directory:

```text
config/composed.yaml      # after config composition and CLI overrides
config/resolved.yaml      # fully resolved with generated values + defaults
config/resolved.json
config/cli.txt            # invoked command and overrides
config/overrides.txt
config/sweep_values.txt   # sweep members only
```

Generated runtime values (`run_id`, `sweep_id`, etc.) appear in the
resolved-config artifacts.

## Template + sweep example

```yaml
runtime:
  run_id: "{coolname}"
  sweep_id: "{coolname}"

output_root: ./runs

model:
  backbone:
    source: torchvision
    name: efficientnet_b0,resnet50

training:
  batch_size: 32,64

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.backbone.name:slug}/bs{training.batch_size:03}/
  existing_run_dir: error

sweep_outputs:
  dir_template: "{experiment.name}/sweep_results/{runtime.sweep_id}"
  enabled: true
  collect:
    - metric: val/species/macro_f1
      source: best
      mode: max
```

## Cross-References

- `02-cli-and-task-types.md` — `dojo inspect config` validates and renders
  this schema; `task.type` semantics.
- `06-results-artifacts-and-metadata.md` — uses `output_root` /
  `*_outputs` resolution rules and consumes the `results:` sub-block.
- `04-data-and-storage.md` — `data:` and `storage:` blocks.
- `05-models-training-and-heads.md` — `model:`, `transforms:`,
  `training:`, `optimizer:`, `scheduler:`, `checkpointing:`, `objectives:`.
- `07-ssl-and-representation-eval.md` — `ssl:` and `representation_eval:`.
- `08-ensembles.md` — `ensemble:` and `ensemble_outputs:`.
- `09-sweeps-and-batch-runs.md` — `sweep:`, `sweep_outputs:`, and Hydra
  integration.
- `glossary.md` — config-key vocabulary.
