
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
  export:
  metrics:
  figures:
  members:
  manifests:

sweep_outputs:
  dir:
  dir_template:
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
- `model.tabular.fusion` is nested; there is no top-level `model.fusion`.
- The old top-level `outputs:` block is gone. `training_outputs` is its
  replacement.
- `logging` lives under `training_outputs.logging`. Only model-training
  runs initialize experiment logging; `dojo ensemble` and
  `dojo ensemble candidates` do not.
- `ensemble` is the algorithmic block (selection / combine / candidate
  config); `ensemble_outputs` is the corresponding output block.
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

Dojo-owned Python-style template tokens:

- `{experiment.name}`
- `{runtime.run_id}`
- `{model.backbone.name:slug}`
- `{training.batch_size:03}`

Templates resolve **after** config composition, validation, and
runtime-value generation (so generated `run_id` / `sweep_id` values are
available). Prefer this syntax over OmegaConf `${...}` for output paths.
See `09-sweeps-and-batch-runs.md` for the unified runtime ID and output
resolution order.

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
