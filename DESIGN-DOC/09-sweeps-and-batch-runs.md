
# 09. Sweeps and Batch Runs

## Purpose

Defines Hydra-driven sweeps, batch-run-style grid sweeps, deferred
Bayesian / AutoML sweep schema, and the `sweep_outputs:` block. Sweeps
can explore training hyperparameters, ensemble strategy / combine-mode
combinations, random-seed sensitivity, or feed candidate manifests into
a subsequent ensembling step. Bayesian / AutoML HPO is deferred.

## Hydra multirun

Use Hydra multirun (`-m`) for sweeps. Dojo treats Hydra as the sweep
expansion mechanism, while Dojo owns runtime IDs, output-directory
resolution, existing-directory policy, and canonical artifacts.

Example sweep:

```bash
dojo train -m \
  experiment=ifcb/experimentA \
  model.backbone.source=torchvision,timm \
  model.backbone.name=resnet50,convnext_tiny \
  optimizer.lr=1e-4,3e-4 \
  training.batch_size=32,64
```

### Hydra directories vs. Dojo output directories

Dojo's run directory (resolved from `training_outputs.dir_template`
under `output_root`) is the canonical location for checkpoints, exports,
metrics, and results. It is independent of Hydra's working directory.

Hydra still needs directories for its own launch metadata, per-job
bookkeeping, and Hydra logs. Those are **not** Dojo run directories.
Dojo keeps them under an `_hydra/` coordination area below
`output_root`, while canonical Dojo artifacts go only to the resolved
`training_outputs.dir`, `ensemble_outputs.dir`, and `sweep_outputs.dir`.

Default Hydra config:

```yaml
# configs/hydra/default.yaml
hydra:
  run:
    dir: ${output_root}/_hydra/single/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ${output_root}/_hydra/sweeps/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
  job:
    chdir: false
```

Key points:

- `chdir: false` keeps the process CWD at the project root. All Dojo
  paths are absolute or relative to `output_root`. This avoids the
  Hydra footgun where `os.getcwd()` silently changes per job.
- `hydra.run.dir`, `hydra.sweep.dir`, and `hydra.sweep.subdir` are only
  for Hydra metadata / logs. Dojo code must not derive checkpoints,
  results, exports, metrics, or figures from those paths.
- For sweeps, `hydra.sweep.dir` is the parent coordination directory
  and `hydra.sweep.subdir` is the per-job coordination directory. Dojo
  artifacts still land under each job's resolved `training_outputs.dir`
  / `ensemble_outputs.dir`.
- `runtime.sweep_id` is generated once per sweep before concrete runs
  are expanded. `runtime.run_id` is generated once per concrete run
  after its sweep-axis values are known.
- Hydra's `hydra.job.num` and `hydra.job.id` may be used in
  `runtime.run_id` templates, but Dojo does not rely on pattern
  inspection to prove uniqueness. For sweep invocations, Dojo generates
  all run IDs before any run starts and raises an error if collisions
  exist.

> [!NOTE]
> Hydra config uses `${...}` OmegaConf / Hydra interpolation notation,
> evaluated by Hydra while it sets up its own bookkeeping directories.
> Dojo-owned output templates omit the `$` and use `{...}` patterns
> resolved by Dojo, not OmegaConf / Hydra interpolation.

## Sweep definition block

`sweep:` is the top-level algorithmic block for sweep generation. It is
separate from `sweep_outputs:`, which controls sweep-level aggregation
artifacts.

```yaml
sweep:
  mode: grid
  conflict_policy: default
  grid:
    runtime.seed: [101, 102, 103]
    optimizer.lr: [1.0e-4, 3.0e-4]
    training.batch_size: [32, 64]
    model.backbone.name: [resnet50, convnext_tiny]
```

`sweep.mode` values:

- `grid` — functional; explicit cartesian product over parameter lists.
- `bayesian` — schema slot only in the initial implementation; runtime
  raises `NotImplementedError`.

### Grid sweep syntax

Inside `sweep.grid`, values are YAML lists. This is intentionally
different from Hydra CLI comma syntax:

```bash
dojo train -m optimizer.lr=1e-4,3e-4 training.batch_size=32,64
```

CLI comma values are accepted at the command surface, but Dojo
normalizes all sweep axes into top-level `sweep.grid` during config
compilation. Source configs should use YAML lists under `sweep.grid`
rather than comma-separated strings.

Sweep axes found outside the top-level `sweep:` block are also
normalized into `sweep.grid` during config compilation. After
normalization, each concrete run receives ordinary resolved scalar
config values at the target paths.

### Batch-run-style grid sweeps

A batch run is a grid sweep where the only intentional run-varying
parameter is `runtime.seed`. This is useful for measuring model
sensitivity to randomness while holding architecture, data, training
parameters, and evaluation settings fixed.

```yaml
sweep:
  mode: grid
  grid:
    runtime.seed: [101, 102, 103, 104, 105]
```

Sweep aggregation can then summarize metrics across seeds, for example
with box plots / five-number summaries for `macro_f1` or per-class F1.

### Sweep conflict policy

`sweep.conflict_policy` controls what happens when a swept target path
also has a value in the non-sweep config:

- `default` — default. If the target config value is scalar, the sweep
  value overrides it for each concrete run. If both locations define
  different sweep ranges for the same target path, config compilation
  raises an error.
- `strict` — swept target fields must be missing or null outside
  `sweep:`. If a target path has any non-sweep value, config
  compilation raises an error.

CLI overrides supersede both the target config and `sweep:` definitions.
If a CLI override supplies sweep values, those values are normalized into
`sweep.grid`.

### Active run metadata

`sweep.active_run` is generated only in resolved config artifacts and
runtime config objects. It should not be written in source configs.

Example resolved-config fragment for one concrete run:

```yaml
sweep:
  mode: grid
  conflict_policy: default
  active_run:
    index: 3
    values:
      runtime.seed: 103
      optimizer.lr: 0.0001
      training.batch_size: 64
      model.backbone.name: convnext_tiny
```

### Bayesian sweep schema

Bayesian / AutoML HPO is deferred. The schema slot is reserved so config
shape has a clear destination when the runtime is later implemented:

```yaml
sweep:
  mode: bayesian
  conflict_policy: default
  bayesian:
    engine: optuna
    metric: val/loss
    mode: min
    max_trials: 50
    sampler: tpe
    params:
      optimizer.lr:
        type: float
        low: 1.0e-5
        high: 1.0e-3
        log: true
      optimizer.weight_decay:
        type: float
        low: 0.0
        high: 0.1
      training.batch_size:
        type: categorical
        values: [16, 32, 64]
      model.backbone.name:
        type: categorical
        values: [resnet50, convnext_tiny, efficientnet_b0]
```

Initial Bayesian parameter types:

- `categorical` — explicit `values`.
- `int` — `low`, `high`, optional `step`, optional `log`.
- `float` — `low`, `high`, optional `step`, optional `log`.
- `bool` — boolean choice.

Runtime support for `mode: bayesian` raises `NotImplementedError` in the
initial implementation; see `appendix-deferred-features.md`.

## Runtime ID and output resolution order

The following order applies to both single-run and sweep invocations.
Sections that reference sweep-level state fall through when there is no
active sweep.

1. **Address sweep-level state**
   1. Determine whether this invocation is a Hydra multirun / sweep.
   2. If there is no active sweep:
      1. Set `runtime.sweep_id = null` unless explicitly configured.
      2. Set `sweep_hash = null`.
      3. Skip sweep-level aggregation setup.
   3. If there is an active sweep:
      1. Normalize sweep axes from source config and CLI inputs into
         the top-level `sweep:` block.
      2. Compose the sweep launcher definition: base config plus
         normalized sweep definition.
      3. Compute `sweep_hash` from stable sweep-definition inputs:
         base config, sweep mode, sweep axes / search params, and
         explicit sweep metadata.
      4. Exclude generated IDs, output paths, `output_root`, and all
         `*_outputs` blocks from `sweep_hash`.
      5. Generate `runtime.sweep_id` once for the whole sweep:
         manual value as-is; template rendered from non-generated
         sweep / base config fields; `{coolname}` as a fresh unseeded
         coolname; unset value as a seedname from `sweep_hash`.
      6. Expand the sweep into concrete per-run jobs.
      7. Pass finalized `runtime.sweep_id` and `sweep_hash` into every
         run.
2. **Determine run-level hashes and run IDs for each run**
   1. Compose each concrete run config:
      1. Start from the base config.
      2. Apply CLI overrides.
      3. If this is a sweep run, apply that job's sweep-axis values.
      4. Include finalized sweep-level values when present.
   2. Validate the composed config structurally with Pydantic:
      1. ID fields may still contain Dojo template strings.
      2. Output paths may still be unresolved.
   3. Compute stable run-level hashes:
      1. Compute `config_hash` from the concrete resolved config.
      2. Exclude runtime-generated values, output paths, `output_root`,
         and all `*_outputs` blocks.
      3. Compute any cheap / local config-time hashes available, such
         as dataset hash or compatibility hashes.
   4. Generate `runtime.run_id` for each concrete run:
      1. Manual value: use as-is.
      2. Template value: render from concrete non-generated config
         fields plus finalized sweep-level values, if present.
      3. `{coolname}`: fresh unseeded coolname per run.
      4. Unset value: use the configured default run-id template.
   5. For sweep invocations, check all generated `run_id` values for
      collisions before any run starts:
      1. If collisions exist, raise an error.
      2. The error shows the colliding `run_id` values and affected
         sweep jobs.
      3. The error suggests adding a uniqueness-pattern value such as
         `hydra.job.num`, `hydra.job.id`, or a realized sweep value
         such as `runtime.seed`.
3. **Resolve per-run output directories**
   1. For each concrete run, resolve output templates:
      1. `training_outputs.dir_template` resolves to
         `training_outputs.dir`.
      2. `ensemble_outputs.dir_template` resolves to
         `ensemble_outputs.dir`.
      3. `sweep_outputs.dir_template` is not resolved here except for
         commands that produce per-run sweep-scoped artifacts.
   2. Resolve paths:
      1. Absolute paths are used as-is.
      2. `./path` resolves relative to process CWD.
      3. Bare-relative top-level `*_outputs.dir` values resolve under
         `output_root`.
      4. Bare-relative sub-block dirs resolve under their parent output
         block's resolved `dir`.
   3. Apply existing-directory policy:
      1. Evaluate `existing_run_dir` once per resolved physical
         directory.
      2. If multiple output blocks share one physical directory, apply
         the policy once.
      3. Later phases in the same command must not re-apply
         `overwrite`.
4. **Resolve sweep-level output directories**
   1. If there is no active sweep, skip this section.
   2. Resolve `sweep_outputs.dir_template` to `sweep_outputs.dir` once
      for the sweep.
   3. Resolve sweep-output sub-block paths under `sweep_outputs.dir`.
   4. Apply the sweep-output existing-directory policy once for the
      resolved sweep directory.
   5. Keep sweep-level aggregation outputs separate from per-run
      `training_outputs` / `ensemble_outputs`.
5. **Finalize runtime config artifacts**
   1. For each run, write:
      1. `config/composed.yaml`
      2. `config/resolved.yaml`
      3. `config/resolved.json`
      4. `config/cli.txt`
      5. `config/overrides.txt`
      6. `config/sweep_values.txt` for sweep runs.
   2. For active sweeps, write sweep-level config artifacts under
      `sweep_outputs.dir/config/`, including:
      1. sweep launcher / base config;
      2. sweep axes and value lists;
      3. finalized `sweep_id`;
      4. `sweep_hash`;
      5. generated run IDs and resolved per-run output dirs.
6. **Execute runs and sweep aggregation**
   1. Execute each concrete run.
   2. Result rows include `run_id`, `config_hash`, and related run
      provenance.
   3. Sweep-produced result rows also include `sweep_id` and
      `sweep_hash`.
   4. After all sweep runs finish, execute configured
      `sweep_outputs` aggregation:
      1. collect per-run metrics / artifacts;
      2. write sweep-level metrics and figures;
      3. write sweep-level exports only if explicitly configured.

## `sweep_outputs:` block

Peer to `training_outputs:` and `ensemble_outputs:`. Holds **sweep-level
aggregation** outputs.

Sub-blocks: `dir_template`, `export`, `metrics`, `figures`. There is
**no** `results` sub-block — per-row data comes from the underlying
per-job `training_outputs` / `ensemble_outputs`.

Default on-disk sub-directories under the resolved `sweep_outputs.dir`:

```text
sweep_outputs.dir/
  config/
  exports/
  metrics/
  figures/
```

### Sweep ID

`runtime.sweep_id` may be manually set, rendered from a template (e.g.
`{coolname}`), or fall back to a seedname from `sweep_hash`. See
`06-results-artifacts-and-metadata.md` for the full ID / hash rules.

Result rows produced as part of a sweep carry both `sweep_id` and
`sweep_hash` provenance columns, and may include `sweep_id` in their
`partition_by` list.

### Sweep output example

```yaml
runtime:
  sweep_id: "{coolname}"

model:
  backbone:
    source: torchvision
    name: resnet50

training:
  batch_size: 32

optimizer:
  lr: 0.0003

sweep:
  mode: grid
  conflict_policy: default
  grid:
    model.backbone.name: [resnet50, efficientnet_b0, convnext_tiny]
    training.batch_size: [32, 64]
    optimizer.lr: [0.0003, 0.0001]

output_root: ./runs

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.backbone.name:slug}/bs{training.batch_size:03}/lr{optimizer.lr:slug}/

sweep_outputs:
  dir_template: >-
    {experiment.name}/sweep_results/{runtime.sweep_id}
  enabled: true
  collect:
    - metric: val/species/macro_f1
      source: best
      mode: max
    - metric: val/species/per_class_f1
      source: best
      mode: max
  metrics:
    summary_csv: true
    summary_json: true
    summary_parquet: true
  figures:
    metric_rankings: true
    per_class_heatmap: true
```

Sweep aggregation outputs include:

- compare best / final F1 across best / final epoch for model trainings;
- compare per-class F1 across best / final epoch;
- comparison metrics in CSV / JSON / Parquet;
- comparison figures.

## Sweeps over ensembling

Hydra sweeps can explore ensemble selection strategy / combine-mode
combinations against a candidate manifest:

```bash
dojo ensemble -m \
  experiment=ifcb/ensemble_search \
  ensemble.selection.strategy=top_k,greedy_forward_selection \
  ensemble.inference.combine.classification=probabilities_mean,logits_mean
```

Sweep aggregation summarizes ensemble metrics across the swept axes.

## Sweeps as candidate-source feeders

A training sweep may write to a shared candidate manifest directory that
a subsequent `dojo ensemble` invocation reads:

```bash
dojo train -m experiment=ifcb/sweep_for_ensembling
dojo ensemble candidates \
  experiment=ifcb/post_sweep_candidates \
  ensemble.candidates.sources.0.type=run_dir_glob \
  ensemble.candidates.sources.0.uri_glob=./runs/ifcb_sweep/*/ \
  ensemble_outputs.manifests.dir=./shared_manifests
dojo ensemble \
  experiment=ifcb/post_sweep_ensemble \
  ensemble.candidates.manifest_uri=./shared_manifests/ifcb_post_sweep_candidates.json
```

This is the supported pattern for "use a sweep's outputs as ensemble
candidates" — `dojo ensemble candidates` over an explicit
run-directory glob, not a registry-driven discovery (deferred).

## Cross-References

- `02-cli-and-task-types.md` — `-m` works with every command family.
- `03-configuration.md` — `output_root`, `*_outputs.dir_template`
  resolution, run / sweep directory collision handling.
- `06-results-artifacts-and-metadata.md` — `sweep_id`, `sweep_hash`,
  partition columns; `metrics/` and `figures/` sub-block semantics
  shared with `training_outputs` and `ensemble_outputs`.
- `08-ensembles.md` — `dojo ensemble candidates` for candidate manifests
  built from sweep outputs.
- `appendix-deferred-features.md` — Bayesian / AutoML HPO;
  registry-based candidate discovery.
- `glossary.md` — `sweep_id`, `sweep_hash` definitions.
