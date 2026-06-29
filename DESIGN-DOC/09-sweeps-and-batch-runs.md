
# 09. Sweeps and Batch Runs

## Purpose

Defines config-defined sweeps (Dojo-owned expansion over the Hydra
Compose API), batch-run-style grid sweeps, deferred Bayesian / AutoML
sweep schema, and the `sweep_outputs:` block. Sweeps
can explore training hyperparameters, ensemble strategy / combine-mode
combinations, random-seed sensitivity, or feed candidate manifests into
a subsequent ensembling step. Bayesian / AutoML HPO is deferred.

## Sweep preparation and manual execution

Dojo composes configs through the Hydra **Compose API** (`hydra.compose`),
not `@hydra.main`, so there is no Hydra launcher, no Hydra-managed working
directory, and no `chdir`. A sweep is prepared when `dojo sweep prepare`
composes a config whose `sweep:` block defines axes (see "Sweep definition
block" below). Dojo expands the cartesian product itself and owns runtime
IDs, output-directory resolution, run-directory collision handling, and
all canonical artifacts.

`dojo train` executes one concrete training run. It rejects authored /
composed configs with an active `sweep.grid`; use `dojo sweep prepare`
first, then run concrete jobs from the prepared sweep.

A sweep is prepared from either an experiment config that includes a
`sweep:` block, or axes overridden onto `sweep.grid` from the command line
(dash-free config overrides):

```bash
# the experiment config carries a sweep: block
dojo sweep prepare experiment=ifcb/sweep_lr_bs

# or supply axes as config overrides (Hydra list-value syntax)
dojo sweep prepare experiment=ifcb/experimentA \
  sweep.mode=grid \
  'sweep.grid.optimizer.lr=[1e-4,3e-4]' \
  'sweep.grid.training.batch_size=[32,64]'
```

There is no `-m` / `--multirun` flag and no CLI comma-list sweep
shorthand; the `sweep:` block is the single sweep definition. No `+` is
needed for `sweep.grid.*` overrides because `sweep.grid` is an open mapping
in the root schema.

Process CWD never changes between runs. All Dojo paths are absolute or
resolved relative to `output_root` (`03-configuration.md`), so there is no
Hydra `os.getcwd()` footgun and no `_hydra/` coordination area — Dojo's
resolved `training_outputs.dir`, `ensemble_outputs.dir`, and
`sweep_outputs.dir` are the only output locations.

- `runtime.sweep_id` is generated once per sweep before concrete runs are
  expanded. `runtime.run_id` is generated once per concrete run after its
  sweep-axis values are known.
- Dojo generates all run IDs before any run starts and raises an error if
  collisions exist (see the resolution order below). Uniqueness comes from
  realized sweep values or the `{job_num}` sweep index
  (`sweep.active_run.index`), not from Hydra job metadata.

> [!NOTE]
> OmegaConf `${...}` interpolation is resolved by OmegaConf during
> composition. Dojo-owned output templates omit the `$` and use `{...}`
> patterns resolved by Dojo after composition, validation, and
> runtime-value generation (`03-configuration.md`). The two interpolation
> layers do not overlap.

> [!NOTE]
> **Deferred automated runners.** The initial execution mode is manual.
> Automated local sequential execution and Slurm / HPC queue submission
> are deferred to P4.16. Dojo owns expansion and artifact layout regardless
> of the later execution mechanism.

Initial sweep execution is manual:

```bash
dojo sweep prepare experiment=ifcb/experimentA \
  'sweep.grid.optimizer.lr=[1e-4,3e-4]'
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --index 0
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --run-id RUN_ID
dojo sweep status ./runs/ifcb/sweep_results/SWEEP_ID
dojo sweep report ./runs/ifcb/sweep_results/SWEEP_ID
```

`dojo sweep train` reads the sweep manifest, selects one training job by
`--index` or `--run-id`, and invokes the same internal training path as
`dojo train --resolved-config JOB_DIR/config/resolved.yaml`. Per-run status
is written by the run itself; the central manifest remains the immutable
prepared job index.

## Sweep definition block

`sweep:` is the top-level algorithmic block for sweep generation. It is
separate from `sweep_outputs:`, which controls sweep-level reporting
artifacts.

```yaml
sweep:
  mode: grid
  conflict_policy: default
  execution:
    mode: manual
  grid:
    runtime.seed: [101, 102, 103]
    optimizer.lr: [1.0e-4, 3.0e-4]
    training.batch_size: [32, 64]
    model.image_input.backbone.architecture.name: [resnet50, convnext_tiny]
```

`sweep.mode` values:

- `grid` — functional; explicit cartesian product over parameter lists.
- `bayesian` — schema slot only in the initial implementation; runtime
  raises `NotImplementedError`.

`sweep.execution.mode` values:

- `manual` — functional initial mode. `dojo sweep prepare` writes the
  prepared sweep artifacts; users run concrete jobs explicitly.
- `local_sequential` — deferred to P4.16.
- `slurm` — deferred to P4.16.

### Grid sweep syntax

Inside `sweep.grid`, each normalized key is a target config path and each
value is a YAML list; the concrete runs are the cartesian product of those
lists. Authored YAML may use flat dotted keys:

```yaml
sweep:
  grid:
    optimizer.lr: [1.0e-4, 3.0e-4]
```

CLI overrides naturally compose as nested mappings, e.g.
`'sweep.grid.optimizer.lr=[1e-4,3e-4]'`. After Hydra composition and before
validation, Dojo normalizes `sweep.grid` by flattening nested leaves under
`sweep.grid` with dots, so that override becomes the target path
`optimizer.lr`.

The `sweep:` block is the single sweep definition: there is no CLI
comma-list shorthand and no normalization of axes scattered elsewhere in
the config. After expansion, each concrete run receives ordinary resolved
scalar config values at the target paths. `dojo train` rejects active
`sweep.grid` values in authored / composed configs; `sweep.grid` is only
expanded by `dojo sweep prepare`.

### Commas, lists, and shell quoting

Comma handling differs between CLI overrides and config files:

- **CLI overrides** use Hydra's override grammar — Dojo relies on it for
  all `key=value` composition. The whole grammar is in play, with one
  exception, the *bare top-level comma*:

  - **Bare top-level comma = multirun, which Dojo rejects the direct use of.** 
    In Hydra's grammar a comma that sits directly in the value
    (`optimizer.lr=1e-4,3e-4`) means "run once per listed value" — a
    *multirun sweep*. It is the only construct that requires Hydra's
    multirun machinery, and Dojo never runs _hydra_ multirun: it composes branching/grids of single
    configs through the Compose API. So this comma styling override **errors during
    composition**. Sweeping is handled differently.
  - **Sweeps come only from the `sweep:` config block**.
  - **Commas inside a value are fine.** The rest of the grammar works
    normally, including commas that are structurally part of a value: list
    literals `[a,b]`, dict literals `{a:1,b:2}`, and Hydra-quoted strings
    `key='a,b'`. These are not top-level commas, so they never trigger
    multirun.

  The trap: `optimizer.lr=1e-4,3e-4` (top-level comma → errors) and
  `optimizer.lr=[1e-4,3e-4]` (list value → legal) look almost identical but
  mean opposite things. Author sweep axes as the bracketed form on
  `sweep.grid`, e.g. `'sweep.grid.optimizer.lr=[1e-4,3e-4]'`, never
  `optimizer.lr=1e-4,3e-4`.
- **Shell quoting:** single-quote any override token containing `[...]` or
  `{...}` so the shell does not glob- or brace-expand it before Hydra sees
  it (e.g. `'sweep.grid.training.batch_size=[32,64]'`). Tokens without
  brackets (`sweep.mode=grid`) need no quotes.
- **Config files** follow ordinary YAML: commas separate elements only
  inside flow collections (`[a, b]`, `{a: 1}`) and are literal characters
  in plain or quoted scalars. Hydra's comma-as-sweep grammar is a
  CLI-override behavior only — it never applies inside YAML, so
  `sweep.grid` axes are written as ordinary YAML lists.

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

Sweep reporting can then summarize metrics across seeds, for example
with box plots / five-number summaries for `f1_macro` or per-class F1.

### Sweep conflict policy

`sweep.conflict_policy` controls what happens when a swept target path
also has a value in the non-sweep config:

- `default` — default. A `sweep.grid` entry for a target path overrides
  any value at that path elsewhere in the composed config, per concrete
  run.
- `strict` — the target path must be absent or null outside `sweep.grid`;
  any non-sweep value there raises a config-compilation error.

CLI config overrides supersede both the target config and `sweep:`
definitions. Sweep axes supplied on the command line are written directly
onto `sweep.grid` (e.g. `'sweep.grid.optimizer.lr=[1e-4,3e-4]'`), not
inferred from comma-separated scalar overrides.

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
      model.image_input.backbone.architecture.name: convnext_tiny
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
      model.image_input.backbone.architecture.name:
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

## Sweep preparation, training, status, and reporting order

Single-run `dojo train` uses the same run-level resolution rules as a
prepared sweep job. If an authored / composed config has a non-empty
`sweep.grid`, `dojo train` errors and points the user to
`dojo sweep prepare`.

1. **Prepare the sweep with `dojo sweep prepare`**
   1. Compose the authored config and CLI overrides through Hydra.
   2. Validate `sweep.mode`, `sweep.grid`, and `sweep.execution`.
      `sweep.execution.mode: manual` is functional; deferred modes raise
      `NotImplementedError`.
   3. Compute `sweep_hash` from stable sweep-definition inputs: base
      config, sweep mode, sweep axes / search params, and explicit sweep
      metadata. Exclude generated IDs, output paths, `output_root`, and
      all `*_outputs` blocks.
   4. Generate `runtime.sweep_id` once for the sweep: manual value as-is;
      template rendered from non-generated sweep / base config fields or
      already-computed hash sources such as `{coolname:sweep_hash}`;
      `{coolname:noseed}` as a fresh unseeded coolname; unset value falls
      back to `{coolname:sweep_hash}`. Bare `{coolname}` is deterministic
      from `runtime.seed` and is not the default.
   5. Expand the cartesian product into concrete jobs.
   6. For each job, apply that job's sweep-axis values, validate the
      concrete config, compute `config_hash`, generate `runtime.run_id`
      (default `{coolname:noseed}`; hash-seeded templates such as
      `{coolname:config_hash}` can render because `config_hash` already
      exists), and resolve per-run output directories in memory.
   7. Check all generated `run_id` values and resolved per-run output
      directories for collisions before writing any per-run artifacts.
      Collision errors name the affected sweep indices and suggest adding
      `{job_num}` or a realized sweep value such as `runtime.seed` to the
      run-id / directory template.
   8. Write per-run config artifacts:
      - `config/composed.yaml`
      - `config/resolved.yaml`
      - `config/resolved.json`
      - `config/cli.txt`
      - `config/overrides.txt`
      - `config/sweep_values.txt`
   9. Resolve `sweep_outputs.dir_template` to `sweep_outputs.dir` once for
      the sweep, then resolve sweep-output sub-block paths under it.
   10. Write sweep-level config artifacts under `sweep_outputs.dir/config/`:
      - `composed.yaml` — sweep-level composed config with `sweep.grid`;
      - `resolved.yaml` — sweep-level resolved config with finalized
        `runtime.sweep_id`, `sweep_hash`, and output paths, but no
        concrete `sweep.active_run`;
      - `resolved.json`;
      - `sweep_manifest.json` — immutable concrete job index;
      - `cli.txt`;
      - `overrides.txt`.

2. **Run one training job with `dojo sweep train`**
   1. Read `SWEEP_DIR/config/sweep_manifest.json`.
   2. Select exactly one job by `--index` or `--run-id`.
   3. Invoke the same internal path as
      `dojo train --resolved-config JOB_DIR/config/resolved.yaml`.
   4. Use the same resolved-config guard as `dojo train`: if the selected
      run directory contains artifacts beyond prepared config / status
      files, require explicit `--resume` or `--clobber` behavior.
   5. Write a simple per-run status file at `JOB_DIR/status.json`.
      `pending` is inferred from the manifest when the file does not
      exist. Written states are `initializing`, `training`, `exporting`,
      `done`, and `failed`.
   6. Result rows include `run_id`, `config_hash`, and related run
      provenance. Sweep-produced result rows also include `sweep_id` and
      `sweep_hash`.

3. **Check status with `dojo sweep status`**
   1. `dojo sweep status SWEEP_DIR` lists all sweep indices, `run_id`
      values, key sweep values, and current status.
   2. `--index` or `--run-id` reports one job.
   3. Status is read live from the immutable manifest, per-run
      `status.json` files, and expected run artifacts. It does not mutate
      the manifest and does not write a sweep-level status cache.
   4. The command clearly reports whether all prepared jobs are `done`.

4. **Write sweep reports with `dojo sweep report`**
   1. Read `SWEEP_DIR/config/sweep_manifest.json`.
   2. Require all non-skipped manifest jobs to be `done`; otherwise list
      pending / running / failed jobs and exit without writing partial
      report artifacts.
   3. If `sweep_outputs.enabled: false`, exit cleanly after confirming
      reporting is disabled.
   4. Read each run's resolved config and metric artifacts from paths
      recorded in the manifest.
   5. Apply `sweep_outputs.collect`, write sweep-level metrics and
      figures, and promote / export sweep-level artifacts only if
      explicitly configured.

## `sweep_outputs:` block

Peer to `training_outputs:` and `ensemble_outputs:`. Holds **sweep-level
reporting** outputs.

Sub-blocks: `dir_template`, `enabled`, `collect`, `artifacts`, `metrics`,
`figures`. There is **no** `results` sub-block — per-row data comes from
the underlying per-job `training_outputs` / `ensemble_outputs`.

`sweep_outputs.enabled` defaults to `true`. When `false`, Dojo still
prepares the sweep and concrete jobs may still run, but `dojo sweep report`
skips sweep-level collection, summary metrics, aggregate figures, and
sweep artifact promotion / export.

`sweep_outputs.collect` is a list of metric / artifact collection specs.
Each item names what to collect from every concrete run and which
per-run source to read. For metric specs, optional `mode` is `min` or
`max` and controls sweep-level ranking / best-run selection for that
collected metric:

```yaml
sweep_outputs:
  collect:
    - metric: val/species/f1_macro
      source: best
      mode: max
```

Initial `source` values:

- `best` — collect the value associated with the run's best checkpoint.
  By default this uses the run's `checkpointing.monitor` /
  `checkpointing.mode`; an explicit collect `mode` may be supplied for
  aggregation ranking.
- `last` — collect the final recorded value for the run.
- `all` — collect all recorded values for that metric across epochs /
  steps, for trend plots or post-hoc summaries.

Sweep reporting does not discover runs by scanning directories. During
sweep preparation, Dojo writes `sweep_outputs.dir/config/sweep_manifest.json`
as the immutable job index. The manifest records each concrete run's
index, `run_id`, `config_hash`, realized sweep values, resolved
`training_outputs.dir` / `ensemble_outputs.dir`, resolved config artifact
paths, and per-run status path. `dojo sweep status` and
`dojo sweep report` read the manifest, then read each run's
`config/resolved.yaml`, `status.json`, and metric artifacts from those
recorded locations.

`sweep_outputs.artifacts` is a report-time artifact promotion block. It
does not create a separate "sweep model"; it copies or converts selected
model artifacts from completed concrete jobs recorded in
`sweep_manifest.json`, such as the best run's best checkpoint according to
a configured collected metric.

Default on-disk sub-directories under the resolved `sweep_outputs.dir`:

```text
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
```

### Sweep ID

`runtime.sweep_id` may be manually set, rendered from a coolname source
template such as `{coolname:sweep_hash}`, or fall back to
`{coolname:sweep_hash}` when unset. See `06-results-artifacts-and-metadata.md`
for the full ID / hash rules.

Result rows produced as part of a sweep carry both `sweep_id` and
`sweep_hash` provenance columns, and may include `sweep_id` in their
`partition_by` list.

### Sweep output example

```yaml
runtime:
  sweep_id: "{coolname:sweep_hash}"

model:
  image_input:
    backbone:
      architecture:
        source: torchvision
        name: resnet50
      weights:
        source: library
        name: DEFAULT

training:
  batch_size: 32

optimizer:
  lr: 0.0003

sweep:
  mode: grid
  conflict_policy: default
  execution:
    mode: manual
  grid:
    model.image_input.backbone.architecture.name: [resnet50, efficientnet_b0, convnext_tiny]
    training.batch_size: [32, 64]
    optimizer.lr: [0.0003, 0.0001]

output_root: ./runs

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.image_input.backbone.architecture.name:slug}/bs{training.batch_size:03}/lr{optimizer.lr:slug}/

sweep_outputs:
  dir_template: >-
    {experiment.name}/sweep_results/{runtime.sweep_id}
  enabled: true
  collect:
    - metric: val/species/f1_macro
      source: best
      mode: max
    - metric: val/species/f1_per_class/*
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

Sweep report outputs include:

- compare best / final F1 across best / final epoch for model trainings;
- compare per-class F1 across best / final epoch;
- comparison metrics in CSV / JSON / Parquet;
- comparison figures.

## Sweeps over ensembling

Sweeps can explore ensemble selection strategy / combine-mode
combinations against a candidate manifest, via a `sweep:` block or
`sweep.grid.*` overrides:

```bash
dojo sweep prepare experiment=ifcb/ensemble_search \
  sweep.mode=grid \
  'sweep.grid.ensemble.selection.strategy=[top_k,greedy_forward_selection]' \
  'sweep.grid.ensemble.inference.combine.classification=[probabilities_mean,logits_mean]'
```

Each prepared concrete ensemble job can then run through the lower-level
command-specific replay path, for example
`dojo ensemble --resolved-config JOB_DIR/config/resolved.yaml`. The
`dojo sweep train` convenience wrapper is for training jobs. Sweep status
and report commands still operate on the prepared sweep directory and
summarize ensemble metrics across the swept axes.

## Sweeps as candidate-source feeders

A training sweep may write to a shared candidate manifest directory that
a subsequent `dojo ensemble` invocation reads:

```bash
dojo sweep prepare experiment=ifcb/sweep_for_ensembling
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --index 0
dojo sweep train ./runs/ifcb/sweep_results/SWEEP_ID --index 1
dojo sweep report ./runs/ifcb/sweep_results/SWEEP_ID
dojo ensemble candidates \
  experiment=ifcb/post_sweep_candidates \
  ensemble.candidates.sources.0.type=run_dir_glob \
  ensemble.candidates.sources.0.uri_glob=./runs/ifcb_sweep/*/ \
  ensemble_outputs.manifests.dir=./shared_manifests
dojo ensemble \
  experiment=ifcb/post_sweep_ensemble \
  ensemble.candidates.sources.0.type=manifest \
  ensemble.candidates.sources.0.manifest_uri=./shared_manifests/ifcb_post_sweep_candidates.json
```

This is the supported pattern for "use a sweep's outputs as ensemble
candidates" — `dojo ensemble candidates` over an explicit
run-directory glob, not a registry-driven discovery (deferred).

## Cross-References

- `02-cli-and-task-types.md` — CLI architecture; `dojo sweep prepare`,
  `dojo sweep train`, `dojo sweep status`, and `dojo sweep report`.
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
