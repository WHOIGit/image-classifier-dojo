
# 09. Sweeps and Batch Runs

## Purpose

Defines config-defined sweeps (Dojo-owned expansion over the Hydra
Compose API), batch-run-style grid sweeps, deferred Bayesian / AutoML
sweep schema, and the `sweep_outputs:` block. Sweeps
can explore training hyperparameters, ensemble strategy / combine-mode
combinations, random-seed sensitivity, or feed candidate manifests into
a subsequent ensembling step. Bayesian / AutoML HPO is deferred.

## Composition and Dojo-owned sweep expansion

Dojo composes configs through the Hydra **Compose API** (`hydra.compose`),
not `@hydra.main`, so there is no Hydra launcher, no Hydra-managed working
directory, and no `chdir`. A run is a sweep when the composed config's
`sweep:` block defines axes (see "Sweep definition block" below); Dojo
expands the cartesian product itself and owns runtime IDs,
output-directory resolution, existing-directory policy, and all canonical
artifacts.

A sweep is launched by composing a config whose `sweep:` block has axes —
either an experiment config that includes the block, or axes overridden
onto `sweep.grid` from the command line (dash-free config overrides):

```bash
# the experiment config carries a sweep: block
dojo train experiment=ifcb/sweep_lr_bs

# or supply axes as config overrides (Hydra list-value syntax)
dojo train experiment=ifcb/experimentA \
  +sweep.mode=grid \
  '+sweep.grid.optimizer.lr=[1e-4,3e-4]' \
  '+sweep.grid.training.batch_size=[32,64]'
```

There is no `-m` / `--multirun` flag and no CLI comma-list sweep
shorthand; the `sweep:` block is the single sweep definition.

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
> **TBD — sweep run execution.** Once Dojo has expanded a sweep into
> concrete per-run resolved configs and written `sweep_manifest.json`,
> *how* those runs execute — in-process sequentially, or handed to
> external orchestration — is an open question to be decided later. Dojo
> owns expansion and artifact layout regardless of the execution
> mechanism.

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
    model.image_input.backbone.architecture.: [resnet50, convnext_tiny]
```

`sweep.mode` values:

- `grid` — functional; explicit cartesian product over parameter lists.
- `bayesian` — schema slot only in the initial implementation; runtime
  raises `NotImplementedError`.

### Grid sweep syntax

Inside `sweep.grid`, each key is a target config path and each value is a
YAML list; the concrete runs are the cartesian product of those lists.
Axes may also be supplied as config overrides onto `sweep.grid` using
Hydra list-value syntax (e.g. `'+sweep.grid.optimizer.lr=[1e-4,3e-4]'`).

The `sweep:` block is the single sweep definition: there is no CLI
comma-list shorthand and no normalization of axes scattered elsewhere in
the config. After expansion, each concrete run receives ordinary resolved
scalar config values at the target paths.

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
  `sweep.grid`, e.g. `'+sweep.grid.optimizer.lr=[1e-4,3e-4]'`, never
  `optimizer.lr=1e-4,3e-4`.
- **Shell quoting:** single-quote any override token containing `[...]` or
  `{...}` so the shell does not glob- or brace-expand it before Hydra sees
  it (e.g. `'+sweep.grid.training.batch_size=[32,64]'`). Tokens without
  brackets (`+sweep.mode=grid`) need no quotes.
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

Sweep aggregation can then summarize metrics across seeds, for example
with box plots / five-number summaries for `macro_f1` or per-class F1.

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
onto `sweep.grid` (e.g. `'+sweep.grid.optimizer.lr=[1e-4,3e-4]'`), not
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
      model.image_input.backbone.name: convnext_tiny
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
      model.image_input.backbone.name:
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
   1. Determine whether the composed config defines a sweep (non-empty
      `sweep.grid`).
   2. If there is no active sweep:
      1. Set `runtime.sweep_id = null` unless explicitly configured.
      2. Set `sweep_hash = null`.
      3. Skip sweep-level aggregation setup.
   3. If there is an active sweep:
      1. Read the sweep axes from the composed `sweep:` block (including
         any `+sweep.grid.*` CLI overrides).
      2. Assemble the sweep definition: base config plus the `sweep:`
         block.
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
      3. The error suggests adding a uniqueness-pattern value such as the
         `{job_num}` sweep index or a realized sweep value such as
         `runtime.seed`.
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
      1. `sweep_base.yaml` — composed base config before per-run sweep
         values;
      2. `sweep_definition.yaml` — normalized `sweep.mode`, axes /
         search params, conflict policy, finalized `sweep_id`, and
         `sweep_hash`;
      3. `sweep_manifest.json` — concrete run index containing generated
         run IDs, config hashes, realized sweep values, resolved per-run
         output dirs, and resolved per-run config artifact paths;
      4. `cli.txt`;
      5. `overrides.txt`.
6. **Execute runs and sweep aggregation**
   1. Execute each concrete run.
   2. Result rows include `run_id`, `config_hash`, and related run
      provenance.
   3. Sweep-produced result rows also include `sweep_id` and
      `sweep_hash`.
   4. If `sweep_outputs.enabled: false`, stop after run execution; the
      sweep happened, but sweep-level aggregation outputs are skipped.
   5. After all sweep runs finish, execute configured
      `sweep_outputs` aggregation:
      1. collect per-run metrics / artifacts;
      2. write sweep-level metrics and figures;
      3. write sweep-level exports only if explicitly configured.

## `sweep_outputs:` block

Peer to `training_outputs:` and `ensemble_outputs:`. Holds **sweep-level
aggregation** outputs.

Sub-blocks: `dir_template`, `enabled`, `collect`, `export`, `metrics`,
`figures`. There is **no** `results` sub-block — per-row data comes from
the underlying per-job `training_outputs` / `ensemble_outputs`.

`sweep_outputs.enabled` defaults to `true`. When `false`, Dojo still
expands and executes the sweep runs, but skips sweep-level collection,
summary metrics, aggregate figures, and sweep exports.

`sweep_outputs.collect` is a list of metric / artifact collection specs.
Each item names what to collect from every concrete run and which
per-run source to read. For metric specs, optional `mode` is `min` or
`max` and controls sweep-level ranking / best-run selection for that
collected metric:

```yaml
sweep_outputs:
  collect:
    - metric: val/species/macro_f1
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

Sweep aggregation does not discover runs by scanning directories. During
sweep expansion, Dojo writes `sweep_outputs.dir/config/sweep_manifest.json`
as the run index. The manifest records each concrete run's `run_id`,
`config_hash`, realized sweep values, resolved `training_outputs.dir` /
`ensemble_outputs.dir`, and resolved config artifact paths. Aggregation
reads the manifest, then reads each run's `config/resolved.yaml` and
metric artifacts from those recorded locations.

Default on-disk sub-directories under the resolved `sweep_outputs.dir`:

```text
sweep_outputs.dir/
  config/
    sweep_base.yaml
    sweep_definition.yaml
    sweep_manifest.json
    cli.txt
    overrides.txt
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
  image_input:
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
    model.image_input.backbone.name: [resnet50, efficientnet_b0, convnext_tiny]
    training.batch_size: [32, 64]
    optimizer.lr: [0.0003, 0.0001]

output_root: ./runs

training_outputs:
  dir_template: >-
    {experiment.name}/sweep_runs/{model.image_input.backbone.name:slug}/bs{training.batch_size:03}/lr{optimizer.lr:slug}/

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

Sweeps can explore ensemble selection strategy / combine-mode
combinations against a candidate manifest, via a `sweep:` block or
`+sweep.grid.*` overrides:

```bash
dojo ensemble experiment=ifcb/ensemble_search \
  +sweep.mode=grid \
  '+sweep.grid.ensemble.selection.strategy=[top_k,greedy_forward_selection]' \
  '+sweep.grid.ensemble.inference.combine.classification=[probabilities_mean,logits_mean]'
```

Sweep aggregation summarizes ensemble metrics across the swept axes.

## Sweeps as candidate-source feeders

A training sweep may write to a shared candidate manifest directory that
a subsequent `dojo ensemble` invocation reads:

```bash
dojo train experiment=ifcb/sweep_for_ensembling
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

- `02-cli-and-task-types.md` — CLI architecture; config-defined sweeps
  (the `sweep:` block) work with every command family.
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
