
# 09. Sweeps and Batch Runs

## Purpose

Defines Hydra-driven sweeps and the `sweep_outputs:` block. Sweeps can
explore training hyperparameters, ensemble strategy / combine-mode
combinations, or feed candidate manifests into a subsequent ensembling
step. Bayesian / AutoML HPO is deferred.

## Hydra multirun

Use Hydra multirun (`-m`) for sweeps. Each Hydra job:

1. composes configuration;
2. applies CLI overrides;
3. validates with Pydantic;
4. resolves run directory and renders path templates;
5. runs training / evaluation / ensembling;
6. writes configured artifacts and canonical results;
7. logs to configured sinks (training runs only).

Example sweep:

```bash
dojo train -m \
  experiment=ifcb/experimentA \
  model.backbone.source=torchvision,timm \
  model.backbone.name=resnet50,convnext_tiny \
  optimizer.lr=1e-4,3e-4 \
  training.batch_size=32,64
```

### Hydra working directory and `run_id`

Dojo's run directory (resolved from `training_outputs.dir_template`
under `output_root`) is the canonical location for checkpoints, exports,
metrics, and results. It is independent of Hydra's working directory.

To avoid scattering output across Hydra's CWD (`outputs/...` or
`multirun/...`), Dojo's resolved run dirs, and the logger sink's
artifact store, Dojo ships Hydra defaults that align Hydra's job dirs
with the resolved output directory and keep CWD stable:

```yaml
# configs/hydra/default.yaml
hydra:
  run:
    dir: ${output_root}/${runtime.run_id}
  sweep:
    dir: ${output_root}/_sweeps/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${runtime.run_id}
  job:
    chdir: false
```

Key points:

- `runtime.run_id` is generated at config-resolve time and is stable
  across the run, multirun sweep, and logger sinks. It must be available
  as an OmegaConf interpolation before Hydra creates its job directory.
- `chdir: false` keeps the process CWD at the project root. All Dojo
  paths are absolute or relative to `output_root`. This avoids the
  Hydra footgun where `os.getcwd()` silently changes per job.
- For sweeps, `hydra.sweep.dir` is a Hydra coordination directory for
  per-job metadata; Dojo artifacts still land under each job's
  `training_outputs.dir`.
- Hydra's `hydra.job.num` and `hydra.job.id` can be folded into
  `runtime.run_id` (e.g. `{experiment.name}-{timestamp}-{job_num}`) to
  make sweep-member IDs unique and ordered.

Bayesian / AutoML HPO is deferred — see
`appendix-deferred-features.md`.

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
    name: resnet50  # swept: resnet50, efficientnet_b0, convnext_tiny

training:
  batch_size: 32  # swept: 32, 64

optimizer:
  lr: 0.0003  # swept: 0.0003, 0.0001

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
