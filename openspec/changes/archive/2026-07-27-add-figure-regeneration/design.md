## Context

Standalone HTML figures are currently emitted only as a side effect of a
training run (`Training figures` requirement in `supervised-training`). Every
input the figures need already persists in the finished run directory, so
iterating on figure quality — new plot types, restyling, bug fixes — should not
require a retrain. `write_training_figures()` in `src/dojo/training/figures.py`
is already a self-contained entry point; the missing piece is a command that
points it at a completed run.

A finished run directory has a stable, self-describing layout:

```
<run_dir>/
├── config/resolved.yaml    # OmegaConf dump of the resolved RootConfig
├── metrics/metrics.csv     # line plots
├── results/                # classification_output rows + _metadata.json
└── figures/                # write target
```

## Goals / Non-Goals

**Goals:**
- A `dojo render run <run_dir>` command that regenerates the full HTML figure
  set from a completed run directory without retraining.
- Regenerated figures are byte-identical to the training-time output for the
  same inputs (reuse the exact `write_training_figures()` entry point).
- Robust to the random-run-id trap: no config recomposition, no model load.

**Non-Goals:**
- Running any model or inference (this is pure post-processing of artifacts).
- Changing the objective/head config schema (whether to collapse
  objective≡head is a separate follow-up change).
- Other `render` targets (`embeddings`, `ensemble`, `sweep`) — the group is
  introduced with room for them, but only `render run` ships here.

## Decisions

### Run-directory positional input, not config recomposition
`dojo render run <run_dir>` takes the run's top-level directory as a positional
argument and reads `metrics/`, `results/`, `figures/`, and `config/` by
convention.

*Why not config-driven (like `eval holdout`'s `compose_and_resolve`)?* The
default `run_id` template is `{coolname:noseed}`, which is random on each
resolve — recomposing a config would invent a **new** run directory and could
never reliably re-find an existing run. Pointing at the run dir sidesteps this
and matches the spec scenario ("against a finished run directory").

### `objective_to_head` from the snapshot config, identity fallback
`write_training_figures()` needs `objective_to_head` to map line-plot
**objectives** (from `metrics.csv` keys like `val/species/loss`) to
result-figure **head** subdirectories. This mapping is a genuine config concept:
it is not present in `metrics.csv` (objective names only) or in
`classification_output` / `_metadata.json` (head names only). The general case
is many-to-one — two objectives (e.g. a blended `ce` + `focal` loss) can share
one head — which only the config can disambiguate.

The command loads `<run_dir>/config/resolved.yaml` via
`RootConfig.model_validate` and rebuilds
`{name: obj.head or name for name, obj in cfg.objectives.items() if obj.enabled}`
— the exact expression the trainer uses. If the snapshot is absent (older runs),
fall back to the identity mapping (objective name == head name), which is correct
for every 1:1 config.

*Alternatives considered:* (a) assume identity always — silently wrong for the
multi-loss configs someone would most want to iterate figures on; (c) persist
`objective_to_head` into `_metadata.json` at train time — touches the train path
and schema and only helps future runs. Reading the snapshot is nearly free given
we already hold the run dir and is faithful in every case.

### New top-level `dojo render` command group
A new Typer sub-app registered in `cli/main.py` alongside `eval`/`infer`/
`inspect`. `render` is deliberately *not* filed under `eval`: `eval` means "run
a checkpoint over data to produce result rows" (it loads a model and runs
inference), whereas `render` runs no model — it is artifacts-from-artifacts. The
group leaves room for `render embeddings`, `render ensemble`, `render sweep`.

### Overwrite-by-default with `--backup`
The whole point is re-rendering against a fixed run, so `figures/` is
overwritten in place by default (unlike result writers, which refuse without
`--clobber`). `--backup` first copies an existing `figures/` to a
numeric-incremented sibling (`figures.1/`, `figures.2/`, …) and then overwrites,
so a prior render can be diffed against the new one.

## Risks / Trade-offs

- **Stale/foreign run dir** → the command reads whatever is on disk; if
  `metrics/` or `results/` are missing it should abort with a clear message
  naming the missing artifact rather than writing a partial figure set.
- **Snapshot config drift** → `resolved.yaml` is a JSON-mode OmegaConf dump; it
  must round-trip through `RootConfig.model_validate`. Reuse the existing
  resolved-config load path (`compare.py`'s resolved branch) rather than a
  bespoke parser so the two stay in sync.
- **`objective_to_head` divergence for pre-snapshot runs** → identity fallback
  is correct for 1:1 configs and only misplaces subdirs for the rare
  many-to-one case on runs old enough to lack `config/resolved.yaml`; acceptable
  and documented.

## Open Questions

- Whether to later collapse the objective/head namespaces (require objective
  name == head name, drop `objective.head`) so `objective_to_head` can be
  deleted from the figure builder entirely — tracked as a separate follow-up,
  out of scope here.
