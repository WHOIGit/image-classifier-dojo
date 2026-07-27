## Why

The standalone HTML training figures (`src/dojo/training/figures.py`) are a
minimal reimplementation of the far richer interactive Aim plots that used to
live in `dojo_deprecated/multiclass/callbacks.py`: each figure is a single
static `Plotly.newPlot` with no toggles, no reordering, sparse hover, a
non-square confusion matrix, and duplicate per-head plots even for single-head
models. This makes the figures hard to read and interrogate during model
evaluation. Restoring the interactive affordances — while keeping the figures
computable purely from persisted run artifacts — closes that regression.
Source design: `DESIGN-DOC/05-models-training-and-heads.md` (training figures)
and `DESIGN-DOC/06-results-artifacts-and-metadata.md` (canonical
`classification_output` rows / `_metadata.json`); workplan P2 (core supervised
platform, figures).

## What Changes

- Adopt a richer shared HTML template that precomputes all data variants into
  the embedded payload and renders real controls (buttons / `<select>`) plus a
  small reusable JS layer that restyles/recomputes on change — replacing the
  one-`newPlot`-per-file template. Still no Python Plotly dependency; still
  Plotly.js from the CDN.
- **Loss curves**: merge `loss_curves.html` and `loss_curves_normalized.html`
  into a single document with a raw⇄normalized toggle below the title, legend
  moved inside the plot (top-right), and the y-axis floor pinned at `0.0`.
- **Confusion matrix**: force the matrix area square; blank (background-color)
  cells where the count is zero; add a raw-Counts⇄row-normalized (recall)
  toggle under the title; add an order `<select>` (same option set as
  per-class metrics, symmetric on both axes); hover always shows Actual,
  Predicted, Actual sum, Predicted sum, cell Count, and Normalized value.
- **F1 line plots**: pin `yaxis` max at `1.0` with auto min.
- **Per-class metrics**: two independent controls under the title — a metric
  `<select>` (f1 / precision / recall / validation counts) and an order-by
  `<select>` (`auto:<selected-metric>`, class-list order, then each metric
  incl. counts).
- **Misclassification explorer** (new, per head): port
  `PlotPerclassDropdownAim` — pick a class and see stacked horizontal bars of
  false positives (mistaken-as) and false negatives (mistaken-for), built from
  the confusion payload.
- **Single- vs multi-head layout**: for single-head models write figures
  directly under `figures/` with no per-head subfolder and omit the redundant
  averaged loss/F1 plots; for multi-head models keep per-head subfolders plus
  top-level averaged plots, and give the averaged loss/F1 line plots an
  optional (default-off) min–max envelope toggle band across heads.
- All figures remain fully computable from `metrics/metrics.csv`,
  `classification_output` rows, and `_metadata.json`, so the
  `add-figure-regeneration` command still regenerates them offline.

## Capabilities

### New Capabilities
<!-- none: this reworks existing figure behavior -->

### Modified Capabilities
- `supervised-training`: the "Training figures" requirement changes — the
  emitted figure-file set, the single- vs multi-head directory layout, and the
  required interactive controls / hover / axis behavior of each figure.

## Impact

- Code: `src/dojo/training/figures.py` (template, builders, single/multi-head
  branching); the shared figure entry point consumed by both the trainer and
  the `add-figure-regeneration` command.
- Coordination: overlaps `add-figure-regeneration`, which also modifies the
  same "Training figures" requirement (figure-file set changes here must be
  reflected there); both must stay computable from the same persisted inputs.
- Data inputs: may need per-head validation class counts surfaced via
  `_metadata.json` if not already derivable from `classification_output` row
  supports (validation "Counts" only; training counts are intentionally out of
  scope).
- No config-schema or Python-dependency changes; output is still CDN Plotly
  HTML.
