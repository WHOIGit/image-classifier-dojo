## Context

`src/dojo/training/figures.py` emits standalone HTML figures for supervised
runs. Today each figure is a single static `Plotly.newPlot(traces, layout)`
written from a raw `{traces, layout}` dict via `_write_plotly_html`. Inputs are
already fully persisted: line plots read `metrics/metrics.csv`, and the
confusion-matrix / per-class plots read canonical `classification_output` rows
via `ResultReader`. There is no Python Plotly dependency — Plotly.js loads from
a CDN and the payload is `json.dumps`-ed into the page.

The predecessor implementation (`dojo_deprecated/multiclass/callbacks.py`) was
much richer: `go.Figure` objects with `updatemenus` dropdowns/buttons, rich
per-cell hover, reorderable axes, and a per-class misclassification explorer,
logged live to Aim. This change ports that interactivity back into the
standalone-HTML world without reintroducing the Python Plotly / Aim
dependencies, and without breaking the `add-figure-regeneration` contract that
these figures be regenerable offline from persisted artifacts.

## Goals / Non-Goals

**Goals:**
- Interactive figures (toggles, reorder selectors, rich hover) driven by a
  small reusable client-side JS layer over embedded, precomputed data.
- Correct, readable confusion matrix: square plot area, blanked zero cells,
  raw⇄row-normalized toggle, symmetric reorder, full hover.
- Single-head runs produce a flat, non-redundant figure set; multi-head runs
  keep per-head plus averaged plots, with an optional min–max envelope band.
- Every figure remains computable from `metrics/metrics.csv` +
  `classification_output` rows + `_metadata.json` so regeneration stays
  byte-identical.

**Non-Goals:**
- No Python Plotly dependency; no Aim logging path.
- No training-count series (validation "Counts" only).
- No fancier searchable/query FP–FN selector — the explorer keeps the
  deprecated dropdown (Plotly `updatemenus`) for now.
- No config-schema changes.

## Decisions

### D1: Richer shared HTML template with a small JS layer (option B)
Extend `_write_plotly_html` (or add a sibling) so the page can carry, besides
`traces`/`layout`/`config`, a `variants` blob and a declarative `controls`
spec, plus a compact reusable JS function that wires buttons/`<select>`s to
`Plotly.restyle`/`Plotly.relayout`/`Plotly.react`.

*Why over pure Plotly `updatemenus` (option A):* four of the five asks need
state Plotly buttons can't express without combinatorial precomputed button
sets — two *independent* dropdowns (per-class metric × order-by), a
normalize×order matrix, and "always show both raw and normalized in hover."
`updatemenus` would explode to N×M buttons and still can't recompute. A tiny JS
layer precomputes each independent axis once and composes them at runtime.
*Trade-off:* the template stops being trivial and gains JS to own and test;
mitigated by keeping one shared, documented helper rather than per-figure JS.

### D2: Precompute variants server-side, switch client-side
For each figure, the Python builder computes every data variant up front
(e.g. raw and row-normalized matrices; per-metric bar values; each ordering
permutation) and embeds them keyed in `variants`. The JS only selects and
applies — no math in the browser beyond index lookups. Keeps the "figures are a
pure function of persisted inputs" property and keeps regeneration
deterministic.

### D3: Confusion matrix construction
- **Square area:** set `yaxis.scaleanchor='x'` and `scaleratio=1` (plus equal
  category counts) so the inner heatmap is square regardless of label width;
  rely on `automargin` for labels.
- **Blank zero cells:** map zero counts to `null` in `z` with
  `hoverongaps=false` so they render as background and suppress hover (matches
  the deprecated `if cell_count==0: hovertext=''`).
- **Normalize toggle:** embed both the raw count matrix and the row-normalized
  (`true`/recall) matrix; toggle swaps `z` and the colorbar title. Row = actual
  (y), column = predicted (x); row-normalize = each actual row sums to 1.
- **Reorder:** embed the permutation index list for each order key; reordering
  applies the *same* permutation to rows and columns (keeps the diagonal), and
  reindexes labels and the precomputed hover matrix. Order option set is shared
  with per-class metrics (class-list order + each metric incl. counts).
- **Hover:** precompute an HTML hover string per non-zero cell containing
  Actual, Predicted, Actual sum, Predicted sum, Count, and Normalized value, so
  all six are present in both toggle states.

### D4: Per-class metrics — two independent selectors
Embed per-metric bar-value arrays (f1, precision, recall, validation counts)
and, separately, order permutations keyed by `auto`, `classlist`, and each
metric. The metric `<select>` chooses which value array is shown; the order-by
`<select>` chooses the permutation; `auto` resolves to "order by the currently
selected metric." The two are composed independently in JS.

### D5: Loss & F1 line plots
- Loss: one document; embed raw and first-epoch-normalized series; toggle below
  the title swaps them (via `Plotly.react`/`restyle`). Legend inside top-right
  (`legend.x/y` inside plot); `yaxis.rangemode` floored at `0.0`.
- F1: `yaxis.range=[null, 1.0]`? Plotly needs a concrete pair for a hard max
  with auto min, so pin `yaxis.autorange` off only for the max via
  `range=[min_seen_or_0, 1.0]` with `rangemode` handling — chosen approach:
  compute a data-driven min and set `range=[data_min, 1.0]`. (Resolve exact
  Plotly incantation in tasks.)

### D6: Misclassification explorer (port PlotPerclassDropdownAim)
Reuse the confusion payload already built for the matrix. For each class emit
two horizontal bar traces (FP = mistaken-as, FN = mistaken-for) and a Plotly
`updatemenus` dropdown that toggles trace visibility + title, exactly as the
deprecated `plot(cm, classes)`. Per head; skipped when a head has no
off-diagonal mass.

### D7: Single- vs multi-head layout
Branch in `_write_metric_lines` / `_write_result_figures` on head count. One
head → write directly under `figures/`, drop the averaged loss/F1 plots. >1
head → keep `figures/<head>/…` per-head plus top-level averaged plots. Averaged
loss/F1 gain an optional (default-off) min–max envelope: emit `min` (invisible
line), `max` (`fill='tonexty'`, faint) and `mean` traces; the toggle flips the
two bound traces' visibility. Min/max come free from the same per-head keys the
mean is computed from.

## Risks / Trade-offs

- **Shared requirement collision with `add-figure-regeneration`** → both modify
  the "Training figures" requirement and its figure-file list (this change
  removes `loss_curves_normalized.html` and changes the per-head layout).
  Mitigation: whichever archives second must re-copy the other's full
  requirement text; call this out at apply/archive time.
- **JS maintenance burden** → keep exactly one documented helper; snapshot-test
  the emitted HTML/payload rather than the rendered DOM.
- **Square matrix vs long labels** → `scaleanchor` can fight `automargin`;
  mitigation is to keep labels on axis ticks with `automargin` and let the plot
  area (not the figure) be square.
- **Validation "Counts" data source** → if `classification_output` row supports
  are insufficient (e.g. classes with zero val samples must still appear),
  surface per-head validation class counts via `_metadata.json`. Confirm before
  implementing the per-class "validation counts" metric.

## Open Questions

- Exact Plotly recipe for "hard ymax=1.0, auto ymin" (D5) — settle during
  implementation.
- Whether validation class counts are fully recoverable from
  `classification_output` supports or must be threaded through
  `_metadata.json` (D-risk above).
