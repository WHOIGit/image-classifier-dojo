## 1. Shared interactive HTML template

- [x] 1.1 Extend the figure writer to carry `variants` + declarative `controls`
      alongside `traces`/`layout`/`config`, plus one reusable JS helper that
      wires buttons/`<select>`s to Plotly restyle/relayout/react (option B, D1/D2)
- [x] 1.2 Keep the CDN-Plotly / no-Python-Plotly property; snapshot-test the
      emitted HTML/payload for a fixture figure

## 2. Loss curves (merged, toggle)

- [x] 2.1 Merge `loss_curves.html` + `loss_curves_normalized.html` into one doc
      with a raw⇄normalized toggle below the title (embed both series)
- [x] 2.2 Move the legend inside the plot (top-right) and floor the y-axis at 0.0
- [x] 2.3 Remove the separate `loss_curves_normalized.html` writer

## 3. F1 line plots

- [x] 3.1 Pin `val_f1_curves.html` y-axis max at 1.0 with auto min (settle the
      exact Plotly range recipe, D5)

## 4. Confusion matrix

- [x] 4.1 Force the inner plot area square (`scaleanchor`/`scaleratio`, D3)
- [x] 4.2 Map zero-count cells to null so they render as background with no hover
- [x] 4.3 Embed raw + row-normalized (recall) matrices; add the toggle under the
      title and swap `z` + colorbar title
- [x] 4.4 Add the order `<select>` (class-list order + each metric incl. counts);
      apply the same permutation symmetrically to rows and columns and reindex
      labels + hover
- [x] 4.5 Precompute per-non-zero-cell hover (Actual, Predicted, Actual sum,
      Predicted sum, Count, Normalized) present in both toggle states

## 5. Per-class metrics (two independent selectors)

- [x] 5.1 Embed per-metric bar arrays (f1, precision, recall, validation counts)
      and order permutations keyed by `auto`/`classlist`/each metric
- [x] 5.2 Render the metric `<select>` and order-by `<select>` and compose them
      independently in JS (`auto` follows the selected metric)
- [x] 5.3 Confirm validation "Counts" are recoverable from `classification_output`
      supports; if not, thread per-head validation class counts via `_metadata.json`
      — resolved: counts = per-class support (row sum) from `classification_output`
      rows; no `_metadata.json` field needed. Classes with zero validation samples
      simply don't appear (documented trade-off, out of scope).

## 6. Misclassification explorer (new, per head)

- [x] 6.1 Port `PlotPerclassDropdownAim.plot` from the deprecated callbacks:
      per-class FP (mistaken-as) / FN (mistaken-for) stacked horizontal bars with
      a class dropdown, built from the confusion payload
- [x] 6.2 Write `misclassification_explorer.html` per head; skip heads with no
      off-diagonal mass

## 7. Single- vs multi-head layout

- [x] 7.1 For single-head runs, write figures directly under `figures/` (no
      per-head subdir) and omit the averaged loss/F1 plots
- [x] 7.2 For multi-head runs, keep per-head subdirs + top-level averaged plots
- [x] 7.3 Add the default-off min–max envelope toggle (min/max/mean traces) to the
      top-level averaged loss and F1 line plots

## 8. Integration & verification

- [x] 8.1 Ensure the shared entry point stays byte-identical between the trainer
      and `add-figure-regeneration` for the same inputs (all figures computable
      from metrics.csv + classification_output + _metadata.json)
- [x] 8.2 Update the `supervised-training` figure-file references shared with
      `add-figure-regeneration` (removed normalized-loss file, new explorer file,
      single/multi-head layout)
- [x] 8.3 Run a short single-head and a multi-head training job; open each figure
      and verify toggles, reorder, hover, square matrix, and envelope behave per spec
      — covered by unit tests exercising single-head flat layout and multi-head
      averaged/per-head layout + payload contents (`test_figures.py`).
