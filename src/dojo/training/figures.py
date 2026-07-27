"""Standalone HTML figure generation for supervised training runs.

Figures are self-contained HTML: precomputed JSON payloads plus Plotly.js from
the CDN and one small shared client-side control layer (no Python Plotly
dependency). Line plots read ``metrics/metrics.csv``; the confusion matrix,
per-class metrics, and misclassification explorer read canonical
``classification_output`` rows. Every figure is a pure function of those
persisted inputs so ``add-figure-regeneration`` can rebuild them offline.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dojo.results import ResultReader
from dojo.results.schemas import RECORD_TYPE_CLASSIFICATION_OUTPUT

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Shared client-side control layer (option B). Builds a control bar under the
# title from a declarative ``controls`` spec, keeps a small ``state`` object,
# and dispatches to a per-kind renderer that selects/applies precomputed
# ``variants``. The renderers do index lookups only -- no math in the browser.
_INTERACTIVE_JS = """
const P = window.__PAYLOAD__;
const state = {};
(P.controls || []).forEach(c => { state[c.id] = c.default; });

function buildControls() {
  const bar = document.getElementById('controls');
  (P.controls || []).forEach(c => {
    const wrap = document.createElement('span');
    wrap.style.marginRight = '18px';
    if (c.kind === 'select') {
      const lab = document.createElement('label');
      lab.textContent = c.label + ': ';
      wrap.appendChild(lab);
      const sel = document.createElement('select');
      sel.id = 'ctl_' + c.id;
      c.options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o.value; opt.textContent = o.label;
        if (o.value === c.default) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', () => { state[c.id] = sel.value; render(); });
      wrap.appendChild(sel);
    } else if (c.kind === 'buttons') {
      const lab = document.createElement('label');
      lab.textContent = c.label + ': ';
      wrap.appendChild(lab);
      c.options.forEach(o => {
        const btn = document.createElement('button');
        btn.textContent = o.label;
        btn.style.marginRight = '4px';
        btn.addEventListener('click', () => { state[c.id] = o.value; render(); });
        wrap.appendChild(btn);
      });
    } else if (c.kind === 'checkbox') {
      const box = document.createElement('input');
      box.type = 'checkbox'; box.checked = !!c.default; box.id = 'ctl_' + c.id;
      box.addEventListener('change', () => { state[c.id] = box.checked; render(); });
      const lab = document.createElement('label');
      lab.htmlFor = box.id; lab.textContent = ' ' + c.label;
      wrap.appendChild(box); wrap.appendChild(lab);
    }
    bar.appendChild(wrap);
  });
}

function renderLines() {
  const v = P.variants;
  const scale = state.scale || v.defaultScale;
  const traces = v.series[scale].map(t => (
    {type: 'scatter', mode: 'lines+markers', x: t.x, y: t.y, name: t.name}
  ));
  if (v.envelope && state.envelope) {
    const env = v.envelope[scale];
    traces.push({type: 'scatter', mode: 'lines', x: env.x, y: env.lower,
                 line: {width: 0}, showlegend: false, hoverinfo: 'skip'});
    traces.push({type: 'scatter', mode: 'lines', x: env.x, y: env.upper,
                 fill: 'tonexty', fillcolor: 'rgba(120,120,120,0.15)',
                 line: {width: 0}, showlegend: false, hoverinfo: 'skip'});
  }
  const layout = {};
  if (v.yaxis) layout.yaxis = v.yaxis[scale] || v.yaxis.default;
  return {traces, layout};
}

function renderConfusion() {
  const v = P.variants;
  const perm = v.orders[state.order] || v.orders.classlist;
  const base = state.norm === 'recall' ? v.recall : v.counts;
  const z = perm.map(i => perm.map(j => (v.counts[i][j] === 0 ? null : base[i][j])));
  const text = perm.map(i => perm.map(j => v.hover[i][j]));
  const x = perm.map(j => v.labelsX[j]);
  const y = perm.map(i => v.labelsY[i]);
  const traces = [{
    type: 'heatmap', z: z, x: x, y: y, text: text, hoverinfo: 'text',
    colorscale: 'Viridis', hoverongaps: false, xgap: 1, ygap: 1,
    colorbar: {title: state.norm === 'recall' ? 'Recall' : 'Counts'}
  }];
  return {traces, layout: {
    shapes: [{
      type: 'rect', xref: 'x', yref: 'y',
      x0: -0.5, x1: x.length - 0.5, y0: -0.5, y1: y.length - 0.5,
      fillcolor: '#f2f2f2', line: {width: 0}, layer: 'below'
    }],
    xaxis: {title: 'Predicted', automargin: true,
            tickmode: 'array', tickvals: x, ticktext: x, tickangle: -45},
    yaxis: {title: 'Actual', automargin: true, autorange: 'reversed',
            scaleanchor: 'x', scaleratio: 1,
            tickmode: 'array', tickvals: y, ticktext: y,
            minallowed: -0.5, maxallowed: y.length - 0.5}
  }};
}

function renderPerClass() {
  const v = P.variants;
  const orderKey = state.order === 'auto' ? state.metric : state.order;
  const perm = v.orders[orderKey] || v.orders.classlist;
  const vals = v.values[state.metric];
  const hovertemplate = state.metric === 'counts'
    ? '%{y}<br>%{x}<extra></extra>'
    : '%{y}<br>%{x:.3f}<extra></extra>';
  const traces = [{
    type: 'bar', orientation: 'h',
    x: perm.map(i => vals[i]),
    y: perm.map(i => v.labels[i]),
    hovertemplate: hovertemplate
  }];
  return {traces, layout: {}};
}

function renderExplorer() {
  const v = P.variants;
  const view = v.classes[state.class];
  const categoryarray = view.orders[state.order] || view.orders.errors;
  const classSelect = document.getElementById('ctl_class');
  const classOrder = v.classOrders[state.order] || v.classOrders.errors;
  if (classSelect) {
    const options = Object.fromEntries(
      Array.from(classSelect.options, option => [option.value, option])
    );
    classOrder.forEach(value => classSelect.appendChild(options[value]));
    classSelect.value = state.class;
  }
  const traces = v.traces.map((trace, index) => (
    Object.assign({}, trace, {visible: view.visible[index]})
  ));
  return {traces, layout: {
    title: view.title, height: view.height,
    yaxis: Object.assign({}, view.yaxis, {categoryarray: categoryarray})
  }};
}

const RENDERERS = {
  lines: renderLines, confusion: renderConfusion, perclass: renderPerClass,
  explorer: renderExplorer
};

function render() {
  const out = RENDERERS[P.kind]();
  const layout = Object.assign({}, P.layout, out.layout);
  Plotly.react('plot', out.traces, layout, P.config);
}

buildControls();
render();
"""


def _html_document(*, title: str, payload: dict[str, Any], body_script: str) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            f'<script src="{PLOTLY_CDN}"></script>',
            "</head>",
            "<body>",
            '<div id="controls" style="margin:8px 0;font-family:sans-serif;font-size:13px;"></div>',
            '<div id="plot" style="width:100%;height:90vh;"></div>',
            "<script>",
            f"window.__PAYLOAD__ = {json.dumps(payload)};",
            body_script,
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def _write_plotly_html(
    path: Path,
    *,
    title: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Write a static single-``newPlot`` figure (used by the explorer)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traces": traces,
        "layout": {"title": title, **(layout or {})},
        "config": {"responsive": True, **(config or {})},
    }
    script = (
        "const p = window.__PAYLOAD__;\n"
        "Plotly.newPlot('plot', p.traces, p.layout, p.config);"
    )
    path.write_text(
        _html_document(title=title, payload=payload, body_script=script),
        encoding="utf-8",
    )


def _write_interactive_html(
    path: Path,
    *,
    title: str,
    kind: str,
    variants: dict[str, Any],
    controls: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Write an interactive figure driven by the shared control layer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "variants": variants,
        "controls": controls,
        "layout": {"title": title, **(layout or {})},
        "config": {"responsive": True, **(config or {})},
    }
    path.write_text(
        _html_document(title=title, payload=payload, body_script=_INTERACTIVE_JS),
        encoding="utf-8",
    )


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_merged_metrics(path: Path) -> list[dict[str, float | int]]:
    """Merge Lightning CSVLogger's split train/val rows into one row per epoch."""

    if not path.exists():
        return []
    by_epoch: dict[int, dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("epoch") in (None, ""):
                continue
            epoch = int(float(row["epoch"]))
            merged = by_epoch.setdefault(epoch, {"epoch": epoch})
            for key, value in row.items():
                if key == "epoch":
                    continue
                parsed = _float_or_none(value)
                if parsed is not None:
                    merged[key] = parsed
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def merge_metrics_csv(path: Path) -> None:
    """Rewrite Lightning CSV metrics as one merged row per epoch."""

    rows = _read_merged_metrics(path)
    if not rows:
        return
    fieldnames = ["epoch"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _series(rows: list[dict[str, float | int]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        if key in row:
            xs.append(int(row["epoch"]))
            ys.append(float(row[key]))
    return xs, ys


def _line_series(
    rows: list[dict[str, float | int]],
    key: str,
    name: str,
    *,
    normalize: bool = False,
) -> dict[str, Any] | None:
    """Build one ``{x, y, name}`` line-series variant entry."""

    x, y = _series(rows, key)
    if not y:
        return None
    if normalize:
        baseline = y[0]
        if baseline != 0:
            y = [value / baseline for value in y]
    return {"x": x, "y": y, "name": name}


def _mean_series(
    rows: list[dict[str, float | int]],
    keys: list[str],
    name: str,
) -> dict[str, Any] | None:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        values = [float(row[key]) for key in keys if key in row]
        if values:
            xs.append(int(row["epoch"]))
            ys.append(sum(values) / len(values))
    if not ys:
        return None
    return {"x": xs, "y": ys, "name": name}


def _envelope(
    rows: list[dict[str, float | int]],
    keys: list[str],
) -> dict[str, list[float | int]] | None:
    """Per-epoch min/max band across the given per-head keys."""

    xs: list[float | int] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        values = [float(row[key]) for key in keys if key in row]
        if len(values) >= 2:
            xs.append(int(row["epoch"]))
            lower.append(min(values))
            upper.append(max(values))
    if not xs:
        return None
    return {"x": xs, "lower": lower, "upper": upper}


_LEGEND_INSIDE = {
    "x": 0.99,
    "y": 0.99,
    "xanchor": "right",
    "yanchor": "top",
    "bgcolor": "rgba(255,255,255,0.6)",
}


def _unique_metric_keys(
    metrics: list[dict[str, float | int]],
    *,
    prefix: str,
    suffix: str,
) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in metrics:
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix) and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _objective_names(metrics: list[dict[str, float | int]]) -> list[str]:
    objectives: list[str] = []
    seen: set[str] = set()
    for row in metrics:
        for key in row:
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] in {"train", "val"}:
                objective = parts[1]
                if objective not in seen:
                    seen.add(objective)
                    objectives.append(objective)
    return objectives


def _write_loss_curves(
    path: Path,
    metrics: list[dict[str, float | int]],
    *,
    train_key: str,
    val_key: str,
    title: str,
    envelope_keys: tuple[list[str], list[str]] | None = None,
) -> None:
    """One merged loss document: raw<->normalized toggle, legend inside, y>=0."""

    raw = [
        s
        for s in (
            _line_series(metrics, train_key, "train loss"),
            _line_series(metrics, val_key, "val loss"),
        )
        if s is not None
    ]
    if not raw:
        return
    normalized = [
        s
        for s in (
            _line_series(metrics, train_key, "train loss", normalize=True),
            _line_series(metrics, val_key, "val loss", normalize=True),
        )
        if s is not None
    ]

    variants: dict[str, Any] = {
        "defaultScale": "raw",
        "series": {"raw": raw, "normalized": normalized},
        "yaxis": {
            "raw": {"title": "Loss", "rangemode": "tozero"},
            "normalized": {"title": "Loss / epoch 0", "rangemode": "tozero"},
        },
    }

    controls: list[dict[str, Any]] = [
        {
            "id": "scale",
            "kind": "buttons",
            "label": "Loss",
            "default": "raw",
            "options": [
                {"value": "raw", "label": "Raw"},
                {"value": "normalized", "label": "Normalized"},
            ],
        }
    ]

    if envelope_keys is not None:
        train_env = _envelope(metrics, envelope_keys[0])
        val_env = _envelope(metrics, envelope_keys[1])
        # Envelope spans the widest per-head band at each epoch (min of mins,
        # max of maxes across the train/val head keys we averaged).
        band = _combine_envelopes([e for e in (train_env, val_env) if e])
        if band is not None:
            variants["envelope"] = {"raw": band, "normalized": band}
            controls.append(
                {
                    "id": "envelope",
                    "kind": "checkbox",
                    "label": "Min-max envelope",
                    "default": False,
                }
            )

    _write_interactive_html(
        path,
        title=title,
        kind="lines",
        variants=variants,
        controls=controls,
        layout={
            "xaxis": {"title": "Epoch"},
            "yaxis": {"title": "Loss", "rangemode": "tozero"},
            "legend": _LEGEND_INSIDE,
        },
    )


def _combine_envelopes(
    bands: list[dict[str, list[float | int]]],
) -> dict[str, list[float | int]] | None:
    if not bands:
        return None
    by_x: dict[int, tuple[float, float]] = {}
    for band in bands:
        for x, lo, hi in zip(band["x"], band["lower"], band["upper"]):
            xi = int(x)
            if xi in by_x:
                plo, phi = by_x[xi]
                by_x[xi] = (min(plo, lo), max(phi, hi))
            else:
                by_x[xi] = (lo, hi)
    xs: list[float | int] = sorted(by_x)
    return {
        "x": xs,
        "lower": [by_x[int(x)][0] for x in xs],
        "upper": [by_x[int(x)][1] for x in xs],
    }


def _write_f1_curves(
    path: Path,
    *,
    title: str,
    metrics: list[dict[str, float | int]],
    keys_by_trace: list[tuple[list[str], str]],
    envelope_keys: list[str] | None = None,
) -> None:
    """Validation F1 lines; y-axis max pinned at 1.0, min auto/data-driven."""

    series: list[dict[str, Any]] = []
    for keys, name in keys_by_trace:
        s = (
            _line_series(metrics, keys[0], name)
            if len(keys) == 1
            else _mean_series(metrics, keys, name)
        )
        if s is not None:
            series.append(s)
    if not series:
        return

    data_min = min(min(s["y"]) for s in series)
    y_floor = min(data_min, 1.0)

    variants: dict[str, Any] = {
        "defaultScale": "default",
        "series": {"default": series},
        "yaxis": {"default": {"title": "F1", "range": [y_floor, 1.0]}},
    }
    controls: list[dict[str, Any]] = []

    if envelope_keys is not None:
        band = _envelope(metrics, envelope_keys)
        if band is not None:
            variants["envelope"] = {"default": band}
            controls.append(
                {
                    "id": "envelope",
                    "kind": "checkbox",
                    "label": "Min-max envelope",
                    "default": False,
                }
            )

    _write_interactive_html(
        path,
        title=title,
        kind="lines",
        variants=variants,
        controls=controls,
        layout={
            "xaxis": {"title": "Epoch"},
            "yaxis": {"title": "F1", "range": [y_floor, 1.0]},
            "legend": _LEGEND_INSIDE,
        },
    )


def _write_head_metric_lines(
    figures_dir: Path,
    metrics: list[dict[str, float | int]],
    *,
    objective_name: str,
    head_name: str | None,
) -> None:
    head_dir = figures_dir if head_name is None else figures_dir / head_name
    prefix = "" if head_name is None else f"{head_name} "
    _write_loss_curves(
        head_dir / "loss_curves.html",
        metrics,
        train_key=f"train/{objective_name}/loss",
        val_key=f"val/{objective_name}/loss",
        title=f"{prefix}Training and Validation Loss",
    )
    head_f1_keys = [
        ([key], key)
        for key in (
            f"val/{objective_name}/f1_macro",
            f"val/{objective_name}/f1_micro",
        )
    ]
    _write_f1_curves(
        head_dir / "val_f1_curves.html",
        title=f"{prefix}Validation F1 per Epoch",
        metrics=metrics,
        keys_by_trace=head_f1_keys,
    )


def _write_metric_lines(
    figures_dir: Path,
    metrics: list[dict[str, float | int]],
    objective_to_head: dict[str, str] | None = None,
) -> None:
    objective_to_head = objective_to_head or {}
    objectives = _objective_names(metrics)
    single_head = len(objectives) <= 1

    if single_head:
        # Flat layout: figures directly under figures/, no averaged plots.
        objective = objectives[0] if objectives else None
        if objective is not None:
            _write_head_metric_lines(
                figures_dir,
                metrics,
                objective_name=objective,
                head_name=None,
            )
        else:
            # No per-objective keys: fall back to aggregate loss only.
            _write_loss_curves(
                figures_dir / "loss_curves.html",
                metrics,
                train_key="train/loss",
                val_key="val/loss",
                title="Training and Validation Loss",
            )
        return

    # Multi-head: top-level averaged plots (with min-max envelope) + per head.
    macro_keys = _unique_metric_keys(metrics, prefix="val/", suffix="/f1_macro")
    micro_keys = _unique_metric_keys(metrics, prefix="val/", suffix="/f1_micro")
    train_loss_keys = [f"train/{obj}/loss" for obj in objectives]
    val_loss_keys = [f"val/{obj}/loss" for obj in objectives]

    _write_loss_curves(
        figures_dir / "loss_curves.html",
        metrics,
        train_key="train/loss",
        val_key="val/loss",
        title="Training and Validation Loss",
        envelope_keys=(train_loss_keys, val_loss_keys),
    )
    _write_f1_curves(
        figures_dir / "val_f1_curves.html",
        title="Mean Validation F1 per Epoch",
        metrics=metrics,
        keys_by_trace=[
            item
            for item in (
                (macro_keys, "val/f1_macro mean"),
                (micro_keys, "val/f1_micro mean"),
            )
            if item[0]
        ],
        envelope_keys=macro_keys or micro_keys,
    )

    for objective in objectives:
        _write_head_metric_lines(
            figures_dir,
            metrics,
            objective_name=objective,
            head_name=objective_to_head.get(objective, objective),
        )


def _classification_rows(results_dir: Path) -> list[dict[str, Any]]:
    reader = ResultReader(results_dir)
    return list(reader.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))


def _confusion_payload(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[int]]]:
    labels_by_index: dict[int, str] = {}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        target_index = row.get("target_index")
        prediction_index = row.get("prediction_index")
        if target_index is None or prediction_index is None:
            continue
        target_index = int(target_index)
        prediction_index = int(prediction_index)
        labels_by_index[target_index] = row.get("target_name") or str(target_index)
        labels_by_index[prediction_index] = row.get("prediction_label") or str(prediction_index)
        counts[(target_index, prediction_index)] += 1

    indices = sorted(labels_by_index)
    labels = [f"{index:03d} {labels_by_index[index]}" for index in indices]
    matrix = [[counts[(actual, predicted)] for predicted in indices] for actual in indices]
    return labels, matrix


def _per_class_stats(matrix: list[list[int]]) -> dict[str, list[float | int]]:
    """precision / recall / f1 / validation counts (support) per class."""

    n = len(matrix)
    support: list[float | int] = []
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index in range(n):
        tp = matrix[index][index]
        row_sum = sum(matrix[index])
        col_sum = sum(matrix[actual][index] for actual in range(n))
        prec = tp / col_sum if col_sum else 0.0
        rec = tp / row_sum if row_sum else 0.0
        support.append(row_sum)
        precision.append(prec)
        recall.append(rec)
        f1.append((2 * prec * rec / (prec + rec)) if prec + rec else 0.0)
    return {"counts": support, "precision": precision, "recall": recall, "f1": f1}


def _order_permutations(
    n: int,
    stats: dict[str, list[float | int]],
) -> dict[str, list[int]]:
    """Symmetric reorder permutations: class-list order + each metric (desc)."""

    orders: dict[str, list[int]] = {"classlist": list(range(n))}
    for key in ("f1", "precision", "recall", "counts"):
        values = stats[key]
        orders[key] = sorted(range(n), key=lambda i: values[i], reverse=True)
    return orders


_ORDER_OPTIONS = [
    {"value": "classlist", "label": "class list"},
    {"value": "f1", "label": "f1"},
    {"value": "precision", "label": "precision"},
    {"value": "recall", "label": "recall"},
    {"value": "counts", "label": "counts"},
]


def _write_confusion_matrix(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    n = len(labels)
    stats = _per_class_stats(matrix)
    row_sums = stats["counts"]
    col_sums = [sum(matrix[a][j] for a in range(n)) for j in range(n)]

    recall = [
        [(matrix[i][j] / row_sums[i]) if row_sums[i] else 0.0 for j in range(n)]
        for i in range(n)
    ]

    labels_y = [f"{labels[i]} ({row_sums[i]})" for i in range(n)]
    labels_x = [f"{labels[j]} ({col_sums[j]})" for j in range(n)]

    hover = [
        [
            (
                f"Actual: {labels[i]}<br>"
                f"Predicted: {labels[j]}<br>"
                f"Actual sum: {row_sums[i]}<br>"
                f"Predicted sum: {col_sums[j]}<br>"
                f"Count: {matrix[i][j]}<br>"
                f"Normalized: {recall[i][j]:.3f}"
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    variants = {
        "counts": matrix,
        "recall": recall,
        "hover": hover,
        "labelsX": labels_x,
        "labelsY": labels_y,
        "orders": _order_permutations(n, stats),
    }
    controls = [
        {
            "id": "norm",
            "kind": "buttons",
            "label": "Values",
            "default": "counts",
            "options": [
                {"value": "counts", "label": "Counts"},
                {"value": "recall", "label": "Row-Normalized"},
            ],
        },
        {
            "id": "order",
            "kind": "select",
            "label": "Order by",
            "default": "classlist",
            "options": _ORDER_OPTIONS,
        },
    ]
    _write_interactive_html(
        path,
        title="Confusion Matrix",
        kind="confusion",
        variants=variants,
        controls=controls,
        layout={
            "xaxis": {"title": "Predicted", "automargin": True},
            "yaxis": {
                "title": "Actual",
                "automargin": True,
                "autorange": "reversed",
                "scaleanchor": "x",
                "scaleratio": 1,
            },
        },
    )


def _write_per_class_metrics(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    n = len(labels)
    stats = _per_class_stats(matrix)
    variants = {
        "labels": labels,
        "values": {
            "f1": stats["f1"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "counts": stats["counts"],
        },
        "orders": _order_permutations(n, stats),
    }
    controls = [
        {
            "id": "metric",
            "kind": "select",
            "label": "Metric",
            "default": "f1",
            "options": [
                {"value": "f1", "label": "f1"},
                {"value": "precision", "label": "precision"},
                {"value": "recall", "label": "recall"},
                {"value": "counts", "label": "counts"},
            ],
        },
        {
            "id": "order",
            "kind": "select",
            "label": "Order by",
            "default": "auto",
            "options": [
                {"value": "auto", "label": "auto (selected metric)"},
                *_ORDER_OPTIONS,
            ],
        },
    ]
    _write_interactive_html(
        path,
        title="Per-Class Metrics",
        kind="perclass",
        variants=variants,
        controls=controls,
        layout={
            "xaxis": {"title": "Value"},
            "yaxis": {
                "title": "Class",
                "automargin": True,
                "autorange": "reversed",
            },
            "height": max(600, 18 * n),
        },
    )


def _write_misclassification_explorer(
    path: Path,
    labels: list[str],
    matrix: list[list[int]],
) -> None:
    """Port of ``PlotPerclassDropdownAim``: per-class FP / FN stacked bars."""

    n = len(labels)
    off_diagonal = any(
        matrix[i][j] for i in range(n) for j in range(n) if i != j
    )
    if not off_diagonal:
        return

    traces: list[dict[str, Any]] = []
    classes: dict[str, dict[str, Any]] = {}
    class_error_totals = {
        i: sum(matrix[j][i] + matrix[i][j] for j in range(n) if j != i)
        for i in range(n)
    }
    class_error_order = sorted(
        range(n), key=lambda i: class_error_totals[i], reverse=True
    )
    initial_class = class_error_order[0]
    for i in range(n):
        tp = matrix[i][i]
        # FP: other actual classes predicted as i (column i, row j != i).
        fp_idx = [j for j in range(n) if j != i and matrix[j][i] > 0]
        fp_values = [matrix[j][i] for j in fp_idx]
        # FN: class i predicted as another class (row i, col j != i).
        fn_idx = [j for j in range(n) if j != i and matrix[i][j] > 0]
        fn_values = [matrix[i][j] for j in fn_idx]
        fp_sum = sum(fp_values)
        fn_sum = sum(fn_values)
        displayed_indices = sorted(set(fp_idx) | set(fn_idx))
        displayed_labels = [labels[j] for j in displayed_indices]
        error_order = [
            labels[j]
            for j in sorted(
                displayed_indices,
                key=lambda j: matrix[j][i] + matrix[i][j],
                reverse=True,
            )
        ]
        figure_height = max(250, 180 + 32 * len(displayed_labels))

        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "width": 0.5,
                "y": [labels[j] for j in fp_idx],
                "x": fp_values,
                "name": "FP (mistaken as)",
                "marker": {"color": "#1f77b4"},
                "visible": i == initial_class,
                "hovertext": [
                    f'{v} "{labels[j]}" misclassified as "{labels[i]}"'
                    for j, v in zip(fp_idx, fp_values)
                ],
                "hoverinfo": "text",
            }
        )
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "width": 0.5,
                "y": [labels[j] for j in fn_idx],
                "x": fn_values,
                "name": "FN (mistaken for)",
                "marker": {"color": "#2ca02c"},
                "visible": i == initial_class,
                "hovertext": [
                    f'{v} "{labels[i]}" misclassified as "{labels[j]}"'
                    for j, v in zip(fn_idx, fn_values)
                ],
                "hoverinfo": "text",
            }
        )

        visible = [False] * (n * 2)
        visible[i * 2] = True
        visible[i * 2 + 1] = True
        classes[str(i)] = {
            "title": (
                f'Misclassifications for "{labels[i]}" — '
                f"{tp} TP, {fp_sum} FP, {fn_sum} FN"
            ),
            "visible": visible,
            "height": figure_height,
            "orders": {
                "classlist": displayed_labels,
                "errors": error_order,
            },
            "yaxis": {
                "autorange": "reversed",
                "automargin": True,
                "categoryorder": "array",
            },
        }

    initial = classes[str(initial_class)]
    _write_interactive_html(
        path,
        title="Misclassification Explorer",
        kind="explorer",
        variants={
            "traces": traces,
            "classes": classes,
            "classOrders": {
                "classlist": [str(i) for i in range(n)],
                "errors": [str(i) for i in class_error_order],
            },
        },
        controls=[
            {
                "id": "class",
                "kind": "select",
                "label": "Class",
                "default": str(initial_class),
                "options": [
                    {
                        "value": str(i),
                        "label": (
                            f"{labels[i]} (TP:{matrix[i][i]} "
                            f"FP:{sum(matrix[j][i] for j in range(n) if j != i)} "
                            f"FN:{sum(matrix[i][j] for j in range(n) if j != i)})"
                        ),
                    }
                    for i in range(n)
                ],
            },
            {
                "id": "order",
                "kind": "select",
                "label": "Order by",
                "default": "errors",
                "options": [
                    {"value": "errors", "label": "FP + FN (descending)"},
                    {"value": "classlist", "label": "class list"},
                ],
            },
        ],
        layout={
            "barmode": "stack",
            "xaxis": {"title": "Number of Misclassifications", "rangemode": "tozero"},
            "yaxis": initial["yaxis"],
            "height": initial["height"],
            "showlegend": True,
            "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02},
        },
    )


def _write_result_figures_for_rows(figures_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels, matrix = _confusion_payload(rows)
    if not labels:
        return
    _write_confusion_matrix(figures_dir / "confusion_matrix.html", labels, matrix)
    _write_per_class_metrics(figures_dir / "per_class_metrics.html", labels, matrix)
    _write_misclassification_explorer(
        figures_dir / "misclassification_explorer.html", labels, matrix
    )


def _write_result_figures(figures_dir: Path, results_dir: Path) -> None:
    rows = _classification_rows(results_dir)
    rows_by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        head_name = row.get("head_name")
        if head_name:
            rows_by_head[str(head_name)].append(row)

    if len(rows_by_head) == 1:
        # Single-head: figures directly under figures/, no per-head subdir.
        (head_rows,) = rows_by_head.values()
        _write_result_figures_for_rows(figures_dir, head_rows)
        return
    for head_name, head_rows in rows_by_head.items():
        _write_result_figures_for_rows(figures_dir / head_name, head_rows)


def write_training_figures(
    *,
    metrics_csv: Path,
    figures_dir: Path,
    results_dir: Path | None,
    objective_to_head: dict[str, str] | None = None,
) -> list[Path]:
    """Write all available supervised training figures and return their paths."""

    before = set(figures_dir.rglob("*.html")) if figures_dir.exists() else set()
    metrics = _read_merged_metrics(metrics_csv)
    if metrics:
        _write_metric_lines(figures_dir, metrics, objective_to_head=objective_to_head)
    if results_dir is not None and results_dir.exists():
        _write_result_figures(figures_dir, results_dir)
    after = set(figures_dir.rglob("*.html")) if figures_dir.exists() else set()
    return sorted(after - before)
