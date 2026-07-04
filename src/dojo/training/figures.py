"""Standalone HTML figure generation for supervised training runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dojo.results import ResultReader
from dojo.results.schemas import RECORD_TYPE_CLASSIFICATION_OUTPUT

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _write_plotly_html(
    path: Path,
    *,
    title: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traces": traces,
        "layout": {"title": title, **(layout or {})},
        "config": {"responsive": True, **(config or {})},
    }
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                f"<title>{title}</title>",
                f'<script src="{PLOTLY_CDN}"></script>',
                "</head>",
                "<body>",
                '<div id="plot" style="width:100%;height:95vh;"></div>',
                "<script>",
                f"const payload = {json.dumps(payload)};",
                "Plotly.newPlot('plot', payload.traces, payload.layout, payload.config);",
                "</script>",
                "</body>",
                "</html>",
            ]
        ),
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


def _line_trace(
    rows: list[dict[str, float | int]],
    key: str,
    name: str,
    *,
    normalize: bool = False,
) -> dict[str, Any] | None:
    x, y = _series(rows, key)
    if not y:
        return None
    if normalize:
        baseline = y[0]
        if baseline != 0:
            y = [value / baseline for value in y]
    return {"type": "scatter", "mode": "lines+markers", "x": x, "y": y, "name": name}


def _mean_metric_trace(
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
    return {"type": "scatter", "mode": "lines+markers", "x": xs, "y": ys, "name": name}


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


def _write_loss_lines(
    figures_dir: Path,
    metrics: list[dict[str, float | int]],
    *,
    train_key: str,
    val_key: str,
    title: str,
) -> None:
    loss_traces = [
        trace
        for trace in (
            _line_trace(metrics, train_key, train_key),
            _line_trace(metrics, val_key, val_key),
        )
        if trace is not None
    ]
    if loss_traces:
        _write_plotly_html(
            figures_dir / "loss_curves.html",
            title=title,
            traces=loss_traces,
            layout={"xaxis": {"title": "Epoch"}, "yaxis": {"title": "Loss"}},
        )

    normalized_loss_traces = [
        trace
        for trace in (
            _line_trace(metrics, train_key, f"{train_key} normalized", normalize=True),
            _line_trace(metrics, val_key, f"{val_key} normalized", normalize=True),
        )
        if trace is not None
    ]
    if normalized_loss_traces:
        _write_plotly_html(
            figures_dir / "loss_curves_normalized.html",
            title="Loss Normalized to First Epoch",
            traces=normalized_loss_traces,
            layout={"xaxis": {"title": "Epoch"}, "yaxis": {"title": "Loss / epoch0"}},
        )


def _write_f1_lines(
    path: Path,
    *,
    title: str,
    metrics: list[dict[str, float | int]],
    keys_by_trace: list[tuple[list[str], str]],
) -> None:
    f1_traces = []
    for keys, name in keys_by_trace:
        trace = (
            _line_trace(metrics, keys[0], name)
            if len(keys) == 1
            else _mean_metric_trace(metrics, keys, name)
        )
        if trace is not None:
            f1_traces.append(trace)
    if f1_traces:
        _write_plotly_html(
            path,
            title=title,
            traces=f1_traces,
            layout={"xaxis": {"title": "Epoch"}, "yaxis": {"title": "F1"}},
        )


def _write_head_metric_lines(
    figures_dir: Path,
    metrics: list[dict[str, float | int]],
    *,
    objective_name: str,
    head_name: str,
) -> None:
    head_dir = figures_dir / head_name
    _write_loss_lines(
        head_dir,
        metrics,
        train_key=f"train/{objective_name}/loss",
        val_key=f"val/{objective_name}/loss",
        title=f"{head_name} Training and Validation Loss",
    )
    head_f1_keys = [
        ([key], key)
        for key in (
            f"val/{objective_name}/f1_macro",
            f"val/{objective_name}/f1_micro",
        )
    ]
    _write_f1_lines(
        head_dir / "val_f1_curves.html",
        title=f"{head_name} Validation F1 per Epoch",
        metrics=metrics,
        keys_by_trace=head_f1_keys,
    )


def _write_metric_lines(
    figures_dir: Path,
    metrics: list[dict[str, float | int]],
    objective_to_head: dict[str, str] | None = None,
) -> None:
    _write_loss_lines(
        figures_dir,
        metrics,
        train_key="train/loss",
        val_key="val/loss",
        title="Training and Validation Loss",
    )

    macro_keys = _unique_metric_keys(metrics, prefix="val/", suffix="/f1_macro")
    micro_keys = _unique_metric_keys(metrics, prefix="val/", suffix="/f1_micro")
    top_level_f1 = [
        (macro_keys, "val/f1_macro mean"),
        (micro_keys, "val/f1_micro mean"),
    ]
    _write_f1_lines(
        figures_dir / "val_f1_curves.html",
        title="Mean Validation F1 per Epoch",
        metrics=metrics,
        keys_by_trace=[item for item in top_level_f1 if item[0]],
    )

    objective_to_head = objective_to_head or {}
    for objective in _objective_names(metrics):
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


def _per_class_metrics(
    labels: list[str],
    matrix: list[list[int]],
) -> dict[str, list[float | int | str]]:
    metrics: dict[str, list[float | int | str]] = {
        "class": labels,
        "support": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for index, row in enumerate(matrix):
        tp = row[index]
        support = sum(row)
        predicted = sum(matrix[actual][index] for actual in range(len(matrix)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        metrics["support"].append(support)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
    return metrics


def _write_result_figures_for_rows(figures_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels, matrix = _confusion_payload(rows)
    if not labels:
        return

    _write_plotly_html(
        figures_dir / "confusion_matrix.html",
        title="Confusion Matrix",
        traces=[
            {
                "type": "heatmap",
                "z": matrix,
                "x": labels,
                "y": labels,
                "colorscale": "Viridis",
                "hoverongaps": False,
            }
        ],
        layout={
            "xaxis": {"title": "Predicted", "automargin": True},
            "yaxis": {"title": "Actual", "automargin": True},
        },
    )

    metrics = _per_class_metrics(labels, matrix)
    metric_names = ["f1", "precision", "recall", "support"]
    traces = [
        {
            "type": "bar",
            "x": metrics[name],
            "y": metrics["class"],
            "orientation": "h",
            "name": name,
            "visible": name == "f1",
        }
        for name in metric_names
    ]
    buttons = [
        {
            "label": name,
            "method": "update",
            "args": [
                {"visible": [candidate == name for candidate in metric_names]},
                {"title": f"Per-Class {name}"},
            ],
        }
        for name in metric_names
    ]
    _write_plotly_html(
        figures_dir / "per_class_metrics.html",
        title="Per-Class Metrics",
        traces=traces,
        layout={
            "xaxis": {"title": "Value"},
            "yaxis": {"title": "Class", "automargin": True},
            "updatemenus": [{"buttons": buttons, "direction": "down"}],
            "height": max(600, 18 * len(labels)),
        },
    )


def _write_result_figures(figures_dir: Path, results_dir: Path) -> None:
    rows = _classification_rows(results_dir)
    rows_by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        head_name = row.get("head_name")
        if head_name:
            rows_by_head[str(head_name)].append(row)
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
