"""Standalone training figure generation."""

from __future__ import annotations

from dojo.results import Provenance, ResultWriter, classification_output_record
from dojo.training.figures import merge_metrics_csv, write_training_figures


def test_write_training_figures_from_metrics_and_results(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metrics_csv = metrics_dir / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "epoch",
                        "step",
                        "train/loss",
                        "val/loss",
                        "train/species/loss",
                        "val/species/loss",
                        "val/species/f1_macro",
                        "val/species/f1_micro",
                        "train/genus/loss",
                        "val/genus/loss",
                        "val/genus/f1_macro",
                        "val/genus/f1_micro",
                    ]
                ),
                "0,1,2.0,3.0,1.0,1.5,0.2,0.3,1.0,1.5,0.4,0.5",
                "1,2,1.0,2.0,0.5,1.0,0.4,0.5,0.5,1.0,0.6,0.7",
            ]
        ),
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir, partition_by=["record_type"])
    prov = Provenance(run_id="r", config_hash="sha256:c", dataset_hash="sha256:d")
    writer.write_records(
        [
            classification_output_record(
                prov,
                sample_id="s0",
                split="val",
                head_name="species",
                prediction_index=0,
                prediction_label="A",
                prediction_confidence=0.9,
                logits=[2.0, 0.0],
                probabilities=[0.9, 0.1],
                head_hash="sha256:h",
                target_index=0,
                target_name="A",
            ),
            classification_output_record(
                prov,
                sample_id="s1",
                split="val",
                head_name="species",
                prediction_index=0,
                prediction_label="A",
                prediction_confidence=0.6,
                logits=[1.0, 0.5],
                probabilities=[0.6, 0.4],
                head_hash="sha256:h",
                target_index=1,
                target_name="B",
            ),
            classification_output_record(
                prov,
                sample_id="s0",
                split="val",
                head_name="genus",
                prediction_index=1,
                prediction_label="Y",
                prediction_confidence=0.8,
                logits=[0.0, 2.0],
                probabilities=[0.2, 0.8],
                head_hash="sha256:g",
                target_index=1,
                target_name="Y",
            ),
            classification_output_record(
                prov,
                sample_id="s1",
                split="val",
                head_name="genus",
                prediction_index=0,
                prediction_label="X",
                prediction_confidence=0.7,
                logits=[1.5, 0.5],
                probabilities=[0.7, 0.3],
                head_hash="sha256:g",
                target_index=1,
                target_name="Y",
            ),
        ]
    )

    paths = write_training_figures(
        metrics_csv=metrics_csv,
        figures_dir=tmp_path / "figures",
        results_dir=results_dir,
    )

    figures_dir = tmp_path / "figures"
    names = {path.relative_to(figures_dir).as_posix() for path in paths}
    assert {
        "loss_curves.html",
        "loss_curves_normalized.html",
        "val_f1_curves.html",
        "species/loss_curves.html",
        "species/loss_curves_normalized.html",
        "species/val_f1_curves.html",
        "species/confusion_matrix.html",
        "species/per_class_metrics.html",
        "genus/loss_curves.html",
        "genus/loss_curves_normalized.html",
        "genus/val_f1_curves.html",
        "genus/confusion_matrix.html",
        "genus/per_class_metrics.html",
    } <= names
    assert not (figures_dir / "confusion_matrix.html").exists()
    assert "val/f1_macro mean" in (figures_dir / "val_f1_curves.html").read_text()
    assert "Plotly.newPlot" in (figures_dir / "species" / "confusion_matrix.html").read_text()


def test_merge_metrics_csv_combines_train_and_val_rows(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "epoch,step,train/loss,val/loss",
                "0,1,,3.0",
                "0,1,2.0,",
                "1,2,,2.5",
                "1,2,1.5,",
            ]
        ),
        encoding="utf-8",
    )

    merge_metrics_csv(metrics_csv)

    lines = metrics_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "train/loss" in lines[0]
    assert "val/loss" in lines[0]
    assert "2.0" in lines[1]
    assert "3.0" in lines[1]
