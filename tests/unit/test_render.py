"""`dojo render run` regeneration behavior."""

from __future__ import annotations

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.results import Provenance, ResultWriter, classification_output_record
from tests.fixtures.configs import toy_root_config


def _finished_run(tmp_path, *, include_config: bool = True):
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "\n".join(
            (
                "epoch,step,train/loss,val/loss,train/species/loss,val/species/loss,val/species/f1_macro",
                "0,1,2.0,3.0,1.0,1.5,0.2",
                "1,2,1.0,2.0,0.5,1.0,0.4",
            )
        ),
        encoding="utf-8",
    )
    results_dir = run_dir / "results"
    writer = ResultWriter(results_dir, partition_by=["record_type"])
    provenance = Provenance(run_id="r", config_hash="sha256:c", dataset_hash="sha256:d")
    writer.write_records(
        [
            classification_output_record(
                provenance,
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
                provenance,
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
        ]
    )
    if include_config:
        config_dir = run_dir / "config"
        config_dir.mkdir()
        cfg = toy_root_config(output_root=str(tmp_path / "outputs"))
        OmegaConf.save(
            OmegaConf.create(cfg.model_dump(mode="json", exclude_none=True)),
            config_dir / "resolved.yaml",
        )
    return run_dir


def test_render_run_writes_full_figure_set_from_finished_run(tmp_path):
    run_dir = _finished_run(tmp_path)

    result = CliRunner().invoke(app, ["render", "run", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert {
        "loss_curves.html",
        "val_f1_curves.html",
        "confusion_matrix.html",
        "per_class_metrics.html",
        "misclassification_explorer.html",
    } <= {path.name for path in (run_dir / "figures").glob("*.html")}


def test_render_run_backup_preserves_previous_figures(tmp_path):
    run_dir = _finished_run(tmp_path)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir()
    (figures_dir / "previous.html").write_text("old figure", encoding="utf-8")

    result = CliRunner().invoke(app, ["render", "run", str(run_dir), "--backup"])

    assert result.exit_code == 0, result.output
    assert (run_dir / "figures.1" / "previous.html").read_text(encoding="utf-8") == "old figure"
    assert (figures_dir / "loss_curves.html").exists()
    assert not (figures_dir / "previous.html").exists()


def test_render_run_uses_identity_mapping_without_resolved_config(tmp_path):
    run_dir = _finished_run(tmp_path, include_config=False)

    result = CliRunner().invoke(app, ["render", "run", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert (run_dir / "figures" / "loss_curves.html").exists()
