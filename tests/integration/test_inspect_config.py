from typer.testing import CliRunner

from dojo.cli.main import app


def test_inspect_config_json_for_p1_experiment():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "config",
            "experiment=p1/plankton-toy",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"valid": true' in result.output
    assert "classification_output" in result.output
    assert "train_validation" in result.output


def test_inspect_config_experiment_selector_rejects_file_path():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "config",
            "experiment=configs/experiment/p1/plankton-mini_efficientnet.yaml",
        ],
    )

    assert result.exit_code == 1
    assert "experiment= selects a Hydra experiment group" in result.output
    assert "--config configs/experiment/p1/plankton-mini_efficientnet.yaml" in result.output


def test_inspect_config_experiment_selector_rejects_configs_prefix():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "config",
            "experiment=configs/experiment/p1/plankton-mini_efficientnet",
        ],
    )

    assert result.exit_code == 1
    assert "experiment= selects a Hydra experiment group" in result.output
    assert "--config configs/experiment/p1/plankton-mini_efficientnet.yaml" in result.output
    assert "experiment=p1/plankton-mini_efficientnet" in result.output
