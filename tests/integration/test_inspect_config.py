import json

import torch
from typer.testing import CliRunner
from omegaconf import OmegaConf

from dojo.cli.main import app
from dojo.config_loader import resolve_runtime_and_paths
from dojo.training.inference_contract import build_inference_contract
from dojo.training.task import SupervisedTaskModule
from tests.fixtures.configs import toy_root_config


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


def test_inspect_config_compare_json_for_authored_configs(tmp_path):
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    config_a.write_text(
        """
# @package _global_
defaults:
  - /experiment/p1/plankton-toy
  - _self_
""",
        encoding="utf-8",
    )
    config_b.write_text(
        """
# @package _global_
defaults:
  - /experiment/p1/plankton-toy
  - _self_
training:
  batch_size: 8
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "config-compare",
            "--config-a",
            str(config_a),
            "--config-b",
            str(config_b),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config_a"]["source_kind"] == "authored"
    assert payload["config_b"]["source_kind"] == "authored"
    assert payload["sections"]["config_hash"]["equal"] is False
    assert any(
        diff["path"] == "training.batch_size"
        for diff in payload["sections"]["config_hash"]["diffs"]
    )
    assert payload["sections"]["other"]["diffs"] == []


def test_inspect_config_compare_accepts_resolved_config(tmp_path):
    authored = tmp_path / "authored.yaml"
    authored.write_text(
        """
# @package _global_
defaults:
  - /experiment/p1/plankton-toy
  - _self_
""",
        encoding="utf-8",
    )
    resolved = resolve_runtime_and_paths(
        toy_root_config(output_root=str(tmp_path))
    ).config
    resolved_path = tmp_path / "resolved.yaml"
    OmegaConf.save(
        OmegaConf.create(resolved.model_dump(mode="json", exclude_none=True)),
        resolved_path,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "config-compare",
            "--config-a",
            str(authored),
            "--config-b",
            str(resolved_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config_a"]["source_kind"] == "authored"
    assert payload["config_b"]["source_kind"] == "resolved"
    assert "preprocessing_hash" in payload["sections"]


def test_inspect_backbone_json():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "backbone",
            "experiment=p1/plankton-toy",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"output_dim": 1280' in result.output


def test_inspect_checkpoint_json(tmp_path):
    cfg = toy_root_config(output_root=str(tmp_path))
    contract = build_inference_contract(
        cfg,
        class_mapping={index: str(index) for index in range(6)},
    )
    module = SupervisedTaskModule(
        model_config=cfg.model,
        training_config=cfg.training,
        objectives_config=cfg.objectives,
        optimizer_config=cfg.optimizer,
        class_counts_by_head={"species": {index: 1 for index in range(6)}},
        inference_contract=contract,
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "dojo_inference_contract": contract},
        checkpoint,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inspect", "checkpoint", str(checkpoint), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert '"has_inference_contract": true' in result.output
    assert "target_schema_hash" in result.output
