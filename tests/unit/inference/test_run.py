"""Checkpoint-backed inference helpers."""

from __future__ import annotations

import json

import torch

from dojo.inference import execute_holdout_eval, execute_infer
from dojo.results import ResultReader
from dojo.results.schemas import (
    RECORD_TYPE_CLASSIFICATION_OUTPUT,
    RECORD_TYPE_EMBEDDING,
    STAGE_HOLDOUT_EVAL,
    STAGE_INFER,
)
from dojo.training.inference_contract import build_inference_contract
from dojo.training.task import SupervisedTaskModule
from tests.fixtures.configs import toy_root_config


def _checkpoint(tmp_path):
    cfg = toy_root_config(canvas=(32, 32), batch_size=8, output_root=str(tmp_path))
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
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "dojo_inference_contract": contract,
        },
        path,
    )
    return cfg, path


def test_execute_infer_predictions_writes_infer_rows(tmp_path):
    cfg, checkpoint = _checkpoint(tmp_path)

    result = execute_infer(
        cfg,
        checkpoint_path=checkpoint,
        output_kind="predictions",
    )

    reader = ResultReader(result.results_dir)
    rows = list(reader.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))
    assert rows
    assert rows[0]["stage"] == STAGE_INFER
    assert result.manifest_path.exists()


def test_execute_infer_embeddings_writes_embedding_rows(tmp_path):
    cfg, checkpoint = _checkpoint(tmp_path)

    result = execute_infer(
        cfg,
        checkpoint_path=checkpoint,
        output_kind="embeddings",
    )

    reader = ResultReader(result.results_dir)
    rows = list(reader.read({"record_type": RECORD_TYPE_EMBEDDING}))
    assert rows
    assert rows[0]["embedding_kind"] == "head_input_embedding"
    assert rows[0]["embedding_dim"] == 1280


def test_execute_holdout_eval_writes_holdout_stage(tmp_path):
    cfg, checkpoint = _checkpoint(tmp_path)

    result = execute_holdout_eval(cfg, checkpoint_path=checkpoint)

    reader = ResultReader(result.results_dir)
    rows = list(reader.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT}))
    assert rows
    assert rows[0]["stage"] == STAGE_HOLDOUT_EVAL
    assert result.metric_summary is not None
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metric_summary"] == result.metric_summary
    assert "f1_macro" in manifest["metric_summary"]["objectives"]["species"]
