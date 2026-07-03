"""checkpoint_hash + filename convention."""

from __future__ import annotations

from dojo.training.checkpoint import checkpoint_filename, checkpoint_hash
from dojo.training.inference_contract import build_inference_contract
from dojo.training.task import SupervisedTaskModule
from tests.fixtures.configs import toy_root_config


def test_checkpoint_hash_is_sha256_of_bytes(tmp_path):
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"weights-bytes")
    h1 = checkpoint_hash(path)
    assert h1.startswith("sha256:")
    # Deterministic, content-addressed.
    assert checkpoint_hash(path) == h1
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"different")
    assert checkpoint_hash(other) != h1


def test_checkpoint_filename_embeds_first6_hex():
    name = checkpoint_filename("loss-1.23_epoch-003", "sha256:7ff91abc1234", ext="ckpt")
    assert name == "loss-1.23_epoch-003.7ff91a.ckpt"


def test_task_embeds_inference_contract_on_checkpoint_save():
    cfg = toy_root_config()
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
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["dojo_inference_contract"]["compatibility"][
        "target_schema_hash"
    ].startswith("sha256:")
