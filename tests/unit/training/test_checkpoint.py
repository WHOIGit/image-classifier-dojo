"""checkpoint_hash + filename convention."""

from __future__ import annotations

from dojo.training.checkpoint import checkpoint_filename, checkpoint_hash


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
