"""End-to-end P1 thin slice: ``dojo train`` on the plankton-toyset fixture.

Exercises every architectural boundary once: a resolved run directory with
``config/`` / ``checkpoints/`` / ``results/`` / ``metrics/``; result Parquet
that round-trips through a fresh reader and filters by ``stage`` /
``record_type``; a ``_metadata.json`` sidecar that validates against the schema;
and deterministic, reproducible hashes across two runs.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.training.checkpoint import checkpoint_hash
from dojo.training.run import execute_train
from dojo.results import ResultReader, ResultsMetadata
from dojo.results.schemas import (
    RECORD_TYPE_CLASSIFICATION_OUTPUT,
    RECORD_TYPE_SAMPLE_METADATA,
    STAGE_TRAIN_VALIDATION,
)

pytestmark = pytest.mark.expensive


def _toy_cfg(output_root):
    from tests.fixtures.configs import toy_root_config

    return toy_root_config(
        canvas=(32, 32), max_epochs=1, batch_size=8, output_root=str(output_root)
    )


def _run(output_root):
    # No limit_train_batches: run the real train split so a trailing size-1
    # batch (which crashes BatchNorm without drop_last) is exercised.
    return execute_train(
        _toy_cfg(output_root),
        accelerator="cpu",
        precision="32-true",
        max_epochs=1,
        num_sanity_val_steps=0,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("run"))


def test_run_directory_has_canonical_layout(trained):
    assert (trained.config_dir / "resolved.yaml").exists()
    assert (trained.config_dir / "resolved.json").exists()
    assert list(trained.checkpoint_dir.glob("*.ckpt"))
    assert (trained.checkpoint_dir / "last.ckpt").exists()
    assert trained.best_checkpoint.exists()
    assert (trained.metrics_dir / "metrics.csv").exists()


def test_direct_run_omits_provenance_derived_config_artifacts(trained):
    # execute_train was called with a bare RootConfig, so there is no composition
    # history to record; these must be absent, not reconstructed.
    assert not (trained.config_dir / "composed.yaml").exists()
    assert not (trained.config_dir / "cli.txt").exists()
    assert not (trained.config_dir / "overrides.txt").exists()
    assert trained.results_dir is not None
    assert (trained.results_dir / "_metadata.json").exists()
    assert trained.checkpoint_hash.startswith("sha256:")


def test_results_round_trip_and_filter(trained):
    reader = ResultReader(trained.results_dir)

    total = reader.count()
    assert total == trained.result_record_count > 0

    classifications = list(
        reader.read({"record_type": RECORD_TYPE_CLASSIFICATION_OUTPUT})
    )
    sample_meta = list(reader.read({"record_type": RECORD_TYPE_SAMPLE_METADATA}))
    assert classifications and sample_meta
    assert len(classifications) + len(sample_meta) == total

    # Every row is the single P1 stage, and a stage filter selects them all.
    assert reader.count({"stage": STAGE_TRAIN_VALIDATION}) == total

    row = classifications[0]
    assert row["stage"] == STAGE_TRAIN_VALIDATION
    assert row["head_name"] == "species"
    assert row["head_hash"].startswith("sha256:")
    assert row["target_index"] in range(6)
    assert isinstance(row["target_name"], str)
    assert row["checkpoint_hash"] == trained.checkpoint_hash
    assert len(row["probabilities"]) == 6
    assert row["dataset_hash"] == trained.dataset_hash
    assert row["config_hash"] == trained.config_hash


def test_metadata_sidecar_validates(trained):
    payload = json.loads((trained.results_dir / "_metadata.json").read_text())
    meta = ResultsMetadata.model_validate(payload)

    assert meta.run_id
    assert meta.compatibility is not None
    assert meta.compatibility.target_schema_hash.startswith("sha256:")
    assert meta.compatibility.class_mapping_hash.startswith("sha256:")
    assert meta.compatibility.model_config_hash.startswith("sha256:")
    assert meta.compatibility.preprocessing_hash.startswith("sha256:")
    head = meta.record_types.classification_output.heads["species"]
    assert head.target == "species"
    assert head.head_hash.startswith("sha256:")
    assert len(head.classes) == 6
    assert head.class_mapping["0"] == head.classes[0]


def test_packaged_experiment_runs_via_cli(tmp_path):
    # The way a user runs it: `dojo train experiment=...` with inline overrides,
    # no --resolved-config. Exercises composition + the Typer wrapper end-to-end.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "experiment=p1/plankton-toy",
            "runtime.num_workers=0",
            "runtime.precision=32-true",
            "runtime.progress_bar=false",
            "training.max_epochs=1",
            f"output_root={tmp_path}",
        ],
    )

    assert result.exit_code == 0, result.output
    runs = list(tmp_path.glob("plankton-toy_p1_efficientnet_b0/*"))
    assert len(runs) == 1
    run_dir = runs[0]
    # A composed run records its full config artifact set (sweep_values.txt is
    # sweep-member-only and not expected here).
    config_dir = run_dir / "config"
    assert {path.name for path in config_dir.iterdir()} == {
        "composed.yaml",
        "resolved.yaml",
        "resolved.json",
        "cli.txt",
        "overrides.txt",
    }
    assert (config_dir / "overrides.txt").read_text().splitlines() == [
        "experiment=p1/plankton-toy",
        "runtime.num_workers=0",
        "runtime.precision=32-true",
        "runtime.progress_bar=false",
        "training.max_epochs=1",
        f"output_root={tmp_path}",
    ]
    assert (config_dir / "cli.txt").read_text().strip()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "results" / "_metadata.json").exists()
    assert (run_dir / "metrics" / "metrics.csv").exists()


def test_runs_with_multiprocess_dataloader_workers(tmp_path):
    # num_workers>0 forks DataLoader workers, so the dataset must pickle and the
    # shared collation must work across processes (fit loader and the results
    # scoring loader both read runtime.num_workers).
    from dojo.config_schemas import RootConfig
    from dojo.config_loader import resolve_runtime_and_paths
    from tests.fixtures.configs import toy_config_dict

    cfg_dict = toy_config_dict(
        canvas=(32, 32), max_epochs=1, batch_size=8, output_root=str(tmp_path)
    )
    cfg_dict["runtime"] = {"num_workers": 2, "progress_bar": False}
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(cfg_dict)).config
    assert cfg.runtime.num_workers == 2

    result = execute_train(
        cfg,
        accelerator="cpu",
        precision="32-true",
        max_epochs=1,
        num_sanity_val_steps=0,
    )

    assert (result.checkpoint_dir / "last.ckpt").exists()
    assert result.results_dir is not None
    assert result.result_record_count > 0


def test_hashes_are_deterministic_across_runs(tmp_path_factory, trained):
    second = _run(tmp_path_factory.mktemp("run2"))
    assert second.config_hash == trained.config_hash
    assert second.dataset_hash == trained.dataset_hash
    # Distinct run_ids resolve to distinct run directories.
    assert second.run_dir != trained.run_dir


def _state_dict_hash(ckpt_path) -> str:
    """Content hash of the trained weights alone (order-independent over keys)."""
    import hashlib

    import torch

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def test_hashes_reproducible_across_runs(tmp_path_factory):
    # Hashes are deterministic and reproducible across two independent runs of
    # the same config. config_hash / dataset_hash derive from the config / data
    # alone and must match exactly. For the checkpoint, the reproducible
    # invariant is the *trained weights*: execute_train seeds via runtime.seed
    # and deterministic=True pins the remaining kernels, so two CPU fits converge
    # to the same state_dict. checkpoint_hash itself is the SHA-256 of the whole
    # .ckpt file (artifact identity), which embeds run-specific ModelCheckpoint
    # state (absolute paths), so it is deterministic per file but differs between
    # run directories.
    def _deterministic_run(name):
        return execute_train(
            _toy_cfg(tmp_path_factory.mktemp(name)),
            accelerator="cpu",
            precision="32-true",
            max_epochs=1,
            num_sanity_val_steps=0,
            deterministic=True,
        )

    first = _deterministic_run("repro1")
    second = _deterministic_run("repro2")

    # Reproducible across runs.
    assert second.config_hash == first.config_hash
    assert second.dataset_hash == first.dataset_hash
    assert _state_dict_hash(second.best_checkpoint) == _state_dict_hash(first.best_checkpoint)

    # checkpoint_hash is a well-formed, deterministic function of the .ckpt file
    # (re-hashing the same artifact is stable), even though the two run files
    # differ by embedded run metadata.
    assert first.checkpoint_hash.startswith("sha256:")
    assert checkpoint_hash(first.best_checkpoint) == first.checkpoint_hash
    assert first.checkpoint_hash != second.checkpoint_hash
