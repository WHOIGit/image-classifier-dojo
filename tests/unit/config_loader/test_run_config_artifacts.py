"""The run ``config/`` artifact set.

`DESIGN-DOC/03-configuration.md` specifies five files; the resolved pair is
unconditional and the provenance-derived three appear only when the caller knows
how the config was composed.
"""

import json

import pytest
from omegaconf import OmegaConf

from dojo.config_loader import (
    ConfigProvenance,
    compose_and_resolve,
    write_run_config_artifacts,
)


@pytest.fixture
def resolved():
    return compose_and_resolve(overrides=["experiment=p1/plankton-toy"])


def test_provenance_writes_the_full_artifact_set(tmp_path, resolved):
    provenance = ConfigProvenance(
        composed=OmegaConf.create({"training": {"max_epochs": 3}}),
        overrides=("training.max_epochs=3", "runtime.num_workers=0"),
        invoked_command="dojo train experiment=p1/plankton-toy",
    )

    write_run_config_artifacts(tmp_path, resolved.config, provenance=provenance)

    assert {path.name for path in tmp_path.iterdir()} == {
        "composed.yaml",
        "resolved.yaml",
        "resolved.json",
        "cli.txt",
        "overrides.txt",
    }


def test_without_provenance_only_the_resolved_pair_is_written(tmp_path, resolved):
    # A bare RootConfig has no composition history; the other three must be
    # absent rather than reconstructed from the resolved config.
    write_run_config_artifacts(tmp_path, resolved.config)

    assert {path.name for path in tmp_path.iterdir()} == {
        "resolved.yaml",
        "resolved.json",
    }


def test_resolved_yaml_and_json_agree(tmp_path, resolved):
    write_run_config_artifacts(tmp_path, resolved.config)

    from_yaml = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "resolved.yaml"), resolve=True
    )
    from_json = json.loads((tmp_path / "resolved.json").read_text())

    assert from_yaml == from_json


def test_resolved_artifacts_carry_generated_runtime_values(tmp_path, resolved):
    write_run_config_artifacts(tmp_path, resolved.config)

    from_json = json.loads((tmp_path / "resolved.json").read_text())

    assert from_json["runtime"]["run_id"] == resolved.config.runtime.run_id
    assert "{" not in from_json["runtime"]["run_id"]


def test_overrides_recorded_verbatim_one_per_line(tmp_path, resolved):
    provenance = ConfigProvenance(
        composed=OmegaConf.create({}),
        overrides=("training.batch_size=8", "runtime.seed=7"),
    )

    write_run_config_artifacts(tmp_path, resolved.config, provenance=provenance)

    assert (tmp_path / "overrides.txt").read_text() == (
        "training.batch_size=8\nruntime.seed=7\n"
    )


def test_empty_overrides_still_writes_the_file(tmp_path, resolved):
    provenance = ConfigProvenance(composed=OmegaConf.create({}), overrides=())

    write_run_config_artifacts(tmp_path, resolved.config, provenance=provenance)

    assert (tmp_path / "overrides.txt").read_text() == ""


def test_cli_txt_omitted_when_the_invocation_is_unknown(tmp_path, resolved):
    # Only the CLI knows the real command line; a non-CLI caller composing a
    # config gets composed.yaml and overrides.txt but no fabricated cli.txt.
    provenance = ConfigProvenance(
        composed=OmegaConf.create({}), overrides=("runtime.seed=1",)
    )

    write_run_config_artifacts(tmp_path, resolved.config, provenance=provenance)

    assert not (tmp_path / "cli.txt").exists()
    assert (tmp_path / "composed.yaml").exists()


def test_composed_yaml_preserves_the_pre_resolution_tree(tmp_path, resolved):
    # composed.yaml is the composed DictConfig, not a Pydantic round-trip, so it
    # keeps template tokens that resolution would have rendered away.
    provenance = ConfigProvenance(
        composed=OmegaConf.create({"runtime": {"run_id": "{coolname:noseed}"}}),
        overrides=(),
    )

    write_run_config_artifacts(tmp_path, resolved.config, provenance=provenance)

    composed = OmegaConf.load(tmp_path / "composed.yaml")
    assert composed.runtime.run_id == "{coolname:noseed}"


def test_writer_creates_a_missing_config_directory(tmp_path, resolved):
    config_dir = tmp_path / "run" / "config"

    write_run_config_artifacts(config_dir, resolved.config)

    assert (config_dir / "resolved.yaml").exists()


def test_experiment_selector_survives_into_provenance():
    # The compositor consumes `experiment=` as the config root and drops it from
    # ComposedConfig.overrides; provenance must keep it, or overrides.txt cannot
    # reproduce the run.
    result = compose_and_resolve(
        overrides=["experiment=p1/plankton-toy", "training.max_epochs=1"]
    )

    assert result.provenance is not None
    assert result.provenance.overrides == (
        "experiment=p1/plankton-toy",
        "training.max_epochs=1",
    )


def test_writer_returns_the_paths_it_wrote(tmp_path, resolved):
    provenance = ConfigProvenance(
        composed=OmegaConf.create({}), overrides=(), invoked_command="dojo train"
    )

    written = write_run_config_artifacts(
        tmp_path, resolved.config, provenance=provenance
    )

    assert sorted(path.name for path in written) == [
        "cli.txt",
        "composed.yaml",
        "overrides.txt",
        "resolved.json",
        "resolved.yaml",
    ]
    assert all(path.exists() for path in written)
