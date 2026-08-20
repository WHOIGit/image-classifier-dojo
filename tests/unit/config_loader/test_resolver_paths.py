"""Output-path resolution: name sanitization and platform warnings."""

from __future__ import annotations

from pathlib import Path

from dojo.config_loader import resolve_runtime_and_paths
from dojo.config_schemas import RootConfig
from dojo.config_loader.resolver import render_template
from tests.fixtures.configs import toy_config_dict


def _pretend_windows(monkeypatch, enabled: bool) -> None:
    """Flip the one platform hook shared by sanitization and the warnings.

    Patching ``os.name`` itself is not an option: pathlib reads it at call time
    and would hand back an uninstantiable ``WindowsPath``.
    """

    monkeypatch.setattr("dojo.storage.paths.is_windows", lambda: enabled)


def _config(**overrides) -> RootConfig:
    raw = toy_config_dict()
    for key, value in overrides.items():
        raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
    return RootConfig.model_validate(raw)


def test_template_literals_still_separate_directories(monkeypatch):
    _pretend_windows(monkeypatch, True)
    values = {"experiment": {"name": "p2/08_multihead"}}
    rendered = render_template("{experiment.name}/run", values)
    # The "/" inside the substituted value nests, as it does today on Linux.
    assert rendered == "p2/08_multihead/run"


def test_windows_illegal_characters_are_replaced_in_run_paths(monkeypatch):
    _pretend_windows(monkeypatch, True)
    values = {"experiment": {"name": 'sweep: lr=1e-3?'}}
    assert render_template("{experiment.name}", values) == "sweep_ lr=1e-3_"


def test_substituted_names_are_untouched_on_posix(monkeypatch):
    _pretend_windows(monkeypatch, False)
    values = {"experiment": {"name": "sweep: lr=1e-3?"}}
    assert render_template("{experiment.name}", values) == "sweep: lr=1e-3?"


def test_absolute_substitutions_are_left_alone(monkeypatch):
    _pretend_windows(monkeypatch, True)
    values = {"output_root": "C:/runs/dojo"}
    assert render_template("{output_root}/x", values) == "C:/runs/dojo/x"


def test_run_dir_is_creatable_for_a_hostile_experiment_name(tmp_path, monkeypatch):
    _pretend_windows(monkeypatch, True)
    cfg = _config(
        experiment={"name": "lr:1e-3"},
        output_root=str(tmp_path),
        training_outputs={"dir_template": "{experiment.name}/{runtime.run_id}"},
    )
    resolved = resolve_runtime_and_paths(cfg, cwd=tmp_path)
    training_dir = Path(resolved.config.training_outputs.dir)
    assert "lr_1e-3" in training_dir.parts


def test_num_workers_warns_only_on_windows(tmp_path, monkeypatch):
    cfg = _config(output_root=str(tmp_path), runtime={"num_workers": 8})

    _pretend_windows(monkeypatch, False)
    assert not [w for w in resolve_runtime_and_paths(cfg, cwd=tmp_path).warnings
                if "num_workers" in w]

    _pretend_windows(monkeypatch, True)
    warnings = resolve_runtime_and_paths(cfg, cwd=tmp_path).warnings
    assert any("num_workers=8" in warning and "spawned" in warning for warning in warnings)


def test_no_num_workers_warning_when_workers_are_disabled(tmp_path, monkeypatch):
    _pretend_windows(monkeypatch, True)
    cfg = _config(output_root=str(tmp_path), runtime={"num_workers": 0})
    assert not [
        warning
        for warning in resolve_runtime_and_paths(cfg, cwd=tmp_path).warnings
        if "num_workers" in warning
    ]
