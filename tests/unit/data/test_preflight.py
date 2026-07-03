"""Dataset preflight checks."""

from __future__ import annotations

from dojo.data import build_datasets, run_dataset_preflight
from tests.fixtures.configs import toy_root_config


def test_empty_train_classes_are_errors_by_default():
    cfg = toy_root_config()
    cfg.model.heads["species"].num_classes = 7
    bundle = build_datasets(cfg)

    issues = run_dataset_preflight(cfg, bundle)

    assert any(
        issue.check == "empty_train_classes" and issue.severity == "error"
        for issue in issues
    )


def test_imbalance_ratio_uses_configured_warn_threshold():
    cfg = toy_root_config()
    cfg.runtime.preflight.checks.empty_train_classes = "ignore"
    cfg.runtime.preflight.checks.imbalance_ratio_gt.threshold = 2.0
    cfg.runtime.preflight.checks.imbalance_ratio_gt.severity = "warn"
    bundle = build_datasets(cfg)

    issues = run_dataset_preflight(cfg, bundle)

    assert any(
        issue.check == "imbalance_ratio_gt" and issue.severity == "warn"
        for issue in issues
    )


def test_preflight_can_be_disabled():
    cfg = toy_root_config()
    cfg.runtime.preflight.enabled = False
    cfg.model.heads["species"].num_classes = 7
    bundle = build_datasets(cfg)

    assert run_dataset_preflight(cfg, bundle) == []
