"""Dataset preflight checks consumed before supervised training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dojo.config_schemas.root import RootConfig, Severity
from dojo.data.parquet_images import DataBundle


PreflightCheckName = Literal[
    "empty_train_classes",
    "non_contiguous_class_indices",
    "imbalance_ratio_gt",
]


@dataclass(frozen=True)
class PreflightIssue:
    check: PreflightCheckName
    severity: Severity
    message: str


def _issue(
    *,
    check: PreflightCheckName,
    severity: Severity,
    message: str,
) -> PreflightIssue | None:
    if severity == "ignore":
        return None
    return PreflightIssue(check=check, severity=severity, message=message)


def run_dataset_preflight(cfg: RootConfig, bundle: DataBundle) -> list[PreflightIssue]:
    """Run configured dataset preflight checks against frozen class counts."""

    preflight = cfg.runtime.preflight
    if not preflight.enabled:
        return []

    checks = preflight.checks
    issues: list[PreflightIssue] = []
    for head_name, head in cfg.model.heads.items():
        train_counts = bundle.class_counts_by_target.get("train", {}).get(
            head.target,
            {},
        )
        observed = sorted(train_counts)
        missing = [
            index
            for index in range(head.num_classes)
            if train_counts.get(index, 0) == 0
        ]
        item = _issue(
            check="empty_train_classes",
            severity=checks.empty_train_classes,
            message=(
                f"head {head_name!r} has {len(missing)} empty train classes: "
                f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
            ),
        ) if missing else None
        if item is not None:
            issues.append(item)

        if observed:
            expected = list(range(min(observed), max(observed) + 1))
            if observed != expected:
                item = _issue(
                    check="non_contiguous_class_indices",
                    severity=checks.non_contiguous_class_indices,
                    message=(
                        f"head {head_name!r} train class indices are non-contiguous: "
                        f"observed={observed[:20]}{'...' if len(observed) > 20 else ''}"
                    ),
                )
                if item is not None:
                    issues.append(item)

        nonzero_counts = [count for count in train_counts.values() if count > 0]
        if len(nonzero_counts) >= 2:
            ratio = max(nonzero_counts) / min(nonzero_counts)
            threshold_cfg = checks.imbalance_ratio_gt
            if ratio > threshold_cfg.threshold:
                item = _issue(
                    check="imbalance_ratio_gt",
                    severity=threshold_cfg.severity,
                    message=(
                        f"head {head_name!r} train class imbalance ratio {ratio:.3g} "
                        f"exceeds threshold {threshold_cfg.threshold:.3g}"
                    ),
                )
                if item is not None:
                    issues.append(item)

    return issues
