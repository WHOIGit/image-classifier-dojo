# datasets — local and committed datasets

## Purpose

Repository-root datasets and local data mirrors consumed by Dojo data configs.
This tree supports manifest-backed image classification experiments without
making generated run artifacts part of the source contract.

## Local Contracts

- Keep dataset references repo-root relative in configs and manifests when
  practical, matching `./datasets/...` paths.
- Do not place generated run outputs, model checkpoints, metrics, or result
  ledgers under `datasets/`; use `runs/` or the configured output roots.
- Preserve existing manifest column names used by configs, including split,
  image path/bytes, sample ID, and target label columns.
- Treat large dataset binaries as data assets, not documentation; summarize
  stable behavior in Markdown instead of duplicating large contents.

## Work Guidance

- Before changing a manifest, check all `configs/data/*.yaml` references that
  consume it.
- Prefer additive manifest columns over renaming or deleting columns used by
  existing configs.
- Keep small reusable dev datasets committed only when they are intended for
  tests, examples, or reproducible local experiments.

## Verification

- Dataset config/schema checks are covered by `pytest`.
- For dataset-specific changes, run or document a relevant `dojo inspect data
  data=<name>` check when the vision stack is available.

## Child DOX Index
