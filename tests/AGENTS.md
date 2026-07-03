# tests — test suite

## Purpose

Unit and integration tests for the `dojo` package, plus committed fixtures.

- `unit/` — mirrors the package (`config_schemas`, `data`, `storage`, `training`,
  `model`, `results`, `inference`).
- `integration/` — config fall-through, init, inspect, supervised train, real
  fit.
- `fixtures/` — the `plankton-toyset` dataset (LFS), synthetic builders,
  defect injectors (`defects.py`), and config fixtures.

## Ownership

Owns test code and fixtures. `conftest.py` provides the `--run-expensive`
option and the `defect_dataset` factory.

## Local Contracts

- Real Lightning fits / multiprocess dataloaders are marked
  `@pytest.mark.expensive` and skipped by default.
- `defect_dataset(injector, ...)` mutates the clean `plankton-toyset` fixture to
  produce malformed datasets under `tmp_path` for preflight tests.
- Fixture Parquet is Git-LFS-tracked.

## Verification

- `pytest` (fast); `pytest --run-expensive` (everything).
</content>
