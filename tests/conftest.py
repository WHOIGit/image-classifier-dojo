"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.fixtures.defects import build_defect


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Also run @pytest.mark.expensive tests (real training fits / "
        "multiprocess dataloaders). Skipped by default.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        return
    skip_expensive = pytest.mark.skip(reason="expensive: pass --run-expensive to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)


@pytest.fixture
def defect_dataset(tmp_path):
    """Factory: materialize a malformed ``parquet_images`` dataset under ``tmp_path``.

    Mutates the clean ``plankton-toyset`` fixture with one injector from
    ``tests/fixtures/defects.py`` and writes it as a throwaway dataset dir.

    Usage::

        def test_preflight_rejects_empty_eval(defect_dataset):
            path = defect_dataset("empty_eval_class", classname="Trichodesmium")
            # point a parquet_images data config at `path` and assert preflight fails

    Returns the dataset directory (contains ``data/00000.parquet``). Calling more
    than once is fine; repeated defect names get a numeric suffix so the dirs
    don't collide.
    """
    seen: dict[str, int] = {}

    def _make(defect: str, **kwargs):
        n = seen.get(defect, 0)
        seen[defect] = n + 1
        name = defect if n == 0 else f"{defect}_{n}"
        return build_defect(defect, tmp_path / name, **kwargs)

    return _make
