"""Smoke tests for the defect-injection fixture tooling.

These guard `tests/fixtures/defects.py` against rot: every registered injector
must build, with its default kwargs, a non-empty parquet_images dataset off the
clean toy fixture. (The preflight/data layer that consumes these malformed
datasets is not built yet, so behavioral assertions live elsewhere later.)
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from tests.fixtures.defects import DEFECTS, load_fixture_table


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_defect_builds_parquet_images(defect, defect_dataset):
    path = defect_dataset(defect)
    files = list((path / "data").glob("*.parquet"))
    assert files, f"{defect}: no parquet written"
    table = pq.read_table(files[0])
    assert table.num_rows > 0


def test_clean_fixture_loads():
    table = load_fixture_table("plankton-toyset")
    assert table.num_rows == 62
    assert {"image", "label", "classname", "split"} <= set(table.schema.names)
