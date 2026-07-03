"""`dojo inspect dataset` reports targets and writes the stats cache."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
from typer.testing import CliRunner

from dojo.cli.main import app


def test_inspect_dataset_json_and_stats_cache(tmp_path):
    cache_path = tmp_path / "stats.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "dataset",
            "experiment=p1/plankton-toy",
            f"+data.stats_cache_uri={cache_path}",
            "--stats",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert payload["dataset_hash"].startswith("sha256:")
    assert payload["wrote_stats_cache"] is True
    assert payload["aspects"]["class_counts"]["per_head"]["species"]["total"] == 49
    assert payload["aspects"]["class_mapping"]["per_head"]["species"]["0"] == "bubble"
    assert payload["aspects"]["class_counts"]["per_head"]["species"]["counts"]["0"] == 19

    cached = json.loads(cache_path.read_text())
    assert cached["dataset_hash"] == payload["dataset_hash"]
    assert cached["dataset_content_hash"].startswith("sha256:")
    assert cached["created_by"] == "dojo inspect dataset"
    assert "rows" in payload["aspects"]["dimensions"]
    assert "rows" not in cached["aspects"]["dimensions"]
    dimensions_uri = cached["aspects"]["dimensions"]["parquet_uri"]
    assert dimensions_uri == "stats.dimensions.parquet"
    dimensions_path = cache_path.with_name(dimensions_uri)
    dimensions = pq.read_table(dimensions_path)
    assert dimensions.num_rows == 62
    assert dimensions.column_names == [
        "sample_id",
        "split",
        "native_width_px",
        "native_height_px",
    ]


def test_inspect_dataset_text_shows_class_mapping():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "dataset",
            "experiment=p1/plankton-toy",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Class Mapping" in result.output
    assert "Index" in result.output
    assert "Label" in result.output
    assert "Train Count" in result.output
    assert "bubble" in result.output
