"""`dojo init` materializes editable starter assets."""

from __future__ import annotations

from typer.testing import CliRunner

from dojo.cli.main import app


def test_init_materializes_packaged_configs_and_skips_existing(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output
    starter = tmp_path / "configs" / "experiment" / "p1" / "plankton-toy.yaml"
    assert starter.exists()

    starter.write_text("local edit\n", encoding="utf-8")
    second = runner.invoke(app, ["init", str(tmp_path)])

    assert second.exit_code == 0, second.output
    assert "skip" in second.output
    assert starter.read_text(encoding="utf-8") == "local edit\n"


def test_init_data_copies_fixture_data(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path), "--data"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "example-data" / "plankton-toyset" / "data.parquet").exists()
