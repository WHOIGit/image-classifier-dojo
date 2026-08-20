"""dataset_hash: deterministic over path+size, sensitive to bindings/content."""

from __future__ import annotations

from dojo.config_schemas import RootConfig
from dojo.data.identity import compute_dataset_hash
from tests.fixtures.configs import toy_config_dict


def _data_cfg(**overrides) -> RootConfig:
    raw = toy_config_dict()
    raw["data"].update(overrides)
    return RootConfig.model_validate(raw).data


FILES = [("train-00000.parquet", 100), ("validation-00000.parquet", 50)]


def test_same_inputs_same_hash():
    cfg = _data_cfg()
    h1, p1 = compute_dataset_hash(cfg, FILES)
    h2, p2 = compute_dataset_hash(cfg, list(reversed(FILES)))  # sorted internally
    assert h1 == h2
    assert p1 == p2 == "uri_size"
    assert h1.startswith("sha256:")


def test_file_size_change_changes_hash():
    cfg = _data_cfg()
    base, _ = compute_dataset_hash(cfg, FILES)
    bigger, _ = compute_dataset_hash(cfg, [("train-00000.parquet", 101), FILES[1]])
    assert base != bigger


def test_binding_change_changes_hash():
    base, _ = compute_dataset_hash(_data_cfg(), FILES)
    # Rebinding the sample id column is a different dataset contract.
    rebound, _ = compute_dataset_hash(_data_cfg(sample_id_column="classname"), FILES)
    assert base != rebound


def test_mtime_is_not_part_of_identity():
    # Identity is path + size only; the same (path, size) list always hashes the
    # same regardless of when/where the files were written.
    cfg = _data_cfg()
    assert compute_dataset_hash(cfg, FILES) == compute_dataset_hash(cfg, FILES)


def test_windows_separators_hash_as_posix_separators():
    # The same nested dataset discovered on Windows ("2022\\train-0.parquet")
    # and on Linux ("2022/train-0.parquet") is one dataset, one hash.
    cfg = _data_cfg()
    posix = [("2022/train-00000.parquet", 100), ("2022/validation-00000.parquet", 50)]
    windows = [
        ("2022\\train-00000.parquet", 100),
        ("2022\\validation-00000.parquet", 50),
    ]
    assert compute_dataset_hash(cfg, posix) == compute_dataset_hash(cfg, windows)


def test_file_order_is_settled_on_normalized_paths():
    # "/" (0x2f) and "\\" (0x5c) sort differently against "0" (0x30), so the
    # sort has to happen after normalization or the two platforms disagree.
    cfg = _data_cfg()
    posix = [("a/b.parquet", 1), ("a0.parquet", 2)]
    windows = [("a\\b.parquet", 1), ("a0.parquet", 2)]
    assert compute_dataset_hash(cfg, posix) == compute_dataset_hash(cfg, windows)
