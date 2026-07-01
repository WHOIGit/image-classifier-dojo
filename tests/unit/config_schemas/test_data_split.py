import pytest
from pydantic import ValidationError

from dojo.config_schemas.root import DataConfig


def _base(**overrides):
    data = dict(
        backend="parquet_images",
        manifest_uri="x",
        sample_id_column="id",
        images={"column": "image"},
        targets={"species": {"label_index_column": "label", "type": "multiclass_classification"}},
    )
    data.update(overrides)
    return data


def test_split_column_alone_is_valid():
    cfg = DataConfig(**_base(split_column="split"))
    assert cfg.split_column == "split"
    assert cfg.split_from_filename is None


def test_split_from_filename_alone_is_valid():
    cfg = DataConfig(**_base(split_from_filename={"train": "train-*", "val": "validation-*"}))
    assert cfg.split_column is None
    assert cfg.split_from_filename == {"train": "train-*", "val": "validation-*"}


def test_both_split_sources_rejected():
    with pytest.raises(ValidationError, match="exactly one of data.split_column"):
        DataConfig(**_base(split_column="split", split_from_filename={"train": "train-*"}))


def test_no_split_source_rejected():
    with pytest.raises(ValidationError, match="exactly one of data.split_column"):
        DataConfig(**_base())


def test_non_vocabulary_split_name_rejected():
    # "validation" is not a canonical split name; "val" is.
    with pytest.raises(ValidationError):
        DataConfig(**_base(split_from_filename={"validation": "validation-*"}))
