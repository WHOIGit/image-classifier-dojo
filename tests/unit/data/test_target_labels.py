"""Class-label resolution for data targets: label_index_column / label_name_column.

Covers the three ways a target names its classes against the committed
plankton-toyset fixture, whose rows carry a contiguous ``label`` index and a
readable ``classname`` (e.g. label 0 -> "bubble").
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dojo.config_schemas import RootConfig
from dojo.config_loader import resolve_runtime_and_paths
from dojo.data import build_datasets
from tests.fixtures.configs import toy_config_dict

# Distinct classnames in the toy fixture, in ASCII-sorted order (upper before lower).
TOY_NAMES_ALPHABETICAL = [
    "Copepod_nauplii",
    "Dinophysis_acuminata",
    "Skeletonema",
    "Trichodesmium",
    "bubble",
    "fiber",
]


def _bundle(target: dict):
    cfg_dict = toy_config_dict(canvas=(32, 32))
    cfg_dict["data"]["targets"]["species"] = {
        "type": "multiclass_classification",
        "missing_policy": "error",
        **target,
    }
    cfg = resolve_runtime_and_paths(RootConfig.model_validate(cfg_dict)).config
    return build_datasets(cfg)


def test_index_and_name_columns_pair_into_class_mapping():
    bundle = _bundle({"label_index_column": "label", "label_name_column": "classname"})
    # Names come from the data, keyed by the fixture's own contiguous indices.
    assert bundle.class_mapping[0] == "bubble"
    assert set(bundle.class_mapping.values()) == set(TOY_NAMES_ALPHABETICAL)
    assert sorted(bundle.class_mapping) == list(range(6))


def test_index_only_leaves_names_unresolved():
    bundle = _bundle({"label_index_column": "label"})
    # No name source -> class_mapping empty; run.py falls back to index strings.
    assert bundle.class_mapping == {}


def test_name_only_assigns_indices_alphabetically():
    bundle = _bundle({"label_name_column": "classname"})
    assert [bundle.class_mapping[i] for i in range(6)] == TOY_NAMES_ALPHABETICAL
    # The dataset must still yield the matching integer targets for those names.
    train = bundle.datasets["train"]
    class_index_by_name = {name: index for index, name in bundle.class_mapping.items()}
    sample = train[0]
    assert 0 <= sample["target"] < 6
    assert sample["target"] in class_index_by_name.values()


def test_target_requires_a_class_source():
    with pytest.raises(ValidationError, match="label_index_column"):
        RootConfig.model_validate(
            {
                **toy_config_dict(canvas=(32, 32)),
                "data": {
                    **toy_config_dict(canvas=(32, 32))["data"],
                    "targets": {
                        "species": {
                            "type": "multiclass_classification",
                            "missing_policy": "error",
                        }
                    },
                },
            }
        )
