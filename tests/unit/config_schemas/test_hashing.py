"""Unit tests for the config hashing canonicalizer.

Covers the canonicalizer's edge cases (key-order independence, float
normalization, NaN/Infinity, tuple/list equivalence) and the ``config_hash``
identity contract (which fields are excluded, what changes the hash).
"""

from __future__ import annotations

import math

from dojo.config_schemas.hashing import (
    _canonicalize,
    canonical_json_bytes,
    config_hash,
    config_hash_source,
    sha256_json,
)


# --- canonicalizer ---------------------------------------------------------


def test_key_order_does_not_affect_canonical_bytes():
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert canonical_json_bytes(a) == b'{"a":2,"b":1,"c":{"x":2,"y":1}}'


def test_tuple_and_list_are_equivalent():
    assert _canonicalize((1, 2, 3)) == _canonicalize([1, 2, 3]) == [1, 2, 3]
    assert canonical_json_bytes({"x": (1, 2)}) == canonical_json_bytes({"x": [1, 2]})


def test_floats_normalized_to_twelve_significant_figures():
    # 0.1 + 0.2 == 0.30000000000000004; %.12g collapses it to 0.3.
    assert _canonicalize(0.1 + 0.2) == 0.3
    assert canonical_json_bytes({"x": 0.1 + 0.2}) == canonical_json_bytes({"x": 0.3})


def test_non_finite_floats_become_strings():
    assert _canonicalize(math.nan) == "NaN"
    assert _canonicalize(math.inf) == "Infinity"
    assert _canonicalize(-math.inf) == "-Infinity"
    # And the result is therefore JSON-serializable (raw json can't dump inf).
    assert canonical_json_bytes({"x": math.inf}) == b'{"x":"Infinity"}'


def test_nested_keys_sorted_at_every_level():
    value = {"z": [{"b": 1, "a": 2}], "a": 1}
    assert canonical_json_bytes(value) == b'{"a":1,"z":[{"a":2,"b":1}]}'


def test_sha256_json_is_prefixed_deterministic_and_content_sensitive():
    digest = sha256_json({"a": 1})
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert sha256_json({"a": 1}) == sha256_json({"a": 1})
    # Reordered keys hash identically; changed content does not.
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


# --- config_hash identity contract -----------------------------------------


def _resolved_toy(**kwargs):
    from tests.fixtures.configs import toy_root_config

    return toy_root_config(**kwargs)


def test_config_hash_source_excludes_runtime_and_output_identity_fields():
    cfg = _resolved_toy()
    source = config_hash_source(cfg)

    assert "output_root" not in source
    assert "training_outputs" not in source
    assert "run_id" not in source["runtime"]
    assert "sweep_id" not in source["runtime"]
    assert "inference_pipeline" not in source["transforms"]

    # ...but the substantive config is still there.
    assert "model" in source
    assert "data" in source
    assert source["runtime"]["seed"] == cfg.runtime.seed
    assert source["transforms"]["pipeline"]


def test_config_hash_is_invariant_to_excluded_fields():
    # Two independent resolves of the same authored config: distinct random
    # run_ids and distinct output dirs, identical config_hash.
    a = _resolved_toy(output_root="/tmp/run_a")
    b = _resolved_toy(output_root="/tmp/run_b")

    assert a.runtime.run_id != b.runtime.run_id
    assert a.training_outputs.dir != b.training_outputs.dir
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_a_substantive_field():
    assert config_hash(_resolved_toy(max_epochs=1)) != config_hash(
        _resolved_toy(max_epochs=2)
    )


def test_config_hash_is_prefixed_and_deterministic():
    cfg = _resolved_toy()
    digest = config_hash(cfg)
    assert digest.startswith("sha256:")
    assert config_hash(cfg) == digest


def test_config_hash_source_does_not_mutate_config():
    cfg = _resolved_toy()
    run_id_before = cfg.runtime.run_id
    config_hash_source(cfg)
    # model_dump produces a copy; the live config is untouched.
    assert cfg.runtime.run_id == run_id_before
    assert cfg.transforms.inference_pipeline is not None
