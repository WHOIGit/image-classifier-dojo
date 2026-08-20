"""Platform-appropriate sanitization of config-derived path components."""

from __future__ import annotations

import pytest

from dojo.storage.paths import (
    is_absolute_uri,
    sanitize_path_component,
    sanitize_relative_path,
)


def test_posix_flavor_only_strips_separators():
    # Everything except "/" and NUL is a legal POSIX filename character, so
    # Linux run directories keep the names they have always had.
    assert sanitize_path_component("head:A1?", flavor="posix") == "head:A1?"
    assert sanitize_path_component("a/b", flavor="posix") == "a_b"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("head:A1", "head_A1"),
        ('a"b<c>d|e?f*g', "a_b_c_d_e_f_g"),
        ("back\\slash", "back_slash"),
        ("trailing.", "trailing"),
        ("trailing space ", "trailing space"),
        ("bell\x07", "bell_"),
    ],
)
def test_windows_flavor_replaces_illegal_characters(raw, expected):
    assert sanitize_path_component(raw, flavor="windows") == expected


@pytest.mark.parametrize("reserved", ["CON", "nul", "COM1", "LPT9", "NUL.txt"])
def test_windows_reserved_device_names_are_escaped(reserved):
    sanitized = sanitize_path_component(reserved, flavor="windows")
    assert sanitized == f"_{reserved}"


def test_empty_and_dot_components_become_placeholders():
    for flavor in ("posix", "windows"):
        assert sanitize_path_component("", flavor=flavor) == "_"
        assert sanitize_path_component(".", flavor=flavor) == "_"
        assert sanitize_path_component("..", flavor=flavor) == "_"


def test_relative_path_keeps_intentional_nesting():
    assert (
        sanitize_relative_path("p2/08_multihead", flavor="posix") == "p2/08_multihead"
    )
    assert sanitize_relative_path("p2/08:multi", flavor="windows") == "p2/08_multi"


def test_relative_path_refuses_to_escape_its_parent():
    assert sanitize_relative_path("/etc/passwd", flavor="posix") == "etc/passwd"
    assert sanitize_relative_path("../../secrets", flavor="posix") == "_/_/secrets"
    assert sanitize_relative_path("", flavor="posix") == "_"


@pytest.mark.parametrize(
    "uri",
    ["s3://bucket/key", "file:///data/a.png", "/data/a.png", "C:\\data\\a.png", "C:/data/a.png"],
)
def test_absolute_uris_are_recognized_from_either_platform(uri):
    assert is_absolute_uri(uri) is True


@pytest.mark.parametrize("uri", ["images/a.png", "./images/a.png", "a.png"])
def test_relative_uris_are_not_absolute(uri):
    assert is_absolute_uri(uri) is False
