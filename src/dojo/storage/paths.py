"""Filesystem-safe path components for artifact paths.

Run directories, figure subdirectories, and cache entries are named from
config-supplied values (experiment name, ``run_id``, head names). The legal
character set for a path component is filesystem-dependent, so sanitization is
too: POSIX only forbids ``/`` and NUL, while Windows additionally forbids
``<>:"\\|?*`` and the C0 control characters, silently strips trailing dots and
spaces, and reserves the legacy DOS device names (``CON``, ``NUL``, ``COM1``…)
in *every* directory.

Sanitizing to the union of both would rename directories on Linux that have
worked for every run to date, so each platform is sanitized to its own rules:
identical configs produce identical paths on Linux before and after this
module, and produce *creatable* paths on Windows. Path components are therefore
not guaranteed byte-identical across platforms — ``*_hash`` provenance, which
must be, never derives from these strings (see ``data/identity.py``).
"""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath, PureWindowsPath


Flavor = str  # "posix" | "windows"

# Windows reserves the DOS device names in any directory, with or without an
# extension ("NUL.txt" is as unusable as "NUL").
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)

_POSIX_FORBIDDEN_RE = re.compile(r"[/\x00]")
_WINDOWS_FORBIDDEN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_REPLACEMENT = "_"


def is_absolute_uri(uri: str) -> bool:
    """Whether ``uri`` already addresses a location on its own.

    True for schemed URIs and for absolute paths in *either* platform's syntax,
    so a manifest written on Windows (``C:\\data\\a.png``) is not mistaken for a
    manifest-root-relative URI when read on Linux, or vice versa.
    """

    if "://" in uri:
        return True
    return PurePosixPath(uri).is_absolute() or PureWindowsPath(uri).is_absolute()


def is_windows() -> bool:
    """Whether the running platform has Windows filesystem semantics.

    The single place the platform is detected, so tests (and any future
    ``--target-platform`` preview) can flip both sanitization and the
    platform-specific config warnings with one patch.
    """

    return os.name == "nt"


def default_flavor() -> Flavor:
    """The sanitization flavor for the running platform."""

    return "windows" if is_windows() else "posix"


def sanitize_path_component(name: str, *, flavor: Flavor | None = None) -> str:
    """Return ``name`` made usable as a single path component.

    ``flavor`` defaults to the running platform; pass it explicitly to exercise
    the other platform's rules (tests, cross-platform path previews).
    """

    flavor = flavor or default_flavor()
    if flavor == "windows":
        cleaned = _WINDOWS_FORBIDDEN_RE.sub(_REPLACEMENT, name)
        # Windows drops trailing dots/spaces when creating the entry, so a name
        # ending in one resolves to a *different* directory than requested.
        cleaned = cleaned.rstrip(". ")
        stem = cleaned.split(".", 1)[0]
        if stem.upper() in _WINDOWS_RESERVED_STEMS:
            cleaned = f"{_REPLACEMENT}{cleaned}"
    else:
        cleaned = _POSIX_FORBIDDEN_RE.sub(_REPLACEMENT, name)

    # "." and ".." are directory references on both platforms, never names.
    if cleaned in {"", ".", ".."}:
        return _REPLACEMENT
    return cleaned


def sanitize_relative_path(value: str, *, flavor: Flavor | None = None) -> str:
    """Sanitize each ``/``-separated segment of ``value``, keeping the joins.

    Used where a config value may intentionally nest (``experiment.name:
    p2/08_multihead``) but must not smuggle in characters the filesystem
    rejects, an absolute-path escape, or a ``..`` traversal.
    """

    flavor = flavor or default_flavor()
    segments = [
        sanitize_path_component(segment, flavor=flavor)
        for segment in value.replace("\\", "/").split("/")
        if segment != ""
    ]
    if not segments:
        return _REPLACEMENT
    return "/".join(segments)
