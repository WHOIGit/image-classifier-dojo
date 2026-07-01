"""Checkpoint identity helpers.

``checkpoint_hash`` is the SHA-256 of the ``.ckpt`` file bytes. The filename
convention embeds the first 6 hex characters of that hash:
``{stem}.{first6_hex}.{ext}``.
"""

from __future__ import annotations

import hashlib
import os

_CHUNK = 1 << 20


def checkpoint_hash(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def checkpoint_filename(stem: str, hash_str: str, ext: str = "ckpt") -> str:
    first6 = hash_str.split(":")[-1][:6]
    return f"{stem}.{first6}.{ext}"
