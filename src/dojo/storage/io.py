"""Dojo storage IO interface.

A thin, URI-oriented surface over ``amplify-storage-utils`` so dataset,
results, and checkpoint code never import the amplify storage APIs directly
(see ``04-data-and-storage.md``: "Dataset / training / export code should not
leak amplify-storage-utils APIs directly — go through the Dojo storage
interface").

P1 is local-only. The local backend is implemented over amplify's
``FilesystemStore``; ``s3://`` (and any other non-local scheme) is a lazy seam
that raises a clear :class:`StorageError` until an object-store backend is
wired. ``boto3`` is intentionally absent from the P1 dependency set, so an
``s3://`` URI must fail loudly rather than silently.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from storage.fs import FilesystemStore

from dojo.config_schemas.root import StorageConfig


# Schemes the Dojo storage interface treats as local filesystem.
_LOCAL_SCHEMES = ("", "file")


class StorageError(RuntimeError):
    """Raised for unsupported URIs or storage backends."""


def _scheme(uri: str) -> str:
    # A bare Windows-free POSIX path has no scheme; urlparse leaves it empty.
    return urlparse(uri).scheme


class Storage(ABC):
    """URI-oriented byte storage. Implementations back a set of URI schemes."""

    @abstractmethod
    def read_bytes(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri`` (raises ``FileNotFoundError``)."""

    @abstractmethod
    def write_bytes(self, uri: str, data: bytes) -> None:
        """Write ``data`` to ``uri``, creating parent directories as needed."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Return whether ``uri`` currently resolves to stored bytes."""

    @abstractmethod
    def localize(self, uri: str) -> Path:
        """Return a local filesystem path for ``uri``.

        For local backends this is the path itself; object-store backends
        download to the local cache and return the cached path.
        """


class LocalStorage(Storage):
    """Local-filesystem storage backed by amplify ``FilesystemStore``.

    ``FilesystemStore`` joins its key onto ``root_path`` with ``os.path.join``,
    which returns the key unchanged when the key is absolute. Rooting the store
    at ``""`` and always passing absolute keys therefore makes it a faithful
    passthrough to the local filesystem while still exercising the amplify
    store contract.
    """

    def __init__(self) -> None:
        self._store = FilesystemStore("")

    @staticmethod
    def _key(uri: str) -> str:
        scheme = _scheme(uri)
        if scheme not in _LOCAL_SCHEMES:
            raise StorageError(
                f"{scheme}:// storage is not enabled in this build. P1 is "
                "local-only; object-store backends (e.g. s3) are deferred."
            )
        path = urlparse(uri).path if scheme == "file" else uri
        return os.fspath(Path(path).expanduser().resolve())

    def read_bytes(self, uri: str) -> bytes:
        try:
            return self._store.get(self._key(uri))
        except KeyError:
            raise FileNotFoundError(uri) from None

    def write_bytes(self, uri: str, data: bytes) -> None:
        self._store.put(self._key(uri), data)

    def exists(self, uri: str) -> bool:
        return self._store.exists(self._key(uri))

    def localize(self, uri: str) -> Path:
        return Path(self._key(uri))


def get_storage(config: StorageConfig | None = None) -> Storage:
    """Return the Dojo storage backend for ``config``.

    P1 always returns :class:`LocalStorage`. The signature takes the resolved
    ``storage`` config block so the s3 / cache-aware backend can slot in here
    without changing call sites.
    """

    return LocalStorage()
