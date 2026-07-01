"""Storage backends and the URI-based storage interface.

Config path / template resolution moved to :mod:`dojo.config_loader.resolver`.
"""

from dojo.storage.io import (
    LocalStorage,
    Storage,
    StorageError,
    get_storage,
)

__all__ = ["Storage", "LocalStorage", "StorageError", "get_storage"]
