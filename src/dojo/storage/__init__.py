"""Storage backends and the URI-based storage interface.

Config path / template resolution moved to :mod:`dojo.config_loader.resolver`.
"""

from dojo.storage.io import (
    LocalStorage,
    Storage,
    StorageError,
    get_storage,
)
from dojo.storage.paths import (
    is_absolute_uri,
    sanitize_path_component,
    sanitize_relative_path,
)

__all__ = [
    "Storage",
    "LocalStorage",
    "StorageError",
    "get_storage",
    "is_absolute_uri",
    "sanitize_path_component",
    "sanitize_relative_path",
]
