"""``dataset_hash`` for the P1 ``parquet_images`` backend.

Cheap, deterministic dataset identity: it never reads image pixels. The basis
is the manifest file identities (**relative path + byte size** — deliberately
not mtime, which is not reproducible across copies/machines and would break the
two-run determinism guarantee) plus the backend type and the declared column
bindings. Embedded image bytes belong to ``dataset_content_hash`` (deferred,
P2.2), never to this hash. See ``04-data-and-storage.md``.
"""

from __future__ import annotations

from typing import Any

from dojo.config_schemas.hashing import sha256_json
from dojo.config_schemas.root import DataConfig

# P1 records this provenance label; doc 04's enum
# (manifest_content / uri_etag / uri_only) is reconciled in P2.2 dataset-hash
# hardening. "uri_size" = path + size, stronger than uri_only, mtime excluded.
DATASET_HASH_PROVENANCE = "uri_size"


def _normalize_separators(path: str) -> str:
    """POSIX-style separators, so manifest identity is platform-independent."""

    return path.replace("\\", "/")


def _binding_source(cfg: DataConfig) -> dict[str, Any]:
    return {
        "backend": cfg.backend,
        "sample_id_column": cfg.sample_id_column,
        "split_column": cfg.split_column,
        "split_from_filename": cfg.split_from_filename,
        "image": cfg.images.model_dump() if cfg.images is not None else None,
        "image_uri_column": cfg.image_uri_column,
        "targets": {
            name: {
                "label_index_column": t.label_index_column,
                "label_name_column": t.label_name_column,
                "type": t.type,
            }
            for name, t in sorted(cfg.targets.items())
        },
    }


def compute_dataset_hash(
    cfg: DataConfig,
    files: list[tuple[str, int]],
) -> tuple[str, str]:
    """Return ``(dataset_hash, dataset_hash_provenance)``.

    ``files`` is the list of ``(relative_path, size_bytes)`` for the manifest
    Parquet files, sorted by path. Identical content in the same layout hashes
    identically regardless of absolute location, modification time, or the
    platform the manifest was discovered on: separators are normalized to
    ``/`` here so a Windows run and a Linux run over the same dataset agree.
    """

    source = {
        "version": "1",
        "bindings": _binding_source(cfg),
        # Normalize before sorting: "\\" and "/" sort differently against the
        # other characters legal in a filename, so ordering must be settled on
        # the normalized form to be platform-independent too.
        "files": [
            {"path": path, "size": size}
            for path, size in sorted(
                (_normalize_separators(path), size) for path, size in files
            )
        ],
    }
    return sha256_json(source), DATASET_HASH_PROVENANCE
