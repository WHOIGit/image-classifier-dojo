"""Tiny synthetic ``parquet_images`` datasets for tests that need split-by-file.

The committed ``plankton-toyset`` fixture uses ``split_column``; these helpers write
small throwaway Parquet files (inlined PNG bytes, no split column) so the
``split_from_filename`` path can be exercised without committing more binaries.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _png_bytes(seed: int, size: tuple[int, int] = (12, 8)) -> bytes:
    img = Image.new("RGB", size, color=(seed % 256, (2 * seed) % 256, (3 * seed) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_parquet_images_file(path: Path, *, labels: list[int], start_id: int = 0) -> None:
    """Write one parquet_images file with an ``image`` struct and a ``label`` col."""

    n = len(labels)
    images = [{"bytes": _png_bytes(start_id + i), "path": None} for i in range(n)]
    table = pa.table(
        {
            "image": pa.array(
                images,
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "label": pa.array(labels, pa.int64()),
            "sample_id": pa.array([f"s{start_id + i}" for i in range(n)], pa.string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
