"""Tiny synthetic ``parquet_images`` datasets for tests that need split-by-file.

The committed ``plankton-toyset`` fixture uses ``split_column``; these helpers write
small throwaway Parquet files (inlined PNG bytes, no split column) so the
``split_from_filename`` path can be exercised without committing more binaries.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from PIL import Image


def _png_bytes(seed: int, size: tuple[int, int] = (12, 8)) -> bytes:
    img = Image.new("RGB", size, color=(seed % 256, (2 * seed) % 256, (3 * seed) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_parquet_images_file(
    path: Path,
    *,
    labels: list[int],
    start_id: int = 0,
    sizes: list[tuple[int, int]] | None = None,
) -> None:
    """Write one parquet_images file with an ``image`` struct and a ``label`` col."""

    n = len(labels)
    sizes = sizes or [(12, 8)] * n
    images = [
        {"bytes": _png_bytes(start_id + i, size=sizes[i]), "path": None}
        for i in range(n)
    ]
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


def write_manifest_images_dataset(
    root: Path,
    *,
    labels: list[int],
    backend: str = "parquet_manifest",
    start_id: int = 0,
) -> Path:
    """Write external PNG images plus a CSV or Parquet manifest."""

    rows = []
    for offset, label in enumerate(labels):
        sample_id = f"s{start_id + offset}"
        image_name = f"{sample_id}.png"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "images" / image_name).write_bytes(_png_bytes(start_id + offset))
        rows.append(
            {
                "sample_id": sample_id,
                "image_uri": f"images/{image_name}",
                "split": "train" if offset < max(1, len(labels) - 1) else "val",
                "label": label,
                "classname": f"class-{label}",
            }
        )

    table = pa.table(
        {
            "sample_id": pa.array([row["sample_id"] for row in rows], pa.string()),
            "image_uri": pa.array([row["image_uri"] for row in rows], pa.string()),
            "split": pa.array([row["split"] for row in rows], pa.string()),
            "label": pa.array([row["label"] for row in rows], pa.int64()),
            "classname": pa.array([row["classname"] for row in rows], pa.string()),
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    if backend == "csv_manifest":
        path = root / "manifest.csv"
        pacsv.write_csv(table, path)
    else:
        path = root / "manifest.parquet"
        pq.write_table(table, path)
    return path
