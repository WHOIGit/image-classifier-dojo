"""Shared sample contract and batch collation for P1 datasets.

One decoded sample carries the image tensor, its supervised target index, and
the provenance the result writer needs (``sample_id``, ``split``, native /
resize dimensions, source-extra passthrough). See ``04-data-and-storage.md``
("Shared sample contract") and ``06-results-artifacts-and-metadata.md``
(``sample_metadata`` columns).
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


class DecodedSample(TypedDict):
    image: torch.Tensor
    target: int
    sample_id: str
    uri: str | None
    split: str
    native_width_px: int
    native_height_px: int
    resize_width_px: int
    resize_height_px: int
    aspect_bucket: str | None
    source_extra: dict[str, Any] | None


class SampleBatch(TypedDict):
    image: torch.Tensor          # (B, C, H, W)
    target: torch.Tensor         # (B,) int64
    sample_id: list[str]
    uri: list[str | None]
    split: list[str]
    native_width_px: list[int]
    native_height_px: list[int]
    resize_width_px: list[int]
    resize_height_px: list[int]
    aspect_bucket: list[str | None]
    source_extra: list[dict[str, Any] | None]


def collate_samples(batch: list[DecodedSample]) -> SampleBatch:
    """Stack image tensors and targets; keep provenance as per-sample lists."""

    return SampleBatch(
        image=torch.stack([s["image"] for s in batch]),
        target=torch.tensor([s["target"] for s in batch], dtype=torch.int64),
        sample_id=[s["sample_id"] for s in batch],
        uri=[s["uri"] for s in batch],
        split=[s["split"] for s in batch],
        native_width_px=[s["native_width_px"] for s in batch],
        native_height_px=[s["native_height_px"] for s in batch],
        resize_width_px=[s["resize_width_px"] for s in batch],
        resize_height_px=[s["resize_height_px"] for s in batch],
        aspect_bucket=[s["aspect_bucket"] for s in batch],
        source_extra=[s["source_extra"] for s in batch],
    )
