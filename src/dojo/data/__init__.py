"""Dataset backends, the shared sample contract, transforms, and dataset identity."""

from __future__ import annotations

from torch.utils.data import DataLoader

from dojo.data.contract import DecodedSample, SampleBatch, collate_samples
from dojo.data.dataset import ManifestImagesDataset, ParquetImagesDataset
from dojo.data.identity import compute_dataset_hash
from dojo.data.parquet_images import (
    DataBundle,
    DatasetConfigError,
    build_datasets,
)
from dojo.data.preflight import PreflightIssue, run_dataset_preflight
from dojo.data.samplers import (
    AspectBucketBatchSampler,
    WeightedBatchSampler,
    sample_weights_for_dataset,
)
from dojo.data.transforms import build_image_transform

__all__ = [
    "DecodedSample",
    "SampleBatch",
    "collate_samples",
    "ParquetImagesDataset",
    "ManifestImagesDataset",
    "compute_dataset_hash",
    "DataBundle",
    "DatasetConfigError",
    "build_datasets",
    "PreflightIssue",
    "run_dataset_preflight",
    "AspectBucketBatchSampler",
    "WeightedBatchSampler",
    "sample_weights_for_dataset",
    "build_image_transform",
    "build_dataloader",
]


def build_dataloader(
    dataset: ParquetImagesDataset | ManifestImagesDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    batch_aspect_buckets: bool = False,
    seed: int = 0,
    sampler_type: str = "default",
    class_counts: dict[int, int] | None = None,
    class_weight_scheme: str = "inverse_frequency",
    class_weight_beta: float = 0.9999,
) -> DataLoader:
    """A DataLoader over a supervised dataset using the shared sample collation.

    ``drop_last`` drops a trailing partial batch; the training loop sets it so a
    size-1 final batch can't crash BatchNorm in train mode.
    """

    if batch_aspect_buckets and sampler_type == "default":
        sampler_type = "batch_aspect_buckets"

    weights_by_index: list[float] | None = None
    if sampler_type in {"class_balanced", "weighted"}:
        if class_counts is None:
            raise ValueError(f"{sampler_type} sampler requires train-split class_counts")
        weights_by_index = sample_weights_for_dataset(
            dataset,
            class_counts,
            scheme=class_weight_scheme,
            beta=class_weight_beta,
        )

    has_aspect_buckets = (
        bool(dataset.has_aspect_buckets())
        if hasattr(dataset, "has_aspect_buckets")
        else False
    )
    if sampler_type == "batch_aspect_buckets" or (
        has_aspect_buckets and weights_by_index is not None
    ):
        return DataLoader(
            dataset,
            batch_sampler=AspectBucketBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
                weights_by_index=weights_by_index,
            ),
            num_workers=num_workers,
            collate_fn=collate_samples,
        )

    if weights_by_index is not None:
        return DataLoader(
            dataset,
            batch_sampler=WeightedBatchSampler(
                dataset,
                batch_size=batch_size,
                weights_by_index=weights_by_index,
                drop_last=drop_last,
                seed=seed,
            ),
            num_workers=num_workers,
            collate_fn=collate_samples,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_samples,
    )
