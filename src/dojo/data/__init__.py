"""Dataset backends, the shared sample contract, transforms, and dataset identity."""

from torch.utils.data import DataLoader

from dojo.data.contract import DecodedSample, SampleBatch, collate_samples
from dojo.data.dataset import ParquetImagesDataset
from dojo.data.identity import compute_dataset_hash
from dojo.data.parquet_images import (
    DataBundle,
    DatasetConfigError,
    build_datasets,
)
from dojo.data.transforms import build_image_transform

__all__ = [
    "DecodedSample",
    "SampleBatch",
    "collate_samples",
    "ParquetImagesDataset",
    "compute_dataset_hash",
    "DataBundle",
    "DatasetConfigError",
    "build_datasets",
    "build_image_transform",
    "build_dataloader",
]


def build_dataloader(
    dataset: ParquetImagesDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """A DataLoader over a P1 dataset using the shared sample collation.

    ``drop_last`` drops a trailing partial batch; the training loop sets it so a
    size-1 final batch can't crash BatchNorm in train mode.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_samples,
    )
