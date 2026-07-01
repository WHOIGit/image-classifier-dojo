"""Torch dataset over an in-memory ``parquet_images`` split table."""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import torch
from PIL import Image
from torch.utils.data import Dataset

from dojo.config_schemas.root import DataConfig
from dojo.data.contract import DecodedSample
from dojo.data.transforms import ImageTransform


class ParquetImagesDataset(Dataset[DecodedSample]):
    """One split's rows, decoding inlined image bytes on access.

    Columns are pre-extracted to Python lists at construction so ``__getitem__``
    only decodes + transforms the image. ``split`` is constant for the dataset
    (the backend groups rows by split before constructing it).
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        cfg: DataConfig,
        target_name: str,
        split: str,
        transform: ImageTransform,
        class_index_by_name: dict[str, int] | None = None,
    ) -> None:
        self._split = split
        self._transform = transform
        self._sample_ids = table.column(cfg.sample_id_column).to_pylist()

        image = cfg.images
        assert image is not None  # guaranteed by parquet_images backend
        # table.column(...) is a ChunkedArray of struct type; pull the bytes
        # sub-field with struct_field (ChunkedArray has no .field()).
        struct = table.column(image.column)
        self._image_bytes = pc.struct_field(struct, image.bytes_field).to_pylist()

        # Per-sample integer label index: read the index column directly, or map
        # the name column through the backend-assigned class_index_by_name.
        target = cfg.targets[target_name]
        if target.label_index_column is not None:
            self._targets = [int(v) for v in table.column(target.label_index_column).to_pylist()]
        else:
            assert class_index_by_name is not None
            names = table.column(target.label_name_column).to_pylist()
            self._targets = [class_index_by_name[str(n)] for n in names]

        self._source_extra_cols = list(cfg.source_extra_columns)
        self._source_extra = {
            col: table.column(col).to_pylist() for col in self._source_extra_cols
        }

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> DecodedSample:
        img = Image.open(io.BytesIO(self._image_bytes[index]))
        img.load()
        native_w, native_h = img.size

        tensor = self._transform(img)
        _, resize_h, resize_w = tensor.shape

        source_extra: dict[str, Any] | None = None
        if self._source_extra_cols:
            source_extra = {
                col: self._source_extra[col][index] for col in self._source_extra_cols
            }

        return DecodedSample(
            image=tensor,
            target=int(self._targets[index]),
            sample_id=str(self._sample_ids[index]),
            split=self._split,
            native_width_px=int(native_w),
            native_height_px=int(native_h),
            resize_width_px=int(resize_w),
            resize_height_px=int(resize_h),
            source_extra=source_extra,
        )
