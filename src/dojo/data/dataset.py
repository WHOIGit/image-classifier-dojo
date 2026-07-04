"""Torch datasets over lightweight manifest split tables."""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from dojo.config_schemas.root import AspectBucketStep, DataConfig
from dojo.data.contract import MISSING_TARGET_INDEX, DecodedSample
from dojo.data.parquet_rows import (
    ROW_FILE_PATH,
    ROW_GROUP_INDEX,
    ROW_IMAGE_PATH,
    ROW_IN_GROUP_INDEX,
    ROW_MATERIALIZED_IMAGE_PATH,
    LazyParquetImageReader,
)
from dojo.data.transforms import ImageTransform, choose_aspect_bucket
from dojo.storage import Storage


class ParquetImagesDataset(Dataset[DecodedSample]):
    """One split's rows, decoding inlined image bytes on access.

    Only lightweight metadata and row references are pre-extracted to Python
    lists. Embedded image bytes stay in the source Parquet file and are read in
    ``__getitem__`` from the referenced row group.
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        cfg: DataConfig,
        target_names: tuple[str, ...],
        split: str,
        transform: ImageTransform,
        aspect_bucket_step: AspectBucketStep | None = None,
        class_index_by_name: dict[str, dict[str, int] | None] | None = None,
    ) -> None:
        self._split = split
        self._transform = transform
        self._aspect_bucket_step = aspect_bucket_step
        self._target_names = target_names
        self._primary_target_name = target_names[0]
        self._sample_ids = table.column(cfg.sample_id_column).to_pylist()

        image = cfg.images
        assert image is not None  # guaranteed by parquet_images backend
        self._image_column = image.column
        self._bytes_field = image.bytes_field
        self._file_paths = [str(v) for v in table.column(ROW_FILE_PATH).to_pylist()]
        self._row_groups = [int(v) for v in table.column(ROW_GROUP_INDEX).to_pylist()]
        self._row_in_groups = [
            int(v) for v in table.column(ROW_IN_GROUP_INDEX).to_pylist()
        ]
        self._materialized_paths: list[str] | None = None
        if ROW_MATERIALIZED_IMAGE_PATH in table.column_names:
            self._materialized_paths = [
                str(value) for value in table.column(ROW_MATERIALIZED_IMAGE_PATH).to_pylist()
            ]
        self._image_reader: LazyParquetImageReader | None = None
        if image.path_field is None:
            self._uris: list[str | None] = [None] * len(self._sample_ids)
        else:
            self._uris = [
                None if value is None else str(value)
                for value in table.column(ROW_IMAGE_PATH).to_pylist()
            ]

        self._targets_by_name: dict[str, list[int]] = {}
        for target_name in target_names:
            target = cfg.targets[target_name]
            if target.label_index_column is not None:
                self._targets_by_name[target_name] = _index_targets(
                    table.column(target.label_index_column).to_pylist(),
                    target_name=target_name,
                    missing_policy=target.missing_policy,
                )
            else:
                assert class_index_by_name is not None
                name_mapping = class_index_by_name[target_name]
                assert name_mapping is not None
                self._targets_by_name[target_name] = _name_targets(
                    table.column(target.label_name_column).to_pylist(),
                    name_mapping=name_mapping,
                    target_name=target_name,
                    missing_policy=target.missing_policy,
                )

        self._source_extra_cols = list(cfg.source_extra_columns)
        self._source_extra = {
            col: table.column(col).to_pylist() for col in self._source_extra_cols
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_image_reader"] = None
        return state

    def __len__(self) -> int:
        return len(self._sample_ids)

    def target_for_index(self, index: int, target_name: str | None = None) -> int:
        target_name = target_name or self._primary_target_name
        return int(self._targets_by_name[target_name][index])

    def has_aspect_buckets(self) -> bool:
        return self._aspect_bucket_step is not None

    def _read_image_bytes(self, index: int) -> bytes:
        if self._image_reader is None:
            self._image_reader = LazyParquetImageReader(
                image_column=self._image_column,
                bytes_field=self._bytes_field,
            )
        return self._image_reader.read_bytes(
            file_path=self._file_paths[index],
            row_group=self._row_groups[index],
            row_in_group=self._row_in_groups[index],
        )

    def __getitem__(self, index: int) -> DecodedSample:
        if self._materialized_paths is None:
            img = Image.open(io.BytesIO(self._read_image_bytes(index)))
        else:
            img = Image.open(self._materialized_paths[index])
        img.load()
        native_w, native_h = img.size
        aspect_bucket = self.aspect_bucket_for_index(index, native_size=(native_w, native_h))

        tensor = self._transform(img)
        _, resize_h, resize_w = tensor.shape

        source_extra: dict[str, Any] | None = None
        if self._source_extra_cols:
            source_extra = {
                col: self._source_extra[col][index] for col in self._source_extra_cols
            }

        return DecodedSample(
            image=tensor,
            target=self.target_for_index(index),
            targets={
                target_name: int(values[index])
                for target_name, values in self._targets_by_name.items()
            },
            sample_id=str(self._sample_ids[index]),
            uri=self._uris[index],
            split=self._split,
            native_width_px=int(native_w),
            native_height_px=int(native_h),
            resize_width_px=int(resize_w),
            resize_height_px=int(resize_h),
            aspect_bucket=aspect_bucket,
            source_extra=source_extra,
        )

    def aspect_bucket_for_index(
        self,
        index: int,
        *,
        native_size: tuple[int, int] | None = None,
    ) -> str | None:
        if self._aspect_bucket_step is None:
            return None
        if native_size is None:
            if self._materialized_paths is None:
                img = Image.open(io.BytesIO(self._read_image_bytes(index)))
            else:
                img = Image.open(self._materialized_paths[index])
            native_size = img.size
        bucket = choose_aspect_bucket(
            width=int(native_size[0]),
            height=int(native_size[1]),
            buckets=self._aspect_bucket_step.buckets,
        )
        return bucket.name


class ManifestImagesDataset(Dataset[DecodedSample]):
    """One split's rows, reading image bytes from ``image_uri_column`` on demand."""

    def __init__(
        self,
        table: pa.Table,
        *,
        cfg: DataConfig,
        target_names: tuple[str, ...],
        split: str,
        transform: ImageTransform,
        storage: Storage,
        manifest_root: str,
        aspect_bucket_step: AspectBucketStep | None = None,
        class_index_by_name: dict[str, dict[str, int] | None] | None = None,
    ) -> None:
        self._split = split
        self._transform = transform
        self._storage = storage
        self._manifest_root = manifest_root
        self._aspect_bucket_step = aspect_bucket_step
        self._target_names = target_names
        self._primary_target_name = target_names[0]
        self._sample_ids = table.column(cfg.sample_id_column).to_pylist()

        assert cfg.image_uri_column is not None
        self._uris = [
            None if value is None else str(value)
            for value in table.column(cfg.image_uri_column).to_pylist()
        ]

        self._targets_by_name: dict[str, list[int]] = {}
        for target_name in target_names:
            target = cfg.targets[target_name]
            if target.label_index_column is not None:
                self._targets_by_name[target_name] = _index_targets(
                    table.column(target.label_index_column).to_pylist(),
                    target_name=target_name,
                    missing_policy=target.missing_policy,
                )
            else:
                assert class_index_by_name is not None
                name_mapping = class_index_by_name[target_name]
                assert name_mapping is not None
                self._targets_by_name[target_name] = _name_targets(
                    table.column(target.label_name_column).to_pylist(),
                    name_mapping=name_mapping,
                    target_name=target_name,
                    missing_policy=target.missing_policy,
                )

        self._source_extra_cols = list(cfg.source_extra_columns)
        self._source_extra = {
            col: table.column(col).to_pylist() for col in self._source_extra_cols
        }

    def __len__(self) -> int:
        return len(self._sample_ids)

    def target_for_index(self, index: int, target_name: str | None = None) -> int:
        target_name = target_name or self._primary_target_name
        return int(self._targets_by_name[target_name][index])

    def has_aspect_buckets(self) -> bool:
        return self._aspect_bucket_step is not None

    def _resolve_uri(self, uri: str) -> str:
        if "://" in uri or uri.startswith("/"):
            return uri
        return str((self._storage.localize(self._manifest_root) / uri).resolve())

    def __getitem__(self, index: int) -> DecodedSample:
        uri = self._uris[index]
        if uri is None:
            raise ValueError(f"missing image URI for sample {self._sample_ids[index]!r}")
        img = Image.open(io.BytesIO(self._storage.read_bytes(self._resolve_uri(uri))))
        img.load()
        native_w, native_h = img.size
        aspect_bucket = self.aspect_bucket_for_index(index, native_size=(native_w, native_h))

        tensor = self._transform(img)
        _, resize_h, resize_w = tensor.shape

        source_extra: dict[str, Any] | None = None
        if self._source_extra_cols:
            source_extra = {
                col: self._source_extra[col][index] for col in self._source_extra_cols
            }

        return DecodedSample(
            image=tensor,
            target=self.target_for_index(index),
            targets={
                target_name: int(values[index])
                for target_name, values in self._targets_by_name.items()
            },
            sample_id=str(self._sample_ids[index]),
            uri=uri,
            split=self._split,
            native_width_px=int(native_w),
            native_height_px=int(native_h),
            resize_width_px=int(resize_w),
            resize_height_px=int(resize_h),
            aspect_bucket=aspect_bucket,
            source_extra=source_extra,
        )

    def aspect_bucket_for_index(
        self,
        index: int,
        *,
        native_size: tuple[int, int] | None = None,
    ) -> str | None:
        if self._aspect_bucket_step is None:
            return None
        uri = self._uris[index]
        if uri is None:
            return None
        if native_size is None:
            img = Image.open(io.BytesIO(self._storage.read_bytes(self._resolve_uri(uri))))
            native_size = img.size
        bucket = choose_aspect_bucket(
            width=int(native_size[0]),
            height=int(native_size[1]),
            buckets=self._aspect_bucket_step.buckets,
        )
        return bucket.name


def _missing_target_value(*, target_name: str, missing_policy: str) -> int:
    if missing_policy == "mask_objective":
        return MISSING_TARGET_INDEX
    raise ValueError(
        f"target {target_name!r} has missing labels but missing_policy is "
        f"{missing_policy!r}"
    )


def _index_targets(
    values: list[Any],
    *,
    target_name: str,
    missing_policy: str,
) -> list[int]:
    targets: list[int] = []
    for value in values:
        if value is None:
            targets.append(
                _missing_target_value(
                    target_name=target_name,
                    missing_policy=missing_policy,
                )
            )
        else:
            targets.append(int(value))
    return targets


def _name_targets(
    names: list[Any],
    *,
    name_mapping: dict[str, int],
    target_name: str,
    missing_policy: str,
) -> list[int]:
    targets: list[int] = []
    for name in names:
        if name is None:
            targets.append(
                _missing_target_value(
                    target_name=target_name,
                    missing_policy=missing_policy,
                )
            )
        else:
            targets.append(name_mapping[str(name)])
    return targets
