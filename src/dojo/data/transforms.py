"""Image transform builder for supervised Dojo image pipelines.

Builds a ``PIL.Image -> torch.Tensor`` callable from resolved transform steps.
Implemented steps include ``letterbox``, ``foreground_crop``,
``aspect_bucket``, ``grayscale``, ``normalize``, and the stochastic
``train_only`` augmentations ``rotate`` (multiples of 90), ``horizontal_flip``,
and ``vertical_flip``.

Value-range convention follows ``05-models-training-and-heads.md``:
**decode → scale to ``[0, 1]`` (by ``input_bit_depth`` divisor) → normalize**.
``image_mode`` sets the channel contract before the pipeline runs.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image

from dojo.config_schemas.root import (
    AspectBucketConfig,
    AspectBucketStep,
    ForegroundCropStep,
    GrayscaleStep,
    HorizontalFlipStep,
    LetterboxStep,
    NormalizeStep,
    RotateStep,
    TransformStep,
    VerticalFlipStep,
)

ImageTransform = Callable[[Image.Image], torch.Tensor]

_BIT_DEPTH_DIVISOR = {8: 255.0, 12: 4095.0, 16: 65535.0}


def _decode_to_unit_chw(
    img: Image.Image,
    image_mode: str,
    input_bit_depth: str | int,
) -> torch.Tensor:
    """Decode a PIL image to a float CHW tensor scaled to ``[0, 1]``."""

    if image_mode == "rgb":
        img = img.convert("RGB")
    else:  # grayscale | grayscale_repeat3 decode a single luminance channel
        img = img.convert("L")

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]

    if input_bit_depth == "auto":
        divisor = 65535.0 if arr.dtype == np.uint16 else 255.0
    else:
        divisor = _BIT_DEPTH_DIVISOR[int(input_bit_depth)]

    tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1) / divisor
    if image_mode == "grayscale_repeat3":
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def _letterbox(tensor: torch.Tensor, canvas_size: tuple[int, int]) -> torch.Tensor:
    """Resize preserving aspect ratio, then pad to a centered ``canvas_size``.

    ``canvas_size`` is ``(height, width)``.
    """

    target_h, target_w = canvas_size
    _, h, w = tensor.shape
    scale = min(target_h / h, target_w / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = F.resize(tensor, [new_h, new_w], antialias=True)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    left = pad_w // 2
    # torchvision pad order is [left, top, right, bottom].
    return F.pad(resized, [left, top, pad_w - left, pad_h - top])


def _foreground_crop(
    tensor: torch.Tensor,
    *,
    threshold: float,
    padding_px: int,
) -> torch.Tensor:
    mask = tensor.amax(dim=0) > threshold
    if not bool(mask.any()):
        return tensor
    rows = torch.nonzero(mask.any(dim=1), as_tuple=False).flatten()
    cols = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten()
    top = max(0, int(rows[0]) - padding_px)
    bottom = min(tensor.shape[1], int(rows[-1]) + padding_px + 1)
    left = max(0, int(cols[0]) - padding_px)
    right = min(tensor.shape[2], int(cols[-1]) + padding_px + 1)
    return tensor[:, top:bottom, left:right]


def choose_aspect_bucket(
    *,
    width: int,
    height: int,
    buckets: Sequence[AspectBucketConfig],
) -> AspectBucketConfig:
    aspect = width / height
    long_side = max(width, height)
    for bucket in buckets:
        if bucket.min_aspect is not None and aspect < bucket.min_aspect:
            continue
        if bucket.max_aspect is not None and aspect > bucket.max_aspect:
            continue
        if (
            bucket.min_native_long_side is not None
            and long_side < bucket.min_native_long_side
        ):
            continue
        if (
            bucket.max_native_long_side is not None
            and long_side > bucket.max_native_long_side
        ):
            continue
        return bucket
    raise ValueError(
        f"no aspect_bucket bucket matches width={width}, height={height}, aspect={aspect:.3g}"
    )


def find_aspect_bucket_step(steps: Sequence[TransformStep]) -> AspectBucketStep | None:
    for step in steps:
        if isinstance(step, AspectBucketStep) and step.enabled:
            return step
    return None


def _grayscale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == 1:
        return tensor
    lum = (
        0.2989 * tensor[0:1]
        + 0.5870 * tensor[1:2]
        + 0.1140 * tensor[2:3]
    )
    return lum.repeat(tensor.shape[0], 1, 1)


def _normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


class _CompiledImageTransform:
    """A picklable ``PIL.Image -> CHW float tensor`` transform.

    A module-level callable (not a closure) so it survives pickling to
    ``DataLoader`` worker processes, which Python 3.14 spawns via ``forkserver``
    on non-mac POSIX and therefore requires picklable arguments.
    """

    def __init__(
        self,
        active: Sequence[TransformStep],
        image_mode: str,
        input_bit_depth: str | int,
    ) -> None:
        self._active = list(active)
        self._image_mode = image_mode
        self._input_bit_depth = input_bit_depth

    def __call__(self, img: Image.Image) -> torch.Tensor:
        native_w, native_h = img.size
        tensor = _decode_to_unit_chw(img, self._image_mode, self._input_bit_depth)
        for step in self._active:
            if isinstance(step, ForegroundCropStep):
                tensor = _foreground_crop(
                    tensor,
                    threshold=step.threshold,
                    padding_px=step.padding_px,
                )
            elif isinstance(step, LetterboxStep):
                tensor = _letterbox(tensor, step.canvas_size)
            elif isinstance(step, AspectBucketStep):
                bucket = choose_aspect_bucket(
                    width=native_w,
                    height=native_h,
                    buckets=step.buckets,
                )
                tensor = _letterbox(tensor, bucket.canvas_size)
            elif isinstance(step, GrayscaleStep):
                if random.random() < step.p:
                    tensor = _grayscale(tensor)
            elif isinstance(step, RotateStep):
                if random.random() < step.p:
                    tensor = torch.rot90(tensor, random.randint(1, 3), dims=(1, 2))
            elif isinstance(step, HorizontalFlipStep):
                if random.random() < step.p:
                    tensor = torch.flip(tensor, dims=[2])
            elif isinstance(step, VerticalFlipStep):
                if random.random() < step.p:
                    tensor = torch.flip(tensor, dims=[1])
            elif isinstance(step, NormalizeStep):
                tensor = _normalize(tensor, step.mean, step.std)
        return tensor.contiguous()


def build_image_transform(
    steps: Sequence[TransformStep],
    *,
    image_mode: str,
    input_bit_depth: str | int,
) -> ImageTransform:
    """Compile ``steps`` into a ``PIL.Image -> CHW float tensor`` callable.

    Pass ``transforms.pipeline`` for train stages and the resolved
    ``transforms.inference_pipeline`` for non-train stages; ``train_only``
    selection is the caller's choice of list, not re-decided here. Disabled
    steps are skipped.
    """

    active = [step for step in steps if step.enabled]
    return _CompiledImageTransform(active, image_mode, input_bit_depth)
