"""Image transform builder: letterbox, normalize, value range, train_only steps."""

from __future__ import annotations

import torch
from PIL import Image

from dojo.config_schemas.root import TransformsConfig
from dojo.data.transforms import build_image_transform


def _transforms(pipeline) -> TransformsConfig:
    return TransformsConfig(image_mode="rgb", input_bit_depth=8, pipeline=pipeline)


def _letterbox_normalize(canvas=(32, 32)):
    return [
        {"name": "letterbox", "canvas_size": list(canvas)},
        {
            "name": "normalize",
            "mode": "fixed",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    ]


def test_letterbox_pads_to_canvas_and_scales_to_unit_range():
    cfg = _transforms(_letterbox_normalize((32, 32)))
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )
    # A wide white image: aspect preserved, padded to a square canvas.
    out = transform(Image.new("RGB", (20, 10), color=(255, 255, 255)))
    assert out.shape == (3, 32, 32)
    # mean=0/std=1 -> values are the [0,1] scaled pixels; white -> 1.0, pad -> 0.0.
    assert out.max().item() <= 1.0 + 1e-6
    assert out.min().item() >= 0.0 - 1e-6
    assert out.max().item() > 0.9  # the white content survived
    assert out.min().item() < 0.1  # zero padding present


def test_normalize_centers_values():
    pipeline = [
        {"name": "letterbox", "canvas_size": [8, 8]},
        {
            "name": "normalize",
            "mode": "fixed",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    ]
    cfg = _transforms(pipeline)
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )
    out = transform(Image.new("RGB", (8, 8), color=(255, 255, 255)))
    # white (1.0) under mean/std 0.5 -> +1.0
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_inference_pipeline_is_deterministic():
    # Without train_only augmentation, repeated calls match exactly.
    cfg = _transforms(_letterbox_normalize((16, 16)))
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )
    img = Image.new("RGB", (10, 14), color=(120, 30, 200))
    assert torch.equal(transform(img), transform(img))


def test_disabled_step_is_skipped():
    pipeline = [
        {"name": "letterbox", "canvas_size": [8, 8]},
        {"name": "horizontal_flip", "p": 1.0, "enabled": False},
        {
            "name": "normalize",
            "mode": "fixed",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    ]
    cfg = _transforms(pipeline)
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )
    # An asymmetric image stays put because the flip is disabled (deterministic).
    img = Image.new("RGB", (8, 8))
    img.putpixel((0, 0), (255, 0, 0))
    out = transform(img)
    assert out[0, 0, 0].item() > 0.9  # red corner remained top-left


def test_aspect_bucket_selects_canvas_from_native_shape():
    pipeline = [
        {
            "name": "aspect_bucket",
            "buckets": [
                {"name": "wide", "min_aspect": 1.2, "canvas_size": [16, 32]},
                {"name": "tall", "max_aspect": 0.8, "canvas_size": [32, 16]},
                {"name": "square", "canvas_size": [24, 24]},
            ],
        },
        {
            "name": "normalize",
            "mode": "fixed",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    ]
    cfg = _transforms(pipeline)
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )

    assert transform(Image.new("RGB", (40, 10))).shape == (3, 16, 32)
    assert transform(Image.new("RGB", (10, 40))).shape == (3, 32, 16)
    assert transform(Image.new("RGB", (20, 20))).shape == (3, 24, 24)


def test_foreground_crop_removes_empty_border_before_letterbox():
    pipeline = [
        {"name": "foreground_crop", "threshold": 0.1, "padding_px": 0},
        {"name": "letterbox", "canvas_size": [8, 8]},
    ]
    cfg = _transforms(pipeline)
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )
    img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    for x in range(3, 5):
        for y in range(3, 5):
            img.putpixel((x, y), (255, 255, 255))

    out = transform(img)

    assert out.shape == (3, 8, 8)
    assert torch.all(out > 0.9)


def test_grayscale_step_preserves_channel_count():
    pipeline = [
        {"name": "grayscale", "p": 1.0},
        {"name": "letterbox", "canvas_size": [8, 8]},
    ]
    cfg = _transforms(pipeline)
    transform = build_image_transform(
        cfg.pipeline, image_mode=cfg.image_mode, input_bit_depth=cfg.input_bit_depth
    )

    out = transform(Image.new("RGB", (8, 8), color=(255, 0, 0)))

    assert out.shape == (3, 8, 8)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[1], out[2])
