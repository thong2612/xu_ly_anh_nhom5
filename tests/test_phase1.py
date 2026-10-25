from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.color.pipeline import _process_one, _cache_key
from src.color.preprocess import (
    UnsupportedImageError,
    bgr_to_rgb,
    decode_image,
    gray_world,
    hsv_to_rgb,
    lab_to_rgb,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_bgr,
)


def test_gray_world_equal_channels_is_identity() -> None:
    image = np.full((5, 7, 3), 117, dtype=np.uint8)
    result = gray_world(image)
    assert result.applied
    assert result.gains == pytest.approx((1.0, 1.0, 1.0))
    np.testing.assert_array_equal(result.rgb, image)
    assert result.clipping_fraction == 0.0


def test_gray_world_near_zero_channel_is_skipped_without_mutation() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    image[:, :, 1] = 140
    original = image.copy()
    result = gray_world(image)
    assert not result.applied
    assert result.reason == "near_zero_channel_mean"
    np.testing.assert_array_equal(result.rgb, original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("value", [0, 1, 2, 250, 254, 255])
def test_gray_world_extreme_dark_bright_and_saturated_values_are_bounded(value: int) -> None:
    image = np.array([[[255, value, 0], [255, value, 255]]], dtype=np.uint8)
    original = image.copy()
    result = gray_world(image)
    assert result.rgb.dtype == np.uint8
    assert result.rgb.shape == image.shape
    assert 0.0 <= result.clipping_fraction <= 1.0
    assert result.rgb.min() >= 0 and result.rgb.max() <= 255
    np.testing.assert_array_equal(image, original)


def test_gray_world_preserves_geometry_and_reports_clipping() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 100
    image[:, :, 2] = np.array([[255, 0, 255], [255, 0, 255]], dtype=np.uint8)
    result = gray_world(image)
    assert result.rgb.shape == image.shape
    assert result.clipping_fraction > 0
    assert len(result.gains) == 3


def test_bgr_rgb_channel_order_round_trip() -> None:
    bgr = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(rgb_to_bgr(rgb), bgr)


def test_decode_grayscale_and_rgba(tmp_path) -> None:
    grayscale_path = tmp_path / "gray.png"
    assert cv2.imwrite(str(grayscale_path), np.array([[0, 127, 255]], dtype=np.uint8))
    grayscale = decode_image(grayscale_path)
    assert grayscale.rgb.shape == (1, 3, 3)
    np.testing.assert_array_equal(grayscale.rgb[0, 1], [127, 127, 127])
    assert grayscale.alpha is None

    rgba_path = tmp_path / "rgba.png"
    rgba = np.array([[[13, 71, 191, 0], [220, 20, 30, 255]]], dtype=np.uint8)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    assert cv2.imwrite(str(rgba_path), bgra)
    decoded = decode_image(rgba_path)
    np.testing.assert_array_equal(decoded.rgb, rgba[:, :, :3])
    np.testing.assert_array_equal(decoded.alpha, rgba[:, :, 3])


def test_decode_reports_corrupt_and_unsupported_images(tmp_path) -> None:
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(OSError, match="Unreadable or corrupt"):
        decode_image(corrupt)
    high_depth = tmp_path / "16bit.png"
    assert cv2.imwrite(str(high_depth), np.full((3, 3), 1024, dtype=np.uint16))
    with pytest.raises(UnsupportedImageError, match="8-bit"):
        decode_image(high_depth)


def test_hsv_lab_scales_float32_and_rgb_round_trips() -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [60, 90, 130]]], dtype=np.uint8)
    hsv = rgb_to_hsv(rgb)
    lab = rgb_to_lab(rgb)
    assert hsv.dtype == np.float32 and lab.dtype == np.float32
    assert hsv[:, :, 0].min() >= 0 and hsv[:, :, 0].max() < 360
    assert hsv[:, :, 1:].min() >= 0 and hsv[:, :, 1:].max() <= 1
    assert lab[:, :, 0].min() >= 0 and lab[:, :, 0].max() <= 100
    assert lab[:, :, 1:].min() >= -128 and lab[:, :, 1:].max() <= 128
    np.testing.assert_allclose(hsv_to_rgb(hsv), rgb, atol=1)
    np.testing.assert_allclose(lab_to_rgb(lab), rgb, atol=2)


def test_color_conversion_does_not_mutate_source() -> None:
    rgb = np.array([[[12, 88, 210], [210, 40, 60]]], dtype=np.uint8)
    original = rgb.copy()
    rgb_to_hsv(rgb)
    rgb_to_lab(rgb)
    gray_world(rgb)
    np.testing.assert_array_equal(rgb, original)


def test_cache_key_separates_gray_world_configuration() -> None:
    enabled = _cache_key("id1", "abc123", True, 512)
    disabled = _cache_key("id1", "abc123", False, 512)
    assert enabled != disabled
    assert enabled == _cache_key("id1", "abc123", True, 512)


def test_image_cache_reuses_npz_and_keeps_original_link(tmp_path) -> None:
    image_path = tmp_path / "mango.png"
    rgb = np.full((8, 12, 3), [120, 90, 40], dtype=np.uint8)
    assert cv2.imwrite(str(image_path), rgb_to_bgr(rgb))
    sample = {"absolute_image_path": str(image_path), "image_id": "sample-001", "image_path": "raw/mango.png"}
    first_key, first_payload, first_preview, first_hit = _process_one(sample, tmp_path / "cache", True, 64, False)
    second_key, second_payload, second_preview, second_hit = _process_one(sample, tmp_path / "cache", True, 64, False)
    assert first_key == second_key
    assert not first_hit and second_hit
    assert first_payload["source_sha256"] == second_payload["source_sha256"]
    assert first_payload["native_width"] == 12 and first_payload["native_height"] == 8
    for name in first_preview:
        np.testing.assert_array_equal(first_preview[name], second_preview[name])

    off_key, off_payload, _, off_hit = _process_one(sample, tmp_path / "cache", False, 64, False)
    assert off_key != first_key
    assert not off_hit
    assert off_payload["gray_world"]["enabled"] is False
    assert off_payload["gray_world"]["reason"] == "disabled"
    np.testing.assert_array_equal(
        np.asarray(off_payload["experiments"][0]["HSV"]["H"]["histogram_counts"]),
        np.asarray(off_payload["experiments"][1]["HSV"]["H"]["histogram_counts"]),
    )
