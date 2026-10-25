from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class UnsupportedImageError(ValueError):
    """Raised when an image cannot be represented safely as 8-bit RGB."""


@dataclass(frozen=True)
class ImageFrame:
    rgb: np.ndarray
    alpha: np.ndarray | None = None


@dataclass(frozen=True)
class GrayWorldResult:
    rgb: np.ndarray
    gains: tuple[float, float, float]
    channel_means: tuple[float, float, float]
    target_mean: float
    clipping_fraction: float
    clipped_channel_value_fraction: float
    applied: bool
    reason: str


def decode_image(path: str | Path) -> ImageFrame:
    """Decode an 8-bit grayscale, BGR, or BGRA file into RGB plus optional alpha."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Unreadable or corrupt image: {path}")
    if image.dtype != np.uint8:
        raise UnsupportedImageError(f"Expected 8-bit image, got {image.dtype}: {path}")
    if image.ndim == 2:
        return ImageFrame(rgb=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    if image.ndim != 3:
        raise UnsupportedImageError(f"Unsupported image dimensions {image.shape}: {path}")
    if image.shape[2] == 3:
        return ImageFrame(rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return ImageFrame(rgb=rgba[:, :, :3].copy(), alpha=rgba[:, :, 3].copy())
    raise UnsupportedImageError(f"Expected 1, 3, or 4 channels, got {image.shape[2]}: {path}")


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("BGR input must have shape HxWx3")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB input must have shape HxWx3")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def gray_world(rgb: np.ndarray, alpha: np.ndarray | None = None, epsilon: float = 1e-6) -> GrayWorldResult:
    """Apply Gray World to uint8 RGB without mutating the caller's array.

    Means use visible pixels when alpha is supplied. The clipping fraction is
    the fraction of pixels with at least one channel outside [0, 255] before
    deliberate clipping. Channel-value fraction is also returned separately.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("gray_world expects an HxWx3 uint8 RGB image")
    visible = np.ones(rgb.shape[:2], dtype=bool) if alpha is None else alpha > 0
    if visible.shape != rgb.shape[:2]:
        raise ValueError("alpha shape must match image height and width")
    if not np.any(visible):
        return GrayWorldResult(rgb.copy(), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, False, "no_visible_pixels")
    channels = rgb[visible].astype(np.float32, copy=False)
    means = channels.mean(axis=0, dtype=np.float32)
    target = float(means.mean(dtype=np.float32))
    means_tuple = tuple(float(x) for x in means)
    if np.any(means <= epsilon):
        return GrayWorldResult(rgb.copy(), (1.0, 1.0, 1.0), means_tuple, target, 0.0, 0.0, False, "near_zero_channel_mean")
    gains_arr = target / means
    gains = tuple(float(x) for x in gains_arr)
    adjusted = rgb.astype(np.float32) * gains_arr.reshape(1, 1, 3)
    overflow = np.logical_or(adjusted < 0.0, adjusted > 255.0)
    visible_overflow = overflow[visible]
    clipping_fraction = float(np.count_nonzero(np.any(visible_overflow, axis=1)) / visible_overflow.shape[0])
    channel_value_fraction = float(np.count_nonzero(visible_overflow) / visible_overflow.size)
    balanced = np.clip(np.rint(adjusted), 0.0, 255.0).astype(np.uint8)
    return GrayWorldResult(balanced, gains, means_tuple, target, clipping_fraction, channel_value_fraction, True, "applied")


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float32 OpenCV HSV: H [0,360), S/V [0,1]."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_hsv expects an HxWx3 uint8 RGB image")
    normalized = rgb.astype(np.float32) / np.float32(255.0)
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2HSV).astype(np.float32, copy=False)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float32 CIELAB: L* [0,100], a*/b* signed."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_lab expects an HxWx3 uint8 RGB image")
    normalized = rgb.astype(np.float32) / np.float32(255.0)
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB).astype(np.float32, copy=False)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert canonical float32 HSV (degrees, unit S/V) to uint8 RGB."""
    if hsv.dtype != np.float32 or hsv.ndim != 3 or hsv.shape[2] != 3:
        raise ValueError("hsv_to_rgb expects float32 HxWx3 HSV")
    normalized_hsv = hsv.copy()
    # OpenCV's float HSV uses degrees (unlike uint8 HSV, which uses half-degrees).
    normalized_hsv[:, :, 0] = np.mod(normalized_hsv[:, :, 0], 360.0)
    normalized_hsv[:, :, 1:] = np.clip(normalized_hsv[:, :, 1:], 0.0, 1.0)
    rgb01 = cv2.cvtColor(normalized_hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.uint8)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert float32 OpenCV CIELAB back to uint8 RGB."""
    if lab.dtype != np.float32 or lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_to_rgb expects float32 HxWx3 CIELAB")
    rgb01 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.uint8)


def visible_rgb(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Composite RGBA on white for display without changing source data."""
    if alpha is None:
        return rgb.copy()
    opacity = alpha.astype(np.float32)[:, :, None] / 255.0
    composite = rgb.astype(np.float32) * opacity + 255.0 * (1.0 - opacity)
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


def channel_descriptions(color_space: str) -> tuple[tuple[str, float, float, int], ...]:
    if color_space == "HSV":
        return (("H", 0.0, 360.0, 72), ("S", 0.0, 1.0, 64), ("V", 0.0, 1.0, 64))
    if color_space == "Lab":
        return (("L*", 0.0, 100.0, 64), ("a*", -128.0, 128.0, 64), ("b*", -128.0, 128.0, 64))
    raise ValueError(f"Unknown color space: {color_space}")


def describe_channels(values: np.ndarray, color_space: str, valid_pixels: np.ndarray | None = None) -> dict[str, dict[str, object]]:
    """Summarize float32 HSV or CIELAB channels, optionally excluding transparent pixels."""
    descriptions = channel_descriptions(color_space)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("values must be a float32 HxWx3 color array")
    if valid_pixels is not None and valid_pixels.shape != values.shape[:2]:
        raise ValueError("valid_pixels shape must match the color array")
    result: dict[str, dict[str, object]] = {}
    percentiles = (1, 5, 25, 50, 75, 95, 99)
    for index, (name, low, high, bins) in enumerate(descriptions):
        channel = values[:, :, index]
        selected = channel.ravel() if valid_pixels is None else channel[valid_pixels]
        if selected.size == 0:
            raise ValueError("No visible pixels to summarize")
        counts, edges = np.histogram(selected, bins=bins, range=(low, high))
        quantiles = np.percentile(selected, percentiles).astype(np.float64)
        result[name] = {
            "mean": float(selected.mean(dtype=np.float64)),
            "median": float(np.median(selected)),
            "std": float(selected.std(dtype=np.float64)),
            "min": float(selected.min()),
            "max": float(selected.max()),
            **{f"p{p}": float(q) for p, q in zip(percentiles, quantiles)},
            "histogram_counts": counts.astype(np.int64),
            "histogram_edges": edges.astype(np.float32),
            "valid_pixel_count": int(selected.size),
        }
    return result
