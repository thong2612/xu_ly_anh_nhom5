from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.color.preprocess import rgb_to_hsv, rgb_to_lab


CHANNELS = {
    "lab_a": (-128.0, 128.0, "Lab a*"),
    "lab_b": (-128.0, 128.0, "Lab b*"),
    "hsv_h": (0.0, 360.0, "HSV H (degrees; circular)"),
    "hsv_s": (0.0, 1.0, "HSV S"),
}


@dataclass
class SegmentationResult:
    channel: str
    method: str
    polarity: str
    threshold_native: float
    threshold_u8: float
    raw_mask: np.ndarray
    proposed_mask: np.ndarray
    status: str
    component_count: int
    selected_area_fraction: float
    score: float
    details: str


def extract_channel(rgb: np.ndarray, channel: str) -> np.ndarray:
    if channel not in CHANNELS:
        raise ValueError(f"Unsupported segmentation channel: {channel}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be HxWx3 uint8")
    if channel.startswith("lab_"):
        lab = rgb_to_lab(rgb)
        return lab[:, :, 1 if channel == "lab_a" else 2]
    hsv = rgb_to_hsv(rgb)
    return hsv[:, :, 0 if channel == "hsv_h" else 1]


def channel_to_u8(values: np.ndarray, channel: str) -> np.ndarray:
    if channel not in CHANNELS or values.ndim != 2:
        raise ValueError("Expected a supported 2D channel array")
    low, high, _ = CHANNELS[channel]
    scaled = (values.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def apply_morphology(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 9,
                     open_iterations: int = 1, close_iterations: int = 2) -> np.ndarray:
    _validate_binary_mask(mask)
    for value, name in ((open_kernel, "open_kernel"), (close_kernel, "close_kernel")):
        if value < 1 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    output = mask.copy()
    if open_kernel > 1 and open_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    if close_kernel > 1 and close_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    _validate_binary_mask(mask)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _validate_binary_mask(mask: np.ndarray) -> None:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("Mask must be a 2D uint8 array")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Mask pixels must be binary 0/255")


def _components(mask: np.ndarray, min_area_fraction: float, max_area_fraction: float,
                target_area_fraction: float) -> list[tuple[float, int, np.ndarray, dict[str, float]]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    total = height * width
    ranked = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        fraction = area / total
        if fraction < min_area_fraction or fraction > max_area_fraction:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contour, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contour[0], True) if contour else 0.0
        hull_area = cv2.contourArea(cv2.convexHull(contour[0])) if contour else 0.0
        solidity = min(1.0, area / hull_area) if hull_area else 0.0
        aspect = w / max(1, h)
        aspect_score = float(np.exp(-abs(np.log(max(aspect, 1e-6) / 1.0))))
        cx, cy = centroids[label]
        center_dist = np.sqrt(((cx - width / 2) / (width / 2)) ** 2 + ((cy - height / 2) / (height / 2)) ** 2)
        center_score = float(np.clip(1.0 - center_dist / np.sqrt(2), 0, 1))
        area_score = float(np.exp(-abs(np.log(max(fraction, 1e-6) / target_area_fraction))))
        border_touch = int(x == 0) + int(y == 0) + int(x + w >= width) + int(y + h >= height)
        border_score = max(0.0, 1.0 - 0.18 * border_touch)
        # Heuristic ranking only: balance size plausibility, centrality, compactness,
        # aspect ratio, and a mild border penalty. This is not a learned confidence.
        score = (0.35 * area_score + 0.25 * center_score + 0.20 * solidity
                 + 0.10 * aspect_score + 0.10 * border_score)
        ranked.append((score, area, component, {"area_fraction": fraction, "solidity": solidity,
                                                 "aspect_ratio": aspect, "center_score": center_score,
                                                 "border_touches": border_touch, "perimeter": perimeter}))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def segment_channel(rgb: np.ndarray, channel: str = "lab_b", method: str = "fixed",
                    polarity: str = "binary", fixed_threshold: float = 8.0,
                    open_kernel: int = 3, close_kernel: int = 9,
                    open_iterations: int = 1, close_iterations: int = 2,
                    fill_hole_pixels: bool = True, min_area_fraction: float = 0.03,
                    max_area_fraction: float = 0.90, target_area_fraction: float = 0.32,
                    minimum_score: float = 0.45, ambiguity_margin: float = 0.025) -> SegmentationResult:
    values = extract_channel(rgb, channel)
    quantized = channel_to_u8(values, channel)
    if method not in {"fixed", "otsu"}:
        raise ValueError("method must be 'fixed' or 'otsu'")
    if polarity not in {"binary", "binary_inv"}:
        raise ValueError("polarity must be 'binary' or 'binary_inv'")
    low, high, _ = CHANNELS[channel]
    mode = cv2.THRESH_BINARY if polarity == "binary" else cv2.THRESH_BINARY_INV
    threshold_u8 = float(fixed_threshold - low) * 255.0 / (high - low)
    if method == "otsu":
        threshold_u8, raw = cv2.threshold(quantized, 0, 255, mode | cv2.THRESH_OTSU)
    else:
        _, raw = cv2.threshold(quantized, threshold_u8, 255, mode)
    threshold_native = low + float(threshold_u8) * (high - low) / 255.0
    ratio = float(np.count_nonzero(raw) / raw.size)
    if ratio <= 0.0001:
        return SegmentationResult(channel, method, polarity, threshold_native, float(threshold_u8), raw,
                                  raw.copy(), "empty_threshold", 0, 0.0, 0.0, "Threshold produced no foreground")
    if ratio >= 0.9999:
        return SegmentationResult(channel, method, polarity, threshold_native, float(threshold_u8), raw,
                                  raw.copy(), "all_foreground", 0, ratio, 0.0, "Threshold produced nearly full-frame foreground")
    cleaned = apply_morphology(raw, open_kernel, close_kernel, open_iterations, close_iterations)
    ranked = _components(cleaned, min_area_fraction, max_area_fraction, target_area_fraction)
    if not ranked:
        return SegmentationResult(channel, method, polarity, threshold_native, float(threshold_u8), raw,
                                  cleaned, "no_plausible_component", 0, 0.0, 0.0,
                                  "No connected component met the configured area range")
    best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    chosen = fill_holes(best[2]) if fill_hole_pixels else best[2]
    area_fraction = float(np.count_nonzero(chosen) / chosen.size)
    status = "accepted_heuristic"
    details = f"ranked={len(ranked)}; runner_up_score={second_score:.4f}; features={best[3]}"
    if best[0] < minimum_score:
        status, details = "low_confidence", f"score={best[0]:.4f} below {minimum_score:.4f}; {details}"
    elif len(ranked) > 1 and best[0] - second_score < ambiguity_margin:
        status, details = "ambiguous_components", f"top component scores differ by < {ambiguity_margin:.4f}; {details}"
    return SegmentationResult(channel, method, polarity, threshold_native, float(threshold_u8), raw,
                              chosen, status, len(ranked), area_fraction, float(best[0]), details)

