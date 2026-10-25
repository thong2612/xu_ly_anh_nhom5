from __future__ import annotations

import numpy as np

from src.segmentation.phase3 import (
    _overlay,
    _cache_key,
    cluster_and_detect,
    defect_area_ratio,
    select_suspicious_clusters,
    _summarize_metrics,
)


def _test_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((80, 90, 3), (245, 245, 245), dtype=np.uint8)
    fruit = np.zeros((80, 90), dtype=np.uint8)
    fruit[10:70, 15:75] = 255
    image[10:70, 15:75] = (190, 140, 45)
    image[35:50, 42:58] = (75, 55, 35)
    return image, fruit


def test_kmeans_runs_for_each_k_and_is_reproducible() -> None:
    image, fruit = _test_image()
    original = image.copy()
    for k in (2, 3, 4, 5):
        first = cluster_and_detect(image, fruit, "Lab", k, seed=17, max_fit_pixels=2000)
        second = cluster_and_detect(image, fruit, "Lab", k, seed=17, max_fit_pixels=2000)
        assert first["status"] == second["status"] == "ok"
        assert np.array_equal(first["cluster_labels"], second["cluster_labels"])
        assert np.array_equal(first["defect_mask"], second["defect_mask"])
        assert set(np.unique(first["cluster_labels"][fruit == 0])) == {-1}
        assert np.all(first["defect_mask"][fruit == 0] == 0)
    assert np.array_equal(image, original)


def test_empty_and_small_fruit_masks_are_not_reported_as_zero_area() -> None:
    image, _ = _test_image()
    empty = np.zeros(image.shape[:2], dtype=np.uint8)
    outcome = cluster_and_detect(image, empty, "HSV", 3, seed=1)
    assert outcome["status"] == "insufficient_fruit_pixels"
    assert defect_area_ratio(outcome["defect_mask"], empty) is None
    tiny = empty.copy()
    tiny[4, 4] = 255
    assert cluster_and_detect(image, tiny, "HSV", 3, seed=1)["status"] == "insufficient_fruit_pixels"


def test_suspicious_selection_uses_color_and_continuity_not_darkness_alone() -> None:
    labels = np.zeros((20, 24), dtype=np.int16)
    labels[2:7, 2:8] = 1   # darkest, but nearly the dominant cluster's color
    labels[11:19, 15:22] = 2  # brighter, but strongly color-distinct
    features = np.zeros((20, 24, 3), dtype=np.float32)
    brightness = np.full((20, 24), 0.6, dtype=np.float32)
    features[labels == 0] = (0.6, 0.5, 0.5)
    features[labels == 1] = (0.6, 0.505, 0.5)
    brightness[labels == 1] = 0.2
    features[labels == 2] = (0.95, 0.8, 0.2)
    brightness[labels == 2] = 0.85
    defect, selected, details = select_suspicious_clusters(labels, features, brightness, "Lab")
    assert selected == [2]
    assert not np.any(defect[labels == 1])
    assert np.all(defect[labels == 2] == 255)
    assert next(row for row in details if row["cluster_id"] == 1)["selected"] is False


def test_defect_ratio_uses_only_valid_fruit_pixels() -> None:
    fruit = np.zeros((10, 10), dtype=np.uint8)
    fruit[:4, :5] = 255
    defect = np.zeros_like(fruit)
    defect[:2, :5] = 255
    assert defect_area_ratio(defect, fruit) == 50.0
    assert defect_area_ratio(np.zeros_like(fruit), fruit) == 0.0
    assert defect_area_ratio(defect, np.zeros_like(fruit)) is None


def test_overlay_is_written_as_rgb_image_with_only_masked_pixels_tinted() -> None:
    image = np.full((5, 6, 3), (100, 120, 140), dtype=np.uint8)
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[2, 3] = 255
    overlay_bgr = _overlay(image, mask)
    assert overlay_bgr.shape == (5, 6, 3)
    assert np.array_equal(overlay_bgr[0, 0], (140, 120, 100))
    assert overlay_bgr[2, 3, 2] > overlay_bgr[2, 3, 0]


def test_phase3_cache_key_changes_with_source_mask_and_configuration() -> None:
    row = {"_source_sha256": "source-a"}
    mask = np.zeros((8, 8), dtype=np.uint8)
    settings = {"attempts": 1, "max_iterations": 30, "epsilon": 0.001,
                "max_fit_pixels": 100, "min_fruit_pixels": 10}
    rule = {"minimum_color_distance": 0.12}
    base = _cache_key(row, mask, "heuristic", "original", "Lab", 3, 5, settings, rule)
    assert base == _cache_key(row, mask.copy(), "heuristic", "original", "Lab", 3, 5, settings, rule)
    assert base != _cache_key(row, mask, "heuristic", "original", "Lab", 4, 5, settings, rule)
    assert base != _cache_key(row, mask, "heuristic", "original", "Lab", 3, 5, settings, {**rule, "minimum_color_distance": 0.2})


def test_metric_aggregation_keeps_zero_iou_images_in_denominator() -> None:
    base = {
        "mode": "heuristic", "variant": "original", "color_space": "Lab", "k": 3,
        "phase02_split": "validation", "kmeans_status": "ok", "dice": 0.0,
        "precision": 0.0, "recall": 0.0, "detection_hit": 0,
        "false_positive_pixels_within_fruit": 0, "defect_area_percent": 0.0,
    }
    rows = [{**base, "iou": 0.0}, {**base, "iou": 1.0, "dice": 1.0, "precision": 1.0,
             "recall": 1.0, "detection_hit": 1, "false_positive_pixels_within_fruit": 2,
             "defect_area_percent": 10.0}]
    summary = _summarize_metrics(rows)
    assert summary[0]["n_images"] == 2
    assert summary[0]["iou_mean"] == 0.5
    assert summary[0]["detection_rate"] == 0.5
