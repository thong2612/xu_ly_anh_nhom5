from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.segmentation.baseline import (
    _components,
    apply_morphology,
    fill_holes,
    segment_channel,
)
from src.segmentation.metrics import segmentation_metrics
from src.segmentation.annotation_gui import heuristic_mask_to_polygon
from src.segmentation.pipeline import approve_fruit_masks, import_fruit_masks, make_cache_key


def _color_patch() -> np.ndarray:
    image = np.full((100, 120, 3), 128, dtype=np.uint8)
    image[25:75, 35:85] = (255, 0, 0)
    return image


def test_fixed_threshold_and_polarity_produce_binary_masks() -> None:
    image = _color_patch()
    positive = segment_channel(image, "hsv_s", "fixed", "binary", 0.5,
                               min_area_fraction=0.02, target_area_fraction=0.2)
    inverse = segment_channel(image, "hsv_s", "fixed", "binary_inv", 0.5,
                              min_area_fraction=0.02, target_area_fraction=0.7)
    assert set(np.unique(positive.raw_mask)) <= {0, 255}
    assert positive.raw_mask[40, 50] == 255
    assert inverse.raw_mask[40, 50] == 0
    assert positive.threshold_native == 0.5


def test_otsu_threshold_handles_nonuniform_channel() -> None:
    result = segment_channel(_color_patch(), "hsv_s", "otsu", "binary",
                             min_area_fraction=0.02, target_area_fraction=0.2)
    assert 0 <= result.threshold_u8 <= 255
    assert result.raw_mask[40, 50] == 255
    assert result.raw_mask[5, 5] == 0


def test_morphology_open_close_and_fill_holes() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask[25:35, 25:35] = 0
    mask[3, 3] = 255
    cleaned = apply_morphology(mask, open_kernel=3, close_kernel=5, open_iterations=1, close_iterations=1)
    filled = fill_holes(cleaned)
    assert cleaned[3, 3] == 0
    assert filled[30, 30] == 255
    assert filled[0, 0] == 0


def test_component_ranking_does_not_always_choose_largest() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:90, 50:90] = 255  # central component, area .16
    mask[0:45, 0:45] = 255   # larger component, but touches two borders
    ranked = _components(mask, 0.01, 0.9, target_area_fraction=0.16)
    assert ranked
    chosen = ranked[0][2]
    # The scoring balances plausible area, shape, and position; ensure both stay candidates.
    assert len(ranked) == 2
    assert np.count_nonzero(chosen) > 0
    assert chosen[60, 60] == 255
    assert chosen[10, 10] == 0
    assert ranked[0][0] >= ranked[1][0]


def test_empty_and_full_threshold_are_reported_without_forcing_mask() -> None:
    gray = np.full((40, 40, 3), 100, dtype=np.uint8)
    saturated = np.full((40, 40, 3), (255, 0, 0), dtype=np.uint8)
    empty = segment_channel(gray, "hsv_s", "fixed", "binary", 0.5)
    full = segment_channel(saturated, "hsv_s", "fixed", "binary", 0.5)
    assert empty.status == "empty_threshold"
    assert full.status == "all_foreground"
    assert not np.any(empty.proposed_mask == 255)


def test_input_array_is_not_mutated() -> None:
    image = _color_patch()
    before = image.copy()
    segment_channel(image, "hsv_s", "otsu", "binary", min_area_fraction=0.02)
    assert np.array_equal(image, before)


def test_metrics_iou_dice_and_empty_masks() -> None:
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[2:6, 2:6] = 255
    pred = np.zeros_like(truth)
    pred[2:6, 3:7] = 255
    scores = segmentation_metrics(pred, truth)
    assert scores["iou"] == 12 / 20
    assert scores["dice"] == 24 / 32
    empty = segmentation_metrics(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
    assert empty["iou"] == empty["dice"] == 1.0


def test_cache_key_includes_all_settings() -> None:
    base = {"channel": "lab_b", "method": "fixed", "gray_world_enabled": False}
    changed = {**base, "method": "otsu"}
    assert make_cache_key("id", "hash", base) != make_cache_key("id", "hash", changed)


def test_manual_mask_import_requires_exact_dimensions_and_binary_values(tmp_path: Path) -> None:
    image_rel = "data/raw/a.png"
    image_path = tmp_path / image_rel
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 20), (120, 90, 70)).save(image_path)
    candidate_path = tmp_path / "data/metadata/fruit_gt_candidates.csv"
    candidate_path.parent.mkdir(parents=True)
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "dataset", "task", "label", "source_image_path", "annotation_status"])
        writer.writeheader()
        writer.writerow({"image_id": "sample1", "dataset": "a", "task": "ripeness_classification", "label": "ripe", "source_image_path": image_rel, "annotation_status": "pending_manual_gt"})
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    mask = np.zeros((20, 24), np.uint8)
    mask[3:17, 5:20] = 255
    cv2.imwrite(str(incoming / "sample1.png"), mask)
    config = {"_project_root": str(tmp_path)}
    manifest = import_fruit_masks(config, "incoming")
    assert manifest.exists()
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        record = next(csv.DictReader(stream))
    assert record["validation_status"] == "valid_format_pending_semantic_review"
    assert (tmp_path / record["ground_truth_path"]).exists()
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidate = next(csv.DictReader(stream))
    assert candidate["annotation_status"] == "imported_pending_semantic_review"
    approve_fruit_masks(config, ["sample1"])
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        record = next(csv.DictReader(stream))
    assert record["validation_status"] == "valid_whole_fruit_mask"
    assert record["semantic_review_status"] == "human_confirmed_whole_fruit"
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidate = next(csv.DictReader(stream))
    assert candidate["annotation_status"] == "annotated"


def test_heuristic_mask_becomes_editable_polygon_without_being_ground_truth() -> None:
    mask = np.zeros((50, 60), np.uint8)
    cv2.ellipse(mask, (30, 25), (15, 10), 0, 0, 360, 255, -1)
    polygon = heuristic_mask_to_polygon(mask)
    assert len(polygon) >= 3
    assert all(0 <= x < 60 and 0 <= y < 50 for x, y in polygon)
    with np.testing.assert_raises(ValueError):
        heuristic_mask_to_polygon((mask // 255).astype(np.uint8))
