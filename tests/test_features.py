from __future__ import annotations

import numpy as np

from src.color.preprocess import hsv_to_rgb
from src.features.extraction import (
    _build_feature_columns,
    _cache_key,
    _color_features,
    _feature_schema,
    _kmeans_features_from_outcome,
    _analysis_raster,
    _label_feature_summary,
)


def test_color_features_use_only_accepted_roi_and_named_hue_bands() -> None:
    hsv = np.array([[[25.0, 1.0, 1.0], [50.0, 1.0, 1.0],
                     [90.0, 1.0, 1.0], [0.0, 0.0, 0.5]]], dtype=np.float32)
    rgb = hsv_to_rgb(hsv)
    before = rgb.copy()
    roi = np.full((1, 4), 255, dtype=np.uint8)
    features, status = _color_features(rgb, roi, "original_color", {
        "minimum_saturation": 0.15, "minimum_value": 0.08,
        "hue_ranges_degrees": {"orange": [15, 35], "yellow": [35, 70], "green": [70, 165]},
    })
    assert status == "ok"
    assert features["original_color__orange_fraction"] == 0.25
    assert features["original_color__yellow_fraction"] == 0.25
    assert features["original_color__green_fraction"] == 0.25
    assert np.array_equal(rgb, before)


def test_invalid_or_missing_fruit_mask_leaves_color_features_blank() -> None:
    rgb = np.full((8, 9, 3), 120, dtype=np.uint8)
    features, status = _color_features(rgb, None, "original_color", {})
    assert status == "fruit_mask_unavailable"
    assert features
    assert set(features.values()) == {""}


def test_analysis_raster_resizes_without_mutating_raw_frame_and_preserves_aspect() -> None:
    rgb = np.zeros((400, 800, 3), dtype=np.uint8)
    rgb[:, :400] = (250, 20, 10)
    original = rgb.copy()
    analysis, alpha = _analysis_raster(rgb, None, 200)
    assert analysis.shape == (100, 200, 3)
    assert alpha is None
    assert np.array_equal(rgb, original)


def test_successful_kmeans_no_detection_uses_zero_area_but_missing_region_moments() -> None:
    rgb = np.full((5, 5, 3), (80, 110, 140), dtype=np.uint8)
    fruit = np.full((5, 5), 255, dtype=np.uint8)
    outcome = {
        "status": "ok", "defect_mask": np.zeros((5, 5), dtype=np.uint8),
        "selected_cluster_ids": [],
        "cluster_details": [{"cluster_id": 0, "area_fraction": 1.0, "selected": False}],
    }
    features = _kmeans_features_from_outcome(rgb, fruit, outcome, "original_lab_k3")
    assert features["original_lab_k3__detected_suspicious"] == 0
    assert features["original_lab_k3__suspicious_area_fraction"] == 0.0
    assert features["original_lab_k3__region_lab_l_mean"] == ""


def test_failed_kmeans_keeps_entire_branch_missing() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fruit = np.full((4, 4), 255, dtype=np.uint8)
    output = _kmeans_features_from_outcome(rgb, fruit, {"status": "insufficient_fruit_pixels"}, "x")
    assert set(output.values()) == {""}


def test_defect_kmeans_branches_share_feature_row_and_metadata_are_not_inputs() -> None:
    columns, groups = _build_feature_columns("defect")
    assert "original_lab_k3" in groups
    assert "original_hsv_k5" in groups
    assert len(columns) == len(set(columns))
    schema = _feature_schema({"ripeness": [], "defect": [{"image_id": "one"}]},
                             {"ripeness": _build_feature_columns("ripeness")[1], "defect": groups}, {})
    defect_inputs = set(schema["feature_columns"]["defect"])
    assert not {"image_id", "label", "phase02_split", "image_path", "dataset"} & defect_inputs
    assert len(groups["original_lab_k3"]) == len(groups["original_hsv_k5"])
    assert schema["sample_counts"]["defect"] == 1


def test_feature_cache_key_tracks_masks_source_and_threshold_config() -> None:
    row = {"image_id": "sample"}
    masks = {"original": np.ones((6, 6), dtype=np.uint8) * 255, "gray_world": None}
    statuses = {"original": "accepted_heuristic", "gray_world": "not_found"}
    feature_settings = {"min_fruit_pixels": 10, "color_proportions": {"minimum_saturation": 0.15}}
    kmeans_settings = {"seed": 7, "attempts": 1, "max_iterations": 30, "epsilon": 0.001,
              "max_fit_pixels": 100, "min_fruit_pixels": 10, "selection_rule": {"minimum_suspicion_score": 0.3}}
    base = _cache_key(row, "source", masks, statuses, feature_settings, kmeans_settings)
    assert base == _cache_key(row, "source", {**masks, "original": masks["original"].copy()}, statuses, feature_settings, kmeans_settings)
    assert base != _cache_key(row, "different-source", masks, statuses, feature_settings, kmeans_settings)
    changed = {**feature_settings, "color_proportions": {"minimum_saturation": 0.2}}
    assert base != _cache_key(row, "source", masks, statuses, changed, kmeans_settings)


def test_label_summary_covers_all_features_and_keeps_missing_values_missing() -> None:
    row = {"image_id": "one", "label": "ripe", "feature": 0.25}
    schema = {"feature_columns": {"ripeness": ["feature", "missing_feature"]}}
    summary = _label_feature_summary({"ripeness": [row]}, schema)
    assert len(summary) == 2
    assert summary[0]["n_images"] == 1
    assert summary[0]["n_nonmissing"] == 1
    assert summary[0]["mean"] == 0.25
    assert summary[1]["n_nonmissing"] == 0
    assert summary[1]["mean"] == ""
