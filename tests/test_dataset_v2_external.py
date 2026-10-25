import json

import numpy as np
import pytest

from src.validation.dataset_v2_external import (
    crop_exif_corrected_rgb, evaluate_predictions, primary_exclusion_reason,
)


def test_crop_uses_normalized_exif_corrected_xyxy_without_mutation():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    original = image.copy()
    bbox = json.dumps({
        "format": "normalized_xyxy_0_1",
        "coordinates": "EXIF-corrected displayed image",
        "x1": 0.25, "y1": 1 / 6, "x2": 0.75, "y2": 2 / 3,
    })
    crop = crop_exif_corrected_rgb(image, bbox)
    assert crop.shape == (3, 4, 3)
    np.testing.assert_array_equal(crop, image[1:4, 2:6])
    np.testing.assert_array_equal(image, original)


def test_crop_clamps_only_source_rounding_tolerance():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = json.dumps({
        "format": "normalized_xyxy_0_1",
        "coordinates": "EXIF-corrected displayed image",
        "x1": -0.0000005, "y1": 0.0, "x2": 0.5, "y2": 1.0,
    })
    assert crop_exif_corrected_rgb(image, bbox).shape == (10, 5, 3)


@pytest.mark.parametrize("bbox", [
    '{"format":"wrong","coordinates":"EXIF-corrected displayed image","x1":0,"y1":0,"x2":1,"y2":1}',
    '{"format":"normalized_xyxy_0_1","coordinates":"raw","x1":0,"y1":0,"x2":1,"y2":1}',
    '{"format":"normalized_xyxy_0_1","coordinates":"EXIF-corrected displayed image","x1":0.8,"y1":0,"x2":0.2,"y2":1}',
])
def test_crop_rejects_unknown_or_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        crop_exif_corrected_rgb(np.zeros((10, 10, 3), dtype=np.uint8), bbox)


def test_primary_metric_excludes_conflict_and_duplicate_risk():
    row = {"source": "MangoFruitBD_v2", "label_conflict_status": "none_observed", "curated_label": "defective"}
    assert primary_exclusion_reason(row) == ""
    assert primary_exclusion_reason({**row, "label_conflict_status": "fine_grained_conflict_binary_consistent"}) == "source_fine_label_conflict"
    assert primary_exclusion_reason(row, "exact_match_to_training") == "exact_cross_dataset_duplicate"
    assert primary_exclusion_reason(row, "near_similarity_candidate_excluded") == "unresolved_near_duplicate_risk"


def test_harumanis_does_not_get_an_unsupported_class_mapping():
    row = {"source": "Harumanis_v4", "label_conflict_status": "none_observed", "curated_label": "semi_ripe"}
    assert primary_exclusion_reason(row) == "no_label_mapping_for_external_metric"


def test_evaluation_reports_raw_and_one_vote_per_source_group_metrics():
    rows = [
        {"image_id": "a1", "source_group": "frame-a", "evaluation_eligible": True, "prediction_available": True, "true_label": "healthy", "prediction": "healthy"},
        {"image_id": "a2", "source_group": "frame-a", "evaluation_eligible": True, "prediction_available": True, "true_label": "healthy", "prediction": "healthy"},
        {"image_id": "b1", "source_group": "frame-b", "evaluation_eligible": True, "prediction_available": True, "true_label": "defective", "prediction": "healthy"},
    ]
    result = evaluate_predictions(rows, ["healthy", "defective"])
    assert result["n_evaluated"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["group_weighted_accuracy"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == [[2, 0], [1, 0]]


def test_unavailable_inference_is_not_counted_as_metric_prediction():
    rows = [
        {"image_id": "a", "source_group": "f1", "evaluation_eligible": True, "prediction_available": True, "true_label": "healthy", "prediction": "healthy"},
        {"image_id": "b", "source_group": "f2", "evaluation_eligible": True, "prediction_available": False, "true_label": "defective", "prediction": ""},
    ]
    result = evaluate_predictions(rows, ["healthy", "defective"])
    assert result["n_evaluated"] == 1
    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"] == [[1, 0], [0, 0]]
