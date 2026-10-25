from __future__ import annotations

import numpy as np


def _binary(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask")
    if array.dtype == np.bool_:
        return array
    if not np.isin(array, (0, 1, 255)).all():
        raise ValueError(f"{name} must contain only binary values (0/1/255)")
    return array != 0


def segmentation_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute pixel metrics for a verified whole-fruit ground-truth mask."""
    pred = _binary(prediction, "prediction")
    truth = _binary(ground_truth, "ground_truth")
    if pred.shape != truth.shape:
        raise ValueError("prediction and ground_truth must have identical shapes")
    intersection = int(np.count_nonzero(pred & truth))
    pred_count = int(np.count_nonzero(pred))
    truth_count = int(np.count_nonzero(truth))
    union = pred_count + truth_count - intersection
    dice_denominator = pred_count + truth_count
    return {
        "iou": 1.0 if union == 0 else intersection / union,
        "dice": 1.0 if dice_denominator == 0 else 2 * intersection / dice_denominator,
        "precision": 1.0 if pred_count == 0 and truth_count == 0 else (intersection / pred_count if pred_count else 0.0),
        "recall": 1.0 if pred_count == 0 and truth_count == 0 else (intersection / truth_count if truth_count else 0.0),
    }
