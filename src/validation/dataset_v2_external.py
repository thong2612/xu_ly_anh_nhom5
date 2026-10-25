from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def crop_exif_corrected_rgb(rgb: np.ndarray, bbox_json: str) -> np.ndarray:
    """Crop a normalized YOLO xyxy box in an already EXIF-corrected RGB image."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    bbox = json.loads(bbox_json)
    if bbox.get("format") != "normalized_xyxy_0_1" or bbox.get("coordinates") != "EXIF-corrected displayed image":
        raise ValueError("Unsupported crop coordinate metadata")
    values = [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]
    # The source's YOLO validator allows a 1e-6 floating-point edge tolerance;
    # clamp only that tolerance after validating, without extending the crop.
    if not all(math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in values):
        raise ValueError("Crop coordinates must be finite and normalized")
    values = [min(1.0, max(0.0, v)) for v in values]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop box is empty or reversed")
    height, width = rgb.shape[:2]
    left, top = max(0, round(x1 * width)), max(0, round(y1 * height))
    right, bottom = min(width, round(x2 * width)), min(height, round(y2 * height))
    if right <= left or bottom <= top:
        raise ValueError("Crop box is empty after pixel conversion")
    return rgb[top:bottom, left:right].copy()


def make_dhash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def pixel_sha256(rgb: np.ndarray) -> str:
    import hashlib

    height, width = rgb.shape[:2]
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}:RGB:".encode("ascii"))
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def primary_exclusion_reason(row: dict[str, Any], duplicate_status: str = "none") -> str:
    """Strict rules for the mapped MangoFruitBD binary external metric."""
    if row.get("source") != "MangoFruitBD_v2":
        return "no_label_mapping_for_external_metric"
    if row.get("label_conflict_status") != "none_observed":
        return "source_fine_label_conflict"
    if row.get("curated_label") not in {"healthy", "defective"}:
        return "invalid_or_missing_binary_label"
    if duplicate_status == "exact_match_to_training":
        return "exact_cross_dataset_duplicate"
    if duplicate_status == "near_similarity_candidate_excluded":
        return "unresolved_near_duplicate_risk"
    return ""


def evaluate_predictions(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Return row-level and one-vote-per-source-group metrics on available predictions."""
    valid = [r for r in rows if r.get("evaluation_eligible") and r.get("prediction_available")]
    if not valid:
        return {"n_evaluated": 0, "accuracy": None, "macro_f1": None,
                "group_weighted_accuracy": None, "group_weighted_macro_f1": None,
                "confusion_matrix": [[0 for _ in labels] for _ in labels]}
    y_true = [str(r["true_label"]) for r in valid]
    y_pred = [str(r["prediction"]) for r in valid]
    groups = Counter(str(r.get("source_group") or r["image_id"]) for r in valid)
    weights = np.asarray([1.0 / groups[str(r.get("source_group") or r["image_id"])] for r in valid])
    return {
        "n_evaluated": len(valid),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "group_weighted_accuracy": float(accuracy_score(y_true, y_pred, sample_weight=weights)),
        "group_weighted_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", sample_weight=weights, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "group_weighted_confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels, sample_weight=weights).tolist(),
    }


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source_group") or row["image_id"])].append(row)
    result = []
    for group_id, samples in sorted(groups.items()):
        primary = [r for r in samples if r.get("evaluation_eligible")]
        successful = [r for r in primary if r.get("prediction_available")]
        result.append({
            "source_group": group_id,
            "candidate_crops": len(samples),
            "primary_eligible_crops": len(primary),
            "inference_available_crops": len(successful),
            "coverage": len(successful) / len(primary) if primary else "",
            "correct_crops": sum(r.get("true_label") == r.get("prediction") for r in successful),
            "error_crops": sum(r.get("true_label") != r.get("prediction") for r in successful),
            "source_disease_labels": "|".join(sorted({str(r.get("source_disease_label", "")) for r in samples if r.get("source_disease_label")})),
            "binary_labels": "|".join(sorted({str(r.get("true_label", "")) for r in samples if r.get("true_label")})),
        })
    return result

