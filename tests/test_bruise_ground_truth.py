from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.validation.bruise_ground_truth import (
    MANIFEST_FIELDS, _save_candidate, _summaries, approve_masks, import_mask, rasterize_polygons,
)


def _project_with_one_source(root: Path) -> tuple[dict[str, str], np.ndarray]:
    source = root / "data/raw/source.png"
    source.parent.mkdir(parents=True)
    rgb = np.full((80, 100, 3), 248, dtype=np.uint8)
    cv2.ellipse(rgb, (50, 40), (25, 32), 0, 0, 360, (125, 170, 45), -1)
    assert cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    image_id = "test_image_01"
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update({
        "selection_order": "1", "image_id": image_id, "dataset": "mango_fruit_defect",
        "task": "defect_analysis", "source_group": "test_source",
        "source_lineage_status": "test", "source_image_path": "data/raw/source.png",
        "source_sha256": digest, "pixel_sha256": "test-pixel-hash", "source_label": "defective",
        "width": "100", "height": "80", "workflow_status": "pending",
        "lesion_presence": "", "approval_status": "pending", "technical_validation_status": "not_checked",
    })
    manifest = root / "data/metadata/bruise_gt_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return row, rgb


def _read_manifest(root: Path) -> dict[str, str]:
    with (root / "data/metadata/bruise_gt_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        return next(csv.DictReader(stream))


def test_rasterize_polygons_unions_regions_as_binary_native_mask() -> None:
    mask = rasterize_polygons((40, 60), [
        [(5, 5), (15, 5), (15, 15), (5, 15)],
        [(35, 20), (45, 20), (45, 30), (35, 30)],
    ])

    assert mask.shape == (40, 60)
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) > 0
    assert np.count_nonzero(mask[:, :25]) > 0
    assert np.count_nonzero(mask[:, 30:]) > 0


def test_saved_candidate_is_not_approved_until_explicit_human_action(tmp_path: Path) -> None:
    row, rgb = _project_with_one_source(tmp_path)
    candidate = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.ellipse(candidate, (48, 35), (5, 7), 0, 0, 360, 255, -1)

    saved = _save_candidate(tmp_path, row["image_id"], candidate, "present", "clearly bounded brown spot")
    after_save = _read_manifest(tmp_path)
    assert saved["technical_validation_status"] == "passed"
    assert after_save["workflow_status"] == "candidate"
    assert after_save["approval_status"] == "pending"

    with pytest.raises(ValueError, match="Reviewer identity"):
        approve_masks(tmp_path, [row["image_id"]], "")
    approved = approve_masks(tmp_path, [row["image_id"]], "reviewer-1", "RGB annotation reviewed")
    after_approval = _read_manifest(tmp_path)
    assert approved["approved_count"] == 1
    assert after_approval["workflow_status"] == "approved"
    assert after_approval["approval_status"] == "approved"
    assert after_approval["mask_sha256"]


def test_explicit_no_lesion_can_be_approved_with_empty_mask(tmp_path: Path) -> None:
    row, rgb = _project_with_one_source(tmp_path)
    empty = np.zeros(rgb.shape[:2], dtype=np.uint8)
    saved = _save_candidate(tmp_path, row["image_id"], empty, "absent", "no visible surface injury")

    assert saved["mask_pixels"] == 0
    assert _read_manifest(tmp_path)["approval_status"] == "pending"
    approve_masks(tmp_path, [row["image_id"]], "reviewer-1", "Confirmed no visible injury")
    approved = _read_manifest(tmp_path)
    assert approved["lesion_presence"] == "absent"
    assert approved["approval_status"] == "approved"


def test_import_mask_requires_binary_source_aligned_png_and_blocks_processed_outputs(tmp_path: Path) -> None:
    row, rgb = _project_with_one_source(tmp_path)
    imported = tmp_path / "manual.png"
    invalid = np.zeros(rgb.shape[:2], dtype=np.uint8)
    invalid[20:30, 45:55] = 128
    cv2.imwrite(str(imported), invalid)
    with pytest.raises(ValueError, match="mask_values_must_be_0_or_255"):
        import_mask(tmp_path, row["image_id"], imported, "present")

    processed = tmp_path / "data/processed/automatic.png"
    processed.parent.mkdir(parents=True)
    cv2.imwrite(str(processed), np.zeros(rgb.shape[:2], dtype=np.uint8))
    with pytest.raises(ValueError, match="cannot be imported"):
        import_mask(tmp_path, row["image_id"], processed, "absent")


def test_unavailable_prediction_remains_in_metric_denominator() -> None:
    row = {
        "image_id": "case-1", "config_id": "original_lab_k3", "lesion_presence": "present",
        "prediction_available": "false", "true_positive_pixels": 0,
        "false_positive_pixels": 0, "false_negative_pixels": 12,
        "iou": 0.0, "dice": 0.0,
    }
    candidates = [{"image_id": "case-1", "approval_status": "approved",
                   "workflow_status": "approved", "mask_path": "mask.png", "mask_sha256": "sha"}]

    summary = _summaries([row], candidates)[0]

    assert summary["n_ground_truth_valid"] == 1
    assert summary["n_no_inference"] == 1
    assert summary["inference_failure_rate"] == 1.0
    assert summary["false_negative_pixels"] == 12
