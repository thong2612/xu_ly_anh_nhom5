from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from src.segmentation.annotation_gui import FruitMaskAnnotator
from src.validation.evaluation import (
    EXTERNAL_FIELDS, _read_csv, _wilson_interval, _write_fruit_summary,
    _encode_rgb_within_limit, _robustness_variants, _write_csv, import_external_images, prepare_review_workflows,
    evaluate_reviewed_kmeans_pairs, set_external_review, set_kmeans_pair_review,
)
from src.validation.kmeans_prereview import run_kmeans_prereview


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (170, 95, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_external_manifest(root: Path, row: dict[str, str]) -> None:
    _write_csv(root / "data/external_validation/external_manifest.csv", [row], EXTERNAL_FIELDS)


def test_external_import_is_idempotent_and_leaves_labels_blank(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/external_validation/images"
    image_dir.mkdir(parents=True)
    (image_dir / "new.png").write_bytes(_png_bytes())

    first = import_external_images(tmp_path)
    second = import_external_images(tmp_path)
    rows = _read_csv(tmp_path / "data/external_validation/external_manifest.csv")

    assert first["added"] == 1
    assert second["added"] == 0
    assert len(rows) == 1
    assert rows[0]["ripeness_label"] == rows[0]["defect_label"] == ""
    assert rows[0]["review_status"] == "pending"


def test_external_review_requires_human_label_and_reviewer(tmp_path: Path) -> None:
    image_id = "ext_test"
    _write_external_manifest(tmp_path, {
        "image_id": image_id, "image_path": "data/external_validation/images/a.png",
        "ripeness_label": "", "defect_label": "", "review_status": "pending",
        "source_group": "", "reviewer": "", "reviewed_at": "", "notes": "", "import_status": "valid_40x30",
    })

    with pytest.raises(ValueError, match="at least one manually supplied label"):
        set_external_review(tmp_path, image_id, "approved")
    with pytest.raises(ValueError, match="reviewer identity"):
        set_external_review(tmp_path, image_id, "approved", defect_label="defective")

    set_external_review(tmp_path, image_id, "approved", defect_label="defective", reviewer="reviewer-1")
    row = _read_csv(tmp_path / "data/external_validation/external_manifest.csv")[0]
    assert row["review_status"] == "approved"
    assert row["defect_label"] == "defective"
    assert row["ripeness_label"] == ""


def test_exact_training_duplicate_is_not_external(tmp_path: Path) -> None:
    payload = _png_bytes()
    image_dir = tmp_path / "data/external_validation/images"
    image_dir.mkdir(parents=True)
    (image_dir / "copied.png").write_bytes(payload)
    split_path = tmp_path / "data/splits/phase051/ripeness_phase051.csv"
    split_path.parent.mkdir(parents=True)
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256"])
        writer.writeheader()
        writer.writerow({"sha256": hashlib.sha256(payload).hexdigest()})

    import_external_images(tmp_path)
    row = _read_csv(tmp_path / "data/external_validation/external_manifest.csv")[0]
    assert "exact_duplicate_phase051_data_not_external" in row["import_status"]
    with pytest.raises(ValueError, match="cannot be approved as external"):
        set_external_review(tmp_path, row["image_id"], "approved", defect_label="defective", reviewer="reviewer-1")


def test_kmeans_approval_requires_auditable_human_review(tmp_path: Path) -> None:
    path = tmp_path / "data/metadata/phase07_kmeans_pair_review.csv"
    image_path = tmp_path / "data/images/c-image.png"
    mask_path = tmp_path / "data/masks/c-image.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), (170, 95, 40)).save(image_path)
    gt = np.zeros((30, 40), dtype=np.uint8)
    gt[8:20, 8:20] = 255
    Image.fromarray(gt).save(mask_path)
    _write_csv(path, [{
        "image_id": "c-image", "review_status": "pending", "reviewer": "", "review_notes": "",
        "source_image_path": "data/images/c-image.png", "ground_truth_mask_path": "data/masks/c-image.png",
    }], ["image_id", "review_status", "reviewer", "review_notes", "source_image_path", "ground_truth_mask_path"])
    with pytest.raises(ValueError, match="requires reviewer identity and notes"):
        set_kmeans_pair_review(tmp_path, "c-image", "approved")

    set_kmeans_pair_review(tmp_path, "c-image", "approved", reviewer="reviewer-1", notes="Target mask and alignment reviewed")
    assert _read_csv(path)[0]["review_status"] == "approved"


def test_kmeans_review_evaluation_pairs_configs_and_scores_failure_as_empty(tmp_path: Path) -> None:
    image_path = tmp_path / "data/images/c-image.png"
    mask_path = tmp_path / "data/masks/c-image.png"
    prediction_path = tmp_path / "data/predictions/lab.png"
    for path in (image_path, mask_path, prediction_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 80, 40)).save(image_path)
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:6, 2:6] = 255
    Image.fromarray(gt).save(mask_path)
    Image.fromarray(gt).save(prediction_path)
    metadata = tmp_path / "data/metadata/phase07_kmeans_pair_review.csv"
    _write_csv(metadata, [{
        "image_id": "c-image", "dataset": "mango_segmentation", "review_status": "approved",
        "source_image_path": "data/images/c-image.png", "ground_truth_mask_path": "data/masks/c-image.png",
        "alignment_status": "human_checked", "lab_k3_prediction_path": "data/predictions/lab.png",
        "hsv_k5_prediction_path": "data/predictions/missing.png",
    }], ["image_id", "dataset", "review_status", "source_image_path", "ground_truth_mask_path",
        "alignment_status", "lab_k3_prediction_path", "hsv_k5_prediction_path"])
    _write_csv(tmp_path / "reports/phase03_results.csv", [
        {"mode": "heuristic", "image_id": "c-image", "variant": "original", "color_space": "Lab", "k": "3",
         "kmeans_status": "ok", "predicted_mask_path": "data/predictions/lab.png", "fruit_mask_path": ""},
        {"mode": "heuristic", "image_id": "c-image", "variant": "original", "color_space": "HSV", "k": "5",
         "kmeans_status": "failed", "predicted_mask_path": "data/predictions/missing.png", "fruit_mask_path": ""},
    ], ["mode", "image_id", "variant", "color_space", "k", "kmeans_status", "predicted_mask_path", "fruit_mask_path"])

    result = evaluate_reviewed_kmeans_pairs(tmp_path)
    metrics = _read_csv(tmp_path / "reports/phase07_kmeans_metrics.csv")
    summary = _read_csv(tmp_path / "reports/phase07_kmeans_summary.csv")

    assert result["approved_pairs"] == 1
    assert len(metrics) == 2
    assert {row["image_id"] for row in metrics} == {"c-image"}
    assert metrics[0]["iou"] == "1.0"
    assert metrics[1]["iou"] == "0.0"
    assert all(row["n_ground_truth_valid"] == "1" for row in summary)
    assert sum(int(row["n_prediction_valid"]) for row in summary) == 1
    assert sum(int(row["n_no_valid_result"]) for row in summary) == 1


def test_kmeans_prereview_suggests_without_changing_human_status(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/images"
    mask_dir = tmp_path / "data/masks"
    fruit_dir = tmp_path / "data/fruit"
    for folder in (image_dir, mask_dir, fruit_dir):
        folder.mkdir(parents=True)
    Image.new("RGB", (10, 10), (100, 160, 45)).save(image_dir / "clean.png")
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[3:7, 3:7] = 255
    Image.fromarray(gt).save(mask_dir / "clean.png")
    fruit = np.zeros((10, 10), dtype=np.uint8)
    fruit[1:9, 1:9] = 255
    Image.fromarray(fruit).save(fruit_dir / "clean.png")
    Image.new("RGB", (10, 10), (100, 160, 45)).save(image_dir / "bad.png")
    bad = np.zeros((10, 10), dtype=np.uint8)
    bad[3:7, 3:7] = 128
    Image.fromarray(bad).save(mask_dir / "bad.png")
    Image.fromarray(fruit).save(fruit_dir / "bad.png")
    manifest_path = tmp_path / "data/metadata/phase07_kmeans_pair_review.csv"
    _write_csv(manifest_path, [
        {"image_id": "clean", "review_status": "pending", "source_image_path": "data/images/clean.png",
         "ground_truth_mask_path": "data/masks/clean.png"},
        {"image_id": "bad", "review_status": "pending", "source_image_path": "data/images/bad.png",
         "ground_truth_mask_path": "data/masks/bad.png"},
    ], ["image_id", "review_status", "source_image_path", "ground_truth_mask_path"])
    _write_csv(tmp_path / "reports/phase03_results.csv", [
        {"image_id": image_id, "mode": "heuristic", "variant": "original",
         "fruit_mask_path": f"data/fruit/{image_id}.png", "fruit_mask_status": "accepted_heuristic"}
        for image_id in ("clean", "bad")
    ], ["image_id", "mode", "variant", "fruit_mask_path", "fruit_mask_status"])

    result = run_kmeans_prereview(tmp_path)
    rows = _read_csv(manifest_path)
    by_id = {row["image_id"]: row for row in rows}

    assert result["pairs"] == 2
    assert by_id["clean"]["suggested_status"] == "likely_approved"
    assert by_id["bad"]["suggested_status"] == "likely_rejected"
    assert by_id["clean"]["review_status"] == by_id["bad"]["review_status"] == "pending"
    assert by_id["clean"]["vision_status"] == "not_configured"


def test_prepare_does_not_erase_existing_metric_results(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("data_root: data\n", encoding="utf-8")
    metadata = tmp_path / "data/metadata"
    metadata.mkdir(parents=True)
    (metadata / "fruit_gt_candidates.csv").write_text("image_id,annotation_status\n", encoding="utf-8")
    (metadata / "fruit_gt_manifest.csv").write_text("image_id,validation_status,semantic_review_status\n", encoding="utf-8")
    metrics_path = tmp_path / "reports/phase07_kmeans_metrics.csv"
    _write_csv(metrics_path, [{"image_id": "reviewed", "metric_status": "human_review_approved_exploratory_lineage"}],
               ["image_id", "metric_status"])

    prepare_review_workflows(tmp_path, review_limit=50)

    rows = _read_csv(metrics_path)
    assert rows == [{"image_id": "reviewed", "metric_status": "human_review_approved_exploratory_lineage"}]


def test_wilson_interval_is_bounded_and_handles_empty_sample() -> None:
    low, high = _wilson_interval(7, 10)
    assert 0 < low < 0.7 < high < 1
    assert _wilson_interval(0, 0) == (None, None)


def test_robustness_variants_are_paired_and_within_inference_size_limit() -> None:
    variants = _robustness_variants(_png_bytes(), "source.png")
    assert [row[0] for row in variants] == [
        "original", "brightness_105", "contrast_105", "jpeg_quality_90", "resize_80pct",
    ]
    assert all(row[2] for row in variants)
    assert "aspect_preserved=true" in variants[-1][1]

    large_noise = np.random.default_rng(19).integers(0, 256, (2400, 2400, 3), dtype=np.uint8)
    encoded, encoding = _encode_rgb_within_limit(large_noise, "PNG")
    assert len(encoded) <= 15 * 1024 * 1024
    assert encoding.startswith("JPEG_quality_")


def test_annotation_draft_resumes_without_approving(tmp_path: Path) -> None:
    candidates_path = tmp_path / "data/metadata/fruit_gt_candidates.csv"
    _write_csv(candidates_path, [{"image_id": "fruit-1", "source_image_path": "raw.jpg",
                                 "annotation_status": "pending_manual_gt", "notes": ""}],
               ["image_id", "source_image_path", "annotation_status", "notes"])
    annotator = FruitMaskAnnotator.__new__(FruitMaskAnnotator)
    annotator.config = {"_project_root": str(tmp_path)}
    annotator.candidates_path = candidates_path
    annotator.draft_dir = tmp_path / "data/annotations/fruit_drafts"
    annotator.rows = [{"image_id": "fruit-1", "source_image_path": "raw.jpg"}]
    annotator.index = 0
    annotator.frame = SimpleNamespace(rgb=np.zeros((30, 40, 3), dtype=np.uint8))
    annotator.points = [(2, 3), (35, 4), (20, 25)]
    annotator._write_draft("in_progress_needs_edit")

    resumed = annotator._load_draft(annotator.rows[0])
    candidate = _read_csv(candidates_path)[0]
    draft = (annotator.draft_dir / "fruit-1.json").read_text(encoding="utf-8")
    assert resumed == annotator.points
    assert '"approved": false' in draft
    assert candidate["annotation_status"] == "in_progress_needs_edit"


def test_annotator_queue_keeps_saved_pending_candidates_resumable(tmp_path: Path) -> None:
    candidates_path = tmp_path / "data/metadata/fruit_gt_candidates.csv"
    _write_csv(candidates_path, [
        {"image_id": "pending", "annotation_status": "pending_manual_gt"},
        {"image_id": "saved", "annotation_status": "annotation_saved_pending_import"},
        {"image_id": "approved", "annotation_status": "annotated"},
    ], ["image_id", "annotation_status"])
    annotator = FruitMaskAnnotator.__new__(FruitMaskAnnotator)
    annotator.candidates_path = candidates_path
    assert [row["image_id"] for row in annotator._read_candidates()] == ["pending", "saved"]


def test_fruit_metrics_summary_compares_original_and_gray_world_paired(tmp_path: Path) -> None:
    rows = []
    for variant, offset in (("original", 0.0), ("gray_world", 0.05)):
        rows.append({"image_id": "same-image", "variant": variant, "channel": "lab_a",
                     "method": "otsu", "polarity": "binary", "metric_status": "human_confirmed_subset_exploratory",
                     "iou": 0.7 + offset, "dice": 0.8 + offset, "precision": 0.75 + offset, "recall": 0.85 + offset})
    summary, plot_path = _write_fruit_summary(tmp_path, rows, approved_count=1)
    assert len(summary) == 2
    assert all(row["n_paired_images"] == 1 for row in summary)
    assert summary[0]["paired_mean_iou"] in {0.7, 0.75}
    assert plot_path and (tmp_path / plot_path).is_file()


def test_approved_fruit_summary_writes_chart_when_no_config_pair_is_valid(tmp_path: Path) -> None:
    summary, plot_path = _write_fruit_summary(tmp_path, [], approved_count=1)
    assert summary == []
    assert plot_path == "reports/phase07_segmentation_comparison.png"
    assert (tmp_path / plot_path).is_file()

