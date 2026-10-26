from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.color.preprocess import decode_image
from src.segmentation.metrics import segmentation_metrics
from src.utils.config import load_config


SELECTION_PATH = "data/metadata/bruise_gt_selection.csv"
MANIFEST_PATH = "data/metadata/bruise_gt_manifest.csv"
ANNOTATION_ROOT = "data/annotations/bruise_gt"
MASK_DIR = "data/annotations/bruise_gt/masks"
DRAFT_DIR = "data/annotations/bruise_gt/drafts"
PREDICTION_DIR = "data/processed/bruise_gt_kmeans_predictions"
METRICS_PATH = "reports/bruise_gt_metrics.csv"
SUMMARY_PATH = "reports/bruise_gt_summary.csv"
REPORT_PATH = "reports/bruise_gt_evaluation.md"
CHART_PATH = "reports/bruise_gt_comparison.png"
SUCCESS_SHEET_PATH = "reports/bruise_gt_success_cases.png"
FAILURE_SHEET_PATH = "reports/bruise_gt_failure_cases.png"
CANDIDATE_SHEET_PATH = "reports/bruise_gt_candidate_contact_sheet.png"
GENERATED_FRUIT_MANIFEST = "data/metadata/bruise_gt_fruit_masks.csv"

TARGET_COUNT = 30
CONFIGURATIONS = (("original_lab_k3", "Lab", 3), ("original_hsv_k5", "HSV", 5))
MANIFEST_FIELDS = [
    "selection_order", "image_id", "dataset", "task", "source_group", "source_lineage_status",
    "source_augmentation_status",
    "source_image_path", "source_sha256", "pixel_sha256", "dhash64", "source_label",
    "width", "height", "selection_source", "selection_note", "workflow_status",
    "lesion_presence", "injury_types", "approval_status", "mask_path", "mask_sha256", "mask_width",
    "mask_height", "mask_values", "technical_validation_status", "technical_validation_note",
    "annotation_saved_at", "approved_by", "approved_at", "semantic_review_notes",
    "case_category", "observed_issue", "cause_hypothesis", "case_review_notes",
]
METRIC_FIELDS = [
    "image_id", "dataset", "source_image_path", "source_sha256", "ground_truth_mask_path",
    "ground_truth_mask_sha256", "lesion_presence", "config_id", "color_space", "k",
    "fruit_mask_status", "fruit_mask_path", "kmeans_status", "inference_error", "prediction_available",
    "prediction_path", "ground_truth_pixels", "prediction_pixels", "true_positive_pixels",
    "false_positive_pixels", "false_negative_pixels", "iou", "dice", "pixel_precision",
    "pixel_recall", "detection_hit", "metric_status", "case_category", "observed_issue",
    "cause_hypothesis", "case_review_notes",
]
SUMMARY_FIELDS = [
    "config_id", "color_space", "k", "n_candidates", "n_masks_saved", "n_approved",
    "n_rejected", "n_uncertain", "n_ground_truth_valid", "n_evaluable", "n_positive",
    "n_negative", "n_prediction_available", "n_no_inference", "inference_failure_rate",
    "mean_iou", "mean_dice", "pixel_precision", "pixel_recall", "detection_rate",
    "false_positive_pixels", "false_negative_pixels", "metric_status", "denominator_note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.15 * (attempt + 1))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), array):
        raise OSError(f"Could not write PNG: {path}")
    temporary.replace(path)


def rasterize_polygons(shape: tuple[int, int], polygons: list[list[tuple[int, int]]]) -> np.ndarray:
    """Union user-drawn polygons into a native-size 0/255 PNG mask."""
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        points[:, 0, 0] = np.clip(points[:, 0, 0], 0, width - 1)
        points[:, 0, 1] = np.clip(points[:, 0, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)
    return mask


def _rough_fruit_region(rgb: np.ndarray) -> np.ndarray:
    """Estimate obvious foreground for a technical background-overlap check only."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    low = small.min(axis=2)
    high = small.max(axis=2)
    brightness = small.astype(np.uint16).mean(axis=2)
    # Candidate images were visually screened against light studio backgrounds.
    # Light, low-chroma pixels connected to the frame edge (including soft shadows)
    # are treated as background for this coarse technical check only.
    background = ((brightness >= 188) & ((high.astype(np.int16) - low.astype(np.int16)) <= 48)).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=8)
    edge_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
    edge_labels.discard(0)
    if not edge_labels:
        small_fruit = np.ones(labels.shape, dtype=np.uint8)
    else:
        edge_background = np.isin(labels, list(edge_labels))
        small_fruit = (~edge_background).astype(np.uint8)
    if small_fruit.shape != (height, width):
        small_fruit = cv2.resize(small_fruit, (width, height), interpolation=cv2.INTER_NEAREST)
    return small_fruit.astype(bool)


def validate_mask_array(rgb: np.ndarray, mask: np.ndarray, lesion_presence: str,
                        source_hash_matches: bool = True) -> dict[str, Any]:
    """Check technical mask constraints without granting semantic approval."""
    errors: list[str] = []
    if not source_hash_matches:
        errors.append("source_image_checksum_mismatch")
    if mask.ndim != 2 or mask.dtype != np.uint8:
        errors.append("mask_must_be_single_channel_uint8")
        return {"passed": False, "status": "failed", "note": ";".join(errors), "outside_fraction": ""}
    if mask.shape != rgb.shape[:2]:
        errors.append("mask_dimensions_do_not_match_source")
    if not np.isin(mask, (0, 255)).all():
        errors.append("mask_values_must_be_0_or_255")
    positives = int(np.count_nonzero(mask == 255))
    if lesion_presence == "present" and positives == 0:
        errors.append("present_lesion_requires_nonempty_mask")
    if lesion_presence == "absent" and positives != 0:
        errors.append("absent_lesion_requires_empty_mask")
    outside_fraction: float | None = None
    if mask.shape == rgb.shape[:2] and np.isin(mask, (0, 255)).all() and positives:
        fruit_region = _rough_fruit_region(rgb)
        outside_fraction = float(np.count_nonzero((mask > 0) & ~fruit_region) / positives)
        if outside_fraction > 0.02:
            errors.append(f"mask_outside_plausible_fruit_region:{outside_fraction:.4f}")
    return {
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "note": ";".join(errors) if errors else "binary, source-aligned, and within rough foreground region",
        "outside_fraction": "" if outside_fraction is None else f"{outside_fraction:.6f}",
    }


def prepare_candidates(root: str | Path) -> dict[str, Any]:
    """Materialize the pre-screened, source-linked 30-image annotation manifest."""
    root = Path(root).resolve()
    selection_rows = _read_csv(root / SELECTION_PATH)
    if len(selection_rows) != TARGET_COUNT:
        raise ValueError(f"Selection file must contain exactly {TARGET_COUNT} rows; found {len(selection_rows)}")
    selected_ids = [row.get("image_id", "") for row in selection_rows]
    if len(set(selected_ids)) != TARGET_COUNT or any(not image_id for image_id in selected_ids):
        raise ValueError("Selection file must contain 30 unique non-empty image_id values")

    source_rows = _read_csv(root / "data/metadata/manifest.csv")
    source_by_id = {row.get("image_id", ""): row for row in source_rows}
    perceptual_rows = _read_csv(root / "data/metadata/perceptual_hashes.csv")
    dhash_by_sha = {row.get("sha256", ""): row.get("dhash64", "") for row in perceptual_rows}
    existing_rows = _read_csv(root / MANIFEST_PATH)
    existing_by_id = {row.get("image_id", ""): row for row in existing_rows}

    prepared: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    for selection in selection_rows:
        image_id = selection["image_id"]
        source = source_by_id.get(image_id)
        if source is None:
            raise ValueError(f"Selected image_id is absent from the clean source manifest: {image_id}")
        if source.get("dataset") not in {"mango_fruit_defect", "mango_segmentation"}:
            raise ValueError(f"Only Dataset B/C image sources are allowed: {image_id}")
        if source.get("task") != "defect_analysis" or source.get("defect_label") != "defective":
            raise ValueError(f"Selected source row is not a defective image: {image_id}")
        if source.get("original_or_augmented") == "augmented":
            raise ValueError(f"Augmented images are not allowed in the annotation sample: {image_id}")
        image_path = _project_path(root, source.get("image_path", ""))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        current_sha = sha256_file(image_path)
        expected_sha = source.get("sha256", "")
        if expected_sha and current_sha != expected_sha:
            raise ValueError(f"Source checksum does not match the clean manifest: {image_id}")
        pixel_hash = source.get("pixel_sha256", "")
        if current_sha in source_hashes or (pixel_hash and pixel_hash in pixel_hashes):
            raise ValueError(f"Exact duplicate selected more than once: {image_id}")
        source_hashes.add(current_sha)
        if pixel_hash:
            pixel_hashes.add(pixel_hash)
        frame = decode_image(image_path)
        height, width = frame.rgb.shape[:2]
        if source.get("width") and int(source["width"]) != width:
            raise ValueError(f"Source width differs from manifest: {image_id}")
        if source.get("height") and int(source["height"]) != height:
            raise ValueError(f"Source height differs from manifest: {image_id}")

        old = existing_by_id.get(image_id, {})
        if old and old.get("source_sha256") and old["source_sha256"] != current_sha:
            raise ValueError(f"Existing annotation row points to a changed source image: {image_id}")
        source_path = source.get("image_path", "")
        source_parts = Path(source_path).parts
        source_group = "/".join(source_parts[:5]) if len(source_parts) >= 5 else source.get("dataset", "")
        item = {
            "selection_order": selection["selection_order"], "image_id": image_id,
            "dataset": source["dataset"], "task": source["task"], "source_group": source_group,
            "source_lineage_status": "release_and_path_known; fruit_instance_lineage_not_provided",
            "source_augmentation_status": source.get("original_or_augmented", "unknown"),
            "source_image_path": source_path, "source_sha256": current_sha,
            "pixel_sha256": pixel_hash, "dhash64": dhash_by_sha.get(current_sha, ""),
            "source_label": source.get("defect_label", ""), "width": width, "height": height,
            "selection_source": selection.get("selection_source", ""),
            "selection_note": selection.get("selection_note", ""),
            "workflow_status": "pending", "lesion_presence": "", "injury_types": "",
            "approval_status": "pending",
            "mask_path": "", "mask_sha256": "", "mask_width": "", "mask_height": "",
            "mask_values": "", "technical_validation_status": "not_checked",
            "technical_validation_note": "", "annotation_saved_at": "", "approved_by": "",
            "approved_at": "", "semantic_review_notes": "", "case_category": "",
            "observed_issue": "", "cause_hypothesis": "", "case_review_notes": "",
        }
        for field in MANIFEST_FIELDS:
            if old.get(field, "") not in (None, "") and field not in {
                "selection_order", "image_id", "dataset", "task", "source_group", "source_lineage_status",
                "source_augmentation_status",
                "source_image_path", "source_sha256", "pixel_sha256", "dhash64", "source_label",
                "width", "height", "selection_source", "selection_note",
            }:
                item[field] = old[field]
        prepared.append(item)

    extra = set(existing_by_id) - set(selected_ids)
    if extra:
        raise ValueError("Existing bruise manifest contains IDs outside the locked 30-image selection: "
                         + ", ".join(sorted(extra)))
    _write_csv(root / MANIFEST_PATH, prepared, MANIFEST_FIELDS)
    sheet_path = _write_candidate_sheet(root, prepared)
    counts = Counter(row["workflow_status"] for row in prepared)
    return {
        "candidate_count": len(prepared), "dataset_counts": dict(Counter(row["dataset"] for row in prepared)),
        "workflow_status_counts": dict(counts), "manifest": MANIFEST_PATH,
        "candidate_contact_sheet": str(sheet_path.relative_to(root)),
        "provenance": "Dataset B source RGB files; selection was visual-only and exact duplicates were excluded.",
    }


def _write_candidate_sheet(root: Path, rows: list[dict[str, Any]]) -> Path:
    columns, tile_w, tile_h = 5, 250, 220
    sheet = Image.new("RGB", (columns * tile_w, math.ceil(len(rows) / columns) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        frame = decode_image(_project_path(root, row["source_image_path"]))
        image = Image.fromarray(frame.rgb)
        image.thumbnail((tile_w - 12, tile_h - 42), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_w + (tile_w - image.width) // 2
        y = (index // columns) * tile_h + 4
        sheet.paste(image, (x, y))
        label = f"{index + 1:02d}  {row['image_id'][:8]}  {Path(row['source_image_path']).name}"
        draw.text(((index % columns) * tile_w + 5, (index // columns) * tile_h + tile_h - 30),
                  label[:39], fill="black")
    target = root / CANDIDATE_SHEET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    rows = _read_csv(root / MANIFEST_PATH)
    if not rows:
        raise FileNotFoundError(f"Run `python scripts/bruise_gt.py prepare` first ({MANIFEST_PATH})")
    return rows


def _replace_row(root: Path, row_update: dict[str, Any]) -> None:
    rows = _manifest_rows(root)
    found = False
    for row in rows:
        if row.get("image_id") == row_update.get("image_id"):
            row.update({key: str(value) if value is not None else "" for key, value in row_update.items()})
            found = True
            break
    if not found:
        raise ValueError(f"Unknown image_id in bruise manifest: {row_update.get('image_id')}")
    _write_csv(root / MANIFEST_PATH, rows, MANIFEST_FIELDS)


def _load_source_and_mask(root: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, Path]:
    source_path = _project_path(root, row["source_image_path"])
    source_matches = source_path.is_file() and sha256_file(source_path) == row.get("source_sha256", "")
    if not source_matches:
        raise ValueError(f"Source image checksum changed or file missing for {row['image_id']}")
    frame = decode_image(source_path)
    mask_path = _project_path(root, row.get("mask_path", ""))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.is_file() else None
    if mask is None:
        raise FileNotFoundError(f"No saved candidate mask for {row['image_id']}")
    return frame.rgb, mask, mask_path


def _save_candidate(root: Path, image_id: str, mask: np.ndarray, lesion_presence: str,
                    reviewer_notes: str = "", injury_types: list[str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    rows = _manifest_rows(root)
    row = next((item for item in rows if item.get("image_id") == image_id), None)
    if row is None:
        raise ValueError(f"Unknown image_id: {image_id}")
    if row.get("approval_status") == "approved":
        raise ValueError(f"{image_id} is already semantically approved; revoke approval before editing")
    if lesion_presence not in {"present", "absent", "uncertain"}:
        raise ValueError("lesion_presence must be present, absent, or uncertain")
    source_path = _project_path(root, row["source_image_path"])
    if not source_path.is_file() or sha256_file(source_path) != row.get("source_sha256", ""):
        raise ValueError(f"Source image checksum changed or file missing for {image_id}")
    frame = decode_image(source_path)
    validation = validate_mask_array(frame.rgb, mask, lesion_presence, source_hash_matches=True)
    target = root / MASK_DIR / f"{image_id}.png"
    if not validation["passed"]:
        raise ValueError("Mask failed technical validation: " + validation["note"])
    _atomic_png(target, mask)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    workflow_status = "uncertain" if lesion_presence == "uncertain" else "candidate"
    types = list(injury_types) if injury_types is not None else ([] if lesion_presence == "absent" else ["unknown"])
    update = {
        "image_id": image_id, "workflow_status": workflow_status, "lesion_presence": lesion_presence,
        "injury_types": json.dumps(types, ensure_ascii=False),
        "approval_status": "pending", "mask_path": target.relative_to(root).as_posix(),
        "mask_sha256": sha256_file(target), "mask_width": mask.shape[1], "mask_height": mask.shape[0],
        "mask_values": "0/255", "technical_validation_status": validation["status"],
        "technical_validation_note": validation["note"], "annotation_saved_at": timestamp,
        "approved_by": "", "approved_at": "", "semantic_review_notes": reviewer_notes,
    }
    _replace_row(root, update)
    _invalidate_evaluation_outputs(root)
    return {"image_id": image_id, "workflow_status": workflow_status, "approval_status": "pending",
            "mask_path": update["mask_path"], "mask_sha256": update["mask_sha256"],
            "technical_validation_status": validation["status"], "mask_pixels": int(np.count_nonzero(mask))}


def import_mask(root: str | Path, image_id: str, mask_path: str | Path,
                lesion_presence: str, reviewer_notes: str = "") -> dict[str, Any]:
    root = Path(root).resolve()
    source = Path(mask_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    blocked_roots = [root / "data/processed", root / "reports", root / "data/annotations/fruit_masks",
                     root / "data/annotations/fruit_drafts"]
    for blocked in blocked_roots:
        try:
            source.relative_to(blocked.resolve())
            raise ValueError(f"Masks from {blocked.name}/automatic or prior-phase outputs cannot be imported as independent ground truth")
        except ValueError as exc:
            if "cannot be imported" in str(exc):
                raise
    known_masks: set[Path] = set()
    for source_row in _read_csv(root / "data/metadata/manifest.csv"):
        for field in ("mask_path", "normalized_mask_path"):
            if source_row.get(field):
                known_masks.add(_project_path(root, source_row[field]).resolve())
    for known_mask in known_masks:
        if source == known_mask:
            raise ValueError("A source-dataset mask cannot be imported as independent ground truth")
    mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not decode mask PNG: {source}")
    return _save_candidate(root, image_id, mask, lesion_presence, reviewer_notes)


def review_status(root: str | Path, image_ids: list[str], status: str, reviewer: str,
                  notes: str) -> dict[str, Any]:
    root = Path(root).resolve()
    if status not in {"pending", "rejected", "uncertain"}:
        raise ValueError("Review status must be pending, rejected, or uncertain")
    if not reviewer.strip():
        raise ValueError("Reviewer identity is required for a semantic decision")
    if status in {"rejected", "uncertain"} and not notes.strip():
        raise ValueError("Add a short note when rejecting or marking uncertain")
    rows = _manifest_rows(root)
    by_id = {row["image_id"]: row for row in rows}
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs")
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown image ID(s): " + ", ".join(sorted(unknown)))
    if status == "pending":
        for image_id in image_ids:
            row = by_id[image_id]
            if row.get("approval_status") == "approved":
                raise ValueError(f"Revoke approved rows explicitly before returning them to pending: {image_id}")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for image_id in image_ids:
        row = by_id[image_id]
        if row.get("approval_status") == "approved":
            raise ValueError(f"Approved mask cannot be changed by review-status: {image_id}")
        row["workflow_status"] = status
        row["approval_status"] = status
        row["approved_by"] = reviewer if status != "pending" else ""
        row["approved_at"] = now if status != "pending" else ""
        row["semantic_review_notes"] = notes
    _write_csv(root / MANIFEST_PATH, rows, MANIFEST_FIELDS)
    _invalidate_evaluation_outputs(root)
    return {"updated": len(image_ids), "status": status, "image_ids": image_ids}


def approve_masks(root: str | Path, image_ids: list[str], reviewer: str,
                  notes: str = "") -> dict[str, Any]:
    root = Path(root).resolve()
    if not reviewer.strip():
        raise ValueError("Reviewer identity is required for semantic approval")
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs; no implicit approve-all is supported")
    rows = _manifest_rows(root)
    by_id = {row["image_id"]: row for row in rows}
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown image ID(s): " + ", ".join(sorted(unknown)))
    prepared: list[tuple[dict[str, str], dict[str, Any]]] = []
    for image_id in image_ids:
        row = by_id[image_id]
        if row.get("workflow_status") not in {"candidate", "uncertain"}:
            raise ValueError(f"{image_id} has no saved mask candidate")
        if row.get("workflow_status") == "uncertain" or row.get("lesion_presence") == "uncertain":
            raise ValueError(f"Resolve uncertain annotation before approval: {image_id}")
        if row.get("approval_status") in {"rejected", "uncertain"}:
            raise ValueError(f"Resolve rejected/uncertain status before approval: {image_id}")
        rgb, mask, mask_path = _load_source_and_mask(root, row)
        if sha256_file(mask_path) != row.get("mask_sha256", ""):
            raise ValueError(f"Saved mask checksum mismatch for {image_id}")
        validation = validate_mask_array(rgb, mask, row.get("lesion_presence", ""), True)
        if not validation["passed"]:
            raise ValueError(f"Technical validation failed for {image_id}: {validation['note']}")
        prepared.append((row, validation))
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for row, validation in prepared:
        row["workflow_status"] = "approved"
        row["approval_status"] = "approved"
        row["technical_validation_status"] = validation["status"]
        row["technical_validation_note"] = validation["note"]
        row["approved_by"] = reviewer
        row["approved_at"] = timestamp
        row["semantic_review_notes"] = notes
    _write_csv(root / MANIFEST_PATH, rows, MANIFEST_FIELDS)
    _invalidate_evaluation_outputs(root)
    return {"approved_count": len(prepared), "image_ids": image_ids, "approved_by": reviewer,
            "approval_timestamp": timestamp}


def revoke_approval(root: str | Path, image_ids: list[str], reviewer: str, notes: str) -> dict[str, Any]:
    root = Path(root).resolve()
    if not reviewer.strip() or not notes.strip():
        raise ValueError("Reviewer identity and a revocation note are required")
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs")
    rows = _manifest_rows(root)
    by_id = {row["image_id"]: row for row in rows}
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown image ID(s): " + ", ".join(sorted(unknown)))
    for image_id in image_ids:
        row = by_id[image_id]
        if row.get("approval_status") != "approved":
            raise ValueError(f"Cannot revoke a mask that is not approved: {image_id}")
        row["workflow_status"] = "candidate" if row.get("mask_path") else "pending"
        row["approval_status"] = "pending"
        row["approved_by"] = ""
        row["approved_at"] = ""
        row["semantic_review_notes"] = f"Approval revoked by {reviewer}: {notes}"
    _write_csv(root / MANIFEST_PATH, rows, MANIFEST_FIELDS)
    _invalidate_evaluation_outputs(root)
    return {"revoked_count": len(image_ids), "image_ids": image_ids, "reviewer": reviewer}


def validate_saved_masks(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    rows = _manifest_rows(root)
    checked = 0
    passed = 0
    failed: list[dict[str, str]] = []
    for row in rows:
        if not row.get("mask_path"):
            continue
        checked += 1
        mask: np.ndarray | None = None
        try:
            rgb, mask, mask_path = _load_source_and_mask(root, row)
            checksum_ok = sha256_file(mask_path) == row.get("mask_sha256", "")
            result = validate_mask_array(rgb, mask, row.get("lesion_presence", ""), checksum_ok)
        except (OSError, ValueError) as exc:
            result = {"passed": False, "status": "failed", "note": f"{type(exc).__name__}:{exc}", "outside_fraction": ""}
        row["technical_validation_status"] = result["status"]
        row["technical_validation_note"] = result["note"]
        row["mask_width"] = str(mask.shape[1]) if isinstance(mask, np.ndarray) else row.get("mask_width", "")
        row["mask_height"] = str(mask.shape[0]) if isinstance(mask, np.ndarray) else row.get("mask_height", "")
        row["mask_values"] = "0/255" if result["passed"] else row.get("mask_values", "")
        if result["passed"]:
            passed += 1
        else:
            failed.append({"image_id": row["image_id"], "note": result["note"]})
    _write_csv(root / MANIFEST_PATH, rows, MANIFEST_FIELDS)
    return {"masks_checked": checked, "technical_pass": passed, "technical_fail": len(failed), "failures": failed,
            "note": "Technical validation never changes semantic approval status."}


def record_case_note(root: str | Path, image_id: str, category: str, observed_issue: str,
                     cause_hypothesis: str, notes: str) -> dict[str, Any]:
    allowed = {"undersegmentation", "normal_peel_false_positive", "shadow_false_positive",
               "small_lesion_missed", "fruit_mask_failure", "lab_hsv_difference", "other"}
    if category not in allowed:
        raise ValueError("Unsupported case category")
    if not observed_issue.strip() and not cause_hypothesis.strip():
        raise ValueError("Record an observed issue or a separate cause hypothesis")
    _replace_row(Path(root).resolve(), {"image_id": image_id, "case_category": category,
                                        "observed_issue": observed_issue,
                                        "cause_hypothesis": cause_hypothesis, "case_review_notes": notes})
    return {"image_id": image_id, "case_category": category, "recorded": True}


def _annotation_status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "candidate": len(rows),
        "masks_saved": sum(bool(row.get("mask_path")) and bool(row.get("mask_sha256")) for row in rows),
        "approved": sum(row.get("approval_status") == "approved" for row in rows),
        "rejected": sum(row.get("approval_status") == "rejected" for row in rows),
        "uncertain": sum(row.get("approval_status") == "uncertain" or row.get("workflow_status") == "uncertain" for row in rows),
    }


def _invalidate_evaluation_outputs(root: Path) -> None:
    """Remove stale scores after any mask or semantic-review state change."""
    candidates = _manifest_rows(root)
    _write_csv(root / METRICS_PATH, [], METRIC_FIELDS)
    _write_csv(root / SUMMARY_PATH, _summaries([], candidates), SUMMARY_FIELDS)
    _write_comparison_chart(root, _summaries([], candidates), 0)
    _write_pending_sheets(root)
    _report(root, candidates, [], _summaries([], candidates), [])


def _write_pending_sheets(root: Path) -> None:
    for relative in (SUCCESS_SHEET_PATH, FAILURE_SHEET_PATH):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.60, "Chưa có mask được người dùng duyệt", ha="center", va="center",
                fontsize=16, weight="bold", color="#6b4f00", transform=ax.transAxes)
        ax.text(0.5, 0.35, "Contact sheet thành công/thất bại sẽ được tạo sau khi có ground truth độc lập.",
                ha="center", va="center", fontsize=11, wrap=True, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _write_comparison_chart(root: Path, summaries: list[dict[str, Any]], n: int,
                            relative_path: str = CHART_PATH) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Original + Lab K=3", "Original + HSV K=5"]
    names = [("mean_iou", "Mean IoU"), ("mean_dice", "Mean Dice"),
             ("pixel_precision", "Pixel precision"), ("pixel_recall", "Pixel recall")]
    x = np.arange(len(names))
    width = 0.36
    for index, summary in enumerate(summaries):
        values = [float(summary[key]) if summary.get(key) not in (None, "") else 0.0 for key, _ in names]
        axes[0].bar(x + (index - 0.5) * width, values, width, label=labels[index])
    axes[0].set_xticks(x, [name for _, name in names], rotation=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pixel overlap and pooled metrics")
    axes[0].legend(fontsize=8)
    rate_keys = ["detection_rate", "inference_failure_rate"]
    rate_labels = ["Positive detection", "No inference"]
    x2 = np.arange(len(rate_keys))
    for index, summary in enumerate(summaries):
        values = [float(summary[key]) if summary.get(key) not in (None, "") else 0.0 for key in rate_keys]
        axes[1].bar(x2 + (index - 0.5) * width, values, width, label=labels[index])
    axes[1].set_xticks(x2, rate_labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Coverage and positive-case detection")
    axes[1].legend(fontsize=8)
    if n == 0:
        for axis in axes:
            axis.text(0.5, 0.5, "Waiting for user-approved annotations", ha="center", va="center",
                      transform=axis.transAxes, color="darkred", wrap=True)
    fig.suptitle(f"Independent bruise ground truth: paired locked configurations | n={n}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _composite(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    selected = mask > 0
    tint = np.asarray(color, dtype=np.float32)
    out[selected] = out[selected] * 0.40 + tint * 0.60
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_case_sheet(root: Path, rows: list[dict[str, Any]], relative_path: str, title: str) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No reviewed cases yet", ha="center", va="center", fontsize=15,
                transform=ax.transAxes)
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    columns, tile_w, tile_h = 2, 500, 320
    sheet = Image.new("RGB", (columns * tile_w, math.ceil(len(rows) / columns) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        rgb = np.asarray(Image.open(_project_path(root, row["source_image_path"])).convert("RGB"))
        gt = cv2.imread(str(_project_path(root, row["ground_truth_mask_path"])), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(_project_path(root, row.get("lab_prediction_path", ""))), cv2.IMREAD_GRAYSCALE)
        hsv = cv2.imread(str(_project_path(root, row.get("hsv_prediction_path", ""))), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros(rgb.shape[:2], dtype=np.uint8)
        if lab is None:
            lab = np.zeros(rgb.shape[:2], dtype=np.uint8)
        if hsv is None:
            hsv = np.zeros(rgb.shape[:2], dtype=np.uint8)
        panels = [rgb, _composite(rgb, gt, (30, 210, 70)), _composite(rgb, lab, (245, 55, 45)),
                  _composite(rgb, hsv, (245, 170, 25))]
        preview_w, preview_h = (tile_w - 14) // 2, 126
        for panel_index, panel in enumerate(panels):
            image = Image.fromarray(panel)
            image.thumbnail((preview_w, preview_h), Image.Resampling.LANCZOS)
            px = (index % columns) * tile_w + (panel_index % 2) * (preview_w + 4) + (preview_w - image.width) // 2
            py = (index // columns) * tile_h + (panel_index // 2) * 145 + 2
            sheet.paste(image, (px, py))
        label = f"{row['image_id'][:10]} | {row.get('lesion_presence', '')} | " \
                f"Lab IoU={row.get('lab_iou', '')} HSV IoU={row.get('hsv_iou', '')}"
        draw.text(((index % columns) * tile_w + 4, (index // columns) * tile_h + 215), label[:48], fill="black")
    sheet.save(target)


def _summaries(rows: list[dict[str, Any]], candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = _annotation_status_counts(candidate_rows)
    summaries: list[dict[str, Any]] = []
    for config_id, color_space, k in CONFIGURATIONS:
        group = [row for row in rows if row.get("config_id") == config_id]
        positive = [row for row in group if row.get("lesion_presence") == "present"]
        available = [row for row in group if row.get("prediction_available") == "true"]
        tp = sum(int(row.get("true_positive_pixels") or 0) for row in group)
        fp = sum(int(row.get("false_positive_pixels") or 0) for row in group)
        fn = sum(int(row.get("false_negative_pixels") or 0) for row in group)
        n_ground_truth_valid = len({row["image_id"] for row in group})
        n_no_inference = len(group) - len(available)
        positive_hits = [row for row in positive if row.get("detection_hit") == "1"]
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = tp / precision_denominator if precision_denominator else (1.0 if not fn else 0.0)
        recall = tp / recall_denominator if recall_denominator else (1.0 if fp == 0 else 0.0)
        metric_status = "waiting_for_approved_ground_truth" if not group else "paired_human_approved_subset"
        summaries.append({
            "config_id": config_id, "color_space": color_space, "k": k,
            "n_candidates": counts["candidate"], "n_masks_saved": counts["masks_saved"],
            "n_approved": counts["approved"], "n_rejected": counts["rejected"],
            "n_uncertain": counts["uncertain"], "n_ground_truth_valid": n_ground_truth_valid,
            "n_evaluable": n_ground_truth_valid, "n_positive": len(positive),
            "n_negative": sum(row.get("lesion_presence") == "absent" for row in group),
            "n_prediction_available": len(available), "n_no_inference": n_no_inference,
            "inference_failure_rate": n_no_inference / n_ground_truth_valid if n_ground_truth_valid else "",
            "mean_iou": float(np.mean([float(row["iou"]) for row in group])) if group else "",
            "mean_dice": float(np.mean([float(row["dice"]) for row in group])) if group else "",
            "pixel_precision": precision if group else "", "pixel_recall": recall if group else "",
            "detection_rate": len(positive_hits) / len(positive) if positive else "",
            "false_positive_pixels": fp, "false_negative_pixels": fn,
            "metric_status": metric_status,
            "denominator_note": "All technically valid, explicitly approved masks; unavailable predictions score as empty masks. "
                                "Detection rate uses positive lesion masks only. Empty/empty IoU and Dice are 1 by convention.",
        })
    return summaries


def _report(root: Path, candidates: list[dict[str, str]], metric_rows: list[dict[str, Any]],
            summaries: list[dict[str, Any]], technical_failures: list[dict[str, str]]) -> Path:
    counts = _annotation_status_counts(candidates)
    eligible_ids = {row["image_id"] for row in metric_rows}
    report = [
        "# MangoResearch — Independent Bruise Ground Truth", "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "## Scope and sample", "",
        "This phase prepares a 30-image human annotation set for evaluating the locked Phase 3 color K-Means configurations. Dataset A/C remain the main research data; Dataset v2 remains external/domain-shift context only. No Phase 5.1 models, dataset splits, existing Phase 1–7 artifacts, or algorithm settings are changed.",
        "The 30 selected images come from Dataset B (`mango_fruit_defect`, source path recorded per image). The source label was used only to locate the defective candidate pool; masks and lesion presence still await independent human annotation. Selection used RGB-only visual screening for visible surface lesions, clear focus, a largely visible fruit, and variation in lesion size, location, color, and count. Exact file/pixel duplicates were removed. No mask, IoU, or K-Means output was used to select them. The local source manifest marks image-level original/augmentation status as unknown and Dataset B provides no per-fruit lineage IDs; same-fruit relationships beyond exact duplicates therefore cannot be ruled out. These provenance limits are recorded per image.",
        "The sample is purposive and small. Results, once available, describe only these annotated images and do not estimate performance across all of Dataset C or B.", "",
        "## Annotation rules and review gate", "",
        "Annotate only clearly visible abnormal dark/brown spots or bounded peel damage, scars, or tears consistent with the agreed surface-injury definition; record the per-polygon type when it is identifiable. Do not mark shadows, the background, natural green/yellow color, generally dark peel, or ambiguous discoloration. Mark uncertain cases as `uncertain`; an explicitly reviewed no-lesion image may be approved with an empty mask and `lesion_presence=absent`. These are visual image annotations, not disease diagnoses or agricultural expert confirmation.",
        "The annotator shows the original RGB image and the user's current polygons only. It does not load Dataset C masks, Lab/HSV clusters, or K-Means predictions. A saved candidate mask is not approved. Only the explicit `approve --image-ids ... --reviewer ...` action after technical checks contributes to metrics.",
        "Technical checks confirm source and mask checksums, exact native dimensions, single-channel 0/255 values, lesion-presence consistency, and overlap with a conservative light-background foreground estimate connected to image edges. Passing these checks never means semantic approval.", "",
        "## Current counts", "",
        f"- Candidate images: **{len(candidates)}**",
        f"- Masks saved as candidates: **{counts['masks_saved']}**",
        f"- Semantically approved: **{counts['approved']}**",
        f"- Rejected: **{counts['rejected']}**",
        f"- Uncertain: **{counts['uncertain']}**",
        f"- Images eligible for paired evaluation: **{len(eligible_ids)}**",
        f"- Approved masks failing technical re-check: **{len(technical_failures)}**", "",
        "No metrics are reported until at least one independent mask is explicitly approved.", "",
        "## Locked paired evaluation", "",
        "The two configurations are Original + CIELAB + K=3 and Original + HSV + K=5, using the Phase 3 `cluster_and_detect` routine, fixed Phase 3 seeds/settings, and the existing Phase 2 default fruit-mask baseline. No threshold, K, cluster rule, or model is tuned on this 30-image set. Both configurations use the same approved images. Fruit-mask or K-Means failures remain in the denominator and score as empty predictions; availability and failure rate are reported separately.",
        "Metrics: mean image IoU and Dice; pooled pixel precision/recall; positive-case detection rate; inference availability/failure rate; false-positive and false-negative pixel totals. Empty/empty IoU and Dice follow the existing metric convention of 1; detection rate uses positive masks only.", "",
        "## Summary", "",
        "| Configuration | Approved GT | Inference available | Mean IoU | Mean Dice | Pixel precision | Pixel recall | Positive detection | No inference | FP pixels | FN pixels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(value: Any) -> str:
            return "—" if value in (None, "") else (f"{float(value):.4f}" if isinstance(value, float) or isinstance(value, np.floating) else str(value))
        report.append("| {config_id} | {n_ground_truth_valid} | {n_prediction_available}/{n_ground_truth_valid} | {mean_iou} | {mean_dice} | {pixel_precision} | {pixel_recall} | {detection_rate} | {inference_failure_rate} | {false_positive_pixels} | {false_negative_pixels} |".format(
            config_id=row["config_id"], n_ground_truth_valid=row["n_ground_truth_valid"],
            n_prediction_available=row["n_prediction_available"], mean_iou=fmt(row["mean_iou"]),
            mean_dice=fmt(row["mean_dice"]), pixel_precision=fmt(row["pixel_precision"]),
            pixel_recall=fmt(row["pixel_recall"]), detection_rate=fmt(row["detection_rate"]),
            inference_failure_rate=fmt(row["inference_failure_rate"]), false_positive_pixels=row["false_positive_pixels"],
            false_negative_pixels=row["false_negative_pixels"]))
    reviewed_notes = [row for row in candidates if row.get("case_category")]
    report.extend([
        "", "## Case review", "",
        "Contact sheets show RGB, independent ground truth, and the two predictions with separate colors. Visual explanations are recorded only after human case review. `observed_issue` and `cause_hypothesis` are separate manifest fields; the cause is not inferred from a score. Use `case-note` to record undersegmentation, normal peel color, shadow, small-lesion miss, fruit-mask failure, or a Lab/HSV difference.",
        f"Human case notes recorded: **{len(reviewed_notes)}**. Until case notes exist, no claims are made about which visible error type caused a metric difference.", "",
        "## Outputs and commands", "",
        f"- Annotation manifest: `{MANIFEST_PATH}`",
        f"- Candidate RGB contact sheet: `{CANDIDATE_SHEET_PATH}`",
        f"- Candidate masks and drafts: `{ANNOTATION_ROOT}/`",
        f"- Metrics: `{METRICS_PATH}`; summary: `{SUMMARY_PATH}`; chart: `{CHART_PATH}`",
        f"- Success/failure sheets: `{SUCCESS_SHEET_PATH}` and `{FAILURE_SHEET_PATH}`",
        "- `python scripts/bruise_gt.py prepare` — validate and prepare the locked candidate list.",
        "- `python scripts/bruise_gt.py annotate` — open/resume the RGB-only polygon editor.",
        "- `python scripts/bruise_gt.py import-mask --image-id ID --mask-path PATH --lesion-presence present|absent|uncertain` — import a manual binary PNG as an unapproved candidate.",
        "- `python scripts/bruise_gt.py approve --image-ids ID1 ID2 --reviewer NAME` — approve exactly the listed candidates after technical validation.",
        "- `python scripts/bruise_gt.py review --image-ids ID --status rejected|uncertain --reviewer NAME --notes TEXT` — record an explicit semantic decision.",
        "- `python scripts/bruise_gt.py revoke --image-ids ID --reviewer NAME --notes TEXT` — revoke a prior approval before editing that mask.",
        "- `python scripts/bruise_gt.py evaluate` — evaluate approved masks and export metrics/report/contact sheets.",
        "- `python scripts/bruise_gt.py case-note --image-id ID --category CATEGORY --observed-issue TEXT --cause-hypothesis TEXT` — separate visible errors from possible explanations.",
        "- `python scripts/bruise_gt.py report` — export the current status/report without running K-Means.", "",
    ])
    target = root / REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(report), encoding="utf-8")
    return target


def export_report(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    candidates = _manifest_rows(root)
    valid_approved = []
    technical_failures = []
    for row in candidates:
        if row.get("approval_status") != "approved":
            continue
        try:
            rgb, mask, mask_path = _load_source_and_mask(root, row)
            checksum_ok = sha256_file(mask_path) == row.get("mask_sha256", "")
            validation = validate_mask_array(rgb, mask, row.get("lesion_presence", ""), checksum_ok)
            if validation["passed"]:
                valid_approved.append(row)
            else:
                technical_failures.append({"image_id": row["image_id"], "note": validation["note"]})
        except (OSError, ValueError) as exc:
            technical_failures.append({"image_id": row["image_id"], "note": f"{type(exc).__name__}:{exc}"})
    metrics = _read_csv(root / METRICS_PATH)
    summaries = _summaries(metrics, candidates)
    _write_csv(root / METRICS_PATH, metrics, METRIC_FIELDS)
    _write_csv(root / SUMMARY_PATH, summaries, SUMMARY_FIELDS)
    _write_comparison_chart(root, summaries, len(valid_approved))
    if not metrics:
        _write_pending_sheets(root)
    report = _report(root, candidates, metrics, summaries, technical_failures)
    counts = _annotation_status_counts(candidates)
    return {**counts, "eligible_for_evaluation": len(valid_approved), "metric_rows": len(metrics),
            "technical_failures": technical_failures, "report": str(report.relative_to(root)),
            "metrics": METRICS_PATH, "summary": SUMMARY_PATH, "comparison_chart": CHART_PATH}


def _paired_predictions(root: Path, approved_rows: list[dict[str, str]],
                        prediction_dir: str = PREDICTION_DIR,
                        generated_fruit_manifest: str = GENERATED_FRUIT_MANIFEST,
                        fruit_mask_dir: str = "data/processed/bruise_gt_fruit_masks") -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from src.segmentation.kmeans import Phase2FruitMaskProvider, cluster_and_detect

    config = load_config(root / "config.yaml")
    phase3 = dict(config.get("kmeans", {}))
    phase_settings = {
        "seed": int(phase3.get("seed", 20261023)),
        "attempts": int(phase3.get("attempts", 1)),
        "max_iterations": int(phase3.get("max_iterations", 30)),
        "epsilon": float(phase3.get("epsilon", 0.001)),
        "max_fit_pixels": int(phase3.get("max_fit_pixels", 4000)),
        "min_fruit_pixels": int(phase3.get("min_fruit_pixels", 100)),
    }
    rule = dict(phase3.get("selection_rule", {}))
    isolated_config = dict(config)
    isolated_config["segmentation"] = dict(config.get("segmentation", {}))
    isolated_config["segmentation"]["mask_dir"] = fruit_mask_dir
    generated_manifest = root / generated_fruit_manifest
    provider = Phase2FruitMaskProvider(isolated_config, dict(config.get("segmentation", {})), generated_manifest)
    metric_rows: list[dict[str, Any]] = []
    paired_by_image: dict[str, dict[str, Any]] = {}
    for row in approved_rows:
        source_path = _project_path(root, row["source_image_path"])
        frame = decode_image(source_path)
        source_hash = sha256_file(source_path)
        try:
            fruit_mask, fruit_status, _fruit_source, fruit_rel = provider.get(
                {"image_id": row["image_id"], "dataset": row["dataset"]}, frame.rgb, frame.alpha, source_hash, "E1"
            )
            fruit_error = ""
        except Exception as exc:
            fruit_mask, fruit_rel = None, ""
            fruit_status = f"fruit_mask_error:{type(exc).__name__}"
            fruit_error = f"{type(exc).__name__}: {exc}"
        target_path = _project_path(root, row["mask_path"])
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        image_pair: dict[str, Any] = {"source_image_path": row["source_image_path"],
                                      "ground_truth_mask_path": row["mask_path"],
                                      "lesion_presence": row.get("lesion_presence", "")}
        for config_id, color_space, k in CONFIGURATIONS:
            seed_text = f"{phase_settings['seed']}|{row['image_id']}|heuristic|original|{color_space}|{k}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "little") & 0x7FFFFFFF
            prediction_path = root / prediction_dir / row["image_id"] / f"{config_id}.png"
            fruit_mask_available = fruit_mask is not None
            inference_error = fruit_error
            if not fruit_mask_available:
                kmeans_status = "fruit_mask_unavailable"
                prediction_available = False
                prediction = np.zeros(target.shape, dtype=np.uint8)
            else:
                try:
                    result = cluster_and_detect(
                        frame.rgb, fruit_mask, color_space, k, seed,
                        attempts=phase_settings["attempts"], max_iterations=phase_settings["max_iterations"],
                        epsilon=phase_settings["epsilon"], max_fit_pixels=phase_settings["max_fit_pixels"],
                        min_fruit_pixels=phase_settings["min_fruit_pixels"], rule=rule,
                    )
                    kmeans_status = str(result["status"])
                    prediction_available = kmeans_status == "ok"
                    prediction = result["defect_mask"] if prediction_available else np.zeros(target.shape, dtype=np.uint8)
                    if not prediction_available:
                        inference_error = str(result.get("error", ""))
                except Exception as exc:
                    kmeans_status = f"processing_error:{type(exc).__name__}"
                    inference_error = f"{type(exc).__name__}: {exc}"
                    prediction_available = False
                    prediction = np.zeros(target.shape, dtype=np.uint8)
            if prediction_available:
                _atomic_png(prediction_path, prediction)
            else:
                prediction_path = Path()
            values = segmentation_metrics(prediction, target)
            truth = target > 0
            predicted = prediction > 0
            tp = int(np.count_nonzero(truth & predicted))
            fp = int(np.count_nonzero(~truth & predicted))
            fn = int(np.count_nonzero(truth & ~predicted))
            detection_hit = "" if row.get("lesion_presence") != "present" else str(int(tp > 0))
            metric_rows.append({
                "image_id": row["image_id"], "dataset": row["dataset"],
                "source_image_path": row["source_image_path"], "source_sha256": row["source_sha256"],
                "ground_truth_mask_path": row["mask_path"], "ground_truth_mask_sha256": row["mask_sha256"],
                "lesion_presence": row.get("lesion_presence", ""), "config_id": config_id,
                "color_space": color_space, "k": k, "fruit_mask_status": fruit_status,
                "fruit_mask_path": fruit_rel, "kmeans_status": kmeans_status,
                "inference_error": inference_error,
                "prediction_available": str(prediction_available).lower(),
                "prediction_path": prediction_path.relative_to(root).as_posix() if prediction_available else "",
                "ground_truth_pixels": int(np.count_nonzero(truth)),
                "prediction_pixels": int(np.count_nonzero(predicted)), "true_positive_pixels": tp,
                "false_positive_pixels": fp, "false_negative_pixels": fn, "iou": values["iou"],
                "dice": values["dice"], "pixel_precision": values["precision"],
                "pixel_recall": values["recall"], "detection_hit": detection_hit,
                "metric_status": "human_approved" if prediction_available else "prediction_unavailable_scored_as_empty",
                "case_category": row.get("case_category", ""), "observed_issue": row.get("observed_issue", ""),
                "cause_hypothesis": row.get("cause_hypothesis", ""),
                "case_review_notes": row.get("case_review_notes", ""),
            })
            image_pair[config_id] = {"prediction": prediction, "prediction_available": prediction_available,
                                     "prediction_path": prediction_path, "iou": values["iou"],
                                     "detection_hit": detection_hit, "kmeans_status": kmeans_status,
                                     "fruit_mask_status": fruit_status}
        paired_by_image[row["image_id"]] = image_pair
    provider.checkpoint()
    return metric_rows, paired_by_image


def evaluate_approved(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    candidates = _manifest_rows(root)
    valid_approved: list[dict[str, str]] = []
    technical_failures: list[dict[str, str]] = []
    for row in candidates:
        if row.get("approval_status") != "approved" or row.get("workflow_status") != "approved":
            continue
        try:
            rgb, mask, mask_path = _load_source_and_mask(root, row)
            validation = validate_mask_array(rgb, mask, row.get("lesion_presence", ""),
                                             sha256_file(mask_path) == row.get("mask_sha256", ""))
            if validation["passed"]:
                valid_approved.append(row)
            else:
                technical_failures.append({"image_id": row["image_id"], "note": validation["note"]})
        except (OSError, ValueError) as exc:
            technical_failures.append({"image_id": row["image_id"], "note": f"{type(exc).__name__}:{exc}"})
    if valid_approved:
        metrics, paired = _paired_predictions(root, valid_approved)
        _write_case_sheet(root, _case_rows(root, valid_approved, paired, True), SUCCESS_SHEET_PATH,
                          "Independent bruise GT — cases with detection / correct absence")
        _write_case_sheet(root, _case_rows(root, valid_approved, paired, False), FAILURE_SHEET_PATH,
                          "Independent bruise GT — missed / false positive / unavailable")
    else:
        metrics, paired = [], {}
        _write_pending_sheets(root)
    summaries = _summaries(metrics, candidates)
    _write_csv(root / METRICS_PATH, metrics, METRIC_FIELDS)
    _write_csv(root / SUMMARY_PATH, summaries, SUMMARY_FIELDS)
    _write_comparison_chart(root, summaries, len(valid_approved))
    report_path = _report(root, candidates, metrics, summaries, technical_failures)
    counts = _annotation_status_counts(candidates)
    return {**counts, "eligible_for_evaluation": len(valid_approved), "paired_metric_rows": len(metrics),
            "technical_failures": technical_failures, "summary": SUMMARY_PATH, "metrics": METRICS_PATH,
            "comparison_chart": CHART_PATH, "success_contact_sheet": SUCCESS_SHEET_PATH,
            "failure_contact_sheet": FAILURE_SHEET_PATH, "report": str(report_path.relative_to(root))}


def _case_rows(root: Path, approved: list[dict[str, str]], paired: dict[str, dict[str, Any]],
               success: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in approved:
        predictions = paired[row["image_id"]]
        status_ok = all(predictions[cfg]["prediction_available"] for cfg, _, _ in CONFIGURATIONS)
        if row.get("lesion_presence") == "present":
            is_success = status_ok and all(predictions[cfg]["detection_hit"] == "1" for cfg, _, _ in CONFIGURATIONS)
        else:
            is_success = status_ok and all(int(np.count_nonzero(predictions[cfg]["prediction"])) == 0
                                           for cfg, _, _ in CONFIGURATIONS)
        if is_success != success:
            continue
        selected.append({
            **row,
            "lab_prediction_path": str(predictions[CONFIGURATIONS[0][0]]["prediction_path"]),
            "hsv_prediction_path": str(predictions[CONFIGURATIONS[1][0]]["prediction_path"]),
            "lab_iou": f"{predictions[CONFIGURATIONS[0][0]]['iou']:.3f}",
            "hsv_iou": f"{predictions[CONFIGURATIONS[1][0]]['iou']:.3f}",
        })
    return selected


def current_status(root: str | Path) -> dict[str, Any]:
    rows = _manifest_rows(Path(root).resolve())
    counts = _annotation_status_counts(rows)
    eligible = sum(row.get("approval_status") == "approved"
                   and row.get("technical_validation_status") == "passed" for row in rows)
    return {**counts, "eligible_for_evaluation": eligible,
            "pending": sum(row.get("workflow_status") in {"pending", "draft", "candidate"} for row in rows)}

