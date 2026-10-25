from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.color.preprocess import decode_image
from src.validation.bruise_ground_truth import (
    CONFIGURATIONS, METRIC_FIELDS, SUMMARY_FIELDS, _case_rows, _paired_predictions,
    _project_path, _read_csv, _rough_fruit_region, _summaries, _write_case_sheet,
    _write_comparison_chart, _write_csv, import_mask, sha256_file,
)


ASSISTED_MANIFEST = "data/metadata/bruise_gt_assisted_manifest.csv"
ASSISTED_ROOT = "data/annotations/bruise_gt_assisted"
ASSISTED_MASK_DIR = f"{ASSISTED_ROOT}/masks"
CONTACT_ROOT = "reports/bruise_gt_assisted"
APPROVAL_AUDIT_PATH = f"{ASSISTED_ROOT}/approval_audit.jsonl"
ASSISTED_METRICS_PATH = "reports/bruise_gt_assisted_metrics.csv"
ASSISTED_SUMMARY_PATH = "reports/bruise_gt_assisted_summary.csv"
ASSISTED_CHART_PATH = "reports/bruise_gt_assisted_comparison.png"
ASSISTED_REPORT_PATH = "reports/bruise_gt_assisted_evaluation.md"
ASSISTED_SUCCESS_SHEET_PATH = "reports/bruise_gt_assisted_success_cases.png"
ASSISTED_FAILURE_SHEET_PATH = "reports/bruise_gt_assisted_failure_cases.png"
ASSISTED_PREDICTION_DIR = "data/processed/bruise_gt_assisted_kmeans_predictions"
ASSISTED_FRUIT_MASK_DIR = "data/processed/bruise_gt_assisted_fruit_masks"
ASSISTED_FRUIT_MANIFEST = "data/metadata/bruise_gt_assisted_fruit_masks.csv"
FRUIT_OVERLAP_WARNING_FRACTION = 0.02
ASSISTED_METRIC_FIELDS = [*METRIC_FIELDS, "annotation_origin", "reviewer", "warning_override", "approval_batch_id"]
ASSISTED_SUMMARY_FIELDS = [*SUMMARY_FIELDS, "annotation_origin", "warning_override_count"]
METHOD = "rgb_to_lab_hsv_local_dark_brown_v1"
ASSISTED_FIELDS = [
    "selection_order", "image_id", "dataset", "source_image_path", "source_sha256",
    "source_width", "source_height", "candidate_mask_path", "candidate_mask_sha256",
    "mask_width", "mask_height", "mask_values", "technical_validation_status",
    "technical_validation_note", "technical_validation_warnings", "outside_fruit_fraction",
    "annotation_origin", "review_status", "annotation_status", "lesion_presence", "uncertain",
    "warning_override", "approval_batch_id", "human_review_notes", "method", "method_parameters_json",
    "candidate_area_fraction", "component_count", "generated_at", "visual_review_status",
    "visual_review_note", "edited_by", "edited_at", "reviewer", "reviewed_at",
]


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), array):
        raise OSError(f"Could not write mask PNG: {path}")
    temporary.replace(path)


def _fruit_region(rgb: np.ndarray) -> np.ndarray:
    """Coarse edge-connected light-background removal; not a ground-truth mask."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    brightness = small.astype(np.uint16).mean(axis=2)
    chroma = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
    light_background = ((brightness >= 188) & (chroma <= 48)).astype(np.uint8)
    count, labels = cv2.connectedComponents(light_background, 8)
    edge = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    bg_labels = np.unique(edge)
    bg_labels = bg_labels[bg_labels != 0]
    background = np.isin(labels, bg_labels) if bg_labels.size else np.zeros(labels.shape, dtype=bool)
    foreground = (~background).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, component_labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        foreground = (component_labels == largest).astype(np.uint8)
    if foreground.shape != (h, w):
        foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_NEAREST)
    return foreground > 0


def _proposal(rgb: np.ndarray, max_side: int = 2400) -> tuple[np.ndarray, dict[str, Any]]:
    """Propose dark/brown surface regions using only the source RGB image."""
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    work = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()
    roi = _fruit_region(work)
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    luminance = lab[:, :, 0].astype(np.float32)
    local_background = cv2.GaussianBlur(luminance, (0, 0), sigmaX=9.0, sigmaY=9.0)
    local_darkness = local_background - luminance
    a_channel = lab[:, :, 1].astype(np.int16)
    b_channel = lab[:, :, 2].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)

    # Dark pigment / black spots plus locally darker, red-brown peel regions.
    very_dark = (luminance <= 84) & ((local_darkness >= 3) | (luminance <= 62))
    brown_local = ((luminance <= 177) & (local_darkness >= 9)
                   & (a_channel >= 130) & (b_channel >= 124) & (saturation >= 22))
    brown_absolute = ((luminance <= 142) & (a_channel >= 137)
                      & (b_channel >= 125) & (saturation >= 35))
    proposal = ((very_dark | brown_local | brown_absolute) & roi).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    proposal = cv2.morphologyEx(proposal, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    if scale < 1:
        proposal = cv2.resize(proposal, (w, h), interpolation=cv2.INTER_NEAREST)
    proposal = np.where(proposal > 0, 255, 0).astype(np.uint8)
    area = int(np.count_nonzero(proposal))
    count, _, stats, _ = cv2.connectedComponentsWithStats((proposal > 0).astype(np.uint8), 8)
    components = int(count - 1)
    roi_pixels = int(np.count_nonzero(roi))
    info = {
        "method": METHOD,
        "parameters": {
            "max_work_side": int(max_side), "local_gaussian_sigma_work_px": 9.0,
            "very_dark_lab_l_u8_max": 84, "brown_local_lab_l_u8_max": 177,
            "brown_local_contrast_min": 9, "brown_local_lab_a_u8_min": 130,
            "brown_local_lab_b_u8_min": 124, "brown_local_hsv_s_u8_min": 22,
            "brown_absolute_lab_l_u8_max": 142, "brown_absolute_lab_a_u8_min": 137,
            "brown_absolute_lab_b_u8_min": 125, "brown_absolute_hsv_s_u8_min": 35,
            "morphology": "single 3x3 elliptical close; no minimum component size",
            "fruit_roi": "edge-connected light low-chroma background removal",
        },
        "candidate_area_fraction_of_image": area / float(h * w),
        "candidate_area_fraction_of_fruit_roi": area / float(max(roi_pixels, 1)),
        "component_count": components,
        "largest_component_area_fraction": (float(stats[1:, cv2.CC_STAT_AREA].max()) / max(area, 1)
                                             if components else 0.0),
        "work_size": [int(work.shape[1]), int(work.shape[0])],
    }
    return proposal, info


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32)
    active = mask > 0
    tint = np.array([255.0, 35.0, 145.0], dtype=np.float32)
    base[active] = 0.48 * base[active] + 0.52 * tint
    output = np.clip(base, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 235, 0), max(1, round(max(rgb.shape[:2]) / 1800)))
    return output


def _validate_candidate(rgb: np.ndarray, mask: np.ndarray,
                        lesion_presence: str | None = None) -> dict[str, Any]:
    """Validate hard mask constraints; rough fruit-region overlap is warning-only."""
    errors: list[str] = []
    warnings: list[dict[str, Any]] = []
    if mask.ndim != 2 or mask.dtype != np.uint8:
        return {"passed": False, "status": "failed", "note": "mask_must_be_single_channel_uint8",
                "outside_fraction": "", "warnings": warnings, "errors": ["mask_must_be_single_channel_uint8"]}
    if mask.shape != rgb.shape[:2]:
        errors.append("mask_dimensions_do_not_match_source")
    is_binary = bool(np.isin(mask, (0, 255)).all())
    if not is_binary:
        errors.append("mask_values_must_be_0_or_255")
    positives = int(np.count_nonzero(mask == 255))
    if lesion_presence == "present" and positives == 0:
        errors.append("present_lesion_requires_nonempty_mask")
    if lesion_presence == "absent" and positives > 0:
        errors.append("absent_lesion_requires_empty_mask")
    outside_fraction = 0.0
    if mask.shape == rgb.shape[:2] and positives and is_binary:
        fruit_region = _rough_fruit_region(rgb)
        outside_fraction = float(np.count_nonzero((mask > 0) & ~fruit_region) / positives)
        if outside_fraction > FRUIT_OVERLAP_WARNING_FRACTION:
            warnings.append({
                "code": "mask_outside_plausible_fruit_region",
                "outside_fraction": round(outside_fraction, 6),
                "threshold": FRUIT_OVERLAP_WARNING_FRACTION,
                "note": "Rough fruit-region heuristic is not ground truth; confirm explicitly if this mask is retained.",
            })
    if lesion_presence not in (None, "", "not_reviewed", "present", "absent", "uncertain"):
        errors.append("invalid_lesion_presence")
    return {
        "passed": not errors,
        "status": "failed" if errors else ("passed_with_warnings" if warnings else "passed"),
        "note": ";".join(errors) if errors else (";".join(w["code"] for w in warnings)
                                                    if warnings else "binary and source-aligned"),
        "outside_fraction": f"{outside_fraction:.6f}" if positives else "",
        "warnings": warnings,
        "errors": errors,
    }


def _write_contact_sheets(root: Path, rows: list[dict[str, Any]], batch_size: int = 10) -> list[str]:
    contact_dir = root / CONTACT_ROOT
    contact_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start:start + batch_size]
        columns, tile_w, tile_h = 2, 720, 335
        sheet = Image.new("RGB", (columns * tile_w, 48 + ((len(batch) + columns - 1) // columns) * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 12), f"Dataset B bruise candidate overlays — batch {batch_index}/{(len(rows)+batch_size-1)//batch_size}", fill="black")
        for local_index, row in enumerate(batch):
            order = start + local_index
            frame = decode_image(_project_path(root, row["source_image_path"]))
            mask = cv2.imread(str(_project_path(root, row["candidate_mask_path"])), cv2.IMREAD_GRAYSCALE)
            overlay = _overlay(frame.rgb, mask)
            panels = [Image.fromarray(frame.rgb), Image.fromarray(overlay)]
            panel_w, panel_h = 350, 270
            tile_x = (local_index % columns) * tile_w
            tile_y = 48 + (local_index // columns) * tile_h
            for panel_index, image in enumerate(panels):
                image.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
                x = tile_x + 5 + panel_index * 355 + (panel_w - image.width) // 2
                y = tile_y + 3
                sheet.paste(image, (x, y))
            area = float(row.get("candidate_area_fraction") or 0.0) * 100.0
            note = row.get("visual_review_note", "")
            label = (f"{order + 1:02d} {row['image_id'][:8]}  "
                     f"[{row.get('annotation_status', '')}]  area={area:.2f}%")
            draw.text((tile_x + 6, tile_y + 277), label[:100], fill="black")
            draw.text((tile_x + 6, tile_y + 296), note[:100], fill="#444")
        target = contact_dir / f"bruise_gt_assisted_contact_sheet_{batch_index:02d}.png"
        sheet.save(target)
        outputs.append(target.relative_to(root).as_posix())
    return outputs


def generate_assisted_masks(root: str | Path, max_side: int = 2400, force: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    manual_rows = _read_csv(root / "data/metadata/bruise_gt_manifest.csv")
    if len(manual_rows) != 30:
        raise ValueError(f"Expected the locked 30-image manifest; found {len(manual_rows)}")
    manifest_path = root / ASSISTED_MANIFEST
    existing = {row.get("image_id", ""): row for row in _read_csv(manifest_path)}
    if manifest_path.is_file() and not force:
        edited_or_reviewed = [row.get("image_id", "") for row in existing.values()
                              if row.get("edited_by") or row.get("review_status") != "pending"
                              or row.get("visual_review_status") == "codex_visual_reviewed"]
        if edited_or_reviewed:
            raise ValueError("Assisted candidates contain edits/reviews; use --force only after saving a copy")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for source_row in manual_rows:
        image_id = source_row["image_id"]
        source_path = _project_path(root, source_row["source_image_path"])
        try:
            if not source_path.is_file() or sha256_file(source_path) != source_row.get("source_sha256", ""):
                raise ValueError("source checksum mismatch or missing file")
            frame = decode_image(source_path)
            mask, info = _proposal(frame.rgb, max_side=max_side)
            validation = _validate_candidate(frame.rgb, mask)
            if not validation["passed"]:
                raise ValueError("Technical validation failed: " + validation["note"])
            relative_mask = f"{ASSISTED_MASK_DIR}/{image_id}.png"
            target_mask = root / relative_mask
            _atomic_png(target_mask, mask)
            row = {
                "selection_order": source_row.get("selection_order", ""), "image_id": image_id,
                "dataset": source_row.get("dataset", ""), "source_image_path": source_row["source_image_path"],
                "source_sha256": source_row["source_sha256"], "source_width": frame.rgb.shape[1],
                "source_height": frame.rgb.shape[0], "candidate_mask_path": relative_mask,
                "candidate_mask_sha256": sha256_file(target_mask), "mask_width": frame.rgb.shape[1],
                "mask_height": frame.rgb.shape[0], "mask_values": "0/255",
                "technical_validation_status": validation["status"],
                "technical_validation_note": validation["note"],
                "technical_validation_warnings": json.dumps(validation["warnings"], ensure_ascii=False),
                "outside_fruit_fraction": validation["outside_fraction"],
                "annotation_origin": "codex_assisted", "review_status": "pending",
                "annotation_status": "generated_pending_visual_review", "uncertain": "false",
                "lesion_presence": "not_reviewed",
                "warning_override": "[]", "approval_batch_id": "", "human_review_notes": "",
                "method": info["method"], "method_parameters_json": json.dumps(info["parameters"], separators=(",", ":")),
                "candidate_area_fraction": f"{info['candidate_area_fraction_of_image']:.8f}",
                "component_count": info["component_count"], "generated_at": generated_at,
                "visual_review_status": "not_reviewed", "visual_review_note": "",
                "edited_by": "", "edited_at": "", "reviewer": "", "reviewed_at": "",
            }
            rows.append(row)
        except Exception as exc:
            failures.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
            rows.append({
                "selection_order": source_row.get("selection_order", ""), "image_id": image_id,
                "dataset": source_row.get("dataset", ""), "source_image_path": source_row.get("source_image_path", ""),
                "source_sha256": source_row.get("source_sha256", ""), "annotation_origin": "codex_assisted",
                "review_status": "pending", "annotation_status": "generation_failed", "uncertain": "true",
                "lesion_presence": "not_reviewed",
                "method": METHOD, "method_parameters_json": "", "candidate_mask_path": "",
                "candidate_mask_sha256": "", "mask_width": "", "mask_height": "", "mask_values": "",
                "technical_validation_status": "failed", "technical_validation_note": failures[-1]["error"],
                "technical_validation_warnings": "[]", "outside_fruit_fraction": "",
                "warning_override": "[]", "approval_batch_id": "", "human_review_notes": "",
                "candidate_area_fraction": "", "component_count": "", "generated_at": generated_at,
                "visual_review_status": "not_reviewed", "visual_review_note": failures[-1]["error"],
                "edited_by": "", "edited_at": "", "reviewer": "", "reviewed_at": "",
            })
    _write_csv(manifest_path, rows, ASSISTED_FIELDS)
    contacts = _write_contact_sheets(root, [row for row in rows if row.get("candidate_mask_path")])
    counts = Counter(row.get("annotation_status", "") for row in rows)
    return {
        "candidate_images": len(rows), "masks_generated_successfully": sum(bool(row.get("candidate_mask_path")) for row in rows),
        "generation_failures": failures, "annotation_status_counts": dict(counts),
        "review_status_counts": dict(Counter(row.get("review_status", "") for row in rows)),
        "manifest": ASSISTED_MANIFEST, "contact_sheets": contacts,
        "annotation_origin": "codex_assisted", "review_status": "pending for every generated candidate",
        "method": METHOD,
    }


def update_visual_review(root: str | Path, reviews: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Record Codex visual review notes without changing pending review status."""
    root = Path(root).resolve()
    path = root / ASSISTED_MANIFEST
    rows = _read_csv(path)
    by_id = {row.get("image_id", ""): row for row in rows}
    unknown = set(reviews) - set(by_id)
    if unknown:
        raise ValueError("Unknown assisted candidate ID(s): " + ", ".join(sorted(unknown)))
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for image_id, (status, note) in reviews.items():
        if status not in {"ready_for_review", "needs_edit", "uncertain", "generation_failed"}:
            raise ValueError(f"Invalid annotation status: {status}")
        row = by_id[image_id]
        row["visual_review_status"] = "codex_visual_reviewed"
        row["visual_review_note"] = note
        row["annotation_status"] = status
        row["uncertain"] = "true" if status in {"uncertain", "generation_failed"} else "false"
        row["review_status"] = "pending"
        row["reviewer"] = ""
        row["reviewed_at"] = ""
    _write_csv(path, rows, ASSISTED_FIELDS)
    contacts = _write_contact_sheets(root, [row for row in rows if row.get("candidate_mask_path")])
    return {"visual_reviewed": len(reviews), "review_status": "pending", "contact_sheets": contacts}


def validate_assisted_masks(root: str | Path) -> dict[str, Any]:
    """Validate checksums, native dimensions, binary values, and rough fruit overlap."""
    root = Path(root).resolve()
    manifest_path = root / ASSISTED_MANIFEST
    rows = _read_csv(manifest_path)
    if not rows:
        raise FileNotFoundError(manifest_path)
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    passed = 0
    for row in rows:
        if not row.get("lesion_presence"):
            row["lesion_presence"] = "not_reviewed"
        try:
            if row.get("annotation_origin") != "codex_assisted":
                raise ValueError("unexpected annotation_origin")
            source_path = _project_path(root, row["source_image_path"])
            mask_path = _project_path(root, row["candidate_mask_path"])
            if sha256_file(source_path) != row.get("source_sha256"):
                raise ValueError("source checksum mismatch")
            if sha256_file(mask_path) != row.get("candidate_mask_sha256"):
                raise ValueError("candidate mask checksum mismatch")
            rgb = decode_image(source_path).rgb
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("candidate mask PNG unreadable")
            if row.get("source_width") != str(rgb.shape[1]) or row.get("source_height") != str(rgb.shape[0]):
                raise ValueError("manifest source dimensions mismatch")
            if row.get("mask_width") != str(mask.shape[1]) or row.get("mask_height") != str(mask.shape[0]):
                raise ValueError("manifest mask dimensions mismatch")
            if row.get("mask_values") != "0/255":
                raise ValueError("manifest binary encoding mismatch")
            validation = _validate_candidate(rgb, mask, row.get("lesion_presence"))
            row["technical_validation_status"] = validation["status"]
            row["technical_validation_note"] = validation["note"]
            row["technical_validation_warnings"] = json.dumps(validation["warnings"], ensure_ascii=False)
            row["outside_fruit_fraction"] = validation["outside_fraction"]
            if not validation["passed"]:
                raise ValueError(validation["note"])
            passed += 1
            if validation["warnings"]:
                warnings.append({"image_id": row.get("image_id", ""), "warnings": validation["warnings"]})
        except Exception as exc:
            row["technical_validation_status"] = "failed"
            row["technical_validation_note"] = f"{type(exc).__name__}: {exc}"
            failures.append({"image_id": row.get("image_id", ""), "error": row["technical_validation_note"]})
    _write_csv(manifest_path, rows, ASSISTED_FIELDS)
    return {"checked": len(rows), "passed": passed, "failed": len(failures), "failures": failures,
            "warning_count": sum(len(item["warnings"]) for item in warnings), "warnings": warnings,
            "lesion_presence_unresolved": sum(row.get("lesion_presence") not in {"present", "absent"} for row in rows),
            "review_status_unchanged": True, "manifest": ASSISTED_MANIFEST}


def preview_assisted_approval(root: str | Path, image_ids: list[str],
                              lesion_presence: str | None = None,
                              presence_by_id: dict[str, str] | None = None) -> dict[str, Any]:
    """Read-only preflight for an exact batch; reports hard blockers and overridable warnings."""
    root = Path(root).resolve()
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs")
    if lesion_presence not in (None, "present", "absent"):
        raise ValueError("Batch presence override must be present or absent")
    presence_by_id = presence_by_id or {}
    if lesion_presence and presence_by_id:
        raise ValueError("Use either a common lesion_presence or a per-image presence map, not both")
    if set(presence_by_id) - set(image_ids):
        raise ValueError("Presence map contains image IDs outside the selected batch")
    if any(value not in {"present", "absent", "uncertain", "not_reviewed"}
           for value in presence_by_id.values()):
        raise ValueError("Presence map values must be present, absent, uncertain, or not_reviewed")
    rows = _read_csv(root / ASSISTED_MANIFEST)
    by_id = {row.get("image_id", ""): row for row in rows}
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown assisted image ID(s): " + ", ".join(sorted(unknown)))

    items: list[dict[str, Any]] = []
    for image_id in image_ids:
        row = by_id[image_id]
        blockers: list[str] = []
        if row.get("review_status") != "pending":
            blockers.append(f"review_status_is_{row.get('review_status') or 'blank'}")
        if row.get("annotation_origin") != "codex_assisted":
            blockers.append("annotation_origin_is_not_codex_assisted")
        presence = presence_by_id.get(image_id, lesion_presence or row.get("lesion_presence", "not_reviewed"))
        if presence not in {"present", "absent"}:
            blockers.append("lesion_presence_unresolved")
        warnings: list[dict[str, Any]] = []
        mask_pixels = 0
        try:
            source_path = _project_path(root, row.get("source_image_path", ""))
            mask_value = row.get("candidate_mask_path", "")
            if not mask_value:
                raise ValueError("candidate_mask_path_missing")
            mask_path = _project_path(root, mask_value)
            if not source_path.is_file():
                blockers.append("source_image_missing")
            elif sha256_file(source_path) != row.get("source_sha256", ""):
                blockers.append("source_checksum_mismatch")
            if not mask_path.is_file():
                blockers.append("candidate_mask_missing")
            elif sha256_file(mask_path) != row.get("candidate_mask_sha256", ""):
                blockers.append("candidate_mask_checksum_mismatch")
            if source_path.is_file() and mask_path.is_file():
                frame = decode_image(source_path)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    blockers.append("candidate_mask_decode_failed")
                else:
                    mask_pixels = int(np.count_nonzero(mask))
                    if frame.rgb.shape[1] != int(row.get("source_width") or -1) or frame.rgb.shape[0] != int(row.get("source_height") or -1):
                        blockers.append("manifest_source_dimensions_mismatch")
                    if row.get("mask_width") != str(mask.shape[1]) or row.get("mask_height") != str(mask.shape[0]):
                        blockers.append("manifest_mask_dimensions_mismatch")
                    if row.get("mask_values") != "0/255":
                        blockers.append("manifest_binary_encoding_mismatch")
                    validation = _validate_candidate(frame.rgb, mask, presence)
                    if not validation["passed"]:
                        blockers.extend(validation["errors"])
                    warnings = validation["warnings"]
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(f"{type(exc).__name__}:{exc}")
        items.append({
            "image_id": image_id, "selection_order": row.get("selection_order", ""),
            "source_image_path": row.get("source_image_path", ""),
            "candidate_mask_path": row.get("candidate_mask_path", ""),
            "lesion_presence_current": row.get("lesion_presence", "not_reviewed"),
            "lesion_presence_for_approval": presence, "mask_pixels": mask_pixels,
            "annotation_status": row.get("annotation_status", ""),
            "review_status": row.get("review_status", ""), "warnings": warnings,
            "blockers": sorted(set(blockers)),
        })
    warning_ids = [item["image_id"] for item in items if item["warnings"]]
    blocker_ids = [item["image_id"] for item in items if item["blockers"]]
    return {
        "selected_count": len(items), "image_ids": image_ids,
        "ready_count": len(items) - len(blocker_ids), "blocked_count": len(blocker_ids),
        "warning_count": sum(len(item["warnings"]) for item in items),
        "warning_image_ids": warning_ids, "blocker_image_ids": blocker_ids,
        "items": items,
    }


def approve_assisted_batch(root: str | Path, image_ids: list[str], reviewer: str,
                           confirmation_phrase: str,
                           lesion_presence: str | None = None,
                           presence_by_id: dict[str, str] | None = None,
                           accepted_warning_ids: list[str] | None = None,
                           notes: str = "", refresh_contact_sheets: bool = True) -> dict[str, Any]:
    """Approve exactly a preflighted human-reviewed batch and record its audit event."""
    root = Path(root).resolve()
    if not reviewer.strip():
        raise ValueError("Reviewer identity is required")
    preview = preview_assisted_approval(root, image_ids, lesion_presence, presence_by_id)
    if preview["blocked_count"]:
        raise ValueError("Approval blocked; resolve all hard blockers first: "
                         + json.dumps(preview["items"], ensure_ascii=False))
    accepted_warning_ids = accepted_warning_ids or []
    if len(set(accepted_warning_ids)) != len(accepted_warning_ids):
        raise ValueError("Warning override IDs must be duplicate-free")
    if set(accepted_warning_ids) != set(preview["warning_image_ids"]):
        if preview["warning_image_ids"]:
            raise ValueError("Explicitly accept exactly the warning image IDs shown in the preflight")
        if accepted_warning_ids:
            raise ValueError("No warnings exist to override")
    if confirmation_phrase != f"APPROVE {len(image_ids)} ASSISTED MASKS":
        raise ValueError(f"Final confirmation must exactly equal: APPROVE {len(image_ids)} ASSISTED MASKS")

    manifest_path = root / ASSISTED_MANIFEST
    rows = _read_csv(manifest_path)
    by_id = {row.get("image_id", ""): row for row in rows}
    item_by_id = {item["image_id"]: item for item in preview["items"]}
    batch_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    batch_id = hashlib.sha256((batch_timestamp + "|" + "|".join(image_ids) + "|" + reviewer).encode()).hexdigest()[:16]
    for image_id in image_ids:
        row = by_id[image_id]
        item = item_by_id[image_id]
        row["review_status"] = "approved"
        row["annotation_status"] = "approved"
        row["lesion_presence"] = item["lesion_presence_for_approval"]
        row["uncertain"] = "false"
        row["annotation_origin"] = "codex_assisted"
        row["reviewer"] = reviewer.strip()
        row["reviewed_at"] = batch_timestamp
        row["human_review_notes"] = notes
        row["approval_batch_id"] = batch_id
        warning_payload = item["warnings"] if image_id in accepted_warning_ids else []
        row["warning_override"] = json.dumps(warning_payload, ensure_ascii=False, separators=(",", ":"))
        row["technical_validation_warnings"] = json.dumps(item["warnings"], ensure_ascii=False, separators=(",", ":"))
        row["technical_validation_status"] = "passed_with_warnings" if item["warnings"] else "passed"
        row["technical_validation_note"] = ";".join(warning["code"] for warning in item["warnings"]) or "binary, source-aligned, and presence-consistent"
    event = {
        "event": "assisted_mask_batch_approved", "batch_id": batch_id, "timestamp": batch_timestamp,
        "annotation_origin": "codex_assisted", "reviewer": reviewer.strip(), "image_ids": image_ids,
        "lesion_presence_by_image": {item["image_id"]: item["lesion_presence_for_approval"] for item in preview["items"]},
        "warning_count": preview["warning_count"], "warning_image_ids": preview["warning_image_ids"],
        "warning_overrides": [{"image_id": item["image_id"], "warnings": item["warnings"]}
                              for item in preview["items"] if item["image_id"] in accepted_warning_ids],
        "final_confirmation": confirmation_phrase, "notes": notes,
    }
    original_rows = _read_csv(manifest_path)
    _write_csv(manifest_path, rows, ASSISTED_FIELDS)
    audit_path = root / APPROVAL_AUDIT_PATH
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
    except OSError:
        _write_csv(manifest_path, original_rows, ASSISTED_FIELDS)
        raise
    contacts = (_write_contact_sheets(root, [row for row in rows if row.get("candidate_mask_path")])
                if refresh_contact_sheets else [])
    return {"approved_count": len(image_ids), "image_ids": image_ids, "reviewer": reviewer.strip(),
            "annotation_origin": "codex_assisted", "batch_id": batch_id,
            "warning_overrides": event["warning_overrides"], "contact_sheets": contacts,
            "audit_log": APPROVAL_AUDIT_PATH, "review_status": "approved"}


def review_assisted_candidates(root: str | Path, image_ids: list[str], status: str,
                               reviewer: str, notes: str = "",
                               lesion_presence: str | None = None,
                               refresh_contact_sheets: bool = True,
                               confirmation_phrase: str | None = None,
                               accepted_warning_ids: list[str] | None = None) -> dict[str, Any]:
    """Apply an explicit human status decision to an exact ID list."""
    root = Path(root).resolve()
    if status not in {"pending", "approved", "rejected", "uncertain"}:
        raise ValueError("Status must be pending, approved, rejected, or uncertain")
    if not reviewer.strip():
        raise ValueError("Reviewer identity is required")
    if status in {"rejected", "uncertain"} and not notes.strip():
        raise ValueError("Add a short note when rejecting or marking uncertain")
    if lesion_presence is not None and lesion_presence not in {"present", "absent", "uncertain", "not_reviewed"}:
        raise ValueError("lesion_presence must be present, absent, uncertain, or not_reviewed")
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs")
    manifest_path = root / ASSISTED_MANIFEST
    rows = _read_csv(manifest_path)
    by_id = {row.get("image_id", ""): row for row in rows}
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown assisted image ID(s): " + ", ".join(sorted(unknown)))
    if status == "approved":
        return approve_assisted_batch(
            root, image_ids, reviewer, confirmation_phrase or "", lesion_presence=lesion_presence,
            accepted_warning_ids=accepted_warning_ids, notes=notes,
            refresh_contact_sheets=refresh_contact_sheets,
        )
    for image_id in image_ids:
        row = by_id[image_id]
        if not row.get("candidate_mask_path"):
            raise ValueError(f"No candidate mask exists for {image_id}")
        if status == "approved":
            presence = lesion_presence or row.get("lesion_presence", "not_reviewed")
            if presence not in {"present", "absent"}:
                raise ValueError(f"Set lesion_presence to present or absent before approving {image_id}")
            source_path = _project_path(root, row["source_image_path"])
            mask_path = _project_path(root, row["candidate_mask_path"])
            if sha256_file(source_path) != row.get("source_sha256"):
                raise ValueError(f"Source checksum mismatch for {image_id}")
            if sha256_file(mask_path) != row.get("candidate_mask_sha256"):
                raise ValueError(f"Candidate checksum mismatch for {image_id}")
            rgb = decode_image(source_path).rgb
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != rgb.shape[:2]:
                raise ValueError(f"Candidate mask shape invalid for {image_id}")
            validation = _validate_candidate(rgb, mask)
            if not validation["passed"]:
                raise ValueError(f"Candidate mask technical validation failed for {image_id}: {validation['note']}")
            positives = int(np.count_nonzero(mask))
            if presence == "present" and positives == 0:
                raise ValueError(f"A present annotation needs a nonempty mask: {image_id}")
            if presence == "absent" and positives:
                raise ValueError(f"An absent annotation needs an empty mask: {image_id}")
            row["lesion_presence"] = presence
        elif lesion_presence is not None:
            row["lesion_presence"] = lesion_presence
        if status == "uncertain":
            row["lesion_presence"] = "uncertain"
        row["review_status"] = status
        row["reviewer"] = reviewer
        row["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds") if status != "pending" else ""
        row["human_review_notes"] = notes or row.get("human_review_notes", "")
        row["uncertain"] = "true" if status == "uncertain" else "false"
        if status in {"approved", "rejected", "uncertain"}:
            row["annotation_status"] = status
    _write_csv(manifest_path, rows, ASSISTED_FIELDS)
    contacts = (_write_contact_sheets(root, [row for row in rows if row.get("candidate_mask_path")])
                if refresh_contact_sheets else [])
    return {"updated": len(image_ids), "image_ids": image_ids, "review_status": status,
            "reviewer": reviewer, "contact_sheets": contacts}


def import_human_approved_assisted(root: str | Path, image_ids: list[str], notes: str = "") -> dict[str, Any]:
    """Copy explicitly human-approved assisted masks into the manual GT queue as pending candidates."""
    root = Path(root).resolve()
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Provide an explicit, duplicate-free list of image IDs")
    rows = _read_csv(root / ASSISTED_MANIFEST)
    by_id = {row.get("image_id", ""): row for row in rows}
    unknown = set(image_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown assisted image ID(s): " + ", ".join(sorted(unknown)))
    imported = []
    for image_id in image_ids:
        row = by_id[image_id]
        if row.get("review_status") != "approved":
            raise ValueError(f"Assisted mask has not been explicitly human-approved: {image_id}")
        if row.get("annotation_origin") != "codex_assisted":
            raise ValueError(f"Unexpected annotation origin for {image_id}")
        mask_path = _project_path(root, row["candidate_mask_path"])
        if sha256_file(mask_path) != row.get("candidate_mask_sha256"):
            raise ValueError(f"Candidate checksum mismatch for {image_id}")
        presence = row.get("lesion_presence", "not_reviewed")
        if presence not in {"present", "absent"}:
            raise ValueError(f"Human review must resolve lesion presence before import: {image_id}")
        result = import_mask(
            root, image_id, mask_path, presence,
            reviewer_notes=(notes.strip() + " " if notes.strip() else "")
                           + "Imported from explicitly human-approved codex_assisted candidate; manual GT approval remains pending.",
        )
        imported.append(result)
    return {"imported_as_pending_manual_candidates": len(imported), "results": imported,
            "next_step": "Use the existing approve command with the exact image IDs after manual review."}


def evaluate_assisted_approved(root: str | Path, expected_count: int = 30) -> dict[str, Any]:
    """Run the locked paired Phase 3 configurations on explicitly approved assisted masks only."""
    root = Path(root).resolve()
    rows = _read_csv(root / ASSISTED_MANIFEST)
    approved = [row for row in rows if row.get("review_status") == "approved"]
    if len(approved) != expected_count:
        raise ValueError(f"Expected {expected_count} approved assisted masks; found {len(approved)}")
    audit_path = root / APPROVAL_AUDIT_PATH
    if not audit_path.is_file():
        raise ValueError("Approval audit log is missing")
    audited_ids: set[str] = set()
    with audit_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "assisted_mask_batch_approved":
                audited_ids.update(event.get("image_ids", []))
    invalid: list[dict[str, str]] = []
    normalized: list[dict[str, str]] = []
    for row in approved:
        image_id = row.get("image_id", "")
        try:
            if row.get("annotation_origin") != "codex_assisted":
                raise ValueError("annotation_origin is not codex_assisted")
            if not row.get("reviewer") or not row.get("reviewed_at"):
                raise ValueError("reviewer/audit timestamp missing")
            if image_id not in audited_ids:
                raise ValueError("image ID not present in approval audit log")
            source_path = _project_path(root, row["source_image_path"])
            mask_path = _project_path(root, row["candidate_mask_path"])
            if sha256_file(source_path) != row.get("source_sha256"):
                raise ValueError("source checksum mismatch")
            if sha256_file(mask_path) != row.get("candidate_mask_sha256"):
                raise ValueError("mask checksum mismatch")
            rgb = decode_image(source_path).rgb
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("mask PNG unreadable")
            presence = row.get("lesion_presence", "")
            validation = _validate_candidate(rgb, mask, presence)
            if mask.shape != (int(row["source_height"]), int(row["source_width"])):
                raise ValueError("mask/source dimensions mismatch")
            if not validation["passed"]:
                raise ValueError(validation["note"])
            overrides = json.loads(row.get("warning_override") or "[]")
            warned_codes = {warning["code"] for warning in validation["warnings"]}
            overridden_codes = {warning.get("code") for warning in overrides}
            if warned_codes - overridden_codes:
                raise ValueError("fruit-overlap warning lacks explicit approval override")
            normalized_row = dict(row)
            normalized_row.update({
                "workflow_status": "approved", "approval_status": "approved",
                "mask_path": row["candidate_mask_path"], "mask_sha256": row["candidate_mask_sha256"],
                "ground_truth_mask_path": row["candidate_mask_path"],
                "ground_truth_mask_sha256": row["candidate_mask_sha256"],
            })
            normalized.append(normalized_row)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            invalid.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
    if invalid:
        raise ValueError("Approved assisted masks failed evaluation preflight: " + json.dumps(invalid, ensure_ascii=False))

    metrics, paired = _paired_predictions(
        root, normalized, prediction_dir=ASSISTED_PREDICTION_DIR,
        generated_fruit_manifest=ASSISTED_FRUIT_MANIFEST, fruit_mask_dir=ASSISTED_FRUIT_MASK_DIR,
    )
    for metric in metrics:
        row = next(row for row in normalized if row["image_id"] == metric["image_id"])
        metric.update({
            "annotation_origin": "codex_assisted", "reviewer": row.get("reviewer", ""),
            "warning_override": row.get("warning_override", "[]"),
            "approval_batch_id": row.get("approval_batch_id", ""),
        })
    summaries = _summaries(metrics, normalized)
    warning_override_count = sum(bool(json.loads(row.get("warning_override") or "[]")) for row in normalized)
    for summary in summaries:
        summary["annotation_origin"] = "codex_assisted"
        summary["warning_override_count"] = warning_override_count

    _write_csv(root / ASSISTED_METRICS_PATH, metrics, ASSISTED_METRIC_FIELDS)
    _write_csv(root / ASSISTED_SUMMARY_PATH, summaries, ASSISTED_SUMMARY_FIELDS)
    _write_comparison_chart(root, summaries, len(normalized), ASSISTED_CHART_PATH)
    _write_case_sheet(root, _case_rows(root, normalized, paired, True), ASSISTED_SUCCESS_SHEET_PATH,
                      "Codex-assisted bruise GT — success cases")
    _write_case_sheet(root, _case_rows(root, normalized, paired, False), ASSISTED_FAILURE_SHEET_PATH,
                      "Codex-assisted bruise GT — failure cases")
    report_lines = [
        "# MangoResearch — Assisted Bruise Ground Truth Evaluation", "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "## Protocol", "",
        "This report evaluates only the 30 explicitly human-approved masks stored with `annotation_origin=codex_assisted`. It is not an independent manual pixel annotation or an agricultural diagnosis. Dataset A/C and external Dataset v2 roles are unchanged.",
        "The comparison is paired on the same approved images using the locked Phase 3 configurations: Original + CIELAB + K=3 and Original + HSV + K=5. No K, threshold, model, split, or cluster-selection rule was tuned. Fruit-region warnings are heuristic only and are recorded with their explicit approval overrides.",
        "Every approved image remains in both configuration denominators. Inference failures are counted as unavailable and score an empty prediction for pixel metrics; coverage is reported separately. The sample is purposive and small, so results describe these 30 images only.", "",
        "## Paired comparison", "",
        "| Configuration | Approved n | Prediction coverage | Mean IoU | Mean Dice | Pixel precision | Pixel recall | Detection rate | FP pixels | FN pixels | No inference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        name = f"Original {summary['color_space']} K={summary['k']}"
        coverage = f"{summary['n_prediction_available']}/{summary['n_evaluable']}"
        values = [summary["n_approved"], coverage, summary["mean_iou"], summary["mean_dice"],
                  summary["pixel_precision"], summary["pixel_recall"], summary["detection_rate"],
                  summary["false_positive_pixels"], summary["false_negative_pixels"], summary["n_no_inference"]]
        rendered = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        report_lines.append("| " + " | ".join([name, *rendered]) + " |")
    report_lines.extend([
        "", "## Outputs", "",
        f"- Metrics: `{ASSISTED_METRICS_PATH}`", f"- Summary: `{ASSISTED_SUMMARY_PATH}`",
        f"- Comparison chart: `{ASSISTED_CHART_PATH}`", f"- Success cases: `{ASSISTED_SUCCESS_SHEET_PATH}`",
        f"- Failure cases: `{ASSISTED_FAILURE_SHEET_PATH}`", f"- Approval audit: `{APPROVAL_AUDIT_PATH}`", "",
        "Mean IoU and Dice are image-level means. Pixel precision and recall, and false-positive/false-negative totals, are pooled across the approved masks. Detection rate uses approved `present` images. Coverage is available predictions divided by the full approved set.",
    ])
    report_path = root / ASSISTED_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "approved_images": len(normalized), "paired_metric_rows": len(metrics),
        "configurations": [{"config_id": config_id, "color_space": color_space, "k": k}
                           for config_id, color_space, k in CONFIGURATIONS],
        "warning_override_count": warning_override_count,
        "summary": ASSISTED_SUMMARY_PATH, "metrics": ASSISTED_METRICS_PATH,
        "comparison_chart": ASSISTED_CHART_PATH, "success_contact_sheet": ASSISTED_SUCCESS_SHEET_PATH,
        "failure_contact_sheet": ASSISTED_FAILURE_SHEET_PATH, "report": ASSISTED_REPORT_PATH,
        "prediction_dir": ASSISTED_PREDICTION_DIR, "fruit_mask_dir": ASSISTED_FRUIT_MASK_DIR,
    }

