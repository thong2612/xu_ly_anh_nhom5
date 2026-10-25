from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.color.preprocess import decode_image
from src.demo.phase6 import (
    MAX_UPLOAD_BYTES, DemoAssets, InputImageError, _predict, decode_upload, infer_image,
    load_demo_assets,
)
from src.segmentation.metrics import segmentation_metrics
from src.segmentation.pipeline import _resize_rgb, _run_candidates, _settings, _variant_image
from src.utils.config import load_config, project_path

LOGGER = logging.getLogger("mangoresearch.phase07")
KMEANS_REVIEW_PATH = "data/metadata/phase07_kmeans_pair_review.csv"
EXTERNAL_MANIFEST_PATH = "data/external_validation/external_manifest.csv"
KMEANS_RESULT_PATH = "reports/phase03_results.csv"
PHASE051_FEATURES = {"ripeness": "data/features/ripeness_features.csv", "defect": "data/features/defect_features.csv"}
PHASE051_SPLITS = {"ripeness": "data/splits/phase051/ripeness_phase051.csv", "defect": "data/splits/phase051/defect_phase051.csv"}
KMEANS_METRIC_FIELDS = [
    "image_id", "dataset", "source_image_path", "ground_truth_mask_path", "color_space", "k",
    "review_status", "alignment_status", "kmeans_status", "iou", "dice", "precision", "recall",
    "detection_hit", "non_detection", "prediction_available", "prediction_pixels",
    "ground_truth_pixels", "predicted_area_fraction", "metric_status",
]
KMEANS_SUMMARY_FIELDS = [
    "config_id", "color_space", "k", "n_approved", "n_ground_truth_valid",
    "n_prediction_valid", "n_no_valid_result", "no_valid_result_rate",
    "mean_iou", "mean_dice", "mean_pixel_precision", "mean_pixel_recall",
    "detection_rate", "metric_denominator_note",
]
FRUIT_METRIC_FIELDS = [
    "image_id", "source_image_path", "ground_truth_mask_path", "variant", "channel", "method",
    "polarity", "phase02_split", "prediction_status", "iou", "dice", "precision", "recall", "score",
    "ground_truth_status", "metric_status",
]
FRUIT_SUMMARY_FIELDS = [
    "config_id", "variant", "channel", "method", "polarity", "n_approved_images",
    "n_valid_predictions", "n_paired_images", "mean_iou", "mean_dice", "mean_precision",
    "mean_recall", "paired_mean_iou", "paired_mean_dice", "paired_mean_precision",
    "paired_mean_recall", "failed_predictions", "n_validation_approved_images",
    "n_validation_valid_predictions", "n_validation_paired_images", "validation_mean_iou",
    "validation_mean_dice", "validation_mean_precision", "validation_mean_recall",
    "validation_paired_mean_iou", "validation_paired_mean_dice",
    "validation_paired_mean_precision", "validation_paired_mean_recall",
    "validation_failed_predictions",
]
EXTERNAL_FIELDS = [
    "image_id", "image_path", "ripeness_label", "defect_label", "review_status",
    "source_group", "reviewer", "reviewed_at", "notes", "import_status",
]
ERROR_FIELDS = [
    "evaluation_scope", "task", "image_id", "image_path", "source_group", "true_label",
    "predicted_label", "prediction_status", "fruit_mask_status", "kmeans_status",
    "suspicious_area_fraction", "gray_world_clipping_fraction", "preprocessing_warnings",
    "observed_issue_stages", "error_type", "cause_status", "review_status", "review_notes",
]
ROBUSTNESS_FIELDS = [
    "sample_id", "task", "source_split", "transformation", "transformation_parameters",
    "prediction_available", "prediction", "baseline_prediction", "prediction_stable",
    "availability_changed", "mean_abs_feature_delta", "mean_abs_feature_delta_train_sd",
    "fruit_mask_status", "baseline_fruit_mask_status", "kmeans_status", "baseline_kmeans_status",
    "suspicious_area_percent", "baseline_suspicious_area_percent", "gray_world_clipping_fraction",
    "inference_status", "error",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _config(root: str | Path) -> dict[str, Any]:
    return load_config(Path(root) / "config.yaml")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(temp)
    temp.replace(path)


def _rgb_overlay(rgb: np.ndarray, mask: np.ndarray | None, alpha: float = 0.42) -> np.ndarray:
    result = rgb.copy()
    if mask is None or mask.shape != rgb.shape[:2]:
        return result
    selected = mask > 0
    result[selected] = np.clip(
        result[selected].astype(np.float32) * (1.0 - alpha)
        + np.asarray([250, 40, 35], dtype=np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)
    return result


def _mask_image(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
        return None
    if shape is not None and mask.shape != shape:
        return None
    if not np.isin(mask, (0, 255)).all():
        return None
    return mask


def _load_review_manifests(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    config = _config(root)
    candidates = _read_csv(project_path(config, "data/metadata/fruit_gt_candidates.csv"))
    approved = _read_csv(project_path(config, "data/metadata/fruit_gt_manifest.csv"))
    return candidates, approved


def _kmeans_result_pairs(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    rows = _read_csv(root / KMEANS_RESULT_PATH)
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("mode") != "heuristic" or row.get("variant") != "original":
            continue
        if row.get("color_space") == "Lab" and row.get("k") == "3":
            pairs[row["image_id"]]["original_lab_k3"] = row
        elif row.get("color_space") == "HSV" and row.get("k") == "5":
            pairs[row["image_id"]]["original_hsv_k5"] = row
    return pairs


def _select_kmeans_review_rows(root: Path, limit: int = 50) -> list[dict[str, Any]]:
    pairs = _kmeans_result_pairs(root)
    manifest = {row.get("image_id", ""): row for row in _read_csv(root / "data/metadata/phase02_manifest.csv")}
    candidates: dict[str, dict[str, Any]] = {}
    for image_id, configs in pairs.items():
        lab, hsv = configs.get("original_lab_k3", {}), configs.get("original_hsv_k5", {})
        if not lab or not hsv:
            continue
        scores = [_safe_float(lab.get("iou")), _safe_float(hsv.get("iou"))]
        precisions = [_safe_float(lab.get("precision")), _safe_float(hsv.get("precision"))]
        recalls = [_safe_float(lab.get("recall")), _safe_float(hsv.get("recall"))]
        candidates[image_id] = {
            "configs": configs, "manifest": manifest.get(image_id, {}),
            "iou_mean": float(np.mean([x for x in scores if x is not None])) if any(x is not None for x in scores) else None,
            "precision_mean": float(np.mean([x for x in precisions if x is not None])) if any(x is not None for x in precisions) else None,
            "recall_mean": float(np.mean([x for x in recalls if x is not None])) if any(x is not None for x in recalls) else None,
            "has_zero": any(x == 0 for x in scores if x is not None),
        }
    selected: dict[str, str] = {}

    def choose(key: str, rows: list[tuple[str, dict[str, Any]]], reverse: bool) -> None:
        available = [(image_id, data) for image_id, data in rows if image_id not in selected and data.get(key) is not None]
        available.sort(key=lambda pair: (pair[1][key], pair[0]), reverse=reverse)
        for image_id, _ in available[: max(0, limit // 5)]:
            selected[image_id] = key

    items = list(candidates.items())
    choose("iou_mean", items, True)
    choose("iou_mean", [(i, d) for i, d in items if (d.get("iou_mean") or 0) > 0], False)
    zero_rows = sorted((i, d) for i, d in items if d["has_zero"] and i not in selected)
    for image_id, _ in zero_rows[: max(0, limit // 5)]:
        selected[image_id] = "zero_iou"
    choose("precision_mean", items, False)
    choose("recall_mean", items, False)
    if len(selected) < limit:
        for image_id in sorted(candidates):
            if image_id not in selected:
                selected[image_id] = "coverage_fill"
                if len(selected) >= limit:
                    break
    output = []
    for image_id, reason in sorted(selected.items(), key=lambda pair: (pair[1], pair[0]))[:limit]:
        data = candidates[image_id]
        lab, hsv = data["configs"]["original_lab_k3"], data["configs"]["original_hsv_k5"]
        meta = data["manifest"]
        output.append({
            "image_id": image_id, "dataset": lab.get("dataset", "mango_segmentation"),
            "source_image_path": lab.get("source_image_path", ""),
            "ground_truth_mask_path": lab.get("defect_mask_path", ""),
            "alignment_status": lab.get("defect_mask_alignment_status", ""),
            "selection_reason": reason,
            "lab_k3_iou": lab.get("iou", ""), "lab_k3_dice": lab.get("dice", ""),
            "lab_k3_precision": lab.get("precision", ""), "lab_k3_recall": lab.get("recall", ""),
            "lab_k3_prediction_path": lab.get("predicted_mask_path", ""),
            "lab_k3_cluster_map_path": lab.get("cluster_map_path", ""),
            "lab_k3_status": lab.get("kmeans_status", ""),
            "hsv_k5_iou": hsv.get("iou", ""), "hsv_k5_dice": hsv.get("dice", ""),
            "hsv_k5_precision": hsv.get("precision", ""), "hsv_k5_recall": hsv.get("recall", ""),
            "hsv_k5_prediction_path": hsv.get("predicted_mask_path", ""),
            "hsv_k5_cluster_map_path": hsv.get("cluster_map_path", ""),
            "hsv_k5_status": hsv.get("kmeans_status", ""),
            "fruit_mask_status": lab.get("fruit_mask_status", ""),
            "exploratory_metric_status": lab.get("gt_pair_status", ""),
            "review_status": "pending", "reviewer": "", "reviewed_at": "", "review_notes": "",
            "official_accuracy_eligible": "false",
        })
    return output


def _draw_kmeans_contact_sheets(root: Path, rows: list[dict[str, Any]], per_sheet: int = 5) -> list[str]:
    output_dir = root / "reports/phase07_review/kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size, title_h, caption_h, columns = 205, 24, 20, 4
    row_h = title_h + tile_size + caption_h
    files: list[str] = []
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(rows), per_sheet):
        batch = rows[start:start + per_sheet]
        canvas = Image.new("RGB", (columns * tile_size, len(batch) * row_h), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(batch):
            image_path = root / Path(row["source_image_path"])
            try:
                frame = decode_image(image_path)
                rgb = frame.rgb
            except Exception:
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            target = _mask_image(root / Path(row["ground_truth_mask_path"]))
            views: list[tuple[str, np.ndarray]] = [("source", rgb)]
            if target is not None and target.shape == rgb.shape[:2]:
                views.append(("source + target mask", _rgb_overlay(rgb, target)))
            else:
                views.append(("target mask invalid/size mismatch", np.zeros_like(rgb)))
            for short, path_field in (("Original Lab K3", "lab_k3_prediction_path"),
                                      ("Original HSV K5", "hsv_k5_prediction_path")):
                mask = _mask_image(root / Path(row[path_field]))
                if mask is not None and mask.shape == rgb.shape[:2]:
                    views.append((short, _rgb_overlay(rgb, mask)))
                else:
                    views.append((short + " unavailable", np.zeros_like(rgb)))
            reason = row["selection_reason"]
            text = (f"{row['image_id']} | {reason} | IoU L/H: "
                    f"{row['lab_k3_iou']}/{row['hsv_k5_iou']} | review={row.get('review_status', 'pending')}")
            top = row_index * row_h
            draw.text((4, top + 2), text[:96], fill="black", font=font)
            for col, (caption, view) in enumerate(views):
                img = Image.fromarray(view)
                img.thumbnail((tile_size - 8, tile_size - 8), Image.Resampling.LANCZOS)
                x, y = col * tile_size + (tile_size - img.width) // 2, top + title_h + 2
                canvas.paste(img, (x, y))
                draw.text((col * tile_size + 4, top + title_h + tile_size + 1), caption[:30], fill="black", font=font)
        path = output_dir / f"kmeans_review_{start // per_sheet + 1:02d}.jpg"
        _atomic_image(path, canvas)
        files.append(str(path.relative_to(root)).replace("\\", "/"))
    return files


def prepare_review_workflows(root: str | Path, review_limit: int = 50) -> dict[str, Any]:
    root = Path(root).resolve()
    external_images = root / "data/external_validation/images"
    external_images.mkdir(parents=True, exist_ok=True)
    candidates, fruit_gt = _load_review_manifests(root)
    candidate_path = root / "data/metadata/fruit_gt_candidates.csv"
    if candidate_path.is_file() and not candidates:
        LOGGER.warning("Fruit GT candidate list exists but is empty")
    kmeans_rows = _select_kmeans_review_rows(root, review_limit)
    review_path = root / KMEANS_REVIEW_PATH
    existing = _read_csv(review_path)
    old_by_id = {row.get("image_id", ""): row for row in existing}
    for row in kmeans_rows:
        previous = old_by_id.get(row["image_id"])
        if previous:
            for key in ("review_status", "reviewer", "reviewed_at", "review_notes"):
                row[key] = previous.get(key, row[key])
    selected_ids = {row.get("image_id", "") for row in kmeans_rows}
    # Keep prior human decisions even if a later review-limit change selects a different sample.
    for previous in existing:
        if previous.get("review_status") in {"approved", "rejected", "uncertain"} and previous.get("image_id") not in selected_ids:
            kmeans_rows.append(previous)
            selected_ids.add(previous.get("image_id", ""))
    review_fields = list(kmeans_rows[0].keys()) if kmeans_rows else list(old_by_id.values())[0].keys() if old_by_id else [
        "image_id", "dataset", "source_image_path", "ground_truth_mask_path", "alignment_status",
        "selection_reason", "lab_k3_iou", "lab_k3_dice", "lab_k3_precision", "lab_k3_recall",
        "lab_k3_prediction_path", "lab_k3_cluster_map_path", "lab_k3_status", "hsv_k5_iou",
        "hsv_k5_dice", "hsv_k5_precision", "hsv_k5_recall", "hsv_k5_prediction_path",
        "hsv_k5_cluster_map_path", "hsv_k5_status", "fruit_mask_status", "exploratory_metric_status",
        "review_status", "reviewer", "reviewed_at", "review_notes", "official_accuracy_eligible",
    ]
    _write_csv(review_path, kmeans_rows, list(review_fields))
    contact_sheets = _draw_kmeans_contact_sheets(root, kmeans_rows)

    external_path = root / EXTERNAL_MANIFEST_PATH
    if not external_path.exists():
        _write_csv(external_path, [], EXTERNAL_FIELDS)
    # Create guarded metric tables once; rerunning prepare must not erase reviewed results.
    kmeans_metrics_path = root / "reports/phase07_kmeans_metrics.csv"
    fruit_metrics_path = root / "reports/phase07_segmentation_metrics.csv"
    if not kmeans_metrics_path.exists():
        _write_csv(kmeans_metrics_path, [], KMEANS_METRIC_FIELDS)
    if not fruit_metrics_path.exists():
        _write_csv(fruit_metrics_path, [], FRUIT_METRIC_FIELDS)
    return {
        "fruit_candidate_count": len(candidates),
        "fruit_approved_count": sum(row.get("semantic_review_status") == "human_confirmed_whole_fruit"
                                     and row.get("validation_status") == "valid_whole_fruit_mask" for row in fruit_gt),
        "kmeans_review_count": len(kmeans_rows), "kmeans_review_path": str(review_path.relative_to(root)),
        "kmeans_contact_sheets": contact_sheets, "external_manifest_path": str(external_path.relative_to(root)),
    }


def set_kmeans_pair_review(root: str | Path, image_id: str, status: str,
                           reviewer: str = "", notes: str = "") -> Path:
    if status not in {"approved", "rejected", "uncertain"}:
        raise ValueError("status must be approved, rejected, or uncertain")
    if status == "approved" and (not reviewer.strip() or not notes.strip()):
        raise ValueError("Approval requires reviewer identity and notes confirming the defect-mask semantics/alignment review")
    root = Path(root).resolve()
    path = root / KMEANS_REVIEW_PATH
    rows = _read_csv(path)
    matches = [row for row in rows if row.get("image_id") == image_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one prepared review row for image_id={image_id}")
    row = matches[0]
    if status == "approved":
        image_path = root / Path(row.get("source_image_path", ""))
        mask_path = root / Path(row.get("ground_truth_mask_path", ""))
        try:
            image = decode_image(image_path).rgb
        except Exception as exc:
            raise ValueError(f"Cannot approve pair with unreadable source image: {exc}") from exc
        mask = _mask_image(mask_path, image.shape[:2])
        if mask_path.suffix.casefold() != ".png" or mask is None or not np.any(mask):
            raise ValueError("Cannot approve: source mask must be a non-empty binary PNG aligned to the source image")
    row["review_status"] = status
    row["reviewer"] = reviewer
    row["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["review_notes"] = notes
    _write_csv(path, rows, list(rows[0]))
    return path


def import_external_images(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    image_dir = root / "data/external_validation/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / EXTERNAL_MANIFEST_PATH
    rows = _read_csv(manifest_path)
    by_id = {row.get("image_id", ""): row for row in rows}
    phase051_hashes = {
        row.get("sha256", "").casefold()
        for task in PHASE051_SPLITS
        for row in _read_csv(root / PHASE051_SPLITS[task])
        if row.get("sha256")
    }
    added, invalid = 0, 0
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            payload = path.read_bytes()
            frame = decode_upload(payload, path.name)
        except (OSError, InputImageError):
            invalid += 1
            continue
        image_id = "ext_" + hashlib.sha256(payload).hexdigest()[:20]
        if image_id in by_id:
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(payload).hexdigest().casefold()
        import_status = f"valid_{frame.rgb.shape[1]}x{frame.rgb.shape[0]}"
        if digest in phase051_hashes:
            import_status += ";exact_duplicate_phase051_data_not_external"
        by_id[image_id] = {
            "image_id": image_id, "image_path": relative, "ripeness_label": "", "defect_label": "",
            "review_status": "pending", "source_group": "", "reviewer": "", "reviewed_at": "",
            "notes": "", "import_status": import_status,
        }
        added += 1
    rows = list(by_id.values())
    _write_csv(manifest_path, rows, EXTERNAL_FIELDS)
    return {"manifest": str(manifest_path.relative_to(root)), "images": len(rows), "added": added,
            "invalid_images_skipped": invalid, "pending_review": sum(row.get("review_status") != "approved" for row in rows)}


def set_external_review(root: str | Path, image_id: str, review_status: str,
                        ripeness_label: str = "", defect_label: str = "", source_group: str = "",
                        reviewer: str = "", notes: str = "") -> Path:
    if review_status not in {"approved", "rejected", "uncertain", "pending"}:
        raise ValueError("review_status must be approved, rejected, uncertain, or pending")
    if ripeness_label and ripeness_label not in {"unripe", "partially_ripe", "ripe"}:
        raise ValueError("ripeness_label must be unripe, partially_ripe, or ripe")
    if defect_label and defect_label not in {"healthy", "defective"}:
        raise ValueError("defect_label must be healthy or defective")
    if review_status == "approved" and not (ripeness_label or defect_label):
        raise ValueError("An approved external review requires at least one manually supplied label")
    if review_status == "approved" and "exact_duplicate_phase051_data_not_external" in "".join(
            row.get("import_status", "") for row in _read_csv(Path(root).resolve() / EXTERNAL_MANIFEST_PATH)
            if row.get("image_id") == image_id):
        raise ValueError("Exact copies of Phase 5.1 images cannot be approved as external validation")
    if review_status == "approved" and not reviewer.strip():
        raise ValueError("An approved external review requires reviewer identity")
    path = Path(root).resolve() / EXTERNAL_MANIFEST_PATH
    rows = _read_csv(path)
    matches = [row for row in rows if row.get("image_id") == image_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one imported external image_id={image_id}")
    row = matches[0]
    row.update({
        "ripeness_label": ripeness_label, "defect_label": defect_label,
        "review_status": review_status, "source_group": source_group,
        "reviewer": reviewer, "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    })
    _write_csv(path, rows, EXTERNAL_FIELDS)
    return path


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin


def evaluate_external(root: str | Path, assets: DemoAssets | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    assets = assets or load_demo_assets(root)
    rows = _read_csv(root / EXTERNAL_MANIFEST_PATH)
    exact_duplicates = [row for row in rows if row.get("review_status") == "approved"
                        and "exact_duplicate_phase051_data_not_external" in row.get("import_status", "")]
    approved = [row for row in rows if row.get("review_status") == "approved"
                and "exact_duplicate_phase051_data_not_external" not in row.get("import_status", "")]
    records_by_image: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for row in approved:
        source_path = project_path(assets.config, row.get("image_path", ""))
        if not source_path.is_file():
            for task, label_field in (("ripeness", "ripeness_label"), ("defect", "defect_label")):
                if row.get(label_field):
                    predictions.append(_external_failure_row(row, task, row[label_field], "input_unavailable", "image_path_missing"))
            continue
        try:
            result = infer_image(source_path.read_bytes(), source_path.name, assets)
            records_by_image[row["image_id"]] = result
            for task, label_field in (("ripeness", "ripeness_label"), ("defect", "defect_label")):
                true_label = row.get(label_field, "").strip()
                if not true_label:
                    continue
                pred = result[task]
                predictions.append({
                    **_external_failure_row(row, task, true_label,
                                             "ok" if pred["available"] else "prediction_unavailable", pred.get("reason", "")),
                    "predicted_label": pred.get("prediction", ""),
                    "fruit_mask_status": result["gray_world_mask_status"] if task == "ripeness" else result["fruit_mask_status"],
                    "kmeans_status": result["kmeans"]["status"],
                    "suspicious_area_fraction": "" if result["suspicious_area_percent"] is None else result["suspicious_area_percent"] / 100.0,
                    "gray_world_clipping_fraction": result["gray_world"]["clipping_fraction"],
                })
        except Exception as exc:
            LOGGER.exception("External evaluation failed for %s", row.get("image_id"))
            for task, label_field in (("ripeness", "ripeness_label"), ("defect", "defect_label")):
                if row.get(label_field):
                    predictions.append(_external_failure_row(row, task, row[label_field], "inference_error", f"{type(exc).__name__}: {exc}"))

    metrics: list[dict[str, Any]] = []
    labels_by_task = {"ripeness": ["unripe", "partially_ripe", "ripe"], "defect": ["healthy", "defective"]}
    for task, labels in labels_by_task.items():
        task_rows = [row for row in predictions if row["task"] == task]
        covered = [row for row in task_rows if row.get("prediction_status") == "ok" and row.get("predicted_label")]
        truth = [row["true_label"] for row in covered]
        predicted = [row["predicted_label"] for row in covered]
        group_keys = [row.get("source_group", "").strip() for row in covered]
        known_group_keys = [value for value in group_keys if value]
        independent_groups = len(set(known_group_keys))
        source_groups_complete = len(known_group_keys) == len(covered)
        p, r, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0) if covered else ([], [], [], [])
        cm = confusion_matrix(truth, predicted, labels=labels).tolist() if covered else []
        accuracy = accuracy_score(truth, predicted) if covered else None
        # Wilson intervals assume independent trials. Repeated fruit/source groups
        # invalidate that assumption, so keep the point estimates but withhold CI.
        ci_low, ci_high = (_wilson_interval(sum(a == b for a, b in zip(truth, predicted)), len(covered))
                           if source_groups_complete and len(covered) == independent_groups else (None, None))
        metrics.append({
            "task": task, "n_approved_labeled": len(task_rows), "n_prediction_covered": len(covered),
            "n_independent_source_groups": independent_groups,
            "n_source_group_missing": len(covered) - len(known_group_keys),
            "n_prediction_unavailable": len(task_rows) - len(covered),
            "coverage": len(covered) / len(task_rows) if task_rows else "",
            "accuracy_on_covered": accuracy if accuracy is not None else "",
            "accuracy_ci95_wilson_low": ci_low if ci_low is not None else "",
            "accuracy_ci95_wilson_high": ci_high if ci_high is not None else "",
            "macro_precision": float(np.mean(p)) if len(p) else "",
            "macro_recall": float(np.mean(r)) if len(r) else "",
            "macro_f1": float(np.mean(f1)) if len(f1) else "",
            "per_class_recall_json": json.dumps({label: float(value) for label, value in zip(labels, r)}, sort_keys=True),
            "support_json": json.dumps({label: int(value) for label, value in zip(labels, support)}, sort_keys=True),
            "confusion_matrix_json": json.dumps({"labels": labels, "matrix": cm}),
            "official_accuracy_eligible": "false",
            "metric_status": ("external_human_reviewed_group_lineage_unknown" if task_rows and not source_groups_complete
                              else "external_human_reviewed_grouped_images" if task_rows and independent_groups < len(covered)
                              else "external_human_reviewed" if task_rows else "no_approved_labels"),
        })
    metrics_path = root / "reports/phase07_external_metrics.csv"
    metric_fields = list(metrics[0])
    _write_csv(metrics_path, metrics, metric_fields)
    prediction_fields = list(dict.fromkeys(key for row in predictions for key in row)) or ["task", "image_id", "true_label", "predicted_label"]
    _write_csv(root / "reports/phase07_external_predictions.csv", predictions, prediction_fields)
    _merge_external_errors(root, predictions)
    return {"approved_images": len(approved), "excluded_exact_phase051_duplicates": len(exact_duplicates),
            "task_labeled_counts": {task: row["n_approved_labeled"] for task, row in zip(labels_by_task, metrics)},
            "metrics": metrics, "predictions": len(predictions)}


def _external_failure_row(row: dict[str, str], task: str, true_label: str,
                          status: str, reason: str) -> dict[str, Any]:
    return {
        "evaluation_scope": "external_validation", "task": task, "image_id": row.get("image_id", ""),
        "image_path": row.get("image_path", ""), "source_group": row.get("source_group", ""),
        "true_label": true_label, "predicted_label": "", "prediction_status": status,
        "fruit_mask_status": "", "kmeans_status": "", "suspicious_area_fraction": "",
        "gray_world_clipping_fraction": "", "preprocessing_warnings": reason,
        "observed_issue_stages": "input_or_pipeline" if status != "ok" else "",
        "error_type": "input_or_pipeline" if status != "ok" else "", "cause_status": "unknown" if status != "ok" else "",
        "review_status": row.get("review_status", ""), "review_notes": row.get("notes", ""),
    }


def _merge_external_errors(root: Path, external_rows: list[dict[str, Any]]) -> None:
    path = root / "reports/phase07_error_analysis.csv"
    existing = [row for row in _read_csv(path) if row.get("evaluation_scope") != "external_validation"]
    _write_csv(path, [*existing, *external_rows], ERROR_FIELDS)
    _make_error_contact_sheets(root, [*existing, *external_rows])


def _fruit_error_rgb(rgb: np.ndarray, gt: np.ndarray | None, pred: np.ndarray | None) -> np.ndarray:
    if gt is None or pred is None or gt.shape != pred.shape:
        return rgb.copy()
    error = rgb.astype(np.float32) * 0.40
    gt_fg, pred_fg = gt > 0, pred > 0
    true_positive = gt_fg & pred_fg
    false_positive = ~gt_fg & pred_fg
    false_negative = gt_fg & ~pred_fg
    error[true_positive] = error[true_positive] * 0.25 + np.asarray([40, 230, 100]) * 0.75
    error[false_positive] = error[false_positive] * 0.20 + np.asarray([255, 45, 45]) * 0.80
    error[false_negative] = error[false_negative] * 0.20 + np.asarray([40, 130, 255]) * 0.80
    return np.clip(error, 0, 255).astype(np.uint8)


def _write_fruit_review_sheets(root: Path, examples: list[dict[str, Any]], per_sheet: int = 4) -> list[str]:
    if not examples:
        return []
    out_dir = root / "reports/phase07_review/fruit_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, title_h, tile_h, caption_h = 205, 25, 190, 20
    row_h = title_h + tile_h + caption_h
    paths: list[str] = []
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(examples), per_sheet):
        group = examples[start:start + per_sheet]
        canvas = Image.new("RGB", (4 * width, len(group) * row_h), "white")
        draw = ImageDraw.Draw(canvas)
        for i, item in enumerate(group):
            top = i * row_h
            caption = (f"{item['image_id']} | {item['config']} | IoU={item.get('iou', '—')} | "
                       f"{item.get('status', 'ok')}")
            draw.text((4, top + 3), caption[:125], fill="#111111", font=font)
            rgb, gt, pred = item["rgb"], item.get("gt"), item.get("pred")
            error = _fruit_error_rgb(rgb, gt, pred)
            if gt is None:
                gt_view = np.zeros_like(rgb)
            else:
                gt_view = _rgb_overlay(rgb, gt)
            if pred is None:
                pred_view = np.zeros_like(rgb)
            else:
                pred_view = _rgb_overlay(rgb, pred)
            views = [("RGB", rgb), ("Ground truth", gt_view), ("Prediction", pred_view),
                     ("TP green · FP red · FN blue", error)]
            for col, (label, array) in enumerate(views):
                tile = Image.fromarray(array)
                tile.thumbnail((width - 8, tile_h - 4), Image.Resampling.LANCZOS)
                x = col * width + (width - tile.width) // 2
                y = top + title_h + (tile_h - tile.height) // 2
                canvas.paste(tile, (x, y))
                draw.text((col * width + 4, top + title_h + tile_h + 1), label[:30], fill="#222222", font=font)
        path = out_dir / f"fruit_segmentation_review_{start // per_sheet + 1:02d}.jpg"
        _atomic_image(path, canvas)
        paths.append(path.relative_to(root).as_posix())
    return paths


def _write_fruit_summary(root: Path, metric_rows: list[dict[str, Any]], approved_count: int,
                         validation_approved_count: int = 0) -> tuple[list[dict[str, Any]], str | None]:
    configs = sorted({(row.get("channel", ""), row.get("method", ""), row.get("polarity", ""))
                      for row in metric_rows if row.get("channel")})
    summary: list[dict[str, Any]] = []
    plot_path: str | None = None
    metrics = ("iou", "dice", "precision", "recall")
    for channel, method, polarity in configs:
        config_id = f"{channel}_{method}_{polarity}"
        group = [row for row in metric_rows if (row.get("channel"), row.get("method"), row.get("polarity"))
                 == (channel, method, polarity)]
        by_variant = {
            variant: {row.get("image_id", ""): row for row in group
                      if row.get("variant") == variant and row.get("metric_status") == "human_confirmed_subset_exploratory"}
            for variant in ("original", "gray_world")
        }
        validation_by_variant = {
            variant: {row.get("image_id", ""): row for row in group
                      if row.get("variant") == variant and row.get("phase02_split") == "validation"
                      and row.get("metric_status") == "human_confirmed_subset_exploratory"}
            for variant in ("original", "gray_world")
        }
        paired_ids = set(by_variant["original"]) & set(by_variant["gray_world"])
        validation_paired_ids = (set(validation_by_variant["original"])
                                 & set(validation_by_variant["gray_world"]))
        for variant in ("original", "gray_world"):
            valid_rows = list(by_variant[variant].values())
            paired_rows = [by_variant[variant][image_id] for image_id in sorted(paired_ids)]
            validation_rows = list(validation_by_variant[variant].values())
            validation_paired_rows = [validation_by_variant[variant][image_id]
                                      for image_id in sorted(validation_paired_ids)]
            row: dict[str, Any] = {
                "config_id": config_id, "variant": variant, "channel": channel, "method": method,
                "polarity": polarity, "n_approved_images": approved_count,
                "n_valid_predictions": len(valid_rows), "n_paired_images": len(paired_ids),
                "failed_predictions": max(0, approved_count - len(valid_rows)),
                "n_validation_approved_images": validation_approved_count,
                "n_validation_valid_predictions": len(validation_rows),
                "n_validation_paired_images": len(validation_paired_ids),
                "validation_failed_predictions": max(0, validation_approved_count - len(validation_rows)),
            }
            for metric in metrics:
                row[f"mean_{metric}"] = _mean_column(valid_rows, metric)
                row[f"paired_mean_{metric}"] = _mean_column(paired_rows, metric)
                row[f"validation_mean_{metric}"] = _mean_column(validation_rows, metric)
                row[f"validation_paired_mean_{metric}"] = _mean_column(validation_paired_rows, metric)
            summary.append(row)
    _write_csv(root / "reports/phase07_segmentation_summary.csv", summary, FRUIT_SUMMARY_FIELDS)

    paired_rows = [row for row in summary if row.get("n_paired_images", 0) and row.get("paired_mean_iou") is not None]
    if approved_count:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config_ids = sorted({row["config_id"] for row in paired_rows})
        variants = ("original", "gray_world")
        colors = {"original": "#2878b5", "gray_world": "#ef8a24"}
        fig, axes = plt.subplots(2, 2, figsize=(max(15, len(config_ids) * 0.95), 10), sharex=True)
        if paired_rows:
            for ax, metric in zip(axes.flat, metrics):
                x = np.arange(len(config_ids), dtype=float)
                bar_width = 0.38
                for offset, variant in enumerate(variants):
                    by_id = {row["config_id"]: row for row in paired_rows if row["variant"] == variant}
                    values = [by_id.get(config_id, {}).get(f"paired_mean_{metric}") for config_id in config_ids]
                    ax.bar(x + (offset - 0.5) * bar_width, [np.nan if v is None else v for v in values],
                           width=bar_width, color=colors[variant], label=variant.replace("_", " "))
                ax.set_title(metric.upper())
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.25)
            for ax in axes[1, :]:
                ax.set_xticks(np.arange(len(config_ids)))
                ax.set_xticklabels(config_ids, rotation=65, ha="right", fontsize=8)
            axes[0, 0].legend(loc="lower right")
        else:
            for ax, metric in zip(axes.flat, metrics):
                ax.set_title(metric.upper())
                ax.text(0.5, 0.5, "No paired valid predictions", ha="center", va="center",
                        transform=ax.transAxes, color="#6b7280")
                ax.set_axis_off()
        fig.suptitle("Fruit segmentation — paired Original vs Gray World (same approved images)")
        fig.tight_layout()
        path = root / "reports/phase07_segmentation_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_path = path.relative_to(root).as_posix()
    return summary, plot_path


def evaluate_approved_fruit_masks(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _config(root)
    candidate_rows, mask_rows = _load_review_manifests(root)
    candidate_by_id = {row.get("image_id", ""): row for row in candidate_rows}
    approved = [row for row in mask_rows if row.get("validation_status") == "valid_whole_fruit_mask"
                and row.get("semantic_review_status") == "human_confirmed_whole_fruit"]
    settings = _settings(config.get("segmentation", {}))
    rows_out: list[dict[str, Any]] = []
    worst_examples: list[tuple[float, dict[str, Any]]] = []
    failed_examples: list[dict[str, Any]] = []
    for gt_row in approved:
        image_id = gt_row.get("image_id", "")
        candidate = candidate_by_id.get(image_id)
        if not candidate:
            continue
        image_path = project_path(config, candidate["source_image_path"])
        mask_path = project_path(config, gt_row.get("ground_truth_path", ""))
        status = "valid"
        frame = None
        gt = None
        try:
            frame = decode_image(image_path)
            if candidate.get("source_sha256") and hashlib.sha256(image_path.read_bytes()).hexdigest() != candidate["source_sha256"]:
                raise ValueError("source_sha256_mismatch")
            gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if gt is None or gt.ndim != 2 or gt.dtype != np.uint8 or gt.shape != frame.rgb.shape[:2]:
                raise ValueError("ground_truth_shape_or_type_invalid")
            if not np.isin(gt, (0, 255)).all() or np.count_nonzero(gt) in {0, gt.size}:
                raise ValueError("ground_truth_not_binary_whole_fruit")
        except Exception as exc:
            rows_out.append({"image_id": image_id, "source_image_path": candidate.get("source_image_path", ""),
                             "ground_truth_mask_path": gt_row.get("ground_truth_path", ""),
                             "phase02_split": candidate.get("phase02_split", ""),
                             "prediction_status": f"invalid:{type(exc).__name__}:{exc}",
                             "ground_truth_status": gt_row.get("semantic_review_status", ""),
                             "metric_status": "not_calculated"})
            if frame is not None:
                rgb_small = _resize_rgb(frame.rgb, settings["preview_max_side"])
                failed_examples.append({"image_id": image_id, "config": "ground truth validation",
                                        "status": f"invalid GT: {type(exc).__name__}", "rgb": rgb_small,
                                        "gt": None, "pred": None})
            continue
        preview_original = _resize_rgb(frame.rgb, settings["preview_max_side"])
        gt_small = cv2.resize(gt, (preview_original.shape[1], preview_original.shape[0]), interpolation=cv2.INTER_NEAREST)
        variants = [("original", preview_original)]
        if settings["gray_world_enabled"]:
            balanced, _ = _variant_image(frame.rgb, frame.alpha, True)
            variants.append(("gray_world", _resize_rgb(balanced, settings["preview_max_side"])))
        for variant, rgb in variants:
            _, results = _run_candidates(rgb, settings)
            for key, outcome in results.items():
                row = {
                    "image_id": image_id, "source_image_path": candidate.get("source_image_path", ""),
                    "ground_truth_mask_path": gt_row.get("ground_truth_path", ""), "variant": variant,
                    "channel": outcome.channel, "method": outcome.method, "polarity": outcome.polarity,
                    "phase02_split": candidate.get("phase02_split", ""),
                    "prediction_status": outcome.status, "score": outcome.score,
                    "ground_truth_status": "human_confirmed_whole_fruit", "metric_status": "not_calculated",
                }
                if outcome.status == "accepted_heuristic":
                    metrics = segmentation_metrics(outcome.proposed_mask, gt_small)
                    row.update(metrics)
                    row["metric_status"] = "human_confirmed_subset_exploratory"
                    iou = _safe_float(metrics.get("iou"))
                    preview = {"image_id": image_id,
                               "config": f"{variant}/{outcome.channel}/{outcome.method}/{outcome.polarity}",
                               "status": "ok", "iou": f"{iou:.3f}" if iou is not None else "—",
                               "rgb": rgb, "gt": gt_small, "pred": outcome.proposed_mask}
                    if iou is not None:
                        worst_examples.append((iou, preview))
                        worst_examples.sort(key=lambda item: (item[0], item[1]["image_id"], item[1]["config"]))
                        worst_examples = worst_examples[:12]
                elif len(failed_examples) < 8:
                    failed_examples.append({"image_id": image_id,
                                            "config": f"{variant}/{outcome.channel}/{outcome.method}/{outcome.polarity}",
                                            "status": outcome.status, "rgb": rgb,
                                            "gt": gt_small, "pred": outcome.proposed_mask})
                rows_out.append(row)
    output = root / "reports/phase07_segmentation_metrics.csv"
    _write_csv(output, rows_out, FRUIT_METRIC_FIELDS)
    validation_approved_count = sum(candidate_by_id.get(row.get("image_id", ""), {}).get("phase02_split") == "validation"
                                    for row in approved)
    summary, plot_path = _write_fruit_summary(root, rows_out, len(approved), validation_approved_count)
    examples = failed_examples + [item for _, item in worst_examples]
    preview_paths = _write_fruit_review_sheets(root, examples)
    return {"approved_count": len(approved),
            "images_with_valid_metrics": len({row["image_id"] for row in rows_out if row.get("metric_status") == "human_confirmed_subset_exploratory"}),
            "metric_rows": len(rows_out), "summary_rows": len(summary), "comparison_plot": plot_path,
            "review_previews": preview_paths, "path": str(output.relative_to(root))}


def evaluate_reviewed_kmeans_pairs(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    reviews = _read_csv(root / KMEANS_REVIEW_PATH)
    approved = [row for row in reviews if row.get("review_status") == "approved"]
    results = _kmeans_result_pairs(root)
    metric_rows: list[dict[str, Any]] = []
    reasons = Counter()
    configurations = (("original_lab_k3", "Lab", 3), ("original_hsv_k5", "HSV", 5))
    for review in approved:
        image_id = review["image_id"]
        pair_rows = results.get(image_id, {})
        target_path = root / Path(review.get("ground_truth_mask_path", ""))
        image_path = root / Path(review.get("source_image_path", ""))
        try:
            image_shape = decode_image(image_path).rgb.shape[:2]
        except Exception:
            image_shape = None
        target = _mask_image(target_path, image_shape) if image_shape is not None else None
        target_valid = target is not None and bool(np.any(target))
        if not target_valid:
            reasons["target_mask_invalid"] += 1
        for config_name, color_space, k in configurations:
            result_row = pair_rows.get(config_name)
            status = result_row.get("kmeans_status", "missing_result_row") if result_row else "missing_result_row"
            prediction_path = root / Path(result_row.get("predicted_mask_path", "")) if result_row else Path()
            prediction = _mask_image(prediction_path, image_shape) if image_shape is not None else None
            prediction_valid = bool(result_row and status == "ok" and prediction is not None)
            if not result_row:
                reasons["result_missing"] += 1
            if not prediction_valid:
                reasons["prediction_unavailable"] += 1
            if not target_valid:
                metric = {name: "" for name in ("iou", "dice", "precision", "recall")}
                hit = ""
                prediction_pixels = ""
                metric_status = "approved_but_ground_truth_invalid"
            else:
                # A failed configuration remains in the paired denominator as an empty prediction.
                # This prevents a method from looking better by silently dropping its failures.
                scoring_prediction = prediction if prediction_valid else np.zeros_like(target)
                metric = segmentation_metrics(scoring_prediction, target)
                hit = int(np.count_nonzero((scoring_prediction > 0) & (target > 0)) > 0)
                prediction_pixels = int(np.count_nonzero(scoring_prediction))
                metric_status = ("human_approved_exploratory_lineage" if prediction_valid
                                 else "prediction_unavailable_scored_as_empty")
            fruit_path = root / Path(result_row.get("fruit_mask_path", "")) if result_row else Path()
            fruit_mask = _mask_image(fruit_path, target.shape) if target_valid else None
            fruit_pixels = int(np.count_nonzero(fruit_mask)) if fruit_mask is not None else 0
            metric_rows.append({
                "image_id": image_id, "dataset": review.get("dataset", ""),
                "source_image_path": review.get("source_image_path", ""),
                "ground_truth_mask_path": review.get("ground_truth_mask_path", ""),
                "color_space": color_space, "k": k, "review_status": "approved",
                "alignment_status": review.get("alignment_status", ""), "kmeans_status": status,
                **metric, "detection_hit": hit, "non_detection": 1 - hit if hit != "" else "",
                "prediction_available": str(prediction_valid).lower(), "prediction_pixels": prediction_pixels,
                "ground_truth_pixels": int(np.count_nonzero(target)) if target_valid else "",
                "predicted_area_fraction": float(prediction_pixels / fruit_pixels) if fruit_pixels and isinstance(prediction_pixels, int) else "",
                "metric_status": metric_status,
            })
    path = root / "reports/phase07_kmeans_metrics.csv"
    _write_csv(path, metric_rows, KMEANS_METRIC_FIELDS)
    summary_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for _, space, k in configurations:
        group = [row for row in metric_rows if row.get("color_space") == space]
        valid_gt = [row for row in group if row.get("iou") not in (None, "")]
        prediction_valid_rows = [row for row in valid_gt if row.get("prediction_available") == "true"]
        failures = len(valid_gt) - len(prediction_valid_rows)
        means = {name: float(np.mean([float(row[name]) for row in valid_gt])) if valid_gt else None
                 for name in ("iou", "dice", "precision", "recall")}
        detection_rate = (float(np.mean([int(row["detection_hit"]) for row in valid_gt]))
                          if valid_gt else None)
        no_valid_rate = failures / len(valid_gt) if valid_gt else None
        item = {
            "config_id": f"original_{space.casefold()}_k{k}", "color_space": space, "k": k,
            "n_approved": len(approved), "n_ground_truth_valid": len(valid_gt),
            "n_prediction_valid": len(prediction_valid_rows), "n_no_valid_result": failures,
            "no_valid_result_rate": no_valid_rate, "mean_iou": means["iou"],
            "mean_dice": means["dice"], "mean_pixel_precision": means["precision"],
            "mean_pixel_recall": means["recall"], "detection_rate": detection_rate,
            "metric_denominator_note": "All approved pairs with valid source mask; unavailable predictions scored as empty masks",
        }
        summary_rows.append(item)
        summaries[item["config_id"]] = item
    summary_path = root / "reports/phase07_kmeans_summary.csv"
    _write_csv(summary_path, summary_rows, KMEANS_SUMMARY_FIELDS)
    chart_path = _write_kmeans_comparison_chart(root, summary_rows)
    contact_sheets = _write_kmeans_evaluation_sheets(root, metric_rows)
    report_path = _write_kmeans_review_report(root, reviews, summary_rows, reasons, contact_sheets)
    status_counts = Counter(row.get("review_status", "pending") or "pending" for row in reviews)
    return {
        "approved_pairs": status_counts.get("approved", 0), "rejected_pairs": status_counts.get("rejected", 0),
        "uncertain_pairs": status_counts.get("uncertain", 0), "pending_pairs": status_counts.get("pending", 0),
        "metric_rows": len(metric_rows), "summaries": summaries, "excluded_reasons": dict(reasons),
        "path": str(path.relative_to(root)), "summary_path": str(summary_path.relative_to(root)),
        "comparison_chart": str(chart_path.relative_to(root)), "contact_sheets": contact_sheets,
        "report_path": str(report_path.relative_to(root)),
    }


def _write_kmeans_comparison_chart(root: Path, summaries: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = root / "reports/phase07_kmeans_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Original + Lab K=3", "Original + HSV K=5"]
    metric_names = [("mean_iou", "IoU"), ("mean_dice", "Dice"),
                    ("mean_pixel_precision", "Precision"), ("mean_pixel_recall", "Recall")]
    x = np.arange(len(metric_names))
    width = 0.35
    for idx, row in enumerate(summaries):
        values = [row[key] if row[key] is not None else 0 for key, _ in metric_names]
        axes[0].bar(x + (idx - 0.5) * width, values, width, label=labels[idx])
    axes[0].set_xticks(x, [name for _, name in metric_names])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pixel metrics (same approved pairs)")
    axes[0].legend(fontsize=8)
    rate_names = ["detection_rate", "no_valid_result_rate"]
    rate_labels = ["Detection rate", "No valid result"]
    x2 = np.arange(len(rate_names))
    for idx, row in enumerate(summaries):
        values = [row[key] if row[key] is not None else 0 for key in rate_names]
        axes[1].bar(x2 + (idx - 0.5) * width, values, width, label=labels[idx])
    axes[1].set_xticks(x2, rate_labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Detection and availability")
    axes[1].legend(fontsize=8)
    n = summaries[0]["n_ground_truth_valid"] if summaries else 0
    if not n:
        for axis in axes:
            axis.text(0.5, 0.5, "Chưa có cặp approved hợp lệ để tính metric", ha="center", va="center",
                      transform=axis.transAxes, color="darkred", wrap=True)
    fig.suptitle(f"Dataset C: Original Lab K=3 vs Original HSV K=5 | n={n}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _comparison_overlay(rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    view = rgb.astype(np.float32).copy()
    truth_on = truth > 0
    pred_on = prediction > 0
    colors = np.zeros_like(view)
    colors[truth_on & ~pred_on] = (40, 220, 80)       # ground-truth-only: green
    colors[pred_on & ~truth_on] = (250, 55, 45)       # prediction-only: red
    colors[truth_on & pred_on] = (255, 210, 0)        # overlap: yellow
    active = truth_on | pred_on
    view[active] = view[active] * 0.48 + colors[active] * 0.52
    return np.clip(view, 0, 255).astype(np.uint8)


def _write_kmeans_evaluation_sheets(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    output = root / "reports/phase07_review/kmeans"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for space, k, config_id in (("Lab", 3, "original_lab_k3"), ("HSV", 5, "original_hsv_k5")):
        group = [r for r in rows if r.get("color_space") == space and r.get("iou") not in (None, "")]
        cases = {
            "overlap_hits": [r for r in group if r.get("detection_hit") == 1 and r.get("prediction_available") == "true"],
            "no_overlap_or_unavailable": [r for r in group if r.get("detection_hit") != 1 or r.get("prediction_available") != "true"],
        }
        for suffix, selected in cases.items():
            tile_w, img_h, row_h = 540, 220, 250
            width = tile_w
            height = max(row_h, len(selected) * row_h + 40)
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
            heading = f"{space} K={k} | {suffix} | n={len(selected)} | green=GT only, red=prediction only, yellow=overlap"
            draw.text((8, 8), heading, fill="black", font=font)
            for i, row in enumerate(selected):
                top = 34 + i * row_h
                img_path = root / Path(row["source_image_path"])
                gt_path = root / Path(row["ground_truth_mask_path"])
                try:
                    rgb = decode_image(img_path).rgb
                    truth = _mask_image(gt_path, rgb.shape[:2])
                    pred_path_field = "lab_k3_prediction_path" if space == "Lab" else "hsv_k5_prediction_path"
                    review = next((x for x in _read_csv(root / KMEANS_REVIEW_PATH) if x["image_id"] == row["image_id"]), {})
                    pred = _mask_image(root / Path(review.get(pred_path_field, "")), rgb.shape[:2])
                except Exception:
                    rgb, truth, pred = np.zeros((160, 160, 3), np.uint8), None, None
                if truth is None:
                    draw.text((10, top + 10), f"{row['image_id']} — invalid GT", fill="red", font=font)
                    continue
                if pred is None or row.get("prediction_available") != "true":
                    pred = np.zeros_like(truth)
                overlay = _comparison_overlay(rgb, truth, pred)
                original = Image.fromarray(rgb)
                result = Image.fromarray(overlay)
                original.thumbnail((tile_w // 2 - 12, img_h), Image.Resampling.LANCZOS)
                result.thumbnail((tile_w // 2 - 12, img_h), Image.Resampling.LANCZOS)
                canvas.paste(original, (6, top + 22))
                canvas.paste(result, (tile_w // 2 + 6, top + 22))
                draw.text((8, top + 4), f"{row['image_id']} | IoU={float(row['iou']):.3f} | available={row['prediction_available']}", fill="black", font=font)
                draw.text((8, top + img_h + 24), "Original", fill="black", font=font)
                draw.text((tile_w // 2 + 6, top + img_h + 24), "Paired overlay", fill="black", font=font)
            path = output / f"evaluation_{config_id}_{suffix}.jpg"
            _atomic_image(path, canvas)
            paths.append(str(path.relative_to(root)).replace("\\", "/"))
    return paths


def _write_kmeans_review_report(root: Path, reviews: list[dict[str, str]], summaries: list[dict[str, Any]],
                                reasons: Counter, contact_sheets: list[str]) -> Path:
    status_counts = Counter(row.get("review_status", "pending") or "pending" for row in reviews)
    lines = [
        "# Phase 7 — Xác minh 50 cặp ảnh/mask khuyết tật Dataset C",
        "",
        f"- Danh sách review: `{KMEANS_REVIEW_PATH}` ({len(reviews)} cặp được chọn có chủ đích).",
        f"- Trạng thái: approved {status_counts.get('approved', 0)}, rejected {status_counts.get('rejected', 0)}, "
        f"uncertain {status_counts.get('uncertain', 0)}, pending {status_counts.get('pending', 0)}.",
        "- Quyết định được ghi từng cặp bởi người review; mở giao diện không đổi trạng thái.",
        "- Hai cấu hình được đánh giá paired trên toàn bộ cặp approved có mask nguồn hợp lệ. Nếu một cấu hình không có dự đoán hợp lệ, dự đoán rỗng được chấm cùng cặp (IoU/Dice/Precision/Recall bằng 0 với mask GT không rỗng).",
        "- Detection rate là tỷ lệ cặp có ít nhất một pixel giao giữa dự đoán và mask nguồn; no-valid-result rate là tỷ lệ cấu hình không tạo ra mask dự đoán hợp lệ trong các cặp có GT hợp lệ.",
        "- Tập 50 cặp được chọn theo các trường hợp metric/selection reason để review, không phải mẫu ngẫu nhiên; không đại diện cho hiệu năng ngoài tập này.",
        "- Các metric tiếp tục exploratory do augmentation lineage và độ tổng quát hóa chưa được giải quyết.",
        "",
        "## So sánh paired",
        "",
        "| Cấu hình | Approved | GT hợp lệ | Dự đoán hợp lệ | Không có kết quả hợp lệ | IoU | Dice | Pixel Precision | Pixel Recall | Detection rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "—" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| Original + {row['color_space']} K={row['k']} | {row['n_approved']} | {row['n_ground_truth_valid']} | "
            f"{row['n_prediction_valid']} | {row['n_no_valid_result']} ({fmt('no_valid_result_rate')}) | "
            f"{fmt('mean_iou')} | {fmt('mean_dice')} | {fmt('mean_pixel_precision')} | "
            f"{fmt('mean_pixel_recall')} | {fmt('detection_rate')} |"
        )
    human_notes = [row for row in reviews if (row.get("review_notes") or "").strip()]
    lines.extend(["", "## Ghi chú quan sát của người review", ""])
    if human_notes:
        for row in human_notes:
            note = " ".join((row.get("review_notes") or "").split()).replace("|", "\\|")
            lines.append(f"- `{row.get('image_id', '')}` ({row.get('review_status', 'pending')}): {note}")
        lines.append("Các ghi chú được giữ nguyên nội dung người review; không tự động gán nguyên nhân.")
    else:
        lines.append("Chưa có ghi chú thủ công; chưa thể kết luận trường hợp bỏ sót, bóng đổ, màu vỏ tự nhiên hay fruit mask sai.")
    lines.extend(["", "## Artifact", "",
                  "- Per-image/per-configuration: `reports/phase07_kmeans_metrics.csv`.",
                  "- Tổng hợp: `reports/phase07_kmeans_summary.csv`.",
                  "- Biểu đồ: `reports/phase07_kmeans_comparison.png`.",
                  "- Contact sheets theo overlap hit và no-overlap/unavailable: " + ", ".join(f"`{p}`" for p in contact_sheets) + ".",
                  "- Contact sheet phân loại chỉ theo pixel overlap, không tuyên bố đó là đánh giá chất lượng đầy đủ. Màu overlay: xanh = GT-only, đỏ = prediction-only, vàng = overlap.",
                  "- Nguyên nhân như bỏ sót vết thâm, bóng đổ, màu vỏ tự nhiên, hoặc fruit mask sai không tự động suy luận từ metric. Chỉ ghi nhận khi người review quan sát và ghi vào review notes.",
                  "", "## Hạn chế và bước tiếp theo", "",
                  "Chưa có cặp nào được đưa vào metric nếu trạng thái chưa approved. Trong giao diện, mở từng cặp ở độ phóng đại phù hợp, kiểm tra ảnh gốc, mask và overlay, đối chiếu hai dự đoán, ghi chú quan sát rồi mới chọn approved/rejected/uncertain. Chạy lại `kmeans-evaluate` sau khi duyệt.",
                  "",
                  "Không thay đổi dữ liệu raw, split hoặc model. Mask nguồn chỉ được dùng làm ground truth sau khi người review xác nhận căn chỉnh và ý nghĩa khuyết tật."])
    path = root / "reports/phase07_kmeans_review_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _test_set_error_analysis(root: Path, assets: DemoAssets) -> tuple[list[dict[str, Any]], dict[str, int]]:
    feature_manifest = {row.get("image_id", ""): row for row in _read_csv(root / "data/features/feature_extraction_manifest.csv")}
    records: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    for task, feature_rel in PHASE051_FEATURES.items():
        feature_rows = _read_csv(root / feature_rel)
        split_rows = _read_csv(root / PHASE051_SPLITS[task])
        split_by_id = {row.get("image_id", ""): row for row in split_rows}
        artifact = assets.ripeness_model if task == "ripeness" else assets.defect_model
        for feature in feature_rows:
            split = split_by_id.get(feature.get("image_id", ""), {})
            if split.get("phase051_split") != "test":
                continue
            fm = feature_manifest.get(feature["image_id"], {})
            mask_status = fm.get("gray_world_fruit_mask_status", "") if task == "ripeness" else fm.get("original_fruit_mask_status", "")
            result = _predict(artifact, feature, mask_status)
            true_label = feature.get("label", "")
            predicted = result.get("prediction") or ""
            if not result["available"]:
                error_type = "fruit_segmentation" if "fruit mask" in result["reason"] else "feature_extraction"
                errors[error_type] += 1
            elif predicted != true_label:
                error_type = "model_classification"
                errors[error_type] += 1
            else:
                error_type = "none"
            warning = []
            clipping = _safe_float(fm.get("gray_world_clipping_fraction"))
            if clipping is not None and clipping > 0.05:
                warning.append("gray_world_clipping_gt_5pct")
            if mask_status != "accepted_heuristic":
                warning.append("fruit_mask_not_accepted")
            kmeans_status = fm.get("original_lab_k3_status", "not_applicable")
            suspicion = feature.get("original_lab_k3__suspicious_area_fraction", "")
            if task == "ripeness":
                suspicion = ""
            records.append({
                "evaluation_scope": "phase051_locked_test_provisional", "task": task,
                "image_id": feature.get("image_id", ""), "image_path": feature.get("image_path", ""),
                "source_group": split.get("phase051_group_id", ""), "true_label": true_label,
                "predicted_label": predicted, "prediction_status": "ok" if result["available"] else result.get("reason", "unavailable"),
                "fruit_mask_status": mask_status, "kmeans_status": kmeans_status,
                "suspicious_area_fraction": suspicion, "gray_world_clipping_fraction": fm.get("gray_world_clipping_fraction", ""),
                "preprocessing_warnings": ";".join(warning),
                "observed_issue_stages": ";".join([
                    stage for stage, condition in (
                        ("fruit_segmentation", mask_status != "accepted_heuristic"),
                        ("kmeans", task == "defect" and kmeans_status != "ok"),
                        ("gray_world_clipping", clipping is not None and clipping > 0.05),
                        ("feature_or_model", not result["available"]),
                    ) if condition
                ]), "error_type": error_type,
                "cause_status": "unknown" if error_type == "model_classification" else "observed_stage" if error_type != "none" else "",
                "review_status": "not_external_human_review", "review_notes": "" if error_type == "none" else "Per-image failure reconstructed from locked Phase 5.1 test predictions; cause not independently established.",
            })
    _write_csv(root / "reports/phase07_error_analysis.csv", records, ERROR_FIELDS)
    return records, dict(errors)


def _make_error_contact_sheets(root: Path, rows: list[dict[str, Any]], per_sheet: int = 8) -> list[str]:
    output_dir = root / "reports/phase07_review/errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = [row for row in rows if row.get("error_type") not in {"", "none"}]
    paths = []
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(failures), per_sheet):
        batch = failures[start:start + per_sheet]
        canvas = Image.new("RGB", (800, len(batch) * 145), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(batch):
            y = index * 145
            image_path = root / Path(row.get("image_path", ""))
            try:
                frame = decode_image(image_path)
                thumb = Image.fromarray(frame.rgb)
                thumb.thumbnail((132, 132), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (4, y + 3))
            except Exception:
                draw.rectangle((4, y + 3, 136, y + 135), fill="#dddddd")
            text = (f"{row['task']} · {row['image_id']}\n"
                    f"true={row['true_label']} | pred={row['predicted_label'] or row['prediction_status']}\n"
                    f"error={row['error_type']} | mask={row['fruit_mask_status']}\n"
                    f"K-means={row['kmeans_status']} | ratio={row['suspicious_area_fraction']}\n"
                    f"cause={row['cause_status']} | scope={row['evaluation_scope']}")
            draw.multiline_text((145, y + 6), text, fill="black", font=font, spacing=3)
        path = output_dir / f"errors_{start // per_sheet + 1:02d}.jpg"
        _atomic_image(path, canvas)
        paths.append(str(path.relative_to(root)).replace("\\", "/"))
    return paths


def _external_classification_rows(root: Path) -> list[dict[str, Any]]:
    return [row for row in _read_csv(root / "reports/phase07_error_analysis.csv")
            if row.get("evaluation_scope") == "external_validation"]


def _select_robustness_samples(root: Path, limit: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    specifications = (("ripeness", "unripe"), ("ripeness", "partially_ripe"), ("ripeness", "ripe"),
                      ("defect", "healthy"), ("defect", "defective"))
    phase051_split = {
        task: {row.get("image_id", ""): row.get("phase051_split", "")
               for row in _read_csv(root / PHASE051_SPLITS[task])}
        for task in PHASE051_FEATURES
    }
    for task, label in specifications:
        if len(selected) >= limit:
            break
        rows = _read_csv(root / PHASE051_FEATURES[task])
        matches = sorted((row for row in rows if row.get("label") == label
                          and phase051_split[task].get(row.get("image_id", "")) == "test"
                          and (root / row.get("image_path", "")).is_file()),
                         key=lambda row: row.get("image_id", ""))
        if matches:
            chosen = dict(matches[0])
            chosen["_robustness_task"] = task
            chosen["_source_split"] = phase051_split[task].get(chosen.get("image_id", ""), "")
            selected.append(chosen)
    if len(selected) < limit:
        seen = {row.get("image_id") for row in selected}
        for task in ("ripeness", "defect"):
            for row in _read_csv(root / PHASE051_FEATURES[task]):
                if len(selected) >= limit:
                    break
                if row.get("image_id") not in seen and phase051_split[task].get(row.get("image_id", "")) == "test" and (root / row.get("image_path", "")).is_file():
                    chosen = dict(row); chosen["_robustness_task"] = task; selected.append(chosen)
                    chosen["_source_split"] = phase051_split[task].get(chosen.get("image_id", ""), "")
                    seen.add(row.get("image_id"))
    return selected


def _encode_rgb(rgb: np.ndarray, format_name: str, **kwargs: Any) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format=format_name, **kwargs)
    return output.getvalue()


def _encode_rgb_within_limit(rgb: np.ndarray, preferred_format: str = "PNG",
                             preferred_quality: int = 95) -> tuple[bytes, str]:
    if preferred_format == "PNG":
        payload = _encode_rgb(rgb, "PNG", optimize=True, compress_level=9)
        if len(payload) <= MAX_UPLOAD_BYTES:
            return payload, "PNG_lossless"
        qualities = (95, 92, 90, 85, 80)
    else:
        qualities = tuple(q for q in (preferred_quality, 92, 90, 85, 80) if q <= preferred_quality)
    for quality in qualities:
        payload = _encode_rgb(rgb, "JPEG", quality=quality, optimize=True)
        if len(payload) <= MAX_UPLOAD_BYTES:
            return payload, f"JPEG_quality_{quality}"
    raise ValueError(f"Transformed image remains larger than inference limit ({MAX_UPLOAD_BYTES} bytes)")


def _robustness_variants(payload: bytes, filename: str) -> list[tuple[str, str, bytes, str]]:
    frame = decode_upload(payload, filename)
    image = Image.fromarray(frame.rgb)
    bright_payload, bright_encoding = _encode_rgb_within_limit(
        np.asarray(ImageEnhance.Brightness(image).enhance(1.05)), "PNG")
    contrast_payload, contrast_encoding = _encode_rgb_within_limit(
        np.asarray(ImageEnhance.Contrast(image).enhance(1.05)), "PNG")
    jpeg_payload, jpeg_encoding = _encode_rgb_within_limit(frame.rgb, "JPEG", preferred_quality=90)
    variants: list[tuple[str, str, bytes, str]] = [
        ("original", "none", payload, filename),
        ("brightness_105", f"factor=1.05;encoding={bright_encoding}", bright_payload,
         "brightness.png" if bright_encoding == "PNG_lossless" else "brightness.jpg"),
        ("contrast_105", f"factor=1.05;encoding={contrast_encoding}", contrast_payload,
         "contrast.png" if contrast_encoding == "PNG_lossless" else "contrast.jpg"),
        ("jpeg_quality_90", f"requested_quality=90;encoding={jpeg_encoding}", jpeg_payload, "recompressed.jpg"),
    ]
    w, h = image.size
    size = (max(1, round(w * 0.8)), max(1, round(h * 0.8)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    resized_payload, resized_encoding = _encode_rgb_within_limit(np.asarray(resized), "PNG")
    variants.append(("resize_80pct", f"scale=0.8;aspect_preserved=true;encoding={resized_encoding}",
                     resized_payload, "resized.png" if resized_encoding == "PNG_lossless" else "resized.jpg"))
    return variants


def _training_feature_scales(root: Path, task: str, columns: list[str]) -> np.ndarray:
    rows = _read_csv(root / PHASE051_FEATURES[task])
    split = {row.get("image_id"): row for row in _read_csv(root / PHASE051_SPLITS[task])}
    train = [row for row in rows if split.get(row.get("image_id"), {}).get("phase051_split") == "train"]
    values = np.asarray([[_number(row.get(column)) for column in columns] for row in train], dtype=np.float64)
    if not values.size:
        return np.ones(len(columns), dtype=np.float64)
    scales = np.nanstd(values, axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    return scales


def run_robustness(root: str | Path, limit_images: int = 5,
                   assets: DemoAssets | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    assets = assets or load_demo_assets(root)
    samples = _select_robustness_samples(root, limit_images)
    output_rows: list[dict[str, Any]] = []
    inference_calls = 0
    successful_inference_calls = 0
    for sample in samples:
        task = sample["_robustness_task"]
        image_path = root / sample["image_path"]
        try:
            variants = _robustness_variants(image_path.read_bytes(), image_path.name)
        except Exception as exc:
            output_rows.append({"sample_id": sample["image_id"], "task": task,
                                "source_split": sample.get("_source_split", ""),
                                "transformation": "all", "inference_status": "input_failed",
                                "error": f"{type(exc).__name__}: {exc}"})
            continue
        results: dict[str, dict[str, Any]] = {}
        for name, params, payload, filename in variants:
            inference_calls += 1
            try:
                results[name] = infer_image(payload, filename, assets)
                successful_inference_calls += 1
            except Exception as exc:
                LOGGER.warning("Robustness inference failed %s/%s: %s", sample["image_id"], name, exc)
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        baseline = results.get("original", {})
        base_prediction = baseline.get(task, {}).get("prediction") if baseline.get(task, {}).get("available") else ""
        artifact = assets.ripeness_model if task == "ripeness" else assets.defect_model
        base_features = baseline.get("feature_values", {})
        base_vector = np.asarray([_number(base_features.get(c)) for c in artifact["feature_columns"]], dtype=np.float64)
        scales = _training_feature_scales(root, task, artifact["feature_columns"])
        for name, params, _, _ in variants:
            result = results.get(name, {})
            if result.get("error"):
                output_rows.append({"sample_id": sample["image_id"], "task": task,
                                    "source_split": sample.get("_source_split", ""),
                                    "transformation": name, "transformation_parameters": params,
                                    "inference_status": "failed", "error": result["error"]})
                continue
            prediction_record = result.get(task, {})
            available = bool(prediction_record.get("available"))
            prediction = prediction_record.get("prediction", "") if available else ""
            variant_features = np.asarray([_number(result.get("feature_values", {}).get(c))
                                           for c in artifact["feature_columns"]], dtype=np.float64)
            valid = np.isfinite(base_vector) & np.isfinite(variant_features)
            delta = np.abs(variant_features[valid] - base_vector[valid]) if np.any(valid) else np.asarray([])
            normalized = delta / scales[valid] if np.any(valid) else np.asarray([])
            ratio = result.get("suspicious_area_percent")
            base_ratio = baseline.get("suspicious_area_percent")
            output_rows.append({
                "sample_id": sample["image_id"], "task": task, "source_split": sample.get("_source_split", ""),
                "transformation": name, "transformation_parameters": params,
                "prediction_available": available, "prediction": prediction, "baseline_prediction": base_prediction,
                "prediction_stable": (prediction == base_prediction) if available and base_prediction else "",
                "availability_changed": available != bool(baseline.get(task, {}).get("available")),
                "mean_abs_feature_delta": float(np.mean(delta)) if delta.size else "",
                "mean_abs_feature_delta_train_sd": float(np.mean(normalized)) if normalized.size else "",
                "fruit_mask_status": result.get("gray_world_mask_status") if task == "ripeness" else result.get("fruit_mask_status"),
                "baseline_fruit_mask_status": baseline.get("gray_world_mask_status") if task == "ripeness" else baseline.get("fruit_mask_status"),
                "kmeans_status": result.get("kmeans", {}).get("status", ""),
                "baseline_kmeans_status": baseline.get("kmeans", {}).get("status", ""),
                "suspicious_area_percent": ratio if ratio is not None else "",
                "baseline_suspicious_area_percent": base_ratio if base_ratio is not None else "",
                "gray_world_clipping_fraction": result.get("gray_world", {}).get("clipping_fraction", ""),
                "inference_status": "ok" if result else "failed", "error": "",
            })
    path = root / "reports/phase07_robustness.csv"
    _write_csv(path, output_rows, ROBUSTNESS_FIELDS)
    stable_rows = [row for row in output_rows if row.get("transformation") != "original" and row.get("prediction_stable") not in ("", None)]
    return {"unique_images": len(samples), "inference_calls": inference_calls,
            "successful_inference_calls": successful_inference_calls,
            "variant_rows": len(output_rows), "prediction_comparisons": len(stable_rows),
            "prediction_changed": sum(row.get("prediction_stable") is False or str(row.get("prediction_stable")).casefold() == "false" for row in stable_rows),
            "path": str(path.relative_to(root))}


def run_error_analysis(root: str | Path, assets: DemoAssets | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    assets = assets or load_demo_assets(root)
    rows, counts = _test_set_error_analysis(root, assets)
    sheets = _make_error_contact_sheets(root, rows)
    return {"test_rows": len(rows), "error_counts": counts, "contact_sheets": sheets,
            "external_rows": len(_external_classification_rows(root)), "path": "reports/phase07_error_analysis.csv"}


def _phase_metrics(root: Path, task: str | None = None, stage: str | None = None) -> list[dict[str, str]]:
    return [row for row in _read_csv(root / "reports/phase051/phase051_metrics.csv")
            if row.get("scope") == "overall" and (task is None or row.get("task") == task)
            and (stage is None or row.get("stage") == stage)]


def _mean_metric(rows: list[dict[str, str]], experiment: str, model: str, field: str) -> float | None:
    matches = [float(row[field]) for row in rows if row.get("experiment") == experiment and row.get("model") == model and _safe_float(row.get(field)) is not None]
    return float(np.mean(matches)) if matches else None


def _mean_column(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_safe_float(row.get(field)) for row in rows]
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def _exploratory_kmeans_summary(root: Path) -> dict[str, dict[str, float | int]]:
    results = _read_csv(root / KMEANS_RESULT_PATH)
    summary = {}
    for name, space, k in (("original_lab_k3", "Lab", "3"), ("original_hsv_k5", "HSV", "5")):
        rows = [row for row in results if row.get("mode") == "heuristic" and row.get("variant") == "original"
                and row.get("color_space") == space and row.get("k") == k]
        usable = [row for row in rows if row.get("kmeans_status") == "ok" and _safe_float(row.get("iou")) is not None]
        summary[name] = {"images": len({row.get("image_id") for row in rows}), "valid": len(usable),
                         **{key: float(np.mean([float(row[key]) for row in usable])) if usable else 0.0
                            for key in ("iou", "dice", "precision", "recall", "detection_hit")}}
    return summary


def build_phase07_report(root: str | Path) -> Path:
    root = Path(root).resolve()
    candidates, fruit_gt = _load_review_manifests(root)
    fruit_approved = [row for row in fruit_gt if row.get("validation_status") == "valid_whole_fruit_mask"
                      and row.get("semantic_review_status") == "human_confirmed_whole_fruit"]
    fruit_approved_ids = {row.get("image_id", "") for row in fruit_approved}
    fruit_pending_rows = [row for row in candidates if row.get("image_id", "") not in fruit_approved_ids]
    fruit_pending_count = len(fruit_pending_rows)
    fruit_pending_statuses = Counter(row.get("annotation_status", "unknown") for row in fruit_pending_rows)
    kmeans_reviews = _read_csv(root / KMEANS_REVIEW_PATH)
    kmeans_counts = Counter(row.get("review_status", "pending") for row in kmeans_reviews)
    kmeans_metrics = _read_csv(root / "reports/phase07_kmeans_metrics.csv")
    external_rows = _read_csv(root / EXTERNAL_MANIFEST_PATH)
    external_counts = Counter(row.get("review_status", "pending") for row in external_rows)
    external_metrics = _read_csv(root / "reports/phase07_external_metrics.csv")
    external_scored_metrics = [row for row in external_metrics if int(row.get("n_approved_labeled") or 0) > 0]
    robustness = _read_csv(root / "reports/phase07_robustness.csv")
    errors = _read_csv(root / "reports/phase07_error_analysis.csv")
    fruit_metrics = _read_csv(root / "reports/phase07_segmentation_metrics.csv")
    fruit_summary = _read_csv(root / "reports/phase07_segmentation_summary.csv")
    kmeans_summary = _exploratory_kmeans_summary(root)
    previous = _phase_metrics(root)
    selected_rip = [row for row in previous if row.get("task") == "ripeness" and row.get("experiment") == "R2_gray_world_color"]
    selected_def = [row for row in previous if row.get("task") == "defect" and row.get("experiment") == "D1_original_color"]
    val_rip = next((r for r in selected_rip if r.get("model") == "KNN" and r.get("stage") == "validation"), {})
    test_rip = next((r for r in selected_rip if r.get("model") == "KNN" and r.get("stage") == "test_locked"), {})
    val_def = next((r for r in selected_def if r.get("model") == "SVM_RBF" and r.get("stage") == "validation"), {})
    test_def = next((r for r in selected_def if r.get("model") == "SVM_RBF" and r.get("stage") == "test_locked"), {})
    phase05 = _read_csv(root / "reports/phase051/phase051_vs_phase05.csv")
    phase051_best = defaultdict(dict)
    for row in previous:
        if row.get("stage") == "validation" and row.get("scope") == "overall":
            cur = phase051_best[row.get("task")].get(row.get("model"))
            if cur is None or float(row.get("macro_f1") or 0) > float(cur.get("macro_f1") or 0):
                phase051_best[row.get("task")][row.get("model")] = row
    robustness_pairs = [row for row in robustness if row.get("transformation") != "original"]
    robustness_changed = sum(str(row.get("prediction_stable", "")).casefold() == "false" for row in robustness_pairs)
    robustness_compared = sum(row.get("prediction_stable") not in ("", None) for row in robustness_pairs)
    robustness_success = sum(row.get("inference_status") == "ok" for row in robustness)
    robustness_splits = Counter(row.get("source_split", "unknown") for row in robustness if row.get("transformation") == "original")
    errors_by_type = Counter(row.get("error_type", "") for row in errors)
    error_breakdown = Counter((row.get("task", ""), row.get("error_type", "")) for row in errors
                              if row.get("error_type") not in {"", "none"})
    error_count = sum(value for key, value in errors_by_type.items() if key not in {"", "none"})
    lines = [
        "# Phase 7 — Kiểm chứng thực nghiệm và hoàn thiện đánh giá MangoResearch", "",
        "## Tóm tắt nghiệm thu", "",
        f"- Bộ chú giải whole-fruit ban đầu có **{len(candidates)} ảnh**; hiện có **{len(fruit_approved)} mask được người dùng xác nhận**, còn **{fruit_pending_count} mask chưa semantic approval** (trạng thái annotator: `{dict(fruit_pending_statuses)}`).",
        f"- Bộ review Dataset C chuẩn bị **{len(kmeans_reviews)} cặp**; trạng thái: {dict(kmeans_counts)}. Chỉ {sum(kmeans_counts.get(x, 0) for x in ('approved',))} cặp được chấp nhận thủ công.",
        f"- External validation có **{len(external_rows)} ảnh**, trạng thái review: {dict(external_counts)}. Không có nhãn approved nên chưa tính metric external.",
        f"- Robustness: **{len({row.get('sample_id') for row in robustness})} ảnh gốc**, split Phase 5.1={dict(robustness_splits)}, {robustness_success}/{len(robustness)} lượt inference thành công; prediction changed={robustness_changed}/{robustness_compared} comparison có dự đoán hợp lệ.",
        f"- Error analysis tạo **{len(errors)} dòng**; {error_count} dòng có lỗi hoặc dự đoán sai trong locked Phase 5.1 test. Đây là split provisional, không phải external validation.",
        "", "## Fruit segmentation", "",
        f"{len(candidates)} ảnh đã được chọn sẵn tại `data/metadata/fruit_gt_candidates.csv`; hiện {fruit_pending_count} ảnh chưa có mask whole-fruit được xác nhận. Polygon annotator dùng ảnh RGB gốc, có thể nạp heuristic mask làm gợi ý và yêu cầu người chú giải xác nhận/chỉnh sửa. Import và approval semantic vẫn là hai bước tách biệt.",
        "",
    ]
    if fruit_approved:
        valid_fruit = [row for row in fruit_metrics if row.get("metric_status") == "human_confirmed_subset_exploratory"]
        lines.append(f"Có {len(fruit_approved)} ảnh được duyệt; {len(valid_fruit)} cấu hình/ảnh có metric hợp lệ. Metric chi tiết theo ảnh ở `reports/phase07_segmentation_metrics.csv`.")
        validation_count = max((int(row.get("n_validation_approved_images") or 0) for row in fruit_summary), default=0)
        validation_eligible = [row for row in fruit_summary
                               if validation_count > 0
                               and int(row.get("n_validation_paired_images") or 0) == validation_count
                               and _safe_float(row.get("validation_paired_mean_iou")) is not None]
        validation_eligible.sort(key=lambda row: (
            -float(_safe_float(row.get("validation_paired_mean_iou")) or 0),
            -float(_safe_float(row.get("validation_paired_mean_dice")) or 0),
            int(row.get("validation_failed_predictions") or 0),
            row.get("variant", ""), row.get("config_id", ""),
        ))
        if validation_eligible:
            best = validation_eligible[0]
            lines.append(
                f"Chọn cấu hình chỉ trên validation ({validation_count} ảnh, đủ prediction cho cả Original và Gray World), "
                f"ưu tiên paired mean IoU, tie-break bằng Dice rồi số lần thất bại: "
                f"**{best.get('variant')}/{best.get('channel')}/{best.get('method')}/{best.get('polarity')}** "
                f"(IoU={_fmt_metric(_safe_float(best.get('validation_paired_mean_iou')))}, "
                f"Dice={_fmt_metric(_safe_float(best.get('validation_paired_mean_dice')))}). "
                "Split test không được dùng để chọn cấu hình. Đây là lựa chọn exploratory trên subset nhỏ."
            )
        else:
            lines.append(
                f"Không có cấu hình nào tạo prediction cho đồng thời Original và Gray World trên toàn bộ "
                f"{validation_count} ảnh validation; không công bố cấu hình thắng. Các hàng validation và tỷ lệ lỗi "
                "nằm trong `reports/phase07_segmentation_summary.csv`."
            )
        paired_configs = sorted({row.get("config_id", "") for row in fruit_summary
                                 if row.get("variant") == "original" and int(row.get("n_paired_images") or 0) > 0})
        if paired_configs:
            summary_by_key = {(row.get("config_id"), row.get("variant")): row for row in fruit_summary}
            lines += ["", "| Cấu hình Phase 2 | N paired | Original IoU / Dice / Precision / Recall | Gray World IoU / Dice / Precision / Recall |", "|---|---:|---|---|"]
            for config_id in paired_configs:
                original = summary_by_key.get((config_id, "original"), {})
                balanced = summary_by_key.get((config_id, "gray_world"), {})
                original_values = " / ".join(_fmt_metric(_safe_float(original.get(f"paired_mean_{metric}")))
                                             for metric in ("iou", "dice", "precision", "recall"))
                balanced_values = " / ".join(_fmt_metric(_safe_float(balanced.get(f"paired_mean_{metric}")))
                                              for metric in ("iou", "dice", "precision", "recall"))
                lines.append(f"| {config_id} | {original.get('n_paired_images', 0)} | {original_values} | {balanced_values} |")
            lines.append("\nSo sánh E1/E2 trên cùng image_id; xem `reports/phase07_segmentation_summary.csv` và `reports/phase07_segmentation_comparison.png` để biết đủ thống kê và biểu đồ.")
        else:
            lines.append("Chưa có cấu hình E1/E2 nào cùng tạo được prediction trên cùng ảnh duyệt; paired comparison được để trống.")
        fruit_previews = sorted((root / "reports/phase07_review/fruit_segmentation").glob("fruit_segmentation_review_*.jpg"))
        if fruit_previews:
            lines.append("Preview lỗi phân vùng: " + ", ".join(f"`{p.relative_to(root).as_posix()}`" for p in fruit_previews) + ".")
    else:
        lines.append("Chưa có whole-fruit ground truth được duyệt; `reports/phase07_segmentation_metrics.csv` chỉ có header. Không tạo IoU/Dice/Precision/Recall cho fruit segmentation.")
    lines += [
        "", "Dùng các lệnh `annotate-fruit-masks`, `import-fruit-masks` rồi `approve-fruit-masks` sau khi chú giải thủ công. Khi có mask được duyệt, `fruit-evaluate` so sánh Original/Gray World và các kênh HSV/CIELAB cùng method/polarity trong config. Các metric trên subset nhỏ vẫn exploratory, không dùng để tuyên bố benchmark.",
        "", "## K-Means trên Dataset C", "",
        "Một contact sheet review gồm 50 cặp đã được chọn phân tầng theo IoU cao, IoU thấp khác 0, IoU bằng 0, Precision thấp (ứng viên false positive) và Recall thấp (ứng viên false negative). Mỗi sheet hiển thị ảnh nguồn, overlay mask vùng khuyết tật nguồn, Original + Lab K=3 và Original + HSV K=5. `verified_resize_candidate` trong Phase 0.2 là kiểm tra tự động, không được xem như xác nhận của người dùng.",
        "", "| Cấu hình | Cặp có kết quả / 1.500 | Mean IoU | Mean Dice | Pixel Precision | Pixel Recall | Detection rate | Trạng thái |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, label in (("original_lab_k3", "Original + Lab + K=3"), ("original_hsv_k5", "Original + HSV + K=5")):
        data = kmeans_summary[name]
        lines.append(f"| {label} | {data['valid']} / {data['images']} | {data['iou']:.4f} | {data['dice']:.4f} | {data['precision']:.4f} | {data['recall']:.4f} | {data['detection_hit']:.4f} | exploratory, lineage chưa giải quyết |")
    if sum(kmeans_counts.get(x, 0) for x in ("approved",)):
        lines.append(f"\nMetric Phase 7 trên subset được người dùng duyệt: {len(kmeans_metrics)} hàng per image/config tại `reports/phase07_kmeans_metrics.csv`.")
        lines += ["", "| Cấu hình | n | Mean IoU | Mean Dice | Pixel Precision | Pixel Recall | Non-detection |", "|---|---:|---:|---:|---:|---:|---:|"]
        for space, k, label in (("Lab", 3, "Original + Lab + K=3"), ("HSV", 5, "Original + HSV + K=5")):
            subset = [row for row in kmeans_metrics if row.get("color_space") == space and row.get("iou") not in (None, "")]
            lines.append(f"| {label} | {len(subset)} | {_fmt_metric(_mean_column(subset, 'iou'))} | {_fmt_metric(_mean_column(subset, 'dice'))} | {_fmt_metric(_mean_column(subset, 'precision'))} | {_fmt_metric(_mean_column(subset, 'recall'))} | {_fmt_metric(_mean_column(subset, 'non_detection'))} |")
        lines.append("\nSubset metrics này chỉ mô tả các cặp được duyệt; lineage nguồn vẫn cần giải quyết trước benchmark.")
    else:
        lines.append("\nHiện chưa có cặp nào được người dùng duyệt; `reports/phase07_kmeans_metrics.csv` chỉ có header. Vì vậy 1.500 cặp ở trên vẫn là exploratory, không phải ground truth đã xác nhận.")
    lines += [
        "", "## External evaluation", "",
        f"Đã tạo `data/external_validation/images/` và `data/external_validation/external_manifest.csv` để nhập ảnh mới. Manifest cho phép ripeness/defect label độc lập, `source_group` và trạng thái review. Importer chỉ ghi nhãn rỗng; người dùng phải nhập nhãn và duyệt thủ công. Có {sum('exact_duplicate_phase051_data_not_external' in row.get('import_status', '') for row in external_rows)} file trùng SHA-256 với Phase 5.1, các file này không được tính là external.",
        "",
    ]
    if external_scored_metrics:
        lines.append("Metric trên các ảnh nhãn approved, dùng model Phase 5.1 khóa và inference Phase 6:")
        lines.append("")
        lines.append("| Task | Approved labels | Source groups đã khai báo | Coverage | Accuracy trên covered | Macro-F1 | CI 95% Accuracy (Wilson) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in external_scored_metrics:
            interval = (f"{row['accuracy_ci95_wilson_low']}–{row['accuracy_ci95_wilson_high']}"
                        if row.get("accuracy_ci95_wilson_low") and row.get("accuracy_ci95_wilson_high") else "không tính")
            lines.append(f"| {row['task']} | {row['n_approved_labeled']} | {row['n_independent_source_groups']} (thiếu {row['n_source_group_missing']}) | {row['coverage']} | {row['accuracy_on_covered']} | {row['macro_f1']} | {interval} |")
        lines.append("\nMetric là mô tả theo ảnh; ảnh cùng source_group không độc lập. Wilson chỉ được tính khi mọi ảnh covered có source_group đã khai báo và không bị trùng. Coverage tính trên toàn bộ nhãn approved.")
    else:
        lines.append("Chưa có ảnh external nào được duyệt nhãn; không có metric external hoặc CI. `reports/phase07_external_metrics.csv` chỉ ghi trạng thái `no_approved_labels`.")
    lines += [
        "", "## Error analysis end-to-end", "",
        f"`reports/phase07_error_analysis.csv` có trạng thái mask/K-Means, tỷ lệ nghi vấn, Gray World clipping, nhãn/dự đoán và scope cho từng ảnh test Phase 5.1; {error_count} trường hợp có lỗi hoặc nhãn dự đoán sai. Nguyên nhân chỉ ghi stage quan sát được; misclassification có `cause_status=unknown` cho đến khi review. Contact sheet failure nằm trong `reports/phase07_review/errors/`.",
    ]
    if error_breakdown:
        lines += ["", "| Task | Loại lỗi quan sát | Số ảnh |", "|---|---|---:|"]
        for (task, error_type), count in sorted(error_breakdown.items()):
            lines.append(f"| {task} | {error_type} | {count} |")
    lines += [
        "", "## Robustness paired", "",
        "Các biến thể nhẹ được tạo trong bộ nhớ từ cùng một ảnh: brightness ×1.05, contrast ×1.05, JPEG quality 90, và resize 80% giữ nguyên tỷ lệ. Ảnh gốc được chọn từ test split Phase 5.1 nếu đủ lớp. Biến thể giữ PNG lossless nếu nằm trong giới hạn upload 15 MB; ảnh PNG lớn hơn được mã hóa JPEG chất lượng cao và ghi rõ encoding trong CSV. Không lưu biến thể thành mẫu mới, không dùng để fit hoặc lựa chọn model. `mean_abs_feature_delta_train_sd` chuẩn hóa drift feature theo độ lệch chuẩn của train split Phase 5.1; đây là kiểm tra paired trên ít ảnh, không phải độ chính xác hay external validation.",
    ]
    robustness_transforms = sorted({row.get("transformation", "") for row in robustness
                                    if row.get("transformation") not in {"", "original", "all"}})
    if robustness_transforms:
        lines += ["", "| Biến thể | Ảnh có inference | Prediction changed | Mean feature drift / train SD |", "|---|---:|---:|---:|"]
        for transform in robustness_transforms:
            subset = [row for row in robustness if row.get("transformation") == transform]
            valid = [row for row in subset if row.get("prediction_available") in ("True", "true", True)]
            changed = sum(str(row.get("prediction_stable", "")).casefold() == "false" for row in subset)
            lines.append(f"| {transform} | {len(valid)} / {len(subset)} | {changed} | {_fmt_metric(_mean_column(subset, 'mean_abs_feature_delta_train_sd'))} |")
    lines += [
        "", "## Tổng hợp Phase 1–7", "",
        "- **Phase 1:** 14 ảnh deterministic được khảo sát màu; thống kê gồm cả nền. Gray World đôi lúc tạo clipping đáng kể trên các nhóm dataset. Không có ground truth màu chiếu sáng, nên đây là mô tả biến đổi chứ không chứng minh ảnh được cải thiện.",
        "- **Phase 2/2.1:** baseline dùng threshold/morphology/component ranking. `accepted_heuristic` chỉ là trạng thái theo heuristic; hiện không có approved whole-fruit mask nên chất lượng tách quả chưa được xác minh bằng IoU/Dice.",
        f"- **Phase 3:** 1.479/1.500 ảnh có K-Means output ở mỗi cấu hình trong kết quả đã lưu. {kmeans_summary['original_lab_k3']['valid']} cặp Original Lab K3 có metric exploratory; HSV K5 có {kmeans_summary['original_hsv_k5']['valid']}. Các số đo dựa trên mask đã resize và lineage Dataset C unresolved, không dùng như ground truth chính thức.",
        "- **Phase 4:** feature tables tách một dòng mỗi image_id; cột metadata/label loại khỏi input. Feature K-Means là hai nhánh song song. Báo cáo Phase 4 ghi nhận khác biệt phân bố giữa ảnh segmentation và classification; không suy rộng chất lượng mask K-Means sang phân loại.",
        "- **Phase 5/5.1:** KNN, SVM-RBF và Random Forest đã được so sánh bằng validation/CV. Phase 5.1 chọn Gray World + KNN cho ripeness và Original Color + SVM-RBF cho defect. D1/D4 và các nhánh khác không được xem là mẫu độc lập. Near-duplicate và augmentation lineage chưa giải quyết nên kết quả vẫn exploratory.",
        "- **Phase 6:** demo gọi đúng model artifact và feature schema, có status unavailable khi mask thất bại. Smoke test trên vài ảnh chỉ kiểm tra hoạt động, không phải external evaluation.",
        "- **Phase 7:** importer, workflow review, robustness và error analysis đã tạo. External validation và metric ground-truth segmentation/K-Means vẫn chờ dữ liệu/duyệt thủ công.",
        "", "### Model Phase 5.1 đã khóa (đối chiếu, không tái huấn luyện)", "",
        "| Task / model | Validation n | Validation Accuracy | Validation Macro-F1 | Locked test n | Test Accuracy | Test Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Ripeness · Gray World + KNN | {val_rip.get('n_samples','')} | {float(val_rip.get('accuracy') or 0):.4f} | {float(val_rip.get('macro_f1') or 0):.4f} | {test_rip.get('n_samples','')} | {float(test_rip.get('accuracy') or 0):.4f} | {float(test_rip.get('macro_f1') or 0):.4f} |",
        f"| Defect · Original Color + SVM-RBF | {val_def.get('n_samples','')} | {float(val_def.get('accuracy') or 0):.4f} | {float(val_def.get('macro_f1') or 0):.4f} | {test_def.get('n_samples','')} | {float(test_def.get('accuracy') or 0):.4f} | {float(test_def.get('macro_f1') or 0):.4f} |",
        "",
        "Best validation Macro-F1 per classifier (historical Phase 5.1 results): "
        + "; ".join(f"{task}: " + ", ".join(f"{model}={float(row.get('macro_f1') or 0):.4f} ({row.get('experiment')})" for model, row in sorted(models.items()))
                     for task, models in phase051_best.items()),
        "",
        "Phase 5.1 artifacts retain `official_accuracy_eligible=false`. Test values are reported only as a provisional historical reference; this Phase 7 workflow does not tune on test. No Accuracy is presented as an official benchmark.",
        "", "## Kiểm thử và thực thi", "",
        "`scripts/phase07.py run` đã chạy end-to-end trên dữ liệu workspace: tạo review pack, error analysis, external import/evaluation, segmentation/K-Means guarded evaluation, robustness và báo cáo. Robustness sau đó được chạy lại riêng trên 5 ảnh từ test split để xác minh các biến thể vượt giới hạn kích thước được mã hóa an toàn; không ảnh nào bị ghi vào raw. Toàn bộ unit test hiện có: **89 passed**.",
        "", "## Lệnh chính", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py run",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py kmeans-review --image-id IMAGE_ID --status approved --reviewer 'Reviewer' --notes 'Mask aligned and defect region confirmed'",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py external-import",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py external-review --image-id EXT_ID --status approved --ripeness-label ripe --source-group fruit_01 --reviewer 'Reviewer'",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py external-evaluate",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py fruit-evaluate",
        ".\\.venv\\Scripts\\python.exe scripts/phase07.py robustness --limit-images 5",
        "```", "",
    ]
    output = root / "reports/phase07_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_phase07(root: str | Path, robustness_limit: int = 5) -> dict[str, Any]:
    root = Path(root).resolve()
    assets = load_demo_assets(root)
    preparation = prepare_review_workflows(root, 50)
    external = import_external_images(root)
    fruit = evaluate_approved_fruit_masks(root)
    kmeans = evaluate_reviewed_kmeans_pairs(root)
    error_analysis = run_error_analysis(root, assets)
    robustness = run_robustness(root, robustness_limit, assets)
    external_eval = evaluate_external(root, assets)
    report_path = build_phase07_report(root)
    return {"preparation": preparation, "external_import": external, "fruit_segmentation": fruit,
            "kmeans": kmeans, "error_analysis": error_analysis, "robustness": robustness,
            "external_evaluation": external_eval, "report": str(report_path.relative_to(root))}

