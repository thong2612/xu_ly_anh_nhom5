from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color.preprocess import decode_image, gray_world, rgb_to_hsv, rgb_to_lab
from src.segmentation.phase3 import Phase2FruitMaskProvider, cluster_and_detect
from src.utils.config import project_path


LOGGER = logging.getLogger("mangoresearch.phase4")
PIPELINE_VERSION = "phase4-features-v1"
CHANNELS = ("hsv_h", "hsv_s", "hsv_v", "lab_l", "lab_a", "lab_b")
STATISTICS = ("mean", "median", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")
COLOR_PROPORTIONS = ("green_fraction", "yellow_fraction", "orange_fraction")
KMEANS_FEATURES = (
    "detected_suspicious", "suspicious_cluster_count", "suspicious_area_fraction",
    "dominant_cluster_fraction", "selected_mean_color_distance",
    "selected_mean_brightness_delta", "selected_mean_suspicion_score",
    "selected_mean_continuity", "selected_max_suspicion_score",
    "selected_max_component_fraction", "region_hsv_h_circular_mean",
    "region_hsv_h_circular_resultant", "region_hsv_s_mean", "region_hsv_v_mean",
    "region_lab_l_mean", "region_lab_a_mean", "region_lab_b_mean",
)
METADATA_FIELDS = (
    "image_id", "dataset", "task", "label", "phase02_split", "image_path",
    "source_sha256", "split_risk_status", "official_accuracy_eligible",
)
MANIFEST_FIELDS = (
    "image_id", "dataset", "task", "source_image_path", "source_sha256",
    "target_label", "phase02_split", "split_risk_status", "official_accuracy_eligible",
    "image_status", "image_width", "image_height", "analysis_width", "analysis_height",
    "original_fruit_mask_status",
    "original_fruit_mask_source", "original_fruit_area_fraction", "gray_world_applied",
    "gray_world_gains", "gray_world_clipping_fraction", "gray_world_fruit_mask_status",
    "gray_world_fruit_mask_source", "gray_world_fruit_area_fraction",
    "original_lab_k3_status", "gray_world_lab_k3_status",
    "original_hsv_k5_status", "gray_world_hsv_k5_status",
    "feature_status", "feature_column_count", "missing_feature_count",
    "cache_hit", "error",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.15 * (attempt + 1))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray | None) -> str:
    if array is None:
        return "none"
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _feature_names(prefix: str) -> list[str]:
    names = [f"{prefix}__{channel}_{stat}" for channel in CHANNELS for stat in STATISTICS]
    names.extend((f"{prefix}__hsv_h_circular_mean", f"{prefix}__hsv_h_circular_resultant"))
    names.extend(f"{prefix}__{item}" for item in COLOR_PROPORTIONS)
    names.append(f"{prefix}__fruit_area_fraction")
    return names


def _kmeans_feature_names(prefix: str) -> list[str]:
    return [f"{prefix}__{name}" for name in KMEANS_FEATURES]


def _empty_features(columns: list[str]) -> dict[str, Any]:
    return {name: "" for name in columns}


def _circular_hue(hue: np.ndarray) -> tuple[float, float]:
    if hue.size == 0:
        return math.nan, math.nan
    radians = np.deg2rad(hue.astype(np.float64))
    sine = float(np.sin(radians).mean())
    cosine = float(np.cos(radians).mean())
    return float(np.rad2deg(np.arctan2(sine, cosine)) % 360.0), float(math.hypot(sine, cosine))


def _color_features(
    rgb: np.ndarray,
    fruit_mask: np.ndarray | None,
    prefix: str,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    output = _empty_features(_feature_names(prefix))
    if fruit_mask is None:
        return output, "fruit_mask_unavailable"
    mask = np.asarray(fruit_mask)
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != rgb.shape[:2] or not np.isin(mask, (0, 255)).all():
        return output, "fruit_mask_invalid"
    roi = mask > 0
    count = int(np.count_nonzero(roi))
    if count < 1:
        return output, "fruit_mask_empty"

    hsv = rgb_to_hsv(rgb)[roi]
    lab = rgb_to_lab(rgb)[roi]
    channels = {
        "hsv_h": hsv[:, 0], "hsv_s": hsv[:, 1], "hsv_v": hsv[:, 2],
        "lab_l": lab[:, 0], "lab_a": lab[:, 1], "lab_b": lab[:, 2],
    }
    for channel, data in channels.items():
        values = data.astype(np.float64, copy=False)
        stats = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values)), "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        quantiles = np.percentile(values, (10, 25, 50, 75, 90))
        stats.update({f"p{p}": float(q) for p, q in zip((10, 25, 50, 75, 90), quantiles)})
        for stat, value in stats.items():
            output[f"{prefix}__{channel}_{stat}"] = value
    hue_mean, hue_resultant = _circular_hue(hsv[:, 0])
    output[f"{prefix}__hsv_h_circular_mean"] = hue_mean
    output[f"{prefix}__hsv_h_circular_resultant"] = hue_resultant

    saturation_min = float(thresholds.get("minimum_saturation", 0.15))
    value_min = float(thresholds.get("minimum_value", 0.08))
    chromatic = (hsv[:, 1] >= saturation_min) & (hsv[:, 2] >= value_min)
    hue_ranges = dict(thresholds.get("hue_ranges_degrees", {}))
    for name in COLOR_PROPORTIONS:
        range_name = name.removesuffix("_fraction")
        low, high = (float(x) for x in hue_ranges.get(range_name, (0.0, 0.0)))
        hue = hsv[:, 0]
        in_range = (hue >= low) & (hue < high) if low <= high else ((hue >= low) | (hue < high))
        output[f"{prefix}__{name}"] = float(np.count_nonzero(chromatic & in_range) / count)
    output[f"{prefix}__fruit_area_fraction"] = float(count / (rgb.shape[0] * rgb.shape[1]))
    return output, "ok"


def _kmeans_features_from_outcome(
    rgb: np.ndarray,
    fruit_mask: np.ndarray,
    outcome: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    output = _empty_features(_kmeans_feature_names(prefix))
    if outcome.get("status") != "ok":
        return output
    fruit_pixels = int(np.count_nonzero(fruit_mask))
    if fruit_pixels < 1:
        return output
    prediction = np.asarray(outcome.get("defect_mask"))
    details = list(outcome.get("cluster_details", []))
    selected = [item for item in details if item.get("selected")]
    selected_ids = list(outcome.get("selected_cluster_ids", []))
    predicted_pixels = int(np.count_nonzero((prediction > 0) & (fruit_mask > 0)))
    output.update({
        f"{prefix}__detected_suspicious": int(predicted_pixels > 0),
        f"{prefix}__suspicious_cluster_count": len(selected_ids),
        f"{prefix}__suspicious_area_fraction": float(predicted_pixels / fruit_pixels),
        f"{prefix}__dominant_cluster_fraction": max((float(row["area_fraction"]) for row in details), default=""),
    })
    if selected:
        columns = {
            "selected_mean_color_distance": "color_distance_from_dominant",
            "selected_mean_brightness_delta": "brightness_delta_from_dominant",
            "selected_mean_suspicion_score": "suspicion_score",
            "selected_mean_continuity": "continuity",
        }
        for output_name, detail_name in columns.items():
            output[f"{prefix}__{output_name}"] = float(np.mean([float(item[detail_name]) for item in selected]))
        output[f"{prefix}__selected_max_suspicion_score"] = max(float(x["suspicion_score"]) for x in selected)
        output[f"{prefix}__selected_max_component_fraction"] = max(
            int(x["largest_component_pixels"]) for x in selected
        ) / fruit_pixels
        selected_pixels = (prediction > 0) & (fruit_mask > 0)
        hsv = rgb_to_hsv(rgb)[selected_pixels]
        lab = rgb_to_lab(rgb)[selected_pixels]
        hue_mean, hue_resultant = _circular_hue(hsv[:, 0])
        output.update({
            f"{prefix}__region_hsv_h_circular_mean": hue_mean,
            f"{prefix}__region_hsv_h_circular_resultant": hue_resultant,
            f"{prefix}__region_hsv_s_mean": float(np.mean(hsv[:, 1])),
            f"{prefix}__region_hsv_v_mean": float(np.mean(hsv[:, 2])),
            f"{prefix}__region_lab_l_mean": float(np.mean(lab[:, 0])),
            f"{prefix}__region_lab_a_mean": float(np.mean(lab[:, 1])),
            f"{prefix}__region_lab_b_mean": float(np.mean(lab[:, 2])),
        })
    return output


def _kmeans_features_from_saved(
    rgb: np.ndarray,
    fruit_mask: np.ndarray | None,
    record: dict[str, str] | None,
    config: dict[str, Any],
    prefix: str,
) -> tuple[dict[str, Any], str]:
    empty = _empty_features(_kmeans_feature_names(prefix))
    if fruit_mask is None:
        return empty, "fruit_mask_unavailable"
    if not record or record.get("kmeans_status") != "ok":
        return empty, record.get("kmeans_status", "phase3_record_missing") if record else "phase3_record_missing"
    try:
        prediction_path = project_path(config, record.get("predicted_mask_path", ""))
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
        if prediction is None or prediction.ndim != 2 or prediction.shape != fruit_mask.shape:
            raise ValueError("Phase 3 predicted mask missing or has wrong dimensions")
        if not np.isin(prediction, (0, 255)).all():
            raise ValueError("Phase 3 predicted mask is not binary")
        details = json.loads(record.get("cluster_details_json", "[]") or "[]")
        selected_ids = json.loads(record.get("selected_cluster_ids", "[]") or "[]")
        outcome = {"status": "ok", "defect_mask": prediction,
                   "cluster_details": details, "selected_cluster_ids": selected_ids}
        return _kmeans_features_from_outcome(rgb, fruit_mask, outcome, prefix), "ok"
    except Exception as exc:
        return empty, f"phase3_artifact_invalid:{type(exc).__name__}"


def _derived_seed(seed: int, image_id: str, variant: str, color_space: str, k: int) -> int:
    text = f"{seed}|{image_id}|heuristic|{variant}|{color_space}|{k}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") & 0x7FFFFFFF


def _cache_key(
    row: dict[str, str], source_hash: str, masks: dict[str, np.ndarray | None],
    statuses: dict[str, str], phase4: dict[str, Any], phase3: dict[str, Any],
) -> str:
    payload = {
        "version": PIPELINE_VERSION, "image_id": row.get("image_id"), "source_sha256": source_hash,
        "masks": {key: _sha256_array(value) for key, value in masks.items()}, "mask_status": statuses,
        "phase4": phase4, "phase3": {key: phase3.get(key) for key in (
            "seed", "attempts", "max_iterations", "epsilon", "max_fit_pixels", "min_fruit_pixels", "selection_rule"
        )},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata(row: dict[str, str], source_hash: str | None = None) -> dict[str, Any]:
    return {
        "image_id": row.get("image_id", ""), "dataset": row.get("dataset", ""),
        "task": row.get("task", ""),
        "label": row.get("ripeness_label", "") or row.get("defect_label", ""),
        "phase02_split": row.get("phase02_split", "") or row.get("split", ""),
        "image_path": row.get("image_path", ""), "source_sha256": source_hash or row.get("sha256", ""),
        "split_risk_status": row.get("split_risk_status", ""),
        "official_accuracy_eligible": row.get("official_accuracy_eligible", "false"),
    }


def _manifest_template(row: dict[str, str]) -> dict[str, Any]:
    return {
        "image_id": row.get("image_id", ""), "dataset": row.get("dataset", ""),
        "task": row.get("task", ""), "source_image_path": row.get("image_path", ""),
        "source_sha256": row.get("sha256", ""),
        "target_label": row.get("ripeness_label", "") or row.get("defect_label", ""),
        "phase02_split": row.get("phase02_split", "") or row.get("split", ""),
        "split_risk_status": row.get("split_risk_status", ""),
        "official_accuracy_eligible": row.get("official_accuracy_eligible", "false"),
        "image_status": "pending", "image_width": "", "image_height": "",
        "analysis_width": "", "analysis_height": "",
        "original_fruit_mask_status": "not_processed", "original_fruit_mask_source": "",
        "original_fruit_area_fraction": "", "gray_world_applied": "",
        "gray_world_gains": "", "gray_world_clipping_fraction": "",
        "gray_world_fruit_mask_status": "not_processed", "gray_world_fruit_mask_source": "",
        "gray_world_fruit_area_fraction": "", "original_lab_k3_status": "not_applicable",
        "gray_world_lab_k3_status": "not_applicable", "original_hsv_k5_status": "not_applicable",
        "gray_world_hsv_k5_status": "not_applicable", "feature_status": "pending",
        "feature_column_count": 0, "missing_feature_count": "", "cache_hit": "false", "error": "",
    }


def _valid_working_mask(mask: np.ndarray | None, alpha: np.ndarray | None,
                        min_pixels: int) -> tuple[np.ndarray | None, str]:
    if mask is None:
        return None, "fruit_mask_unavailable"
    if mask.dtype != np.uint8 or mask.ndim != 2 or not np.isin(mask, (0, 255)).all():
        return None, "fruit_mask_invalid"
    result = mask.copy()
    if alpha is not None:
        result[alpha == 0] = 0
    count = int(np.count_nonzero(result))
    if count == 0:
        return None, "fruit_mask_empty"
    if count < min_pixels:
        return None, "insufficient_fruit_pixels"
    return result, "accepted_heuristic"


def _analysis_raster(rgb: np.ndarray, alpha: np.ndarray | None, max_side: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Build a temporary aspect-preserving analysis raster without modifying the source frame."""
    if max_side < 1:
        raise ValueError("max_analysis_side must be positive")
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb.copy(), None if alpha is None else alpha.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    reduced_alpha = None if alpha is None else cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST)
    return reduced_rgb, reduced_alpha


def _build_feature_columns(task_key: str) -> tuple[list[str], dict[str, list[str]]]:
    if task_key == "ripeness":
        sets = {"original_color": _feature_names("original_color"),
                "gray_world_color": _feature_names("gray_world_color")}
    else:
        sets = {
            "original_color": _feature_names("original_color"),
            "gray_world_color": _feature_names("gray_world_color"),
            "original_lab_k3": _kmeans_feature_names("original_lab_k3"),
            "gray_world_lab_k3": _kmeans_feature_names("gray_world_lab_k3"),
            "original_hsv_k5": _kmeans_feature_names("original_hsv_k5"),
            "gray_world_hsv_k5": _kmeans_feature_names("gray_world_hsv_k5"),
        }
    columns = [name for group in sets.values() for name in group]
    return columns, sets


def _empty_dataset_row(meta: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {**meta, **_empty_features(columns)}


def _process_classification_image(
    row: dict[str, str], task_key: str, config: dict[str, Any], provider: Phase2FruitMaskProvider,
    phase4: dict[str, Any], phase3: dict[str, Any], feature_columns: list[str],
    feature_sets: dict[str, list[str]], cache_dir: Path, force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _manifest_template(row)
    output = _empty_dataset_row(_metadata(row), feature_columns)
    image_path = project_path(config, row["image_path"])
    try:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {row['image_path']}")
        source_hash = _sha256_file(image_path)
        if row.get("sha256") and source_hash != row["sha256"]:
            raise ValueError("Raw image SHA-256 differs from Phase 0.2 split manifest")
        frame = decode_image(image_path)
        rgb = frame.rgb
        manifest["image_status"] = "ok"
        manifest["image_width"], manifest["image_height"] = rgb.shape[1], rgb.shape[0]
        analysis_rgb, analysis_alpha = _analysis_raster(
            rgb, frame.alpha, int(phase4.get("max_analysis_side", 512))
        )
        manifest["analysis_width"], manifest["analysis_height"] = analysis_rgb.shape[1], analysis_rgb.shape[0]
        rgb, alpha = analysis_rgb, analysis_alpha
        balanced_result = gray_world(rgb, alpha)
        balanced = balanced_result.rgb
        manifest["gray_world_applied"] = str(balanced_result.applied).lower()
        manifest["gray_world_gains"] = json.dumps(balanced_result.gains)
        manifest["gray_world_clipping_fraction"] = balanced_result.clipping_fraction

        masks: dict[str, np.ndarray | None] = {}
        statuses: dict[str, str] = {}
        for variant, phase_variant in (("original", "E1"), ("gray_world", "E2")):
            raw_mask, status, source, _ = provider.get(row, rgb, alpha, source_hash, phase_variant)
            mask, valid_status = _valid_working_mask(raw_mask, alpha, int(phase4.get("min_fruit_pixels", 100)))
            masks[variant], statuses[variant] = mask, valid_status
            manifest[f"{variant}_fruit_mask_status"] = valid_status
            manifest[f"{variant}_fruit_mask_source"] = source
            if mask is not None:
                manifest[f"{variant}_fruit_area_fraction"] = float(np.count_nonzero(mask) / mask.size)

        cache_key = _cache_key(row, source_hash, masks, statuses, phase4, phase3)
        cache_path = cache_dir / task_key / f"{row['image_id']}.json"
        if cache_path.is_file() and not force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    cached_manifest = cached["manifest"]
                    cached_manifest["cache_hit"] = "true"
                    return cached["features"], cached_manifest
            except (OSError, ValueError, KeyError, TypeError):
                pass

        output = {**_metadata(row, source_hash), **_empty_features(feature_columns)}
        image_by_variant = {"original": rgb, "gray_world": balanced}
        statuses_color: dict[str, str] = {}
        for variant in ("original", "gray_world"):
            prefix = f"{variant}_color"
            block, status = _color_features(image_by_variant[variant], masks[variant], prefix,
                                            phase4.get("color_proportions", {}))
            output.update(block)
            statuses_color[variant] = status
        kmeans_statuses: dict[str, str] = {}
        if task_key == "defect":
            for variant in ("original", "gray_world"):
                for color_space, k, key in (("Lab", 3, "lab_k3"), ("HSV", 5, "hsv_k5")):
                    prefix = f"{variant}_{key}"
                    mask = masks[variant]
                    if mask is None:
                        status = statuses[variant]
                        block = _empty_features(_kmeans_feature_names(prefix))
                    else:
                        seed = _derived_seed(int(phase3.get("seed", 20261023)), row["image_id"],
                                             variant, color_space, k)
                        result = cluster_and_detect(
                            image_by_variant[variant], mask, color_space, k, seed,
                            attempts=int(phase3.get("attempts", 1)),
                            max_iterations=int(phase3.get("max_iterations", 30)),
                            epsilon=float(phase3.get("epsilon", 0.001)),
                            max_fit_pixels=int(phase3.get("max_fit_pixels", 4000)),
                            min_fruit_pixels=int(phase3.get("min_fruit_pixels", 100)),
                            rule=dict(phase3.get("selection_rule", {})),
                        )
                        status = str(result.get("status", "unknown"))
                        block = _kmeans_features_from_outcome(image_by_variant[variant], mask, result, prefix)
                    kmeans_statuses[prefix] = status
                    output.update(block)

        manifest["original_lab_k3_status"] = kmeans_statuses.get("original_lab_k3", "not_applicable")
        manifest["gray_world_lab_k3_status"] = kmeans_statuses.get("gray_world_lab_k3", "not_applicable")
        manifest["original_hsv_k5_status"] = kmeans_statuses.get("original_hsv_k5", "not_applicable")
        manifest["gray_world_hsv_k5_status"] = kmeans_statuses.get("gray_world_hsv_k5", "not_applicable")
        group_status: dict[str, str] = {
            "original_color": statuses_color["original"],
            "gray_world_color": statuses_color["gray_world"],
            **kmeans_statuses,
        }
        required_groups = list(feature_sets)
        manifest["feature_status"] = "complete" if all(group_status.get(g) == "ok" for g in required_groups) else "partial"
        feature_values = [output.get(column, "") for column in feature_columns]
        manifest["feature_column_count"] = len(feature_columns)
        manifest["missing_feature_count"] = sum(
            value == "" or value is None or (isinstance(value, float) and not np.isfinite(value))
            for value in feature_values
        )
        manifest["cache_hit"] = "false"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"cache_key": cache_key, "features": output, "manifest": manifest},
                                        ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(cache_path)
        return output, manifest
    except Exception as exc:
        manifest["image_status"] = "error"
        manifest["feature_status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:500]
        manifest["feature_column_count"] = len(feature_columns)
        manifest["missing_feature_count"] = len(feature_columns)
        return output, manifest


def _load_task_rows(config: dict[str, Any], phase4: dict[str, Any], task_key: str,
                    manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    task_settings = phase4["tasks"][task_key]
    split_path = project_path(config, task_settings["split_path"])
    if not split_path.is_file():
        raise FileNotFoundError(f"Phase 0.2 split file is missing: {split_path}")
    split_rows = _read_csv(split_path)
    manifest_by_id = {row.get("image_id", ""): row for row in manifest_rows}
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for split_row in split_rows:
        image_id = split_row.get("image_id", "")
        if not image_id or image_id in seen:
            raise ValueError(f"Empty or duplicate image_id in split {split_path}: {image_id}")
        seen.add(image_id)
        master = manifest_by_id.get(image_id)
        if not master:
            raise ValueError(f"Split image_id not present in Phase 0.2 manifest: {image_id}")
        eligible_field = "eligible_for_ripeness" if task_key == "ripeness" else "eligible_for_defect_analysis"
        label_field = "ripeness_label" if task_key == "ripeness" else "defect_label"
        if master.get("dataset") != task_settings["dataset"] or master.get("task") != task_settings["task"]:
            raise ValueError(f"Split row {image_id} has mismatched dataset/task in manifest")
        if not _truthy(master.get(eligible_field, "")):
            raise ValueError(f"Split contains ineligible row {image_id}")
        if not master.get(label_field, "").strip():
            raise ValueError(f"Split contains a missing target label for {image_id}")
        if split_row.get("phase02_split") not in {"train", "validation", "test"}:
            raise ValueError(f"Split value is invalid for {image_id}")
        merged = dict(master)
        merged.update(split_row)
        merged[label_field] = master[label_field]
        result.append(merged)
    return result


def _write_feature_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    _atomic_csv(path, rows, list(METADATA_FIELDS) + columns)


def _numeric(values: list[Any]) -> np.ndarray:
    parsed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            parsed.append(number)
    return np.asarray(parsed, dtype=np.float64)


def _summarize_group(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = _numeric([row.get(feature, "") for row in rows])
    if not values.size:
        return {"n": 0, "mean": "", "std": "", "median": "", "p10": "", "p90": ""}
    p10, p90 = np.percentile(values, (10, 90))
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)), "p10": float(p10), "p90": float(p90)}


def _distribution_audit(
    config: dict[str, Any], phase4: dict[str, Any], defect_rows: list[dict[str, Any]],
    defect_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Describe, but do not train on, the C segmentation cohort using cached Phase 3 outputs."""
    phase02_rows = _read_csv(project_path(config, phase4.get("manifest_path", "data/metadata/phase02_manifest.csv")))
    seg_rows = [row for row in phase02_rows if row.get("dataset") == "mango_segmentation"
                and row.get("task") == "defect_segmentation" and row.get("mask_type") == "defect_region"
                and _truthy(row.get("eligible_for_segmentation", ""))]
    phase3_path = project_path(config, config.get("phase3", {}).get("results_path", "reports/phase03_results.csv"))
    phase3_rows = _read_csv(phase3_path)
    lookup = {(row.get("image_id"), row.get("variant"), row.get("color_space"), row.get("k")): row
              for row in phase3_rows if row.get("mode") == "heuristic"}
    seg_features: list[dict[str, Any]] = []
    audit_status = Counter()
    for row in seg_rows:
        try:
            image_path = project_path(config, row["image_path"])
            if not image_path.is_file():
                audit_status["image_missing"] += 1
                continue
            source_hash = _sha256_file(image_path)
            if row.get("sha256") and source_hash != row["sha256"]:
                audit_status["source_hash_mismatch"] += 1
                continue
            frame = decode_image(image_path)
            rgb = frame.rgb
            balanced = gray_world(rgb, frame.alpha).rgb
            item = _empty_dataset_row(_metadata(row, source_hash), defect_columns)
            for variant, phase_variant in (("original", "original"), ("gray_world", "gray_world")):
                lab_rec = lookup.get((row["image_id"], phase_variant, "Lab", "3"))
                hsv_rec = lookup.get((row["image_id"], phase_variant, "HSV", "5"))
                base_rec = lab_rec or hsv_rec
                fruit_mask = None
                if base_rec and base_rec.get("fruit_mask_status") == "accepted_heuristic":
                    mask_path = project_path(config, base_rec.get("fruit_mask_path", ""))
                    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.is_file() else None
                    if raw_mask is not None and raw_mask.ndim == 2 and raw_mask.shape == rgb.shape[:2] \
                            and raw_mask.dtype == np.uint8 and np.isin(raw_mask, (0, 255)).all():
                        fruit_mask = raw_mask.copy()
                        if frame.alpha is not None:
                            fruit_mask[frame.alpha == 0] = 0
                        if int(np.count_nonzero(fruit_mask)) < 100:
                            fruit_mask = None
                variant_rgb = rgb if variant == "original" else balanced
                block, _ = _color_features(variant_rgb, fruit_mask, f"{variant}_color",
                                           phase4.get("color_proportions", {}))
                item.update(block)
                for color_space, k, key in (("Lab", 3, "lab_k3"), ("HSV", 5, "hsv_k5")):
                    record = lookup.get((row["image_id"], phase_variant, color_space, str(k)))
                    prefix = f"{variant}_{key}"
                    block, status = _kmeans_features_from_saved(variant_rgb, fruit_mask, record, config, prefix)
                    item.update(block)
                    audit_status[f"{variant}_{key}_{status}"] += 1
            seg_features.append(item)
            audit_status["processed"] += 1
        except Exception as exc:
            audit_status[f"error_{type(exc).__name__}"] += 1

    groups: dict[str, list[dict[str, Any]]] = {
        "C_segmentation_cohort": seg_features,
        "C_classification_all": defect_rows,
        "C_classification_healthy": [row for row in defect_rows if row.get("label") == "healthy"],
        "C_classification_defective": [row for row in defect_rows if row.get("label") == "defective"],
    }
    output: list[dict[str, Any]] = []
    for cohort_name, cohort_rows in groups.items():
        for feature in defect_columns:
            summary = _summarize_group(cohort_rows, feature)
            output.append({"cohort": cohort_name, "feature": feature, **summary,
                           "smd_vs_C_segmentation": ""})
    seg_by_feature = {row["feature"]: row for row in output if row["cohort"] == "C_segmentation_cohort"}
    class_by_feature = {row["feature"]: row for row in output if row["cohort"] == "C_classification_all"}
    for row in output:
        if row["cohort"] != "C_classification_all":
            continue
        left, right = seg_by_feature[row["feature"]], class_by_feature[row["feature"]]
        try:
            left_std, right_std = float(left["std"]), float(right["std"])
            pooled = math.sqrt((left_std * left_std + right_std * right_std) / 2.0)
            row["smd_vs_C_segmentation"] = (float(right["mean"]) - float(left["mean"])) / pooled if pooled else ""
        except (TypeError, ValueError, ZeroDivisionError):
            row["smd_vs_C_segmentation"] = ""
    meta = {"segmentation_candidate_count": len(seg_rows), "segmentation_processed_count": len(seg_features),
            "segmentation_status_counts": dict(audit_status)}
    return output, meta


def _feature_schema(task_rows: dict[str, list[dict[str, Any]]],
                    sets_by_task: dict[str, dict[str, list[str]]],
                    thresholds: dict[str, Any]) -> dict[str, Any]:
    feature_columns = {task: [name for cols in sets_by_task[task].values() for name in cols]
                       for task in sets_by_task}
    definitions: dict[str, dict[str, str]] = {}
    for columns in feature_columns.values():
        for name in columns:
            prefix, metric = name.split("__", 1)
            if metric == "hsv_h_circular_resultant":
                unit = "resultant [0,1]"
            elif metric.startswith("hsv_h_") or metric == "hsv_h_circular_mean":
                unit = "degrees [0,360)"
            elif metric.startswith("hsv_"):
                unit = "[0,1]"
            elif metric.startswith("lab_l_"):
                unit = "[0,100]"
            elif metric.startswith("lab_"):
                unit = "signed CIELAB units"
            elif metric.endswith("_fraction"):
                unit = "fraction [0,1]"
            elif metric == "detected_suspicious":
                unit = "binary indicator [0,1]"
            elif metric == "suspicious_cluster_count":
                unit = "count"
            elif metric.startswith("region_hsv_h_"):
                unit = "degrees [0,360)" if metric.endswith("circular_mean") else "resultant [0,1]"
            elif metric.startswith("region_hsv_"):
                unit = "[0,1]"
            elif metric.startswith("region_lab_l_"):
                unit = "[0,100]"
            elif metric.startswith("region_lab_"):
                unit = "signed CIELAB units"
            else:
                unit = "pipeline-defined scalar; see feature_origin and Phase 3 selection-rule config"
            definitions[name] = {
                "feature_set": prefix,
                "unit_or_scale": unit,
                "description": _feature_description(metric, prefix),
            }
    return {
        "schema_version": 1, "pipeline_version": PIPELINE_VERSION,
        "description": "CSV feature tables have one row per image_id. Only columns listed under feature_columns are numeric model inputs.",
        "metadata_columns": list(METADATA_FIELDS),
        "target_columns": {"ripeness": "label (ripeness target)", "defect": "label (healthy/defective target)"},
        "split_column": "phase02_split",
        "feature_columns": feature_columns,
        "feature_sets": sets_by_task,
        "feature_counts": {task: len(columns) for task, columns in feature_columns.items()},
        "feature_definitions": definitions,
        "all_feature_columns": sorted(set(name for cols in feature_columns.values() for name in cols)),
        "analysis_resolution": {
            "max_side_pixels": int(thresholds.get("max_analysis_side", 512)),
            "downsampling": "aspect-preserving OpenCV INTER_AREA; alpha mask uses nearest-neighbor",
            "raw_source": "raw files are hash-verified and never modified; features use only the analysis raster",
        },
        "color_proportion_definition": {
            "thresholds": thresholds.get("color_proportions", {}),
            "denominator": "pixels within an accepted heuristic fruit mask and visible alpha region",
            "hue_units": "degrees [0,360); ranges are half-open; hue outside named ranges is not counted",
        },
        "color_scales": {"HSV": {"H": "degrees [0,360)", "S": "[0,1]", "V": "[0,1]"},
                         "CIELAB": {"L": "[0,100]", "a": "signed", "b": "signed"}},
        "feature_origin": {
            "original_color": "HSV/CIELAB pixel statistics within accepted Phase 2 heuristic fruit mask on original RGB",
            "gray_world_color": "same statistics after Phase 1 Gray World using its corresponding Phase 2 heuristic fruit mask",
            "original_lab_k3": "Phase 3 K-Means Lab K=3 on original RGB and accepted heuristic fruit mask",
            "original_hsv_k5": "Phase 3 K-Means HSV K=5 on original RGB and accepted heuristic fruit mask",
            "gray_world_lab_k3": "comparative Gray World K-Means Lab K=3 features",
            "gray_world_hsv_k5": "comparative Gray World K-Means HSV K=5 features",
        },
        "missing_values": "Blank CSV cells mean unavailable; no imputation is performed. If a valid K-Means run has no selected cluster, detection/area are 0 while suspicious-region color moments remain blank.",
        "excluded_from_model_input": [
            "image_id", "dataset", "task", "label", "phase02_split", "image_path", "source_sha256",
            "split_risk_status", "official_accuracy_eligible", "all manifest/status fields",
            "all defect ground-truth mask paths/pixels/metrics", "source filenames and folders",
        ],
        "leakage_controls": [
            "labels, splits, ground-truth masks, and source metadata are never passed to feature functions",
            "ripeness and defect tables are built from the saved Phase 0.2 split CSVs without modifying them",
            "defect K-Means branches share one row/image_id and do not multiply training sample count",
        ],
        "sample_counts": {task: len(rows) for task, rows in task_rows.items()},
    }


def _feature_description(metric: str, prefix: str) -> str:
    if metric.endswith("_fraction"):
        if metric == "green_fraction":
            return "Fraction of accepted fruit-mask pixels in configured green HSV hue range."
        if metric == "yellow_fraction":
            return "Fraction of accepted fruit-mask pixels in configured yellow HSV hue range."
        if metric == "orange_fraction":
            return "Fraction of accepted fruit-mask pixels in configured orange HSV hue range."
        if metric == "fruit_area_fraction":
            return "Fruit-mask area divided by visible analysis-raster area."
        if metric == "suspicious_area_fraction":
            return "Selected suspicious K-Means pixels divided by accepted fruit-mask pixels."
        if metric == "selected_max_component_fraction":
            return "Largest selected connected component divided by accepted fruit-mask pixels."
    if metric.startswith("hsv_h_circular"):
        return "Circular hue statistic in HSV for accepted fruit-mask pixels."
    if metric.startswith("hsv_") or metric.startswith("lab_"):
        return "Configured channel statistic over accepted fruit-mask pixels; statistic is encoded in the column suffix."
    if metric == "detected_suspicious":
        return "Whether Phase 3 selected at least one suspicious cluster."
    if metric == "suspicious_cluster_count":
        return "Number of Phase 3 selected suspicious clusters."
    if metric == "dominant_cluster_fraction":
        return "Largest K-Means cluster area divided by accepted fruit-mask pixels."
    if metric.startswith("selected_"):
        return "Aggregate Phase 3 diagnostic for selected suspicious clusters; exact quantity is encoded in the column name."
    if metric.startswith("region_"):
        return "Color statistic within Phase 3 predicted suspicious pixels intersected with the accepted fruit mask."
    return f"Phase 4 scalar from the {prefix} feature branch."


def _label_feature_summary(
    feature_tables: dict[str, list[dict[str, Any]]], schema: dict[str, Any],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for task, rows in feature_tables.items():
        labels = sorted({str(row.get("label", "")) for row in rows if row.get("label", "") != ""})
        for label in labels:
            group = [row for row in rows if row.get("label") == label]
            for feature in schema["feature_columns"][task]:
                values = _numeric([row.get(feature, "") for row in group])
                if values.size:
                    percentiles = np.percentile(values, (10, 25, 50, 75, 90))
                    mean, std, minimum, maximum = (float(values.mean()), float(values.std()),
                                                    float(values.min()), float(values.max()))
                    p10, p25, median, p75, p90 = map(float, percentiles)
                else:
                    mean = std = minimum = maximum = p10 = p25 = median = p75 = p90 = ""
                summary_rows.append({
                    "task": task, "label": label, "feature": feature, "n_images": len(group),
                    "n_nonmissing": int(values.size), "mean": mean, "std": std,
                    "min": minimum, "p10": p10, "p25": p25, "median": median,
                    "p75": p75, "p90": p90, "max": maximum,
                })
    return summary_rows


def _report(
    path: Path,
    feature_tables: dict[str, list[dict[str, Any]]],
    manifest_rows: list[dict[str, Any]],
    schema: dict[str, Any],
    distribution_rows: list[dict[str, Any]],
    distribution_meta: dict[str, Any],
) -> None:
    lines = [
        "# MangoResearch — Phase 4: tích hợp pipeline và trích xuất đặc trưng", "",
        "## Tóm tắt", "",
        "Pipeline dùng ảnh RGB gốc, Gray World từ Phase 1, fruit mask heuristic Phase 2 và K-Means Phase 3. Không huấn luyện mô hình. Label và split chỉ được giữ làm target/metadata; không nằm trong các cột feature được schema cho phép.", "",
    ]
    for task, rows in feature_tables.items():
        feature_columns = schema["feature_columns"][task]
        expected_task = "ripeness_classification" if task == "ripeness" else "defect_analysis"
        manifest_subset = [row for row in manifest_rows if row.get("task") == expected_task]
        status_counts = Counter(row.get("feature_status", "") for row in manifest_subset)
        label_counts = Counter(str(row.get("label", "")) for row in rows)
        split_counts = Counter(str(row.get("phase02_split", "")) for row in rows)
        successful = sum(row.get("image_status") == "ok" for row in manifest_subset)
        lines.extend([
            f"### {task.title()}", "",
            f"- Số mẫu trong bảng: {len(rows):,}; đọc/hash ảnh thành công: {successful:,}; lỗi ảnh/nguồn: {len(rows)-successful:,}.",
            f"- Trạng thái mức feature row: `{dict(status_counts)}`; số feature numeric: {len(feature_columns)}; số feature sets: {len(schema['feature_sets'][task])}.",
            f"- Nhãn mục tiêu (metadata): `{dict(label_counts)}`.",
            f"- Split Phase 0.2 được giữ nguyên: `{dict(split_counts)}`.",
        ])
        if task == "ripeness":
            groups = ("original_color", "gray_world_color")
            shown_features = [
                "original_color__hsv_h_circular_mean", "original_color__hsv_s_mean",
                "original_color__hsv_v_mean", "original_color__lab_l_mean",
                "original_color__green_fraction", "original_color__yellow_fraction",
                "original_color__orange_fraction",
            ]
        else:
            groups = tuple(schema["feature_sets"][task])
            shown_features = [
                "original_color__hsv_s_mean", "original_color__hsv_v_mean", "original_color__lab_l_mean",
                "original_color__green_fraction", "original_color__yellow_fraction", "original_color__orange_fraction",
                "original_lab_k3__suspicious_area_fraction", "original_lab_k3__selected_mean_color_distance",
                "original_hsv_k5__suspicious_area_fraction", "original_hsv_k5__selected_mean_color_distance",
            ]
        lines.extend(["", "| Feature set | Ảnh có branch hợp lệ | Ô feature thiếu |", "|---|---:|---:|"])
        for group in groups:
            columns = schema["feature_sets"][task][group]
            if group in {"original_color", "gray_world_color"}:
                branch_status = "original_fruit_mask_status" if group == "original_color" else "gray_world_fruit_mask_status"
                usable = sum(row.get(branch_status) == "accepted_heuristic" for row in manifest_subset)
            else:
                usable = sum(row.get(group + "_status") == "ok" for row in manifest_subset)
            total_cells = max(1, len(rows) * len(columns))
            missing = sum(row.get(name, "") == "" or row.get(name) is None for row in rows for name in columns)
            lines.append(f"| `{group}` | {usable:,}/{len(rows):,} | {missing:,}/{total_cells:,} ({missing/total_cells:.2%}) |")
        selected_for_report = [name for name in shown_features if name in feature_columns]
        lines.extend(["", "Phân bố mô tả theo nhãn (không dùng để chọn cấu hình):", ""])
        if selected_for_report:
            lines.append("| Nhãn | N | " + " | ".join(selected_for_report) + " |")
            lines.append("|---|---:|" + "---:|" * len(selected_for_report))
            for label in sorted(label_counts):
                group_rows = [row for row in rows if row.get("label") == label]
                means = []
                for name in selected_for_report:
                    summary = _summarize_group(group_rows, name)
                    means.append("—" if summary["n"] == 0 else f"{float(summary['mean']):.4g} (n={summary['n']})")
                lines.append(f"| {label} | {len(group_rows)} | " + " | ".join(means) + " |")
        lines.append("")

    lines.extend([
        "## So sánh phân bố Dataset C segmentation và classification", "",
        f"Feature extraction mô tả cho nhóm segmentation dùng {distribution_meta.get('segmentation_processed_count',0):,}/{distribution_meta.get('segmentation_candidate_count',0):,} ảnh đủ điều kiện, dựa trên Phase 3 predictions và không đọc ground-truth defect mask để tạo feature. Nhóm classification có {len(feature_tables.get('defect', [])):,} ảnh; đây là các cohort khác nhau về nguồn, số lượng và class mix.",
        "So sánh mean/std/median/percentile và standardized mean difference (SMD) chỉ nhằm nhận diện khác biệt phân bố; không suy ra hiệu quả phân loại hoặc khả năng chuyển giao chất lượng K-Means. Chi tiết đủ feature/cohort nằm ở `reports/phase04_distribution_comparison.csv`.",
        f"Thống kê mean/std/min/max và percentiles cho toàn bộ feature theo từng nhãn được lưu tại `reports/phase04_label_feature_summary.csv` ({sum(len(schema['feature_columns'][task]) * len({str(row.get('label','')) for row in rows if row.get('label','')}) for task, rows in feature_tables.items()):,} dòng).",
        f"- Trạng thái xử lý cohort segmentation: `{distribution_meta.get('segmentation_status_counts', {})}`.",
        "", "| Feature | SMD: classification all vs segmentation | n segmentation | n classification |", "|---|---:|---:|---:|",
    ])
    top = [row for row in distribution_rows if row.get("cohort") == "C_classification_all" and row.get("smd_vs_C_segmentation") not in ("", None)]
    top.sort(key=lambda row: abs(float(row["smd_vs_C_segmentation"])), reverse=True)
    for row in top[:12]:
        seg_row = next((x for x in distribution_rows if x.get("cohort") == "C_segmentation_cohort" and x.get("feature") == row["feature"]), {})
        lines.append(f"| `{row['feature']}` | {float(row['smd_vs_C_segmentation']):.3f} | {seg_row.get('n',0)} | {row.get('n',0)} |")

    successful_manifests = [row for row in manifest_rows if row.get("image_status") == "ok"]
    total_missing = 0
    total_cells = 0
    for task, rows in feature_tables.items():
        cols = schema["feature_columns"][task]
        total_cells += len(rows) * len(cols)
        total_missing += sum(row.get(name, "") == "" or row.get(name) is None for row in rows for name in cols)
    lines.extend([
        "", "## Missing values, provenance, and leakage controls", "",
        f"- Tỷ lệ ô feature trống gộp: {total_missing:,}/{total_cells:,} ({total_missing/max(1,total_cells):.2%}); manifest phân biệt mask/K-Means unavailable, lỗi ảnh và partial row.",
        f"- SHA nguồn được xác minh thành công: {len(successful_manifests):,}/{len(manifest_rows):,} mẫu. Không ghi/sửa raw dataset, manifest Phase 0.2 hay split Phase 0.2.",
        "- `label`, `phase02_split`, `image_path`, `source_sha256`, `dataset` và trạng thái chất lượng là target/metadata. `feature_schema.json` liệt kê input columns; ground-truth defect mask, split và source metadata không được truyền vào hàm trích xuất.",
        "- Giá trị thiếu để trống, không điền 0. Nếu fruit mask heuristic không hợp lệ, color stats và K-Means block tương ứng bị để trống. Nếu K-Means chạy hợp lệ nhưng không chọn cụm nghi vấn, detection và area fraction là 0; thống kê màu vùng nghi vấn để trống.",
        f"- Feature được tính trên raster tạm aspect-preserving, cạnh dài tối đa {schema['analysis_resolution']['max_side_pixels']} px; color conversion, Gray World, fruit segmentation và K-Means cùng dùng raster này. Kích thước gốc và kích thước phân tích được ghi riêng trong extraction manifest.",
        "", "## Giới hạn và sẵn sàng cho Phase 5", "",
        "- Dataset C segmentation có augmentation lineage unresolved; Phase 0.2 defect split còn residual near-duplicate/lineage risk, và `official_accuracy_eligible=false`. Không suy luận lại quan hệ ảnh gốc/augmented.",
        "- Phase 2 fruit masks là heuristic, chưa có approved whole-fruit ground truth. Đặc trưng vùng quả exploratory và một phần có thể missing.",
        "- Phase 3 predictions chỉ dùng để tạo feature ở cohort segmentation; GT defect mask và metric không được dùng. K-Means Lab K=3 và HSV K=5 là hai feature set song song trên cùng mỗi image_id, không nhân đôi sample.",
        "- Sẵn sàng cho Phase 5 khi loader chỉ chọn `feature_columns`, giữ target/split riêng và xử lý missing values trong một bước thí nghiệm được khai báo. Phase 4 không huấn luyện classifier.",
        "", "## Output", "",
        "- `data/features/ripeness_features.csv`", "- `data/features/defect_features.csv`",
        "- `data/features/feature_schema.json`", "- `data/features/feature_extraction_manifest.csv`",
        "- `reports/phase04_distribution_comparison.csv`", "- `reports/phase04_label_feature_summary.csv`",
        "- `reports/phase04_report.md`", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase4(config: dict[str, Any], tasks: list[str] | None = None,
               limit: int | None = None, force: bool = False) -> Path:
    phase4 = dict(config.get("phase4", {}))
    if not phase4:
        raise ValueError("Missing phase4 configuration in config.yaml")
    phase3 = dict(config.get("phase3", {}))
    selected_tasks = tasks or ["ripeness", "defect"]
    invalid = sorted(set(selected_tasks) - {"ripeness", "defect"})
    if invalid:
        raise ValueError(f"Unsupported Phase 4 task(s): {invalid}")
    master_rows = _read_csv(project_path(config, phase4.get("manifest_path", "data/metadata/phase02_manifest.csv")))
    output_dir = project_path(config, phase4.get("output_dir", "data/features"))
    cache_dir = project_path(config, phase4.get("cache_dir", "data/features/cache"))
    generated_fruit_manifest = project_path(config, phase4.get("fruit_mask_manifest", "data/metadata/phase04_fruit_masks.csv"))

    provider_config = dict(config)
    provider_config["phase3"] = dict(phase3)
    provider_config["segmentation"] = dict(config.get("segmentation", {}))
    provider_config["segmentation"]["mask_dir"] = phase4.get("fruit_mask_dir", "data/processed/phase4_fruit_masks")
    provider = Phase2FruitMaskProvider(provider_config, provider_config["segmentation"], generated_fruit_manifest)

    all_features: dict[str, list[dict[str, Any]]] = {}
    all_manifests: list[dict[str, Any]] = []
    all_columns: dict[str, list[str]] = {}
    feature_sets_by_task: dict[str, dict[str, list[str]]] = {}
    for task_key in selected_tasks:
        task_rows = _load_task_rows(config, phase4, task_key, master_rows)
        if limit is not None:
            task_rows = task_rows[:limit]
        columns, feature_sets = _build_feature_columns(task_key)
        all_columns[task_key] = columns
        feature_sets_by_task[task_key] = feature_sets
        task_features: list[dict[str, Any]] = []
        task_manifests: list[dict[str, Any]] = []
        for index, row in enumerate(task_rows, start=1):
            feature_row, manifest_row = _process_classification_image(
                row, task_key, provider_config, provider, phase4, phase3, columns,
                feature_sets, cache_dir, force,
            )
            task_features.append(feature_row)
            task_manifests.append(manifest_row)
            if index % max(1, int(phase4.get("checkpoint_images", 25))) == 0:
                provider.checkpoint()
                _write_feature_table(output_dir / f"{task_key}_features.csv", task_features, columns)
                LOGGER.info("Phase 4 %s: processed %d/%d images", task_key, index, len(task_rows))
        provider.checkpoint()
        _write_feature_table(output_dir / f"{task_key}_features.csv", task_features, columns)
        all_features[task_key] = task_features
        all_manifests.extend(task_manifests)

    manifest_output = output_dir / "feature_extraction_manifest.csv"
    existing_manifest = _read_csv(manifest_output)
    merged_manifests = {(row.get("task"), row.get("image_id")): row for row in existing_manifest}
    for row in all_manifests:
        merged_manifests[(row.get("task"), row.get("image_id"))] = row
    final_manifests = list(merged_manifests.values())
    _atomic_csv(manifest_output, final_manifests, MANIFEST_FIELDS)

    distribution_rows: list[dict[str, Any]] = []
    distribution_meta: dict[str, Any] = {"segmentation_candidate_count": 0, "segmentation_processed_count": 0,
                                         "segmentation_status_counts": {"skipped_task_limited_run": 1}}
    if "defect" in all_features and limit is None:
        distribution_rows, distribution_meta = _distribution_audit(
            config, phase4, all_features["defect"], all_columns["defect"]
        )
        _atomic_csv(project_path(config, phase4.get("distribution_report_csv", "reports/phase04_distribution_comparison.csv")),
                    distribution_rows, ("cohort", "feature", "n", "mean", "std", "median", "p10", "p90", "smd_vs_C_segmentation"))
    elif "defect" in all_features:
        distribution_meta = {"segmentation_candidate_count": 0, "segmentation_processed_count": 0,
                             "segmentation_status_counts": {"skipped_sample_limited_run": 1}}

    schema_rows = dict(all_features)
    for task_key in ("ripeness", "defect"):
        if task_key not in schema_rows:
            existing_path = output_dir / f"{task_key}_features.csv"
            schema_rows[task_key] = _read_csv(existing_path) if existing_path.is_file() else []
            all_columns[task_key], feature_sets_by_task[task_key] = _build_feature_columns(task_key)
    schema = _feature_schema(schema_rows, feature_sets_by_task, {
        "color_proportions": phase4.get("color_proportions", {}),
        "max_analysis_side": phase4.get("max_analysis_side", 512),
    })
    label_summary = _label_feature_summary(schema_rows, schema)
    _atomic_csv(project_path(config, phase4.get(
        "label_summary_csv", "reports/phase04_label_feature_summary.csv")), label_summary,
        ("task", "label", "feature", "n_images", "n_nonmissing", "mean", "std", "min",
         "p10", "p25", "median", "p75", "p90", "max"))
    schema_path = project_path(config, phase4.get("schema_path", "data/features/feature_schema.json"))
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = project_path(config, phase4.get("report_path", "reports/phase04_report.md"))
    _report(report_path, schema_rows, final_manifests, schema, distribution_rows, distribution_meta)
    LOGGER.info("Phase 4 complete: %s", report_path)
    return report_path
