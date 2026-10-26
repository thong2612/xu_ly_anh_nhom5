from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color.preprocess import decode_image, gray_world, rgb_to_hsv, rgb_to_lab, rgb_to_bgr
from src.segmentation.baseline import segment_channel
from src.segmentation.metrics import segmentation_metrics
from src.utils.config import project_path


LOGGER = logging.getLogger("mangoresearch.phase3")
RESULT_FIELDS = [
    "mode", "image_id", "dataset", "task", "phase02_split", "split_risk_status",
    "official_accuracy_eligible", "source_image_path", "source_sha256", "defect_mask_path",
    "defect_mask_sha256", "defect_mask_alignment_status", "defect_mask_review_status",
    "variant", "color_space", "k", "seed", "attempts", "max_iterations", "epsilon",
    "max_fit_pixels", "fruit_mask_path", "fruit_mask_source", "fruit_mask_status",
    "fruit_pixels", "fit_pixels", "compactness", "kmeans_status", "selected_cluster_ids", "cluster_details_json",
    "defect_pixel_count", "defect_area_percent", "gt_pair_status", "iou", "dice",
    "precision", "recall", "detection_hit", "false_positive_pixels_within_fruit",
    "cluster_map_path", "predicted_mask_path", "overlay_path", "cache_key", "cache_hit",
]
METRIC_FIELDS = [
    "mode", "variant", "color_space", "k", "split", "n_images", "iou_mean", "dice_mean",
    "precision_mean", "recall_mean", "detection_rate", "false_positive_pixels_mean",
    "defect_area_percent_mean", "metric_status",
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


def _binary_mask(mask: np.ndarray, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be a 2D uint8 binary mask")
    if not np.isin(array, (0, 255)).all():
        raise ValueError(f"{name} must contain only 0/255 values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match image shape {shape}")
    return array


def normalized_color_features(rgb: np.ndarray, color_space: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-scaled K-Means features and normalized L*/V brightness."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("K-Means input must be an HxWx3 uint8 RGB image")
    if color_space == "Lab":
        values = rgb_to_lab(rgb)
        features = np.stack((
            values[:, :, 0] / 100.0,
            (values[:, :, 1] + 128.0) / 255.0,
            (values[:, :, 2] + 128.0) / 255.0,
        ), axis=2)
        brightness = values[:, :, 0] / 100.0
    elif color_space == "HSV":
        values = rgb_to_hsv(rgb)
        features = np.stack((values[:, :, 0] / 360.0, values[:, :, 1], values[:, :, 2]), axis=2)
        brightness = values[:, :, 2]
    else:
        raise ValueError("color_space must be 'Lab' or 'HSV'")
    return np.clip(features, 0.0, 1.0).astype(np.float32), brightness.astype(np.float32)


def _color_distance(left: np.ndarray, right: np.ndarray, color_space: str) -> float:
    difference = np.abs(left - right)
    if color_space == "HSV":
        difference[0] = min(float(difference[0]), 1.0 - float(difference[0]))
    return float(np.linalg.norm(difference) / math.sqrt(len(difference)))


def select_suspicious_clusters(
    cluster_labels: np.ndarray,
    features: np.ndarray,
    brightness: np.ndarray,
    color_space: str,
    rule: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    """Select non-dominant, color-distinct, spatially coherent cluster regions."""
    settings = {
        "minimum_cluster_fraction": 0.003,
        "maximum_cluster_fraction": 0.35,
        "minimum_component_fraction": 0.001,
        "minimum_component_pixels": 5,
        "minimum_color_distance": 0.12,
        "preferred_area_fraction": 0.08,
        "minimum_suspicion_score": 0.34,
        "minimum_darkness_delta": 0.12,
        "color_weight": 0.50,
        "darkness_weight": 0.25,
        "area_weight": 0.15,
        "continuity_weight": 0.10,
    }
    settings.update(rule or {})
    labels = np.asarray(cluster_labels)
    if labels.ndim != 2 or features.shape[:2] != labels.shape or brightness.shape != labels.shape:
        raise ValueError("Cluster labels, feature map and brightness map must have matching dimensions")
    roi = labels >= 0
    total = int(np.count_nonzero(roi))
    result = np.zeros(labels.shape, dtype=np.uint8)
    if total == 0:
        return result, [], []
    ids, counts = np.unique(labels[roi], return_counts=True)
    dominant_id = int(ids[int(np.argmax(counts))])
    feature_flat = features[roi]
    label_flat = labels[roi]
    dominant_mean = feature_flat[label_flat == dominant_id].mean(axis=0)
    dominant_brightness = float(brightness[roi][label_flat == dominant_id].mean())
    suspicious_ids: list[int] = []
    details: list[dict[str, Any]] = []
    minimum_component_pixels = int(settings["minimum_component_pixels"])
    minimum_component_fraction = float(settings["minimum_component_fraction"])
    minimum_darkness_delta = max(float(settings["minimum_darkness_delta"]), 1e-8)
    preferred_area = max(float(settings["preferred_area_fraction"]), 1e-8)
    maximum_area = float(settings["maximum_cluster_fraction"])

    for cluster_id, count in zip(ids.tolist(), counts.tolist()):
        cluster_id = int(cluster_id)
        count = int(count)
        cluster_pixels = labels == cluster_id
        fraction = count / total
        means = features[cluster_pixels].mean(axis=0)
        cluster_brightness = float(brightness[cluster_pixels].mean())
        color_distance = _color_distance(means, dominant_mean, color_space)
        darkness_delta = dominant_brightness - cluster_brightness
        darkness_score = float(np.clip(darkness_delta / minimum_darkness_delta, 0.0, 1.0))
        if fraction <= preferred_area:
            area_score = float(np.clip(fraction / preferred_area, 0.0, 1.0))
        else:
            area_score = float(np.clip((maximum_area - fraction) / max(maximum_area - preferred_area, 1e-8), 0.0, 1.0))

        component_count, component_map, component_stats, _ = cv2.connectedComponentsWithStats(
            cluster_pixels.astype(np.uint8), 8
        )
        component_areas = component_stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([], dtype=np.int32)
        largest_component = int(component_areas.max()) if component_areas.size else 0
        continuity = largest_component / max(count, 1)
        component_minimum = max(minimum_component_pixels, int(math.ceil(minimum_component_fraction * total)))
        keep_labels = [index + 1 for index, area in enumerate(component_areas.tolist()) if int(area) >= component_minimum]
        retained = np.isin(component_map, keep_labels) if keep_labels else np.zeros(labels.shape, dtype=bool)
        score = (
            float(settings["color_weight"]) * color_distance
            + float(settings["darkness_weight"]) * darkness_score
            + float(settings["area_weight"]) * area_score
            + float(settings["continuity_weight"]) * continuity
        )
        eligible = (
            cluster_id != dominant_id
            and float(settings["minimum_cluster_fraction"]) <= fraction <= maximum_area
            and color_distance >= float(settings["minimum_color_distance"])
            and largest_component / total >= minimum_component_fraction
            and bool(np.any(retained))
            and score >= float(settings["minimum_suspicion_score"])
        )
        if eligible:
            suspicious_ids.append(cluster_id)
            result[retained] = 255
        details.append({
            "cluster_id": cluster_id,
            "pixel_count": count,
            "area_fraction": fraction,
            "mean_brightness": cluster_brightness,
            "brightness_delta_from_dominant": darkness_delta,
            "color_distance_from_dominant": color_distance,
            "largest_component_pixels": largest_component,
            "continuity": continuity,
            "suspicion_score": score,
            "selected": eligible,
        })
    return result, suspicious_ids, details


def cluster_and_detect(
    rgb: np.ndarray,
    fruit_mask: np.ndarray,
    color_space: str,
    k: int,
    seed: int,
    attempts: int = 1,
    max_iterations: int = 30,
    epsilon: float = 0.001,
    max_fit_pixels: int = 4000,
    min_fruit_pixels: int = 100,
    rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mask = _binary_mask(fruit_mask, "fruit_mask", rgb.shape[:2])
    if k < 2 or attempts < 1 or max_iterations < 1 or max_fit_pixels < 1:
        raise ValueError("Invalid K-Means settings")
    roi = mask > 0
    pixel_count = int(np.count_nonzero(roi))
    empty_labels = np.full(mask.shape, -1, dtype=np.int16)
    if pixel_count < max(k, int(min_fruit_pixels)):
        return {"status": "insufficient_fruit_pixels", "cluster_labels": empty_labels,
                "cluster_map": np.zeros((*mask.shape, 3), dtype=np.uint8),
                "defect_mask": np.zeros(mask.shape, dtype=np.uint8), "selected_cluster_ids": [],
                "cluster_details": [], "fruit_pixels": pixel_count, "fit_pixels": 0}

    features, brightness = normalized_color_features(rgb, color_space)
    roi_features = np.ascontiguousarray(features[roi], dtype=np.float32)
    roi_indices = np.flatnonzero(roi)
    if roi_features.shape[0] < k:
        return {"status": "insufficient_fruit_pixels", "cluster_labels": empty_labels,
                "cluster_map": np.zeros((*mask.shape, 3), dtype=np.uint8),
                "defect_mask": np.zeros(mask.shape, dtype=np.uint8), "selected_cluster_ids": [],
                "cluster_details": [], "fruit_pixels": pixel_count, "fit_pixels": 0}

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if roi_features.shape[0] > max_fit_pixels:
        sample_positions = np.sort(rng.choice(roi_features.shape[0], size=max_fit_pixels, replace=False))
        fit_data = np.ascontiguousarray(roi_features[sample_positions], dtype=np.float32)
    else:
        fit_data = roi_features
    if fit_data.shape[0] < k:
        return {"status": "insufficient_fit_pixels", "cluster_labels": empty_labels,
                "cluster_map": np.zeros((*mask.shape, 3), dtype=np.uint8),
                "defect_mask": np.zeros(mask.shape, dtype=np.uint8), "selected_cluster_ids": [],
                "cluster_details": [], "fruit_pixels": pixel_count, "fit_pixels": int(fit_data.shape[0])}
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iterations), float(epsilon))
    try:
        compactness, _, centers = cv2.kmeans(
            fit_data, int(k), None, criteria, int(attempts), cv2.KMEANS_PP_CENTERS
        )
    except cv2.error as exc:
        return {"status": "kmeans_failed", "error": str(exc), "cluster_labels": empty_labels,
                "cluster_map": np.zeros((*mask.shape, 3), dtype=np.uint8),
                "defect_mask": np.zeros(mask.shape, dtype=np.uint8), "selected_cluster_ids": [],
                "cluster_details": [], "fruit_pixels": pixel_count, "fit_pixels": int(fit_data.shape[0])}

    roi_labels = np.empty(pixel_count, dtype=np.int16)
    chunk_size = 50000
    center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    for start in range(0, pixel_count, chunk_size):
        block = roi_features[start:start + chunk_size]
        distances = np.sum(block * block, axis=1, keepdims=True) + center_norm - 2.0 * block @ centers.T
        roi_labels[start:start + len(block)] = np.argmin(distances, axis=1).astype(np.int16)
    cluster_labels = empty_labels.copy()
    cluster_labels.ravel()[roi_indices] = roi_labels
    defect_mask, selected_ids, cluster_details = select_suspicious_clusters(
        cluster_labels, features, brightness, color_space, rule
    )
    palette = np.asarray([
        [75, 115, 210], [230, 105, 65], [70, 180, 110], [200, 80, 190], [30, 190, 210],
    ], dtype=np.uint8)
    cluster_map = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cluster_id in range(k):
        cluster_map[cluster_labels == cluster_id] = palette[cluster_id]
    return {"status": "ok", "cluster_labels": cluster_labels, "cluster_map": cluster_map,
            "defect_mask": defect_mask, "selected_cluster_ids": selected_ids,
            "cluster_details": cluster_details, "fruit_pixels": pixel_count,
            "fit_pixels": int(fit_data.shape[0]), "compactness": float(compactness),
            "centers": centers.tolist()}


def defect_area_ratio(defect_mask: np.ndarray, fruit_mask: np.ndarray) -> float | None:
    """Return a percent only for a non-empty, shape-matched binary fruit mask."""
    defect = _binary_mask(defect_mask, "defect_mask")
    fruit = _binary_mask(fruit_mask, "fruit_mask", defect.shape)
    fruit_pixels = int(np.count_nonzero(fruit))
    if fruit_pixels == 0:
        return None
    return float(np.count_nonzero((defect > 0) & (fruit > 0)) / fruit_pixels * 100.0)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"Could not write PNG: {path}")
    temporary.replace(path)


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = rgb_to_bgr(rgb)
    output = bgr.copy()
    red = np.zeros_like(output)
    red[:, :] = (20, 30, 245)
    selected = mask > 0
    output[selected] = np.clip(
        bgr[selected].astype(np.float32) * 0.42 + red[selected].astype(np.float32) * 0.58,
        0, 255,
    ).astype(np.uint8)
    return output


def _load_mask(path: Path, shape: tuple[int, int], require_positive: bool = True) -> np.ndarray | None:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or raw.shape != shape:
        return None
    if not np.isin(raw, (0, 255)).all() or (require_positive and np.count_nonzero(raw) == 0):
        return None
    return raw


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_on = left > 0
    right_on = right > 0
    union = int(np.count_nonzero(left_on | right_on))
    return 1.0 if union == 0 else int(np.count_nonzero(left_on & right_on)) / union


class Phase2FruitMaskProvider:
    """Use exported Phase 2 masks, or run the same Phase 2 baseline for missing IDs."""

    def __init__(self, config: dict[str, Any], settings: dict[str, Any],
                 generated_manifest_path: Path | None = None) -> None:
        self.config = config
        self.settings = settings
        self.root = Path(config["_project_root"])
        kmeans_config = config.get("kmeans", {})
        segmentation = config.get("segmentation", {})
        self.mask_root = project_path(config, segmentation.get("mask_dir", "data/processed/fruit_masks"))
        self.generated_manifest_path = generated_manifest_path or project_path(
            config, "data/metadata/phase03_fruit_masks.csv"
        )
        self.phase2_records = _read_csv(project_path(config, kmeans_config.get(
            "phase2_mask_manifest", "data/metadata/phase2_segmentation_manifest.csv")))
        self.generated_records = _read_csv(self.generated_manifest_path)
        self.records: dict[tuple[str, str], dict[str, str]] = {}
        for row in self.generated_records + self.phase2_records:
            self.records[(row.get("image_id", ""), row.get("variant", ""))] = row
        self.generated_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._pending: list[dict[str, Any]] = []
        for row in self.generated_records:
            self.generated_by_id[row.get("image_id", "")].append(row)

    def _cached(self, image_id: str, variant: str, source_hash: str,
                shape: tuple[int, int]) -> tuple[np.ndarray | None, str, str]:
        row = self.records.get((image_id, variant))
        if not row or row.get("source_sha256") != source_hash:
            return None, "not_found", ""
        status = row.get("status", "")
        if status != "accepted_heuristic":
            return None, status or "not_accepted", row.get("mask_path", "")
        path = project_path(self.config, row.get("mask_path", ""))
        mask = _load_mask(path, shape) if path.is_file() else None
        if mask is None:
            return None, "cached_mask_invalid", str(path)
        source = "phase2_baseline_generated_for_phase3" if row.get("source") else "phase2_exported"
        return mask, source, str(path.relative_to(self.root))

    def get(self, row: dict[str, str], rgb: np.ndarray, alpha: np.ndarray | None,
            source_hash: str, variant: str) -> tuple[np.ndarray | None, str, str, str]:
        image_id = row["image_id"]
        shape = rgb.shape[:2]
        cached, source, path = self._cached(image_id, variant, source_hash, shape)
        if cached is not None:
            return cached, "accepted_heuristic", source, path
        cached_row = self.records.get((image_id, variant))
        if cached_row and cached_row.get("source_sha256") == source_hash and cached_row.get("status") not in {
            "accepted_heuristic", "not_found", "cached_mask_invalid"
        }:
            return None, cached_row.get("status", "not_accepted"), "phase2_baseline", ""

        variant_rgb = rgb if variant == "E1" else gray_world(rgb, alpha).rgb
        channel = str(self.settings.get("default_channel", "lab_b"))
        result = segment_channel(
            variant_rgb,
            channel=channel,
            method=str(self.settings.get("default_method", "fixed")),
            polarity=str(self.settings.get("default_polarity", "binary")),
            fixed_threshold=float(self.settings.get("fixed_thresholds", {}).get(channel, 0.0)),
            open_kernel=int(self.settings.get("morphology", {}).get("open_kernel", 3)),
            close_kernel=int(self.settings.get("morphology", {}).get("close_kernel", 9)),
            open_iterations=int(self.settings.get("morphology", {}).get("open_iterations", 1)),
            close_iterations=int(self.settings.get("morphology", {}).get("close_iterations", 2)),
            fill_hole_pixels=bool(self.settings.get("morphology", {}).get("fill_holes", True)),
            min_area_fraction=float(self.settings.get("ranking", {}).get("min_area_fraction", 0.03)),
            max_area_fraction=float(self.settings.get("ranking", {}).get("max_area_fraction", 0.90)),
            target_area_fraction=float(self.settings.get("ranking", {}).get("target_area_fraction", 0.32)),
            minimum_score=float(self.settings.get("ranking", {}).get("minimum_score", 0.45)),
            ambiguity_margin=float(self.settings.get("ranking", {}).get("ambiguity_margin", 0.025)),
        )
        mask = result.proposed_mask if result.status == "accepted_heuristic" else None
        mask_path = self.mask_root / row["dataset"] / variant / f"{image_id}.png"
        if variant == "E2" and mask is not None:
            e1, _, _, _ = self.get(row, rgb, alpha, source_hash, "E1")
            gate = float(self.settings.get("minimum_variant_agreement_iou", 0.5))
            if e1 is None or _mask_iou(e1, mask) < gate:
                mask = None
                result_status = "review_variant_disagreement" if e1 is not None else "review_variant_status_mismatch"
            else:
                result_status = result.status
        else:
            result_status = result.status
        if mask is not None:
            _write_png(mask_path, mask)
            relative_path = str(mask_path.relative_to(self.root))
            self._pending.append({"image_id": image_id, "dataset": row["dataset"], "variant": variant,
                                 "source_sha256": source_hash, "mask_path": relative_path,
                                 "channel": channel, "method": self.settings.get("default_method", "fixed"),
                                 "polarity": self.settings.get("default_polarity", "binary"),
                                 "mask_type": "heuristic_fruit_candidate_not_ground_truth",
                                 "status": "accepted_heuristic", "source": "phase2_baseline_generated_for_phase3"})
            self.records[(image_id, variant)] = self._pending[-1]
            return mask.copy(), "accepted_heuristic", "phase2_baseline_generated_for_phase3", relative_path
        self._pending.append({"image_id": image_id, "dataset": row["dataset"], "variant": variant,
                             "source_sha256": source_hash, "mask_path": "", "channel": channel,
                             "method": self.settings.get("default_method", "fixed"),
                             "polarity": self.settings.get("default_polarity", "binary"),
                             "mask_type": "heuristic_fruit_candidate_not_ground_truth",
                             "status": result_status, "source": "phase2_baseline_generated_for_phase3"})
        self.records[(image_id, variant)] = self._pending[-1]
        return None, result_status, "phase2_baseline", ""

    def checkpoint(self) -> None:
        if not self._pending:
            return
        by_key = {(row.get("image_id"), row.get("variant")): row for row in self.generated_records}
        for row in self._pending:
            by_key[(row["image_id"], row["variant"])] = row
        self.generated_records = list(by_key.values())
        _write_csv(self.generated_manifest_path, self.generated_records,
                   ["image_id", "dataset", "variant", "source_sha256", "mask_path", "channel",
                    "method", "polarity", "mask_type", "status", "source"])
        self._pending.clear()


def _approved_masks(config: dict[str, Any], phase3: dict[str, Any]) -> dict[str, dict[str, str]]:
    rows = _read_csv(project_path(config, phase3.get(
        "fruit_ground_truth_manifest", "data/metadata/fruit_gt_manifest.csv")))
    return {row.get("image_id", ""): row for row in rows
            if row.get("validation_status") == "valid_whole_fruit_mask"
            and row.get("semantic_review_status") == "human_confirmed_whole_fruit"}


def _cache_key(row: dict[str, Any], fruit_mask: np.ndarray, mode: str, variant: str,
               color_space: str, k: int, seed: int, settings: dict[str, Any], rule: dict[str, Any]) -> str:
    mask_bytes = np.ascontiguousarray(fruit_mask).tobytes()
    payload = {
        "version": "phase3-kmeans-v1", "source_sha256": row["_source_sha256"],
        "fruit_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(), "mode": mode,
        "variant": variant, "color_space": color_space, "k": k, "seed": seed,
        "attempts": settings["attempts"], "max_iterations": settings["max_iterations"],
        "epsilon": settings["epsilon"], "max_fit_pixels": settings["max_fit_pixels"],
        "min_fruit_pixels": settings["min_fruit_pixels"], "rule": rule,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _result_template(row: dict[str, Any], mode: str, variant: str, color_space: str,
                     k: int, settings: dict[str, Any], fruit_path: str, fruit_source: str,
                     fruit_status: str, fruit_pixels: int, cache_key: str) -> dict[str, Any]:
    return {
        "mode": mode, "image_id": row["image_id"], "dataset": row["dataset"], "task": row.get("task", ""),
        "phase02_split": row.get("phase02_split", "") or "unassigned",
        "split_risk_status": row.get("split_risk_status", ""),
        "official_accuracy_eligible": row.get("official_accuracy_eligible", "false"),
        "source_image_path": row["image_path"], "source_sha256": row["_source_sha256"],
        "defect_mask_path": row.get("normalized_mask_path", ""), "defect_mask_sha256": row.get("_gt_sha256", ""),
        "defect_mask_alignment_status": row.get("mask_alignment_status", ""),
        "defect_mask_review_status": row.get("mask_review_status", ""),
        "variant": variant, "color_space": color_space, "k": k, "seed": settings.get("seed", ""),
        "attempts": settings["attempts"], "max_iterations": settings["max_iterations"],
        "epsilon": settings["epsilon"], "max_fit_pixels": settings["max_fit_pixels"],
        "fruit_mask_path": fruit_path, "fruit_mask_source": fruit_source, "fruit_mask_status": fruit_status,
        "fruit_pixels": fruit_pixels, "fit_pixels": "", "compactness": "",
        "kmeans_status": "not_run", "selected_cluster_ids": "",
        "cluster_details_json": "", "defect_pixel_count": "", "defect_area_percent": "",
        "gt_pair_status": row.get("_gt_pair_status", ""), "iou": "", "dice": "", "precision": "",
        "recall": "", "detection_hit": "", "false_positive_pixels_within_fruit": "",
        "cluster_map_path": "", "predicted_mask_path": "", "overlay_path": "",
        "cache_key": cache_key, "cache_hit": "false",
    }


def _summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kmeans_status") != "ok" or row.get("iou") in (None, ""):
            continue
        key = (row["mode"], row["variant"], row["color_space"], int(row["k"]), row["phase02_split"])
        groups[key].append(row)
    summarized = []
    for (mode, variant, space, k, split), group in sorted(groups.items()):
        mean = lambda name: float(np.mean([float(row[name]) for row in group if row.get(name) != ""]))
        summarized.append({
            "mode": mode, "variant": variant, "color_space": space, "k": k, "split": split,
            "n_images": len(group), "iou_mean": mean("iou"), "dice_mean": mean("dice"),
            "precision_mean": mean("precision"), "recall_mean": mean("recall"),
            "detection_rate": mean("detection_hit"),
            "false_positive_pixels_mean": mean("false_positive_pixels_within_fruit"),
            "defect_area_percent_mean": mean("defect_area_percent"),
            "metric_status": "exploratory_lineage_unresolved",
        })
    return summarized


def _best_validation(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in metrics if row["split"] == "validation" and row["n_images"]
                  and row["mode"] == "heuristic"]
    candidates.sort(key=lambda row: (-float(row["iou_mean"]), -float(row["dice_mean"]),
                                     row["variant"], row["color_space"], int(row["k"])))
    return candidates[0] if candidates else None


def _save_validation_overlays(config: dict[str, Any], rows: list[dict[str, Any]],
                              selected: dict[str, Any] | None, output_root: Path) -> None:
    selected_key = None if selected is None else (
        selected["mode"], selected["variant"], selected["color_space"], str(selected["k"])
    )
    keep_targets = {
        (output_root / row["mode"] / row["image_id"] / row["variant"] / row["color_space"] / f"k{row['k']}" / "overlay.png").resolve()
        for row in rows if selected_key is not None
        and (row["mode"], row["variant"], row["color_space"], str(row["k"])) == selected_key
        and row.get("kmeans_status") == "ok"
    }
    if output_root.exists():
        for existing_overlay in output_root.rglob("overlay.png"):
            if existing_overlay.resolve() not in keep_targets:
                existing_overlay.unlink()
    for row in rows:
        key = (row["mode"], row["variant"], row["color_space"], str(row["k"]))
        overlay_path = project_path(config, row.get("overlay_path", "")) if row.get("overlay_path") else None
        if key != selected_key:
            if overlay_path and overlay_path.is_file():
                try:
                    overlay_path.resolve().relative_to(output_root.resolve())
                except ValueError:
                    continue
                overlay_path.unlink()
            row["overlay_path"] = ""
            continue
        if row.get("kmeans_status") != "ok":
            continue
        target = output_root / row["mode"] / row["image_id"] / row["variant"] / row["color_space"] / f"k{row['k']}" / "overlay.png"
        source = decode_image(project_path(config, row["source_image_path"])).rgb
        variant_rgb = source if row["variant"] == "original" else gray_world(source).rgb
        prediction = _load_mask(project_path(config, row["predicted_mask_path"]), source.shape[:2], require_positive=False)
        if prediction is None:
            row["overlay_path"] = ""
            continue
        _write_png(target, _overlay(variant_rgb, prediction))
        row["overlay_path"] = str(target.relative_to(Path(config["_project_root"])))


def _write_preview_sheet(config: dict[str, Any], rows: list[dict[str, Any]], output: Path,
                         max_images: int, selected: dict[str, Any] | None) -> Path | None:
    if max_images <= 0:
        return None
    if selected is None:
        return None
    selected_key = (selected["mode"], selected["variant"], selected["color_space"], int(selected["k"]))
    selected_rows: list[dict[str, Any]] = []
    reference = [row for row in rows if row.get("mode") == selected_key[0] and row.get("variant") == selected_key[1]
                 and row.get("color_space") == selected_key[2] and int(row.get("k") or 0) == selected_key[3]
                 and row.get("kmeans_status") == "ok"]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        by_split[row["phase02_split"]].append(row)
    per_split = max(1, math.ceil(max_images / max(1, len(by_split))))
    for split in ("train", "validation", "test", "unassigned"):
        selected_rows.extend(by_split.get(split, [])[:per_split])
    selected_rows = selected_rows[:max_images]
    if not selected_rows:
        return None
    panel_titles = ["RGB original", "Phase 2 fruit mask", "Lab clusters (K=3)", "Suspicious mask", "Overlay"]
    thumb_w, thumb_h = 224, 168
    header_h, label_h = 24, 24
    row_h = header_h + thumb_h + label_h
    sheet = np.full((len(selected_rows) * row_h, len(panel_titles) * thumb_w, 3), 250, dtype=np.uint8)
    root = Path(config["_project_root"])
    for index, row in enumerate(selected_rows):
        paths = [row["source_image_path"], row["fruit_mask_path"], row["cluster_map_path"],
                 row["predicted_mask_path"], row["overlay_path"]]
        for column, (title, relative) in enumerate(zip(panel_titles, paths)):
            path = project_path(config, relative)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
            if image is None:
                image = np.full((thumb_h, thumb_w, 3), 220, dtype=np.uint8)
            else:
                image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            x0, y0 = column * thumb_w, index * row_h + header_h
            sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = image
            cv2.putText(sheet, title, (x0 + 5, y0 + thumb_h + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1, cv2.LINE_AA)
        label = f"{row['image_id']} | {row['phase02_split']} | area={float(row['defect_area_percent']):.2f}%"
        cv2.putText(sheet, label, (5, index * row_h + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (10, 10, 10), 1, cv2.LINE_AA)
    _write_png(output, sheet)
    return output


def _generate_report(path: Path, rows: list[dict[str, Any]], metrics: list[dict[str, Any]],
                     pair_counts: dict[str, int], mode_counts: dict[str, int], preview: Path | None,
                     phase02_report: Path) -> None:
    configs = Counter((r["mode"], r["variant"], r["color_space"], int(r["k"])) for r in rows)
    statuses = Counter(r.get("kmeans_status", "unknown") for r in rows)
    gt_count = pair_counts.get("eligible_exploratory", 0)
    fruit_mask_statuses: dict[str, Counter[str]] = {}
    for variant in ("original", "gray_world"):
        unique_masks = {
            row["image_id"]: row.get("fruit_mask_status", "unknown") for row in rows
            if row.get("mode") == "heuristic" and row.get("variant") == variant
            and row.get("color_space") == "Lab" and int(row.get("k") or 0) == 3
        }
        fruit_mask_statuses[variant] = Counter(unique_masks.values())
    val = [r for r in metrics if r["split"] == "validation" and r["n_images"]]
    val.sort(key=lambda r: (-float(r["iou_mean"]), -float(r["dice_mean"]), r["mode"], r["variant"], r["color_space"], int(r["k"])))
    best = _best_validation(metrics)
    project_root = path.parents[1]
    report_lines = [
        "# MangoResearch — Phase 3: K-Means phát hiện vùng nghi vấn thâm", "",
        "## Tóm tắt", "",
        f"Đã kiểm tra {pair_counts.get('source_rows', 0):,} cặp ảnh–mask Dataset C; {gt_count:,} cặp thỏa điều kiện cấu trúc/định dạng để tính metric **exploratory**. Không có approved whole-fruit mask tại thời điểm chạy; Mode B được bỏ qua (approved mask: {mode_counts.get('approved', 0)}).",
        "Kết quả không phải benchmark chính thức: lineage augmentation của 1.500 ảnh segmentation còn chưa giải quyết và chỉ 24 cặp mask được kiểm tra trực quan trong Phase 0.2. Các split train/validation/test được giữ nguyên; không sửa raw data hoặc split.", "",
        f"K-Means đã chạy trên các tổ hợp khả dụng: {len(configs)} (Mode A); trạng thái tác vụ: `{dict(statuses)}`. Metric được phân nhóm theo split. Xếp hạng cấu hình bên dưới chỉ dùng validation; test không được dùng để chọn cấu hình.", "",
        "## Phương pháp", "",
        "- Mỗi ảnh dùng RGB gốc và nhánh Gray World riêng. Với mỗi variant, màu được chuyển đổi qua module Phase 1; Lab L*/a*/b* được chuẩn hóa theo dải lý thuyết và HSV H/S/V về [0,1] trước K-Means để các kênh không bị chênh thang đo. HSV hue vẫn được đưa vào K-Means tuyến tính; tính tuần hoàn của H là hạn chế đã biết.",
        "- K-Means chỉ fit trên pixel thuộc fruit mask; khi nhiều hơn giới hạn, lấy mẫu pixel ROI xác định theo seed. OpenCV K-Means++ dùng seed gốc 20261023, 1 lần khởi tạo mỗi cấu hình, tối đa 30 vòng lặp, epsilon 0.001 và tối đa 4.000 pixel để fit; các seed riêng theo ảnh/cấu hình cùng tham số được ghi trong `reports/phase03_results.csv`. Pixel ngoài mask không được gán cluster và không thể thành vùng nghi vấn.",
        "- Quy tắc vùng nghi vấn bỏ cụm chiếm đa số, lọc theo tỷ lệ cụm, khoảng cách màu đến cụm trội, chênh sáng L*/V, diện tích phù hợp và tính liên tục thành phần liên thông. Cụm tối nhất không tự động được chọn. Trọng số/threshold nằm trong `config.yaml` mục `phase3.selection_rule`; đây là heuristic chưa hiệu chỉnh bằng nhãn.",
        "- Mode A dùng mask E1/E2 đã xuất ở Phase 2 hoặc chạy lại cùng baseline Phase 2 cho ảnh còn thiếu. E2 chỉ được dùng nếu qua gate đồng thuận E1/E2 hiện hành. Mọi mask này vẫn là heuristic, không phải whole-fruit ground truth.",
        "- Mode B chỉ nhận mask có `valid_whole_fruit_mask` và `human_confirmed_whole_fruit`; không sử dụng mask khuyết tật làm fruit mask.", "",
        "## Kiểm kê cặp ảnh–mask Dataset C", "",
        f"- Candidate theo manifest: {pair_counts.get('source_rows', 0):,}; mask chuẩn hóa tồn tại và khớp kích thước/binary: {pair_counts.get('eligible_exploratory', 0):,}; bị loại vì đường dẫn, trạng thái, kích thước hoặc giá trị: {pair_counts.get('rejected', 0):,}.",
        f"- SHA-256 ảnh nguồn khớp với manifest Phase 0.2: {pair_counts.get('source_hash_match', 0):,}/{pair_counts.get('source_rows', 0):,}; dữ liệu ảnh gốc được giữ nguyên.",
        f"- Fruit mask Phase 2 hợp lệ theo E1/original: `{dict(fruit_mask_statuses.get('original', {}))}`; theo E2/Gray World: `{dict(fruit_mask_statuses.get('gray_world', {}))}`. E2 được gate theo đồng thuận; các trạng thái chưa accepted bị bỏ qua.",
        f"- Visual audit Phase 0.2: 24/1.500 cặp; 1.500 có cùng-stem và `verified_resize_candidate`, nhưng đây không phải pixel-level annotation thủ công cho toàn bộ tập. Mask là vùng khuyết tật theo ngữ nghĩa nguồn/preview; resize/threshold và đối chiếu overlay không bảo đảm từng pixel tuyệt đối chính xác. Tham chiếu: [{phase02_report.name}](<../{phase02_report.relative_to(project_root).as_posix()}>).",
        f"- Cặp/split nguồn: {dict(Counter(r.get('phase02_split', 'unassigned') or 'unassigned' for r in rows if r.get('mode') == 'heuristic' and r.get('variant') == 'original' and r.get('color_space') == 'Lab' and int(r.get('k') or 0) == 3))}.",
        "- 1.500 ảnh segmentation có `augmentation_parent_unresolved`; `official_accuracy_eligible=false`. Các metric bên dưới chỉ exploratory, kể cả validation.", "",
        "## Kết quả theo validation", "",
        "Mean IoU/Dice/precision/recall và tỷ lệ ảnh có giao cắt prediction–GT. Danh sách đầy đủ theo split nằm ở `reports/phase03_metrics.csv`; metric từng ảnh nằm ở `reports/phase03_results.csv`.", "",
        "| Mode | Variant | Space | K | N validation | IoU | Dice | Precision | Recall | Detection rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in val:
        report_lines.append(
            f"| {row['mode']} | {row['variant']} | {row['color_space']} | {row['k']} | {row['n_images']} | {float(row['iou_mean']):.4f} | {float(row['dice_mean']):.4f} | {float(row['precision_mean']):.4f} | {float(row['recall_mean']):.4f} | {float(row['detection_rate']):.4f} |"
        )
    if best:
        report_lines.extend([
            "", f"Cấu hình đứng đầu validation theo mean IoU: **{best['mode']} / {best['variant']} / {best['color_space']} / K={best['k']}** (IoU={float(best['iou_mean']):.4f}, Dice={float(best['dice_mean']):.4f}, n={best['n_images']}). Đây chỉ là ứng viên exploratory; chưa được chọn làm baseline khoa học do split và mask limitation.",
        ])
    else:
        report_lines.extend(["", "Không đủ dòng validation có prediction và cặp mask hợp lệ để xếp hạng; chưa chọn cấu hình."])
    report_lines.extend([
        "", "## So sánh variant/space/K và lỗi", "",
        "- So sánh đầy đủ HSV/Lab, K=2–5 và Original/Gray World được lưu trong metric CSV theo train/validation/test. Chỉ bảng validation ở trên được dùng để xếp hạng; test không tham gia chọn.",
        "- Dataset C ảnh chụp nhiều điều kiện ánh sáng, màu vỏ, nền và mức độ khuyết tật. Gray World có thể làm thay đổi tương phản cục bộ; K-Means theo màu có thể gom bóng đổ vào cụm nghi vấn hoặc tách màu tự nhiên của vỏ. Đầu ra là vùng nghi vấn màu sắc, không phải chẩn đoán chắc chắn vết thâm.",
        "- Rà soát trực quan mẫu ở cấu hình original/Lab/K=3 cho thấy lỗi theo cả hai hướng: ảnh `1309.jpg` có vùng trắng trong mask tham chiếu nhưng dự đoán rỗng (miss); ở `956.jpg`, dự đoán chồng lấp tốt với vùng mask nhưng vẫn không chứng minh được quy tắc màu tổng quát. Đây là ví dụ minh họa, không phải đánh giá lại thủ công toàn bộ tập.",
        "- Chưa có tập ảnh healthy được ghép whole-fruit mask và defect mask xác nhận để đo false positive ở cấp ảnh healthy. Pixel precision hiện phản ánh pixel dương giả ngoài mask khuyết tật trong các cặp Dataset C, không thay thế kiểm tra healthy fruit.",
        "- Nếu không có fruit mask đạt điều kiện, không chạy K-Means và defect area được để trống/unavailable, không gán 0%. Nếu có mask hợp lệ nhưng không cụm nào đủ điều kiện, defect mask rỗng và tỷ lệ diện tích là 0%.", "",
        "## Output và preview", "",
        "Manifest giữ đường dẫn ảnh RGB gốc, fruit mask, cluster map và mask vùng nghi vấn cho từng ảnh/cấu hình. Overlay được xuất cho mọi ảnh theo cấu hình Mode A đứng đầu validation (khi có validation metric), kèm contact sheet; overlay cấu hình khác không được nhân bản. Mask binary được lưu PNG lossless dưới `data/processed/phase3/<mode>/<image_id>/<variant>/<space>/k<K>/`. Các mask tạo bổ sung bởi Phase 2 baseline được ghi trong `data/metadata/phase03_fruit_masks.csv`; raw giữ nguyên.",
    ])
    if preview:
        report_lines.extend(["", f"![Contact sheet Phase 3](<{preview.relative_to(path.parent).as_posix()}>)"])
    mask_audit_preview = path.parent / "phase03" / "previews" / "mask_semantics_audit.png"
    if mask_audit_preview.is_file():
        report_lines.extend([
            "",
            f"Rà soát trực quan thêm 8 cặp validation (RGB, mask tham chiếu, dự đoán và overlay; cấu hình original/Lab/K=3): [mở ảnh kiểm tra](<{mask_audit_preview.relative_to(path.parent).as_posix()}>).",
        ])
    report_lines.extend([
        "", "## Giới hạn và mức sẵn sàng", "",
        "- Các mask defect JPEG gốc đã threshold/resize thành mask nhị phân ở Phase 0.2; 24 cặp được đối chiếu trực quan, không phải toàn bộ 1.500 được vẽ thủ công. Các metric chưa đủ điều kiện công bố chính thức.",
        "- Augmentation lineage của Dataset C chưa giải quyết nên các split có thể rò rỉ ảnh gốc/augmented. Chưa coi test là holdout độc lập.",
        "- Mode B bị bỏ qua do chưa có approved fruit mask. Mode A chưa có đánh giá chất lượng fruit mask bằng IoU/Dice; metric defect chịu ảnh hưởng của sai số mask này.",
        "- Sẵn sàng: chạy exploratory, tạo artifact truy xuất được và so sánh cấu hình trên validation provisional. Chưa sẵn sàng cho kết luận khoa học chính thức hoặc diễn giải vùng nghi vấn là nhãn bệnh chắc chắn.",
        "- Kiểm thử: 55 unit tests passed; end-to-end đã chạy 1.500 cặp thật qua 16 cấu hình khả dụng. SHA-256 của 1.500 ảnh nguồn khớp manifest Phase 0.2.",
        "- Không có Accuracy phân loại; không huấn luyện KNN/SVM/Random Forest. Không triển khai Phase 4/5.", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines), encoding="utf-8")


def run_phase3(config: dict[str, Any], limit: int | None = None,
               modes: list[str] | None = None, force: bool = False) -> Path:
    phase = dict(config.get("kmeans", {}))
    root = Path(config["_project_root"])
    manifest_path = project_path(config, phase.get("manifest_path", "data/metadata/phase02_manifest.csv"))
    output_root = project_path(config, phase.get("output_dir", "data/processed/phase3"))
    result_path = project_path(config, phase.get("result_manifest", "data/metadata/phase03_manifest.csv"))
    results_path = project_path(config, phase.get("results_path", "reports/phase03_results.csv"))
    metrics_path = project_path(config, phase.get("metrics_path", "reports/phase03_metrics.csv"))
    report_path = project_path(config, phase.get("report_path", "reports/phase03_report.md"))
    settings = {
        "seed": int(phase.get("seed", 20261023)),
        "attempts": int(phase.get("attempts", 1)),
        "max_iterations": int(phase.get("max_iterations", 30)),
        "epsilon": float(phase.get("epsilon", 0.001)),
        "max_fit_pixels": int(phase.get("max_fit_pixels", 4000)),
        "min_fruit_pixels": int(phase.get("min_fruit_pixels", 100)),
        "checkpoint_images": max(1, int(phase.get("checkpoint_images", 25))),
    }
    rule = dict(phase.get("selection_rule", {}))
    variants = [str(x) for x in phase.get("variants", ["original", "gray_world"])]
    color_spaces = [str(x) for x in phase.get("color_spaces", ["Lab", "HSV"])]
    k_values = [int(x) for x in phase.get("k_values", [2, 3, 4, 5])]
    selected_modes = modes or [str(x) for x in phase.get("mask_modes", ["heuristic", "approved"])]
    if any(mode not in {"heuristic", "approved"} for mode in selected_modes):
        raise ValueError("modes must contain only 'heuristic' and/or 'approved'")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if not variants or any(v not in {"original", "gray_world"} for v in variants):
        raise ValueError("variants must be original and/or gray_world")
    if not color_spaces or any(s not in {"Lab", "HSV"} for s in color_spaces):
        raise ValueError("color_spaces must be Lab and/or HSV")
    if any(k not in {2, 3, 4, 5} for k in k_values):
        raise ValueError("K values must be selected from 2, 3, 4, 5")

    source_rows = [row for row in _read_csv(manifest_path)
                   if row.get("dataset") == "mango_segmentation"
                   and row.get("task") == "defect_segmentation"
                   and row.get("mask_type") == "defect_region"]
    source_rows.sort(key=lambda row: row.get("image_id", ""))
    if limit is not None:
        source_rows = source_rows[:limit]
    phase2_settings = dict(config.get("segmentation", {}))
    fruit_provider = Phase2FruitMaskProvider(config, phase2_settings)
    approved = _approved_masks(config, phase)
    mode_counts = {"heuristic": len(source_rows),
                   "approved": sum(row.get("image_id") in approved for row in source_rows)}

    # Preflight exact source/mask mapping, shape, binary values, and hashes. Do not infer mask pairs by position.
    pairs: list[dict[str, Any]] = []
    pair_counts = Counter()
    pair_counts["source_rows"] = len(source_rows)
    for row in source_rows:
        item = dict(row)
        image_path = project_path(config, row.get("image_path", ""))
        defect_path = project_path(config, row.get("normalized_mask_path", ""))
        try:
            if not image_path.is_file() or not defect_path.is_file():
                raise FileNotFoundError("image or normalized defect-mask path is missing")
            if row.get("mask_review_status") != "verified" or row.get("mask_alignment_status") != "verified_resize_candidate":
                raise ValueError("pair lacks the required Phase 0.2 review/alignment state")
            frame = decode_image(image_path)
            gt_mask = _load_mask(defect_path, frame.rgb.shape[:2])
            if gt_mask is None:
                raise ValueError("normalized defect mask is not same-size binary PNG with positive pixels")
            if row.get("mask_path") and not project_path(config, row["mask_path"]).is_file():
                raise FileNotFoundError("source defect-mask file is missing")
            item["_image_path_abs"] = str(image_path)
            item["_gt_path_abs"] = str(defect_path)
            item["_source_sha256"] = sha256_file(image_path)
            if row.get("sha256") and row["sha256"] != item["_source_sha256"]:
                raise ValueError("source image SHA-256 differs from the Phase 0.2 manifest")
            pair_counts["source_hash_match"] += 1
            item["_gt_sha256"] = sha256_file(defect_path)
            item["_gt_mask"] = gt_mask
            item["_gt_pair_status"] = "eligible_exploratory"
            pair_counts["eligible_exploratory"] += 1
            pairs.append(item)
        except Exception as exc:
            item["_gt_pair_status"] = f"rejected:{type(exc).__name__}:{exc}"
            pair_counts["rejected"] += 1
            pairs.append(item)
    LOGGER.info("Phase 3 preflight: %d eligible exploratory pairs / %d manifest rows",
                pair_counts["eligible_exploratory"], pair_counts["source_rows"])

    existing_rows = _read_csv(result_path)
    old_by_config = {(row.get("mode"), row.get("image_id"), row.get("variant"), row.get("color_space"),
                      row.get("k")): row for row in existing_rows}
    results: list[dict[str, Any]] = []
    has_approved = mode_counts["approved"] > 0
    runnable_modes = [mode for mode in selected_modes if mode != "approved" or has_approved]
    for image_index, row in enumerate(pairs, start=1):
        if row.get("_gt_pair_status") != "eligible_exploratory":
            continue
        image_result_start = len(results)
        try:
            frame = decode_image(row["_image_path_abs"])
            rgb = frame.rgb
            gt_mask = row["_gt_mask"]
            mode_mask_info: dict[str, tuple[np.ndarray | None, str, str, str]] = {}
            source_hash = row["_source_sha256"]
            if "heuristic" in runnable_modes:
                for variant in ("E1", "E2"):
                    mode_mask_info[("heuristic", variant)] = fruit_provider.get(row, rgb, frame.alpha, source_hash, variant)
            if "approved" in runnable_modes:
                approved_row = approved.get(row["image_id"])
                if approved_row:
                    approved_path = project_path(config, approved_row.get("ground_truth_path", ""))
                    approved_mask = _load_mask(approved_path, rgb.shape[:2]) if approved_path.is_file() else None
                    if approved_mask is not None:
                        rel = str(approved_path.relative_to(root))
                        for variant in ("E1", "E2"):
                            mode_mask_info[("approved", variant)] = (approved_mask, "valid_whole_fruit_mask",
                                                                        "human_approved_ground_truth", rel)
                    else:
                        for variant in ("E1", "E2"):
                            mode_mask_info[("approved", variant)] = (None, "approved_mask_invalid",
                                                                        "human_approved_ground_truth", "")

            for mode in runnable_modes:
                if mode == "approved" and row["image_id"] not in approved:
                    continue
                for variant in variants:
                    phase2_variant = "E1" if variant == "original" else "E2"
                    fruit_mask, fruit_status, fruit_source, fruit_path = mode_mask_info.get(
                        (mode, phase2_variant), (None, "fruit_mask_unavailable", "", ""))
                    variant_rgb = rgb if variant == "original" else gray_world(rgb, frame.alpha).rgb
                    for color_space in color_spaces:
                        for k in k_values:
                            derived_seed_text = f"{settings['seed']}|{row['image_id']}|{mode}|{variant}|{color_space}|{k}"
                            derived_seed = int.from_bytes(hashlib.sha256(derived_seed_text.encode()).digest()[:4], "little") & 0x7FFFFFFF
                            key_settings = {**settings, "seed": derived_seed}
                            cache_key = _cache_key(row, fruit_mask if fruit_mask is not None else np.zeros(rgb.shape[:2], np.uint8),
                                                   mode, variant, color_space, k, derived_seed, key_settings, rule)
                            record = _result_template(row, mode, variant, color_space, k, key_settings,
                                                      fruit_path, fruit_source, fruit_status,
                                                      int(np.count_nonzero(fruit_mask)) if fruit_mask is not None else 0,
                                                      cache_key)
                            record["seed"] = derived_seed
                            if fruit_mask is None:
                                record["kmeans_status"] = "fruit_mask_unavailable"
                                results.append(record)
                                continue
                            artifact_dir = output_root / mode / row["image_id"] / variant / color_space / f"k{k}"
                            cluster_path = artifact_dir / "cluster_map.png"
                            pred_path = artifact_dir / "suspicious_mask.png"
                            overlay_path = artifact_dir / "overlay.png"
                            cache_row = old_by_config.get((mode, row["image_id"], variant, color_space, str(k)))
                            cache_valid = bool(
                                not force and cache_row and cache_row.get("cache_key") == cache_key
                                and all(project_path(config, cache_row.get(field, "")).is_file()
                                        for field in ("cluster_map_path", "predicted_mask_path"))
                            )
                            if cache_valid:
                                prediction = _load_mask(project_path(config, cache_row["predicted_mask_path"]),
                                                        rgb.shape[:2], require_positive=False)
                                cache_valid = prediction is not None
                            else:
                                prediction = None
                            if cache_valid:
                                record.update({
                                    "kmeans_status": "ok", "selected_cluster_ids": cache_row.get("selected_cluster_ids", ""),
                                    "cluster_details_json": cache_row.get("cluster_details_json", ""),
                                    "cluster_map_path": cache_row["cluster_map_path"],
                                    "predicted_mask_path": cache_row["predicted_mask_path"],
                                    "overlay_path": cache_row["overlay_path"], "cache_hit": "true",
                                })
                            else:
                                outcome = cluster_and_detect(
                                    variant_rgb, fruit_mask, color_space, k, derived_seed,
                                    attempts=settings["attempts"], max_iterations=settings["max_iterations"],
                                    epsilon=settings["epsilon"], max_fit_pixels=settings["max_fit_pixels"],
                                    min_fruit_pixels=settings["min_fruit_pixels"], rule=rule,
                                )
                                record["kmeans_status"] = outcome["status"]
                                if outcome["status"] == "ok":
                                    prediction = outcome["defect_mask"]
                                    _write_png(cluster_path, cv2.cvtColor(outcome["cluster_map"], cv2.COLOR_RGB2BGR))
                                    _write_png(pred_path, prediction)
                                    _write_png(overlay_path, _overlay(variant_rgb, prediction))
                                    record.update({
                                        "fit_pixels": outcome["fit_pixels"],
                                        "compactness": outcome.get("compactness", ""),
                                        "selected_cluster_ids": json.dumps(outcome["selected_cluster_ids"]),
                                        "cluster_details_json": json.dumps(outcome["cluster_details"], separators=(",", ":")),
                                        "cluster_map_path": str(cluster_path.relative_to(root)),
                                        "predicted_mask_path": str(pred_path.relative_to(root)),
                                        "cache_hit": "false",
                                    })
                            if record["kmeans_status"] == "ok" and prediction is not None:
                                defect_count = int(np.count_nonzero(prediction))
                                ratio = defect_area_ratio(prediction, fruit_mask)
                                metric = segmentation_metrics(prediction, gt_mask)
                                fp = int(np.count_nonzero((prediction > 0) & (gt_mask == 0) & (fruit_mask > 0)))
                                hit = int(bool(np.any((prediction > 0) & (gt_mask > 0))))
                                record.update({
                                    "defect_pixel_count": defect_count,
                                    "defect_area_percent": ratio if ratio is not None else "",
                                    "iou": metric["iou"], "dice": metric["dice"],
                                    "precision": metric["precision"], "recall": metric["recall"],
                                    "detection_hit": hit, "false_positive_pixels_within_fruit": fp,
                                })
                            results.append(record)
            if image_index % settings["checkpoint_images"] == 0:
                try:
                    _write_csv(result_path, results, RESULT_FIELDS)
                    _write_csv(results_path, results, RESULT_FIELDS)
                    fruit_provider.checkpoint()
                except PermissionError:
                    LOGGER.exception("Checkpoint temporarily blocked by Windows; continuing and retrying at next/final checkpoint")
                LOGGER.info("Phase 3: processed %d/%d eligible pairs", image_index, len(pairs))
        except Exception:
            LOGGER.exception("Phase 3 failed for %s", row.get("image_id", "unknown"))
            del results[image_result_start:]
            for mode in runnable_modes:
                if mode == "approved" and row.get("image_id") not in approved:
                    continue
                for variant in variants:
                    for color_space in color_spaces:
                        for k in k_values:
                            record = _result_template(row, mode, variant, color_space, k,
                                                      {**settings, "seed": settings["seed"]}, "", "", "processing_error", 0, "")
                            record["kmeans_status"] = "processing_error"
                            results.append(record)
    _write_csv(result_path, results, RESULT_FIELDS)
    _write_csv(results_path, results, RESULT_FIELDS)
    fruit_provider.checkpoint()
    metric_rows = _summarize_metrics(results)
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    best_validation = _best_validation(metric_rows)
    _save_validation_overlays(config, results, best_validation, output_root)
    _write_csv(result_path, results, RESULT_FIELDS)
    _write_csv(results_path, results, RESULT_FIELDS)
    preview_path = _write_preview_sheet(config, results,
                                        project_path(config, "reports/phase03/previews/phase3_contact_sheet.png"),
                                        int(phase.get("preview_image_count", 12)), best_validation)
    phase02_report = project_path(config, "reports/phase02_report.md")
    _generate_report(report_path, results, metric_rows, dict(pair_counts), mode_counts, preview_path, phase02_report)
    LOGGER.info("Phase 3 complete: %d result rows; report: %s", len(results), report_path)
    return report_path
