from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.color.preprocess import ImageFrame, decode_image, gray_world
from src.features.extraction import (
    _analysis_raster, _color_features, _derived_seed,
    _kmeans_features_from_outcome, _valid_working_mask,
)
from src.segmentation.kmeans import Phase2FruitMaskProvider, cluster_and_detect, defect_area_ratio
from src.utils.config import load_config, project_path

LOGGER = logging.getLogger("mangoresearch.inference")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}
FORBIDDEN_FEATURE_NAMES = {
    "label", "ripeness_label", "defect_label", "phase02_split", "phase051_split",
    "image_id", "dataset", "task", "image_path", "source_sha256", "split_risk_status",
    "official_accuracy_eligible", "mask_path", "ground_truth_mask_path",
}


class InputImageError(ValueError):
    """An uploaded file is not a safe, supported 8-bit JPG/PNG image."""


@dataclass
class DemoAssets:
    root: Path
    config: dict[str, Any]
    schema: dict[str, Any]
    ripeness_model: dict[str, Any]
    defect_model: dict[str, Any]


def load_demo_assets(root: str | Path | None = None) -> DemoAssets:
    project_root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    config = load_config(project_root / "config.yaml")
    schema_path = project_path(config, config.get("features", {}).get("schema_path", "data/features/feature_schema.json"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, Any]] = {}
    for task in ("ripeness", "defect"):
        path = project_root / "models" / "classifiers" / f"{task}_selected.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Phase 5.1 model is missing: {path}")
        artifact = joblib.load(path)
        if artifact.get("task") != task:
            raise ValueError(f"Unexpected task in {path.name}: {artifact.get('task')}")
        columns = artifact.get("feature_columns")
        if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Invalid ordered feature_columns in {path.name}")
        known = set(schema.get("feature_columns", {}).get(task, []))
        metadata = set(schema.get("metadata_columns", [])) | FORBIDDEN_FEATURE_NAMES
        if not set(columns).issubset(known) or set(columns) & metadata:
            raise ValueError(f"Feature schema mismatch or metadata leakage in {path.name}")
        chosen_group = "gray_world_color" if task == "ripeness" else "original_color"
        expected_columns = schema.get("feature_sets", {}).get(task, {}).get(chosen_group, [])
        if columns != expected_columns:
            raise ValueError(f"Ordered features in {path.name} differ from the Phase 4 {chosen_group} schema")
        artifacts[task] = artifact
    if artifacts["ripeness"].get("experiment") != "R2_gray_world_color":
        raise ValueError("Ripeness artifact is not the selected Gray World model")
    if artifacts["defect"].get("experiment") != "D1_original_color":
        raise ValueError("Defect artifact is not the selected Original Color model")
    return DemoAssets(project_root, config, schema, artifacts["ripeness"], artifacts["defect"])


def decode_upload(payload: bytes, filename: str, max_bytes: int = MAX_UPLOAD_BYTES,
                  max_pixels: int = MAX_IMAGE_PIXELS) -> ImageFrame:
    if not payload:
        raise InputImageError("Tệp không có dữ liệu.")
    if len(payload) > max_bytes:
        raise InputImageError(f"Ảnh vượt giới hạn {max_bytes // (1024 * 1024)} MB.")
    suffix = Path(filename or "").suffix.casefold()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise InputImageError("Chỉ chấp nhận tệp JPG hoặc PNG.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in ALLOWED_FORMATS or (image.format == "MPO" and suffix not in {".jpg", ".jpeg"}):
                raise InputImageError("Nội dung tệp không phải JPG hoặc PNG hợp lệ.")
            width, height = image.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise InputImageError(f"Kích thước ảnh vượt giới hạn {max_pixels:,} pixel.")
            image.verify()
    except InputImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputImageError("Không thể đọc ảnh; tệp có thể bị hỏng hoặc không phải ảnh.") from exc

    # Decode through the same OpenCV RGB/BGRA path used by Phases 1 and 4.
    suffix = ".jpg" if filename.casefold().endswith((".jpg", ".jpeg")) else ".png"
    with tempfile.TemporaryDirectory(prefix="mango-upload-") as temp_name:
        temporary = Path(temp_name) / f"upload{suffix}"
        temporary.write_bytes(payload)
        try:
            frame = decode_image(temporary)
        except (OSError, ValueError) as exc:
            raise InputImageError(str(exc)) from exc
    if frame.rgb.shape[0] * frame.rgb.shape[1] > max_pixels:
        raise InputImageError(f"Kích thước ảnh vượt giới hạn {max_pixels:,} pixel.")
    return frame


def _run_config_for_demo(config: dict[str, Any], scratch: Path) -> dict[str, Any]:
    runtime = copy.deepcopy(config)
    runtime["segmentation"]["mask_dir"] = str(scratch / "fruit_masks")
    runtime["kmeans"]["phase2_mask_manifest"] = str(scratch / "no_cached_masks.csv")
    return runtime


def _display_raster(rgb: np.ndarray, max_side: int = 1024) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _predict(artifact: dict[str, Any], features: dict[str, Any], mask_status: str) -> dict[str, Any]:
    columns: list[str] = artifact["feature_columns"]
    vector = build_ordered_model_vector(artifact, features)
    missing = int(np.isnan(vector).sum())
    if mask_status != "accepted_heuristic":
        return {"available": False, "reason": f"fruit mask: {mask_status}", "missing_features": missing,
                "feature_count": len(columns), "prediction": None, "probabilities": None}
    if missing == len(columns):
        return {"available": False, "reason": "Không có đặc trưng hữu hạn để dự đoán.",
                "missing_features": missing, "feature_count": len(columns),
                "prediction": None, "probabilities": None}
    pipeline = artifact["pipeline"]
    try:
        predicted = str(pipeline.predict(vector)[0])
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            raw = np.asarray(pipeline.predict_proba(vector))[0]
            labels = list(getattr(pipeline, "classes_", artifact.get("labels", [])))
            probabilities = {str(label): float(score) for label, score in zip(labels, raw)}
        return {"available": True, "reason": "ok", "missing_features": missing,
                "feature_count": len(columns), "prediction": predicted, "probabilities": probabilities}
    except Exception as exc:
        LOGGER.exception("Model prediction failed")
        return {"available": False, "reason": f"prediction failed: {type(exc).__name__}",
                "missing_features": missing, "feature_count": len(columns),
                "prediction": None, "probabilities": None}


def build_ordered_model_vector(artifact: dict[str, Any], features: dict[str, Any]) -> np.ndarray:
    """Build a one-row array in the immutable feature order stored with the fitted artifact."""
    columns = artifact.get("feature_columns", [])
    if not columns or len(columns) != len(set(columns)) or set(columns) & FORBIDDEN_FEATURE_NAMES:
        raise ValueError("Model artifact has an invalid or unsafe feature order")
    vector = np.full((1, len(columns)), np.nan, dtype=np.float64)
    for index, column in enumerate(columns):
        try:
            numeric = float(features.get(column))
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            vector[0, index] = numeric
    return vector


def infer_image(payload: bytes, filename: str, assets: DemoAssets | None = None) -> dict[str, Any]:
    assets = assets or load_demo_assets()
    frame = decode_upload(payload, filename)
    raw_digest = hashlib.sha256(payload).hexdigest()
    image_id = f"demo_{raw_digest[:20]}"
    phase4 = assets.config.get("features", {})
    phase3 = assets.config.get("kmeans", {})
    analysis_rgb, analysis_alpha = _analysis_raster(
        frame.rgb, frame.alpha, int(phase4.get("max_analysis_side", 512)))
    gray_result = gray_world(analysis_rgb, analysis_alpha)
    gray_rgb = gray_result.rgb
    row = {"image_id": image_id, "dataset": "demo_upload", "task": "demo",
           "image_path": filename, "sha256": raw_digest}

    # Run the Phase 2 baseline in an isolated scratch directory. It uses the
    # same configured thresholding code; no Phase 0.2 mask or GT is used.
    scratch_parent = assets.root / "data" / "features"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inference_", dir=scratch_parent) as temp_name:
        scratch = Path(temp_name)
        runtime_config = _run_config_for_demo(assets.config, scratch)
        provider = Phase2FruitMaskProvider(
            runtime_config, runtime_config.get("segmentation", {}),
            generated_manifest_path=scratch / "generated_masks.csv")
        original_raw, original_status, original_source, _ = provider.get(
            row, analysis_rgb, analysis_alpha, raw_digest, "E1")
        gray_raw, gray_status, gray_source, _ = provider.get(
            row, analysis_rgb, analysis_alpha, raw_digest, "E2")
        min_pixels = int(phase4.get("min_fruit_pixels", 100))
        original_mask, original_validity = _valid_working_mask(original_raw, analysis_alpha, min_pixels)
        gray_mask, gray_validity = _valid_working_mask(gray_raw, analysis_alpha, min_pixels)

        original_features, original_feature_status = _color_features(
            analysis_rgb, original_mask, "original_color", phase4.get("color_proportions", {}))
        gray_features, gray_feature_status = _color_features(
            gray_rgb, gray_mask, "gray_world_color", phase4.get("color_proportions", {}))
        feature_values = {**original_features, **gray_features}

        kmeans = None
        kmeans_feature_status = "fruit_mask_unavailable"
        if original_mask is not None:
            seed = _derived_seed(int(phase3.get("seed", 20261023)), image_id,
                                 "original", "Lab", 3)
            kmeans = cluster_and_detect(
                analysis_rgb, original_mask, "Lab", 3, seed,
                attempts=int(phase3.get("attempts", 1)),
                max_iterations=int(phase3.get("max_iterations", 30)),
                epsilon=float(phase3.get("epsilon", 0.001)),
                max_fit_pixels=int(phase3.get("max_fit_pixels", 4000)),
                min_fruit_pixels=int(phase3.get("min_fruit_pixels", 100)),
                rule=dict(phase3.get("selection_rule", {})),
            )
            kmeans_feature_status = str(kmeans.get("status", "unknown"))
            if kmeans.get("status") == "ok":
                feature_values.update(_kmeans_features_from_outcome(
                    analysis_rgb, original_mask, kmeans, "original_lab_k3"))

        ripeness_prediction = _predict(assets.ripeness_model, feature_values, gray_validity)
        defect_prediction = _predict(assets.defect_model, feature_values, original_validity)

    defect_mask = (kmeans or {}).get("defect_mask")
    cluster_map = (kmeans or {}).get("cluster_map")
    ratio = None
    if kmeans and kmeans.get("status") == "ok" and original_mask is not None and defect_mask is not None:
        ratio = defect_area_ratio(defect_mask, original_mask)
    fruit_pixels = int(np.count_nonzero(original_mask)) if original_mask is not None else 0
    background_removed = None
    if original_mask is not None:
        background_removed = np.zeros_like(analysis_rgb)
        background_removed[original_mask > 0] = analysis_rgb[original_mask > 0]
    overlay = None
    if defect_mask is not None and kmeans and kmeans.get("status") == "ok":
        overlay = analysis_rgb.copy()
        suspicious = defect_mask > 0
        overlay[suspicious] = np.clip(
            analysis_rgb[suspicious].astype(np.float32) * 0.42
            + np.asarray([245, 30, 20], dtype=np.float32) * 0.58,
            0, 255,
        ).astype(np.uint8)
    return {
        "image_id": image_id,
        "original_rgb": _display_raster(frame.rgb),
        "analysis_rgb": analysis_rgb,
        "gray_world_rgb": gray_rgb,
        "fruit_mask": original_mask,
        "gray_world_fruit_mask": gray_mask,
        "background_removed_rgb": background_removed,
        "cluster_map_rgb": cluster_map if kmeans and kmeans.get("status") == "ok" else None,
        "suspicious_mask": defect_mask if kmeans and kmeans.get("status") == "ok" else None,
        "overlay_rgb": overlay,
        "fruit_mask_status": original_validity,
        "gray_world_mask_status": gray_validity,
        "mask_provider_status": {"original": original_status, "gray_world": gray_status},
        "mask_source": {"original": original_source, "gray_world": gray_source},
        "fruit_area_fraction": float(fruit_pixels / (analysis_rgb.shape[0] * analysis_rgb.shape[1])) if fruit_pixels else None,
        "gray_world": {"applied": gray_result.applied, "reason": gray_result.reason,
                       "gains": gray_result.gains, "clipping_fraction": gray_result.clipping_fraction},
        "ripeness": ripeness_prediction,
        "defect": defect_prediction,
        "feature_status": {"original_color": original_feature_status, "gray_world_color": gray_feature_status,
                           "original_lab_k3": kmeans_feature_status},
        "kmeans": {"status": kmeans_feature_status, "color_space": "CIELAB", "k": 3,
                   "variant": "Original", "seed": _derived_seed(int(phase3.get("seed", 20261023)),
                                                                    image_id, "original", "Lab", 3),
                   "selected_clusters": (kmeans or {}).get("selected_cluster_ids", [])},
        "suspicious_area_percent": ratio,
        "fruit_pixels": fruit_pixels,
        "feature_values": feature_values,
        "warning": "Ứng dụng chưa có mô hình xác nhận ảnh có chứa xoài. Với ảnh không có xoài hoặc mask nền không đáng tin cậy, dự đoán có thể không hợp lệ. Kết quả phân loại và vùng màu nghi vấn từ K-Means là hai đầu ra độc lập.",
        "scientific_caveat": "Artifact Phase 5.1 được đánh giá trên split còn provisional; kết quả demo chỉ exploratory, không đại diện hiệu năng thương mại. Xác suất nếu có là giá trị thô, chưa hiệu chuẩn.",
    }


def prediction_summary_for_cli(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe CLI view of the same inference result used by Streamlit."""
    return {
        "image_id": result["image_id"],
        "fruit_mask_status": result["fruit_mask_status"],
        "gray_world_mask_status": result["gray_world_mask_status"],
        "ripeness": {key: result["ripeness"].get(key) for key in
                     ("available", "reason", "prediction", "probabilities", "missing_features", "feature_count")},
        "defect": {key: result["defect"].get(key) for key in
                   ("available", "reason", "prediction", "probabilities", "missing_features", "feature_count")},
        "kmeans_status": result["kmeans"]["status"],
        "suspicious_area_percent": result["suspicious_area_percent"],
    }

