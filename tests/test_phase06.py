from __future__ import annotations

import csv
import io
import json
import joblib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.demo.phase6 import (
    InputImageError, _predict, build_ordered_model_vector, decode_upload,
    infer_image, load_demo_assets, prediction_summary_for_cli,
)
from src.segmentation.baseline import SegmentationResult

ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(rgb: tuple[int, int, int] = (245, 245, 245), size: tuple[int, int] = (80, 64)) -> bytes:
    image = Image.new("RGB", size, rgb)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_decode_upload_accepts_png_and_jpg_and_rejects_invalid() -> None:
    png = _png_bytes()
    decoded = decode_upload(png, "sample.png")
    assert decoded.rgb.shape == (64, 80, 3)
    assert decoded.rgb.dtype == np.uint8

    jpg = io.BytesIO()
    Image.new("RGB", (72, 50), (120, 30, 20)).save(jpg, format="JPEG")
    assert decode_upload(jpg.getvalue(), "sample.JPG").rgb.shape == (50, 72, 3)
    with pytest.raises(InputImageError):
        decode_upload(b"not an image", "bad.png")
    with pytest.raises(InputImageError):
        decode_upload(png, "sample.webp")


def test_upload_resource_limits_are_enforced() -> None:
    payload = _png_bytes()
    with pytest.raises(InputImageError, match="vượt giới hạn"):
        decode_upload(payload, "sample.png", max_bytes=10)
    with pytest.raises(InputImageError, match="pixel"):
        decode_upload(payload, "sample.png", max_pixels=100)


def test_model_vector_uses_artifact_order_and_rejects_metadata() -> None:
    artifact = {"feature_columns": ["f_second", "f_first"]}
    vector = build_ordered_model_vector(artifact, {"f_first": 11, "f_second": 7})
    assert vector.tolist() == [[7.0, 11.0]]
    with pytest.raises(ValueError, match="unsafe"):
        build_ordered_model_vector({"feature_columns": ["label"]}, {"label": "ripe"})


def test_missing_feature_policy_and_failed_mask_return_unavailable() -> None:
    class SpyPipeline:
        def __init__(self):
            self.seen = None

        def predict(self, values):
            self.seen = values.copy()
            return np.asarray(["ok"])

    pipeline = SpyPipeline()
    artifact = {"feature_columns": ["a", "b"], "pipeline": pipeline}
    partial = _predict(artifact, {"a": 2.0}, "accepted_heuristic")
    assert partial["available"] and partial["missing_features"] == 1
    assert pipeline.seen[0, 0] == 2 and np.isnan(pipeline.seen[0, 1])
    assert not _predict(artifact, {}, "fruit_mask_empty")["available"]
    assert not _predict(artifact, {}, "accepted_heuristic")["available"]


def test_selected_models_load_and_match_phase4_order() -> None:
    assets = load_demo_assets(ROOT)
    assert assets.ripeness_model["experiment"] == "R2_gray_world_color"
    assert assets.defect_model["experiment"] == "D1_original_color"
    assert assets.ripeness_model["feature_columns"] == assets.schema["feature_sets"]["ripeness"]["gray_world_color"]
    assert assets.defect_model["feature_columns"] == assets.schema["feature_sets"]["defect"]["original_color"]
    assert len(assets.ripeness_model["feature_columns"]) == 66
    assert len(assets.defect_model["feature_columns"]) == 66


def test_no_mango_like_blank_image_warns_and_failed_segmentation_is_unavailable() -> None:
    assets = load_demo_assets(ROOT)
    blank = _png_bytes()
    actual_result = infer_image(blank, "blank.png", assets)
    assert actual_result["mask_provider_status"]["original"] == "empty_threshold"
    assert not actual_result["ripeness"]["available"]
    assert not actual_result["defect"]["available"]
    assert actual_result["suspicious_area_percent"] is None

    dummy_result = SegmentationResult(
        channel="lab_b", method="fixed", polarity="binary", threshold_native=8.0,
        threshold_u8=10.0, raw_mask=np.zeros((64, 80), np.uint8),
        proposed_mask=np.zeros((64, 80), np.uint8), status="no_plausible_component",
        component_count=0, selected_area_fraction=0.0, score=0.0, details="test fail",
    )
    with patch("src.segmentation.phase3.segment_channel", return_value=dummy_result):
        result = infer_image(_png_bytes(), "blank.png", assets)
    assert "chứa xoài" in result["warning"]
    assert result["fruit_mask"] is None
    assert not result["ripeness"]["available"]
    assert not result["defect"]["available"]
    assert result["suspicious_area_percent"] is None


def _real_ripeness_image() -> Path:
    with (ROOT / "data/features/ripeness_features.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            candidate = ROOT / row["image_path"]
            if candidate.is_file():
                return candidate
    pytest.skip("Phase 4 ripeness images are not available in this checkout")


def test_real_image_and_cli_streamlit_inference_are_identical() -> None:
    from streamlit.testing.v1 import AppTest

    image_path = _real_ripeness_image()
    assets = load_demo_assets(ROOT)
    result = infer_image(image_path.read_bytes(), image_path.name, assets)
    expected = prediction_summary_for_cli(result)
    temporary_model = ROOT / "models/phase051/.phase06_roundtrip_test.joblib"
    try:
        joblib.dump(assets.ripeness_model, temporary_model)
        reloaded_model = joblib.load(temporary_model)
        reloaded_prediction = _predict(reloaded_model, result["feature_values"], result["gray_world_mask_status"])
        assert reloaded_prediction == result["ripeness"]
    finally:
        temporary_model.unlink(missing_ok=True)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase06_predict.py"), str(image_path)],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
    )
    cli_result = json.loads(completed.stdout)
    assert cli_result == expected
    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app_test.file_uploader[0].set_value((image_path.name, image_path.read_bytes(), "image/jpeg"))
    app_test.run()
    app_test.button[0].click().run()
    assert not app_test.exception
    ui_result = prediction_summary_for_cli(app_test.session_state["phase06_result"])
    assert ui_result == expected
    assert result["fruit_mask_status"] == "accepted_heuristic"
    assert result["ripeness"]["available"]
    assert result["defect"]["available"]
    assert result["kmeans"]["status"] == "ok"
    assert result["suspicious_area_percent"] is not None
    assert result["original_rgb"].shape[1] <= 1024

