from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color.preprocess import decode_image
from src.validation.evaluation import KMEANS_REVIEW_PATH, _read_csv, _write_csv


PREREVIEW_FIELDS = [
    "image_id", "source_image_path", "ground_truth_mask_path", "fruit_mask_path",
    "source_image_status", "source_mask_status", "fruit_mask_status",
    "mask_area_fraction", "mask_bbox_xyxy", "touches_image_border",
    "fruit_mask_overlap_fraction", "mask_outside_fruit_fraction",
    "pre_review_warnings", "pre_review_summary", "suggested_status",
    "suggested_status_reason", "vision_status", "vision_suggestion", "vision_reason",
]

# These deliberately broad thresholds create review prompts, never ground truth.
MIN_AREA_FRACTION = 0.001
MAX_AREA_FRACTION = 0.65
PARTIAL_OUTSIDE_FRUIT_WARNING = 0.10
MIN_FRUIT_OVERLAP_FOR_REJECT_SUGGESTION = 0.20
MIN_FRUIT_OVERLAP_FOR_CLEAN_SUGGESTION = 0.90


def _phase2_fruit_mask_paths(root: Path) -> dict[str, dict[str, str]]:
    """Read only Phase 2 fruit-mask references; do not inspect K-Means predictions or metrics."""
    rows = _read_csv(root / "reports/phase03_results.csv")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("mode") != "heuristic" or row.get("variant") != "original":
            continue
        image_id = row.get("image_id", "")
        path = row.get("fruit_mask_path", "")
        if not image_id or not path:
            continue
        current = output.get(image_id)
        if current is None:
            output[image_id] = {"path": path, "status": row.get("fruit_mask_status", "")}
        elif current["path"] != path:
            current["path"] = ""  # Conflicting references must not be guessed between.
            current["status"] = "conflicting_phase2_fruit_mask_paths"
    return output


def _read_binary_mask(path: Path, expected_shape: tuple[int, int] | None = None) -> tuple[np.ndarray | None, str]:
    if path.suffix.casefold() != ".png":
        return None, "mask_not_png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if image is None:
        return None, "mask_unreadable_or_missing"
    if image.ndim != 2:
        return None, "mask_not_single_channel"
    if image.dtype != np.uint8:
        return None, "mask_not_uint8"
    if expected_shape is not None and image.shape != expected_shape:
        return None, "mask_size_mismatch"
    values = np.unique(image)
    if not np.isin(values, (0, 255)).all():
        return None, "mask_not_binary_0_255"
    if not np.any(image):
        return None, "mask_empty"
    return image, "valid_binary_png"


def _review_one(root: Path, row: dict[str, str], fruit_ref: dict[str, str] | None) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        rgb = decode_image(root / Path(row.get("source_image_path", ""))).rgb
        image_status = "readable"
        shape = rgb.shape[:2]
    except Exception:
        rgb = None
        shape = None
        image_status = "unreadable_or_missing"
        warnings.append("source_image_unreadable")

    mask_path = root / Path(row.get("ground_truth_mask_path", ""))
    target, mask_status = _read_binary_mask(mask_path, shape)
    if mask_status != "valid_binary_png":
        warnings.append(mask_status)

    area_fraction: float | str = ""
    bbox: str = ""
    touches_border = ""
    overlap_fraction: float | str = ""
    outside_fraction: float | str = ""
    fruit_status = "unavailable"
    fruit_path = (fruit_ref or {}).get("path", "")
    if fruit_ref and fruit_ref.get("status") == "conflicting_phase2_fruit_mask_paths":
        fruit_status = "conflicting_phase2_fruit_mask_paths"
        warnings.append("phase2_fruit_mask_path_conflict")
    elif fruit_path and shape is not None:
        fruit, fruit_status = _read_binary_mask(root / Path(fruit_path), shape)
        if fruit is None:
            warnings.append("phase2_fruit_mask_invalid_or_unavailable")
        elif target is not None:
            defect = target > 0
            fruit_area = fruit > 0
            overlap_fraction = float(np.count_nonzero(defect & fruit_area) / np.count_nonzero(defect))
            outside_fraction = 1.0 - overlap_fraction
            if outside_fraction > PARTIAL_OUTSIDE_FRUIT_WARNING:
                warnings.append("mask_partially_outside_heuristic_fruit_mask")
            if outside_fraction > 0.20:
                warnings.append("mask_extends_outside_heuristic_fruit_mask")
    else:
        warnings.append("phase2_fruit_mask_unavailable")

    if target is not None and shape is not None:
        defect = target > 0
        mask_pixels = int(np.count_nonzero(defect))
        area_fraction = mask_pixels / int(shape[0] * shape[1])
        ys, xs = np.where(defect)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = f"{x0},{y0},{x1},{y1}"
        touches = x0 == 0 or y0 == 0 or x1 == shape[1] - 1 or y1 == shape[0] - 1
        touches_border = str(bool(touches)).lower()
        if area_fraction < MIN_AREA_FRACTION:
            warnings.append("mask_area_extremely_small")
        if area_fraction > MAX_AREA_FRACTION:
            warnings.append("mask_area_extremely_large")
        if touches:
            warnings.append("mask_touches_image_border")

    technical_valid = image_status == "readable" and mask_status == "valid_binary_png"
    if not technical_valid:
        suggested = "likely_rejected"
        reason = "Mask không qua kiểm tra kỹ thuật; cần xem lại trước khi có thể dùng làm ground truth."
    elif isinstance(overlap_fraction, float) and overlap_fraction < MIN_FRUIT_OVERLAP_FOR_REJECT_SUGGESTION:
        suggested = "likely_rejected"
        reason = "Mask phần lớn nằm ngoài fruit mask heuristic; đây chỉ là cảnh báo vì fruit mask cũng có thể sai."
    elif (isinstance(area_fraction, float) and MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION
          and isinstance(overlap_fraction, float) and overlap_fraction >= MIN_FRUIT_OVERLAP_FOR_CLEAN_SUGGESTION
          and touches_border == "false" and not warnings):
        suggested = "likely_approved"
        reason = "Kiểm tra kỹ thuật và hình học không phát hiện cảnh báo; semantics vùng khuyết tật vẫn cần người xác nhận."
    else:
        suggested = "uncertain"
        reason = "Có cảnh báo hoặc chưa đủ phép kiểm tra hình học; cần người review trực tiếp."

    return {
        "image_id": row.get("image_id", ""), "source_image_path": row.get("source_image_path", ""),
        "ground_truth_mask_path": row.get("ground_truth_mask_path", ""), "fruit_mask_path": fruit_path,
        "source_image_status": image_status, "source_mask_status": mask_status,
        "fruit_mask_status": fruit_status, "mask_area_fraction": area_fraction,
        "mask_bbox_xyxy": bbox, "touches_image_border": touches_border,
        "fruit_mask_overlap_fraction": overlap_fraction, "mask_outside_fruit_fraction": outside_fraction,
        "pre_review_warnings": ";".join(warnings),
        "pre_review_summary": "Không phát hiện cảnh báo kỹ thuật/hình học." if not warnings else "Cảnh báo: " + "; ".join(warnings),
        "suggested_status": suggested, "suggested_status_reason": reason,
        "vision_status": "not_configured", "vision_suggestion": "",
        "vision_reason": "Không tìm thấy cấu hình vision API trong dự án; ảnh không được gửi ra dịch vụ bên ngoài.",
    }


def run_kmeans_prereview(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / KMEANS_REVIEW_PATH
    rows = _read_csv(manifest_path)
    if not rows:
        raise FileNotFoundError(f"Review manifest missing or empty: {manifest_path}")
    old_statuses = [row.get("review_status", "pending") for row in rows]
    fruit_paths = _phase2_fruit_mask_paths(root)
    details = [_review_one(root, row, fruit_paths.get(row.get("image_id", ""))) for row in rows]
    details_by_id = {row["image_id"]: row for row in details}
    for row in rows:
        auto = details_by_id[row.get("image_id", "")]
        for key in PREREVIEW_FIELDS:
            if key not in {"image_id", "source_image_path", "ground_truth_mask_path"}:
                row[key] = auto[key]
    if [row.get("review_status", "pending") for row in rows] != old_statuses:
        raise AssertionError("Pre-review must never modify review_status")
    manifest_fields = list(rows[0])
    _write_csv(manifest_path, rows, manifest_fields)
    report_csv = root / "reports/phase07_kmeans_prereview.csv"
    _write_csv(report_csv, details, PREREVIEW_FIELDS)

    suggested_counts = Counter(row["suggested_status"] for row in details)
    warning_counts = Counter(code for row in details for code in row["pre_review_warnings"].split(";") if code)
    readable_images = sum(row["source_image_status"] == "readable" for row in details)
    valid_source_masks = sum(row["source_mask_status"] == "valid_binary_png" for row in details)
    valid_fruit_masks = sum(row["fruit_mask_status"] == "valid_binary_png" for row in details)
    area_values = [float(row["mask_area_fraction"]) for row in details if row["mask_area_fraction"] != ""]
    overlap_values = [float(row["fruit_mask_overlap_fraction"]) for row in details if row["fruit_mask_overlap_fraction"] != ""]
    warning_pair_count = sum(bool(row["pre_review_warnings"]) for row in details)
    lines = [
        "# Pre-review cục bộ — Dataset C K-Means Ground Truth",
        "",
        f"- Số cặp được kiểm tra: {len(details)}.",
        f"- Ảnh nguồn đọc được: {readable_images}/{len(details)}; mask nguồn đạt PNG đơn kênh uint8, binary 0/255, khớp kích thước và không rỗng: {valid_source_masks}/{len(details)}.",
        f"- Fruit mask heuristic hợp lệ và cùng kích thước: {valid_fruit_masks}/{len(details)}; cặp có ít nhất một cảnh báo: {warning_pair_count}/{len(details)} (các loại cảnh báo có thể trùng nhau).",
        f"- Gợi ý (không phải quyết định): likely_approved {suggested_counts['likely_approved']}, likely_rejected {suggested_counts['likely_rejected']}, uncertain {suggested_counts['uncertain']}.",
        (f"- Mask area fraction min/median/max: {min(area_values):.4f}/{float(np.median(area_values)):.4f}/{max(area_values):.4f}; "
         f"mask nằm trong fruit mask fraction min/median/max: {min(overlap_values):.4f}/{float(np.median(overlap_values)):.4f}/{max(overlap_values):.4f}."
         if area_values and overlap_values else "- Tỷ lệ diện tích/overlap chưa đủ dữ liệu để tổng hợp."),
        f"- Trạng thái người review giữ nguyên: {Counter(old_statuses)}.",
        "- Các gợi ý chỉ nằm trong `suggested_status`; `review_status` chỉ thay đổi khi người review bấm nút trong GUI.",
        "- Kiểm tra gồm đọc ảnh, PNG mask đơn kênh uint8, giá trị 0/255, kích thước, bbox/touch border, tỷ lệ diện tích mask, và phần mask nằm trong fruit mask heuristic Phase 2.",
        f"- Cờ cảnh báo diện tích: dưới {MIN_AREA_FRACTION:.3f} hoặc trên {MAX_AREA_FRACTION:.2f} diện tích ảnh; partial outside-fruit trên {PARTIAL_OUTSIDE_FRUIT_WARNING:.0%}, cảnh báo nặng trên 20%. Các ngưỡng chỉ để sắp xếp review, không phải tiêu chuẩn ground truth.",
        "- So sánh fruit-mask là với heuristic nên có thể sai. Không dùng output/mask K-Means hoặc score K-Means để gợi ý trạng thái.",
        "- Vision API: không được cấu hình trong project; không gọi API và không gửi ảnh ra ngoài.",
        "- Kể cả `likely_approved`, mask vẫn cần được người review xác nhận về căn chỉnh và ý nghĩa vùng khuyết tật.",
        "",
        "## Các cảnh báo thường gặp",
        "",
    ]
    if warning_counts:
        lines.extend(f"- `{code}`: {count}" for code, count in sorted(warning_counts.items()))
    else:
        lines.append("Không có cảnh báo kỹ thuật/hình học trong tập này.")
    lines.extend(["", f"Chi tiết từng cặp: `{report_csv.relative_to(root).as_posix()}`.",
                  "", "Không sửa mask nguồn, raw dataset, split, model hoặc quyết định người review."])
    report_path = root / "reports/phase07_kmeans_prereview.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "pairs": len(details), "suggested_status_counts": dict(suggested_counts),
        "warning_counts": dict(warning_counts), "review_status_unchanged": True,
        "vision_status": "not_configured_no_external_calls",
        "manifest": str(manifest_path.relative_to(root)),
        "details_csv": str(report_csv.relative_to(root)),
        "report": str(report_path.relative_to(root)),
    }
