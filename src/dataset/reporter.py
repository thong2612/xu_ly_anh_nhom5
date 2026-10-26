from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import project_path


def _download_inventory(config: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    download_dir = project_path(
        config,
        Path(config.get("data_root", "data")) / "raw" / dataset_key / "downloads",
    )
    archives = []
    partial_count = 0
    if download_dir.exists():
        partial_count = sum(1 for path in download_dir.glob("*.part") if path.is_file())
        for path in sorted(download_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".zip", ".rar"}:
                continue
            valid_zip = zipfile.is_zipfile(path) if path.suffix.lower() == ".zip" else None
            archives.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "valid_zip": valid_zip,
            })
    return {
        "status": "downloaded" if archives else "not_downloaded",
        "archives": archives,
        "partial_file_count": partial_count,
    }


def generate_report(config: dict[str, Any]) -> dict[str, Any]:
    manifest = project_path(config, "data/metadata/manifest.csv")
    rows: list[dict[str, str]] = []
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    mask_manifest = project_path(config, "reports/mask_inspection.csv")
    mask_rows: list[dict[str, str]] = []
    if mask_manifest.exists():
        with mask_manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            mask_rows = list(csv.DictReader(stream))
    split_report_path = project_path(config, "reports/split_report.json")
    split_plan: dict[str, Any] = {}
    if split_report_path.exists():
        try:
            split_plan = json.loads(split_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            split_plan = {"error": "split_report.json is invalid"}
    dataset_reports: list[dict[str, Any]] = []
    for key, definition in config["datasets"].items():
        dataset_rows = [row for row in rows if row.get("dataset") == key]
        valid_rows = [row for row in dataset_rows if row.get("validation_status") != "corrupt"]
        ripeness = Counter(row.get("ripeness_label", "").strip() for row in valid_rows if row.get("ripeness_label", "").strip())
        defects = Counter(row.get("defect_label", "").strip() for row in valid_rows if row.get("defect_label", "").strip())
        statuses = Counter(row.get("validation_status", "unknown") for row in dataset_rows)
        task_counts = Counter(row.get("task", "unknown") for row in dataset_rows)
        paired = [row for row in dataset_rows if row.get("mask_path", "").strip()]
        typed_pairs = [row for row in paired if row.get("mask_type", "").strip()]
        verified_pairs = [row for row in typed_pairs if row.get("mask_review_status", "").strip().casefold() == "verified"]
        resolutions = Counter(f"{row.get('width')}x{row.get('height')}" for row in valid_rows if row.get("width") and row.get("height"))
        formats = Counter(row.get("file_format", "unknown") for row in valid_rows)
        dataset_masks = [row for row in mask_rows if row.get("dataset") == key]
        mask_modes = Counter(row.get("mode", "unknown") for row in dataset_masks)
        mask_values = Counter(row.get("unique_values_or_colors", "unknown") for row in dataset_masks)
        mask_statuses = Counter(row.get("validation", "unknown") for row in dataset_masks)
        mask_sizes = Counter(row.get("size", "unknown") for row in dataset_masks)
        duplicate_counts = Counter(row.get("pixel_sha256") or row.get("sha256") for row in valid_rows if row.get("pixel_sha256") or row.get("sha256"))
        split_counts = Counter(row.get("split", "") for row in dataset_rows if row.get("split", "").strip())
        lineage_risk = any(row.get("split", "").strip() and row.get("original_or_augmented") != "original" for row in dataset_rows)
        is_ripeness_suitable = key == "mango_ripeness" and all(ripeness.get(label, 0) > 0 for label in ("unripe", "partially_ripe", "ripe"))
        is_defect_suitable = bool(defects)
        segmentation_suitable = (
            bool(verified_pairs)
            and len(verified_pairs) == len(paired) == len(dataset_masks)
            and all(row.get("validation_status") not in {"mask_size_mismatch", "mask_pair_ambiguous", "mask_rejected"} for row in verified_pairs)
        )
        segmentation_review_complete = bool(verified_pairs) and len(verified_pairs) == len(paired) == len(dataset_masks)
        mask_size_mismatch_count = sum(row.get("validation_status") == "mask_size_mismatch" for row in paired)
        if segmentation_suitable:
            segmentation_status = "candidate pairs visually reviewed and mask_type recorded"
        elif segmentation_review_complete and mask_size_mismatch_count:
            segmentation_status = (
                f"mask semantics reviewed, but {mask_size_mismatch_count} image/mask pairs have mismatched dimensions; "
                "ground truth is not ready for segmentation"
            )
        else:
            segmentation_status = "not established; pair/type visual review incomplete"
        dataset_reports.append({
            "key": key,
            "title": definition["title"],
            "source_url": definition["url"],
            "download": _download_inventory(config, key),
            "task": definition["task"],
            "task_counts": dict(task_counts),
            "expected_images_from_source_description": definition.get("expected_images"),
            "expected_images_description": definition.get("expected_images_description", definition.get("expected_images")),
            "image_count": len(dataset_rows),
            "valid_image_count": len(valid_rows),
            "ripeness_label_counts": dict(ripeness),
            "defect_label_counts": dict(defects),
            "mask_count_paired": len(paired),
            "mask_file_count": len(dataset_masks),
            "mask_type_recorded_count": len(typed_pairs),
            "mask_semantics_verified_count": len(verified_pairs),
            "mask_modes": dict(mask_modes),
            "mask_unique_values_or_colors": dict(mask_values),
            "mask_size_counts": dict(mask_sizes),
            "mask_validation_counts": dict(mask_statuses),
            "duplicate_extra_count": sum(max(0, count - 1) for count in duplicate_counts.values()),
            "split_counts": dict(split_counts),
            "split_lineage_risk": lineage_risk,
            "validation_status_counts": dict(statuses),
            "formats": dict(formats),
            "resolutions": dict(resolutions),
            "suitability": {
                "ripeness_classification": "supported by imported labels" if is_ripeness_suitable else "not established from current imported labels",
                "defect_analysis": "supported by explicit imported labels" if is_defect_suitable else "not established; no explicit defect labels in manifest",
                "segmentation_ground_truth": segmentation_status,
            },
        })

    data_quality_notes = [
        "Không có ảnh bị loại tự động; lỗi và trạng thái nghi vấn được giữ trong manifest/report.",
        "Mask ghép theo basename trong cùng component; preview dùng để rà soát trực quan.",
        "Mask type chỉ được ghi sau khi đối chiếu mô tả nguồn và preview; không suy ra từ pixel count.",
        "Mask semantics được đối chiếu ở cấp component với mô tả nguồn và preview mẫu; không phải toàn bộ 1.500 cặp đã được xem thủ công từng cặp.",
        "Exact duplicate được phát hiện theo SHA256 và pixel_sha256; augmentation lineage chưa được suy đoán.",
    ]
    for dataset in dataset_reports:
        mismatch_count = int(dataset["validation_status_counts"].get("mask_size_mismatch", 0))
        if mismatch_count:
            paired_resolutions = Counter(
                f"{row.get('width')}x{row.get('height')}"
                for row in rows
                if row.get("dataset") == dataset["key"] and row.get("mask_path", "").strip()
            )
            data_quality_notes.append(
                f"{dataset['key']}: {mismatch_count} image/mask pairs have different dimensions; "
                f"image sizes={dict(paired_resolutions)}, mask sizes={dataset['mask_size_counts']}."
            )
    if not rows:
        data_quality_notes.insert(
            0,
            "Chưa có ảnh dataset thật được import; Phase 0.1 chưa hoàn tất và chưa thể xác minh nhãn, mask hoặc split trên dữ liệu thực.",
        )
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": config.get("project_name", "MangoResearch"),
        "manifest_path": "data/metadata/manifest.csv",
        "dataset_count": len(config["datasets"]),
        "imported_image_count": len(rows),
        "datasets": dataset_reports,
        "split_plan": split_plan,
        "data_quality_notes": data_quality_notes,
    }
    reports_root = project_path(config, config.get("reports_root", "reports"))
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "dataset_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Báo cáo khảo sát dataset — {result['project']}",
        "",
        f"Tạo lúc: {result['generated_at']}",
        f"Ảnh trong manifest: **{len(rows)}**",
        "",
        "## Tải dữ liệu",
        "",
    ]
    for dataset in dataset_reports:
        archives = dataset["download"]["archives"]
        archive_text = ", ".join(
            f"`{item['name']}` ({item['size_bytes']:,} bytes; ZIP hợp lệ: {item['valid_zip']})"
            for item in archives
        ) or "chưa có archive trong thư mục downloads"
        lines.append(
            f"- {dataset['key']}: **{dataset['download']['status']}**; {archive_text}; "
            f"file tải dở: {dataset['download']['partial_file_count']}"
        )
    lines.extend([
        "",
        "## Kết quả theo nguồn",
        "",
    ])
    for dataset in dataset_reports:
        lines.extend([
            f"### {dataset['key']} — [{dataset['title']}]({dataset['source_url']})",
            "",
            f"- Nhiệm vụ cấu hình: `{dataset['task']}`",
            f"- Số ảnh theo task thực tế: `{dataset['task_counts']}`",
            f"- Ảnh kiểm kê: {dataset['image_count']}; hợp lệ để đọc: {dataset['valid_image_count']}; mô tả số lượng tại nguồn: {dataset['expected_images_description']}",
            f"- Nhãn độ chín thực tế: `{dataset['ripeness_label_counts']}`",
            f"- Nhãn khuyết tật thực tế: `{dataset['defect_label_counts']}`",
            f"- Cặp mask ghép: {dataset['mask_count_paired']}; có mask_type đã nhập: {dataset['mask_type_recorded_count']}",
            f"- Cặp có semantic review marker: {dataset['mask_semantics_verified_count']} (đối chiếu cấp component và preview mẫu; không xem thủ công từng cặp)",
            f"- File mask kiểm tra: {dataset['mask_file_count']}; mode: `{dataset['mask_modes']}`; số giá trị/màu phân biệt: `{dataset['mask_unique_values_or_colors']}`; trạng thái: `{dataset['mask_validation_counts']}`",
            f"- Ảnh trùng nội dung dư: {dataset['duplicate_extra_count']}; split: `{dataset['split_counts']}`; rủi ro lineage: {dataset['split_lineage_risk']}",
            f"- Tình trạng kiểm tra: `{dataset['validation_status_counts']}`",
            f"- Phù hợp phân loại độ chín: {dataset['suitability']['ripeness_classification']}",
            f"- Phù hợp phân tích khuyết tật: {dataset['suitability']['defect_analysis']}",
            f"- Ground truth segmentation: {dataset['suitability']['segmentation_ground_truth']}",
            "",
        ])
    lines.extend([
        "## Ghi chú chất lượng",
        "",
        *[f"- {note}" for note in result["data_quality_notes"]],
        "",
        "Preview và báo cáo mask nằm trong `reports/previews/` và `reports/mask_inspection.csv`.",
        "",
    ])
    if split_plan:
        lines.extend(["## Chia dữ liệu", ""])
        for item in split_plan.get("prepared", []):
            lines.append(f"- {item['dataset']} / {item['task']}: {item['counts']}; lineage_risk={item['lineage_risk']}")
        for item in split_plan.get("skipped", []):
            lines.append(f"- Chưa chia {item['dataset']} / {item['task']}: {item['reason']}")
        lines.append("")
    (reports_root / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
