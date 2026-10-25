from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from typing import Any

from src.dataset.inspector import METADATA_FIELDS
from src.utils.config import project_path


LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


def _targets(n: int, ratios: dict[str, float]) -> dict[str, int]:
    names = list(SPLITS)
    raw = {name: n * float(ratios[name]) for name in names}
    result = {name: int(raw[name]) for name in names}
    remaining = n - sum(result.values())
    for name in sorted(names, key=lambda item: (raw[item] - result[item], -names.index(item)), reverse=True)[:remaining]:
        result[name] += 1
    return result


def _assign_groups(groups: list[list[dict[str, str]]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(len(group) for group in groups)
    target = {name: total * float(ratios[name]) for name in SPLITS}
    current = {name: 0 for name in SPLITS}
    assignment: dict[str, str] = {}
    for group in groups:
        size = len(group)
        # Fill the split furthest below its target. This preserves all three
        # subsets even for small classes while keeping duplicate groups intact.
        chosen = max(SPLITS, key=lambda name: ((target[name] - current[name]) / max(target[name], 1), -SPLITS.index(name)))
        for row in group:
            assignment[row["image_id"]] = chosen
        current[chosen] += size
    return assignment


def prepare_splits(config: dict[str, Any], dataset_keys: list[str] | None = None) -> dict[str, Any]:
    manifest = project_path(config, "data/metadata/manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError("Chưa có manifest.csv; chạy inspect trước.")
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        existing_fields = [field for field in (reader.fieldnames or []) if field]
    requested = set(dataset_keys or config["datasets"].keys())
    seed = int(config.get("split", {}).get("seed", 20261023))
    ratios = config.get("split", {}).get("ratios", {"train": .7, "validation": .15, "test": .15})
    assigned: dict[str, str] = {}
    summary: dict[str, Any] = {"prepared": [], "skipped": []}
    by_task: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    ambiguous_cohorts: set[tuple[str, str]] = set()

    for row in rows:
        key = row.get("dataset", "")
        if key not in requested:
            continue
        row.setdefault("split_origin", "")
        if row.get("split_origin") != "published":
            row["split"] = ""
        if row.get("split_origin") == "ambiguous":
            ambiguous_cohorts.add((key, row.get("task", "")))
            continue
        if row.get("validation_status") == "corrupt":
            continue
        task = row.get("task", "")
        if task == "ripeness_classification":
            label = row.get("ripeness_label", "").strip()
        elif task == "defect_analysis":
            label = row.get("defect_label", "").strip()
        elif task == "defect_segmentation":
            # A mask's class semantics must be inspected and explicitly verified first.
            label = "segmentation" if (
                row.get("validation_status") in {"valid", "duplicate_exact"}
                and row.get("mask_path", "").strip()
                and row.get("mask_type", "").strip()
                and row.get("mask_review_status", "").strip().casefold() == "verified"
            ) else ""
        else:
            label = ""
        if label:
            row["_split_label"] = label
            by_task[(key, task)].append(row)

    for dataset in sorted(requested):
        task = config["datasets"][dataset]["task"]
        if (dataset, task) not in by_task:
            reasons = {
                "ripeness_classification": "không có ảnh với ripeness_label đã xác minh",
                "defect_analysis": "không có ảnh với defect_label tường minh",
                "defect_segmentation": "chưa có cặp ảnh/mask với mask_type đã xác minh",
            }
            reason = "đường dẫn có dấu hiệu split chính thức xung đột" if (dataset, task) in ambiguous_cohorts else reasons.get(task, "không có ảnh đủ metadata để chia")
            if task == "defect_segmentation":
                size_mismatches = sum(
                    row.get("dataset") == dataset
                    and row.get("task") == task
                    and row.get("validation_status") == "mask_size_mismatch"
                    for row in rows
                )
                if size_mismatches:
                    reason = f"{size_mismatches} cặp ảnh/mask sai kích thước; cần căn chỉnh trước khi chia segmentation"
            summary["skipped"].append({
                "dataset": dataset,
                "task": task,
                "reason": reason,
            })

    digest_cohorts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if row.get("validation_status") == "corrupt":
            continue
        digest = row.get("pixel_sha256") or row.get("sha256") or row["image_id"]
        digest_cohorts[digest].add((row.get("dataset", ""), row.get("task", "")))
    cross_cohort_conflict_counts: dict[tuple[str, str], int] = defaultdict(int)
    for cohorts in digest_cohorts.values():
        if len(cohorts) > 1:
            for cohort in cohorts:
                cross_cohort_conflict_counts[cohort] += 1

    for (dataset, task), task_rows in sorted(by_task.items()):
        if (dataset, task) in ambiguous_cohorts:
            summary["skipped"].append({"dataset": dataset, "task": task, "reason": "đường dẫn có dấu hiệu split chính thức xung đột"})
            continue
        published = [row for row in task_rows if row.get("split_origin") == "published"]
        if published:
            valid_split_values = set(SPLITS)
            if all(row.get("split_origin") == "published" and row.get("split") in valid_split_values for row in task_rows):
                source_lineage_risk = any(row.get("original_or_augmented") != "original" for row in task_rows)
                summary["prepared"].append({
                    "dataset": dataset,
                    "task": task,
                    "image_count": len(task_rows),
                    "counts": {name: sum(row.get("split") == name for row in task_rows) for name in SPLITS},
                    "lineage_risk": source_lineage_risk,
                    "source_split_preserved": True,
                    "note": "Split nhận diện từ cấu trúc nguồn được giữ nguyên; lineage augmentation chưa được xác minh." if source_lineage_risk else "Split nhận diện từ cấu trúc nguồn được giữ nguyên.",
                })
            else:
                summary["skipped"].append({"dataset": dataset, "task": task, "reason": "cấu trúc có split chính thức nhưng metadata split không đầy đủ/nhất quán"})
            continue
        digest_labels: dict[str, set[str]] = defaultdict(set)
        for row in task_rows:
            digest = row.get("pixel_sha256") or row.get("sha256") or row["image_id"]
            label_value = row.get("ripeness_label", "").strip() or row.get("defect_label", "").strip() or "segmentation"
            digest_labels[digest].add(label_value)
        conflicting = [digest for digest, labels in digest_labels.items() if len(labels) > 1]
        conflict_reasons: list[str] = []
        cross_count = cross_cohort_conflict_counts.get((dataset, task), 0)
        if cross_count:
            conflict_reasons.append(
                f"{cross_count} ảnh trùng nội dung xuất hiện ở nhiều dataset/task; cần thống nhất split trước"
            )
        if conflicting:
            conflict_reasons.append(f"{len(conflicting)} ảnh trùng nội dung có nhãn xung đột; cần rà soát metadata")
        if conflict_reasons:
            summary["skipped"].append({
                "dataset": dataset,
                "task": task,
                "reason": "; ".join(conflict_reasons),
            })
            continue
        by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            by_label[row.pop("_split_label")].append(row)
        task_assignment: dict[str, str] = {}
        reasons: list[str] = []
        if task == "ripeness_classification":
            expected_labels = set(config["datasets"].get(dataset, {}).get("ripeness_labels", {}))
            missing_labels = expected_labels - set(by_label)
            if missing_labels:
                reasons.append(f"thiếu nhãn độ chín: {', '.join(sorted(missing_labels))}")
        if task == "defect_analysis" and len(by_label) < 2:
            reasons.append("cần ít nhất 2 nhãn khuyết tật tường minh để tạo split phân loại")
        risk = any(row.get("original_or_augmented") != "original" for row in task_rows)
        for label, label_rows in sorted(by_label.items()):
            groups_by_digest: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in label_rows:
                digest = row.get("pixel_sha256") or row.get("sha256") or row["image_id"]
                groups_by_digest[digest].append(row)
            if len(groups_by_digest) < 3:
                reasons.append(f"{label}: chỉ có {len(groups_by_digest)} nhóm ảnh độc lập, cần ít nhất 3")
                continue
            label_seed = seed + int(hashlib.sha256(f"{dataset}/{task}/{label}".encode()).hexdigest()[:8], 16)
            task_assignment.update(_assign_groups(list(groups_by_digest.values()), ratios, label_seed))
        if reasons:
            summary["skipped"].append({"dataset": dataset, "task": task, "reason": "; ".join(reasons)})
            continue
        assigned.update(task_assignment)
        summary["prepared"].append({
            "dataset": dataset,
            "task": task,
            "image_count": len(task_assignment),
            "counts": {name: sum(value == name for value in task_assignment.values()) for name in SPLITS},
            "lineage_risk": risk,
            "note": "Exact pixel duplicates grouped; augmented lineage is not established." if risk else "Only source-confirmed originals included.",
        })

    for row in rows:
        if row["image_id"] in assigned:
            row["split"] = assigned[row["image_id"]]
            row["split_origin"] = "generated"

    metadata_path = manifest
    fields = list(rows[0].keys()) if rows else (existing_fields or METADATA_FIELDS)
    fields = [field for field in fields if not field.startswith("_")]
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    split_root = project_path(config, "data/splits")
    split_root.mkdir(parents=True, exist_ok=True)
    for dataset, task in sorted({(row["dataset"], row["task"]) for row in rows}):
        output = split_root / f"{dataset}_{task}.csv"
        selected = [row for row in rows if row["dataset"] == dataset and row["task"] == task and row.get("split")]
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            fieldnames = ["image_id", "dataset", "image_path", "task", "ripeness_label", "defect_label", "mask_path", "mask_type", "mask_review_status", "split", "split_origin", "original_or_augmented", "sha256", "pixel_sha256"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    reports_root = project_path(config, config.get("reports_root", "reports"))
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "split_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Đã chia %s ảnh; bỏ qua %s task chưa đủ điều kiện", len(assigned), len(summary["skipped"]))
    return summary
