from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.phase6 import decode_upload, infer_image, load_demo_assets
from src.models.phase051 import _ssim_gray
from src.validation.dataset_v2_external import (
    crop_exif_corrected_rgb, evaluate_predictions, make_dhash, pixel_sha256,
    primary_exclusion_reason, summarize_groups,
)
from src.color.preprocess import decode_image

LOGGER = logging.getLogger("mangoresearch.dataset_v2_external")
MANIFEST = ROOT / "data/metadata/dataset_v2_curated_candidate_manifest.csv"
SPLITS = {
    "Harumanis_v4": ROOT / "data/splits/phase051/ripeness_phase051.csv",
    "MangoFruitBD_v2": ROOT / "data/splits/phase051/defect_phase051.csv",
}
ARCHIVES = {
    "harumanis_v4_all.zip": ROOT / "reports/dataset_v2_visual_audit/downloads/harumanis_v4_all.zip",
    "MangoFruitBD.zip": ROOT / "reports/dataset_v2_visual_audit/downloads/MangoFruitBD.zip",
}
OUT = ROOT / "reports/dataset_v2_external"
REPORT = ROOT / "reports/dataset_v2_external_evaluation.md"
PREDICTION_FIELDS = [
    "image_id", "source", "source_path", "source_sha256", "input_sha256", "input_pixel_sha256",
    "input_dhash", "source_group", "duplicate_group", "original_label", "source_disease_label",
    "true_label", "label_conflict_status", "duplicate_status", "nearest_train_image_id",
    "nearest_train_dhash_distance", "nearest_train_ssim_gray_256", "near_duplicate_screen",
    "primary_exclusion_reason", "evaluation_eligible", "prediction_available", "prediction",
    "inference_error", "fruit_mask_status", "ripeness_prediction", "defect_prediction",
    "ripeness_available", "defect_available", "kmeans_status", "model_fingerprint",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_member(row: dict[str, str], zips: dict[str, zipfile.ZipFile]) -> tuple[bytes, str]:
    archive_name, member = row["original_path"].split("::", 1)
    if archive_name not in zips:
        archive_path = ARCHIVES.get(archive_name)
        if archive_path is None or not archive_path.is_file():
            raise FileNotFoundError(f"Archive unavailable: {archive_name}")
        zips[archive_name] = zipfile.ZipFile(archive_path)
    return zips[archive_name].read(member), member


def prepare_input(row: dict[str, str], zips: dict[str, zipfile.ZipFile]) -> tuple[bytes, str, np.ndarray, np.ndarray, str]:
    """Return Phase 6 payload, filename, full source RGB, model-input RGB, source-file SHA."""
    raw, member = archive_member(row, zips)
    with Image.open(io.BytesIO(raw)) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
        source_rgb = np.asarray(source, dtype=np.uint8)
    if row["source"] == "MangoFruitBD_v2":
        model_rgb = crop_exif_corrected_rgb(source_rgb, row["crop_bbox"])
        output = io.BytesIO()
        Image.fromarray(model_rgb, mode="RGB").save(output, format="PNG", optimize=False)
        payload = output.getvalue()
        filename = f"{row['image_id']}.png"
    else:
        model_rgb = source_rgb
        payload = raw
        filename = Path(member).name
    return payload, filename, source_rgb, model_rgb, sha256_bytes(raw)


def letterbox_gray256(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = min(256 / max(width, 1), 256 / max(height, 1))
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((256, 256), 127, dtype=np.uint8)
    top, left = (256 - new_h) // 2, (256 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def load_training_references() -> dict[str, list[dict[str, Any]]]:
    # Reuse hashes already computed and persisted for Phase 0.2/5.1. Decode only
    # the few nearest perceptual-hash candidates for SSIM instead of rereading
    # all high-resolution training images into memory.
    hash_rows = {r["image_id"]: r for r in read_csv(ROOT / "data/metadata/phase02_manifest.csv")}
    refs: dict[str, list[dict[str, Any]]] = {"Harumanis_v4": [], "MangoFruitBD_v2": []}
    for source, split_path in SPLITS.items():
        rows = read_csv(split_path)
        for row in rows:
            if row.get("phase051_split") != "train":
                continue
            image_path = ROOT / row["image_path"]
            cached_hash = hash_rows.get(row["image_id"], {})
            refs[source].append({
                "image_id": row["image_id"], "sha256": row.get("sha256", ""),
                "pixel_sha256": row.get("pixel_sha256", "") or cached_hash.get("pixel_sha256", ""),
                "dhash": int(cached_hash.get("dhash64") or "0", 16),
                "path": image_path, "letterbox": None,
            })
        LOGGER.info("Loaded %d actual Phase 5.1 training images for %s", len(refs[source]), source)
    return refs


def duplicate_audit(source_sha: str, source_rgb: np.ndarray, input_rgb: np.ndarray,
                    train_refs: list[dict[str, Any]]) -> dict[str, Any]:
    source_pixel = pixel_sha256(source_rgb)
    input_pixel = pixel_sha256(input_rgb)
    source_hash = sha256_bytes(source_rgb.tobytes())
    input_hash = sha256_bytes(input_rgb.tobytes())
    file_matches = [r for r in train_refs if r["sha256"] and source_sha == r["sha256"]]
    source_pixel_matches = [r for r in train_refs if source_pixel == r["pixel_sha256"]]
    input_pixel_matches = [r for r in train_refs if input_pixel == r["pixel_sha256"]]
    dhash = make_dhash(input_rgb)
    distances = [((dhash ^ int(r["dhash"])).bit_count(), r) for r in train_refs]
    distances.sort(key=lambda pair: pair[0])
    nearest_distance, nearest = distances[0] if distances else (None, {})
    near_ssim = ""
    if distances and int(nearest_distance) <= 12:
        # Compare the closest perceptual-hash hits after the same aspect-preserving 256px alignment.
        candidates = distances[:3] if int(nearest_distance) <= 12 else distances[:1]
        external_gray = letterbox_gray256(input_rgb)
        scored = []
        for distance, ref in candidates:
            if ref["letterbox"] is None:
                train_frame = decode_image(ref["path"])
                ref["letterbox"] = letterbox_gray256(train_frame.rgb)
            score = _ssim_gray(external_gray, ref["letterbox"])
            scored.append((score, distance, ref))
        best_score, best_distance, best_ref = max(scored, key=lambda x: x[0])
        nearest_ssim = round(float(best_score), 6)
        nearest_distance, nearest = best_distance, best_ref
        near_ssim = nearest_ssim
    exact = bool(file_matches or source_pixel_matches or input_pixel_matches)
    near_screen = bool(distances and (nearest_distance <= 4 or (near_ssim != "" and float(near_ssim) >= 0.90)))
    elevated = bool(not exact and near_ssim != "" and int(nearest_distance) <= 4 and float(near_ssim) >= 0.90)
    if exact:
        duplicate_status = "exact_match_to_training"
    elif elevated:
        duplicate_status = "near_similarity_candidate_excluded"
    elif near_screen:
        duplicate_status = "near_similarity_screen_unconfirmed"
    else:
        duplicate_status = "no_exact_or_high_similarity_match"
    return {
        "source_file_sha256_match": "|".join(r["image_id"] for r in file_matches),
        "source_pixel_sha256_match": "|".join(r["image_id"] for r in source_pixel_matches),
        "input_pixel_sha256_match": "|".join(r["image_id"] for r in input_pixel_matches),
        "source_raw_pixel_sha256": source_hash,
        "input_pixel_sha256": input_hash,
        "input_dhash": f"{dhash:016x}",
        "nearest_train_image_id": nearest.get("image_id", ""),
        "nearest_train_dhash_distance": nearest_distance if nearest_distance is not None else "",
        "nearest_train_ssim_gray_256": near_ssim,
        "near_duplicate_screen": "candidate_not_identity_proof" if near_screen else "no_candidate_found",
        "duplicate_status": duplicate_status,
    }


def model_fingerprint(assets: Any) -> str:
    digests = []
    for task in ("ripeness", "defect"):
        path = assets.root / "models" / "phase051" / f"{task}_selected.joblib"
        digests.append(sha256_bytes(path.read_bytes()))
    schema_path = assets.root / assets.config["phase4"]["schema_path"]
    digests.append(sha256_bytes(schema_path.read_bytes()))
    return hashlib.sha256("|".join(digests).encode()).hexdigest()


def confusion_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], directory: Path) -> None:
    labels = ["healthy", "defective"]
    matrix = metrics["confusion_matrix"]
    write_csv(directory / "mangofruitbd_confusion_matrix.csv",
              [{"true_label": label, **{f"predicted_{pred}": matrix[i][j] for j, pred in enumerate(labels)}}
               for i, label in enumerate(labels)], ["true_label", "predicted_healthy", "predicted_defective"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(np.asarray(matrix), cmap="Blues")
    ax.set(xticks=range(2), yticks=range(2), xticklabels=labels, yticklabels=labels,
           xlabel="Dự đoán", ylabel="Nhãn nguồn ánh xạ", title="MangoFruitBD — metric chính (crop-level)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(directory / "mangofruitbd_confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_contact_sheets(rows: list[dict[str, Any]], zips: dict[str, zipfile.ZipFile], output: Path,
                        unavailable_sample_limit: int = 100) -> dict[str, int]:
    def text_tile(image: Image.Image, title: str, subtitle: str) -> Image.Image:
        tile = Image.new("RGB", (248, 275), "white")
        image = image.convert("RGB")
        image.thumbnail((240, 220), Image.Resampling.LANCZOS)
        tile.paste(image, ((248 - image.width) // 2, (220 - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 222), title[:36], fill="black")
        draw.text((4, 240), subtitle[:38], fill="black")
        return tile

    unavailable = [r for r in rows if not r.get("prediction_available")]
    by_group_and_label: dict[tuple[str, str], dict[str, Any]] = {}
    for row in unavailable:
        group = str(row.get("source_group") or row["image_id"])
        label = str(row.get("source_disease_label") or row.get("original_label") or "unlabeled")
        by_group_and_label.setdefault((label, group), row)
    label_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (label, _group), row in sorted(by_group_and_label.items()):
        label_groups[label].append(row)
    # Deterministic round-robin across source labels and distinct frames.
    unavailable_sample: list[dict[str, Any]] = []
    labels = sorted(label_groups)
    cursor = 0
    while labels and len(unavailable_sample) < unavailable_sample_limit:
        label = labels[cursor % len(labels)]
        if label_groups[label]:
            unavailable_sample.append(label_groups[label].pop(0))
        if not label_groups[label]:
            labels.remove(label)
            if labels:
                cursor %= len(labels)
        else:
            cursor += 1
    selected: dict[str, list[dict[str, Any]]] = {
        "mangofruitbd_wrong_predictions": [r for r in rows if r["source"] == "MangoFruitBD_v2" and r.get("evaluation_eligible") and r.get("prediction_available") and r.get("true_label") != r.get("prediction")],
        "inference_unavailable_sample": unavailable_sample,
        "mangofruitbd_label_conflicts": [r for r in rows if r["source"] == "MangoFruitBD_v2" and r.get("label_conflict_status") != "none_observed"],
    }
    pages_by_kind = {}
    for kind, chosen in selected.items():
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("page_*.jpg"):
            stale.unlink()
        tiles = []
        for row in chosen:
            try:
                payload, filename, _, _, _ = prepare_input(row["_manifest"], zips)
                with Image.open(io.BytesIO(payload)) as image:
                    true_label = row.get("true_label") or row.get("original_label", "unmapped")
                    pred = row.get("prediction") or "unavailable"
                    subtitle = f"true={true_label}; pred={pred}; {row.get('primary_exclusion_reason') or row.get('inference_error') or 'prediction error'}"
                    tiles.append(text_tile(image, row["image_id"], subtitle))
            except Exception as exc:
                LOGGER.warning("Could not create contact tile for %s: %s", row["image_id"], exc)
        pages = 0
        for start in range(0, len(tiles), 20):
            page = tiles[start:start + 20]
            tile_w, tile_h = page[0].size
            sheet = Image.new("RGB", (4 * tile_w, math.ceil(len(page) / 4) * tile_h), "#dddddd")
            for index, tile in enumerate(page):
                sheet.paste(tile, ((index % 4) * tile_w, (index // 4) * tile_h))
            pages += 1
            sheet.save(folder / f"page_{pages:02d}.jpg", quality=88)
        pages_by_kind[kind] = pages
    return {f"{kind}_pages": value for kind, value in pages_by_kind.items()}


def render_report(rows: list[dict[str, Any]], metrics: dict[str, Any], groups: list[dict[str, Any]],
                  duplicate_rows: list[dict[str, Any]], model_id: str, contacts: dict[str, int],
                  report_path: Path) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    mfb = by_source["MangoFruitBD_v2"]
    haru = by_source["Harumanis_v4"]
    eligible = [r for r in mfb if r.get("evaluation_eligible")]
    successful = [r for r in eligible if r.get("prediction_available")]
    label_counter = Counter(r["source_disease_label"] for r in mfb)
    mfb_class_lines = []
    for disease in ("Healthy", "Alternaria", "Anthracnose", "Scab", "Alternaria|Anthracnose"):
        subset = [r for r in mfb if r.get("source_disease_label") == disease]
        if not subset:
            continue
        eval_subset = [r for r in subset if r.get("evaluation_eligible")]
        succ = [r for r in eval_subset if r.get("prediction_available")]
        correct = sum(r.get("true_label") == r.get("prediction") for r in succ)
        mfb_class_lines.append(f"| {disease} | {len(subset)} | {len(eval_subset)} | {len(succ)} | {len(succ)/len(eval_subset):.1%} | {correct}/{len(succ)} |" if eval_subset else f"| {disease} | {len(subset)} | 0 | 0 | — | Không tính (xung đột nhãn gốc) |")
    haru_lines = []
    for label in ("unripe", "semi_ripe", "ripe", "rotten"):
        subset = [r for r in haru if r["original_label"] == label]
        if not subset:
            continue
        succ = [r for r in subset if r.get("ripeness_available")]
        pred_distribution = Counter(r["ripeness_prediction"] for r in succ)
        pred_text = ", ".join(f"{k}: {v}" for k, v in sorted(pred_distribution.items())) or "không có dự đoán"
        haru_lines.append(f"| {label} | {len(subset)} | {len(succ)} | {len(succ)/len(subset):.1%} | {pred_text} | Không tính; chưa xác minh ánh xạ nhãn |")
    dup_counts = Counter((r["source"], r["duplicate_status"]) for r in duplicate_rows)
    excluded = Counter(r.get("primary_exclusion_reason", "") for r in mfb if r.get("primary_exclusion_reason"))
    mask_failures = Counter(r.get("fruit_mask_status", "unknown") for r in rows if not r.get("prediction_available"))
    mfb_failure_reasons = Counter(r.get("inference_error", "") for r in eligible if not r.get("prediction_available"))
    haru_failures = Counter(r.get("inference_error", "") for r in haru if not r.get("ripeness_available"))
    evaluated_rows = [r for r in successful]
    labels = ["healthy", "defective"]
    class_report = classification_report([r["true_label"] for r in evaluated_rows], [r["prediction"] for r in evaluated_rows], labels=labels, output_dict=True, zero_division=0) if evaluated_rows else {}
    class_lines = [f"| {label} | {class_report.get(label, {}).get('precision', 0):.3f} | {class_report.get(label, {}).get('recall', 0):.3f} | {class_report.get(label, {}).get('f1-score', 0):.3f} | {int(class_report.get(label, {}).get('support', 0))} |" for label in labels]
    group_eligible = {r["source_group"] for r in eligible}
    group_success = {r["source_group"] for r in successful}
    error_n = sum(r["true_label"] != r["prediction"] for r in successful)
    lines = [
        "# Phase 7 — Dataset v2 External Evaluation", "",
        "## Phạm vi và tính hợp lệ", "",
        "Hai artifact Phase 5.1 được dùng nguyên trạng qua `src/demo/phase6.py::infer_image`; không huấn luyện lại, không chọn lại ngưỡng và không đọc ground-truth mask lúc inference. Mỗi crop MangoFruitBD được cắt từ bbox YOLO trên ảnh đã áp EXIF transpose, sau đó gửi qua cùng decoder/preprocessing/fruit-mask/feature order của Phase 6. Ảnh Harumanis được đưa vào như ảnh gốc.",
        "",
        f"Model fingerprint: `{model_id}`. Kết quả này là **external evaluation thăm dò**, không phải benchmark chính thức: các artifact Phase 5.1 vẫn có `official_accuracy_eligible=false`, split/lineage còn provisional, và nguồn ngoài có khác biệt nhãn/miền ảnh.",
        "",
        f"Đối chiếu exact/near-duplicate với đúng các ảnh train model đã thấy: {sum(r.get('phase051_split') == 'train' for r in read_csv(SPLITS['Harumanis_v4'])):,} ảnh Dataset A cho model độ chín và {sum(r.get('phase051_split') == 'train' for r in read_csv(SPLITS['MangoFruitBD_v2'])):,} ảnh Dataset C cho model defect.",
        "",
        "MangoFruitBD nhãn `Healthy` được ánh xạ sang `healthy`; `Alternaria`, `Anthracnose`, `Scab` sang `defective`. Đây là bệnh được gán nhãn ở cấp quả, không đồng nghĩa hoàn toàn với mọi dạng defect/damage của Dataset C hoặc vết thâm màu sắc. Không dùng nhãn bệnh gốc làm feature. 103 crop có xung đột lớp bệnh gốc Alternaria/Anthracnose bị loại khỏi metric chính theo yêu cầu, dù ánh xạ nhị phân của chúng nhất quán.",
        "",
        "Harumanis giữ `rotten` riêng. Tài liệu nguồn chưa chứng minh `semi_ripe` tương đương `partially_ripe`; do đó không ánh xạ và không tính accuracy/Macro-F1 ba lớp cho Harumanis. Bảng dưới chỉ mô tả coverage và phân bố dự đoán, không gọi dự đoán đúng/sai.",
        "",
        "## Tổng hợp mẫu và near-duplicate", "",
        "| Nguồn | Candidate | Primary eligible | Inference khả dụng | Coverage candidate | Coverage primary | Exact match train | Near similarity candidate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, subset in (("Harumanis_v4", haru), ("MangoFruitBD_v2", mfb)):
        primary = [r for r in subset if r.get("evaluation_eligible")]
        available = [r for r in subset if (r.get("ripeness_available") if source == "Harumanis_v4" else r.get("defect_available"))]
        primary_available = [r for r in primary if (r.get("ripeness_available") if source == "Harumanis_v4" else r.get("defect_available"))]
        exact_n = sum(r["source"] == source and r["duplicate_status"] == "exact_match_to_training" for r in duplicate_rows)
        near_n = sum(r["source"] == source and r["duplicate_status"] in {"near_similarity_candidate_excluded", "near_similarity_screen_unconfirmed"} for r in duplicate_rows)
        lines.append(f"| {source} | {len(subset):,} | {len(primary):,} | {len(available):,} | {len(available)/len(subset):.1%} | {len(primary_available)/len(primary):.1%}" if primary else f"| {source} | {len(subset):,} | 0 | {len(available):,} | {len(available)/len(subset):.1%} | —")
        lines[-1] += f" | {exact_n} | {near_n} |"
    lines.extend(["", "Exact match means file SHA or decoded pixel SHA equality to a **train** image seen by the corresponding model. Near candidates were screened by dHash and by SSIM on aspect-preserving 256×256 grayscale letterboxes. The combined exclusion screen is dHash distance ≤4 and SSIM ≥0.90. This is a conservative risk screen, not proof that two samples depict the same physical fruit. We record lower-similarity candidates as unconfirmed, and do not claim all lineage has been resolved.", "",
                  "### Harumanis — lớp nguồn và dự đoán mô tả", "",
                  "| Nhãn nguồn | Candidate | Dự đoán khả dụng | Coverage | Phân bố nhãn dự đoán | Accuracy |",
                  "|---|---:|---:|---:|---|---|", *haru_lines,
                  "", "`rotten` được giữ nguyên và không gộp với `ripe`. Các hàng bao gồm candidate cho mô tả coverage; không dùng chúng làm metric phân loại do khác/không đủ định nghĩa lớp.",
                  "", "## MangoFruitBD — metric nhị phân", "",
                  f"Metric chính dùng {len(eligible):,}/{len(mfb):,} crop đủ điều kiện về nhãn/conflict/duplicate screen; {len(successful):,} có dự đoán (coverage trên tập primary {len(successful)/len(eligible):.1%} nếu có). Có {len(error_n and [1] or []):,} biến thể lỗi?", ""])
    # Replace an intentionally neutral placeholder with exact error count.
    lines[-2] = f"Metric chính dùng {len(eligible):,}/{len(mfb):,} crop đủ điều kiện về nhãn/conflict/duplicate screen; {len(successful):,} có dự đoán (coverage trên tập primary {len(successful)/len(eligible):.1%} nếu có). Có {error_n:,} crop dự đoán sai và {len(eligible)-len(successful):,} crop inference không khả dụng. Coverage tính cả các lỗi fruit mask; thất bại không bị xóa khỏi mẫu số."
    if metrics.get("n_evaluated"):
        lines.extend([
            f"",
            f"- Crop-level Accuracy: **{metrics['accuracy']:.4f}**.",
            f"- Crop-level Macro-F1: **{metrics['macro_f1']:.4f}**.",
            f"- Frame-equal weighted Accuracy: **{metrics['group_weighted_accuracy']:.4f}**.",
            f"- Frame-equal weighted Macro-F1: **{metrics['group_weighted_macro_f1']:.4f}**.",
            f"- Source groups có crop primary: {len(group_eligible):,}; có ít nhất một inference thành công: {len(group_success):,}.",
            "- Frame-equal weight gán tổng trọng số 1 cho mỗi `source_group`, chia đều cho các crop đủ điều kiện trong group. Các crop cùng frame vẫn được giữ ở cấp quả nhưng không được coi như các lần chụp độc lập.",
            "", "### Classification report", "",
            "| Nhãn | Precision | Recall | F1 | Support crop |", "|---|---:|---:|---:|---:|", *class_lines,
            "", "Confusion matrix (hàng = nhãn nguồn ánh xạ, cột = dự đoán): `reports/dataset_v2_external/mangofruitbd_confusion_matrix.csv` và PNG.",
        ])
    else:
        lines.extend(["", "Không có crop đủ điều kiện và dự đoán khả dụng để tính metric."])
    lines.extend(["", "### Theo nhãn bệnh nguồn", "",
                  "| Nhãn gốc | Candidate crop | Crop đủ điều kiện | Dự đoán khả dụng | Coverage | Kết quả |",
                  "|---|---:|---:|---:|---:|---|", *mfb_class_lines,
                  "", "Nhãn xung đột `Alternaria|Anthracnose` được hiển thị riêng để truy xuất nhưng không đưa vào metric chính.",
                  "", "### Theo source group/frame", "",
                  f"Có {len(groups):,} group trong toàn candidate; {len(group_eligible):,} group có ít nhất một crop đủ điều kiện. Bảng chi tiết mỗi group nằm ở `reports/dataset_v2_external/source_group_summary.csv`.",
                  "", "## Coverage và lỗi", "",
                  f"Trạng thái mask trong các inference không khả dụng: `{dict(mask_failures)}`. Trong {len(eligible)-len(successful):,} crop MangoFruitBD đủ điều kiện nhưng không suy luận được, lý do gồm `{dict(mfb_failure_reasons)}`. Hai ảnh Harumanis không dự đoán được có trạng thái/lý do `{dict(haru_failures)}`. Các lỗi theo từng ảnh/crop, nhãn nguồn, dự đoán, trạng thái fruit mask, near-duplicate screen và lý do loại metric được lưu ở `reports/dataset_v2_external/predictions_by_sample.csv`; subset lỗi/không khả dụng nằm ở `error_analysis.csv`.",
                  "", "Contact sheets:", ""])
    for kind, pages in contacts.items():
        if kind == "mangofruitbd_wrong_predictions_pages":
            description = f"toàn bộ {error_n} dự đoán sai đủ điều kiện"
        elif kind == "inference_unavailable_sample_pages":
            description = "mẫu đại diện tối đa 100 source group không suy luận được"
        else:
            description = f"{sum(r['source'] == 'MangoFruitBD_v2' and r.get('label_conflict_status') != 'none_observed' for r in rows)} crop xung đột nhãn gốc"
        lines.append(f"- `{kind}`: {pages} trang ({description}) dưới `reports/dataset_v2_external/contact_sheets/`.")
    if "inference_unavailable_sample_pages" in contacts:
        lines.append("- Contact sheet inference-unavailable là mẫu đại diện tối đa 100 source group, chia vòng theo nhãn nguồn; toàn bộ mẫu lỗi vẫn có trong prediction/error CSV và coverage.")
    lines.extend(["", "## Giới hạn và diễn giải", "",
                  "1. Đây là external evaluation trên nguồn và bối cảnh khác; không thay thế benchmark chính thức.",
                  "2. MangoFruitBD phân loại bệnh cụ thể, trong khi model defect được fit trên nhãn `healthy/defective` của Dataset C. Metric là phép đối chiếu proxy ở cấp quả; không chứng minh phát hiện vết thâm.",
                  "3. MangoFruitBD có khác biệt nền/độ phân giải theo lớp, đặc biệt Alternaria; kết quả có thể phản ánh domain/capture bias. Weighted metrics giảm ảnh hưởng số crop trên một frame nhưng không khử shortcut nền.",
                  "4. Harumanis source labels chưa được rà soát thủ công, `semi_ripe` chưa ánh xạ được, và `rotten` không có lớp đích riêng trong model.",
                  "5. Heuristic fruit mask Phase 2 có thể thất bại trên crop nhỏ/ảnh nền; tất cả lần thất bại vẫn nằm trong coverage denominator. Ảnh không được xác nhận có xoài bằng OOD model.",
                  "6. dHash/SSIM chỉ sàng lọc gần trùng; các candidate chưa xác minh trực quan vẫn mang cờ rủi ro. Không tuyên bố lineage sạch hoàn toàn.",
                  "7. Curation candidate chưa phải dataset được phê duyệt. Kết quả không được dùng để điều chỉnh cấu hình/model.",
                  "", "## Tệp kết quả", "",
                  "- `reports/dataset_v2_external/metrics.json` — metric, coverage và duplicate counts.",
                  "- `reports/dataset_v2_external/predictions_by_sample.csv` — kết quả từng mẫu.",
                  "- `reports/dataset_v2_external/duplicate_audit.csv` — exact/near screen so với train.",
                  "- `reports/dataset_v2_external/source_group_summary.csv` — thống kê theo frame/source group.",
                  "- `reports/dataset_v2_external/error_analysis.csv` — lỗi và inference unavailable.",
                  "- `reports/dataset_v2_external/contact_sheets/` — preview lỗi/không khả dụng/xung đột.",
                  "- `reports/dataset_v2_external/mangofruitbd_confusion_matrix.png` — confusion matrix.", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(resume: bool = True) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assets = load_demo_assets(ROOT)
    fingerprint = model_fingerprint(assets)
    candidates = read_csv(MANIFEST)
    expected_sources = {"Harumanis_v4", "MangoFruitBD_v2"}
    if not candidates or not {r["source"] for r in candidates} >= expected_sources:
        raise RuntimeError("Dataset v2 curated candidate manifest is missing or incomplete")
    train_refs = load_training_references()
    old_rows = read_csv(OUT / "predictions_by_sample.csv") if resume and (OUT / "predictions_by_sample.csv").is_file() else []
    old_by_id = {r["image_id"]: r for r in old_rows if r.get("model_fingerprint") == fingerprint}
    old_duplicate_rows = read_csv(OUT / "duplicate_audit.csv") if resume and (OUT / "duplicate_audit.csv").is_file() else []
    old_duplicate_by_id = {r["image_id"]: r for r in old_duplicate_rows}
    zip_handles: dict[str, zipfile.ZipFile] = {}
    predictions: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    try:
        for number, source_row in enumerate(candidates, 1):
            LOGGER.info("Preparing candidate %d/%d: %s", number, len(candidates), source_row["image_id"])
            base = dict(source_row)
            record = {"image_id": source_row["image_id"], "source": source_row["source"],
                      "source_path": source_row["original_path"], "source_sha256": "",
                      "input_sha256": "", "input_pixel_sha256": "", "input_dhash": "",
                      "source_group": source_row["source_group"], "duplicate_group": source_row["duplicate_group"],
                      "original_label": source_row["original_label"], "source_disease_label": source_row["original_label"],
                      "true_label": source_row["curated_label"] if source_row["source"] == "MangoFruitBD_v2" else "",
                      "label_conflict_status": source_row["label_conflict_status"],
                      "duplicate_status": "", "nearest_train_image_id": "", "nearest_train_dhash_distance": "",
                      "nearest_train_ssim_gray_256": "", "near_duplicate_screen": "", "primary_exclusion_reason": "",
                      "evaluation_eligible": False, "prediction_available": False, "prediction": "",
                      "inference_error": "", "fruit_mask_status": "", "ripeness_prediction": "",
                      "defect_prediction": "", "ripeness_available": False, "defect_available": False,
                      "kmeans_status": "", "model_fingerprint": fingerprint, "_manifest": base}
            try:
                payload, filename, source_rgb_pil, input_rgb_pil, source_sha = prepare_input(source_row, zip_handles)
                record["source_sha256"] = source_sha
                record["input_sha256"] = sha256_bytes(payload)
                old = old_by_id.get(source_row["image_id"])
                if old and old.get("input_sha256") == record["input_sha256"] and old.get("prediction_available", "").casefold() in {"true", "1"}:
                    for field in ("prediction_available", "prediction", "inference_error", "fruit_mask_status", "ripeness_prediction", "defect_prediction", "ripeness_available", "defect_available", "kmeans_status", "input_pixel_sha256", "input_dhash", "nearest_train_image_id", "nearest_train_dhash_distance", "nearest_train_ssim_gray_256", "near_duplicate_screen", "duplicate_status"):
                        record[field] = old.get(field, record[field])
                    cached_dup = old_duplicate_by_id.get(source_row["image_id"], {})
                    duplicate = {**cached_dup, "duplicate_status": record["duplicate_status"]}
                    LOGGER.info("Resumed cached inference and duplicate audit for %s", source_row["image_id"])
                else:
                    # Keep the candidate in coverage even if Phase 6 rejects its upload size.
                    try:
                        frame = decode_upload(payload, filename)
                        input_rgb = frame.rgb
                    except Exception as decode_exc:
                        input_rgb = input_rgb_pil
                        record["inference_error"] = f"{type(decode_exc).__name__}: {decode_exc}"
                    duplicate = duplicate_audit(source_sha, source_rgb_pil, input_rgb, train_refs[source_row["source"]])
                record.update({k: duplicate[k] for k in ("input_pixel_sha256", "input_dhash", "nearest_train_image_id", "nearest_train_dhash_distance", "nearest_train_ssim_gray_256", "near_duplicate_screen", "duplicate_status")})
                record["primary_exclusion_reason"] = primary_exclusion_reason(source_row, duplicate["duplicate_status"])
                record["evaluation_eligible"] = source_row["source"] == "MangoFruitBD_v2" and not record["primary_exclusion_reason"]
                if not record["prediction_available"] and record["inference_error"] == "":
                    result = infer_image(payload, filename, assets)
                    record["fruit_mask_status"] = str(result.get("fruit_mask_status", "unknown"))
                    record["ripeness_available"] = bool(result["ripeness"].get("available"))
                    record["defect_available"] = bool(result["defect"].get("available"))
                    record["ripeness_prediction"] = result["ripeness"].get("prediction") or ""
                    record["defect_prediction"] = result["defect"].get("prediction") or ""
                    record["prediction"] = record["defect_prediction"] if source_row["source"] == "MangoFruitBD_v2" else record["ripeness_prediction"]
                    record["prediction_available"] = record["defect_available"] if source_row["source"] == "MangoFruitBD_v2" else record["ripeness_available"]
                    record["kmeans_status"] = str(result.get("kmeans", {}).get("status", "unknown"))
                    if not record["prediction_available"]:
                        record["inference_error"] = result["defect" if source_row["source"] == "MangoFruitBD_v2" else "ripeness"].get("reason", "prediction unavailable")
                elif record["inference_error"]:
                    # Phase 6 rejected the payload before inference; retain it as a coverage failure.
                    record["prediction_available"] = False
                    record["ripeness_available"] = False
                    record["defect_available"] = False
                duplicates.append({"image_id": record["image_id"], "source": record["source"], "source_path": record["source_path"],
                                   "source_sha256": record["source_sha256"], **duplicate})
            except Exception as exc:
                LOGGER.exception("Candidate failed: %s", source_row["image_id"])
                record["inference_error"] = f"{type(exc).__name__}: {exc}"
                record["prediction_available"] = False
                record["duplicate_status"] = "not_checked_due_to_input_error"
                record["primary_exclusion_reason"] = "invalid_or_unusable_crop" if source_row["source"] == "MangoFruitBD_v2" and not record["source_sha256"] else primary_exclusion_reason(source_row, record["duplicate_status"])
                record["evaluation_eligible"] = source_row["source"] == "MangoFruitBD_v2" and not record["primary_exclusion_reason"]
                duplicates.append({"image_id": record["image_id"], "source": record["source"], "source_path": record["source_path"],
                                   "source_sha256": "", "source_file_sha256_match": "", "source_pixel_sha256_match": "",
                                   "input_pixel_sha256_match": "", "input_dhash": "", "nearest_train_image_id": "",
                                   "nearest_train_dhash_distance": "", "nearest_train_ssim_gray_256": "",
                                   "near_duplicate_screen": "not_checked_due_to_input_error", "duplicate_status": record["duplicate_status"]})
            # Booleans serialize consistently for later resume.
            predictions.append(record)
            if number % 20 == 0 or number == len(candidates):
                write_csv(OUT / "predictions_by_sample.csv", predictions, PREDICTION_FIELDS)
                write_csv(OUT / "duplicate_audit.csv", duplicates,
                          ["image_id", "source", "source_path", "source_sha256", "source_file_sha256_match", "source_pixel_sha256_match", "input_pixel_sha256_match", "source_raw_pixel_sha256", "input_pixel_sha256", "input_dhash", "nearest_train_image_id", "nearest_train_dhash_distance", "nearest_train_ssim_gray_256", "near_duplicate_screen", "duplicate_status"])
            if number % 50 == 0:
                LOGGER.info("Progress %d/%d", number, len(candidates))
    finally:
        for handle in zip_handles.values():
            handle.close()
    # Add internal helper data only for contact sheets; never serialize it.
    manifest_by_id = {r["image_id"]: r for r in candidates}
    for row in predictions:
        row["_manifest"] = manifest_by_id[row["image_id"]]
        row["evaluation_eligible"] = str(row["evaluation_eligible"]).casefold() in {"true", "1"}
        row["prediction_available"] = str(row["prediction_available"]).casefold() in {"true", "1"}
        row["ripeness_available"] = str(row["ripeness_available"]).casefold() in {"true", "1"}
        row["defect_available"] = str(row["defect_available"]).casefold() in {"true", "1"}
    metrics = evaluate_predictions(predictions, ["healthy", "defective"])
    eligible = [r for r in predictions if r["evaluation_eligible"]]
    available = [r for r in eligible if r["prediction_available"]]
    groups = summarize_groups(predictions)
    group_eligible = {r["source_group"] for r in eligible}
    group_available = {r["source_group"] for r in available}
    metrics.update({
        "candidate_count": len(predictions),
        "by_source": {source: {
            "candidate": sum(r["source"] == source for r in predictions),
            "primary_eligible": sum(r["source"] == source and r["evaluation_eligible"] for r in predictions),
            "prediction_available": sum(r["source"] == source and (r["ripeness_available"] if source == "Harumanis_v4" else r["defect_available"]) for r in predictions),
            "primary_coverage": (sum(r["source"] == source and r["evaluation_eligible"] and r["prediction_available"] for r in predictions) /
                                 max(1, sum(r["source"] == source and r["evaluation_eligible"] for r in predictions))),
        } for source in ("Harumanis_v4", "MangoFruitBD_v2")},
        "mfb_primary_source_groups": len(group_eligible),
        "mfb_source_groups_with_inference": len(group_available),
        "mfb_source_group_coverage": len(group_available) / len(group_eligible) if group_eligible else None,
        "mfb_prediction_errors": sum(r["true_label"] != r["prediction"] for r in available),
        "mfb_inference_unavailable": len(eligible) - len(available),
        "mfb_exclusions": dict(Counter(r["primary_exclusion_reason"] for r in predictions if r["source"] == "MangoFruitBD_v2" and r["primary_exclusion_reason"])),
        "duplicate_status_counts": {source: dict(Counter(r["duplicate_status"] for r in predictions if r["source"] == source)) for source in ("Harumanis_v4", "MangoFruitBD_v2")},
        "model_fingerprint": fingerprint,
        "models": {"ripeness": "Phase 5.1 selected Gray World + KNN", "defect": "Phase 5.1 selected Original Color + SVM-RBF"},
    })
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUT / "source_group_summary.csv", groups,
              ["source_group", "candidate_crops", "primary_eligible_crops", "inference_available_crops", "coverage", "correct_crops", "error_crops", "source_disease_labels", "binary_labels"])
    error_rows = [r for r in predictions if (r["evaluation_eligible"] and (not r["prediction_available"] or r["true_label"] != r["prediction"])) or (r["source"] == "MangoFruitBD_v2" and r["primary_exclusion_reason"])]
    write_csv(OUT / "error_analysis.csv", error_rows, PREDICTION_FIELDS)
    write_csv(OUT / "duplicate_audit.csv", duplicates,
              ["image_id", "source", "source_path", "source_sha256", "source_file_sha256_match", "source_pixel_sha256_match", "input_pixel_sha256_match", "source_raw_pixel_sha256", "input_pixel_sha256", "input_dhash", "nearest_train_image_id", "nearest_train_dhash_distance", "nearest_train_ssim_gray_256", "near_duplicate_screen", "duplicate_status"])
    confusion_outputs(predictions, metrics, OUT)
    with zipfile.ZipFile(ARCHIVES["harumanis_v4_all.zip"]) as hz, zipfile.ZipFile(ARCHIVES["MangoFruitBD.zip"]) as mz:
        contacts = save_contact_sheets(predictions, {"harumanis_v4_all.zip": hz, "MangoFruitBD.zip": mz}, OUT / "contact_sheets")
    render_report(predictions, metrics, groups, duplicates, fingerprint, contacts, REPORT)
    return metrics


def finalize_existing_outputs(rebuild_contact_sheets: bool = True) -> dict[str, Any]:
    """Rebuild report and compact contact sheets without repeating inference."""
    predictions = read_csv(OUT / "predictions_by_sample.csv")
    manifest = {r["image_id"]: r for r in read_csv(MANIFEST)}
    for row in predictions:
        row["_manifest"] = manifest[row["image_id"]]
        for key in ("evaluation_eligible", "prediction_available", "ripeness_available", "defect_available"):
            row[key] = str(row.get(key, "")).casefold() in {"true", "1"}
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    groups = read_csv(OUT / "source_group_summary.csv")
    duplicates = read_csv(OUT / "duplicate_audit.csv")
    if rebuild_contact_sheets:
        with zipfile.ZipFile(ARCHIVES["harumanis_v4_all.zip"]) as hz, zipfile.ZipFile(ARCHIVES["MangoFruitBD.zip"]) as mz:
            contacts = save_contact_sheets(predictions, {"harumanis_v4_all.zip": hz, "MangoFruitBD.zip": mz}, OUT / "contact_sheets")
    else:
        contacts = {f"{folder.name}_pages": len(list(folder.glob("page_*.jpg")))
                    for folder in (OUT / "contact_sheets").iterdir() if folder.is_dir()}
    render_report(predictions, metrics, groups, duplicates, metrics["model_fingerprint"], contacts, REPORT)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="External evaluation on curated Dataset v2 candidates with locked Phase 5.1 models")
    parser.add_argument("--no-resume", action="store_true", help="Ignore cached completed inference rows")
    parser.add_argument("--finalize-only", action="store_true", help="Create report/contact sheets from completed outputs without inference")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report from existing metrics/contact sheets without inference or image decoding")
    args = parser.parse_args()
    result = finalize_existing_outputs(rebuild_contact_sheets=not args.report_only) if (args.finalize_only or args.report_only) else run(resume=not args.no_resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
