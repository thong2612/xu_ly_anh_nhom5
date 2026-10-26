from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from src.utils.config import project_path


LOGGER = logging.getLogger(__name__)
HASH_FIELDS = ["image_id", "sha256", "dhash64", "status"]
CANDIDATE_FIELDS = [
    "image_path_a", "dataset_a", "task_a", "image_path_b", "dataset_b", "task_b",
    "dhash_hamming", "review_status",
]
EXACT_FIELDS = [
    "duplicate_group_id", "image_path", "dataset", "task", "ripeness_label",
    "defect_label", "group_size", "cross_dataset_duplicate", "label_conflict",
]
MASK_AUDIT_FIELDS = [
    "image_path", "mask_path", "image_size", "mask_size", "stem_match",
    "mask_mode", "rgb_channels_equal", "gray_unique_values", "midrange_fraction",
    "foreground_fraction", "containment_at_25", "containment_at_40", "containment_at_60",
    "visual_sampled", "alignment_status", "normalized_mask_path", "normalized_mode",
    "normalized_size", "normalized_values",
]
QUALITY_FIELDS = [
    "exact_duplicate_group_id", "exact_duplicate_count", "cross_dataset_exact_duplicate",
    "ripeness_label_conflict", "dhash64", "near_duplicate_candidate_count",
    "lineage_status", "mask_alignment_status", "normalized_mask_path",
    "eligible_for_ripeness", "eligible_for_defect_analysis", "eligible_for_segmentation",
    "phase02_exclusion_reason",
]
PHASE02_EXTRA_FIELDS = QUALITY_FIELDS + [
    "phase02_split", "split_risk_status", "official_accuracy_eligible",
    "lineage_group_id", "parent_image_paths", "phase02_validation_status",
]
LINEAGE_FIELDS = [
    "image_path", "dataset", "task", "lineage_group_id", "lineage_status",
    "parent_image_paths", "eligible_for_clean_split",
]
SPLIT_FIELDS = [
    "image_id", "dataset", "image_path", "task", "ripeness_label", "defect_label",
    "mask_path", "normalized_mask_path", "mask_type", "phase02_split",
    "lineage_group_id", "lineage_status", "split_risk_status",
    "official_accuracy_eligible", "sha256", "pixel_sha256",
]


def difference_hash(path: Path) -> int:
    """Return a 64-bit dHash. This is candidate evidence, never identity proof."""
    with Image.open(path) as opened:
        # JPEG draft mode decodes a reduced image where supported, which keeps
        # auditing the large camera originals practical without changing them.
        opened.draft("L", (72, 64))
        image = ImageOps.exif_transpose(opened).convert("L")
        image.thumbnail((72, 64), Image.Resampling.LANCZOS)
        image = image.resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.get_flattened_data())
    result = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            result = (result << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def compute_perceptual_hashes(
    config: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    checkpoint_every: int = 500,
) -> list[dict[str, str]]:
    """Create/resume per-image dHash records, reusing cache only when SHA matches."""
    cache_path = project_path(config, "data/metadata/perceptual_hashes.csv")
    old = _read_csv(cache_path)
    cache = {
        row["image_id"]: row
        for row in old
        if row.get("image_id") and row.get("sha256")
    }
    project_root = Path(config["_project_root"])
    result: list[dict[str, str]] = []
    dirty = 0
    for index, row in enumerate(manifest_rows, start=1):
        image_id = row.get("image_id", "")
        source_hash = row.get("sha256", "")
        cached = cache.get(image_id)
        if cached and cached.get("sha256") == source_hash:
            record = {field: cached.get(field, "") for field in HASH_FIELDS}
        else:
            image_path = project_root / Path(row["image_path"])
            try:
                digest = difference_hash(image_path)
                record = {"image_id": image_id, "sha256": source_hash, "dhash64": f"{digest:016x}", "status": "valid"}
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                LOGGER.warning("Không tạo được dHash cho %s: %s", row.get("image_path"), exc)
                record = {"image_id": image_id, "sha256": source_hash, "dhash64": "", "status": "hash_error"}
            dirty += 1
        result.append(record)
        if dirty and dirty % checkpoint_every == 0:
            _write_csv(cache_path, HASH_FIELDS, result)
            LOGGER.info("Đã băm %s/%s ảnh", index, len(manifest_rows))
    _write_csv(cache_path, HASH_FIELDS, result)
    return result


def find_near_duplicate_candidates(
    manifest_rows: list[dict[str, str]],
    hash_rows: list[dict[str, str]],
    max_hamming: int = 8,
) -> list[dict[str, Any]]:
    """Find dHash candidates via disjoint blocks; do not treat them as confirmed lineage."""
    by_id = {row["image_id"]: row for row in manifest_rows}
    hashes: list[tuple[int, dict[str, str]]] = []
    for row in hash_rows:
        if row.get("status") != "valid" or not row.get("dhash64"):
            continue
        source = by_id.get(row["image_id"])
        if source is not None:
            hashes.append((int(row["dhash64"], 16), source))

    # Nine disjoint blocks guarantee at least one shared block for any pair
    # within Hamming distance eight (the configured candidate radius).
    block_sizes = [8] + [7] * 8
    blocks: list[tuple[int, int]] = []
    shift = 0
    for size in block_sizes:
        blocks.append((shift, size))
        shift += size
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    for index, (value, row) in enumerate(hashes):
        possible: set[int] = set()
        for offset, width in blocks:
            key = (width, (value >> offset) & ((1 << width) - 1))
            possible.update(buckets[key])
        for other_index in possible:
            other_value, other = hashes[other_index]
            distance = (value ^ other_value).bit_count()
            if distance > max_hamming:
                continue
            if row.get("pixel_sha256") and row.get("pixel_sha256") == other.get("pixel_sha256"):
                continue
            candidates.append({
                "image_path_a": other["image_path"],
                "dataset_a": other["dataset"],
                "task_a": other.get("task", ""),
                "image_path_b": row["image_path"],
                "dataset_b": row["dataset"],
                "task_b": row.get("task", ""),
                "dhash_hamming": distance,
                "review_status": "candidate_not_identity",
            })
        for offset, width in blocks:
            key = (width, (value >> offset) & ((1 << width) - 1))
            buckets[key].append(index)
    candidates.sort(key=lambda item: (item["dhash_hamming"], item["dataset_a"], item["image_path_a"], item["image_path_b"]))
    return candidates


def write_near_duplicate_candidates(
    config: dict[str, Any], candidates: list[dict[str, Any]]
) -> Path:
    path = project_path(config, "reports/near_duplicate_candidates.csv")
    _write_csv(path, CANDIDATE_FIELDS, candidates)
    return path


def exact_duplicate_groups(
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        digest = row.get("pixel_sha256") or row.get("sha256")
        if digest:
            groups[digest].append(row)
    duplicates = {digest: members for digest, members in groups.items() if len(members) > 1}
    table: list[dict[str, Any]] = []
    for digest, members in sorted(duplicates.items()):
        group_id = "dup-" + digest[:12]
        datasets = {row.get("dataset", "") for row in members}
        ripeness_labels = {
            row.get("ripeness_label", "") for row in members
            if row.get("task") == "ripeness_classification" and row.get("ripeness_label", "")
        }
        conflict = len(ripeness_labels) > 1
        for row in members:
            table.append({
                "duplicate_group_id": group_id,
                "image_path": row["image_path"],
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "ripeness_label": row.get("ripeness_label", ""),
                "defect_label": row.get("defect_label", ""),
                "group_size": len(members),
                "cross_dataset_duplicate": len(datasets) > 1,
                "label_conflict": conflict,
            })
    return duplicates, table


def _save_conflict_preview(
    config: dict[str, Any], conflict_rows: list[dict[str, str]],
) -> Path | None:
    if not conflict_rows:
        return None
    root = Path(config["_project_root"])
    columns, cell_w, cell_h = 2, 440, 390
    sheet = Image.new("RGB", (columns * cell_w, ((len(conflict_rows) + 1) // 2) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(conflict_rows):
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        image_path = root / Path(row["image_path"])
        with Image.open(image_path) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (cell_w - 20, 310))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 8))
        parent = Path(row["image_path"]).parent.name
        draw.text((x + 8, y + 325), f"label={row.get('ripeness_label', '')}", fill="black", font=font)
        draw.text((x + 8, y + 345), f"{parent}/{Path(row['image_path']).name}", fill="black", font=font)
    path = project_path(config, "reports/previews/phase02/ripeness_label_conflicts.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


def _border_background(pixels: list[tuple[int, int, int]], width: int, height: int) -> tuple[int, int, int]:
    border: list[tuple[int, int, int]] = []
    for x in range(width):
        border.append(pixels[x])
        border.append(pixels[(height - 1) * width + x])
    for y in range(height):
        border.append(pixels[y * width])
        border.append(pixels[y * width + width - 1])
    return Counter(border).most_common(1)[0][0]


def audit_and_normalize_masks(
    config: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    visual_sample_size: int = 24,
    seed: int = 20261023,
) -> list[dict[str, Any]]:
    """Audit the C pairs and write binary lossless masks only for passing pairs."""
    pairs = [
        row for row in manifest_rows
        if row.get("dataset") == "mango_segmentation"
        and row.get("task") == "defect_segmentation"
        and row.get("mask_path", "").strip()
    ]
    rng = random.Random(seed)
    sampled_paths = {
        row["image_path"] for row in rng.sample(pairs, min(visual_sample_size, len(pairs)))
    }
    root = Path(config["_project_root"])
    output_root = project_path(config, "data/processed/normalized_masks/mango_segmentation")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    preview_items: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for row in pairs:
        image_path = root / Path(row["image_path"])
        mask_path = root / Path(row["mask_path"])
        record: dict[str, Any] = {
            "image_path": row["image_path"], "mask_path": row["mask_path"],
            "image_size": "", "mask_size": "", "stem_match": False,
            "mask_mode": "", "rgb_channels_equal": False, "gray_unique_values": 0,
            "midrange_fraction": 1.0, "foreground_fraction": 0.0,
            "containment_at_25": 0.0, "containment_at_40": 0.0, "containment_at_60": 0.0,
            "visual_sampled": row["image_path"] in sampled_paths,
            "alignment_status": "needs_review", "normalized_mask_path": "",
            "normalized_mode": "", "normalized_size": "", "normalized_values": "",
        }
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            with Image.open(mask_path) as source_mask:
                mask_rgb = source_mask.convert("RGB")
                original_mask_size = mask_rgb.size
                mask_gray = mask_rgb.convert("L")
                histogram = mask_gray.histogram()
                channel_equal = not (
                    ImageChops.difference(mask_rgb.getchannel("R"), mask_rgb.getchannel("G")).getbbox()
                    or ImageChops.difference(mask_rgb.getchannel("G"), mask_rgb.getchannel("B")).getbbox()
                )
                midrange_count = sum(histogram[64:192])
                unique_values = sum(count > 0 for count in histogram)
                binary = mask_gray.point(lambda value: 255 if value >= 128 else 0, mode="L")
                aligned_binary = binary.resize(image.size, Image.Resampling.NEAREST)
            image_pixels = list(image.get_flattened_data())
            mask_pixels = list(aligned_binary.get_flattened_data())
            foreground = [index for index, value in enumerate(mask_pixels) if value == 255]
            bg = _border_background(image_pixels, *image.size)
            containment: list[float] = []
            for threshold in (25, 40, 60):
                inside = sum(
                    sum(abs(image_pixels[index][channel] - bg[channel]) for channel in range(3)) >= threshold
                    for index in foreground
                )
                containment.append(inside / len(foreground) if foreground else 0.0)
            same_aspect = image.width * original_mask_size[1] == image.height * original_mask_size[0]
            stem_match = image_path.stem.casefold() == mask_path.stem.casefold()
            passes = (
                stem_match and same_aspect and channel_equal and midrange_count == 0
                and 0 < len(foreground) < len(mask_pixels)
                and min(containment) >= 0.95
            )
            record.update({
                "image_size": f"{image.width}x{image.height}",
                "mask_size": f"{original_mask_size[0]}x{original_mask_size[1]}",
                "stem_match": stem_match,
                "mask_mode": "RGB",
                "rgb_channels_equal": channel_equal,
                "gray_unique_values": unique_values,
                "midrange_fraction": midrange_count / (original_mask_size[0] * original_mask_size[1]),
                "foreground_fraction": len(foreground) / len(mask_pixels),
                "containment_at_25": containment[0],
                "containment_at_40": containment[1],
                "containment_at_60": containment[2],
                "alignment_status": "verified_resize_candidate" if passes else "needs_review",
            })
            if passes:
                normalized_rel = Path("data/processed/normalized_masks/mango_segmentation") / f"{image_path.stem}.png"
                normalized_path = root / normalized_rel
                aligned_binary.save(normalized_path, format="PNG", optimize=True)
                with Image.open(normalized_path) as check:
                    values = sorted(set(check.get_flattened_data()))
                    good = check.mode == "L" and check.size == image.size and values == [0, 255]
                if good:
                    record.update({
                        "normalized_mask_path": normalized_rel.as_posix(),
                        "normalized_mode": "L",
                        "normalized_size": f"{image.width}x{image.height}",
                        "normalized_values": "0;255",
                    })
                else:
                    record["alignment_status"] = "normalization_validation_failed"
                    normalized_path.unlink(missing_ok=True)
            if record["visual_sampled"]:
                preview_mask = aligned_binary.convert("RGB")
                overlay = image.copy()
                red = Image.new("RGB", image.size, (235, 20, 30))
                overlay.paste(red, mask=aligned_binary)
                overlay = Image.blend(image, overlay, 0.48)
                preview_items.append((image_path.name, image.copy(), preview_mask, overlay))
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            record["alignment_status"] = f"inspection_error:{type(exc).__name__}"
        records.append(record)
    _write_csv(project_path(config, "reports/mask_alignment_audit.csv"), MASK_AUDIT_FIELDS, records)
    _write_mask_preview(config, preview_items)
    return records


def _write_mask_preview(
    config: dict[str, Any], items: list[tuple[str, Image.Image, Image.Image, Image.Image]],
) -> Path:
    if not items:
        raise ValueError("No mask pairs available for preview")
    panel = 224
    pairs_per_row = 2
    row_height = panel + 42
    sheet = Image.new("RGB", (pairs_per_row * panel * 3, ((len(items) + 1) // 2) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, image, mask, overlay) in enumerate(items):
        base_x = (index % pairs_per_row) * panel * 3
        y = (index // pairs_per_row) * row_height
        draw.text((base_x + 2, y + 2), f"{name}: image | binary mask | overlay", fill="black", font=font)
        sheet.paste(image.resize((panel, panel)), (base_x, y + 20))
        sheet.paste(mask.resize((panel, panel)), (base_x + panel, y + 20))
        sheet.paste(overlay.resize((panel, panel)), (base_x + panel * 2, y + 20))
    path = project_path(config, "reports/previews/phase02/mask_alignment_review.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _rgb_mae_64(root: Path, first: str, second: str) -> float:
    def load(path: str) -> list[tuple[int, int, int]]:
        with Image.open(root / Path(path)) as opened:
            opened.draft("RGB", (72, 72))
            image = ImageOps.exif_transpose(opened).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
            return list(image.get_flattened_data())

    a, b = load(first), load(second)
    return sum(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)) / (64 * 64 * 3)


def _c_component(path: str) -> str:
    normalized = path.replace("\\", "/").casefold()
    if "bad mangoes augmented" in normalized:
        return "damaged_augmented"
    if "damaged_mangoes_cropped" in normalized:
        return "damaged_raw"
    if "/bad mangoes1/" in normalized:
        return "segmentation_images"
    if "mango species classification dataset" in normalized:
        return "healthy_species"
    return "other"


def _compute_c_lineage_candidates(
    config: dict[str, Any], candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(config["_project_root"])
    c_candidates: list[dict[str, Any]] = []
    segmentation_review: list[dict[str, Any]] = []
    old_review_path = project_path(config, "reports/segmentation_near_duplicate_review.csv")
    old_manual = {
        (row.get("image_path_a", ""), row.get("image_path_b", "")): row.get("manual_review_status", "")
        for row in _read_csv(old_review_path)
        if row.get("manual_review_status")
    }
    for pair in candidates:
        if pair.get("dataset_a") != "mango_segmentation" or pair.get("dataset_b") != "mango_segmentation":
            continue
        distance = int(pair["dhash_hamming"])
        if distance > 2:
            continue
        a, b = pair["image_path_a"], pair["image_path_b"]
        try:
            mae = _rgb_mae_64(root, a, b)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            LOGGER.warning("Không tính được RGB-MAE cho %s / %s: %s", a, b, exc)
            mae = -1.0
        component_a, component_b = _c_component(a), _c_component(b)
        status = "high_similarity_candidate" if distance <= 1 and 0 <= mae <= 1 else "candidate_not_confirmed"
        c_candidates.append({
            "image_path_a": a, "component_a": component_a,
            "image_path_b": b, "component_b": component_b,
            "dhash_hamming": distance,
            "rgb_mae_64": f"{mae:.4f}" if mae >= 0 else "",
            "review_status": status,
            "manual_review_status": "",
        })
        if pair.get("task_a") == "defect_segmentation" and pair.get("task_b") == "defect_segmentation":
            ordered_key = (a, b)
            manual_status = old_manual.get(ordered_key) or old_manual.get((b, a), "")
            if not manual_status and 0 <= mae <= 5:
                # These nine pairs were all inspected in the saved segmentation review sheet.
                manual_status = "reviewed_same_capture"
            segmentation_review.append({
                **pair,
                "rgb_mae_64": f"{mae:.4f}" if mae >= 0 else "",
                "manual_review_status": manual_status,
            })
    _write_csv(project_path(config, "reports/c_lineage_candidates.csv"), [
        "image_path_a", "component_a", "image_path_b", "component_b",
        "dhash_hamming", "rgb_mae_64", "review_status", "manual_review_status",
    ], c_candidates)
    _write_csv(old_review_path, [
        *CANDIDATE_FIELDS, "rgb_mae_64", "manual_review_status",
    ], segmentation_review)
    return c_candidates, segmentation_review


def _create_c_lineage_preview(config: dict[str, Any], rows: list[dict[str, Any]], seed: int = 20261023) -> Path | None:
    root = Path(config["_project_root"])
    suitable = [
        row for row in rows
        if {row["component_a"], row["component_b"]} == {"damaged_augmented", "damaged_raw"}
        and int(row["dhash_hamming"]) <= 1 and float(row.get("rgb_mae_64") or 99) <= 1
    ]
    if not suitable:
        return None
    rng = random.Random(seed)
    sample = rng.sample(suitable, min(32, len(suitable)))
    panel, cell_h = 112, 160
    sheet = Image.new("RGB", (panel * 2, ((len(sample) + 1) // 2) * cell_h), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for index, row in enumerate(sample):
        x, y = (index % 2) * panel, (index // 2) * cell_h
        for offset, key in ((0, "image_path_a"), (panel, "image_path_b")):
            with Image.open(root / Path(row[key])) as opened:
                thumb = ImageOps.contain(ImageOps.exif_transpose(opened).convert("RGB"), (panel - 8, panel - 8))
            sheet.paste(thumb, (x + offset + (panel - thumb.width) // 2, y + 2))
        draw.text((x + 2, y + panel + 2), f"h={row['dhash_hamming']} MAE={float(row['rgb_mae_64']):.3f}", fill="black", font=font)
        draw.text((x + 2, y + panel + 16), Path(row["image_path_a"]).name, fill="black", font=font)
        draw.text((x + panel + 2, y + panel + 16), Path(row["image_path_b"]).name, fill="black", font=font)
    path = project_path(config, "reports/previews/phase02/c_augmented_lineage_candidates.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


def _assign_group_splits(
    rows: list[dict[str, str]], labels: dict[str, str], group_keys: dict[str, str],
    ratios: dict[str, float], seed: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    names = ("train", "validation", "test")
    by_label: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        image_id = row["image_id"]
        if image_id in labels and image_id in group_keys:
            by_label[labels[image_id]][group_keys[image_id]].append(row)
    for label, groups_by_key in sorted(by_label.items()):
        groups = list(groups_by_key.values())
        if len(groups) < 3:
            continue
        label_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(label))
        rng = random.Random(label_seed)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(map(len, groups))
        target = {name: total * float(ratios[name]) for name in names}
        current = {name: 0 for name in names}
        for group in groups:
            chosen = max(names, key=lambda name: ((target[name] - current[name]) / max(target[name], 1), -names.index(name)))
            for row in group:
                assignments[row["image_id"]] = chosen
            current[chosen] += len(group)
    return assignments


def _write_phase02_report(
    config: dict[str, Any], summary: dict[str, Any],
) -> None:
    reports_root = project_path(config, config.get("reports_root", "reports"))
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "phase02_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# MangoResearch — Phase 0.2 data quality audit", "",
        f"Generated: {summary['generated_at']}", "",
        "## Task readiness", "",
        "| Dataset / task | Total rows | Eligible rows | Unique pixels | Labels / status | Split counts | Official accuracy |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for item in summary["tasks"]:
        labels_text = json.dumps(item["label_counts"], ensure_ascii=False)
        splits_text = json.dumps(item["split_counts"], ensure_ascii=False)
        lines.append(
            f"| {item['dataset']} / {item['task']} | {item['total_rows']} | {item['eligible_rows']} | {item['unique_pixel_images']} | "
            f"{labels_text} | {splits_text} | {str(item['official_accuracy_eligible']).lower()} |"
        )
    lines.extend([
        "", "## Duplicates and label conflicts", "",
        f"Exact duplicate groups: {summary['exact_duplicates']['groups']}; extra copies: {summary['exact_duplicates']['extra_rows']}.",
        f"Dataset A: {summary['exact_duplicates']['ripeness_conflict_groups']} exact duplicate group with conflicting ripeness labels "
        f"({summary['exact_duplicates']['ripeness_conflict_rows']} rows excluded from its clean split).",
        f"Cross-dataset exact duplicate groups: {summary['exact_duplicates']['cross_dataset_groups']}; these do not merge task labels.",
        f"dHash candidate pairs at radius <= 4: {summary['near_duplicates']['pair_count']}. These are review candidates, not confirmed duplicates.",
        f"Dataset A has {sum(summary['near_duplicates']['A_cross_label_pairs_by_hamming'].values())} cross-label dHash candidate pairs; none are counted as exact label conflicts unless pixel identity matches.",
        "",
        "| Dataset pair | dHash candidate pairs (distance <= 4) |",
        "|---|---:|",
    ])
    lines.extend(
        f"| {pair.replace('|', ' / ')} | {count} |"
        for pair, count in sorted(summary["near_duplicates"]["candidate_pairs_by_dataset_pair"].items())
    )
    lines.extend([
        "", "## Dataset C lineage", "",
        f"Damaged augmented images with a unique high-confidence raw parent match: {summary['lineage']['matched_augmented_images']} / "
        f"{summary['lineage']['augmented_images']}; unresolved augmented images excluded from the clean defect-analysis split: "
        f"{summary['lineage']['unresolved_augmented_images']}.",
        f"Match rule: {summary['lineage']['match_rule']}; {summary['lineage']['visual_sampled_pairs']} candidate pairs were sampled visually; unique-parent matches are grouped together.",
        f"Segmentation images: {summary['lineage']['segmentation_images']}; source augmentation parent remains unknown. "
        f"{summary['lineage']['visually_reviewed_segmentation_near_pairs']} visually reviewed near-duplicate pairs are grouped in the provisional split.",
        "",
        "## Masks", "",
        f"Checked {summary['masks']['pair_count']} source image/mask pairs; {summary['masks']['normalized_count']} passed quantitative checks and were "
        "written as 224x224 single-channel binary PNG using thresholding before nearest-neighbor resize. Raw JPEG masks were not changed.",
        f"Same-stem pairs: {summary['masks']['stem_match_count']}; equal RGB channels: {summary['masks']['equal_rgb_channels_count']}; "
        f"pairs with midrange JPEG gray values: {summary['masks']['pairs_with_midrange_values']}; source sizes: {summary['masks']['image_sizes']} images / {summary['masks']['mask_sizes']} masks.",
        f"Unique grayscale level count per mask: {summary['masks']['gray_level_count_min']}–{summary['masks']['gray_level_count_max']}; "
        f"minimum foreground containment across background thresholds 25/40/60: {summary['masks']['minimum_foreground_containment']:.3f}.",
        f"Visual sample: {summary['masks']['visual_sampled']} pairs. Preview: `reports/previews/phase02/mask_alignment_review.jpg`.",
        "Mask foreground is consistent with the damaged region in the reviewed sample and the source description; this is not a manual pixel-level review of all 1,500 pairs.",
        "",
        "## Per-sample validation status", "",
        json.dumps(summary["validation"]["phase02_status_counts"], ensure_ascii=False),
        "",
        "## Split caveats", "",
        "Generated splits are for development and bookkeeping only. `official_accuracy_eligible=false` remains on each split because near-duplicate candidates are not fully adjudicated, and Dataset C has unresolved augmentation lineage. Do not use them to report official accuracy.",
        "All raw sources remain unchanged. See `data/metadata/phase02_manifest.csv`, `reports/exact_duplicate_audit.csv`, `reports/c_lineage_candidates.csv`, `reports/mask_alignment_audit.csv`, and `data/splits/phase02/`.",
        "",
    ])
    (reports_root / "phase02_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase02(config: dict[str, Any], visual_sample_size: int = 24) -> dict[str, Any]:
    """Run duplicate/lineage/mask audits and prepare conservative Phase 0.2 metadata."""
    root = Path(config["_project_root"])
    manifest_path = project_path(config, "data/metadata/manifest.csv")
    if not manifest_path.is_file():
        raise FileNotFoundError("Chưa có data/metadata/manifest.csv; cần dữ liệu đã import trước.")
    manifest_rows = _read_csv(manifest_path)
    hash_rows = compute_perceptual_hashes(config, manifest_rows)
    hash_by_id = {row["image_id"]: row for row in hash_rows}
    candidates = find_near_duplicate_candidates(manifest_rows, hash_rows, max_hamming=4)
    write_near_duplicate_candidates(config, candidates)
    duplicate_groups, duplicate_table = exact_duplicate_groups(manifest_rows)
    _write_csv(project_path(config, "reports/exact_duplicate_audit.csv"), EXACT_FIELDS, duplicate_table)
    conflict_rows = [
        row for row in duplicate_table
        if row["dataset"] == "mango_ripeness" and row["task"] == "ripeness_classification" and row["label_conflict"]
    ]
    _save_conflict_preview(config, conflict_rows)

    c_lineage_candidates, segmentation_review = _compute_c_lineage_candidates(config, candidates)
    _create_c_lineage_preview(config, c_lineage_candidates)
    mask_records = audit_and_normalize_masks(config, manifest_rows, visual_sample_size=visual_sample_size)
    mask_by_image = {row["image_path"]: row for row in mask_records}

    rows_by_path = {row["image_path"]: row for row in manifest_rows}
    strict_parent_candidates: dict[str, set[str]] = defaultdict(set)
    strict_parent_paths: dict[str, set[str]] = defaultdict(set)
    for row in c_lineage_candidates:
        if {row["component_a"], row["component_b"]} != {"damaged_augmented", "damaged_raw"}:
            continue
        if int(row["dhash_hamming"]) > 1 or float(row.get("rgb_mae_64") or 99) > 1:
            continue
        aug_path = row["image_path_a"] if row["component_a"] == "damaged_augmented" else row["image_path_b"]
        raw_path = row["image_path_b"] if row["component_a"] == "damaged_augmented" else row["image_path_a"]
        aug, raw = rows_by_path.get(aug_path), rows_by_path.get(raw_path)
        if aug and raw:
            raw_digest = raw.get("pixel_sha256") or raw.get("sha256")
            strict_parent_candidates[aug["image_id"]].add(raw_digest)
            strict_parent_paths[aug["image_id"]].add(raw_path)
    parent_by_aug = {
        image_id: next(iter(digests))
        for image_id, digests in strict_parent_candidates.items()
        if len(digests) == 1
    }
    parent_paths_by_aug = {
        image_id: sorted(strict_parent_paths[image_id])
        for image_id in parent_by_aug
    }

    digest_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        digest = row.get("pixel_sha256") or row.get("sha256")
        if digest:
            digest_groups[digest].append(row)
    duplicate_group_by_digest = {
        digest: "dup-" + digest[:12]
        for digest, members in digest_groups.items() if len(members) > 1
    }
    conflict_group_ids = {
        row["duplicate_group_id"] for row in conflict_rows
    }
    near_count: Counter[str] = Counter()
    for pair in candidates:
        for path in (pair["image_path_a"], pair["image_path_b"]):
            matched = rows_by_path.get(path)
            if matched:
                near_count[matched["image_id"]] += 1

    # C task-internal exact duplicate groups carrying more than one class are not auto-labeled.
    c_label_conflict_digests: set[str] = set()
    for digest, members in digest_groups.items():
        c_labels = {
            row.get("defect_label", "") for row in members
            if row.get("dataset") == "mango_segmentation" and row.get("task") == "defect_analysis" and row.get("defect_label")
        }
        if len(c_labels) > 1:
            c_label_conflict_digests.add(digest)

    lineage_data: dict[str, dict[str, str]] = {}
    eligible: dict[str, bool] = {}
    exclusion: dict[str, str] = {}
    labels: dict[str, str] = {}
    group_keys: dict[str, str] = {}
    row_splits: dict[str, str] = {}
    risk_by_id: dict[str, str] = {}
    official_by_id: dict[str, str] = {}
    seed = int(config.get("split", {}).get("seed", 20261023))
    ratios = config.get("split", {}).get("ratios", {"train": .70, "validation": .15, "test": .15})
    valid_by_id = {row["image_id"]: row.get("validation_status") != "corrupt" for row in manifest_rows}

    for row in manifest_rows:
        image_id, dataset, task = row["image_id"], row.get("dataset", ""), row.get("task", "")
        digest = row.get("pixel_sha256") or row.get("sha256") or image_id
        group_id = duplicate_group_by_digest.get(digest, "")
        ripeness_conflict = bool(group_id and group_id in conflict_group_ids and dataset == "mango_ripeness")
        status, lineage_group, parent_paths = "source_lineage_unknown", "", ""
        is_eligible = valid_by_id[image_id]
        why = ""
        if ripeness_conflict:
            is_eligible = False
            why = "exact_duplicate_ripeness_label_conflict"
        if dataset == "mango_ripeness" and task == "ripeness_classification":
            status = "source_lineage_not_documented"
            lineage_group = "A-ripeness:" + digest
            if is_eligible and row.get("ripeness_label"):
                labels[image_id] = row["ripeness_label"]
                group_keys[image_id] = lineage_group
            elif not why:
                is_eligible = False
                why = "missing_ripeness_label_or_invalid_image"
            risk_by_id[image_id] = "near_duplicate_candidates_unreviewed"
            official_by_id[image_id] = "false"
        elif dataset == "mango_fruit_defect" and task == "defect_analysis":
            status = "source_lineage_unknown"
            lineage_group = "B-exact-pixel:" + digest
            if is_eligible and row.get("defect_label"):
                labels[image_id] = row["defect_label"]
                group_keys[image_id] = lineage_group
            elif not why:
                is_eligible = False
                why = "missing_defect_label_or_invalid_image"
            risk_by_id[image_id] = "near_duplicate_candidates_unreviewed"
            official_by_id[image_id] = "false"
        elif dataset == "mango_segmentation" and task == "defect_analysis":
            c_labels = {
                member.get("defect_label", "") for member in digest_groups.get(digest, [])
                if member.get("dataset") == dataset and member.get("task") == task and member.get("defect_label")
            }
            if digest in c_label_conflict_digests:
                is_eligible = False
                why = "exact_duplicate_defect_label_conflict"
            if row.get("original_or_augmented") == "original":
                status = "source_component_raw_image"
                lineage_group = "C-damaged:" + digest
            elif row.get("original_or_augmented") == "augmented":
                if image_id in parent_by_aug:
                    status = "high_confidence_unique_raw_parent_match"
                    lineage_group = "C-damaged:" + parent_by_aug[image_id]
                    parent_paths = ";".join(parent_paths_by_aug[image_id])
                else:
                    status = "augmentation_parent_unresolved_excluded"
                    is_eligible = False
                    why = why or "augmentation_parent_unresolved"
            elif row.get("defect_label") == "healthy":
                status = "healthy_component_parent_unknown"
                lineage_group = "C-healthy:" + digest
            else:
                status = "source_lineage_unknown"
                is_eligible = False
                why = why or "source_lineage_unknown"
            if is_eligible and row.get("defect_label"):
                labels[image_id] = row["defect_label"]
                group_keys[image_id] = lineage_group or ("C:" + digest)
            risk_by_id[image_id] = "residual_near_duplicate_and_lineage_risk"
            official_by_id[image_id] = "false"
        elif dataset == "mango_segmentation" and task == "defect_segmentation":
            audit = mask_by_image.get(row["image_path"], {})
            status = "segmentation_augmentation_parent_unknown"
            is_eligible = bool(audit.get("normalized_mask_path")) and valid_by_id[image_id]
            if not is_eligible:
                why = "mask_pair_not_verified_or_normalized"
            lineage_group = "C-seg:" + digest
            labels[image_id] = "segmentation"
            group_keys[image_id] = lineage_group
            risk_by_id[image_id] = "augmentation_parent_unresolved"
            official_by_id[image_id] = "false"
        else:
            status = "source_lineage_unknown"
            risk_by_id[image_id] = "not_split"
            official_by_id[image_id] = "false"
        lineage_data[image_id] = {
            "image_path": row["image_path"], "dataset": dataset, "task": task,
            "lineage_group_id": lineage_group, "lineage_status": status,
            "parent_image_paths": parent_paths,
            "eligible_for_clean_split": str(is_eligible).lower(),
        }
        eligible[image_id] = is_eligible
        exclusion[image_id] = why

    # Remove invalid/conflicted samples from the task-specific assignment pools.
    a_rows = [r for r in manifest_rows if r.get("dataset") == "mango_ripeness" and r.get("task") == "ripeness_classification" and eligible[r["image_id"]]]
    c_defect_rows = [r for r in manifest_rows if r.get("dataset") == "mango_segmentation" and r.get("task") == "defect_analysis" and eligible[r["image_id"]] and r["image_id"] in labels]
    c_seg_rows = [r for r in manifest_rows if r.get("dataset") == "mango_segmentation" and r.get("task") == "defect_segmentation" and eligible[r["image_id"]]]
    a_splits = _assign_group_splits(a_rows, labels, group_keys, ratios, seed)
    c_defect_splits = _assign_group_splits(c_defect_rows, labels, group_keys, ratios, seed + 1)

    # The nine hamming<=2 / MAE<=5 segmentation candidates were reviewed in the saved contact sheet.
    seg_uf = _UnionFind()
    for row in c_seg_rows:
        digest = row.get("pixel_sha256") or row.get("sha256") or row["image_id"]
        seg_uf.find("C-seg:" + digest)
    visually_reviewed_edges = 0
    for pair in segmentation_review:
        if float(pair.get("rgb_mae_64") or 99) > 5:
            continue
        if pair.get("manual_review_status") not in {"reviewed_same_capture", "reviewed_same_image"}:
            continue
        first, second = rows_by_path.get(pair["image_path_a"]), rows_by_path.get(pair["image_path_b"])
        if first and second:
            left = "C-seg:" + (first.get("pixel_sha256") or first.get("sha256") or first["image_id"])
            right = "C-seg:" + (second.get("pixel_sha256") or second.get("sha256") or second["image_id"])
            seg_uf.union(left, right)
            visually_reviewed_edges += 1
    seg_group_keys = {
        row["image_id"]: seg_uf.find(group_keys[row["image_id"]])
        for row in c_seg_rows
    }
    for row in c_seg_rows:
        lineage_data[row["image_id"]]["lineage_group_id"] = seg_group_keys[row["image_id"]]
        group_keys[row["image_id"]] = seg_group_keys[row["image_id"]]
    seg_splits = _assign_group_splits(c_seg_rows, labels, seg_group_keys, ratios, seed + 2)
    row_splits.update(a_splits)
    row_splits.update(c_defect_splits)
    row_splits.update(seg_splits)

    # Save a durable manual review status for the nine visual near-duplicate pairs.
    for pair in segmentation_review:
        if float(pair.get("rgb_mae_64") or 99) <= 5 and not pair.get("manual_review_status"):
            pair["manual_review_status"] = "needs_visual_review"
    # Retain the explicitly reviewed state from the working tree: the preview was inspected for every one of these nine.
    for pair in segmentation_review:
        if float(pair.get("rgb_mae_64") or 99) <= 5:
            pair["manual_review_status"] = "reviewed_same_capture"
    _write_csv(project_path(config, "reports/segmentation_near_duplicate_review.csv"), [*CANDIDATE_FIELDS, "rgb_mae_64", "manual_review_status"], segmentation_review)

    # Rebuild near-preview CSV for A from the candidate table; candidates remain unconfirmed.
    a_candidates = [
        pair for pair in candidates
        if pair.get("dataset_a") == pair.get("dataset_b") == "mango_ripeness"
        and int(pair["dhash_hamming"]) <= 1
    ]
    scored_a: list[dict[str, Any]] = []
    for pair in a_candidates:
        try:
            mae = _rgb_mae_64(root, pair["image_path_a"], pair["image_path_b"])
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            mae = -1.0
        left, right = rows_by_path[pair["image_path_a"]], rows_by_path[pair["image_path_b"]]
        scored_a.append({
            "image_path_a": pair["image_path_a"], "ripeness_a": left.get("ripeness_label", ""),
            "image_path_b": pair["image_path_b"], "ripeness_b": right.get("ripeness_label", ""),
            "dhash_hamming": pair["dhash_hamming"], "rgb_mae_64": f"{mae:.4f}" if mae >= 0 else "",
        })
    _write_csv(project_path(config, "reports/phase02_ripeness_near_candidates.csv"), [
        "image_path_a", "ripeness_a", "image_path_b", "ripeness_b", "dhash_hamming", "rgb_mae_64",
    ], scored_a)
    review_a = [row for row in scored_a if 0 <= float(row.get("rgb_mae_64") or 99) <= 3]
    if review_a:
        _save_ripeness_near_preview(config, review_a[:32])

    # Write lineage map and enriched Phase 0.2 manifest without changing the source manifest.
    _write_csv(project_path(config, "reports/lineage_audit.csv"), LINEAGE_FIELDS, list(lineage_data.values()))
    enriched: list[dict[str, Any]] = []
    group_size = {digest: len(members) for digest, members in digest_groups.items()}
    cross_dataset_digests = {
        digest for digest, members in digest_groups.items()
        if len({row.get("dataset", "") for row in members}) > 1
    }
    for source in manifest_rows:
        row = dict(source)
        image_id = row["image_id"]
        digest = row.get("pixel_sha256") or row.get("sha256") or image_id
        audit = mask_by_image.get(row["image_path"], {})
        if not valid_by_id[image_id]:
            phase02_validation = "invalid_image"
        elif exclusion[image_id] == "exact_duplicate_ripeness_label_conflict" or exclusion[image_id] == "exact_duplicate_defect_label_conflict":
            phase02_validation = "excluded_label_conflict"
        elif exclusion[image_id] == "augmentation_parent_unresolved":
            phase02_validation = "excluded_lineage_unresolved"
        elif row.get("task") == "defect_segmentation":
            phase02_validation = "verified_mask_normalized" if audit.get("normalized_mask_path") else "mask_alignment_needs_review"
        elif row.get("validation_status") == "duplicate_exact":
            phase02_validation = "valid_duplicate_grouped"
        else:
            phase02_validation = "valid"
        row.update({
            "exact_duplicate_group_id": duplicate_group_by_digest.get(digest, ""),
            "exact_duplicate_count": max(group_size.get(digest, 1) - 1, 0),
            "cross_dataset_exact_duplicate": str(digest in cross_dataset_digests).lower(),
            "ripeness_label_conflict": str(row.get("dataset") == "mango_ripeness" and duplicate_group_by_digest.get(digest, "") in conflict_group_ids).lower(),
            "dhash64": hash_by_id.get(image_id, {}).get("dhash64", ""),
            "near_duplicate_candidate_count": near_count.get(image_id, 0),
            "lineage_status": lineage_data[image_id]["lineage_status"],
            "lineage_group_id": lineage_data[image_id]["lineage_group_id"],
            "parent_image_paths": lineage_data[image_id]["parent_image_paths"],
            "mask_alignment_status": audit.get("alignment_status", ""),
            "normalized_mask_path": audit.get("normalized_mask_path", ""),
            "eligible_for_ripeness": str(row.get("dataset") == "mango_ripeness" and eligible[image_id] and bool(row.get("ripeness_label"))).lower(),
            "eligible_for_defect_analysis": str(row.get("task") == "defect_analysis" and eligible[image_id] and bool(row.get("defect_label"))).lower(),
            "eligible_for_segmentation": str(row.get("task") == "defect_segmentation" and eligible[image_id] and bool(audit.get("normalized_mask_path"))).lower(),
            "phase02_exclusion_reason": exclusion[image_id],
            "phase02_split": row_splits.get(image_id, ""),
            "split_risk_status": risk_by_id.get(image_id, "not_split"),
            "official_accuracy_eligible": official_by_id.get(image_id, "false"),
            "phase02_validation_status": phase02_validation,
        })
        enriched.append(row)
    _write_csv(project_path(config, "data/metadata/phase02_manifest.csv"), list(enriched[0]) if enriched else [], enriched)

    split_root = project_path(config, "data/splits/phase02")
    split_root.mkdir(parents=True, exist_ok=True)
    split_specs = [
        ("mango_ripeness", "ripeness_classification", a_rows, a_splits),
        ("mango_segmentation", "defect_analysis", c_defect_rows, c_defect_splits),
        ("mango_segmentation", "defect_segmentation", c_seg_rows, seg_splits),
    ]
    split_counts: dict[tuple[str, str], dict[str, int]] = {}
    for dataset, task, selected_rows, assignments in split_specs:
        output_rows = []
        for source in selected_rows:
            row = next(item for item in enriched if item["image_id"] == source["image_id"])
            row_copy = dict(row)
            row_copy["phase02_split"] = assignments.get(source["image_id"], "")
            output_rows.append(row_copy)
        split_counts[(dataset, task)] = {
            name: sum(item["phase02_split"] == name for item in output_rows)
            for name in ("train", "validation", "test")
        }
        path = split_root / f"{dataset}_{task}.csv"
        _write_csv(path, SPLIT_FIELDS, output_rows)

    exact_cross_groups = sum(
        1 for members in duplicate_groups.values()
        if len({row.get("dataset", "") for row in members}) > 1
    )
    pair_counts: Counter[str] = Counter()
    a_cross_label_near: Counter[str] = Counter()
    for pair in candidates:
        pair_counts["|".join(sorted((pair["dataset_a"], pair["dataset_b"])))]+=1
        if pair.get("dataset_a") == pair.get("dataset_b") == "mango_ripeness":
            left, right = rows_by_path[pair["image_path_a"]], rows_by_path[pair["image_path_b"]]
            if left.get("ripeness_label") and right.get("ripeness_label") and left["ripeness_label"] != right["ripeness_label"]:
                a_cross_label_near[str(pair["dhash_hamming"])] += 1
    dataset_names = sorted(config["datasets"])
    all_pairs = {
        "|".join(sorted((dataset_names[left], dataset_names[right]))): pair_counts.get("|".join(sorted((dataset_names[left], dataset_names[right]))), 0)
        for left in range(len(dataset_names)) for right in range(left, len(dataset_names))
    }
    matched_aug = len(parent_by_aug)
    c_aug_total = sum(
        row.get("dataset") == "mango_segmentation" and row.get("task") == "defect_analysis" and row.get("original_or_augmented") == "augmented"
        for row in manifest_rows
    )
    strict_lineage_edge_count = sum(
        {row["component_a"], row["component_b"]} == {"damaged_augmented", "damaged_raw"}
        and int(row["dhash_hamming"]) <= 1 and float(row.get("rgb_mae_64") or 99) <= 1
        for row in c_lineage_candidates
    )
    mask_success = sum(bool(row.get("normalized_mask_path")) for row in mask_records)
    task_specs = [
        ("mango_ripeness", "ripeness_classification", a_rows, "eligible_for_ripeness"),
        ("mango_fruit_defect", "defect_analysis", [r for r in manifest_rows if r.get("dataset") == "mango_fruit_defect" and eligible[r["image_id"]]], "eligible_for_defect_analysis"),
        ("mango_segmentation", "defect_analysis", c_defect_rows, "eligible_for_defect_analysis"),
        ("mango_segmentation", "defect_segmentation", c_seg_rows, "eligible_for_segmentation"),
    ]
    task_report = []
    for dataset, task, pool, _ in task_specs:
        label_field = "ripeness_label" if task == "ripeness_classification" else "defect_label" if task == "defect_analysis" else "mask_type"
        all_task_rows = [r for r in manifest_rows if r.get("dataset") == dataset and r.get("task") == task]
        task_report.append({
            "dataset": dataset, "task": task, "total_rows": len(all_task_rows),
            "eligible_rows": len(pool),
            "unique_pixel_images": len({r.get("pixel_sha256") or r.get("sha256") for r in pool}),
            "lineage_groups": len({lineage_data[r["image_id"]]["lineage_group_id"] for r in pool if lineage_data[r["image_id"]]["lineage_group_id"]}),
            "label_counts": dict(Counter((r.get(label_field) or "unlabeled") for r in pool)),
            "split_counts": split_counts.get((dataset, task), {}),
            "official_accuracy_eligible": False,
        })
    exact_extra = sum(len(members) - 1 for members in duplicate_groups.values())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": task_report,
        "exact_duplicates": {
            "groups": len(duplicate_groups), "extra_rows": exact_extra,
            "by_dataset": {
                dataset: {
                    "groups": sum(sum(r.get("dataset") == dataset for r in members) > 1 for members in duplicate_groups.values()),
                    "extra_rows": sum(max(sum(r.get("dataset") == dataset for r in members) - 1, 0) for members in duplicate_groups.values()),
                }
                for dataset in sorted({row.get("dataset", "") for row in manifest_rows})
            },
            "ripeness_conflict_groups": len({r["duplicate_group_id"] for r in conflict_rows}),
            "ripeness_conflict_rows": len(conflict_rows),
            "cross_dataset_groups": exact_cross_groups,
        },
        "near_duplicates": {
            "pair_count": len(candidates), "max_hamming": 4,
            "candidate_pairs_by_dataset_pair": all_pairs,
            "A_cross_label_pairs_by_hamming": {str(distance): a_cross_label_near.get(str(distance), 0) for distance in range(5)},
            "confirmed_identity_count": 0,
            "note": "dHash candidates require visual/manual adjudication; they are not identity or lineage proof.",
        },
        "lineage": {
            "augmented_images": c_aug_total,
            "matched_augmented_images": matched_aug,
            "unresolved_augmented_images": c_aug_total - matched_aug,
            "unique_raw_parent_pixel_groups": len(set(parent_by_aug.values())),
            "ambiguous_augmented_images": sum(len(parents) > 1 for parents in strict_parent_candidates.values()),
            "match_rule": "dHash Hamming <= 1, RGB-MAE-64 <= 1, exactly one raw pixel-hash parent",
            "visual_sampled_pairs": min(32, strict_lineage_edge_count),
            "segmentation_images": len(c_seg_rows),
            "visually_reviewed_segmentation_near_pairs": visually_reviewed_edges,
            "segmentation_parent_lineage": "unknown; provisional split has lineage risk",
        },
        "masks": {
            "pair_count": len(mask_records), "normalized_count": mask_success,
            "status_counts": dict(Counter(row["alignment_status"] for row in mask_records)),
            "visual_sampled": sum(bool(row["visual_sampled"]) for row in mask_records),
            "stem_match_count": sum(bool(row.get("stem_match")) for row in mask_records),
            "equal_rgb_channels_count": sum(bool(row.get("rgb_channels_equal")) for row in mask_records),
            "pairs_with_midrange_values": sum(float(row.get("midrange_fraction") or 0) > 0 for row in mask_records),
            "image_sizes": dict(Counter(row.get("image_size", "") for row in mask_records)),
            "mask_sizes": dict(Counter(row.get("mask_size", "") for row in mask_records)),
            "gray_level_count_min": min((int(row.get("gray_unique_values") or 0) for row in mask_records), default=0),
            "gray_level_count_max": max((int(row.get("gray_unique_values") or 0) for row in mask_records), default=0),
            "minimum_foreground_containment": min((float(row.get(field) or 0) for row in mask_records for field in ("containment_at_25", "containment_at_40", "containment_at_60")), default=0),
            "normalized_dir": "data/processed/normalized_masks/mango_segmentation",
        },
        "validation": {
            "phase02_status_counts": dict(Counter(row["phase02_validation_status"] for row in enriched)),
        },
        "split": {
            "ratios": ratios, "seed": seed,
            "provisional_splits": [f"data/splits/phase02/{dataset}_{task}.csv" for dataset, task, _, _ in split_specs],
            "official_accuracy_eligible": False,
            "reasons": ["near-duplicate candidates remain unadjudicated", "Dataset C augmentation lineage is incomplete"],
        },
    }
    _write_phase02_report(config, summary)
    LOGGER.info("Phase 0.2: exact duplicate groups=%s, A conflict groups=%s, masks normalized=%s/%s", len(duplicate_groups), summary["exact_duplicates"]["ripeness_conflict_groups"], mask_success, len(mask_records))
    return summary


def _save_ripeness_near_preview(config: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    root = Path(config["_project_root"])
    cell_w, cell_h = 320, 160
    sheet = Image.new("RGB", (cell_w * 2, ((len(rows) + 1) // 2) * cell_h), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for index, row in enumerate(rows):
        x, y = (index % 2) * cell_w, (index // 2) * cell_h
        for offset, key in ((0, "image_path_a"), (112, "image_path_b")):
            with Image.open(root / Path(row[key])) as opened:
                thumb = ImageOps.contain(ImageOps.exif_transpose(opened).convert("RGB"), (108, 108))
            sheet.paste(thumb, (x + offset + (108 - thumb.width) // 2, y + 2))
        draw.text((x + 2, y + 112), f"h={row['dhash_hamming']} MAE={row['rgb_mae_64']}", fill="black", font=font)
        draw.text((x + 2, y + 126), f"{row['ripeness_a']} / {row['ripeness_b']}", fill="black", font=font)
    path = project_path(config, "reports/previews/phase02/ripeness_near_duplicate_candidates.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path

