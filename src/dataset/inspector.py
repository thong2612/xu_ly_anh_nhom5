from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

from src.utils.config import project_path
from src.utils.files import sha256_file


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".pnm", ".ico",
}
METADATA_FIELDS = [
    "image_id", "dataset", "image_path", "task", "ripeness_label", "defect_label",
    "mask_path", "mask_type", "mask_review_status", "split", "split_origin", "original_or_augmented", "validation_status",
    "validation_details", "sha256", "pixel_sha256", "file_format", "width", "height", "file_size_bytes",
]
OVERRIDE_FIELDS = ["image_path", "ripeness_label", "defect_label", "mask_type", "mask_review_status", "original_or_augmented"]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _read_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=OVERRIDE_FIELDS)
            writer.writeheader()
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return {
            row.get("image_path", "").replace("\\", "/").casefold(): row
            for row in csv.DictReader(stream)
            if row.get("image_path", "").strip()
        }


def _is_mask(path: Path) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.casefold()))
    if "mask" in tokens or "masks" in tokens:
        return True
    for parent in path.parents:
        if _norm(parent.name) in {"source", "downloads"}:
            break
        if set(_norm(parent.name).split()) & {"mask", "masks"}:
            return True
    return False


def _mask_key(path: Path) -> str:
    name = path.stem.casefold()
    name = re.sub(r"(^mask[_ -]?|[_ -]?mask$)", "", name)
    return _norm(name)


def _mask_scope(path: Path) -> Path:
    """Return the component root that contains a dedicated mask directory."""
    for parent in path.parents:
        if _norm(parent.name) in {"source", "downloads"}:
            break
        if set(_norm(parent.name).split()) & {"mask", "masks"}:
            return parent.parent
    # Filename-marked masks usually sit beside their image, so their parent is
    # the narrowest reliable scope available.
    return path.parent


def _lineage(path: Path, override: str = "") -> str:
    value = override.strip().casefold()
    if value in {"original", "augmented", "unknown"}:
        return value
    components = list(path.parts)
    boundary_positions = [index for index, part in enumerate(components) if _norm(part) in {"source", "downloads"}]
    if boundary_positions:
        components = components[boundary_positions[-1] + 1:]
    parts = [set(_norm(part).split()) for part in components]
    if any(tokens & {"aug", "augmented", "augment", "augmentation"} for tokens in parts):
        return "augmented"
    if any(tokens & {"original", "raw"} for tokens in parts):
        return "original"
    return "unknown"


def _split_from_path(path: Path) -> tuple[str, str]:
    components = list(path.parts)
    boundary_positions = [index for index, part in enumerate(components) if _norm(part) in {"source", "downloads"}]
    if boundary_positions:
        components = components[boundary_positions[-1] + 1:]
    found: set[str] = set()
    aliases = {
        "train": "train", "training": "train",
        "val": "validation", "valid": "validation", "validation": "validation",
        "test": "test", "testing": "test",
    }
    for part in components[:-1]:
        found.update(aliases[token] for token in _norm(part).split() if token in aliases)
    if len(found) == 1:
        return found.pop(), "published"
    if len(found) > 1:
        return "", "ambiguous"
    return "", ""


def _labels_from_path(path: Path, definition: dict[str, Any]) -> tuple[str, str]:
    components = {_norm(part) for part in path.parent.parts}
    ripeness = ""
    defect = ""
    for label, aliases in definition.get("ripeness_labels", {}).items():
        if any(_norm(str(alias)) in components for alias in aliases):
            ripeness = label
            break
    for label, aliases in definition.get("defect_labels", {}).items():
        if any(_norm(str(alias)) in components for alias in aliases):
            defect = label
            break
    return ripeness, defect


def _component_metadata(path: Path, definition: dict[str, Any]) -> dict[str, str]:
    """Apply only source-documented component rules matched to exact path components."""
    components = {_norm(part) for part in path.parts}
    metadata: dict[str, str] = {}
    for rule in definition.get("component_rules", []):
        component = _norm(str(rule.get("path_component", "")))
        if component and component in components:
            for field in ("task", "ripeness_label", "defect_label", "original_or_augmented"):
                value = str(rule.get(field, "")).strip()
                if value:
                    metadata[field] = value
    return metadata


def _override_label(value: str, inferred: str) -> str:
    value = value.strip()
    if value.casefold() == "__clear__":
        return ""
    return value or inferred


def _image_record(path: Path, project_root: Path, dataset_key: str, definition: dict[str, Any], override: dict[str, str]) -> dict[str, Any]:
    relative = path.relative_to(project_root).as_posix()
    record: dict[str, Any] = {
        field: "" for field in METADATA_FIELDS
    }
    source_split, split_origin = _split_from_path(path)
    record.update({
        "dataset": dataset_key,
        "image_path": relative,
        "task": definition["task"],
        "split": source_split,
        "split_origin": split_origin,
    })
    record["image_id"] = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
    component = _component_metadata(path, definition)
    record["task"] = component.get("task", record["task"])
    ripeness, defect = _labels_from_path(path, definition)
    record["ripeness_label"] = _override_label(
        override.get("ripeness_label", ""), component.get("ripeness_label", ripeness)
    )
    record["defect_label"] = _override_label(
        override.get("defect_label", ""), component.get("defect_label", defect)
    )
    record["mask_type"] = override.get("mask_type", "").strip()
    record["mask_review_status"] = override.get("mask_review_status", "").strip().casefold()
    record["original_or_augmented"] = _lineage(
        path, override.get("original_or_augmented", "") or component.get("original_or_augmented", "")
    )
    record["file_size_bytes"] = path.stat().st_size
    record["sha256"] = sha256_file(path)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record["width"], record["height"] = image.size
            record["file_format"] = (image.format or path.suffix.lstrip(".")).lower()
            with image.convert("RGB") as rgb:
                pixel_hash = hashlib.sha256()
                pixel_hash.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
                pixel_hash.update(rgb.tobytes())
                record["pixel_sha256"] = pixel_hash.hexdigest()
        record["validation_status"] = "valid"
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        record["validation_status"] = "corrupt"
        record["validation_details"] = str(exc)
        LOGGER.warning("Ảnh lỗi: %s (%s)", relative, exc)
    if split_origin == "ambiguous" and record["validation_status"] == "valid":
        record["validation_status"] = "split_metadata_ambiguous"
        record["validation_details"] = "Đường dẫn chứa nhiều dấu hiệu split chính thức xung đột"
    return record


def _mask_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": path}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            info["size"] = image.size
            info["mode"] = image.mode
            colors = image.getcolors(maxcolors=257)
            info["unique_values_or_colors"] = len(colors) if colors is not None else ">256"
            info["validation"] = "valid"
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        info["validation"] = f"corrupt: {exc}"
        info["size"] = None
        info["mode"] = ""
        info["unique_values_or_colors"] = ""
    return info


def _write_contact_sheet(records: list[dict[str, Any]], output: Path, project_root: Path, title: str) -> None:
    if not records:
        return
    cell_w, cell_h = 240, 220
    columns = 4
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + 38), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, record in enumerate(records):
        x = (index % columns) * cell_w
        y = 38 + (index // columns) * cell_h
        path = project_root / Path(record["image_path"])
        try:
            with Image.open(path) as image:
                image.thumbnail((cell_w - 12, 165))
                sheet.paste(image.convert("RGB"), (x + (cell_w - image.width) // 2, y))
        except (OSError, ValueError):
            draw.rectangle((x + 8, y, x + cell_w - 8, y + 160), outline="red")
            draw.text((x + 14, y + 16), "Ảnh lỗi", fill="red")
        caption = f"{record.get('ripeness_label') or record.get('defect_label') or 'unlabeled'}\n{Path(record['image_path']).name[:34]}"
        draw.text((x + 6, y + 170), caption, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _write_mask_preview(pairs: list[tuple[dict[str, Any], dict[str, Any]]], output: Path, project_root: Path) -> None:
    if not pairs:
        return
    cell_w, cell_h, columns = 460, 230, 2
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Ảnh / mask ghép theo tên và component; trạng thái review lấy từ manifest", fill="black")
    for index, (record, mask) in enumerate(pairs):
        x = (index % columns) * cell_w
        y = 34 + (index // columns) * cell_h
        for offset, path in ((0, project_root / Path(record["image_path"])), (cell_w // 2, mask["path"])):
            try:
                with Image.open(path) as image:
                    image.thumbnail((cell_w // 2 - 12, 185))
                    sheet.paste(image.convert("RGB"), (x + offset + 6, y))
            except (OSError, ValueError):
                draw.text((x + offset + 8, y + 16), "Không đọc được", fill="red")
        draw.text((x + 6, y + 190), f"Ảnh | {record.get('width')}x{record.get('height')}", fill="black")
        mask_type = record.get("mask_type") or "type pending"
        review = record.get("mask_review_status") or "review pending"
        draw.text(
            (x + cell_w // 2 + 6, y + 190),
            f"MASK | {mask['size']} {mask['mode']} | {mask_type} / {review}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def inspect_datasets(config: dict[str, Any], dataset_keys: list[str] | None = None, sample_limit: int = 16) -> list[dict[str, Any]]:
    keys = dataset_keys or list(config["datasets"])
    project_root = Path(config["_project_root"])
    metadata_root = project_path(config, config.get("metadata_root", "data/metadata"))
    metadata_root.mkdir(parents=True, exist_ok=True)
    overrides = _read_overrides(metadata_root / "labels_overrides.csv")
    all_records: list[dict[str, Any]] = []
    all_mask_info: list[dict[str, Any]] = []
    all_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    report_root = project_path(config, config.get("reports_root", "reports"))

    for key in keys:
        definition = config["datasets"][key]
        roots = [
            project_path(config, f"data/raw/{key}/downloads"),
            project_path(config, f"data/raw/{key}/source"),
        ]
        files = sorted({
            path
            for root in roots if root.exists()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        })
        mask_paths = [path for path in files if _is_mask(path)]
        image_paths = [path for path in files if path not in mask_paths]
        masks_by_scope_key: dict[tuple[Path, str], list[Path]] = defaultdict(list)
        mask_infos: dict[Path, dict[str, Any]] = {}
        for mask_path in mask_paths:
            masks_by_scope_key[(_mask_scope(mask_path), _mask_key(mask_path))].append(mask_path)
            mask_infos[mask_path] = _mask_info(mask_path)
            all_mask_info.append({"dataset": key, **mask_infos[mask_path]})

        records = []
        for path in image_paths:
            relative = path.relative_to(project_root).as_posix()
            record = _image_record(path, project_root, key, definition, overrides.get(relative.casefold(), {}))
            if record["validation_status"] == "valid":
                mask_key = _mask_key(path)
                matches = [
                    candidate
                    for scope in path.parents
                    for candidate in masks_by_scope_key.get((scope, mask_key), [])
                ]
                valid_matches = [candidate for candidate in matches if mask_infos[candidate]["validation"] == "valid"]
                if len(valid_matches) == 1:
                    mask_path = valid_matches[0]
                    info = mask_infos[mask_path]
                    record["mask_path"] = mask_path.relative_to(project_root).as_posix()
                    if tuple(info.get("size") or ()) != (record["width"], record["height"]):
                        record["validation_status"] = "mask_size_mismatch"
                        record["validation_details"] = "Kích thước ảnh/mask không khớp"
                    elif record["mask_review_status"] == "rejected":
                        record["validation_status"] = "mask_rejected"
                        record["validation_details"] = "Cặp ảnh/mask bị đánh dấu không hợp lệ khi rà soát"
                    elif not record["mask_type"] or record["mask_review_status"] != "verified":
                        record["validation_status"] = "mask_review_required"
                        record["mask_review_status"] = record["mask_review_status"] or "pending"
                        record["validation_details"] = "Cần xác nhận mask_type và mask_review_status=verified sau khi xem preview"
                    all_pairs.append((record, info))
                elif len(matches) > 1:
                    record["validation_status"] = "mask_pair_ambiguous"
                    record["validation_details"] = "Nhiều mask trùng khóa tên; chưa ghép"
            records.append(record)

        # Exact byte duplicates stay visible in the metadata and are never
        # discarded. Later splitting groups by digest.
        digest_counts = Counter(row["sha256"] for row in records if row["sha256"])
        for record in records:
            if record["validation_status"] == "valid" and digest_counts[record["sha256"]] > 1:
                record["validation_status"] = "duplicate_exact"

        all_records.extend(records)
        label_counts = Counter(
            row["ripeness_label"] or row["defect_label"] or "unlabeled"
            for row in records if row["validation_status"] != "corrupt"
        )
        format_counts = Counter(row["file_format"] or "unknown" for row in records if row["validation_status"] != "corrupt")
        size_counts = Counter(f"{row['width']}x{row['height']}" for row in records if row["width"] and row["height"])
        summaries.append({
            "dataset": key,
            "title": definition["title"],
            "task": definition["task"],
            "image_count": len(image_paths),
            "valid_image_count": sum(row["validation_status"] != "corrupt" for row in records),
            "corrupt_image_count": sum(row["validation_status"] == "corrupt" for row in records),
            "mask_count": len(mask_paths),
            "paired_mask_count": sum(bool(row["mask_path"]) for row in records),
            "mask_review_required_count": sum(row["validation_status"] == "mask_review_required" for row in records),
            "duplicate_exact_extra_count": sum(max(0, n - 1) for n in digest_counts.values()),
            "label_counts": dict(label_counts),
            "format_counts": dict(format_counts),
            "resolution_counts": dict(size_counts),
            "expected_images": definition.get("expected_images"),
        })
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            label = record["ripeness_label"] or record["defect_label"] or "unlabeled"
            groups[label].append(record)
        for label, group_records in groups.items():
            preview_records = sorted(group_records, key=lambda row: row["image_path"])[:sample_limit]
            _write_contact_sheet(preview_records, report_root / "previews" / key / f"{label.replace(' ', '_')}.jpg", project_root, f"{key} / {label}")

    metadata_path = metadata_root / "manifest.csv"
    if metadata_path.exists():
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
            old_records = list(csv.DictReader(stream))
        all_records = [row for row in old_records if row.get("dataset") not in keys] + all_records
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # Keep the collected preview limited for large collections.
    _write_mask_preview(all_pairs[:max(sample_limit, 8)], report_root / "previews" / "image_mask_pairs.jpg", project_root)
    mask_report_path = report_root / "mask_inspection.csv"
    if mask_report_path.exists():
        with mask_report_path.open("r", newline="", encoding="utf-8-sig") as stream:
            old_masks = [row for row in csv.DictReader(stream) if row.get("dataset") not in keys]
        all_mask_info = old_masks + all_mask_info
    mask_report_path.parent.mkdir(parents=True, exist_ok=True)
    with mask_report_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["dataset", "path", "size", "mode", "unique_values_or_colors", "validation"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_mask_info:
            mask_path = row["path"]
            relative_mask = mask_path.relative_to(project_root).as_posix() if isinstance(mask_path, Path) else str(mask_path)
            writer.writerow({**row, "path": relative_mask, "size": row.get("size")})
    return summaries
