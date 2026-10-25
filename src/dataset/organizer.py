from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any

from src.utils.config import project_path
from src.utils.files import slugify, sha256_file


LOGGER = logging.getLogger(__name__)


def organize_images(config: dict[str, Any]) -> dict[str, int]:
    manifest = project_path(config, "data/metadata/manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError("Chưa có manifest.csv; chạy inspect trước.")
    project_root = Path(config["_project_root"])
    destination_root = project_path(config, "data/processed/organized")
    counts: dict[str, int] = {}
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        image = project_root / Path(row["image_path"])
        if not image.is_file() or row.get("validation_status") == "corrupt":
            continue
        label = row.get("ripeness_label", "").strip() or row.get("defect_label", "").strip() or "unlabeled"
        label_dir = slugify(label)
        target_dir = destination_root / slugify(row.get("dataset", "dataset")) / slugify(row.get("task", "task")) / label_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{row['image_id']}_{image.name}"
        if target.exists():
            if sha256_file(target) == sha256_file(image):
                continue
            raise FileExistsError(f"Không ghi đè file khác nội dung: {target}")
        shutil.copy2(image, target)
        counts[label_dir] = counts.get(label_dir, 0) + 1
    LOGGER.info("Đã tổ chức/copy ảnh: %s", sum(counts.values()))
    return counts
