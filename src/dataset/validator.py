from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.dataset.inspector import METADATA_FIELDS
from src.utils.config import project_path


LOGGER = logging.getLogger(__name__)


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = project_path(config, "data/metadata/manifest.csv")
    report_path = project_path(config, "reports/validation.csv")
    project_root = Path(config["_project_root"])
    issues: list[dict[str, str]] = []
    if not metadata_path.exists():
        raise FileNotFoundError("Chưa có manifest.csv; chạy inspect trước.")
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    for required in METADATA_FIELDS:
        if required not in fields:
            issues.append({"image_path": "", "issue": f"Thiếu cột metadata: {required}"})
    digest_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image = project_root / Path(row.get("image_path", ""))
        if not image.is_file():
            issues.append({"image_path": row.get("image_path", ""), "issue": "image_path_missing"})
            continue
        try:
            with Image.open(image) as im:
                im.verify()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues.append({"image_path": row.get("image_path", ""), "issue": f"image_unreadable: {exc}"})
        mask_path = row.get("mask_path", "").strip()
        if mask_path:
            mask = project_root / Path(mask_path)
            if not mask.is_file():
                issues.append({"image_path": row.get("image_path", ""), "issue": "mask_path_missing"})
            else:
                try:
                    with Image.open(image) as im, Image.open(mask) as mask_im:
                        if im.size != mask_im.size:
                            issues.append({"image_path": row.get("image_path", ""), "issue": "image_mask_size_mismatch"})
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    issues.append({"image_path": row.get("image_path", ""), "issue": f"mask_unreadable: {exc}"})
        digest = row.get("pixel_sha256") or row.get("sha256")
        split = row.get("split", "").strip()
        if digest and split:
            digest_splits[digest].add(split)
    for digest, splits in digest_splits.items():
        if len(splits) > 1:
            issues.append({"image_path": digest, "issue": f"duplicate_across_splits: {','.join(sorted(splits))}"})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "issue"])
        writer.writeheader()
        writer.writerows(issues)
    result = {"rows": len(rows), "issue_count": len(issues), "issues": issues}
    LOGGER.info("Validation: %s rows, %s issue(s)", len(rows), len(issues))
    return result
