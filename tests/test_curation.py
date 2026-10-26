from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from src.dataset.curation import (
    _assign_group_splits,
    audit_and_normalize_masks,
    exact_duplicate_groups,
    find_near_duplicate_candidates,
)


def test_exact_duplicate_audit_marks_ripeness_label_conflict() -> None:
    rows = [
        {"image_id": "ripe", "dataset": "mango_ripeness", "task": "ripeness_classification", "image_path": "ripe/a.jpg", "ripeness_label": "ripe", "pixel_sha256": "same"},
        {"image_id": "unripe", "dataset": "mango_ripeness", "task": "ripeness_classification", "image_path": "unripe/a.jpg", "ripeness_label": "unripe", "pixel_sha256": "same"},
    ]
    groups, table = exact_duplicate_groups(rows)
    assert len(groups) == 1
    assert len(table) == 2
    assert {row["label_conflict"] for row in table} == {True}


def test_perceptual_hash_returns_candidates_without_claiming_identity() -> None:
    rows = [
        {"image_id": "a", "dataset": "A", "task": "ripeness_classification", "image_path": "a.jpg", "pixel_sha256": "pa"},
        {"image_id": "b", "dataset": "A", "task": "ripeness_classification", "image_path": "b.jpg", "pixel_sha256": "pb"},
    ]
    hashes = [
        {"image_id": "a", "dhash64": "0000000000000000", "status": "valid"},
        {"image_id": "b", "dhash64": "0000000000000001", "status": "valid"},
    ]
    candidates = find_near_duplicate_candidates(rows, hashes, max_hamming=1)
    assert len(candidates) == 1
    assert candidates[0]["review_status"] == "candidate_not_identity"


def test_group_split_keeps_linked_images_together_and_is_reproducible() -> None:
    rows = [{"image_id": str(index)} for index in range(30)]
    labels = {str(index): "ripe" for index in range(30)}
    groups = {str(index): f"parent-{index // 2}" for index in range(30)}
    first = _assign_group_splits(rows, labels, groups, {"train": .7, "validation": .15, "test": .15}, 17)
    second = _assign_group_splits(rows, labels, groups, {"train": .7, "validation": .15, "test": .15}, 17)
    assert first == second
    for index in range(0, 30, 2):
        assert first[str(index)] == first[str(index + 1)]
    assert set(first.values()) == {"train", "validation", "test"}


def test_mask_audit_writes_binary_lossless_nearest_neighbor_output(tmp_path: Path) -> None:
    image_rel = "data/raw/mango_segmentation/images/1.png"
    mask_rel = "data/raw/mango_segmentation/masks/1.png"
    image_path = tmp_path / image_rel
    mask_path = tmp_path / mask_rel
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), (245, 245, 245))
    ImageDraw.Draw(image).ellipse((12, 10, 52, 55), fill=(100, 80, 65))
    image.save(image_path)
    mask = Image.new("RGB", (128, 128), (0, 0, 0))
    ImageDraw.Draw(mask).ellipse((24, 20, 104, 110), fill=(255, 255, 255))
    mask.save(mask_path)

    config = {"_project_root": str(tmp_path), "reports_root": "reports"}
    rows = [{
        "image_id": "one", "dataset": "mango_segmentation", "task": "defect_segmentation",
        "image_path": image_rel, "mask_path": mask_rel,
    }]
    records = audit_and_normalize_masks(config, rows, visual_sample_size=1, seed=8)
    assert records[0]["alignment_status"] == "verified_resize_candidate"
    output = tmp_path / records[0]["normalized_mask_path"]
    with Image.open(output) as normalized:
        assert normalized.mode == "L"
        assert normalized.size == (64, 64)
        assert set(normalized.get_flattened_data()) == {0, 255}
    with (tmp_path / "reports" / "mask_alignment_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1
