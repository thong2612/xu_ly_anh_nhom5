from __future__ import annotations

import numpy as np
import csv

from src.models.phase051 import (PAIR_FIELDS, _grouped_split_indices, _ssim_gray,
                                 _stable_group_ids, apply_visual_pair_review)


def test_ssim_identical_and_changed_images():
    image = np.full((256, 256), 127, dtype=np.uint8)
    assert _ssim_gray(image, image) == 1.0
    changed = image.copy()
    changed[80:150, 90:160] = 220
    assert _ssim_gray(image, changed) < 0.97


def test_grouped_split_is_repeatable_and_never_splits_groups():
    labels = np.asarray(["a", "b", "c"] * 40, dtype=object)
    groups = np.asarray([f"group-{i // 2}" for i in range(len(labels))], dtype=object)
    first = _grouped_split_indices(labels, groups, 20261023,
                                   {"train": 0.70, "validation": 0.15, "test": 0.15})
    second = _grouped_split_indices(labels, groups, 20261023,
                                    {"train": 0.70, "validation": 0.15, "test": 0.15})
    assert np.array_equal(first, second)
    assert set(first.tolist()) == {"train", "validation", "test"}
    for group in set(groups.tolist()):
        assert len(set(first[groups == group].tolist())) == 1
    assert abs(np.mean(first == "train") - 0.70) < 0.08


def test_known_lineage_hash_and_confirmed_pair_are_unioned():
    rows = [
        {"image_id": "a", "image_path": "a.jpg"},
        {"image_id": "b", "image_path": "b.jpg"},
        {"image_id": "c", "image_path": "c.jpg"},
    ]
    split = {
        "a": {"lineage_group_id": "lineage-1", "sha256": "sha-a", "pixel_sha256": "pix-a"},
        "b": {"lineage_group_id": "lineage-2", "sha256": "sha-b", "pixel_sha256": "pix-b"},
        "c": {"lineage_group_id": "lineage-2", "sha256": "sha-c", "pixel_sha256": "pix-c"},
    }
    groups = _stable_group_ids(rows, split, [("a.jpg", "b.jpg")])
    assert groups["a"] == groups["b"] == groups["c"]


def test_unlinked_images_keep_distinct_group_ids():
    rows = [{"image_id": "a", "image_path": "a.jpg"}, {"image_id": "b", "image_path": "b.jpg"}]
    split = {"a": {"lineage_group_id": "a", "sha256": "sha-a", "pixel_sha256": "pix-a"},
             "b": {"lineage_group_id": "b", "sha256": "sha-b", "pixel_sha256": "pix-b"}}
    groups = _stable_group_ids(rows, split, [])
    assert groups["a"] != groups["b"]


def test_pair_review_uses_pair_specific_decision_not_dhash(tmp_path):
    output = tmp_path / "reports" / "phase051"
    output.mkdir(parents=True)
    rows = [
        {"task": "ripeness", "image_id_a": "a", "image_id_b": "b", "path_a": "a.jpg", "path_b": "b.jpg",
         "dhash_hamming": "0", "same_file_sha256": "False", "same_decoded_pixel_sha256": "False"},
        {"task": "defect", "image_id_a": "c", "image_id_b": "d", "path_a": "c.jpg", "path_b": "d.jpg",
         "dhash_hamming": "0", "same_file_sha256": "True", "same_decoded_pixel_sha256": "False"},
    ]
    with (output / "near_duplicate_pair_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    decision_fields = ["task", "image_id_a", "image_id_b", "review_decision", "review_notes"]
    with (output / "pair_review_decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=decision_fields)
        writer.writeheader()
        writer.writerow({"task": "ripeness", "image_id_a": "a", "image_id_b": "b",
                         "review_decision": "rejected_different_fruit", "review_notes": "visual differences"})
    apply_visual_pair_review({"_project_root": str(tmp_path)})
    with (output / "near_duplicate_pair_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        reviewed = {r["image_id_a"]: r for r in csv.DictReader(stream)}
    assert reviewed["a"]["review_decision"] == "rejected_different_fruit"
    assert reviewed["c"]["review_decision"] == "confirmed_same_source"
