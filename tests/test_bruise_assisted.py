from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pytest
from PIL import Image

from src.validation.bruise_assisted import (
    ASSISTED_FIELDS, ASSISTED_MANIFEST, _proposal, _validate_candidate,
    approve_assisted_batch, preview_assisted_approval, review_assisted_candidates,
)
from src.validation.bruise_ground_truth import _write_csv


def test_proposal_returns_source_sized_binary_mask_without_external_masks():
    rgb = np.full((240, 320, 3), 248, dtype=np.uint8)
    cv2.ellipse(rgb, (160, 125), (70, 100), 0, 0, 360, (90, 175, 45), -1)
    cv2.circle(rgb, (150, 110), 11, (35, 52, 24), -1)
    cv2.circle(rgb, (180, 145), 5, (25, 34, 18), -1)

    mask, info = _proposal(rgb, max_side=240)

    assert mask.shape == rgb.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask) > 0
    assert info["method"] == "rgb_to_lab_hsv_local_dark_brown_v1"
    assert info["component_count"] >= 1


def test_assisted_review_changes_only_explicit_ids_and_leaves_pending_by_default(tmp_path):
    source = np.full((90, 120, 3), 240, dtype=np.uint8)
    cv2.ellipse(source, (60, 45), (29, 38), 0, 0, 360, (75, 155, 55), -1)
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (52, 40), 6, 255, -1)
    mask_path = tmp_path / "candidate.png"
    Image.fromarray(mask).save(mask_path)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "selection_order": "1", "image_id": "img-one", "dataset": "test",
        "source_image_path": "source.png", "source_sha256": sha(source_path),
        "source_width": "120", "source_height": "90", "candidate_mask_path": "candidate.png",
        "candidate_mask_sha256": sha(mask_path), "mask_width": "120", "mask_height": "90",
        "mask_values": "0/255", "annotation_origin": "codex_assisted", "review_status": "pending",
        "annotation_status": "needs_edit", "lesion_presence": "present", "uncertain": "false", "method": "test",
        "method_parameters_json": "{}", "candidate_area_fraction": "0.01", "component_count": "1",
        "generated_at": "", "visual_review_status": "codex_visual_reviewed", "visual_review_note": "",
        "edited_by": "", "edited_at": "", "reviewer": "", "reviewed_at": "",
    }
    _write_csv(tmp_path / ASSISTED_MANIFEST, [row], ASSISTED_FIELDS)

    result = review_assisted_candidates(
        tmp_path, ["img-one"], "approved", "human-reviewer", "checked",
        confirmation_phrase="APPROVE 1 ASSISTED MASKS", refresh_contact_sheets=False,
    )

    saved = (tmp_path / ASSISTED_MANIFEST).read_text(encoding="utf-8")
    assert result["review_status"] == "approved"
    assert "codex_assisted,approved,approved" in saved
    assert "human-reviewer" in saved


def test_approval_requires_nonempty_exact_id_list():
    with pytest.raises(ValueError, match="explicit"):
        review_assisted_candidates(".", [], "approved", "human")
    with pytest.raises(ValueError, match="duplicate-free"):
        review_assisted_candidates(".", ["x", "x"], "approved", "human")


def test_empty_candidate_is_not_assumed_positive_and_can_be_explicitly_absent(tmp_path):
    source = np.full((72, 96, 3), 248, dtype=np.uint8)
    cv2.ellipse(source, (48, 36), (22, 30), 0, 0, 360, (75, 155, 55), -1)
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_path = tmp_path / "empty.png"
    Image.fromarray(mask).save(mask_path)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "selection_order": "1", "image_id": "empty-one", "dataset": "test",
        "source_image_path": "source.png", "source_sha256": sha(source_path),
        "source_width": "96", "source_height": "72", "candidate_mask_path": "empty.png",
        "candidate_mask_sha256": sha(mask_path), "mask_width": "96", "mask_height": "72",
        "mask_values": "0/255", "technical_validation_status": "passed",
        "annotation_origin": "codex_assisted", "review_status": "pending",
        "annotation_status": "needs_edit", "lesion_presence": "not_reviewed", "uncertain": "false",
        "method": "test", "method_parameters_json": "{}", "candidate_area_fraction": "0",
        "component_count": "0", "generated_at": "", "visual_review_status": "codex_visual_reviewed",
        "visual_review_note": "", "edited_by": "", "edited_at": "", "reviewer": "", "reviewed_at": "",
    }
    _write_csv(tmp_path / ASSISTED_MANIFEST, [row], ASSISTED_FIELDS)

    with pytest.raises(ValueError, match="lesion_presence_unresolved"):
        review_assisted_candidates(tmp_path, ["empty-one"], "approved", "human")
    result = review_assisted_candidates(tmp_path, ["empty-one"], "approved", "human",
                                        lesion_presence="absent",
                                        confirmation_phrase="APPROVE 1 ASSISTED MASKS",
                                        refresh_contact_sheets=False)
    saved = (tmp_path / ASSISTED_MANIFEST).read_text(encoding="utf-8")
    assert result["review_status"] == "approved"
    assert "absent" in saved


def test_rough_fruit_overlap_is_warning_not_hard_failure():
    rgb = np.full((100, 100, 3), 250, dtype=np.uint8)
    cv2.circle(rgb, (50, 50), 30, (70, 160, 45), -1)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 5, 255, -1)
    mask[:5, :5] = 255

    checked = _validate_candidate(rgb, mask, "present")

    assert checked["passed"] is True
    assert checked["status"] == "passed_with_warnings"
    assert checked["warnings"][0]["code"] == "mask_outside_plausible_fruit_region"
    assert mask[2, 2] == 255  # warning calculation never clips or edits the mask


def test_batch_approval_requires_final_confirmation_and_warning_override(tmp_path):
    source = np.full((100, 100, 3), 250, dtype=np.uint8)
    cv2.circle(source, (50, 50), 30, (70, 160, 45), -1)
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (50, 50), 5, 255, -1)
    mask[:5, :5] = 255
    mask_path = tmp_path / "candidate.png"
    Image.fromarray(mask).save(mask_path)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "selection_order": "1", "image_id": "warning-one", "dataset": "test",
        "source_image_path": "source.png", "source_sha256": sha(source_path),
        "source_width": "100", "source_height": "100", "candidate_mask_path": "candidate.png",
        "candidate_mask_sha256": sha(mask_path), "mask_width": "100", "mask_height": "100",
        "mask_values": "0/255", "technical_validation_status": "passed",
        "annotation_origin": "codex_assisted", "review_status": "pending",
        "annotation_status": "needs_edit", "lesion_presence": "present", "uncertain": "false",
        "technical_validation_warnings": "[]", "warning_override": "[]", "approval_batch_id": "",
        "human_review_notes": "", "method": "test", "method_parameters_json": "{}",
        "candidate_area_fraction": "0.01", "component_count": "2", "generated_at": "",
        "visual_review_status": "codex_visual_reviewed", "visual_review_note": "",
        "edited_by": "", "edited_at": "", "reviewer": "", "reviewed_at": "",
    }
    _write_csv(tmp_path / ASSISTED_MANIFEST, [row], ASSISTED_FIELDS)

    preview = preview_assisted_approval(tmp_path, ["warning-one"])
    assert preview["blocked_count"] == 0
    assert preview["warning_image_ids"] == ["warning-one"]
    with pytest.raises(ValueError, match="accept exactly the warning image IDs"):
        approve_assisted_batch(tmp_path, ["warning-one"], "human",
                               "APPROVE 1 ASSISTED MASKS", accepted_warning_ids=[])
    pending = (tmp_path / ASSISTED_MANIFEST).read_text(encoding="utf-8")
    assert "codex_assisted,pending" in pending

    result = approve_assisted_batch(tmp_path, ["warning-one"], "human",
                                    "APPROVE 1 ASSISTED MASKS",
                                    accepted_warning_ids=["warning-one"], refresh_contact_sheets=False)
    approved_rows = (tmp_path / ASSISTED_MANIFEST).read_text(encoding="utf-8")
    audit = (tmp_path / "data/annotations/bruise_gt_assisted/approval_audit.jsonl").read_text(encoding="utf-8")
    assert result["approved_count"] == 1
    assert "codex_assisted,approved,approved" in approved_rows
    assert "warning-one" in audit and "mask_outside_plausible_fruit_region" in audit
