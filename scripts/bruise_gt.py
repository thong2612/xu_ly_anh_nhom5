from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.bruise_ground_truth import (
    approve_masks, current_status, evaluate_approved, export_report, import_mask,
    prepare_candidates, record_case_note, review_status, revoke_approval, validate_saved_masks,
)
from src.validation.bruise_assisted import (
    ASSISTED_MANIFEST, approve_assisted_batch, evaluate_assisted_approved,
    generate_assisted_masks, import_human_approved_assisted, preview_assisted_approval,
    review_assisted_candidates, validate_assisted_masks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent bruise ground-truth annotation and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Validate the locked 30-image list and prepare its RGB-only contact sheet")
    annotate = sub.add_parser("annotate", help="Open/resume the RGB-only multi-polygon annotator")
    annotate.add_argument("--image-id", help="Open one candidate by its manifest ID")

    imported = sub.add_parser("import-mask", help="Import a manual binary PNG as an unapproved mask candidate")
    imported.add_argument("--image-id", required=True)
    imported.add_argument("--mask-path", required=True)
    imported.add_argument("--lesion-presence", required=True, choices=["present", "absent", "uncertain"])
    imported.add_argument("--notes", default="")

    approve = sub.add_parser("approve", help="Semantically approve exactly the listed mask candidates")
    approve.add_argument("--image-ids", nargs="+", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--notes", default="")

    revoke = sub.add_parser("revoke", help="Revoke explicit prior approvals before editing a mask")
    revoke.add_argument("--image-ids", nargs="+", required=True)
    revoke.add_argument("--reviewer", required=True)
    revoke.add_argument("--notes", required=True)

    review = sub.add_parser("review", help="Mark exactly the listed images pending, rejected, or uncertain")
    review.add_argument("--image-ids", nargs="+", required=True)
    review.add_argument("--status", required=True, choices=["pending", "rejected", "uncertain"])
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")

    sub.add_parser("validate", help="Recheck saved masks without changing semantic approval")
    sub.add_parser("evaluate", help="Run paired locked K-Means evaluation only on approved masks")
    sub.add_parser("report", help="Export current status tables, chart, and report without running K-Means")
    sub.add_parser("status", help="Show candidate, saved-mask, review, and evaluation counts")

    case = sub.add_parser("case-note", help="Record observed visual errors separately from cause hypotheses")
    case.add_argument("--image-id", required=True)
    case.add_argument("--category", required=True, choices=["undersegmentation", "normal_peel_false_positive",
                      "shadow_false_positive", "small_lesion_missed", "fruit_mask_failure",
                      "lab_hsv_difference", "other"])
    case.add_argument("--observed-issue", default="")
    case.add_argument("--cause-hypothesis", default="")
    case.add_argument("--notes", default="")

    assist_generate = sub.add_parser("assist-generate", help="Generate separate RGB-only Codex-assisted candidate masks")
    assist_generate.add_argument("--max-side", type=int, default=2400)
    assist_generate.add_argument("--force", action="store_true", help="Regenerate candidates; make a copy first if edited")
    assist_annotate = sub.add_parser("assist-annotate", help="Open the brush editor for assisted candidates")
    assist_annotate.add_argument("--image-id", help="Open a specific candidate ID")
    assist_review = sub.add_parser("assist-review", help="Set an explicit review status on an exact assisted ID list")
    assist_review.add_argument("--image-ids", nargs="+", required=True)
    assist_review.add_argument("--status", required=True, choices=["pending", "approved", "rejected", "uncertain"])
    assist_review.add_argument("--reviewer", required=True)
    assist_review.add_argument("--notes", default="")
    assist_review.add_argument("--lesion-presence", choices=["present", "absent", "uncertain", "not_reviewed"],
                               help="Explicit presence decision, needed before approval")
    assist_review.add_argument("--confirm", help="Exact final confirmation, e.g. APPROVE 1 ASSISTED MASKS")
    assist_review.add_argument("--accept-warning-ids", nargs="*", default=[])
    assist_batch_preview = sub.add_parser("assist-batch-preview", help="Show exact IDs, blockers, and fruit-region warnings without changes")
    assist_batch_preview.add_argument("--image-ids", nargs="+", required=True)
    assist_batch_preview.add_argument("--lesion-presence", choices=["present", "absent"])
    assist_batch_preview.add_argument("--presence-map-csv", help="CSV with image_id,lesion_presence for mixed batches")
    assist_batch_approve = sub.add_parser("assist-batch-approve", help="Approve an exact assisted batch after preview and explicit confirmation")
    assist_batch_approve.add_argument("--image-ids", nargs="+", required=True)
    assist_batch_approve.add_argument("--reviewer", required=True)
    assist_batch_approve.add_argument("--lesion-presence", choices=["present", "absent"])
    assist_batch_approve.add_argument("--presence-map-csv", help="CSV with image_id,lesion_presence for mixed batches")
    assist_batch_approve.add_argument("--notes", default="")
    assist_import = sub.add_parser("assist-import", help="Import human-approved assisted masks as pending manual GT candidates")
    assist_import.add_argument("--image-ids", nargs="+", required=True)
    assist_import.add_argument("--notes", default="")
    sub.add_parser("assist-validate", help="Validate assisted mask shape, binary values, checksums, and fruit overlap")
    sub.add_parser("assist-status", help="Show counts for the separate assisted candidate set")
    sub.add_parser("assist-evaluate", help="Run paired locked Lab K=3 / HSV K=5 evaluation on 30 audited approved assisted masks")
    return parser


def _read_presence_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    import csv
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["image_id"]: row["lesion_presence"] for row in csv.DictReader(stream)}


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_candidates(ROOT)
            result["current_status"] = export_report(ROOT)
        elif args.command == "annotate":
            from src.validation.bruise_annotator import run_annotator
            run_annotator(ROOT, args.image_id)
            result = current_status(ROOT)
        elif args.command == "import-mask":
            result = import_mask(ROOT, args.image_id, args.mask_path, args.lesion_presence, args.notes)
        elif args.command == "approve":
            result = approve_masks(ROOT, args.image_ids, args.reviewer, args.notes)
            result["current_status"] = export_report(ROOT)
        elif args.command == "revoke":
            result = revoke_approval(ROOT, args.image_ids, args.reviewer, args.notes)
            result["current_status"] = export_report(ROOT)
        elif args.command == "review":
            result = review_status(ROOT, args.image_ids, args.status, args.reviewer, args.notes)
            result["current_status"] = export_report(ROOT)
        elif args.command == "validate":
            result = validate_saved_masks(ROOT)
            result["current_status"] = export_report(ROOT)
        elif args.command == "evaluate":
            result = evaluate_approved(ROOT)
        elif args.command == "report":
            result = export_report(ROOT)
        elif args.command == "status":
            result = current_status(ROOT)
        elif args.command == "case-note":
            result = record_case_note(ROOT, args.image_id, args.category, args.observed_issue,
                                      args.cause_hypothesis, args.notes)
            result["report"] = export_report(ROOT)["report"]
        elif args.command == "assist-generate":
            result = generate_assisted_masks(ROOT, args.max_side, args.force)
        elif args.command == "assist-annotate":
            from src.validation.bruise_assisted_annotator import run_assisted_annotator
            run_assisted_annotator(ROOT, args.image_id)
            result = {"manifest": ASSISTED_MANIFEST, "editor_closed": True}
        elif args.command == "assist-review":
            result = review_assisted_candidates(ROOT, args.image_ids, args.status, args.reviewer,
                                                args.notes, args.lesion_presence,
                                                confirmation_phrase=args.confirm,
                                                accepted_warning_ids=args.accept_warning_ids)
        elif args.command == "assist-batch-preview":
            result = preview_assisted_approval(ROOT, args.image_ids, args.lesion_presence,
                                               _read_presence_map(args.presence_map_csv))
        elif args.command == "assist-batch-approve":
            presence_map = _read_presence_map(args.presence_map_csv)
            preview = preview_assisted_approval(ROOT, args.image_ids, args.lesion_presence, presence_map)
            print(json.dumps({"preflight": preview}, ensure_ascii=False, indent=2), flush=True)
            if preview["blocked_count"]:
                raise ValueError("Batch approval stopped; resolve the listed blockers, then run a new preview.")
            accepted_warning_ids = []
            if preview["warning_image_ids"]:
                warning_phrase = "OVERRIDE WARNINGS: " + ",".join(sorted(preview["warning_image_ids"]))
                typed_warning_phrase = input(f"To retain these masks, type exactly `{warning_phrase}`: ").strip()
                if typed_warning_phrase != warning_phrase:
                    raise ValueError("Warning override was not confirmed; no masks were approved")
                accepted_warning_ids = list(preview["warning_image_ids"])
            confirmation_phrase = f"APPROVE {len(args.image_ids)} ASSISTED MASKS"
            typed_confirmation = input(f"Final confirmation — type exactly `{confirmation_phrase}`: ").strip()
            result = approve_assisted_batch(
                ROOT, args.image_ids, args.reviewer, typed_confirmation,
                lesion_presence=args.lesion_presence, presence_by_id=presence_map,
                accepted_warning_ids=accepted_warning_ids, notes=args.notes,
            )
        elif args.command == "assist-import":
            result = import_human_approved_assisted(ROOT, args.image_ids, args.notes)
        elif args.command == "assist-validate":
            result = validate_assisted_masks(ROOT)
        elif args.command == "assist-evaluate":
            result = evaluate_assisted_approved(ROOT)
        elif args.command == "assist-status":
            from collections import Counter
            from src.validation.bruise_ground_truth import _read_csv
            rows = _read_csv(ROOT / ASSISTED_MANIFEST)
            result = {"candidate_images": len(rows),
                      "masks_saved": sum(bool(row.get("candidate_mask_path")) for row in rows),
                      "annotation_status_counts": dict(Counter(row.get("annotation_status", "") for row in rows)),
                      "review_status_counts": dict(Counter(row.get("review_status", "") for row in rows)),
                      "lesion_presence_counts": dict(Counter(row.get("lesion_presence", "not_reviewed") for row in rows))}
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
