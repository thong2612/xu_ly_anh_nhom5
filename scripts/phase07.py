from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.phase07 import (
    build_phase07_report, evaluate_approved_fruit_masks, evaluate_external,
    evaluate_reviewed_kmeans_pairs, import_external_images, prepare_review_workflows,
    run_error_analysis, run_phase07, run_robustness, set_external_review,
    set_kmeans_pair_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MangoResearch Phase 7 validation workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create external manifest, 50-pair review pack, and empty guarded metric tables")
    prepare.add_argument("--review-limit", type=int, default=50)
    review = sub.add_parser("kmeans-review", help="Record a human review decision for one prepared Dataset C pair")
    review.add_argument("--image-id", required=True)
    review.add_argument("--status", required=True, choices=["approved", "rejected", "uncertain"])
    review.add_argument("--reviewer", default="")
    review.add_argument("--notes", default="")
    sub.add_parser("kmeans-review-gui", help="Open the interactive Dataset C pair review window")
    sub.add_parser("kmeans-pre-review", help="Run local image/mask checks without changing review decisions")
    sub.add_parser("kmeans-evaluate", help="Calculate metrics only for manually approved Dataset C pairs")
    sub.add_parser("external-import", help="Index new JPG/PNG files under data/external_validation/images")
    external_review = sub.add_parser("external-review", help="Manually record external image labels and review status")
    external_review.add_argument("--image-id", required=True)
    external_review.add_argument("--status", required=True, choices=["approved", "rejected", "uncertain", "pending"])
    external_review.add_argument("--ripeness-label", default="", choices=["", "unripe", "partially_ripe", "ripe"])
    external_review.add_argument("--defect-label", default="", choices=["", "healthy", "defective"])
    external_review.add_argument("--source-group", default="")
    external_review.add_argument("--reviewer", default="")
    external_review.add_argument("--notes", default="")
    sub.add_parser("external-evaluate", help="Run locked Phase 5.1 artifacts on manually labeled external images")
    sub.add_parser("fruit-evaluate", help="Evaluate Phase 2 configurations only against human-approved whole-fruit masks")
    error = sub.add_parser("error-analysis", help="Create per-image Phase 5.1 test error analysis and contact sheets")
    robustness = sub.add_parser("robustness", help="Run paired mild input perturbations on real mango images")
    robustness.add_argument("--limit-images", type=int, default=5)
    sub.add_parser("report", help="Regenerate the Phase 7 research synthesis report")
    run = sub.add_parser("run", help="Prepare workflows and run feasible Phase 7 checks on current data")
    run.add_argument("--robustness-limit", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_review_workflows(ROOT, args.review_limit)
        elif args.command == "kmeans-review":
            result = str(set_kmeans_pair_review(ROOT, args.image_id, args.status, args.reviewer, args.notes))
        elif args.command == "kmeans-review-gui":
            from src.validation.kmeans_review_gui import run_review_gui
            run_review_gui(ROOT)
            result = {"review_window": "closed", "manifest": "data/metadata/phase07_kmeans_pair_review.csv"}
        elif args.command == "kmeans-pre-review":
            from src.validation.kmeans_prereview import run_kmeans_prereview
            result = run_kmeans_prereview(ROOT)
        elif args.command == "kmeans-evaluate":
            result = evaluate_reviewed_kmeans_pairs(ROOT)
        elif args.command == "external-import":
            result = import_external_images(ROOT)
        elif args.command == "external-review":
            result = str(set_external_review(ROOT, args.image_id, args.status, args.ripeness_label,
                                             args.defect_label, args.source_group, args.reviewer, args.notes))
        elif args.command == "external-evaluate":
            result = evaluate_external(ROOT)
        elif args.command == "fruit-evaluate":
            result = evaluate_approved_fruit_masks(ROOT)
        elif args.command == "error-analysis":
            result = run_error_analysis(ROOT)
        elif args.command == "robustness":
            result = run_robustness(ROOT, args.limit_images)
        elif args.command == "report":
            result = str(build_phase07_report(ROOT))
        elif args.command == "run":
            result = run_phase07(ROOT, args.robustness_limit)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

