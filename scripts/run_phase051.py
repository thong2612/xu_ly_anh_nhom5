from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.phase051 import (
    apply_visual_pair_review,
    audit_candidates,
    prepare_splits,
    run_phase051_models,
    write_phase051_report,
)
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="MangoResearch Phase 5.1 leakage audit and model evaluation")
    parser.add_argument("stage", choices=("audit", "review", "split", "train", "report", "all"),
                        help="audit candidates, persist reviewed decisions, prepare grouped split, train, or run all stages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    if args.stage == "audit":
        result = audit_candidates(config)
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "review":
        result = apply_visual_pair_review(config)
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "split":
        result = prepare_splits(config)
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "train":
        result = run_phase051_models(config)
        audit_path = config["_project_root"] + "/reports/phase051/near_duplicate_audit_summary.json"
        split_path = config["_project_root"] + "/data/splits/phase051/split_summary.json"
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        report = write_phase051_report(config, audit, split, result)
        print(report)
        return 0
    if args.stage == "report":
        root = Path(config["_project_root"])
        report_dir = root / "reports/phase051"
        def read_rows(filename: str):
            with (report_dir / filename).open("r", encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))
        summary = json.loads((report_dir / "phase051_model_results.json").read_text(encoding="utf-8"))
        model_result = {"summary": summary,
            "paired": read_rows("phase051_d1_d4_validation_comparison.csv"),
            "stability": read_rows("phase051_d1_d4_cv_stability.csv"),
            "search_rows": read_rows("phase051_search_results.csv"),
            "metric_rows": read_rows("phase051_metrics.csv"),
            "confusion_rows": read_rows("phase051_confusion_matrices.csv"),
            "errors": read_rows("phase051_misclassifications.csv"),
            "phase05_comparison": read_rows("phase051_vs_phase05.csv")}
        audit = json.loads((report_dir / "near_duplicate_audit_summary.json").read_text(encoding="utf-8"))
        split = json.loads((root / "data/splits/phase051/split_summary.json").read_text(encoding="utf-8"))
        print(write_phase051_report(config, audit, split, model_result))
        return 0
    audit = audit_candidates(config)
    reviewed = apply_visual_pair_review(config)
    split = prepare_splits(config)
    models = run_phase051_models(config)
    report = write_phase051_report(config, audit, split, models)
    print(json.dumps({"pair_review": reviewed, "split": split, "models": models["summary"], "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
