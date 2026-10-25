from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.color.preprocess import decode_image, gray_world, visible_rgb
from src.segmentation.baseline import CHANNELS, SegmentationResult, extract_channel, segment_channel
from src.utils.config import project_path


LOGGER = logging.getLogger("mangoresearch.phase2")
PIPELINE_VERSION = "phase2-single-channel-v1"
ELIGIBLE = {
    "ripeness_classification": "eligible_for_ripeness",
    "defect_analysis": "eligible_for_defect_analysis",
    "defect_segmentation": "eligible_for_segmentation",
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_cache_key(image_id: str, source_hash: str, settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(f"{PIPELINE_VERSION}|{image_id}|{source_hash}|{payload}".encode()).hexdigest()


def _label(row: dict[str, str]) -> str:
    if row.get("task") == "ripeness_classification":
        return row.get("ripeness_label", "")
    if row.get("task") == "defect_segmentation":
        return row.get("mask_type", "") or row.get("defect_label", "")
    return row.get("defect_label", "")


def select_candidates(rows: list[dict[str, str]], project_root: Path, datasets: list[str],
                      per_group: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically sample eligible, non-conflicting Phase 0.2 manifest rows."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task = row.get("task", "")
        flag = ELIGIBLE.get(task)
        if row.get("dataset") not in datasets or not flag or not _truthy(row.get(flag, "")):
            continue
        if task == "ripeness_classification" and _truthy(row.get("ripeness_label_conflict", "")):
            continue
        path = project_path({"_project_root": str(project_root)}, row.get("image_path", ""))
        if path.is_file():
            groups[(row.get("dataset", ""), task, _label(row))].append(row)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for key in sorted(groups):
        pool = list(groups[key])
        rng.shuffle(pool)
        group_count = 0
        for row in pool:
            pixel_hash = row.get("pixel_sha256", "") or row.get("image_id", "")
            if pixel_hash in selected_hashes:
                continue
            selected_hashes.add(pixel_hash)
            record = dict(row)
            record["display_label"] = _label(row)
            record["absolute_image_path"] = str(project_path({"_project_root": str(project_root)}, row["image_path"]))
            chosen.append(record)
            group_count += 1
            if group_count >= per_group:
                break
    return chosen


def _settings(seg: dict[str, Any]) -> dict[str, Any]:
    return {
        "channels": list(seg.get("candidate_channels", list(CHANNELS))),
        "methods": list(seg.get("methods", ["fixed", "otsu"])),
        "polarities": list(seg.get("polarities", ["binary", "binary_inv"])),
        "fixed_thresholds": dict(seg.get("fixed_thresholds", {"lab_a": 0.0, "lab_b": 8.0, "hsv_h": 120.0, "hsv_s": 0.15})),
        "morphology": dict(seg.get("morphology", {})),
        "ranking": dict(seg.get("ranking", {})),
        "preview_max_side": int(seg.get("preview_max_side", 512)),
        "gray_world_enabled": bool(seg.get("gray_world_enabled", True)),
        "minimum_variant_agreement_iou": float(seg.get("minimum_variant_agreement_iou", 0.5)),
    }


def _resize_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    factor = min(1.0, max_side / max(h, w))
    if factor >= 1:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA)


def _variant_image(rgb: np.ndarray, alpha: np.ndarray | None, use_gray_world: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if use_gray_world:
        result = gray_world(rgb, alpha)
        return result.rgb, {"gains": list(result.gains), "clipping_fraction": result.clipping_fraction,
                            "clipped_channel_value_fraction": result.clipped_channel_value_fraction,
                            "applied": result.applied, "reason": result.reason}
    return rgb, {"gains": [1.0, 1.0, 1.0], "clipping_fraction": 0.0,
                 "clipped_channel_value_fraction": 0.0, "applied": False, "reason": "disabled"}


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left > 0, right > 0
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def _run_candidates(rgb: np.ndarray, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, SegmentationResult]]:
    morphology = settings["morphology"]
    ranking = settings["ranking"]
    fixed_thresholds = settings["fixed_thresholds"]
    rows: list[dict[str, Any]] = []
    results: dict[str, SegmentationResult] = {}
    for channel in settings["channels"]:
        for method in settings["methods"]:
            for polarity in settings["polarities"]:
                result = segment_channel(
                    rgb, channel=channel, method=method, polarity=polarity,
                    fixed_threshold=float(fixed_thresholds.get(channel, 0.0)),
                    open_kernel=int(morphology.get("open_kernel", 3)),
                    close_kernel=int(morphology.get("close_kernel", 9)),
                    open_iterations=int(morphology.get("open_iterations", 1)),
                    close_iterations=int(morphology.get("close_iterations", 2)),
                    fill_hole_pixels=bool(morphology.get("fill_holes", True)),
                    min_area_fraction=float(ranking.get("min_area_fraction", 0.03)),
                    max_area_fraction=float(ranking.get("max_area_fraction", 0.90)),
                    target_area_fraction=float(ranking.get("target_area_fraction", 0.32)),
                    minimum_score=float(ranking.get("minimum_score", 0.45)),
                    ambiguity_margin=float(ranking.get("ambiguity_margin", 0.025)),
                )
                key = f"{channel}|{method}|{polarity}"
                results[key] = result
                rows.append({"channel": channel, "channel_name": CHANNELS[channel][2], "method": method,
                             "polarity": polarity, "threshold_native": result.threshold_native,
                             "threshold_u8": result.threshold_u8,
                             "raw_foreground_fraction": float(np.count_nonzero(result.raw_mask) / result.raw_mask.size),
                             "component_count_in_area_range": result.component_count,
                             "selected_area_fraction": result.selected_area_fraction,
                             "rank_score_heuristic": result.score, "status": result.status,
                             "details": result.details})
    return rows, results


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 30, 30), max(1, round(max(rgb.shape[:2]) / 500)))
    return output


def _preview(sample: dict[str, Any], original: np.ndarray, balanced: np.ndarray, alpha: np.ndarray | None,
             e1result: SegmentationResult, e2result: SegmentationResult | None, target: Path,
             default_channel: str, default_method: str, default_polarity: str) -> None:
    shown_rgb = visible_rgb(original, alpha)
    shown_gw = visible_rgb(balanced, alpha)
    index = {"lab_a": 1, "lab_b": 2, "hsv_h": 0, "hsv_s": 1}[default_channel]
    is_lab = default_channel.startswith("lab_")
    channel_image = extract_channel(shown_rgb, default_channel)
    low, high, _ = CHANNELS[default_channel]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    panels = [
        (shown_rgb, "RGB original", None),
        (shown_gw, "E2 Gray World", None),
        (channel_image, f"Single channel: {CHANNELS[default_channel][2]}", "gray"),
        (e1result.raw_mask, f"E1 raw threshold ({e1result.threshold_native:.2f})", "gray"),
        (e1result.proposed_mask, f"E1 postprocess: {e1result.status}", "gray"),
        (np.where(e1result.proposed_mask[:, :, None] > 0, shown_rgb, 255).astype(np.uint8),
         "E1 foreground on white" if e1result.status == "accepted_heuristic" else "E1 proposal (not saved)", None),
        (_overlay(shown_rgb, e1result.proposed_mask), "E1 contour overlay", None),
    ]
    if e2result is not None:
        panels.append((e2result.proposed_mask, f"E2 postprocess: {e2result.status}", "gray"))
    else:
        panels.append((np.zeros(e1result.proposed_mask.shape, np.uint8), "E2 comparison disabled", "gray"))
    for axis, (image, title, cmap) in zip(axes.ravel(), panels):
        if cmap:
            axis.imshow(image, cmap=cmap, vmin=low if title.startswith("Single channel") else 0,
                        vmax=high if title.startswith("Single channel") else 255)
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(f"{sample['dataset']} | {sample['task']} / {sample['display_label']} | {sample['image_id']}\n"
                 f"Preview uses {default_channel} / {default_method} / {default_polarity}; heuristic outputs are not ground truth",
                 fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)


def _write_candidates_manifest(path: Path, samples: list[dict[str, Any]]) -> None:
    existing_rows = _read_csv(path) if path.exists() else []
    existing_by_id = {row.get("image_id"): row for row in existing_rows}
    merged = dict(existing_by_id)
    for sample in samples:
        previous = existing_by_id.get(sample["image_id"], {})
        merged[sample["image_id"]] = {**previous, "image_id": sample["image_id"], "dataset": sample["dataset"],
                                      "task": sample["task"], "label": sample["display_label"],
                                      "source_image_path": sample["image_path"],
                                      "width": sample.get("width", previous.get("width", "")),
                                      "height": sample.get("height", previous.get("height", "")),
                                      "phase02_split": sample.get("phase02_split", "") or sample.get("split", ""),
                                      "candidate_ground_truth_path": previous.get("candidate_ground_truth_path", f"data/annotations/fruit_masks/{sample['image_id']}.png"),
                                      "annotation_status": previous.get("annotation_status", "pending_manual_gt"),
                                      "notes": previous.get("notes", "No heuristic prediction is treated as ground truth.")}
    _write_csv(path, list(merged.values()))


def _plot_status_overview(sample_rows: list[dict[str, Any]], output: Path) -> None:
    if not sample_rows:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(sample_rows) * 0.35), 5))
    names = [f"{r['dataset'].replace('mango_', '')}\n{r['image_id'][:7]}" for r in sample_rows]
    statuses = [r.get("e1_status", "failed") for r in sample_rows]
    palette = {"accepted_heuristic": "#4daf4a", "ambiguous_components": "#ffb000",
               "low_confidence": "#e41a1c", "no_plausible_component": "#984ea3",
               "empty_threshold": "#377eb8", "all_foreground": "#a65628", "error": "#555555"}
    ax.bar(range(len(names)), [1] * len(names), color=[palette.get(s, "#999999") for s in statuses])
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("One configured E1 baseline outcome per candidate (heuristic status, not quality metric)")
    for index, status in enumerate(statuses):
        ax.text(index, 0.5, status.replace("_", "\n"), rotation=90, ha="center", va="center", fontsize=6, color="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _report(path: Path, samples: list[dict[str, Any]], sample_rows: list[dict[str, Any]],
            candidate_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]],
            config: dict[str, Any], gt_count: int, preview_index: Path, status_plot: Path,
            candidate_manifest: Path) -> None:
    counts = Counter((row["dataset"], row["task"], row["display_label"]) for row in samples)
    results_by_id = {row.get("image_id"): row for row in sample_rows}
    outcome = Counter(row.get("e1_status", "error") for row in sample_rows)
    all_status = Counter(row["status"] for row in candidate_rows)
    enabled = bool(config.get("gray_world_enabled", True))
    gains = np.asarray([json.loads(row["gray_world_gains_rgb"]) for row in sample_rows if row.get("gray_world_gains_rgb")], dtype=np.float64)
    clipping = [float(row["gray_world_clipping_fraction"]) for row in sample_rows]
    channel_clipping = [float(row["gray_world_clipped_channel_value_fraction"]) for row in sample_rows]
    agreements = [float(row["variant_mask_iou_preview"]) for row in sample_rows if row.get("variant_mask_iou_preview") not in (None, "")]
    agreement_threshold = float(config.get("minimum_variant_agreement_iou", 0.5))
    low_agreement = sum(value < agreement_threshold for value in agreements)
    representative = next((row for row in sample_rows if row.get("task") == "ripeness_classification" and row.get("e1_status") == "accepted_heuristic"), None)
    failed_case = next((row for row in sample_rows if row.get("e1_status") != "accepted_heuristic"), None)
    variant_case = min((row for row in sample_rows if row.get("variant_mask_iou_preview") not in (None, "")),
                       key=lambda row: float(row["variant_mask_iou_preview"]), default=None)
    case_samples = []
    for case_label, row in (("Example heuristic proposal (visual review still required)", representative),
                            ("Threshold failure / no saved mask", failed_case),
                            ("Lowest E1/E2 agreement in this sample", variant_case)):
        if row and row["image_id"] not in {item[1]["image_id"] for item in case_samples}:
            rel = Path(row["preview_path"])
            try:
                rel = rel.relative_to("reports")
            except ValueError:
                pass
            case_samples.append((case_label, {**row, "report_preview_path": rel.as_posix()}))
    lines = [
        "# MangoResearch — Phase 2 baseline fruit segmentation", "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
        "## Scope and method", "",
        f"Processed {len(sample_rows)} real eligible image(s) from a deterministic Phase 0.2 sample; {len(failure_rows)} failed to decode/process. Candidate channel/method/polarity evaluations: {len(candidate_rows)}.",
        "E1 uses the original RGB image. E2 applies the Phase 1 Gray World function to RGB; E2 is a comparison only. The same channel scales, threshold rules, morphology, and component ranking were used for both variants.",
        f"Configured reference output is {config.get('default_channel', 'lab_b')} / {config.get('default_method', 'fixed')} / {config.get('default_polarity', 'binary')}. This is a reproducible baseline choice, not a validated or optimal configuration.",
        "Only one color channel is thresholded at a time. Fixed thresholds and Otsu are tested in both foreground polarities. Hue Otsu is included as a diagnostic even though hue is circular and ordinary Otsu does not model circularity.",
        "Connected components are ranked by configured size plausibility, center, compactness, aspect ratio, and border contact. The score is a heuristic for review; accepted_heuristic is not a verified fruit mask. Empty, full-frame, ambiguous, low-score, or implausible results are not saved as accepted masks.",
        f"Gray World comparison enabled: {enabled}. Phase 0.2 manifest and splits were read only; raw images and existing labels/splits were not modified.", "",
        "## Ground-truth audit", "",
                  "The available sources do not provide verified whole-fruit masks: Dataset A provides ripeness classes; Dataset B is a defect image dataset; Dataset C provides pixel-level damage-region masks, not whole-fruit silhouettes. Dataset C masks are therefore excluded from fruit-segmentation evaluation.",
        "Sources: [Dataset A](https://data.mendeley.com/datasets/tyghd6gxw2/1), [Dataset B](https://data.mendeley.com/datasets/jnp9tmyrt6/1), [Dataset C](https://data.mendeley.com/datasets/cy3ccgkx3n/1). Local Phase 0.2 audit: `reports/phase02_report.md`.",
                  f"Whole-fruit masks approved for evaluation at this run: **{gt_count}**. Imported masks are format-checked but remain `valid_format_pending_semantic_review` until a person confirms that they trace the whole fruit. No IoU, Dice, precision, or recall results are reported because there is no semantically reviewed whole-fruit ground truth.",
        f"A set of {len(samples)} source-linked manual annotation candidates was exported to `{candidate_manifest.as_posix()}`. The lightweight polygon annotation tool and exact-size/binary-mask importer are available; all candidate rows currently remain pending until a human annotates them.", "",
        "## Sample coverage", "", "| Dataset / task / label | Sampled | E1 heuristic candidate | E2 exported after review gate |", "|---|---:|---:|---:|"]
    for key, value in sorted(counts.items()):
        group_rows = [results_by_id[row["image_id"]] for row in samples
                      if (row["dataset"], row["task"], row["display_label"]) == key
                      and row["image_id"] in results_by_id]
        e1_count = sum(row.get("e1_status") == "accepted_heuristic" for row in group_rows)
        e2_count = sum(bool(row.get("e2_mask_path")) for row in group_rows)
        lines.append(f"| {' / '.join(key)} | {value} | {e1_count}/{value} | {e2_count}/{value} |")
    lines.extend(["", "## Baseline outcomes", "", "Configured E1 outcome counts:", ""])
    lines.extend([f"- `{key}`: {value}" for key, value in sorted(outcome.items())] or ["- None"])
    lines.extend(["", f"Across the full E1/E2 channel × threshold × polarity candidate table, statuses: `{dict(all_status)}`. Detailed thresholds, raw foreground fractions, area fractions, rank scores, and component counts are in `reports/phase2_candidate_results.csv`; per-image outcomes are in `reports/phase2_segmentation_results.csv`.",
                  (f"E1/E2 postprocessed-mask agreement at preview resolution (a stability comparison, not ground-truth accuracy) was available for {len(agreements)} pairs: mean {np.mean(agreements):.3f}, median {np.median(agreements):.3f}, minimum {min(agreements):.3f}; {low_agreement} pairs fell below the configured {agreement_threshold:.2f} review threshold." if agreements else "E1/E2 mask agreement was not calculated (Gray World comparison disabled or no pair of accepted proposals)."),
                  "Accepted E1 masks and E2 masks passing the inter-variant review gate were exported as 0/255 PNG at original resolution. Source, variant, threshold, and heuristic score are recorded in `data/metadata/phase2_segmentation_manifest.csv`. All are heuristic candidates, not labels; E2 disagreement/status-mismatch cases remain in previews/results but are not exported.",
                  (f"Gray World gain mean RGB: {np.mean(gains, axis=0).round(4).tolist()}; pixel clipping fraction mean/max: {np.mean(clipping):.4%}/{max(clipping):.4%}; clipped channel-value fraction mean/max: {np.mean(channel_clipping):.4%}/{max(channel_clipping):.4%}." if len(gains) else "Gray World statistics unavailable."),
                  "This report makes no claim that Gray World improves segmentation quality.", "",
                  "## Visual review", "", f"Status overview: ![E1 baseline status overview](<{status_plot.relative_to(path.parent).as_posix()}>).", "",
                  f"Representative cases: [preview contact sheet](<{preview_index.relative_to(path.parent).as_posix()}>). Individual previews are under `phase02_segmentation/previews/` and display RGB, Gray World, a single channel, raw threshold, postprocessed mask, foreground-on-white, and contour overlay. Proposals for rejected cases are explicitly marked not saved.",
                  ""])
    for case_label, row in case_samples:
        lines.append(f"- {case_label}: {row['dataset']} / {row['label']} / `{row['image_id']}` — [preview](<{row['report_preview_path']}>).")
    lines.extend([
                  "The preview sample includes varied datasets/tasks/labels, but automated selection cannot certify visual difficulty categories such as shadow or background similarity. Inspect individual previews before using any heuristic mask.", "",
                  "## Split and leakage limitations", "",
                  "No new train/validation/test split was created: Phase 0.2 assignments were preserved unchanged. Existing splits are provisional and have unresolved near-duplicate/augmentation lineage caveats (`official_accuracy_eligible=false`). The candidate sample itself is for preview/annotation preparation, not a model split.",
                  "## Data quality and readiness", "",
                  f"Image read/process failures: {len(failure_rows)}; details in `reports/phase2_segmentation_failures.csv`. Saved heuristic masks: E1 {sum(bool(r.get('e1_mask_path')) for r in sample_rows)}; E2 {sum(bool(r.get('e2_mask_path')) for r in sample_rows)}. All saved masks preserve native resolution and are binary PNG; raw source files are untouched.",
                  "Current readiness: the baseline pipeline, traceable previews, candidate table, and manual GT import path are ready for expert review. Fruit segmentation is **not quantitatively validated** and is not ready to serve as ground truth or support an accuracy claim. Before Phase 3, annotate and review whole-fruit masks and evaluate thresholds/morphology against those masks; adjudicate rejected/ambiguous cases.",
                  "No K-Means, learned segmentation, feature extraction, or model training was performed.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase2(config: dict[str, Any], dataset_keys: list[str] | None = None,
               per_group: int | None = None, gray_world_enabled: bool | None = None,
               force: bool = False, image_id: str | None = None, limit: int | None = None) -> Path:
    phase = dict(config.get("segmentation", {}))
    project_root = Path(config["_project_root"])
    manifest_path = project_path(config, phase.get("manifest_path", "data/metadata/phase02_manifest.csv"))
    out_dir = project_path(config, phase.get("output_dir", "reports/phase02_segmentation"))
    report_path = project_path(config, phase.get("report_path", "reports/phase02_segmentation_report.md"))
    cache_dir = project_path(config, phase.get("cache_dir", "data/processed/phase2_cache"))
    mask_root = project_path(config, phase.get("mask_dir", "data/processed/fruit_masks"))
    per_group = int(per_group if per_group is not None else phase.get("samples_per_group", 5))
    seed = int(phase.get("seed", 20261023))
    settings = _settings(phase)
    settings["gray_world_enabled"] = bool(phase.get("gray_world_enabled", True)) if gray_world_enabled is None else gray_world_enabled
    if per_group < 1 or settings["preview_max_side"] < 64:
        raise ValueError("samples_per_group must be >= 1 and preview_max_side >= 64")
    keys = dataset_keys or list(config["datasets"])
    invalid = set(keys) - set(config["datasets"])
    if invalid:
        raise ValueError(f"Unknown dataset key(s): {', '.join(sorted(invalid))}")
    manifest_rows = _read_csv(manifest_path)
    samples = select_candidates(manifest_rows, project_root, keys, per_group, seed)
    if image_id:
        samples = [row for row in samples if row["image_id"] == image_id]
        if not samples:
            selected_row = next((row for row in manifest_rows if row.get("image_id") == image_id), None)
            if selected_row is None:
                raise ValueError(f"Unknown image_id: {image_id}")
            flag = ELIGIBLE.get(selected_row.get("task", ""), "")
            if not flag or not _truthy(selected_row.get(flag, "")) or (
                selected_row.get("task") == "ripeness_classification" and _truthy(selected_row.get("ripeness_label_conflict", ""))
            ):
                raise ValueError(f"image_id {image_id} is not eligible in the Phase 0.2 clean manifest")
            absolute = project_path(config, selected_row.get("image_path", ""))
            if not absolute.is_file():
                raise FileNotFoundError(absolute)
            selected_row = dict(selected_row)
            selected_row["display_label"] = _label(selected_row)
            selected_row["absolute_image_path"] = str(absolute)
            samples = [selected_row]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No eligible images found for Phase 2 sample")
    LOGGER.info("Phase 2: %d eligible source images", len(samples))
    default_channel = str(phase.get("default_channel", "lab_b"))
    default_method = str(phase.get("default_method", "fixed"))
    default_polarity = str(phase.get("default_polarity", "binary"))
    agreement_threshold = float(phase.get("minimum_variant_agreement_iou", 0.5))
    if default_channel not in CHANNELS:
        raise ValueError(f"Unknown default channel: {default_channel}")
    method_specs = settings["methods"]
    polarity_specs = settings["polarities"]
    if default_method not in method_specs or default_polarity not in polarity_specs:
        raise ValueError("Default method/polarity must be included in configured candidate methods/polarities")
    sample_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    sample_for_overview: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    preview_dir = out_dir / "previews"
    for index, sample in enumerate(samples, start=1):
        try:
            image_path = Path(sample["absolute_image_path"])
            source_hash = _sha256(image_path)
            cache_key = make_cache_key(sample["image_id"], source_hash, settings)
            cache_path = cache_dir / f"{cache_key}.npz"
            frame = decode_image(image_path)
            preview_original = _resize_rgb(visible_rgb(frame.rgb, frame.alpha), settings["preview_max_side"])
            balanced_native, gw_record = _variant_image(frame.rgb, frame.alpha, settings["gray_world_enabled"])
            preview_balanced = _resize_rgb(visible_rgb(balanced_native, frame.alpha), settings["preview_max_side"])
            # Candidate threshold comparisons use the same bounded preview scale for predictable runtime.
            variants = [("E1", preview_original)]
            if settings["gray_world_enabled"]:
                variants.append(("E2", preview_balanced))
            cached_rows: list[dict[str, Any]] | None = None
            cache_hit = cache_path.is_file() and not force
            if cache_path.is_file() and not force:
                with np.load(cache_path, allow_pickle=False) as archive:
                    cached_rows = json.loads(str(archive["rows_json"].item()))
                LOGGER.info("Cache hit %d/%d: %s", index, len(samples), sample["image_id"])
            if cached_rows is None:
                records = []
                for variant, rgb in variants:
                    LOGGER.info("Processing %d/%d %s %s", index, len(samples), sample["image_id"], variant)
                    rows, _ = _run_candidates(rgb, settings)
                    for row in rows:
                        records.append({**row, "variant": variant})
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, rows_json=np.asarray(json.dumps(records)))
                temporary.replace(cache_path)
                cached_rows = records
            for row in cached_rows:
                candidate_rows.append({"image_id": sample["image_id"], "dataset": sample["dataset"],
                                       "task": sample["task"], "label": sample["display_label"],
                                       "cache_key": cache_key, **row})
            def get_default(rgb: np.ndarray) -> SegmentationResult:
                return segment_channel(
                    rgb, channel=default_channel, method=default_method, polarity=default_polarity,
                    fixed_threshold=float(settings["fixed_thresholds"].get(default_channel, 0.0)),
                    open_kernel=int(settings["morphology"].get("open_kernel", 3)),
                    close_kernel=int(settings["morphology"].get("close_kernel", 9)),
                    open_iterations=int(settings["morphology"].get("open_iterations", 1)),
                    close_iterations=int(settings["morphology"].get("close_iterations", 2)),
                    fill_hole_pixels=bool(settings["morphology"].get("fill_holes", True)),
                    min_area_fraction=float(settings["ranking"].get("min_area_fraction", 0.03)),
                    max_area_fraction=float(settings["ranking"].get("max_area_fraction", 0.90)),
                    target_area_fraction=float(settings["ranking"].get("target_area_fraction", 0.32)),
                    minimum_score=float(settings["ranking"].get("minimum_score", 0.45)),
                    ambiguity_margin=float(settings["ranking"].get("ambiguity_margin", 0.025)))
            e1 = get_default(preview_original)
            e2 = get_default(preview_balanced) if settings["gray_world_enabled"] else None
            agreement = (_mask_iou(e1.proposed_mask, e2.proposed_mask)
                         if e2 and e1.status == "accepted_heuristic" and e2.status == "accepted_heuristic" else None)
            e2_export_eligible = bool(e2 and e2.status == "accepted_heuristic" and e1.status == "accepted_heuristic"
                                      and agreement is not None and agreement >= agreement_threshold)
            preview_path = preview_dir / f"{sample['dataset']}__{sample['image_id']}.png"
            _preview(sample, preview_original, preview_balanced, None, e1, e2, preview_path,
                     default_channel, default_method, default_polarity)
            preview_paths.append(preview_path)
            record = {
                "image_id": sample["image_id"], "dataset": sample["dataset"], "task": sample["task"],
                "label": sample["display_label"], "source_image_path": sample["image_path"],
                "source_sha256": source_hash, "phase02_split": sample.get("phase02_split", "") or sample.get("split", ""),
                "native_width": frame.rgb.shape[1], "native_height": frame.rgb.shape[0],
                "default_channel": default_channel, "default_method": default_method,
                "default_polarity": default_polarity, "e1_status": e1.status,
                "e1_score_heuristic": e1.score, "e1_selected_area_fraction_preview": e1.selected_area_fraction,
                "e1_threshold_native": e1.threshold_native,
                "e2_algorithm_status": e2.status if e2 else "disabled",
                "e2_status": ("accepted_heuristic" if e2_export_eligible else
                              "review_variant_disagreement" if e2 and e2.status == "accepted_heuristic" and e1.status == "accepted_heuristic" else
                              "review_variant_status_mismatch" if e2 and e2.status == "accepted_heuristic" else
                              e2.status if e2 else "disabled"),
                "e2_score_heuristic": e2.score if e2 else "",
                "e2_selected_area_fraction_preview": e2.selected_area_fraction if e2 else "",
                "variant_mask_iou_preview": agreement if agreement is not None else "",
                "e2_export_eligible": str(e2_export_eligible).lower(),
                "gray_world_gains_rgb": json.dumps(gw_record["gains"]),
                "gray_world_clipping_fraction": gw_record["clipping_fraction"],
                "gray_world_clipped_channel_value_fraction": gw_record["clipped_channel_value_fraction"],
                "gray_world_reason": gw_record["reason"], "cache_key": cache_key,
                "cache_path": str(cache_path.relative_to(project_root)), "cache_hit": cache_hit,
                "preview_path": str(preview_path.relative_to(project_root)),
            }
            # Export full-resolution masks only for accepted configured-reference results.
            output_paths: dict[str, str] = {}
            if e1.status == "accepted_heuristic" or (e2 and e2.status == "accepted_heuristic"):
                full_e1 = get_default(frame.rgb) if e1.status == "accepted_heuristic" else None
                full_e2 = get_default(balanced_native) if e2_export_eligible else None
                full_by_variant = {"E1": full_e1, "E2": full_e2}
                for variant_name in ("E1", "E2"):
                    full_result = full_by_variant[variant_name]
                    target_dir = mask_root / sample["dataset"] / variant_name
                    target = target_dir / f"{sample['image_id']}.png"
                    if full_result is None or full_result.status != "accepted_heuristic":
                        # Remove only this pipeline's old derived result when the current
                        # run no longer considers it exportable; never touch raw files.
                        target.unlink(missing_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), full_result.proposed_mask):
                        raise OSError(f"Failed to write heuristic mask: {target}")
                    output_paths[variant_name] = str(target.relative_to(project_root))
                    candidate_records.append({"image_id": sample["image_id"], "dataset": sample["dataset"],
                                              "task": sample["task"], "label": sample["display_label"],
                                              "source_image_path": sample["image_path"], "source_sha256": source_hash,
                                              "variant": variant_name, "channel": default_channel,
                                              "method": default_method, "polarity": default_polarity,
                                              "threshold_native": full_result.threshold_native,
                                              "mask_path": str(target.relative_to(project_root)),
                                              "mask_type": "heuristic_fruit_candidate_not_ground_truth",
                                              "width": frame.rgb.shape[1], "height": frame.rgb.shape[0],
                                              "rank_score_heuristic": full_result.score,
                                              "area_fraction": full_result.selected_area_fraction,
                                              "status": full_result.status, "cache_key": cache_key})
            record["e1_mask_path"] = output_paths.get("E1", "")
            record["e2_mask_path"] = output_paths.get("E2", "")
            sample_rows.append(record)
            sample_for_overview.append(record)
        except Exception as exc:
            LOGGER.exception("Phase 2 failed for %s", sample.get("image_id", "unknown"))
            failure_rows.append({"image_id": sample.get("image_id", ""), "dataset": sample.get("dataset", ""),
                                 "image_path": sample.get("image_path", ""), "error": repr(exc)})

    _write_csv(project_path(config, "reports/phase2_candidate_results.csv"), candidate_rows)
    _write_csv(project_path(config, "reports/phase2_segmentation_results.csv"), sample_rows)
    _write_csv(project_path(config, "data/metadata/phase2_segmentation_manifest.csv"), candidate_records)
    _write_csv(project_path(config, "reports/phase2_segmentation_failures.csv"), failure_rows,
               ["image_id", "dataset", "image_path", "error"])
    _write_csv(project_path(config, "reports/phase2_metrics.csv"), [],
               ["image_id", "ground_truth_path", "prediction_path", "iou", "dice", "precision", "recall", "status"])
    candidate_manifest = project_path(config, "data/metadata/fruit_gt_candidates.csv")
    _write_candidates_manifest(candidate_manifest, samples)
    gt_manifest = project_path(config, "data/metadata/fruit_gt_manifest.csv")
    gt_rows = _read_csv(gt_manifest) if gt_manifest.exists() else []
    verified_gt = sum(row.get("validation_status") == "valid_whole_fruit_mask" for row in gt_rows)
    overview = out_dir / "baseline_status_overview.png"
    _plot_status_overview(sample_for_overview, overview)
    # A compact contact sheet references actual per-sample preview images.
    contact_path = out_dir / "preview_contact_sheet.png"
    if preview_paths:
        thumbs = []
        for item in preview_paths:
            try:
                image = cv2.cvtColor(cv2.imread(str(item)), cv2.COLOR_BGR2RGB)
                thumbs.append(cv2.resize(image, (384, 216), interpolation=cv2.INTER_AREA))
            except Exception:
                continue
        if thumbs:
            columns = 3
            rows_n = (len(thumbs) + columns - 1) // columns
            sheet = np.full((rows_n * 216, columns * 384, 3), 245, dtype=np.uint8)
            for i, image in enumerate(thumbs):
                y, x = divmod(i, columns)
                sheet[y * 216:(y + 1) * 216, x * 384:(x + 1) * 384] = image
            contact_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(contact_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    _report(report_path, samples, sample_rows, candidate_rows, failure_rows, {
        **phase, "gray_world_enabled": settings["gray_world_enabled"], "default_channel": default_channel,
        "default_method": default_method, "default_polarity": default_polarity,
        "minimum_variant_agreement_iou": agreement_threshold,
    }, verified_gt, contact_path, overview, candidate_manifest)
    LOGGER.info("Phase 2 complete: %d processed, %d failures, %d accepted E1", len(sample_rows),
                len(failure_rows), sum(row.get("e1_status") == "accepted_heuristic" for row in sample_rows))
    return report_path


def import_fruit_masks(config: dict[str, Any], input_dir: str | Path) -> Path:
    """Import human-created masks only after strict ID, size and binary checks."""
    project_root = Path(config["_project_root"])
    candidates_path = project_path(config, "data/metadata/fruit_gt_candidates.csv")
    candidate_rows = _read_csv(candidates_path)
    input_root = Path(input_dir)
    if not input_root.is_absolute():
        input_root = project_root / input_root
    output_root = project_path(config, "data/annotations/fruit_masks")
    imported: list[dict[str, Any]] = []
    rejected: list[str] = []
    known_ids = {row.get("image_id", "") for row in candidate_rows}
    for incoming in input_root.glob("*.png"):
        if incoming.stem not in known_ids:
            rejected.append(f"{incoming.name}: image_id is not in fruit_gt_candidates.csv")
    for row in candidate_rows:
        image_id = row["image_id"]
        candidate = input_root / f"{image_id}.png"
        if not candidate.is_file():
            continue
        image_path = project_path(config, row["source_image_path"])
        frame = decode_image(image_path)
        mask = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            rejected.append(f"{image_id}: expected a single-channel PNG mask")
            continue
        if mask.shape != frame.rgb.shape[:2]:
            rejected.append(f"{image_id}: mask size {mask.shape[1]}x{mask.shape[0]} does not match image {frame.rgb.shape[1]}x{frame.rgb.shape[0]}")
            continue
        if not np.isin(mask, (0, 255)).all():
            rejected.append(f"{image_id}: mask values must be exactly 0/255")
            continue
        if np.count_nonzero(mask) in {0, mask.size}:
            rejected.append(f"{image_id}: empty or full-frame mask is not accepted")
            continue
        target = output_root / f"{image_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), mask):
            rejected.append(f"{image_id}: failed to write imported mask")
            continue
        imported.append({**row, "ground_truth_path": str(target.relative_to(project_root)),
                         "validation_status": "valid_format_pending_semantic_review", "mask_source": "human_import",
                         "foreground_pixels": int(np.count_nonzero(mask))})
    if rejected:
        rejection_path = project_path(config, "reports/fruit_gt_import_rejections.csv")
        _write_csv(rejection_path, [{"reason": item} for item in rejected], ["reason"])
    manifest_path = project_path(config, "data/metadata/fruit_gt_manifest.csv")
    existing = _read_csv(manifest_path) if manifest_path.exists() else []
    by_id = {row.get("image_id"): row for row in existing}
    for row in imported:
        by_id[row["image_id"]] = row
    _write_csv(manifest_path, list(by_id.values()))
    if imported:
        imported_ids = {row["image_id"] for row in imported}
        for row in candidate_rows:
            if row.get("image_id") in imported_ids:
                row["annotation_status"] = "imported_pending_semantic_review"
                row["candidate_ground_truth_path"] = str((output_root / f"{row['image_id']}.png").relative_to(project_root))
        _write_csv(candidates_path, candidate_rows)
    if rejected:
        raise ValueError(f"Imported {len(imported)} mask(s); rejected {len(rejected)}. See reports/fruit_gt_import_rejections.csv")
    return manifest_path


def approve_fruit_masks(config: dict[str, Any], image_ids: list[str]) -> Path:
    """Record explicit human semantic review after format validation/import."""
    if not image_ids:
        raise ValueError("Provide one or more image IDs that a person has visually reviewed")
    project_root = Path(config["_project_root"])
    manifest_path = project_path(config, "data/metadata/fruit_gt_manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError("No imported whole-fruit masks are available to approve")
    rows = _read_csv(manifest_path)
    by_id = {row.get("image_id"): row for row in rows}
    candidates_path = project_path(config, "data/metadata/fruit_gt_candidates.csv")
    candidate_list = _read_csv(candidates_path)
    candidate_rows = {row["image_id"]: row for row in candidate_list}
    for image_id in image_ids:
        row = by_id.get(image_id)
        candidate = candidate_rows.get(image_id)
        if row is None or candidate is None:
            raise ValueError(f"Unknown or not-imported candidate image_id: {image_id}")
        if row.get("validation_status") != "valid_format_pending_semantic_review":
            raise ValueError(f"{image_id} is not awaiting semantic review")
        source = decode_image(project_path(config, candidate["source_image_path"]))
        mask_path = project_path(config, row["ground_truth_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.shape != source.rgb.shape[:2]:
            raise ValueError(f"{image_id}: mask is missing, not grayscale, or does not match the source dimensions")
        if not np.isin(mask, (0, 255)).all() or np.count_nonzero(mask) in {0, mask.size}:
            raise ValueError(f"{image_id}: mask is not a non-empty binary 0/255 whole-fruit candidate")
        row["validation_status"] = "valid_whole_fruit_mask"
        row["semantic_review_status"] = "human_confirmed_whole_fruit"
        row["semantic_review_timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        candidate["annotation_status"] = "annotated"
        candidate["notes"] = "Imported binary mask passed format checks and was explicitly reviewed as whole-fruit ground truth."
    _write_csv(manifest_path, rows)
    _write_csv(candidates_path, candidate_list)
    return manifest_path

