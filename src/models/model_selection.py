from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import sklearn
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

from src.models.training import (
    FEATURE_EXPERIMENTS, LABELS, _make_pipeline, _metric_payload,
    _matrix_for_experiment,
)
from src.utils.config import project_path

LOGGER = logging.getLogger("mangoresearch.phase051")
TASK_FEATURES = {
    "ripeness": "data/features/ripeness_features.csv",
    "defect": "data/features/defect_features.csv",
}
TASK_SPLITS = {
    "ripeness": "data/splits/phase02/mango_ripeness_ripeness_classification.csv",
    "defect": "data/splits/phase02/mango_segmentation_defect_analysis.csv",
}
PAIR_FIELDS = [
    "task", "image_id_a", "image_id_b", "path_a", "path_b", "label_a", "label_b",
    "phase02_split_a", "phase02_split_b", "dhash_hamming", "same_file_sha256",
    "same_decoded_pixel_sha256", "resized_rmse_256", "ssim_gray_256",
    "source_component_a", "source_component_b", "review_decision", "review_notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path.replace("\\", "/")))


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssim_gray(left: np.ndarray, right: np.ndarray) -> float:
    """Windowed SSIM (11x11 Gaussian) for two aligned uint8 grayscale images."""
    x, y = left.astype(np.float32), right.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    c1, c2 = 6.5025, 58.5225
    result = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(np.mean(result))


def _read_rgb(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    if key not in cache:
        image = cv2.imread(key, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {path}")
        cache[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Bound memory for the full defect collection (some source images are
        # several megapixels). Re-decoding an evicted image is safe and rare.
        if len(cache) > 48:
            cache.pop(next(iter(cache)))
    return cache[key]


def _decoded_pixel_sha(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _gray_letterbox256(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    if key not in cache:
        gray = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Unable to decode image: {path}")
        height, width = gray.shape
        scale = min(256 / max(width, 1), 256 / max(height, 1))
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)
        canvas = np.full((256, 256), 127, dtype=np.uint8)
        top, left = (256 - new_h) // 2, (256 - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        cache[key] = canvas
    return cache[key]


def _dhash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _component(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    for marker in ("Damaged dataset", "Damaged_mangoes_cropped", "BAD MANGOES AUGMENTED",
                   "Mango Species Classification Dataset", "segmentation dataset"):
        if marker in parts:
            return marker
    return parts[-2] if len(parts) > 1 else ""


def _make_contact_sheets(rows: list[dict[str, Any]], root: Path, output_dir: Path) -> list[Path]:
    candidates = [r for r in rows if int(r["dhash_hamming"]) <= 1 or float(r["ssim_gray_256"]) >= 0.96]
    candidates.sort(key=lambda r: (int(r["dhash_hamming"]), -float(r["ssim_gray_256"]), r["task"], r["image_id_a"]))
    if not candidates:
        return []
    font = ImageFont.load_default()
    cell_w, cell_h, cols, per_sheet = 440, 188, 2, 12
    created: list[Path] = []
    for start in range(0, len(candidates), per_sheet):
        batch = candidates[start:start + per_sheet]
        sheet = Image.new("RGB", (cell_w * cols, cell_h * math.ceil(len(batch) / cols)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            x = (index % cols) * cell_w
            y = (index // cols) * cell_h
            for image_x, path_key in ((x + 3, "path_a"), (x + 220, "path_b")):
                image_path = root / Path(row[path_key])
                try:
                    rgb = _read_rgb(image_path, {})
                    im = Image.fromarray(rgb)
                    im.thumbnail((210, 142), Image.Resampling.LANCZOS)
                    sheet.paste(im, (image_x, y + 18))
                except Exception:
                    draw.rectangle((image_x, y + 18, image_x + 210, y + 160), fill="#ddd")
                    draw.text((image_x + 5, y + 28), "IMAGE UNREADABLE", fill="red", font=font)
            caption = (f"{row['task']} dH={row['dhash_hamming']} SSIM={float(row['ssim_gray_256']):.3f} "
                       f"{row['image_id_a'][:12]} / {row['image_id_b'][:12]}\n"
                       f"{Path(row['path_a']).name[:38]} | {Path(row['path_b']).name[:38]}")
            draw.text((x + 3, y + 2), caption, fill="black", font=font)
        path = output_dir / f"candidate_contact_sheet_{start // per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=90)
        created.append(path)
    return created


def audit_candidates(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_project_root"])
    output_dir = project_path(config, "reports/model_selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_by_task: dict[str, dict[str, dict[str, str]]] = {}
    source_by_task: dict[str, dict[str, dict[str, str]]] = {}
    for task in TASK_FEATURES:
        features = _read_csv(project_path(config, TASK_FEATURES[task]))
        split_rows = _read_csv(project_path(config, TASK_SPLITS[task]))
        feature_by_task[task] = {_norm(r["image_path"]): r for r in features}
        source_by_task[task] = {_norm(r["image_path"]): r for r in split_rows}

    near_path = project_path(config, "reports/near_duplicate_candidates.csv")
    if not near_path.is_file():
        raise FileNotFoundError(f"Near-duplicate candidate file is missing: {near_path}")
    candidates = _read_csv(near_path)
    gray_cache: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    cross_split_counts: Counter[str] = Counter()
    for candidate in candidates:
        if candidate["task_a"] != candidate["task_b"]:
            continue
        task = {"ripeness_classification": "ripeness", "defect_analysis": "defect"}.get(candidate["task_a"])
        if task is None:
            continue
        key_a, key_b = _norm(candidate["image_path_a"]), _norm(candidate["image_path_b"])
        a, b = feature_by_task[task].get(key_a), feature_by_task[task].get(key_b)
        sa, sb = source_by_task[task].get(key_a), source_by_task[task].get(key_b)
        if not a or not b or not sa or not sb or sa["phase02_split"] == sb["phase02_split"]:
            continue
        cross_split_counts[task] += 1
        path_a, path_b = root / Path(a["image_path"]), root / Path(b["image_path"])
        row: dict[str, Any] = {"task": task, "image_id_a": a["image_id"], "image_id_b": b["image_id"],
            "path_a": a["image_path"], "path_b": b["image_path"], "label_a": a["label"], "label_b": b["label"],
            "phase02_split_a": sa["phase02_split"], "phase02_split_b": sb["phase02_split"],
            "dhash_hamming": candidate["dhash_hamming"],
            "same_file_sha256": bool(sa.get("sha256") and sa.get("sha256") == sb.get("sha256")),
            "same_decoded_pixel_sha256": bool(sa.get("pixel_sha256") and sa.get("pixel_sha256") == sb.get("pixel_sha256")),
            "resized_rmse_256": "", "ssim_gray_256": "",
            "source_component_a": _component(a["image_path"]), "source_component_b": _component(b["image_path"]),
            "review_decision": "unverified", "review_notes": ""}
        try:
            resized_a = _gray_letterbox256(path_a, gray_cache)
            resized_b = _gray_letterbox256(path_b, gray_cache)
            row["resized_rmse_256"] = round(float(np.sqrt(np.mean((resized_a.astype(np.float32) - resized_b.astype(np.float32)) ** 2))), 5)
            row["ssim_gray_256"] = round(_ssim_gray(resized_a, resized_b), 6)
        except Exception as exc:
            row["review_notes"] = f"automatic decode/compare error: {type(exc).__name__}: {exc}"
        audit_rows.append(row)
    audit_rows.sort(key=lambda r: (r["task"], int(r["dhash_hamming"]), -float(r["ssim_gray_256"] or -1)))
    _write_csv(output_dir / "near_duplicate_pair_audit.csv", audit_rows, PAIR_FIELDS)
    sheets = _make_contact_sheets(audit_rows, root, output_dir)
    summary: dict[str, Any] = {"ssim_pairs": len(audit_rows), "all_cross_split_candidate_pairs": dict(cross_split_counts), "contact_sheets": [p.as_posix() for p in sheets], "tasks": {}}
    for task in TASK_FEATURES:
        rows = [r for r in audit_rows if r["task"] == task]
        summary["tasks"][task] = {
            "cross_split_candidate_pairs": cross_split_counts[task],
            "deep_review_pairs_with_ssim": sum(bool(r["ssim_gray_256"] != "") for r in rows),
            "dhash_zero": sum(int(r["dhash_hamming"]) == 0 for r in rows),
            "dhash_le_1": sum(int(r["dhash_hamming"]) <= 1 for r in rows),
            "file_sha_equal": sum(bool(r["same_file_sha256"]) for r in rows),
            "decoded_pixel_equal": sum(bool(r["same_decoded_pixel_sha256"]) for r in rows),
            "ssim_ge_0_96": sum(float(r["ssim_gray_256"] or -1) >= 0.96 for r in rows),
            "ssim_ge_0_90": sum(float(r["ssim_gray_256"] or -1) >= 0.90 for r in rows),
        }
    (output_dir / "near_duplicate_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def apply_visual_pair_review(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the curated, pair-specific review decisions to a fresh audit."""
    path = project_path(config, "reports/model_selection/near_duplicate_pair_audit.csv")
    rows = _read_csv(path)
    decision_path = path.parent / "pair_review_decisions.csv"
    manual_rows = _read_csv(decision_path) if decision_path.is_file() else []
    manual_by_pair = {frozenset((r["image_id_a"], r["image_id_b"])): r for r in manual_rows}
    for row in rows:
        if str(row.get("same_file_sha256", "")).casefold() in {"true", "1"} or str(row.get("same_decoded_pixel_sha256", "")).casefold() in {"true", "1"}:
            row["review_decision"] = "confirmed_same_source"
            row["review_notes"] = "Exact file or decoded-pixel hash match; confirmed by content identity."
            continue
        manual = manual_by_pair.get(frozenset((row["image_id_a"], row["image_id_b"])))
        if manual:
            row["review_decision"] = manual["review_decision"]
            row["review_notes"] = manual["review_notes"]
        else:
            row["review_decision"] = "unverified"
            row["review_notes"] = "No pair-specific visual decision recorded; retained as residual near-duplicate risk."
    _write_csv(path, rows, PAIR_FIELDS)
    counts = Counter((row["task"], row["review_decision"]) for row in rows)
    zero_counts = Counter((row["task"], row["review_decision"]) for row in rows if int(row["dhash_hamming"]) == 0)
    output = {"candidate_status_counts": {f"{task}:{decision}": count for (task, decision), count in counts.items()},
              "dhash_zero": {f"{task}:{decision}": count for (task, decision), count in zero_counts.items()}}
    (path.parent / "pair_review_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _stable_group_ids(rows: list[dict[str, str]], split_by_id: dict[str, dict[str, str]],
                      confirmed_pairs: list[tuple[str, str]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[max(lroot, rroot)] = min(lroot, rroot)

    tokens_by_id: dict[str, list[str]] = {}
    path_to_id = {_norm(r["image_path"]): r["image_id"] for r in rows}
    for row in rows:
        source = split_by_id[row["image_id"]]
        tokens = []
        if source.get("lineage_group_id", "").strip():
            tokens.append("lineage:" + source["lineage_group_id"].strip())
        if source.get("sha256", "").strip():
            tokens.append("sha256:" + source["sha256"].strip())
        if source.get("pixel_sha256", "").strip():
            tokens.append("pixel:" + source["pixel_sha256"].strip())
        if not tokens:
            tokens.append("image:" + row["image_id"])
        for token in tokens:
            find(token)
        for token in tokens[1:]:
            union(tokens[0], token)
        tokens_by_id[row["image_id"]] = tokens
    for path_a, path_b in confirmed_pairs:
        id_a, id_b = path_to_id.get(_norm(path_a)), path_to_id.get(_norm(path_b))
        if id_a and id_b:
            union(tokens_by_id[id_a][0], tokens_by_id[id_b][0])
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id, tokens in tokens_by_id.items():
        groups[find(tokens[0])].append(image_id)
    result: dict[str, str] = {}
    for members in groups.values():
        token = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:20]
        for image_id in members:
            result[image_id] = "p051_" + token
    return result


def _grouped_split_indices(labels: np.ndarray, groups: np.ndarray, seed: int,
                           ratios: dict[str, float]) -> np.ndarray:
    n_groups = len(set(groups.tolist()))
    if n_groups < 20:
        raise ValueError(f"Need at least 20 independent groups for a 70/15/15 fold allocation; found {n_groups}")
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(labels), -1, dtype=np.int16)
    for fold, (_, holdout) in enumerate(splitter.split(np.zeros((len(labels), 1)), labels, groups)):
        fold_of_row[holdout] = fold
    if np.any(fold_of_row < 0):
        raise AssertionError("Group fold assignment omitted rows")
    names = np.asarray(["train", "validation", "test"], dtype=object)
    target_ratios = np.asarray([ratios["train"], ratios["validation"], ratios["test"]])
    class_values = sorted(set(labels.tolist()))
    target = np.asarray([[sum(label == class_name for label in labels) * ratio for class_name in class_values]
                         for ratio in target_ratios])
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray] | None = None
    # Keep whole SGKF folds intact (and therefore all known groups together),
    # choosing among fixed 14/3/3 fold allocations to balance class counts.
    for _ in range(1800):
        order = rng.permutation(20)
        assignment = np.full(20, "train", dtype=object)
        assignment[order[:3]] = "validation"
        assignment[order[3:6]] = "test"
        counts = np.zeros_like(target)
        for split_idx, name in enumerate(names):
            selected = assignment[fold_of_row] == name
            for class_idx, class_name in enumerate(class_values):
                counts[split_idx, class_idx] = np.count_nonzero(selected & (labels == class_name))
        relative_error = np.abs(counts - target) / np.maximum(target, 1.0)
        total_error = np.abs(counts.sum(axis=1) - len(labels) * target_ratios) / max(len(labels), 1)
        objective = float(relative_error.mean() + total_error.sum() * 2.0)
        if best is None or objective < best[0]:
            best = (objective, assignment)
    assert best is not None
    return best[1][fold_of_row]


def prepare_splits(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_project_root"])
    phase051_dir = project_path(config, "data/splits/phase051")
    audit_path = project_path(config, "reports/model_selection/near_duplicate_pair_audit.csv")
    reviewed: list[dict[str, str]] = _read_csv(audit_path) if audit_path.is_file() else []
    confirmed_pairs: list[tuple[str, str]] = []
    review_by_pair: dict[frozenset[str], str] = {}
    for row in reviewed:
        decision = row.get("review_decision", "unverified").strip()
        key = frozenset((row["image_id_a"], row["image_id_b"]))
        review_by_pair[key] = decision
        exact = str(row.get("same_file_sha256", "")).casefold() in {"true", "1"} or str(row.get("same_decoded_pixel_sha256", "")).casefold() in {"true", "1"}
        if decision == "confirmed_same_source" or exact:
            confirmed_pairs.append((row["path_a"], row["path_b"]))
            if exact and decision == "unverified":
                row["review_decision"] = "confirmed_same_source"

    results: dict[str, Any] = {"tasks": {}}
    for task in TASK_FEATURES:
        features = _read_csv(project_path(config, TASK_FEATURES[task]))
        source_rows = _read_csv(project_path(config, TASK_SPLITS[task]))
        split_by_id = {r["image_id"]: r for r in source_rows}
        if {r["image_id"] for r in features} != set(split_by_id):
            raise ValueError(f"Feature and Phase 0.2 split image_id sets differ for {task}")
        task_confirmed = [(a, b) for a, b in confirmed_pairs
                          if any(_norm(r["image_path"]) == _norm(a) for r in features)
                          and any(_norm(r["image_path"]) == _norm(b) for r in features)]
        group_by_id = _stable_group_ids(features, split_by_id, task_confirmed)
        path_to_id = {_norm(r["image_path"]): r["image_id"] for r in features}
        risk_ids: set[str] = set()
        unknown_candidate_pairs = 0
        candidate_path = project_path(config, "reports/near_duplicate_candidates.csv")
        for candidate in _read_csv(candidate_path):
            expected_task = "ripeness_classification" if task == "ripeness" else "defect_analysis"
            if candidate["task_a"] != expected_task or candidate["task_b"] != expected_task:
                continue
            id_a, id_b = path_to_id.get(_norm(candidate["image_path_a"])), path_to_id.get(_norm(candidate["image_path_b"]))
            if not id_a or not id_b:
                continue
            sa, sb = split_by_id[id_a]["phase02_split"], split_by_id[id_b]["phase02_split"]
            if sa == sb:
                continue
            decision = review_by_pair.get(frozenset((id_a, id_b)), "unverified")
            sha_a, sha_b = split_by_id[id_a].get("sha256", ""), split_by_id[id_b].get("sha256", "")
            pix_a, pix_b = split_by_id[id_a].get("pixel_sha256", ""), split_by_id[id_b].get("pixel_sha256", "")
            exact = bool(sha_a and sha_a == sha_b) or bool(pix_a and pix_a == pix_b)
            if decision == "rejected_different_fruit":
                continue
            if decision != "confirmed_same_source" and not exact:
                risk_ids.update((id_a, id_b))
                unknown_candidate_pairs += 1
        labels = np.asarray([r["label"] for r in features], dtype=object)
        groups = np.asarray([group_by_id[r["image_id"]] for r in features], dtype=object)
        ratios = config.get("split", {}).get("ratios", {"train": .70, "validation": .15, "test": .15})
        new_splits = _grouped_split_indices(labels, groups, int(config.get("split", {}).get("seed", 20261023)), ratios)
        output_rows: list[dict[str, Any]] = []
        for row, group, new_split in zip(features, groups, new_splits):
            source = dict(split_by_id[row["image_id"]])
            source["phase051_split"] = str(new_split)
            source["phase051_group_id"] = group
            source["near_duplicate_risk_status"] = "unverified_cross_split_candidate" if row["image_id"] in risk_ids else "no_unresolved_indexed_candidate"
            # Preserve the false eligibility flag from Phase 0.2 without relaxing it.
            source["official_accuracy_eligible"] = "false"
            output_rows.append(source)
        out_path = phase051_dir / f"{task}_phase051.csv"
        fields = list(dict.fromkeys([*source_rows[0].keys(), "phase051_split", "phase051_group_id", "near_duplicate_risk_status"]))
        _write_csv(out_path, output_rows, fields)
        split_counts = Counter(new_splits.tolist())
        class_counts = {name: dict(Counter(row["label"] for row, split in zip(features, new_splits) if split == name))
                        for name in ("train", "validation", "test")}
        results["tasks"][task] = {
            "n_images": len(features), "n_groups": len(set(groups.tolist())),
            "n_confirmed_pair_links": len(task_confirmed), "n_risk_images": len(risk_ids),
            "n_unresolved_cross_split_candidates": unknown_candidate_pairs,
            "split_counts": dict(split_counts), "class_counts": class_counts,
            "phase02_split_path": TASK_SPLITS[task], "phase051_split_path": out_path.relative_to(root).as_posix(),
            "official_accuracy_eligible": False,
        }
    target = phase051_dir / "split_summary.json"
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _phase051_grid(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    settings = config.get("model_selection", {}).get("model_grid", {})
    if model_name == "KNN":
        return {"classifier__n_neighbors": list(settings.get("knn_neighbors", [5, 11])),
                "classifier__weights": list(settings.get("knn_weights", ["uniform", "distance"]))}
    if model_name == "SVM_RBF":
        return {"classifier__C": list(settings.get("svm_C", [1, 10])),
                "classifier__gamma": list(settings.get("svm_gamma", ["scale"]))}
    return {"classifier__n_estimators": list(settings.get("rf_n_estimators", [100, 200])),
            "classifier__max_depth": list(settings.get("rf_max_depth", [None, 20])),
            "classifier__min_samples_leaf": list(settings.get("rf_min_samples_leaf", [1]))}


def _write_confusion(path: Path, records: list[dict[str, Any]]) -> None:
    _write_csv(path, records, ["task", "experiment", "model", "stage", "true_label", "predicted_label", "count"])


def _task_inputs(config: dict[str, Any], schema: dict[str, Any], task: str) -> dict[str, Any]:
    feature_rows = _read_csv(project_path(config, TASK_FEATURES[task]))
    split_rows = _read_csv(project_path(config, f"data/splits/phase051/{task}_phase051.csv"))
    split_by_id = {row["image_id"]: row for row in split_rows}
    if len(feature_rows) != len({row["image_id"] for row in feature_rows}) or {r["image_id"] for r in feature_rows} != set(split_by_id):
        raise ValueError(f"Phase 4 features and Phase 5.1 split IDs differ for {task}")
    label_field = "ripeness_label" if task == "ripeness" else "defect_label"
    for row in feature_rows:
        split_row = split_by_id[row["image_id"]]
        if row["label"] != split_row.get(label_field, ""):
            raise ValueError(f"Feature label differs from Phase 0.2 for image {row['image_id']}")
        if row.get("official_accuracy_eligible", "").casefold() not in {"false", "0", ""}:
            raise ValueError(f"Phase 4 feature has an unexpectedly eligible accuracy flag: {row['image_id']}")
    metadata = set(schema.get("metadata_columns", [])) | {"label", "phase02_split", "phase051_split"}
    all_columns = set(schema.get("feature_columns", {}).get(task, []))
    if metadata.intersection(all_columns):
        raise ValueError(f"Metadata leakage in schema feature columns for {task}: {metadata & all_columns}")
    feature_sets = schema["feature_sets"][task]
    for experiment, group_names in FEATURE_EXPERIMENTS[task].items():
        if not set(group_names).issubset(feature_sets):
            raise ValueError(f"Experiment {experiment} references missing feature set")
    labels = np.asarray([r["label"] for r in feature_rows], dtype=object)
    split_names = np.asarray([split_by_id[r["image_id"]]["phase051_split"] for r in feature_rows], dtype=object)
    groups = np.asarray([split_by_id[r["image_id"]]["phase051_group_id"] for r in feature_rows], dtype=object)
    for name in ("train", "validation", "test"):
        if not np.any(split_names == name):
            raise ValueError(f"Phase 5.1 split has no {name} rows for {task}")
    for name_a, name_b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = set(groups[split_names == name_a]), set(groups[split_names == name_b])
        if a & b:
            raise AssertionError(f"Phase 5.1 group leakage between {name_a} and {name_b} for {task}")
    return {"rows": feature_rows, "split_by_id": split_by_id, "labels": labels,
            "split_names": split_names, "groups": groups, "feature_sets": feature_sets}


def _rows_for_metrics(task: str, experiment: str, model: str, stage: str,
                      payload: dict[str, Any], labels: list[str], params: str = "",
                      cv_std: Any = "") -> list[dict[str, Any]]:
    output = [{"task": task, "experiment": experiment, "model": model, "stage": stage,
        "scope": "overall", "label": "", "n_samples": int(sum(payload["support"])),
        "accuracy": payload["accuracy"], "macro_precision": payload["macro_precision"],
        "macro_recall": payload["macro_recall"], "macro_f1": payload["macro_f1"],
        "macro_f1_std": cv_std, "false_negative_defective": payload["false_negative_defective"],
        "parameters_json": params}]
    for idx, label in enumerate(labels):
        output.append({"task": task, "experiment": experiment, "model": model, "stage": stage,
            "scope": "per_class", "label": label, "n_samples": int(payload["support"][idx]),
            "accuracy": "", "macro_precision": "", "macro_recall": "", "macro_f1": "",
            "macro_f1_std": "", "false_negative_defective": "", "parameters_json": params,
            "precision": float(payload["precision"][idx]), "recall": float(payload["recall"][idx]),
            "f1": float(payload["f1"][idx])})
    return output


def run_phase051_models(config: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(project_path(config, "data/features/feature_schema.json").read_text(encoding="utf-8"))
    settings = config.get("model_selection", {})
    seed = int(settings.get("seed", config.get("split", {}).get("seed", 20261023)))
    requested_folds = int(settings.get("cv_folds", 5))
    model_names = ("KNN", "SVM_RBF", "RandomForest")
    report_dir = project_path(config, "reports/model_selection")
    model_dir = project_path(config, settings.get("model_dir", "models/classifiers"))
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, Any]] = {}
    fold_scores: dict[tuple[str, str], list[float]] = {}
    model_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}

    for task in TASK_FEATURES:
        data = _task_inputs(config, schema, task)
        rows, labels, split_names, groups = data["rows"], data["labels"], data["split_names"], data["groups"]
        train_idx = np.flatnonzero(split_names == "train")
        val_idx = np.flatnonzero(split_names == "validation")
        test_idx = np.flatnonzero(split_names == "test")
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        train_groups = groups[train_idx]
        n_folds = min(requested_folds, len(set(train_groups.tolist())))
        if n_folds < 2:
            raise ValueError(f"Too few independent train groups for group CV in {task}")
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_folds = list(splitter.split(np.zeros((len(train_idx), 1)), y_train, train_groups))
        for left, right in cv_folds:
            if set(train_groups[left]) & set(train_groups[right]):
                raise AssertionError(f"Group leakage within CV for {task}")
        candidates: list[dict[str, Any]] = []

        for experiment, feature_group_names in FEATURE_EXPERIMENTS[task].items():
            columns, X_all = _matrix_for_experiment(rows, data["feature_sets"], feature_group_names)
            if set(columns) & (set(schema.get("metadata_columns", [])) | {"label", "phase02_split", "phase051_split"}):
                raise ValueError(f"Metadata leakage in matrix columns for {task}/{experiment}")
            X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
            for model_name in model_names:
                LOGGER.info("Phase 5.1 %s / %s / %s: tuning on %d train rows", task, experiment, model_name, len(train_idx))
                search = GridSearchCV(_make_pipeline(model_name, seed), _phase051_grid(model_name, config),
                    scoring="f1_macro", refit=True, cv=cv_folds, n_jobs=1, error_score="raise")
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
                val_pred = estimator.predict(X_val)
                payload = _metric_payload(y_val, val_pred, LABELS[task])
                std = float(search.cv_results_["std_test_score"][search.best_index_])
                split_fold_scores = [float(search.cv_results_[f"split{fold}_test_score"][search.best_index_]) for fold in range(n_folds)]
                params_json = json.dumps(search.best_params_, sort_keys=True, default=str)
                candidate = {"task": task, "experiment": experiment, "model": model_name,
                    "feature_groups": feature_group_names, "columns": columns, "estimator": estimator,
                    "params": search.best_params_, "cv_mean": float(search.best_score_), "cv_std": std,
                    "cv_fold_scores": split_fold_scores, "validation": payload, "validation_prediction": val_pred,
                    "X_test": X_test, "y_test": y_test, "n_features": len(columns)}
                candidates.append(candidate)
                model_candidates[(task, experiment, model_name)] = candidate
                fold_scores[(task, experiment, model_name)] = split_fold_scores
                search_rows.append({"task": task, "experiment": experiment, "model": model_name,
                    "feature_sets": ", ".join(feature_group_names), "feature_count": len(columns),
                    "best_parameters_json": params_json, "cv_macro_f1_mean": float(search.best_score_),
                    "cv_macro_f1_std": std, "cv_fold_macro_f1_json": json.dumps(split_fold_scores),
                    "validation_accuracy": payload["accuracy"], "validation_macro_precision": payload["macro_precision"],
                    "validation_macro_recall": payload["macro_recall"], "validation_macro_f1": payload["macro_f1"],
                    "validation_missing_fraction": float(np.isnan(X_val).sum() / X_val.size) if X_val.size else 0.0,
                    "feature_missing_fraction": float(np.isnan(X_all).sum() / X_all.size) if X_all.size else 0.0})
                metric_rows.extend(_rows_for_metrics(task, experiment, model_name, "validation", payload, LABELS[task], params_json, std))
                for i, true in enumerate(LABELS[task]):
                    for j, pred in enumerate(LABELS[task]):
                        confusion_rows.append({"task": task, "experiment": experiment, "model": model_name,
                            "stage": "validation", "true_label": true, "predicted_label": pred,
                            "count": int(payload["confusion"][i, j])})
                for row_idx, pred in zip(val_idx, val_pred):
                    if str(labels[row_idx]) != str(pred):
                        errors.append({"task": task, "stage": "validation", "experiment": experiment,
                            "model": model_name, "image_id": rows[row_idx]["image_id"],
                            "true_label": labels[row_idx], "predicted_label": str(pred)})
                artifact = {"artifact_version": 1, "experiment_version": "phase5.1-group-audit-v1",
                    "task": task, "experiment": experiment, "model_name": model_name,
                    "pipeline": estimator, "feature_columns": columns, "feature_sets": feature_group_names,
                    "labels": LABELS[task], "seed": seed, "hyperparameters": search.best_params_,
                    "cv_macro_f1_mean": float(search.best_score_), "cv_macro_f1_std": std,
                    "validation_metrics": {k: payload[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
                    "train_image_ids": [rows[i]["image_id"] for i in train_idx],
                    "validation_image_ids": [rows[i]["image_id"] for i in val_idx],
                    "phase051_split": f"data/splits/phase051/{task}_phase051.csv",
                    "official_accuracy_eligible": False,
                    "caveat": "Exploratory: unresolved near-duplicate and/or source-lineage risks remain."}
                joblib.dump(artifact, model_dir / f"{task}_{experiment}_{model_name}.joblib", compress=3)
                LOGGER.info("Phase 5.1 %s %s %s: CV %.4f±%.4f, validation %.4f", task, experiment, model_name,
                            candidate["cv_mean"], std, payload["macro_f1"])

        winner = sorted(candidates, key=lambda c: (-c["validation"]["macro_f1"],
                        -c["validation"]["macro_recall"], -c["validation"]["accuracy"],
                        c["n_features"], c["experiment"], c["model"]))[0]
        test_pred = winner["estimator"].predict(winner["X_test"])
        test_payload = _metric_payload(winner["y_test"], test_pred, LABELS[task])
        metric_rows.extend(_rows_for_metrics(task, winner["experiment"], winner["model"],
            "test_locked", test_payload, LABELS[task], json.dumps(winner["params"], sort_keys=True, default=str)))
        for i, true in enumerate(LABELS[task]):
            for j, pred in enumerate(LABELS[task]):
                confusion_rows.append({"task": task, "experiment": winner["experiment"],
                    "model": winner["model"], "stage": "test_locked", "true_label": true,
                    "predicted_label": pred, "count": int(test_payload["confusion"][i, j])})
        for row_idx, pred in zip(test_idx, test_pred):
            if str(labels[row_idx]) != str(pred):
                errors.append({"task": task, "stage": "test_locked", "experiment": winner["experiment"],
                    "model": winner["model"], "image_id": rows[row_idx]["image_id"],
                    "true_label": labels[row_idx], "predicted_label": str(pred)})
        best_path = model_dir / f"{task}_selected.joblib"
        # The underlying estimator was fit on train only; validation selected it,
        # and the test was touched once after that choice was locked.
        winning_artifact = joblib.load(model_dir / f"{task}_{winner['experiment']}_{winner['model']}.joblib")
        winning_artifact["test_metrics_locked_once"] = {k: test_payload[k] for k in
            ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
        joblib.dump(winning_artifact, best_path, compress=3)
        chosen[task] = {"experiment": winner["experiment"], "model": winner["model"],
            "validation_macro_f1": winner["validation"]["macro_f1"],
            "validation_macro_precision": winner["validation"]["macro_precision"],
            "validation_macro_recall": winner["validation"]["macro_recall"],
            "validation_accuracy": winner["validation"]["accuracy"],
            "cv_macro_f1_mean": winner["cv_mean"], "cv_macro_f1_std": winner["cv_std"],
            "test_locked_once": {k: test_payload[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
            "model_path": best_path.relative_to(Path(config["_project_root"])).as_posix()}

    # Paired D1 vs D4 validation and fold comparisons use exactly the same rows
    # and the same group-aware CV folds for each classifier.
    paired_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        d1 = model_candidates.get(("defect", "D1_original_color", model_name))
        d4 = model_candidates.get(("defect", "D4_original_color_plus_lab_k3_hsv_k5", model_name))
        if d1 is None or d4 is None:
            continue
        split_rows = _read_csv(project_path(config, "data/splits/phase051/defect_phase051.csv"))
        split_by_id = {r["image_id"]: r for r in split_rows}
        all_feature_rows = _read_csv(project_path(config, TASK_FEATURES["defect"]))
        y_validation = np.asarray([r["label"] for r in all_feature_rows if split_by_id[r["image_id"]]["phase051_split"] == "validation"], dtype=object)
        p1, p4 = d1["validation_prediction"], d4["validation_prediction"]
        m1, m4 = _metric_payload(y_validation, p1, LABELS["defect"]), _metric_payload(y_validation, p4, LABELS["defect"])
        corrected = int(np.count_nonzero((p1 != y_validation) & (p4 == y_validation)))
        regressed = int(np.count_nonzero((p1 == y_validation) & (p4 != y_validation)))
        paired_rows.append({"model": model_name, "n_validation": len(y_validation),
            "D1_macro_f1": m1["macro_f1"], "D4_macro_f1": m4["macro_f1"],
            "delta_D4_minus_D1": m4["macro_f1"] - m1["macro_f1"],
            "D1_accuracy": m1["accuracy"], "D4_accuracy": m4["accuracy"],
            "D4_corrected_D1_errors": corrected, "D4_new_errors": regressed})
        fs1, fs4 = fold_scores[("defect", "D1_original_color", model_name)], fold_scores[("defect", "D4_original_color_plus_lab_k3_hsv_k5", model_name)]
        differences = [b - a for a, b in zip(fs1, fs4)]
        stability_rows.append({"model": model_name, "fold_count": len(differences),
            "D1_cv_macro_f1_mean": d1["cv_mean"], "D1_cv_macro_f1_std": d1["cv_std"],
            "D4_cv_macro_f1_mean": d4["cv_mean"], "D4_cv_macro_f1_std": d4["cv_std"],
            "mean_paired_fold_delta_D4_minus_D1": float(np.mean(differences)),
            "std_paired_fold_delta": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
            "folds_D4_better": sum(d > 1e-12 for d in differences),
            "folds_tied": sum(abs(d) <= 1e-12 for d in differences),
            "folds_D4_worse": sum(d < -1e-12 for d in differences),
            "fold_delta_json": json.dumps(differences)})

    phase5_search_path = project_path(config, "reports/phase05_search_results.csv")
    old_rows = _read_csv(phase5_search_path) if phase5_search_path.is_file() else []
    old_by_key = {(r.get("task"), r.get("experiment"), r.get("model")): r for r in old_rows}
    comparison_rows = []
    for row in search_rows:
        old = old_by_key.get((row["task"], row["experiment"], row["model"]), {})
        comparison_rows.append({"task": row["task"], "experiment": row["experiment"], "model": row["model"],
            "phase05_validation_macro_f1": old.get("validation_macro_f1", ""),
            "phase051_validation_macro_f1": row["validation_macro_f1"],
            "phase051_cv_macro_f1_mean": row["cv_macro_f1_mean"], "phase051_cv_macro_f1_std": row["cv_macro_f1_std"],
            "validation_macro_f1_delta": (float(row["validation_macro_f1"]) - float(old["validation_macro_f1"])) if old.get("validation_macro_f1") else ""})

    _write_csv(report_dir / "phase051_search_results.csv", search_rows, list(search_rows[0]))
    _write_csv(report_dir / "phase051_metrics.csv", metric_rows, list(dict.fromkeys(k for r in metric_rows for k in r)))
    _write_confusion(report_dir / "phase051_confusion_matrices.csv", confusion_rows)
    _write_csv(report_dir / "phase051_misclassifications.csv", errors,
        ["task", "stage", "experiment", "model", "image_id", "true_label", "predicted_label"])
    _write_csv(report_dir / "phase051_d1_d4_validation_comparison.csv", paired_rows,
        ["model", "n_validation", "D1_macro_f1", "D4_macro_f1", "delta_D4_minus_D1", "D1_accuracy", "D4_accuracy", "D4_corrected_D1_errors", "D4_new_errors"])
    _write_csv(report_dir / "phase051_d1_d4_cv_stability.csv", stability_rows,
        ["model", "fold_count", "D1_cv_macro_f1_mean", "D1_cv_macro_f1_std", "D4_cv_macro_f1_mean", "D4_cv_macro_f1_std", "mean_paired_fold_delta_D4_minus_D1", "std_paired_fold_delta", "folds_D4_better", "folds_tied", "folds_D4_worse", "fold_delta_json"])
    _write_csv(report_dir / "phase051_vs_phase05.csv", comparison_rows,
        ["task", "experiment", "model", "phase05_validation_macro_f1", "phase051_validation_macro_f1", "phase051_cv_macro_f1_mean", "phase051_cv_macro_f1_std", "validation_macro_f1_delta"])
    summary = {"runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "seed": seed, "requested_cv_folds": requested_folds, "models": list(model_names),
        "chosen_by_validation_only": chosen, "official_accuracy_eligible": False,
        "model_search_rows": len(search_rows), "metrics": len(metric_rows), "confusion_matrices": len(confusion_rows)}
    (report_dir / "phase051_model_results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"summary": summary, "paired": paired_rows, "stability": stability_rows,
            "search_rows": search_rows, "metric_rows": metric_rows, "confusion_rows": confusion_rows,
            "errors": errors, "phase05_comparison": comparison_rows}


def write_phase051_report(config: dict[str, Any], audit_summary: dict[str, Any],
                          split_summary: dict[str, Any], model_result: dict[str, Any]) -> Path:
    report_path = project_path(config, "reports/model_selection_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    search_rows = model_result["search_rows"]
    lines = [
        "# MangoResearch — Phase 5.1: Leakage audit và đánh giá lại", "",
        "## Phương pháp và kiểm soát", "",
        "Phase 0.2 split và raw images được giữ nguyên. Split Phase 5.1 được lưu riêng; known lineage groups, raw-file SHA-256, decoded-pixel SHA-256 và các cặp đã xác nhận bằng xem ảnh được gộp thành group. Các cặp chưa xác minh được giữ cờ rủi ro. dHash chỉ tạo candidate, không tự xác nhận danh tính.",
        "SSIM được tính trên ảnh xám sau khi resize giữ tỷ lệ và letterbox vào 256×256 (cửa sổ Gaussian 11×11). File/pixel hashes lấy từ manifest Phase 0.2. Contact sheets và quyết định xem ảnh được lưu dưới `reports/model_selection/`.",
        "Split mới dùng seed=20261023: StratifiedGroupKFold 20 folds, gán 14 fold train, 3 validation, 3 test; chọn trong 1,800 hoán vị cố định cách gán 6 fold holdout có tổng số ảnh và số nhãn gần tỷ lệ mục tiêu nhất. Mọi group đã biết nằm trong một split. Validation chọn feature/model; test chỉ đánh giá một lần cho cấu hình đã khóa.",
        "Imputer, scaler và classifier được fit trong scikit-learn Pipeline trên train mới; GridSearchCV dùng 5-fold StratifiedGroupKFold trên train. Grid cố định: KNN k={5,11}, weights={uniform,distance}; SVM-RBF C={1,10}, gamma=scale; Random Forest n_estimators={100,200}, max_depth={None,20}, min_samples_leaf=1. Chỉ các cột thuộc feature sets trong `feature_schema.json` đi vào X. Label chỉ là y; image_id, đường dẫn, nguồn, split, group, mask và ground-truth mask không phải feature. Không có ground-truth defect mask nào được đưa vào input.",
        "",
        "## Near-duplicate audit", "",
        "| Nhiệm vụ | Candidate xuyên Phase 0.2 split | dHash=0 | Cặp có SSIM | File SHA khớp | Pixel SHA khớp | SSIM ≥ 0.96 | Đã xác nhận cùng nguồn | Bác bỏ | Chưa xác minh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pair_rows = _read_csv(project_path(config, "reports/model_selection/near_duplicate_pair_audit.csv"))
    for task in ("ripeness", "defect"):
        rows = [r for r in pair_rows if r["task"] == task]
        counts = Counter(r.get("review_decision", "unverified") for r in rows)
        stats = audit_summary.get("tasks", {}).get(task, {})
        lines.append(f"| {task} | {stats.get('cross_split_candidate_pairs', 0):,} | {stats.get('dhash_zero', 0):,} | {stats.get('deep_review_pairs_with_ssim', 0):,} | {stats.get('file_sha_equal', 0):,} | {stats.get('decoded_pixel_equal', 0):,} | {stats.get('ssim_ge_0_96', 0):,} | {counts['confirmed_same_source']:,} | {counts['rejected_different_fruit']:,} | {counts['unverified']:,} |")
    label_conflicts = [r for r in pair_rows if r["task"] == "defect" and r.get("review_decision") == "confirmed_same_source" and r.get("label_a") != r.get("label_b")]
    conflict_examples = ", ".join(f"`{Path(r['path_a']).name}` ↔ `{Path(r['path_b']).name}` ({r['label_a']} / {r['label_b']})" for r in label_conflicts)
    lines.extend(["", f"- Dataset A: trong dHash=0 pairs, visually distinct mangoes with different blemish/shape patterns were rejected; one similar same-base-name pair remains unverified. No dHash=0 pair was confirmed to be the same source with different ripeness labels.",
        f"- Defect task: {len(label_conflicts)} visually confirmed same-source pair has conflicting target labels (healthy/defective): {conflict_examples or 'none'}. Both original labels are preserved; this group remains in one split and is flagged in the pair audit.",
        "- dHash candidate index covers distances 0–4. SSIM was computed for all cross-split candidates; contact sheets prioritize dHash≤1 and SSIM≥0.96. Pairs outside the visually reviewed subset remain unverified; SSIM is not identity proof.",
        "", "### Contact sheets", ""])
    for path in audit_summary.get("contact_sheets", []):
        lines.append(f"![Near duplicate review: {Path(path).name}](phase051/{Path(path).name})")
    lines.extend(["", "## Split Phase 5.1", ""])
    for task, data in split_summary.get("tasks", {}).items():
        ratios = data.get("split_counts", {})
        pct = {k: (100 * v / data["n_images"] if data.get("n_images") else 0) for k, v in ratios.items()}
        lines.extend([f"### {task}", "",
            f"- Ảnh: {data['n_images']:,}; known groups sau khi gộp: {data['n_groups']:,}; visually confirmed links: {data['n_confirmed_pair_links']:,}.",
            f"- Split: {ratios}; tỷ lệ xấp xỉ `{pct}`; nhãn theo split: `{data['class_counts']}`.",
            f"- Ảnh bị tác động bởi candidate chưa giải quyết: {data['n_risk_images']:,}; số cặp candidate xuyên split Phase 0.2 còn unresolved: {data['n_unresolved_cross_split_candidates']:,}.",
            f"- Split: `{data['phase051_split_path']}`; Phase 0.2 đối chiếu: `{data['phase02_split_path']}`; official accuracy eligible=`{data['official_accuracy_eligible']}`.", ""])
    lines.extend(["## Huấn luyện và đánh giá", "",
        "Macro-F1 là tiêu chí chính. Các chỉ số validation của tất cả cấu hình dùng cùng mẫu validation theo task; test chỉ có cho winner đã chọn theo validation.", "",
        "| Task | Feature branch | Model | CV Macro-F1 mean±SD | Val Accuracy | Val Precision macro | Val Recall macro | Val Macro-F1 | Phase 5 Val Macro-F1 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    phase5_by_key = {(r.get("task"), r.get("experiment"), r.get("model")): r for r in model_result.get("phase05_comparison", [])}
    for row in search_rows:
        old = phase5_by_key.get((row["task"], row["experiment"], row["model"]), {})
        old_value = f"{float(old['phase05_validation_macro_f1']):.4f}" if old.get("phase05_validation_macro_f1") else "n/a"
        lines.append(f"| {row['task']} | `{row['experiment']}` | {row['model']} | {float(row['cv_macro_f1_mean']):.4f}±{float(row['cv_macro_f1_std']):.4f} | {float(row['validation_accuracy']):.4f} | {float(row['validation_macro_precision']):.4f} | {float(row['validation_macro_recall']):.4f} | {float(row['validation_macro_f1']):.4f} | {old_value} |")
    old_registry_path = project_path(config, "reports/phase05_results.json")
    old_registry = json.loads(old_registry_path.read_text(encoding="utf-8")) if old_registry_path.is_file() else {}
    old_metrics_path = project_path(config, "reports/phase05_metrics.csv")
    old_metrics = _read_csv(old_metrics_path) if old_metrics_path.is_file() else []
    lines.extend(["", "### Đối chiếu model được chọn ở Phase 5 và Phase 5.1", "",
        "Phase 5.1 dùng group-aware split mới nên tập test không gồm cùng image_id với Phase 5. Các con số test dưới đây chỉ để đối chiếu mô tả, không phải so sánh paired trên cùng ảnh.", "",
        "| Task | Phase 5 feature/model | Phase 5 Val Macro-F1 | Phase 5 Test Accuracy/Macro-F1 | Phase 5.1 feature/model | Phase 5.1 Val Macro-F1 | Phase 5.1 Test Accuracy/Macro-F1 |",
        "|---|---|---:|---:|---|---:|---:|"])
    for task in ("ripeness", "defect"):
        old_choice = old_registry.get("selected_models", {}).get(task, {})
        old_val = next((r for r in old_metrics if r["task"] == task and r["experiment"] == "SELECTED" and r["scope"] == "overall" and r["stage"] == "validation"), {})
        old_test = next((r for r in old_metrics if r["task"] == task and r["experiment"] == "SELECTED" and r["scope"] == "overall" and r["stage"] == "test_locked"), {})
        new = model_result["summary"]["chosen_by_validation_only"][task]
        old_test_text = f"{float(old_test['accuracy']):.4f}/{float(old_test['macro_f1']):.4f}" if old_test else "n/a"
        new_test = new["test_locked_once"]
        lines.append(f"| {task} | `{old_choice.get('experiment', 'unknown')} / {old_choice.get('model', 'unknown')}` | {float(old_val.get('macro_f1', 0)):.4f} | {old_test_text} | `{new['experiment']} / {new['model']}` | {new['validation_macro_f1']:.4f} | {new_test['accuracy']:.4f}/{new_test['macro_f1']:.4f} |")
    lines.extend(["", "### Model khóa theo validation", ""])
    for task, winner in model_result["summary"]["chosen_by_validation_only"].items():
        test = winner["test_locked_once"]
        lines.extend([f"**{task}: `{winner['experiment']}` + `{winner['model']}`.** Validation accuracy={winner['validation_accuracy']:.4f}, macro precision/recall/F1={winner['validation_macro_precision']:.4f}/{winner['validation_macro_recall']:.4f}/{winner['validation_macro_f1']:.4f}; train-only group-CV macro-F1={winner['cv_macro_f1_mean']:.4f}±{winner['cv_macro_f1_std']:.4f}. Locked test (one evaluation): accuracy={test['accuracy']:.4f}, macro precision/recall/F1={test['macro_precision']:.4f}/{test['macro_recall']:.4f}/{test['macro_f1']:.4f}. Model: `{winner['model_path']}`.", ""])
    lines.extend(["### Gray World", "",
        "Ripeness R1 Original Color và R2 Gray World Color được so trên cùng validation. Defect G1–G4 (Gray World) được giữ riêng để so với D1–D4 (Original). Diễn giải chỉ dựa trên validation và group-CV; khác biệt chưa chứng minh Gray World gây cải thiện.", ""])
    for task in ("ripeness", "defect"):
        by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in search_rows:
            if row["task"] == task:
                by_exp[row["experiment"]].append(row)
        for exp, rows in by_exp.items():
            best = max(rows, key=lambda r: float(r["validation_macro_f1"]))
            lines.append(f"- `{exp}` best validation Macro-F1={float(best['validation_macro_f1']):.4f} ({best['model']}; CV={float(best['cv_macro_f1_mean']):.4f}±{float(best['cv_macro_f1_std']):.4f}).")
    paired = model_result.get("paired", [])
    stability = model_result.get("stability", [])
    lines.extend(["", "### Ảnh hưởng của K-Means trên defect validation", "",
        "D1 và D4 được tính trên cùng ảnh validation. Paired group-CV fold comparison dùng cùng split folds; sign của chênh lệch thể hiện D4 tốt hơn / hòa / kém hơn ở bao nhiêu folds.", "",
        "| Model | D1 Val Macro-F1 | D4 Val Macro-F1 | Δ D4−D1 | D4 sửa lỗi D1 | Lỗi mới ở D4 | CV Δ mean±SD | Folds D4 tốt/hòa/kém |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    stability_by_model = {r["model"]: r for r in stability}
    for row in paired:
        stab = stability_by_model[row["model"]]
        lines.append(f"| {row['model']} | {float(row['D1_macro_f1']):.4f} | {float(row['D4_macro_f1']):.4f} | {float(row['delta_D4_minus_D1']):+.4f} | {row['D4_corrected_D1_errors']} | {row['D4_new_errors']} | {float(stab['mean_paired_fold_delta_D4_minus_D1']):+.4f}±{float(stab['std_paired_fold_delta']):.4f} | {stab['folds_D4_better']}/{stab['folds_tied']}/{stab['folds_D4_worse']} |")
    lines.extend(["", "## Confusion matrices", ""])
    for task in ("ripeness", "defect"):
        winner = model_result["summary"]["chosen_by_validation_only"][task]
        for stage in ("validation", "test_locked"):
            if stage == "test_locked":
                exp, model = winner["experiment"], winner["model"]
            else:
                exp, model = winner["experiment"], winner["model"]
            cm = {(r["true_label"], r["predicted_label"]): int(r["count"])
                  for r in model_result["confusion_rows"] if r["task"] == task and r["experiment"] == exp
                  and r["model"] == model and r["stage"] == stage}
            class_names = LABELS[task]
            lines.extend([f"### {task} — {stage}", "", "| True \\ Predicted | " + " | ".join(class_names) + " |",
                "|---|" + "---:|" * len(class_names)])
            for truth in class_names:
                lines.append("| " + truth + " | " + " | ".join(str(cm.get((truth, pred), 0)) for pred in class_names) + " |")
            lines.append("")
    err_counts = Counter((r["task"], r["stage"]) for r in model_result["errors"])
    lines.extend(["## Kết luận và giới hạn", "",
        f"- Sai dự đoán validation (tất cả cấu hình) và test (chỉ winner) được lưu tại `reports/model_selection/phase051_misclassifications.csv`; counts: `{dict(err_counts)}`.",
        "- Phase 5 models/reports không bị ghi đè; Phase 5.1 search, metrics, confusion, errors và toàn bộ tuned pipelines nằm riêng trong `reports/model_selection/` và `models/classifiers/`.",
        "- Còn near-duplicate candidate chưa xác minh và augmentation/source lineage chưa đầy đủ. Vì vậy `official_accuracy_eligible=false`, toàn bộ kết quả Phase 5.1 vẫn exploratory; không tuyên bố benchmark chính thức.",
        "- Kết quả defect classification không xác minh ground-truth segmentation. Các feature K-Means có thể phản ánh khác biệt phân bố giữa ảnh classification và segmentation; không suy rộng metric segmentation sang classification.",
        "- Đề xuất: chọn cấu hình theo validation/CV ở trên để tiếp tục nghiên cứu; trước báo cáo accuracy chính thức cần giải quyết lineage/candidate còn rủi ro và xem xét cặp same-source có nhãn healthy/defective mâu thuẫn.",
        ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path



