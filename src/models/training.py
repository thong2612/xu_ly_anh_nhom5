from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
import posixpath
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.config import project_path

LOGGER = logging.getLogger("mangoresearch.phase5")
LABELS = {
    "ripeness": ["unripe", "partially_ripe", "ripe"],
    "defect": ["healthy", "defective"],
}
FEATURE_EXPERIMENTS = {
    "ripeness": {
        "R1_original_color": ["original_color"],
        "R2_gray_world_color": ["gray_world_color"],
    },
    "defect": {
        "D1_original_color": ["original_color"],
        "D2_original_color_plus_lab_k3": ["original_color", "original_lab_k3"],
        "D3_original_color_plus_hsv_k5": ["original_color", "original_hsv_k5"],
        "D4_original_color_plus_lab_k3_hsv_k5": ["original_color", "original_lab_k3", "original_hsv_k5"],
        "G1_gray_world_color": ["gray_world_color"],
        "G2_gray_world_color_plus_lab_k3": ["gray_world_color", "gray_world_lab_k3"],
        "G3_gray_world_color_plus_hsv_k5": ["gray_world_color", "gray_world_hsv_k5"],
        "G4_gray_world_color_plus_both_kmeans": ["gray_world_color", "gray_world_lab_k3", "gray_world_hsv_k5"],
    },
}
METRIC_FIELDS = (
    "task", "experiment", "model", "stage", "scope", "label", "n_samples",
    "accuracy", "precision", "recall", "f1", "macro_precision", "macro_recall",
    "macro_f1", "macro_f1_std", "false_negative_defective", "parameters_json",
)
SEARCH_FIELDS = (
    "task", "experiment", "feature_sets", "feature_count", "model", "best_parameters_json",
    "cv_macro_f1_mean", "cv_macro_f1_std", "cv_macro_precision_mean", "cv_macro_recall_mean",
    "validation_accuracy", "validation_macro_precision", "validation_macro_recall",
    "validation_macro_f1", "validation_missing_fraction", "feature_missing_fraction",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 5 input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            matrix[row_index, column_index] = _number(row.get(column, ""))
    return matrix


def _union_find_groups(rows: list[dict[str, str]], split_rows: dict[str, dict[str, str]]) -> list[str]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    tokens_per_row: list[list[str]] = []
    for row in rows:
        split_row = split_rows[row["image_id"]]
        tokens = []
        lineage = split_row.get("lineage_group_id", "").strip()
        sha = row.get("source_sha256", "").strip()
        if lineage:
            tokens.append("lineage:" + lineage)
        if sha:
            tokens.append("sha256:" + sha)
        if not tokens:
            tokens.append("image_id:" + row["image_id"])
        for token in tokens:
            find(token)
        for token in tokens[1:]:
            union(tokens[0], token)
        tokens_per_row.append(tokens)
    root_digests: dict[str, str] = {}
    groups = []
    for tokens in tokens_per_row:
        root = find(tokens[0])
        if root not in root_digests:
            component = sorted(key for key in parent if find(key) == root)
            root_digests[root] = hashlib.sha256("|".join(component).encode()).hexdigest()[:20]
        groups.append(root_digests[root])
    return groups


def _audit_group_overlap(rows: list[dict[str, str]], split_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    exact: dict[str, set[str]] = defaultdict(set)
    exact_counts: Counter[str] = Counter()
    lineage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_row = split_rows[row["image_id"]]
        split = split_row.get("phase02_split", "")
        sha = row.get("source_sha256", "").strip()
        group = split_row.get("lineage_group_id", "").strip()
        if sha:
            exact[sha].add(split)
            exact_counts[sha] += 1
        if group:
            lineage[group].add(split)
    exact_overlaps = [group for group, splits in exact.items() if len(splits) > 1]
    lineage_overlaps = [group for group, splits in lineage.items() if len(splits) > 1]
    path_to_split = {posixpath.normpath(row.get("image_path", "").replace("\\", "/")).casefold():
                     split_rows[row["image_id"]].get("phase02_split", "") for row in rows}
    near_pairs = 0
    near_cross_split = 0
    near_zero_cross_split = 0
    near_cross_ids: set[str] = set()
    near_path = Path("reports/near_duplicate_candidates.csv")
    if near_path.is_file():
        for candidate in _read_csv(near_path):
            path_a = posixpath.normpath(candidate.get("image_path_a", "").replace("\\", "/")).casefold()
            path_b = posixpath.normpath(candidate.get("image_path_b", "").replace("\\", "/")).casefold()
            if path_a not in path_to_split or path_b not in path_to_split:
                continue
            near_pairs += 1
            if path_to_split[path_a] != path_to_split[path_b]:
                near_cross_split += 1
                try:
                    near_zero_cross_split += int(int(candidate.get("dhash_hamming", "-1")) == 0)
                except ValueError:
                    pass
                near_cross_ids.update((path_a, path_b))
    return {
        "unique_exact_hashes": len(exact),
        "repeated_exact_hash_groups": sum(count > 1 for count in exact_counts.values()),
        "exact_cross_split_groups": len(exact_overlaps),
        "lineage_groups": len(lineage),
        "lineage_cross_split_groups": len(lineage_overlaps),
        "exact_or_lineage_cross_split": bool(exact_overlaps or lineage_overlaps),
        "near_duplicate_candidates_within_task": near_pairs,
        "near_duplicate_candidate_cross_split_pairs": near_cross_split,
        "near_duplicate_dhash_zero_cross_split_pairs": near_zero_cross_split,
        "unique_paths_in_cross_split_candidates": len(near_cross_ids),
        "near_duplicate_candidates_confirmed": False,
        "split_risk_status": sorted({row.get("split_risk_status", "") for row in rows}),
        "official_accuracy_eligible_values": sorted({row.get("official_accuracy_eligible", "") for row in rows}),
    }


def _load_task(config: dict[str, Any], schema: dict[str, Any], task: str) -> dict[str, Any]:
    features = config.get("features", {})
    task_config = features.get("tasks", {}).get(task, {})
    feature_path = project_path(config, f"{features.get('output_dir', 'data/features')}/{task}_features.csv")
    split_path = project_path(config, task_config["split_path"])
    rows, split_list = _read_csv(feature_path), _read_csv(split_path)
    feature_ids = [row.get("image_id", "") for row in rows]
    split_ids = [row.get("image_id", "") for row in split_list]
    if len(feature_ids) != len(set(feature_ids)) or set(feature_ids) != set(split_ids):
        raise ValueError(f"Phase 4 features and canonical split image IDs differ for {task}")
    split_by_id = {row["image_id"]: row for row in split_list}
    label_field = "ripeness_label" if task == "ripeness" else "defect_label"
    for row in rows:
        split_row = split_by_id[row["image_id"]]
        if row.get("phase02_split") != split_row.get("phase02_split"):
            raise ValueError(f"Phase 4 split differs from Phase 0.2 for {row['image_id']}")
        if row.get("label") != split_row.get(label_field):
            raise ValueError(f"Phase 4 target differs from Phase 0.2 for {row['image_id']}")
        if row.get("official_accuracy_eligible", "").casefold() not in {"false", "0", ""}:
            raise ValueError(f"Unexpected official-accuracy flag for {row['image_id']}")
    y = np.asarray([row["label"] for row in rows], dtype=object)
    if set(y.tolist()) != set(LABELS[task]):
        raise ValueError(f"Unexpected labels for {task}: {sorted(set(y.tolist()))}")
    metadata = set(schema.get("metadata_columns", [])) | {"label", "phase02_split"}
    if metadata.intersection(schema.get("feature_columns", {}).get(task, [])):
        raise ValueError(f"Metadata leakage: schema includes metadata in X for {task}")
    feature_sets = schema["feature_sets"][task]
    experiments = FEATURE_EXPERIMENTS[task]
    for name, selected_sets in experiments.items():
        if not set(selected_sets).issubset(feature_sets):
            raise ValueError(f"Feature group {name} refers to absent schema feature set")
    split_names = np.asarray([split_by_id[row["image_id"]]["phase02_split"] for row in rows])
    groups = np.asarray(_union_find_groups(rows, split_by_id))
    return {
        "rows": rows, "split_by_id": split_by_id, "labels": y, "groups": groups,
        "split_names": split_names, "feature_sets": feature_sets, "experiments": experiments,
        "group_audit": _audit_group_overlap(rows, split_by_id),
    }


def _make_pipeline(model_name: str, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    ))]
    if model_name in {"KNN", "SVM_RBF"}:
        steps.append(("scaler", StandardScaler()))
    if model_name == "KNN":
        classifier = KNeighborsClassifier()
    elif model_name == "SVM_RBF":
        classifier = SVC(kernel="rbf", random_state=seed)
    elif model_name == "RandomForest":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _param_grid(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    settings = config.get("classifier_training", {})
    if model_name == "KNN":
        return {"classifier__n_neighbors": list(settings.get("knn_neighbors", [3, 5, 7, 11])),
                "classifier__weights": list(settings.get("knn_weights", ["uniform", "distance"]))}
    if model_name == "SVM_RBF":
        return {"classifier__C": list(settings.get("svm_C", [0.1, 1, 10, 100])),
                "classifier__gamma": list(settings.get("svm_gamma", ["scale"]))}
    return {
        "classifier__n_estimators": list(settings.get("rf_n_estimators", [100, 200])),
        "classifier__max_depth": list(settings.get("rf_max_depth", [None, 20])),
        "classifier__min_samples_leaf": list(settings.get("rf_min_samples_leaf", [1, 2])),
    }


def _folds(X: np.ndarray, y: np.ndarray, groups: np.ndarray, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(count, len(set(groups.tolist())))
    if n_splits < 2:
        raise ValueError("Not enough independent groups for cross-validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))
    for train_index, validation_index in folds:
        if set(groups[train_index]) & set(groups[validation_index]):
            raise AssertionError("Group leakage detected inside a CV fold")
    return folds


def _matrix_for_experiment(
    rows: list[dict[str, Any]], schema_sets: dict[str, list[str]], group_names: list[str],
) -> tuple[list[str], np.ndarray]:
    columns = [column for group_name in group_names for column in schema_sets[group_name]]
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate feature columns in feature group {group_names}")
    return columns, _feature_matrix(rows, columns)


def _missing_fraction(X: np.ndarray) -> float:
    return float(np.isnan(X).sum() / X.size) if X.size else 0.0


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fn_defective = ""
    if "defective" in labels:
        index = labels.index("defective")
        fn_defective = int(matrix[index].sum() - matrix[index, index])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)), "precision": precision, "recall": recall,
        "f1": f1, "support": support, "confusion": matrix,
        "false_negative_defective": fn_defective,
    }


def _add_metric_rows(
    output: list[dict[str, Any]], task: str, experiment: str, model: str, stage: str,
    payload: dict[str, Any], labels: list[str], params_json: str = "", macro_f1_std: Any = "",
) -> None:
    output.append({
        "task": task, "experiment": experiment, "model": model, "stage": stage,
        "scope": "overall", "label": "", "n_samples": int(sum(payload["support"])),
        "accuracy": payload["accuracy"], "precision": "", "recall": "", "f1": "",
        "macro_precision": payload["macro_precision"], "macro_recall": payload["macro_recall"],
        "macro_f1": payload["macro_f1"], "macro_f1_std": macro_f1_std,
        "false_negative_defective": payload["false_negative_defective"], "parameters_json": params_json,
    })
    for index, label in enumerate(labels):
        output.append({
            "task": task, "experiment": experiment, "model": model, "stage": stage,
            "scope": "per_class", "label": label, "n_samples": int(payload["support"][index]),
            "accuracy": "", "precision": float(payload["precision"][index]),
            "recall": float(payload["recall"][index]), "f1": float(payload["f1"][index]),
            "macro_precision": "", "macro_recall": "", "macro_f1": "", "macro_f1_std": "",
            "false_negative_defective": "", "parameters_json": params_json,
        })


def _add_confusion_rows(
    output: list[dict[str, Any]], task: str, experiment: str, model: str, stage: str,
    payload: dict[str, Any], labels: list[str],
) -> None:
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            output.append({"task": task, "experiment": experiment, "model": model,
                           "stage": stage, "true_label": true_label,
                           "predicted_label": predicted_label, "count": int(payload["confusion"][i, j])})


def _learning_curve(
    task: str, experiment: str, model: str, estimator: Pipeline, X: np.ndarray,
    y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], seed: int, report_dir: Path,
) -> list[dict[str, Any]]:
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    sizes, train_scores, validation_scores = learning_curve(
        clone(estimator), X, y, train_sizes=fractions, cv=folds, scoring="f1_macro",
        n_jobs=1, shuffle=True, random_state=seed,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    validation_mean, validation_std = validation_scores.mean(axis=1), validation_scores.std(axis=1)
    output = [{
        "task": task, "experiment": experiment, "model": model,
        "train_samples_per_fold": int(size), "requested_fraction": fraction,
        "train_macro_f1_mean": float(tm), "train_macro_f1_std": float(ts),
        "cv_macro_f1_mean": float(vm), "cv_macro_f1_std": float(vs),
    } for size, fraction, tm, ts, vm, vs in zip(sizes, fractions, train_mean, train_std, validation_mean, validation_std)]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(sizes, train_mean, marker="o", label="Train CV fold")
    axis.fill_between(sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
    axis.plot(sizes, validation_mean, marker="o", label="Held-out CV fold")
    axis.fill_between(sizes, validation_mean-validation_std, validation_mean+validation_std, alpha=0.15)
    axis.set(title=f"Learning curve — {task}: {experiment} / {model}",
             xlabel="Training samples per CV fold", ylabel="Macro F1", ylim=(0, 1.02))
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(report_dir / f"phase05_learning_curve_{task}.png", dpi=150)
    plt.close(fig)
    return output


def _correlation_audit(X: np.ndarray, columns: list[str], task: str) -> list[dict[str, Any]]:
    output = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            valid = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if valid.sum() < 10:
                continue
            left, right = X[valid, i], X[valid, j]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            corr = float(np.corrcoef(left, right)[0, 1])
            if math.isfinite(corr) and abs(corr) >= 0.95:
                output.append({"task": task, "feature_1": columns[i], "feature_2": columns[j],
                               "pearson_r": corr, "n_pairwise": int(valid.sum())})
    output.sort(key=lambda row: abs(row["pearson_r"]), reverse=True)
    return output


def _plot_importance(task: str, rows: list[dict[str, Any]], report_dir: Path) -> None:
    top = rows[:20]
    if not top:
        return
    fig, axis = plt.subplots(figsize=(9, max(4.5, len(top) * 0.28)))
    labels = [row["feature"] for row in reversed(top)]
    values = [float(row["permutation_importance_mean"]) for row in reversed(top)]
    errors = [float(row["permutation_importance_std"]) for row in reversed(top)]
    axis.barh(labels, values, xerr=errors, color="#5589a6", alpha=0.9)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set(title=f"Validation permutation importance — {task}", xlabel="Macro F1 decrease after permutation")
    fig.tight_layout(); fig.savefig(report_dir / f"phase05_feature_importance_{task}.png", dpi=150); plt.close(fig)


def _confusion_markdown(rows: list[dict[str, Any]], task: str, stage: str, labels: list[str]) -> list[str]:
    table = {(row["true_label"], row["predicted_label"]): int(row["count"])
             for row in rows if row["task"] == task and row["stage"] == stage
             and row.get("experiment") == "SELECTED"}
    if not table:
        return ["(Không có ma trận ở giai đoạn này.)"]
    lines = ["| Thực tế \\ Dự đoán | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for true_label in labels:
        lines.append("| " + true_label + " | " + " | ".join(
            str(table.get((true_label, predicted), 0)) for predicted in labels) + " |")
    return lines


def _write_report(
    path: Path, task_data: dict[str, dict[str, Any]], search_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]], confusion_rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]], importance_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]], seed: int, requested_folds: int,
) -> None:
    lines = [
        "# MangoResearch — Phase 5: huấn luyện và đánh giá Machine Learning", "",
        "## Thiết kế và kiểm soát rò rỉ", "",
        f"Dùng feature tables Phase 4, seed={seed}, StratifiedGroupKFold tối đa {requested_folds} folds. Imputer fit bên trong CV Pipeline; KNN/SVM có StandardScaler trong Pipeline, Random Forest chỉ impute. Tuning theo macro F1 chỉ trên train/CV; validation chọn feature group và classifier; test được đánh giá một lần sau khi khóa lựa chọn theo validation.",
        "Grid đã thử: KNN k={3,5,7,11}, weights={uniform,distance}; SVM-RBF C={0.1,1,10,100}, gamma=scale; Random Forest n_estimators={100,200}, max_depth={None,20}, min_samples_leaf={1,2}. Chọn tham số tốt nhất theo mean CV macro F1.",
        "Chỉ cột trong `feature_schema.json` được đưa vào X. Label là y; split, image_id, path, dataset/source, cờ chất lượng và mask/ground truth không phải feature. Missing values được median-impute bên trong từng train fold, kèm missing indicators cho cột từng thiếu ở train; cột rỗng hoàn toàn được giữ ổn định.",
    ]
    for task, data in task_data.items():
        counts = Counter(row["label"] for row in data["rows"])
        splits = Counter(data["split_names"].tolist())
        audit = data["group_audit"]
        lines.extend([
            "", f"### Dữ liệu {task}", "",
            f"- N={len(data['rows']):,}; nhãn `{dict(counts)}`; split giữ nguyên `{dict(splits)}`.",
            f"- Group CV: {data['train_group_count']:,} training groups; exact-hash groups lặp qua split={audit['exact_cross_split_groups']}; lineage group overlap qua split={audit['lineage_cross_split_groups']}; official accuracy eligible=`{audit['official_accuracy_eligible_values']}`.",
            f"- dHash candidate pairs trong task={audit['near_duplicate_candidates_within_task']:,}; cross-split={audit['near_duplicate_candidate_cross_split_pairs']:,}, trong đó dHash=0 có {audit['near_duplicate_dhash_zero_cross_split_pairs']:,} cặp. Đây là ứng viên chưa được xác nhận bằng nhận diện/duyệt ảnh, nhưng đủ để giữ kết quả ở mức exploratory; không dùng các metric này làm kết luận chính cho tới khi audit lineage/near-duplicate và tạo split theo nhóm đáng tin cậy.",
            "", "| Nhóm feature | Model | CV macro F1 mean±std | Val accuracy | Val macro P/R/F1 | Train−Val F1 | Missing cells trên toàn bộ mẫu |", "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in [item for item in search_rows if item["task"] == task]:
            train_row = next(m for m in metric_rows if m["task"] == task and m["experiment"] == row["experiment"]
                             and m["model"] == row["model"] and m["stage"] == "train" and m["scope"] == "overall")
            gap = float(train_row["macro_f1"]) - float(row["validation_macro_f1"])
            lines.append(
                f"| `{row['experiment']}` | {row['model']} | {float(row['cv_macro_f1_mean']):.3f}±{float(row['cv_macro_f1_std']):.3f} | {float(row['validation_accuracy']):.3f} | {float(row['validation_macro_precision']):.3f}/{float(row['validation_macro_recall']):.3f}/{float(row['validation_macro_f1']):.3f} | {gap:+.3f} | {float(row['feature_missing_fraction']):.1%} |"
            )
        chosen = selected[task]
        validation_metric = next(m for m in metric_rows if m["task"] == task and m["experiment"] == "SELECTED"
                                 and m["stage"] == "validation" and m["scope"] == "overall")
        test_metric = next((m for m in metric_rows if m["task"] == task and m["stage"] == "test_locked"
                            and m["scope"] == "overall"), None)
        lines.extend([
            "", f"**Model khóa theo validation:** `{chosen['experiment']}` + `{chosen['model']}`; validation macro F1={chosen['validation_macro_f1']:.3f}, accuracy={chosen['validation_accuracy']:.3f}.",
            "", "Confusion matrix validation:", "", *_confusion_markdown(confusion_rows, task, "validation", LABELS[task]), "",
        ])
        selected_class_rows = [row for row in metric_rows if row["task"] == task
                               and row["experiment"] == "SELECTED" and row["scope"] == "per_class"
                               and row["stage"] in {"validation", "test_locked"}]
        if selected_class_rows:
            lines.extend(["| Split | Lớp | Support | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|"])
            for row in selected_class_rows:
                lines.append(f"| {row['stage']} | {row['label']} | {row['n_samples']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} |")
            lines.append("")
        if test_metric:
            lines.extend([
                "Confusion matrix test (provisional; model đã khóa trước khi tính):", "",
                *_confusion_markdown(confusion_rows, task, "test_locked", LABELS[task]), "",
                f"Test n={test_metric['n_samples']}; accuracy={float(test_metric['accuracy']):.3f}; macro P/R/F1={float(test_metric['macro_precision']):.3f}/{float(test_metric['macro_recall']):.3f}/{float(test_metric['macro_f1']):.3f}.",
            ])
            if task == "defect":
                lines.append(f"False Negative của lớp defective trên test: {test_metric['false_negative_defective']}.")
        if task == "defect":
            best_by_group: dict[str, float] = {}
            for row in search_rows:
                if row["task"] == task:
                    best_by_group[row["experiment"]] = max(best_by_group.get(row["experiment"], -1),
                                                              float(row["validation_macro_f1"]))
            baseline = best_by_group["D1_original_color"]
            lines.extend(["", "Tác động mô tả trên cùng validation 541 ảnh:", ""])
            for group in ("D2_original_color_plus_lab_k3", "D3_original_color_plus_hsv_k5",
                          "D4_original_color_plus_lab_k3_hsv_k5"):
                lines.append(f"- `{group}`: best val macro F1={best_by_group[group]:.3f}, Δ so với D1={best_by_group[group]-baseline:+.3f}.")
            for group in ("G1_gray_world_color", "G2_gray_world_color_plus_lab_k3",
                          "G3_gray_world_color_plus_hsv_k5", "G4_gray_world_color_plus_both_kmeans"):
                lines.append(f"- `{group}`: best val macro F1={best_by_group[group]:.3f}; bảng trên thể hiện missingness.")
            lines.append("So sánh nhiều cấu hình trên validation có selection optimism; không chứng minh K-Means hoặc Gray World gây cải thiện.")
    lines.extend(["", "## Learning curves", ""])
    for task in task_data:
        lines.extend([f"![Learning curve {task}](phase05_learning_curve_{task}.png)", ""])
    lines.extend(["Train/CV only; test không tham gia learning curve. Dữ liệu số nằm ở `reports/phase05_learning_curves.csv`.",
                  "", "## Random Forest permutation importance và feature correlation", ""])
    for task in task_data:
        task_rows = sorted((row for row in importance_rows if row["task"] == task),
                           key=lambda row: float(row["permutation_importance_mean"]), reverse=True)
        lines.extend([f"### {task}", ""])
        if task_rows:
            lines.extend(["| Feature | Mean macro-F1 decrease | Std |", "|---|---:|---:|"])
            for row in task_rows[:10]:
                lines.append(f"| `{row['feature']}` | {float(row['permutation_importance_mean']):.5f} | {float(row['permutation_importance_std']):.5f} |")
            lines.extend(["", f"![Permutation importance {task}](phase05_feature_importance_{task}.png)"])
        corr = [row for row in correlations if row["task"] == task]
        lines.extend(["", f"Cặp feature trong group đã chọn có |Pearson r|≥0.95 trên train: {len(corr)} (chi tiết: `reports/phase05_feature_correlations.csv`). Importance không chứng minh quan hệ nhân quả và có thể bị chia sẻ giữa feature tương quan.", ""])
    lines.extend([
        "## Lỗi, giới hạn và kết luận", "",
        "- Các lỗi trên validation của mọi cấu hình và lỗi test chỉ của model đã khóa được lưu trong `reports/phase05_misclassifications.csv`; metric từng lớp và confusion matrix ở các CSV riêng.",
        "- Feature phụ thuộc fruit mask heuristic Phase 2; missingness đặc biệt cao ở nhánh Gray World của defect. Không phát hiện vùng nghi vấn (0) được giữ khác với feature unavailable (blank).",
        "- Dataset A lineage ảnh gốc chưa được ghi nhận; Dataset C còn augmentation lineage chưa xác định hoàn toàn. Dù exact-hash và known lineage group không overlap giữa split, audit có dHash near-duplicate candidates băng qua các split. Các candidate chưa được xác minh là cùng ảnh, song số lượng đáng kể khiến toàn bộ metric hiện tại chỉ để debug/exploratory; không dùng làm bằng chứng chính hay benchmark cho đến khi phân xử và tạo split nhóm mới.",
        "- `official_accuracy_eligible=false` không bị thay đổi. Không tuyên bố độ chính xác thực tế ngoài miền thu thập.",
        "- Cấu hình đứng đầu validation chỉ là checkpoint nội bộ của thí nghiệm; chưa đưa ra khuyến nghị triển khai hoặc kết luận mô hình tốt nhất cho tới khi candidate near-duplicate xuyên split được duyệt và split theo nhóm được xử lý lại.",
        "", "## Artifact", "",
        "Pipeline đầy đủ và model registry ở `models/phase05/`; kết quả grid search `reports/phase05_search_results.csv`; metric `reports/phase05_metrics.csv`; confusion `reports/phase05_confusion_matrices.csv`; curve `reports/phase05_learning_curves.csv`; importance `reports/phase05_feature_importance.csv`; correlation `reports/phase05_feature_correlations.csv`; lỗi `reports/phase05_misclassifications.csv`; JSON `reports/phase05_results.json`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_phase5_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"pipeline", "feature_columns", "task", "labels", "model_name", "experiment"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Invalid Phase 5 model artifact: {path}")
    return artifact


def predict_from_features(model_path: str | Path, feature_values: dict[str, Any]) -> dict[str, Any]:
    artifact = load_phase5_model(model_path)
    columns = artifact["feature_columns"]
    matrix = np.asarray([[_number(feature_values.get(name, "")) for name in columns]], dtype=np.float64)
    prediction = str(artifact["pipeline"].predict(matrix)[0])
    output = {"task": artifact["task"], "experiment": artifact["experiment"],
              "model": artifact["model_name"], "prediction": prediction}
    if hasattr(artifact["pipeline"], "predict_proba"):
        probabilities = artifact["pipeline"].predict_proba(matrix)[0]
        classifier_classes = artifact["pipeline"].named_steps["classifier"].classes_
        output["class_scores"] = {str(label): float(score) for label, score in zip(classifier_classes, probabilities)}
    return output


def run_phase5(config: dict[str, Any], tasks: list[str] | None = None) -> Path:
    classifier_config = dict(config.get("classifier_training", {}))
    if not classifier_config:
        raise ValueError("Missing classifier_training configuration in config.yaml")
    selected_tasks = tasks or ["ripeness", "defect"]
    if not set(selected_tasks).issubset(LABELS):
        raise ValueError(f"Unsupported Phase 5 tasks: {selected_tasks}")
    schema_path = project_path(config, config.get("features", {}).get("schema_path", "data/features/feature_schema.json"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    seed = int(classifier_config.get("seed", 42))
    requested_folds = int(classifier_config.get("cv_folds", 5))
    models = ("KNN", "SVM_RBF", "RandomForest")
    report_dir = project_path(config, classifier_config.get("report_dir", "reports"))
    model_dir = project_path(config, classifier_config.get("model_dir", "models/classifiers/baseline"))
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    task_data: dict[str, dict[str, Any]] = {}
    search_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    selected_models: dict[str, dict[str, Any]] = {}
    registry: dict[str, Any] = {
        "experiment_version": "phase5-classical-v1", "seed": seed,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "models": {},
    }
    scoring = {"accuracy": "accuracy", "macro_precision": "precision_macro",
               "macro_recall": "recall_macro", "macro_f1": "f1_macro"}

    for task in selected_tasks:
        data = _load_task(config, schema, task)
        rows, y_all, split_names, groups = data["rows"], data["labels"], data["split_names"], data["groups"]
        train_indices = np.flatnonzero(split_names == "train")
        validation_indices = np.flatnonzero(np.isin(split_names, ("validation", "val")))
        test_indices = np.flatnonzero(split_names == "test")
        if not (len(train_indices) and len(validation_indices) and len(test_indices)):
            raise ValueError(f"Expected non-empty train/validation/test for {task}")
        y_train, y_validation = y_all[train_indices], y_all[validation_indices]
        train_groups = groups[train_indices]
        folds = _folds(np.empty((len(train_indices), 1)), y_train, train_groups, requested_folds, seed)
        data.update({
            "train_indices": train_indices, "validation_indices": validation_indices,
            "test_indices": test_indices, "train_group_count": len(set(train_groups.tolist())),
            "folds": folds,
        })
        task_data[task] = data
        candidates: list[dict[str, Any]] = []

        for experiment, feature_group_names in data["experiments"].items():
            columns, X_all = _matrix_for_experiment(rows, data["feature_sets"], feature_group_names)
            X_train, X_validation, X_test = X_all[train_indices], X_all[validation_indices], X_all[test_indices]
            y_test = y_all[test_indices]
            for model_name in models:
                LOGGER.info("Phase 5 tuning %s / %s / %s (%d feature columns)",
                            task, experiment, model_name, len(columns))
                search = GridSearchCV(
                    _make_pipeline(model_name, seed), _param_grid(model_name, config),
                    scoring=scoring, refit="macro_f1", cv=folds,
                    n_jobs=int(classifier_config.get("n_jobs", 1)), return_train_score=True,
                    error_score="raise", verbose=0,
                )
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
                best_index = int(search.best_index_)
                best_params_json = json.dumps(search.best_params_, sort_keys=True, default=str)
                cv_mean = {name: float(search.cv_results_[f"mean_test_{name}"][best_index])
                           for name in scoring}
                train_cv_mean = {name: float(search.cv_results_[f"mean_train_{name}"][best_index])
                                 for name in scoring}
                cv_std = float(search.cv_results_["std_test_macro_f1"][best_index])
                for stage, values in (("cv_train_fold_mean", train_cv_mean),
                                      ("cv_validation_fold_mean", cv_mean)):
                    metric_rows.append({
                        "task": task, "experiment": experiment, "model": model_name, "stage": stage,
                        "scope": "overall", "label": "", "n_samples": len(train_indices),
                        "accuracy": values["accuracy"], "precision": "", "recall": "", "f1": "",
                        "macro_precision": values["macro_precision"], "macro_recall": values["macro_recall"],
                        "macro_f1": values["macro_f1"],
                        "macro_f1_std": cv_std if stage == "cv_validation_fold_mean" else "",
                        "false_negative_defective": "", "parameters_json": best_params_json,
                    })

                # Pooled out-of-fold predictions reuse the exact stratified group folds.
                y_oof = cross_val_predict(estimator, X_train, y_train, cv=folds,
                                          method="predict", n_jobs=1)
                y_train_pred = estimator.predict(X_train)
                y_validation_pred = estimator.predict(X_validation)
                payloads = {
                    "train": _metric_payload(y_train, y_train_pred, LABELS[task]),
                    "cv_oof": _metric_payload(y_train, y_oof, LABELS[task]),
                    "validation": _metric_payload(y_validation, y_validation_pred, LABELS[task]),
                }
                for stage, payload in payloads.items():
                    _add_metric_rows(metric_rows, task, experiment, model_name, stage, payload,
                                     LABELS[task], best_params_json,
                                     cv_std if stage == "cv_oof" else "")
                    _add_confusion_rows(confusion_rows, task, experiment, model_name, stage,
                                        payload, LABELS[task])
                # Preserve all validation errors for all compared configurations.
                for row_index, prediction in zip(validation_indices, y_validation_pred):
                    true_label = str(y_all[row_index])
                    if true_label != str(prediction):
                        error_rows.append({"task": task, "stage": "validation", "image_id": rows[row_index]["image_id"],
                                           "true_label": true_label, "predicted_label": str(prediction),
                                           "model": model_name, "experiment": experiment})
                validation_metric = payloads["validation"]
                defect_recall = (float(validation_metric["recall"][LABELS[task].index("defective")])
                                 if task == "defect" else 0.0)
                missing_fraction = _missing_fraction(X_all)
                validation_missing = _missing_fraction(X_validation)
                candidate = {
                    "task": task, "experiment": experiment, "feature_sets": feature_group_names,
                    "columns": columns, "model": model_name, "estimator": estimator,
                    "best_params": search.best_params_, "cv_metric": cv_mean,
                    "cv_train_metric": train_cv_mean, "cv_std": cv_std,
                    "validation_payload": validation_metric,
                    "validation_macro_f1": validation_metric["macro_f1"],
                    "validation_defective_recall": defect_recall,
                    "validation_accuracy": validation_metric["accuracy"],
                    "feature_missing_fraction": missing_fraction,
                    "validation_missing_fraction": validation_missing,
                    "X_all": X_all, "X_train": X_train, "X_validation": X_validation,
                    "X_test": X_test, "y_train": y_train, "y_validation": y_validation,
                    "y_test": y_test,
                }
                candidates.append(candidate)
                search_rows.append({
                    "task": task, "experiment": experiment,
                    "feature_sets": ", ".join(feature_group_names), "feature_count": len(columns),
                    "model": model_name, "best_parameters_json": best_params_json,
                    "cv_macro_f1_mean": cv_mean["macro_f1"], "cv_macro_f1_std": cv_std,
                    "cv_macro_precision_mean": cv_mean["macro_precision"],
                    "cv_macro_recall_mean": cv_mean["macro_recall"],
                    "validation_accuracy": validation_metric["accuracy"],
                    "validation_macro_precision": validation_metric["macro_precision"],
                    "validation_macro_recall": validation_metric["macro_recall"],
                    "validation_macro_f1": validation_metric["macro_f1"],
                    "validation_missing_fraction": validation_missing,
                    "feature_missing_fraction": missing_fraction,
                })
                LOGGER.info("Phase 5 %s %s %s: CV macro F1 %.4f±%.4f; validation %.4f",
                            task, experiment, model_name, cv_mean["macro_f1"], cv_std,
                            validation_metric["macro_f1"])

        # Validation alone selects the winning feature group and model.
        winner = sorted(candidates, key=lambda item: (
            -item["validation_macro_f1"], -item["validation_defective_recall"],
            len(item["columns"]), item["experiment"], item["model"],
        ))[0]
        data["winner"] = winner
        selected_models[task] = {
            "experiment": winner["experiment"], "model": winner["model"],
            "validation_macro_f1": winner["validation_macro_f1"],
            "validation_accuracy": winner["validation_accuracy"],
        }
        _add_metric_rows(metric_rows, task, "SELECTED", winner["model"], "validation",
                         winner["validation_payload"], LABELS[task],
                         json.dumps(winner["best_params"], sort_keys=True, default=str), winner["cv_std"])
        _add_confusion_rows(confusion_rows, task, "SELECTED", winner["model"], "validation",
                            winner["validation_payload"], LABELS[task])

        # Touch the test split only after the validation selection is fixed.
        test_prediction = winner["estimator"].predict(winner["X_test"])
        test_payload = _metric_payload(winner["y_test"], test_prediction, LABELS[task])
        best_params_json = json.dumps(winner["best_params"], sort_keys=True, default=str)
        _add_metric_rows(metric_rows, task, "SELECTED", winner["model"], "test_locked",
                         test_payload, LABELS[task], best_params_json)
        _add_confusion_rows(confusion_rows, task, "SELECTED", winner["model"], "test_locked",
                            test_payload, LABELS[task])
        for row_index, prediction in zip(data["test_indices"], test_prediction):
            true_label = str(y_all[row_index])
            if true_label != str(prediction):
                error_rows.append({"task": task, "stage": "test_locked", "image_id": rows[row_index]["image_id"],
                                   "true_label": true_label, "predicted_label": str(prediction),
                                   "model": winner["model"], "experiment": winner["experiment"]})

        artifact = {
            "artifact_version": 1, "experiment_version": "phase5-classical-v1",
            "task": task, "experiment": winner["experiment"],
            "model_name": winner["model"], "pipeline": winner["estimator"],
            "feature_columns": winner["columns"], "feature_sets": winner["feature_sets"],
            "labels": LABELS[task], "label_mapping": {label: i for i, label in enumerate(LABELS[task])},
            "seed": seed, "hyperparameters": winner["best_params"],
            "validation_metrics": {name: winner["validation_payload"][name] for name in
                                   ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
            "test_metrics_provisional": {name: test_payload[name] for name in
                                          ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
            "experiment_config": {
                "seed": seed, "cv_folds_requested": requested_folds,
                "cv_strategy": "StratifiedGroupKFold on train only",
                "selection_metric": "macro_f1", "feature_selection_split": "validation",
                "evaluation_test_policy": "evaluate once after validation selection is locked",
                "hyperparameter_grid": _param_grid(winner["model"], config),
                "selected_feature_sets": winner["feature_sets"],
                "imputer": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)",
                "scaler": "StandardScaler" if winner["model"] in {"KNN", "SVM_RBF"} else None,
                "phase5_config": classifier_config,
            },
            "caveat": "Provisional: official_accuracy_eligible is false; residual lineage/near-duplicate risks remain.",
        }
        artifact_path = model_dir / f"{task}_best_pipeline.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        load_phase5_model(artifact_path)
        check_count = min(5, len(winner["X_validation"]))
        for index in range(check_count):
            source_row = rows[data["validation_indices"][index]]
            expected = str(winner["estimator"].predict(winner["X_validation"][index:index + 1])[0])
            loaded = predict_from_features(artifact_path, source_row)["prediction"]
            if expected != loaded:
                raise AssertionError(f"Save/load prediction mismatch for {task}/{source_row['image_id']}")
        registry["models"][task] = {
            "artifact_path": artifact_path.as_posix(), "experiment": winner["experiment"],
            "model_name": winner["model"], "feature_columns": winner["columns"],
            "label_mapping": artifact["label_mapping"], "save_load_consistency_samples": check_count,
            "validation_metrics": artifact["validation_metrics"],
            "test_metrics_provisional": artifact["test_metrics_provisional"],
        }

        # Feature importance is descriptive; choose the best RF by validation, not test.
        rf_winner = max((item for item in candidates if item["model"] == "RandomForest"),
                        key=lambda item: item["validation_macro_f1"])
        importance = permutation_importance(
            rf_winner["estimator"], rf_winner["X_validation"], rf_winner["y_validation"],
            scoring="f1_macro", n_repeats=int(classifier_config.get("permutation_repeats", 8)),
            random_state=seed, n_jobs=1,
        )
        task_importance = [{
            "task": task, "experiment": rf_winner["experiment"], "model": "RandomForest",
            "feature": feature, "permutation_importance_mean": float(mean),
            "permutation_importance_std": float(std),
        } for feature, mean, std in zip(rf_winner["columns"], importance.importances_mean,
                                        importance.importances_std)]
        task_importance.sort(key=lambda row: row["permutation_importance_mean"], reverse=True)
        importance_rows.extend(task_importance)
        _plot_importance(task, task_importance, report_dir)
        correlation_rows.extend(_correlation_audit(winner["X_train"], winner["columns"], task))
        learning_rows.extend(_learning_curve(
            task, winner["experiment"], winner["model"], winner["estimator"],
            winner["X_train"], winner["y_train"], folds, seed, report_dir,
        ))
        LOGGER.info("Phase 5 %s selected %s/%s; validation macro F1 %.4f, locked test %.4f; save/load verified %d samples",
                    task, winner["experiment"], winner["model"], winner["validation_macro_f1"],
                    test_payload["macro_f1"], check_count)

    _atomic_csv(report_dir / "phase05_search_results.csv", search_rows, SEARCH_FIELDS)
    _atomic_csv(report_dir / "phase05_metrics.csv", metric_rows, METRIC_FIELDS)
    _atomic_csv(report_dir / "phase05_confusion_matrices.csv", confusion_rows,
                ("task", "experiment", "model", "stage", "true_label", "predicted_label", "count"))
    _atomic_csv(report_dir / "phase05_learning_curves.csv", learning_rows,
                ("task", "experiment", "model", "train_samples_per_fold", "requested_fraction",
                 "train_macro_f1_mean", "train_macro_f1_std", "cv_macro_f1_mean", "cv_macro_f1_std"))
    _atomic_csv(report_dir / "phase05_feature_importance.csv", importance_rows,
                ("task", "experiment", "model", "feature", "permutation_importance_mean", "permutation_importance_std"))
    _atomic_csv(report_dir / "phase05_feature_correlations.csv", correlation_rows,
                ("task", "feature_1", "feature_2", "pearson_r", "n_pairwise"))
    _atomic_csv(report_dir / "phase05_misclassifications.csv", error_rows,
                ("task", "stage", "image_id", "true_label", "predicted_label", "model", "experiment"))
    registry_path = model_dir / "phase05_model_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {task: data["group_audit"] | {
        "n_samples": len(data["rows"]), "split_counts": dict(Counter(data["split_names"].tolist())),
        "train_groups_for_group_cv": data["train_group_count"], "cv_group_overlap": False,
    } for task, data in task_data.items()}
    results = {
        "seed": seed, "requested_cv_folds": requested_folds,
        "selected_models": selected_models, "leakage_audit": audit,
        "feature_experiments": FEATURE_EXPERIMENTS,
        "test_metrics_are_provisional": True,
        "model_artifacts": {task: registry["models"][task]["artifact_path"] for task in selected_tasks},
    }
    (report_dir / "phase05_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = project_path(config, classifier_config.get("report_path", "reports/phase05_report.md"))
    _write_report(report_path, task_data, search_rows, metric_rows, confusion_rows,
                  selected_models, importance_rows, correlation_rows, seed, requested_folds)
    LOGGER.info("Phase 5 complete: %s", report_path)
    return report_path
