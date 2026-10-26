from __future__ import annotations

import joblib
import numpy as np
from src.models.training import (
    _audit_group_overlap,
    _feature_matrix,
    _folds,
    _make_pipeline,
    _metric_payload,
    _union_find_groups,
    predict_from_features,
)


def test_feature_matrix_uses_only_explicit_schema_columns_and_preserves_missing() -> None:
    rows = [{"image_id": "x", "label": "ripe", "feature_a": "1.5", "feature_b": ""}]
    matrix = _feature_matrix(rows, ["feature_a", "feature_b"])
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == 1.5
    assert np.isnan(matrix[0, 1])


def test_union_find_merges_lineage_and_exact_hash_groups() -> None:
    rows = [
        {"image_id": "a", "source_sha256": "sha-a"},
        {"image_id": "b", "source_sha256": "sha-b"},
        {"image_id": "c", "source_sha256": "sha-c"},
    ]
    splits = {
        "a": {"lineage_group_id": "parent-1"},
        "b": {"lineage_group_id": "parent-1"},
        "c": {"lineage_group_id": "parent-3"},
    }
    groups = _union_find_groups(rows, splits)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_split_audit_flags_exact_or_lineage_overlap_across_splits() -> None:
    rows = [
        {"image_id": "a", "source_sha256": "same", "official_accuracy_eligible": "false"},
        {"image_id": "b", "source_sha256": "same", "official_accuracy_eligible": "false"},
    ]
    splits = {
        "a": {"lineage_group_id": "g1", "phase02_split": "train"},
        "b": {"lineage_group_id": "g2", "phase02_split": "test"},
    }
    audit = _audit_group_overlap(rows, splits)
    assert audit["exact_cross_split_groups"] == 1
    assert audit["exact_or_lineage_cross_split"] is True


def test_group_cv_keeps_groups_separate_in_each_fold() -> None:
    y = np.repeat(["a", "b", "c"], 20)
    groups = np.asarray([f"g{i // 2}" for i in range(len(y))])
    folds = _folds(np.zeros((len(y), 1)), y, groups, 5, 123)
    assert len(folds) == 5
    for train, validation in folds:
        assert not set(groups[train]) & set(groups[validation])


def test_all_missing_train_feature_is_handled_and_scalers_only_on_knn_svm() -> None:
    X = np.column_stack([np.arange(30, dtype=float), np.full(30, np.nan)])
    y = np.asarray(["a", "b"] * 15)
    for model in ("KNN", "SVM_RBF", "RandomForest"):
        pipeline = _make_pipeline(model, 42)
        pipeline.set_params(classifier__n_neighbors=3) if model == "KNN" else None
        pipeline.fit(X, y)
        assert len(pipeline.predict(X)) == len(y)
        assert ("scaler" in pipeline.named_steps) == (model in {"KNN", "SVM_RBF"})


def test_defect_metrics_report_false_negatives() -> None:
    payload = _metric_payload(np.asarray(["healthy", "defective", "defective"]),
                              np.asarray(["defective", "healthy", "defective"]),
                              ["healthy", "defective"])
    assert payload["false_negative_defective"] == 1
    assert payload["support"].tolist() == [1, 2]


def test_saved_pipeline_predicts_same_after_load(tmp_path) -> None:
    X = np.asarray([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]])
    y = np.asarray(["healthy", "healthy", "defective", "defective"])
    pipeline = _make_pipeline("RandomForest", 7)
    pipeline.set_params(classifier__n_estimators=10).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": ["f1", "f2"], "task": "defect",
                 "labels": ["healthy", "defective"], "model_name": "RandomForest",
                 "experiment": "test"}, path)
    first = predict_from_features(path, {"f1": 0.05, "f2": 0.95})
    second = predict_from_features(path, {"f1": 0.05, "f2": 0.95})
    assert first == second
    assert first["prediction"] == "healthy"
    assert set(first["class_scores"]) == {"healthy", "defective"}
