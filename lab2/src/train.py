"""
Модуль обучения моделей.

Содержит функции для:
  - базового решения (LogisticRegression)
  - альтернативных моделей (RandomForest, XGBoost, LightGBM)
  - отбора признаков (SelectKBest + LogReg)
  - работы с дисбалансом классов (class_weight, SMOTE)
"""

import os
import tempfile

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    print_classification_report,
)


def build_baseline_model(**params):
    """Baseline: логистическая регрессия."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=params.get("random_state", 42),
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            solver=params.get("solver", "lbfgs"),
        )),
    ])


def build_random_forest_model(**params):
    """Гипотеза 1: случайный лес."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 15),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=params.get("random_state", 42),
            class_weight=params.get("class_weight", None),
            n_jobs=-1,
        )),
    ])


def build_xgboost_model(**params):
    """Гипотеза 1: XGBoost."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=params.get("random_state", 42),
            eval_metric="logloss",
            use_label_encoder=False,
        )),
    ])


def build_lightgbm_model(**params):
    """Гипотеза 1: LightGBM."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=params.get("random_state", 42),
            verbose=-1,
            n_jobs=-1,
        )),
    ])


def build_feature_selection_model(**params):
    """Гипотеза 2: отбор k лучших признаков + LogisticRegression."""
    k = params.get("k", 30)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=params.get("random_state", 42),
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
        )),
    ])


def build_imbalanced_model(**params):
    """Гипотеза 3: работа с дисбалансом классов.

    - strategy = 'balanced'  → RandomForest с class_weight='balanced'
    - strategy = 'smote'     → SMOTE + RandomForest
    """
    strategy = params.get("strategy", "balanced")
    n_estimators = params.get("n_estimators", 150)
    max_depth = params.get("max_depth", 15)

    if strategy == "balanced":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )),
        ])
    elif strategy == "smote":
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=params.get("random_state", 42))),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Словарь всех builder'ов (используется для одиночного запуска)
MODEL_BUILDERS = {
    # Baseline
    "logistic": build_baseline_model,
    # Гипотеза 1: нелинейные модели
    "random_forest": build_random_forest_model,
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
    # Гипотеза 2: отбор признаков
    "feat_sel": build_feature_selection_model,
    # Гипотеза 3: дисбаланс
    "imbalanced": build_imbalanced_model,
}


def train_and_evaluate(
    model_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names,
    params: dict | None = None,
):
    """
    Обучает модель, логирует параметры/метрики/артефакты в MLflow.
    """
    if params is None:
        params = {}

    params.setdefault("random_state", 42)

    builder = MODEL_BUILDERS[model_name]
    model = builder(**params)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)

        model.fit(X_train, y_train)

        # Валидация
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)

        # Тест
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            fi_path = os.path.join(tmpdir, "feature_importance.png")

            plot_confusion_matrix(y_test, y_test_pred, cm_path)
            plot_roc_curve(y_test, y_test_proba, roc_path)

            final_model = model.named_steps.get("model", model)
            plot_feature_importance(final_model, feature_names, save_path=fi_path)

            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(model, "model")

        print(f"\n=== {model_name} ===")
        print(f"Validation: {val_metrics}")
        print(f"Test:       {test_metrics}")
        print_classification_report(y_test, y_test_pred)

        return val_metrics, test_metrics
    