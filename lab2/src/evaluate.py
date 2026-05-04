"""
Оценка качества модели.

Вычисляет основные метрики классификации и строит кривые.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import validation_curve


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Возвращает словарь с метриками качества.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Рисует матрицу ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Unsatisfied", "Satisfied"],
                yticklabels=["Unsatisfied", "Satisfied"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Рисует ROC-кривую."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Рисует важность признаков.

    Поддерживает модели с атрибутами feature_importances_ (RandomForest, XGBoost)
    и coef_ (LogisticRegression).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        print("Модель не поддерживает извлечение важности признаков.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices][::-1], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def print_classification_report(y_true, y_pred):
    """Отчёт классификации."""
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["Unsatisfied", "Satisfied"],
            digits=4,
        )
    )


def plot_validation_curve(estimator, X, y, param_name, param_range,
                          cv=3, scoring="f1", save_path=None):
    """Строит кривую валидации для одного гиперпараметра."""
    train_scores, test_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, 'o-', color="blue", label="Train score")
    plt.plot(param_range, test_mean, 'o-', color="green", label="Validation score")
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.title(f"Validation Curve for {param_name}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim(0.7, 1.0)
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()