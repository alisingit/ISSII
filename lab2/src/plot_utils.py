"""
Утилиты для построения графиков анализа данных.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_class_distribution(y, save_path=None):
    """Распределение классов."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    ax.bar(["Unsatisfied (0)", "Satisfied (1)"], counts.values, color=["#e74c3c", "#2ecc71"])
    ax.set_ylabel("Количество")
    ax.set_title("Распределение целевой переменной")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center")
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df, top_n=20, save_path=None):
    """Тепловая карта корреляций для top-N признаков."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    # Оставляем top_n признаков, наиболее коррелирующих с is_satisfied
    if "is_satisfied" in corr.columns:
        top_features = corr["is_satisfied"].abs().sort_values(ascending=False).head(top_n).index
        corr_subset = corr.loc[top_features, top_features]
    else:
        corr_subset = corr.iloc[:top_n, :top_n]

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5)
    plt.title("Корреляция признаков (top features)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(roc_data: dict, save_path=None):
    """
    Сравнение ROC-кривых для нескольких моделей.

    Parameters
    ----------
    roc_data : dict
        {model_name: (y_true, y_proba)}.
    """
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_proba) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Сравнение ROC-кривых")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()