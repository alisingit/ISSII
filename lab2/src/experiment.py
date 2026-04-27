"""
Точка входа для запуска экспериментов.

Использование:
    poetry run python -m src.experiment          # все эксперименты
    poetry run python -m src.experiment --model logistic  # один эксперимент
"""

import argparse
import mlflow
import matplotlib
matplotlib.use("Agg")

from .config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from .data_loader import load_dataset, prepare_train_test_val
from .train import train_and_evaluate, MODEL_BUILDERS


def run_all_experiments(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    results = {}

    # ====================
    # Baseline: LogisticRegression
    # ====================
    print("\n" + "=" * 60)
    print("Baseline: LogisticRegression")
    print("=" * 60)
    for C in [0.1, 1.0, 10.0]:
        params = {"C": C, "solver": "lbfgs"}
        run_name = f"baseline_lr_C={C}"
        val_m, test_m = train_and_evaluate(
            "logistic", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"baseline_lr_C={C}"] = test_m

    # ====================
    # Гипотеза 1: Нелинейные модели (RF, XGBoost, LightGBM)
    # ====================
    print("\n" + "=" * 60)
    print("Гипотеза 1: Нелинейные ансамблевые модели")
    print("=" * 60)

    # RandomForest
    for n_est, depth in [(100, 10), (200, 15), (300, 20)]:
        params = {"n_estimators": n_est, "max_depth": depth}
        run_name = f"hyp1_rf_n={n_est}_d={depth}"
        val_m, test_m = train_and_evaluate(
            "random_forest", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp1_rf_n={n_est}_d={depth}"] = test_m

    # XGBoost
    for n_est, lr in [(200, 0.1), (300, 0.05), (500, 0.1)]:
        params = {"n_estimators": n_est, "learning_rate": lr}
        run_name = f"hyp1_xgb_n={n_est}_lr={lr}"
        val_m, test_m = train_and_evaluate(
            "xgboost", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp1_xgb_n={n_est}_lr={lr}"] = test_m

    # LightGBM
    for n_est, lr in [(200, 0.1), (300, 0.05)]:
        params = {"n_estimators": n_est, "learning_rate": lr}
        run_name = f"hyp1_lgb_n={n_est}_lr={lr}"
        val_m, test_m = train_and_evaluate(
            "lightgbm", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp1_lgb_n={n_est}_lr={lr}"] = test_m

    # ====================
    # Гипотеза 2: Отбор признаков (SelectKBest)
    # ====================
    print("\n" + "=" * 60)
    print("Гипотеза 2: Отбор признаков")
    print("=" * 60)
    for k in (20, 30, 50):
        params = {"k": k}
        run_name = f"hyp2_feat_sel_k={k}" 
        val_m, test_m = train_and_evaluate(
            "feat_sel", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp2_feat_sel_k={k}"] = test_m

    # ====================
    # Гипотеза 3: Работа с дисбалансом классов
    # ====================
    print("\n" + "=" * 60)
    print("Гипотеза 3: Дисбаланс классов")
    print("=" * 60)
    # class_weight='balanced'
    for n_est in (100, 200):
        params = {"strategy": "balanced", "n_estimators": n_est, "max_depth": 15}
        run_name = f"hyp3_balanced_rf_n={n_est}"
        val_m, test_m = train_and_evaluate(
            "imbalanced", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp3_balanced_rf_n={n_est}"] = test_m

    # SMOTE
    for n_est in (100, 200):
        params = {"strategy": "smote", "n_estimators": n_est, "max_depth": 15}
        run_name = f"hyp3_smote_rf_n={n_est}"
        val_m, test_m = train_and_evaluate(
            "imbalanced", X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, params, run_name=run_name,
        )
        results[f"hyp3_smote_rf_n={n_est}"] = test_m

    return results


def print_summary(results):
    import pandas as pd
    df = pd.DataFrame(results).T
    df = df.sort_values("f1", ascending=False)
    print("\n" + "=" * 60)
    print("Сводка результатов (отсортировано по F1-мере)")
    print("=" * 60)
    print(df.round(4).to_string())
    print("\nЛучшая модель по F1:")
    best = df.iloc[0]
    print(f"  {df.index[0]}: F1 = {best['f1']:.4f}, ROC-AUC = {best['roc_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Запуск экспериментов ML")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_BUILDERS.keys()),
        default=None,
        help="Запустить только одну модель (по умолчанию — все)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Загрузка данных из MinIO...")
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        prepare_train_test_val(df)

    if args.model:
        train_and_evaluate(
            args.model, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names,
        )
    else:
        results = run_all_experiments(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
        )
        print_summary(results)


if __name__ == "__main__":
    main()
    