# train_decision_tree_regressor.py
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# ===================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ =====================
import sys
from group_split import group_split, groups_from, group_kfold
sys.setrecursionlimit(10000)

CSV_PATH = r"data/material/Datasets/three_curves_10_type_2_for_real_data.csv"
OUT_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data"
EXPERIMENT_SUBDIR = "tree"

TARGET_COLS = ["K", "m"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Обновлённая сетка гиперпараметров: все параметры дерева теперь с префиксом model__regressor__
# Сетка гиперпараметров: 6 x 3 x 3 x 2 x 3 = 324 конфигурации
PARAM_GRID = {
    "model__regressor__max_depth": [4, 6, 8, 10, 12, None],
    "model__regressor__min_samples_leaf": [1, 2, 5],
    "model__regressor__min_samples_split": [2, 5, 10],
    "model__regressor__max_features": [1.0, 0.5],
    "model__regressor__ccp_alpha": [0.0, 1e-4, 1e-3],
}

# ===============================================================


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Читает CSV, понимает строку вида 'sep=;' и съедает BOM."""
    with path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    if first_line.lower().startswith("sep=") and len(first_line.strip()) >= 5:
        sep = first_line.strip()[4]
        df = pd.read_csv(path, sep=sep, skiprows=1, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def rmse_multioutput(y_true, y_pred) -> float:
    """RMSE для многомерного выхода (усреднение по всем целям)."""
    mse = mean_squared_error(y_true, y_pred)  # uniform average
    return float(np.sqrt(mse))


def train_tree_model():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    base_dir = Path(OUT_BASE_DIR)
    dataset_dir = base_dir / csv_path.stem
    out_dir = dataset_dir / EXPERIMENT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю данные: {csv_path}")
    df = read_csv_smart(csv_path)

    for col in TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"В датасете нет столбца '{col}'. Есть: {list(df.columns)}")

    y = df[TARGET_COLS].copy()
    X = df.drop(columns=TARGET_COLS)

    # Только числовые признаки
    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("X пустой: нет числовых признаков.")

    feature_cols = list(X.columns)

    # Групповое разбиение по m (см. group_split.py)
    X_train, X_test, y_train, y_test = group_split(
        X, y, groups=groups_from(y), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Препроцессор признаков (сохраняет порядок колонок)
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Базовое дерево решений (без нормализации таргетов)
    tree_reg = DecisionTreeRegressor(random_state=RANDOM_STATE)

    # Обёртка для нормализации целевых переменных
    # StandardScaler будет применён отдельно к каждому выходу (multioutput поддерживается)
    ttr = TransformedTargetRegressor(
        regressor=tree_reg,
        transformer=StandardScaler()
    )

    # Полный пайплайн: препроцессинг признаков + модель с нормализацией таргетов
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", ttr)
    ])

    scorer = make_scorer(rmse_multioutput, greater_is_better=False)

    print("GridSearchCV...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=group_kfold(CV_FOLDS),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train, groups=groups_from(y_train))

    best_model: Pipeline = grid.best_estimator_
    best_params = grid.best_params_
    best_mean_cv_rmse = float(-grid.best_score_)

    # Предсказания (автоматически в исходном масштабе благодаря TransformedTargetRegressor)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Метрики в исходных единицах
    mse_train = mean_squared_error(y_train, y_train_pred, multioutput="raw_values")
    mse_test = mean_squared_error(y_test, y_test_pred, multioutput="raw_values")

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    r2_train = r2_score(y_train, y_train_pred, multioutput="raw_values")
    r2_test = r2_score(y_test, y_test_pred, multioutput="raw_values")

    metrics = {
        "target_columns": TARGET_COLS,
        "feature_columns": feature_cols,
        "cv": {
            "best_params": best_params,
            "best_mean_cv_rmse": best_mean_cv_rmse,
            "cv_folds": CV_FOLDS,
        },
        "train": {
            "rmse": rmse_train.tolist(),
            "rmse_mean": float(np.mean(rmse_train)),
            "r2": r2_train.tolist(),
        },
        "test": {
            "rmse": rmse_test.tolist(),
            "rmse_mean": float(np.mean(rmse_test)),
            "r2": r2_test.tolist(),
        },
        "n_samples": {
            "total": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        }
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_path = out_dir / "decision_tree_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Модель сохранена: {model_path}")

    # Визуализация дерева (первые уровни) – извлекаем внутреннее дерево
    ttr_fitted = best_model.named_steps["model"]
    tree_internal: DecisionTreeRegressor = ttr_fitted.regressor_
    plt.figure(figsize=(20, 10))
    plot_tree(tree_internal, feature_names=feature_cols, filled=True, max_depth=3)
    plt.title("Decision Tree (первые уровни, max_depth=3)\nОбучено на стандартизированных таргетах")
    plt.savefig(out_dir / "tree_structure_depth3.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Мини-проверка загрузки
    loaded = joblib.load(model_path)
    demo_pred = loaded.predict(X_test.iloc[[0]])  # DataFrame!
    print("Demo predict (K,m) в исходных единицах:", demo_pred[0])


if __name__ == "__main__":
    train_tree_model()