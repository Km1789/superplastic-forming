# train_gradient_boosting.py
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from group_split import group_split, groups_from, group_kfold

# ===================== НАСТРОЙКИ =====================

CSV_PATH = r"data/material/Datasets/three_curves_10_type_2_for_real_data.csv"
OUT_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data"
EXPERIMENT_SUBDIR = "Classic_GB_log"

TARGET_COLS = ["K", "m"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Обновлённая сетка гиперпараметров (удалён некорректный параметр alpha)
# Сетка: 3 x 2 x 3 x 2 x 2 = 72 конфигурации (отдельно для K и для m)
PARAM_GRID = {
    "model__regressor__n_estimators": [200, 400, 800],
    "model__regressor__learning_rate": [0.05, 0.1],
    "model__regressor__max_depth": [2, 3, 4],
    "model__regressor__subsample": [0.8, 1.0],
    "model__regressor__min_samples_leaf": [1, 3],
    # "alpha" удалён, т.к. в GradientBoostingRegressor он отвечает за квантиль потерь, а не за L2
}

# =====================================================


def read_csv_smart(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    if first_line.lower().startswith("sep=") and len(first_line.strip()) >= 5:
        sep = first_line.strip()[4]
        df = pd.read_csv(path, sep=sep, skiprows=1, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def rmse_singleoutput(y_true, y_pred) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_feature_importance_plot(importances, feature_cols, out_path, title, ylabel):
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(max(6, 0.5 * len(order)), 4))
    plt.bar(range(len(order)), np.asarray(importances)[order])
    plt.xticks(range(len(order)), np.array(feature_cols)[order], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_parity_plot(true_vals, pred_vals, out_path, target_name):
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.figure(figsize=(5, 5))
    plt.scatter(true_vals, pred_vals, alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel(f"Истинное {target_name}")
    plt.ylabel(f"Предсказанное {target_name}")
    plt.title(f"Parity plot для {target_name} (GradientBoosting)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residual_hist(residuals, out_path, target_name):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Ошибка (y_true - y_pred)")
    plt.ylabel("Частота")
    plt.title(f"Распределение ошибок для {target_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residuals_vs_pred(pred_vals, residuals, out_path, target_name):
    plt.figure(figsize=(6, 4))
    plt.scatter(pred_vals, residuals, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(f"Предсказанное {target_name}")
    plt.ylabel("Ошибка (y_true - y_pred)")
    plt.title(f"Ошибки vs prediction для {target_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_single_target_pipeline(feature_cols):
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    base_regressor = GradientBoostingRegressor(random_state=RANDOM_STATE)

    # Комбинированное преобразование цели: логарифм -> стандартизация
    # Это приводит логарифмированные значения к нулевому среднему и единичной дисперсии,
    # устраняя проблему разного масштаба исходных K и m.
    target_transformer = Pipeline([
        ("log", FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)),
        ("scale", StandardScaler())
    ])

    ttr = TransformedTargetRegressor(
        regressor=base_regressor,
        transformer=target_transformer
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", ttr),
    ])
    return pipe


def fit_single_target_model(X_train, y_train_target, feature_cols, target_name, groups=None):
    pipe = build_single_target_pipeline(feature_cols)
    scorer = make_scorer(rmse_singleoutput, greater_is_better=False)

    print(f"GridSearchCV для GradientBoostingRegressor (log+scale target), target={target_name}...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=group_kfold(CV_FOLDS),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train_target, groups=groups)

    best_model: Pipeline = grid.best_estimator_
    best_params_full = grid.best_params_
    best_mean_cv_rmse = float(-grid.best_score_)

    best_params_base = {
        k.replace("model__regressor__", ""): v for k, v in best_params_full.items()
    }

    print(f"Лучшие параметры ({target_name}):")
    for k, v in best_params_base.items():
        print(f"  {k}: {v}")
    print(f"Лучший средний CV-RMSE ({target_name}): {best_mean_cv_rmse:.6f}")

    return {
        "grid": grid,
        "best_model": best_model,
        "best_params_full": best_params_full,
        "best_params_base": best_params_base,
        "best_mean_cv_rmse": best_mean_cv_rmse,
    }


def train_boosting_model() -> None:
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    base_dir = Path(OUT_BASE_DIR)
    dataset_dir = base_dir / csv_path.stem
    out_dir = dataset_dir / EXPERIMENT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю данные из {csv_path}...")
    df = read_csv_smart(csv_path)

    for col in TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"В датасете нет столбца '{col}'. Есть: {list(df.columns)}")

    y = df[TARGET_COLS].copy()
    X = df.drop(columns=TARGET_COLS)
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError("Не удалось найти числовые признаки для обучения (X пустой).")

    feature_cols = list(X.columns)

    # Групповое разбиение по m (см. group_split.py)
    X_train, X_test, y_train, y_test = group_split(
        X, y, groups=groups_from(y), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    fitted = {}
    for target_name in TARGET_COLS:
        fitted[target_name] = fit_single_target_model(
            X_train=X_train,
            y_train_target=y_train[target_name],
            feature_cols=feature_cols,
            target_name=target_name,
            groups=groups_from(y_train),
        )

    best_models = {
        target_name: fitted[target_name]["best_model"]
        for target_name in TARGET_COLS
    }

    # Предсказания (автоматически в исходном масштабе благодаря inverse_transform)
    y_train_pred_dict = {}
    y_test_pred_dict = {}

    for target_name in TARGET_COLS:
        model = best_models[target_name]
        y_train_pred_dict[target_name] = model.predict(X_train)
        y_test_pred_dict[target_name] = model.predict(X_test)

        # Небольшой клиппинг для защиты от машинных погрешностей (необязательно, т.к. expm1 даёт >0)
        y_train_pred_dict[target_name] = np.clip(y_train_pred_dict[target_name], a_min=1e-9, a_max=None)
        y_test_pred_dict[target_name] = np.clip(y_test_pred_dict[target_name], a_min=1e-9, a_max=None)

    y_train_pred = np.column_stack([y_train_pred_dict[col] for col in TARGET_COLS])
    y_test_pred = np.column_stack([y_test_pred_dict[col] for col in TARGET_COLS])

    # Метрики
    rmse_train = []
    rmse_test = []
    r2_train = []
    r2_test = []
    metrics_targets = {}

    for i, col in enumerate(TARGET_COLS):
        yt_train = y_train[col].values
        yp_train = y_train_pred[:, i]
        yt_test = y_test[col].values
        yp_test = y_test_pred[:, i]

        rmse_tr = float(np.sqrt(mean_squared_error(yt_train, yp_train)))
        rmse_te = float(np.sqrt(mean_squared_error(yt_test, yp_test)))
        r2_tr = float(r2_score(yt_train, yp_train))
        r2_te = float(r2_score(yt_test, yp_test))

        rmse_train.append(rmse_tr)
        rmse_test.append(rmse_te)
        r2_train.append(r2_tr)
        r2_test.append(r2_te)

        metrics_targets[col] = {
            "train": {"rmse": rmse_tr, "r2": r2_tr},
            "test": {"rmse": rmse_te, "r2": r2_te},
            "cv": {
                "best_params_full": _to_builtin(fitted[col]["best_params_full"]),
                "best_params_base": _to_builtin(fitted[col]["best_params_base"]),
                "best_mean_cv_rmse": fitted[col]["best_mean_cv_rmse"],
            },
        }

    metrics = {
        "train": {"rmse": rmse_train, "rmse_mean": float(np.mean(rmse_train)), "r2": r2_train},
        "test": {"rmse": rmse_test, "rmse_mean": float(np.mean(rmse_test)), "r2": r2_test},
        "cv": {
            "per_target": {col: metrics_targets[col]["cv"] for col in TARGET_COLS},
            "param_grid": _to_builtin(PARAM_GRID),
            "cv_folds": CV_FOLDS,
        },
        "targets": metrics_targets,
        "target_columns": TARGET_COLS,
        "feature_columns": feature_cols,
        "n_samples": {
            "total": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Метрики сохранены в {metrics_path}")

    # Графики важности признаков
    importances_all = []
    for target_name in TARGET_COLS:
        model_pipe = best_models[target_name]
        ttr = model_pipe.named_steps["model"]
        model = ttr.regressor_
        importances = model.feature_importances_
        importances_all.append(importances)

        fi_target_path = out_dir / f"feature_importances_{target_name}.png"
        save_feature_importance_plot(
            importances=importances,
            feature_cols=feature_cols,
            out_path=fi_target_path,
            title=f"Важности признаков (GBRT) для {target_name}",
            ylabel="Важность признака",
        )
        print(f"График важности признаков для {target_name} сохранён в {fi_target_path}")

    importances_all = np.array(importances_all)
    mean_importances = importances_all.mean(axis=0)
    fi_mean_path = out_dir / "feature_importances.png"
    save_feature_importance_plot(
        importances=mean_importances,
        feature_cols=feature_cols,
        out_path=fi_mean_path,
        title="Важности признаков (GBRT, усреднено по таргетам)",
        ylabel="Средняя важность признака",
    )
    print(f"График средней важности признаков сохранён в {fi_mean_path}")

    # Parity и остатки
    for i, col in enumerate(TARGET_COLS):
        true_vals = y_test[col].values
        pred_vals = y_test_pred[:, i]
        residuals = true_vals - pred_vals

        pp_path = out_dir / f"parity_{col}.png"
        save_parity_plot(true_vals, pred_vals, pp_path, col)
        print(f"Parity-график для {col} сохранён в {pp_path}")

        res_hist_path = out_dir / f"residuals_{col}.png"
        save_residual_hist(residuals, res_hist_path, col)
        print(f"Гистограмма ошибок для {col} сохранена в {res_hist_path}")

        res_vs_pred_path = out_dir / f"residuals_vs_pred_{col}.png"
        save_residuals_vs_pred(pred_vals, residuals, res_vs_pred_path, col)
        print(f"График ошибок vs prediction для {col} сохранён в {res_vs_pred_path}")

    # Сохранение top CV результатов
    for target_name in TARGET_COLS:
        grid = fitted[target_name]["grid"]
        cv_df = pd.DataFrame(grid.cv_results_).copy()
        keep_cols = [
            "rank_test_score", "mean_test_score", "std_test_score",
            "mean_fit_time", "mean_score_time", "params"
        ]
        keep_cols = [c for c in keep_cols if c in cv_df.columns]
        cv_top = cv_df.sort_values("rank_test_score").loc[:, keep_cols].head(50).copy()
        if "mean_test_score" in cv_top.columns:
            cv_top["mean_cv_rmse"] = -cv_top["mean_test_score"]
        cv_csv_path = out_dir / f"cv_results_top_{target_name}.csv"
        cv_top.to_csv(cv_csv_path, index=False, encoding="utf-8-sig")
        print(f"Top CV results для {target_name} сохранены в {cv_csv_path}")

    # Сохранение модели
    model_bundle = {
        "models": best_models,
        "target_columns": TARGET_COLS,
        "feature_columns": feature_cols,
        "random_state": RANDOM_STATE,
    }
    model_path = out_dir / "boosting_model_log.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"Модель сохранена в {model_path}")

    # Проверка загрузки
    loaded = joblib.load(model_path)
    demo_X = X_test.iloc[[0]]
    demo_pred = np.array([
        loaded["models"][col].predict(demo_X)[0] for col in loaded["target_columns"]
    ])
    print("Demo predict (K,m) в исходных единицах (положительные):", demo_pred)

    print("Готово.")


if __name__ == "__main__":
    train_boosting_model()