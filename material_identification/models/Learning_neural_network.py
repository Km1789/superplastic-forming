# train_mlp_optuna.py
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

import optuna
from group_split import group_split, groups_from, group_kfold


# ===================== НАСТРОЙКИ =====================

CSV_PATH = r"data/material/Datasets/three_curves_10_type_2_for_real_data.csv"
OUT_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data"
EXPERIMENT_SUBDIR = "mlp_optuna_log"

TARGET_COLS = ["K", "m"]

TEST_SIZE = 0.2
VALID_SIZE = 0.2          # от train_full для Optuna
RANDOM_STATE = 42

OPTUNA_STUDY_NAME = "mlp_regression_study_v2"
N_TRIALS = 30  # случайный поиск: 30 конфигураций

MLP_MAX_ITER = 1000       # увеличенное число эпох

# =====================================================


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Читает CSV, обрабатывает строку 'sep=;' и BOM."""
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
    """RMSE для многомерного выхода (усреднение по целям)."""
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def make_progress_bar_callback(n_trials: int):
    state = {"n": 0}
    bar_width = 30

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        state["n"] += 1
        n = state["n"]
        progress = n / n_trials
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\r[Optuna] Progress: |{bar}| {n}/{n_trials} "
            f"Best RMSE: {study.best_value:.6f}",
            end="",
            flush=True,
        )
        if n == n_trials:
            print()

    return callback


def suggest_hidden_layer_sizes(trial: optuna.Trial) -> tuple[int, ...]:
    """Генерирует кортеж размеров скрытых слоёв."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f"layer_{i}_size", 16, 512, step=16)
        sizes.append(size)
    return tuple(sizes)


def build_mlp_pipeline(trial: optuna.Trial, feature_cols: list[str]) -> Pipeline:
    """Создаёт пайплайн с MLP и преобразованием таргетов: log1p -> StandardScaler."""
    hidden_layer_sizes = suggest_hidden_layer_sizes(trial)

    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)          # L2-регуляризация
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", ["auto", 64, 128, 256])
    n_iter_no_change = trial.suggest_int("n_iter_no_change", 10, 30)
    beta_1 = trial.suggest_float("beta_1", 0.8, 0.99)
    beta_2 = trial.suggest_float("beta_2", 0.99, 0.9999)
    epsilon = trial.suggest_float("epsilon", 1e-9, 1e-7, log=True)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate="adaptive",
        learning_rate_init=learning_rate_init,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        tol=1e-4,
        early_stopping=True,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=0.15,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        verbose=False,
    )

    # ИСПРАВЛЕНО: комбинированное преобразование цели: логарифм -> стандартизация
    # Приводит логарифмированные значения K и m к нулевому среднему и единичной дисперсии.
    target_transformer = Pipeline([
        ("log", FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)),
        ("scale", StandardScaler())
    ])

    ttr = TransformedTargetRegressor(regressor=mlp, transformer=target_transformer)

    # Стандартизация признаков
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([("prep", preprocessor), ("mlp", ttr)])
    return pipeline


def build_mlp_from_best_params(params: dict, feature_cols: list[str]) -> Pipeline:
    """Создаёт пайплайн из лучших параметров Optuna (с преобразованием таргетов: log1p -> StandardScaler)."""
    n_layers = params["n_layers"]
    hidden_layer_sizes = tuple(params[f"layer_{i}_size"] for i in range(n_layers))

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=params["activation"],
        solver="adam",
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        learning_rate="adaptive",
        learning_rate_init=params["learning_rate_init"],
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        tol=1e-4,
        early_stopping=True,
        n_iter_no_change=params["n_iter_no_change"],
        validation_fraction=0.15,
        beta_1=params.get("beta_1", 0.9),
        beta_2=params.get("beta_2", 0.999),
        epsilon=params.get("epsilon", 1e-8),
        verbose=False,
    )

    # ИСПРАВЛЕНО: такое же преобразование, как в build_mlp_pipeline
    target_transformer = Pipeline([
        ("log", FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)),
        ("scale", StandardScaler())
    ])
    ttr = TransformedTargetRegressor(regressor=mlp, transformer=target_transformer)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([("prep", preprocessor), ("mlp", ttr)])
    return pipeline


def save_parity_plot(true_vals, pred_vals, out_path, target_name):
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())

    plt.figure(figsize=(5, 5))
    plt.scatter(true_vals, pred_vals, alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel(f"Истинное {target_name}")
    plt.ylabel(f"Предсказанное {target_name}")
    plt.title(f"Parity plot для {target_name} (MLP + Optuna)")
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


def save_optuna_history(trials_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(trials_df["number"].values, trials_df["value"].values, marker="o", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Validation RMSE")
    plt.title("История значений objective (Optuna)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_optuna_best_so_far(trials_df: pd.DataFrame, out_path: Path):
    best_so_far = np.minimum.accumulate(trials_df["value"].values)

    plt.figure(figsize=(7, 4))
    plt.plot(trials_df["number"].values, best_so_far, marker="o", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Best validation RMSE so far")
    plt.title("Лучшее значение objective по мере trial'ов")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_target_correlation_heatmaps(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, out_path: Path):
    corr_true = y_true_df.corr().values
    corr_pred = y_pred_df.corr().values
    cols = list(y_true_df.columns)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(corr_true, vmin=-1, vmax=1)
    axes[0].set_title("Корреляции true")
    axes[0].set_xticks(range(len(cols)))
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_xticklabels(cols)
    axes[0].set_yticklabels(cols)

    im1 = axes[1].imshow(corr_pred, vmin=-1, vmax=1)
    axes[1].set_title("Корреляции pred")
    axes[1].set_xticks(range(len(cols)))
    axes[1].set_yticks(range(len(cols)))
    axes[1].set_xticklabels(cols)
    axes[1].set_yticklabels(cols)

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_mlp_with_optuna(
    csv_path: str,
    out_base_dir: str | None = None,
    experiment_subdir: str = "mlp_optuna_log",
    target_cols: list[str] | tuple[str, str] = ("K", "m"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    n_trials: int = 150,
) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    base_dir = Path(out_base_dir) if out_base_dir else csv_path.parent
    dataset_name = csv_path.stem
    dataset_dir = base_dir / dataset_name
    out_dir = dataset_dir / experiment_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю данные из {csv_path}...")
    df = read_csv_smart(csv_path)

    target_cols = list(target_cols)
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"В датасете нет столбца '{col}'. Есть: {list(df.columns)}")

    y = df[target_cols].copy()
    X = df.drop(columns=target_cols)
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError("X пустой: нет числовых признаков.")

    feature_cols = list(X.columns)

    # train/test
    # Групповое разбиение по m (см. group_split.py)
    X_train_full, X_test, y_train_full, y_test = group_split(
        X, y, groups=groups_from(y), test_size=test_size, random_state=random_state
    )

    # train/valid для Optuna
    X_train, X_valid, y_train, y_valid = group_split(
        X_train_full, y_train_full, groups=groups_from(y_train_full),
        test_size=valid_size, random_state=random_state + 1,
    )
    g_train_full = groups_from(y_train_full)

    print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

    def objective(trial: optuna.Trial) -> float:
        # пятикратная групповая кросс-валидация на обучающей части
        scores = []
        for tr_idx, va_idx in group_kfold(5).split(X_train_full, groups=g_train_full):
            pipe = build_mlp_pipeline(trial, feature_cols)
            pipe.fit(X_train_full.iloc[tr_idx], y_train_full.iloc[tr_idx])
            pred = pipe.predict(X_train_full.iloc[va_idx])
            scores.append(rmse_multioutput(y_train_full.iloc[va_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(study_name=OPTUNA_STUDY_NAME, direction="minimize", sampler=optuna.samplers.RandomSampler(seed=random_state))
    print(f"Запускаю Optuna (n_trials={n_trials})...")
    study.optimize(objective, n_trials=n_trials, callbacks=[make_progress_bar_callback(n_trials)])

    best_params = study.best_trial.params
    best_val_rmse = float(study.best_value)

    print("\nЛучшие гиперпараметры:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best valid RMSE: {best_val_rmse:.6f}")

    # Обучаем лучшую модель на всём train_full
    best_pipeline = build_mlp_from_best_params(best_params, feature_cols)
    best_pipeline.fit(X_train_full, y_train_full)

    # Предсказания (автоматически в исходном масштабе, положительные)
    y_train_full_pred = np.clip(best_pipeline.predict(X_train_full), a_min=1e-9, a_max=None)
    y_test_pred = np.clip(best_pipeline.predict(X_test), a_min=1e-9, a_max=None)

    # Метрики
    mse_train = mean_squared_error(y_train_full, y_train_full_pred, multioutput="raw_values")
    mse_test = mean_squared_error(y_test, y_test_pred, multioutput="raw_values")

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    r2_train = r2_score(y_train_full, y_train_full_pred, multioutput="raw_values")
    r2_test = r2_score(y_test, y_test_pred, multioutput="raw_values")

    metrics = {
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
        "optuna": {
            "best_params": best_params,
            "best_val_rmse": best_val_rmse,
            "n_trials": int(n_trials),
            "study_name": OPTUNA_STUDY_NAME,
        },
        "target_columns": target_cols,
        "feature_columns": feature_cols,
        "n_samples": {
            "total": int(len(df)),
            "train_full": int(len(X_train_full)),
            "train": int(len(X_train)),
            "valid": int(len(X_valid)),
            "test": int(len(X_test)),
        },
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Метрики сохранены: {metrics_path}")

    # Сохранение таблиц с предсказаниями
    y_train_full_pred_arr = np.asarray(y_train_full_pred)
    y_test_pred_arr = np.asarray(y_test_pred)

    train_pred_df = X_train_full.copy()
    for i, col in enumerate(target_cols):
        train_pred_df[f"{col}_true"] = y_train_full[col].values
        train_pred_df[f"{col}_pred"] = y_train_full_pred_arr[:, i]
        train_pred_df[f"{col}_residual"] = train_pred_df[f"{col}_true"] - train_pred_df[f"{col}_pred"]

    train_pred_path = out_dir / "train_predictions.csv"
    train_pred_df.to_csv(train_pred_path, index=False, encoding="utf-8-sig")
    print(f"Train predictions сохранены: {train_pred_path}")

    test_pred_df = X_test.copy()
    for i, col in enumerate(target_cols):
        test_pred_df[f"{col}_true"] = y_test[col].values
        test_pred_df[f"{col}_pred"] = y_test_pred_arr[:, i]
        test_pred_df[f"{col}_residual"] = test_pred_df[f"{col}_true"] - test_pred_df[f"{col}_pred"]

    test_pred_path = out_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False, encoding="utf-8-sig")
    print(f"Test predictions сохранены: {test_pred_path}")

    # Графики
    for i, col in enumerate(target_cols):
        true_vals = y_test[col].values
        pred_vals = y_test_pred_arr[:, i]
        residuals = true_vals - pred_vals

        pp_path = out_dir / f"parity_{col}.png"
        save_parity_plot(true_vals, pred_vals, pp_path, col)
        print(f"Parity plot сохранён: {pp_path}")

        residual_hist_path = out_dir / f"residuals_{col}.png"
        save_residual_hist(residuals, residual_hist_path, col)
        print(f"Гистограмма residuals сохранена: {residual_hist_path}")

        residual_vs_pred_path = out_dir / f"residuals_vs_pred_{col}.png"
        save_residuals_vs_pred(pred_vals, residuals, residual_vs_pred_path, col)
        print(f"Residuals vs pred сохранён: {residual_vs_pred_path}")

    # Optuna trials
    trials_records = []
    for t in study.trials:
        row = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
        }
        for k, v in t.params.items():
            row[k] = v
        trials_records.append(row)

    trials_df = pd.DataFrame(trials_records).sort_values("number").reset_index(drop=True)
    trials_csv_path = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False, encoding="utf-8-sig")
    print(f"Optuna trials сохранены: {trials_csv_path}")

    completed_trials_df = trials_df[trials_df["value"].notna()].copy()
    if not completed_trials_df.empty:
        hist_path = out_dir / "optuna_history.png"
        save_optuna_history(completed_trials_df, hist_path)
        print(f"График истории Optuna сохранён: {hist_path}")

        best_path = out_dir / "optuna_best_so_far.png"
        save_optuna_best_so_far(completed_trials_df, best_path)
        print(f"График best-so-far сохранён: {best_path}")

    # Корреляции
    y_true_df = y_test[target_cols].reset_index(drop=True).copy()
    y_pred_df = pd.DataFrame(y_test_pred_arr, columns=target_cols)

    corr_path = out_dir / "target_correlation_true_vs_pred.png"
    save_target_correlation_heatmaps(y_true_df, y_pred_df, corr_path)
    print(f"График корреляций true/pred сохранён: {corr_path}")

    # Сохранение модели
    model_path = out_dir / "mlp_optuna_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Модель сохранена: {model_path}")

    # Проверка загрузки
    loaded = joblib.load(model_path)
    demo_pred = loaded.predict(X_test.iloc[[0]])
    print("Demo predict (K,m) в исходных единицах (положительные):", demo_pred[0])

    print("Готово.")


if __name__ == "__main__":
    train_mlp_with_optuna(
        CSV_PATH,
        out_base_dir=OUT_BASE_DIR,
        experiment_subdir=EXPERIMENT_SUBDIR,
        target_cols=TARGET_COLS,
        test_size=TEST_SIZE,
        valid_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )