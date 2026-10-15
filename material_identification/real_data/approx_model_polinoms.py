# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATA_DIR = Path(
    r"data/material/test_real_data/cut_data"
)

# Если нужен одиночный запуск без grid search
# DEG_K = 5
# DEG_M = 5

# Диапазоны степеней для grid search.
# range(1, 6) означает степени 1, 2, 3, 4, 5.
GRID_DEG_K = range(1, 6)
GRID_DEG_M = range(1, 6)

# По какой ошибке выбирать лучший вариант:
# "mean_relative_error", "rmse" или "max_relative_error"
BEST_METRIC = "rmse"

# Файлы вывода
OUTPUT_COEFFS = DATA_DIR / "fitted_coefficients.csv"
OUTPUT_GRID_RESULTS = DATA_DIR / "grid_search_results.csv"
OUTPUT_BEST_COEFFS = DATA_DIR / "best_fitted_coefficients.csv"


# ============================================================
# ЧТЕНИЕ ДАННЫХ
# ============================================================

def extract_strain_rate_from_filename(path: Path) -> float:
    """
    Извлекает скорость деформации из имени файла.

    Например:
        1e-4.txt -> 1e-4
        4e-4.txt -> 4e-4
        8e-4.txt -> 8e-4
    """
    name = path.stem

    match = re.search(
        r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?",
        name
    )

    if match is None:
        raise ValueError(
            f"Не удалось извлечь strain rate из имени файла: {path.name}"
        )

    return float(match.group(0).replace(",", "."))


def read_curve_file(path: Path) -> pd.DataFrame:
    """
    Читает файл с двумя столбцами:
        eps sigma

    Поддерживает:
    - десятичную запятую;
    - десятичную точку;
    - случайные символы после числа, например '15,51496458s'.
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(
                r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?",
                line
            )

            if len(nums) >= 2:
                eps = float(nums[0].replace(",", "."))
                sigma = float(nums[1].replace(",", "."))
                rows.append((eps, sigma))

    if not rows:
        raise ValueError(
            f"Файл пустой или не удалось прочитать данные: {path}"
        )

    df = pd.DataFrame(rows, columns=["eps", "sigma"])
    df["strain_rate"] = extract_strain_rate_from_filename(path)

    return df


def load_all_curves(data_dir: Path) -> pd.DataFrame:
    """
    Загружает все .txt файлы из папки.
    """
    files = sorted(data_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"В папке нет .txt файлов: {data_dir}")

    dfs = []

    for file in files:
        df = read_curve_file(file)
        df["file"] = file.name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Для относительной ошибки и логарифмического начального приближения
    # оставляем только точки с положительным sigma.
    data = data[data["sigma"] > 0].copy()

    if data.empty:
        raise ValueError("После фильтрации sigma > 0 не осталось данных.")

    return data


# ============================================================
# МОДЕЛЬ
# ============================================================

def polynomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Полином:
        c0 + c1*x + c2*x^2 + ...
    """
    y = np.zeros_like(x, dtype=float)

    for i, c in enumerate(coeffs):
        y += c * x**i

    return y


def material_model(
    eps: np.ndarray,
    strain_rate: np.ndarray,
    coeffs_K: np.ndarray,
    coeffs_m: np.ndarray
) -> np.ndarray:
    """
    Модель материала:

        sigma = K(eps) * strain_rate ** m(eps)

    где:
        K(eps) = a0 + a1*eps + ...
        m(eps) = b0 + b1*eps + ...
    """
    K = polynomial(eps, coeffs_K)
    m = polynomial(eps, coeffs_m)

    sigma = K * strain_rate**m

    return sigma


# ============================================================
# НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ
# ============================================================

def initial_guess_log_model(
    eps: np.ndarray,
    sigma: np.ndarray,
    strain_rate: np.ndarray,
    deg_K: int,
    deg_m: int
) -> np.ndarray:
    """
    Строит начальное приближение через логарифмирование:

        sigma = K(eps) * strain_rate ** m(eps)

    Тогда:

        ln(sigma) = ln(K(eps)) + m(eps) * ln(strain_rate)

    На этапе начального приближения временно считаем,
    что ln(K(eps)) тоже аппроксимируется полиномом.

    После этого восстанавливаем K(eps) = exp(ln(K(eps)))
    и аппроксимируем уже K(eps) обычным полиномом нужной степени.
    """
    nK = deg_K + 1
    nm = deg_m + 1

    if np.any(sigma <= 0):
        raise ValueError("Для логарифмирования нужны только sigma > 0.")

    if np.any(strain_rate <= 0):
        raise ValueError("strain_rate должен быть положительным.")

    # Матрица для ln(K)
    A_logK = np.column_stack([eps**i for i in range(nK)])

    # Матрица для m(eps) * ln(strain_rate)
    A_m = np.column_stack([
        np.log(strain_rate) * eps**i
        for i in range(nm)
    ])

    A = np.column_stack([A_logK, A_m])
    y = np.log(sigma)

    coeffs_log, *_ = np.linalg.lstsq(A, y, rcond=None)

    coeffs_logK = coeffs_log[:nK]
    coeffs_m = coeffs_log[nK:]

    # Значения K(eps), восстановленные из logK
    logK_values = polynomial(eps, coeffs_logK)
    K_values = np.exp(logK_values)

    # Аппроксимируем K_values обычным полиномом
    A_K = np.column_stack([eps**i for i in range(nK)])
    coeffs_K, *_ = np.linalg.lstsq(A_K, K_values, rcond=None)

    return np.concatenate([coeffs_K, coeffs_m])


# ============================================================
# ОДИН ПОДБОР ДЛЯ ЗАДАННЫХ СТЕПЕНЕЙ
# ============================================================

def fit_material_model(
    data: pd.DataFrame,
    deg_K: int,
    deg_m: int
):
    """
    Подбирает коэффициенты K(eps) и m(eps)
    для заданных степеней полиномов deg_K и deg_m.
    """
    eps = data["eps"].to_numpy(dtype=float)
    sigma_exp = data["sigma"].to_numpy(dtype=float)
    strain_rate = data["strain_rate"].to_numpy(dtype=float)

    nK = deg_K + 1
    nm = deg_m + 1

    x0 = initial_guess_log_model(
        eps=eps,
        sigma=sigma_exp,
        strain_rate=strain_rate,
        deg_K=deg_K,
        deg_m=deg_m
    )

    def residuals(params):
        coeffs_K = params[:nK]
        coeffs_m = params[nK:nK + nm]

        sigma_pred = material_model(
            eps=eps,
            strain_rate=strain_rate,
            coeffs_K=coeffs_K,
            coeffs_m=coeffs_m
        )

        # Относительная ошибка:
        # так большие значения sigma не будут полностью доминировать.
        return (sigma_pred - sigma_exp) / sigma_exp

    result = least_squares(
        residuals,
        x0,
        max_nfev=50000
    )

    params = result.x

    coeffs_K = params[:nK]
    coeffs_m = params[nK:nK + nm]

    sigma_pred = material_model(
        eps=eps,
        strain_rate=strain_rate,
        coeffs_K=coeffs_K,
        coeffs_m=coeffs_m
    )

    errors = sigma_pred - sigma_exp
    relative_errors = errors / sigma_exp

    rmse = np.sqrt(np.mean(errors**2))
    mean_relative_error = np.mean(np.abs(relative_errors))
    max_relative_error = np.max(np.abs(relative_errors))

    return (
        coeffs_K,
        coeffs_m,
        rmse,
        mean_relative_error,
        max_relative_error,
        result
    )


# ============================================================
# GRID SEARCH ПО СТЕПЕНЯМ ПОЛИНОМОВ
# ============================================================

def grid_search_polynomial_degrees(
    data: pd.DataFrame,
    deg_K_values,
    deg_m_values,
    best_metric: str = "mean_relative_error",
    output_results_path: Path | None = None
):
    """
    Перебирает степени полиномов K(eps) и m(eps).

    Для каждой пары deg_K, deg_m запускает fit_material_model.

    Возвращает:
        results_df — таблицу ошибок по всем вариантам;
        best_info — словарь с лучшим вариантом.
    """

    allowed_metrics = {
        "mean_relative_error",
        "rmse",
        "max_relative_error"
    }

    if best_metric not in allowed_metrics:
        raise ValueError(
            f"Некорректная метрика best_metric={best_metric}. "
            f"Допустимые варианты: {allowed_metrics}"
        )

    results = []
    best_info = None

    for deg_K in deg_K_values:
        for deg_m in deg_m_values:

            print("=" * 70)
            print(f"Пробуем deg_K = {deg_K}, deg_m = {deg_m}")

            n_coeffs_K = deg_K + 1
            n_coeffs_m = deg_m + 1
            n_total_coeffs = n_coeffs_K + n_coeffs_m

            try:
                (
                    coeffs_K,
                    coeffs_m,
                    rmse,
                    mean_relative_error,
                    max_relative_error,
                    result
                ) = fit_material_model(
                    data=data,
                    deg_K=deg_K,
                    deg_m=deg_m
                )

                row = {
                    "deg_K": deg_K,
                    "deg_m": deg_m,
                    "n_coeffs_K": n_coeffs_K,
                    "n_coeffs_m": n_coeffs_m,
                    "n_total_coeffs": n_total_coeffs,
                    "rmse": rmse,
                    "mean_relative_error": mean_relative_error,
                    "max_relative_error": max_relative_error,
                    "cost": result.cost,
                    "optimality": result.optimality,
                    "nfev": result.nfev,
                    "success": result.success,
                    "message": result.message
                }

                results.append(row)

                print(f"RMSE: {rmse:.8g}")
                print(f"Mean relative error: {mean_relative_error:.8g}")
                print(f"Max relative error: {max_relative_error:.8g}")
                print(f"Cost: {result.cost:.8g}")
                print(f"Success: {result.success}")

                current_metric_value = row[best_metric]

                if best_info is None or current_metric_value < best_info["metric_value"]:
                    best_info = {
                        "deg_K": deg_K,
                        "deg_m": deg_m,
                        "n_coeffs_K": n_coeffs_K,
                        "n_coeffs_m": n_coeffs_m,
                        "n_total_coeffs": n_total_coeffs,
                        "coeffs_K": coeffs_K,
                        "coeffs_m": coeffs_m,
                        "rmse": rmse,
                        "mean_relative_error": mean_relative_error,
                        "max_relative_error": max_relative_error,
                        "result": result,
                        "metric_name": best_metric,
                        "metric_value": current_metric_value
                    }

                    print(">>> Новый лучший вариант")

            except Exception as e:
                row = {
                    "deg_K": deg_K,
                    "deg_m": deg_m,
                    "n_coeffs_K": n_coeffs_K,
                    "n_coeffs_m": n_coeffs_m,
                    "n_total_coeffs": n_total_coeffs,
                    "rmse": np.nan,
                    "mean_relative_error": np.nan,
                    "max_relative_error": np.nan,
                    "cost": np.nan,
                    "optimality": np.nan,
                    "nfev": np.nan,
                    "success": False,
                    "message": str(e)
                }

                results.append(row)

                print("Ошибка при подборе:")
                print(e)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=best_metric,
        ascending=True,
        na_position="last"
    ).reset_index(drop=True)

    if output_results_path is not None:
        output_results_path = Path(output_results_path)
        results_df.to_csv(
            output_results_path,
            index=False,
            encoding="utf-8-sig"
        )

    return results_df, best_info


# ============================================================
# СОХРАНЕНИЕ КОЭФФИЦИЕНТОВ
# ============================================================

def save_coefficients(
    coeffs_K: np.ndarray,
    coeffs_m: np.ndarray,
    output_path: Path,
    deg_K: int | None = None,
    deg_m: int | None = None,
    extra_info: dict | None = None
):
    """
    Сохраняет коэффициенты K(eps) и m(eps) в CSV.
    """
    rows = []

    for i, c in enumerate(coeffs_K):
        row = {
            "function": "K(eps)",
            "coefficient": f"a{i}",
            "power": i,
            "value": c
        }

        if deg_K is not None:
            row["deg_K"] = deg_K

        if deg_m is not None:
            row["deg_m"] = deg_m

        if extra_info is not None:
            row.update(extra_info)

        rows.append(row)

    for i, c in enumerate(coeffs_m):
        row = {
            "function": "m(eps)",
            "coefficient": f"b{i}",
            "power": i,
            "value": c
        }

        if deg_K is not None:
            row["deg_K"] = deg_K

        if deg_m is not None:
            row["deg_m"] = deg_m

        if extra_info is not None:
            row.update(extra_info)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# ДОПОЛНИТЕЛЬНО: ОДИНОЧНЫЙ ЗАПУСК
# ============================================================

def run_single_fit():
    """
    Обычный одиночный подбор для DEG_K и DEG_M.
    Можно использовать, если grid search не нужен.
    """
    data = load_all_curves(DATA_DIR)

    (
        coeffs_K,
        coeffs_m,
        rmse,
        mean_relative_error,
        max_relative_error,
        result
    ) = fit_material_model(
        data=data,
        deg_K=DEG_K,
        deg_m=DEG_M
    )

    print("\nАппроксимация завершена")
    print(f"Степень K(eps): {DEG_K}")
    print(f"Степень m(eps): {DEG_M}")
    print(f"RMSE по sigma: {rmse:.8g}")
    print(f"Средняя относительная ошибка: {mean_relative_error:.8g}")
    print(f"Максимальная относительная ошибка: {max_relative_error:.8g}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    print("\nK(eps) coefficients:")
    for i, c in enumerate(coeffs_K):
        print(f"a{i} = {c:.12g}")

    print("\nm(eps) coefficients:")
    for i, c in enumerate(coeffs_m):
        print(f"b{i} = {c:.12g}")

    save_coefficients(
        coeffs_K=coeffs_K,
        coeffs_m=coeffs_m,
        output_path=OUTPUT_COEFFS,
        deg_K=DEG_K,
        deg_m=DEG_M,
        extra_info={
            "rmse": rmse,
            "mean_relative_error": mean_relative_error,
            "max_relative_error": max_relative_error
        }
    )

    print("\nКоэффициенты сохранены в файл:")
    print(OUTPUT_COEFFS)


# ============================================================
# MAIN: GRID SEARCH
# ============================================================

if __name__ == "__main__":

    data = load_all_curves(DATA_DIR)

    results_df, best_info = grid_search_polynomial_degrees(
        data=data,
        deg_K_values=GRID_DEG_K,
        deg_m_values=GRID_DEG_M,
        best_metric=BEST_METRIC,
        output_results_path=OUTPUT_GRID_RESULTS
    )

    print("\n" + "=" * 70)
    print("GRID SEARCH ЗАВЕРШЁН")
    print("=" * 70)

    print("\nТаблица результатов, отсортированная по выбранной метрике:")
    print(results_df.to_string(index=False))

    print("\nТаблица результатов сохранена в файл:")
    print(OUTPUT_GRID_RESULTS)

    if best_info is None:
        print("\nНе удалось найти ни одного успешного варианта.")

    else:
        print("\n" + "=" * 70)
        print("ЛУЧШИЙ ВАРИАНТ")
        print("=" * 70)

        print(f"Выбранная метрика: {best_info['metric_name']}")
        print(f"Значение метрики: {best_info['metric_value']:.8g}")

        print(f"\nСтепень K(eps): {best_info['deg_K']}")
        print(f"Степень m(eps): {best_info['deg_m']}")

        print(f"\nКоличество коэффициентов K(eps): {best_info['n_coeffs_K']}")
        print(f"Количество коэффициентов m(eps): {best_info['n_coeffs_m']}")
        print(f"Общее количество коэффициентов: {best_info['n_total_coeffs']}")

        print(f"\nRMSE по sigma: {best_info['rmse']:.8g}")
        print(f"Средняя относительная ошибка: {best_info['mean_relative_error']:.8g}")
        print(f"Максимальная относительная ошибка: {best_info['max_relative_error']:.8g}")

        print("\nK(eps) coefficients:")
        for i, c in enumerate(best_info["coeffs_K"]):
            print(f"a{i} = {c:.12g}")

        print("\nm(eps) coefficients:")
        for i, c in enumerate(best_info["coeffs_m"]):
            print(f"b{i} = {c:.12g}")

        save_coefficients(
            coeffs_K=best_info["coeffs_K"],
            coeffs_m=best_info["coeffs_m"],
            output_path=OUTPUT_BEST_COEFFS,
            deg_K=best_info["deg_K"],
            deg_m=best_info["deg_m"],
            extra_info={
                "rmse": best_info["rmse"],
                "mean_relative_error": best_info["mean_relative_error"],
                "max_relative_error": best_info["max_relative_error"],
                "best_metric": best_info["metric_name"],
                "best_metric_value": best_info["metric_value"]
            }
        )

        print("\nКоэффициенты лучшего варианта сохранены в файл:")
        print(OUTPUT_BEST_COEFFS)