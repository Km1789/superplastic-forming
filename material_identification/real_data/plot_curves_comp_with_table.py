import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# ЧТЕНИЕ ДАННЫХ
# ============================================================

def extract_strain_rate_from_filename(path: Path) -> float:
    """
    Извлекает скорость деформации из имени файла.

    Примеры:
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
        raise ValueError(f"Не удалось извлечь strain rate из имени файла: {path.name}")

    return float(match.group(0).replace(",", "."))


def read_curve_file(path: Path) -> pd.DataFrame:
    """
    Читает файл с двумя столбцами:
        epsilon sigma

    Поддерживает:
    - десятичную запятую;
    - десятичную точку;
    - случайные символы после чисел.
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
        raise ValueError(f"Файл пустой или не удалось прочитать данные: {path}")

    df = pd.DataFrame(rows, columns=["eps", "sigma"])
    df["strain_rate"] = extract_strain_rate_from_filename(path)

    df = df.sort_values("eps").reset_index(drop=True)

    return df


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
    strain_rate: float,
    coeffs_K: np.ndarray,
    coeffs_m: np.ndarray
) -> np.ndarray:
    """
    sigma = K(eps) * strain_rate ** m(eps)
    """
    K = polynomial(eps, coeffs_K)
    m = polynomial(eps, coeffs_m)

    return K * strain_rate**m


def coefficients_are_defined(coeffs_K, coeffs_m) -> bool:
    """
    Проверяет, заданы ли коэффициенты модели.
    """
    if coeffs_K is None or coeffs_m is None:
        return False

    if len(coeffs_K) == 0 or len(coeffs_m) == 0:
        return False

    return True


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_curve_against_experiment(
    eps_exp: np.ndarray,
    sigma_exp: np.ndarray,
    eps_other: np.ndarray,
    sigma_other: np.ndarray,
) -> float:
    """
    Считает RMSE другой кривой относительно экспериментальной.

    Если сетки eps разные, другая кривая интерполируется
    на экспериментальную сетку.

    RMSE считается только на общей области eps.
    """
    eps_exp = np.asarray(eps_exp, dtype=float)
    sigma_exp = np.asarray(sigma_exp, dtype=float)
    eps_other = np.asarray(eps_other, dtype=float)
    sigma_other = np.asarray(sigma_other, dtype=float)

    # Сортировка другой кривой по eps на всякий случай
    order = np.argsort(eps_other)
    eps_other = eps_other[order]
    sigma_other = sigma_other[order]

    # Общая область по eps
    eps_min = max(np.min(eps_exp), np.min(eps_other))
    eps_max = min(np.max(eps_exp), np.max(eps_other))

    mask = (eps_exp >= eps_min) & (eps_exp <= eps_max)

    if np.sum(mask) < 2:
        return np.nan

    eps_common = eps_exp[mask]
    sigma_exp_common = sigma_exp[mask]

    sigma_other_interp = np.interp(
        eps_common,
        eps_other,
        sigma_other,
    )

    return calculate_rmse(sigma_exp_common, sigma_other_interp)


def build_rmse_table(
    experimental_curves,
    alt_groups_data,
    has_model=False,
    coeffs_K=None,
    coeffs_m=None,
    label_model="Полиномиальная модель",
):
    """
    Строит таблицу RMSE для всех методов относительно эксперимента.

    Возвращает:
        rmse_df
    """
    rows = []

    experimental_by_sr = {
        float(item["strain_rate"]): item
        for item in experimental_curves
    }

    # ------------------------------------------------------------
    # RMSE для полиномиальной модели
    # ------------------------------------------------------------

    if has_model:
        rmse_values = []

        for sr, exp_item in experimental_by_sr.items():
            df_exp = exp_item["data"]

            eps_exp = df_exp["eps"].to_numpy(dtype=float)
            sigma_exp = df_exp["sigma"].to_numpy(dtype=float)

            sigma_model = material_model(
                eps=eps_exp,
                strain_rate=sr,
                coeffs_K=coeffs_K,
                coeffs_m=coeffs_m,
            )

            rmse = calculate_rmse(sigma_exp, sigma_model)
            rmse_values.append(rmse)

            rows.append({
                "method": label_model,
                "strain_rate": sr,
                "rmse": rmse,
                "curve_count_for_average": 1,
            })

        rows.append({
            "method": label_model,
            "strain_rate": "mean",
            "rmse": float(np.nanmean(rmse_values)),
            "curve_count_for_average": int(np.sum(~np.isnan(rmse_values))),
        })

    # ------------------------------------------------------------
    # RMSE для дополнительных групп
    # ------------------------------------------------------------

    for group in alt_groups_data:
        method_label = group["label"]
        rmse_values = []

        for sr, exp_item in experimental_by_sr.items():
            if sr not in group["curves_by_sr"]:
                rmse = np.nan

                rows.append({
                    "method": method_label,
                    "strain_rate": sr,
                    "rmse": rmse,
                    "curve_count_for_average": 0,
                })

                continue

            df_exp = exp_item["data"]
            df_other = group["curves_by_sr"][sr]["data"]

            eps_exp = df_exp["eps"].to_numpy(dtype=float)
            sigma_exp = df_exp["sigma"].to_numpy(dtype=float)

            eps_other = df_other["eps"].to_numpy(dtype=float)
            sigma_other = df_other["sigma"].to_numpy(dtype=float)

            rmse = rmse_curve_against_experiment(
                eps_exp=eps_exp,
                sigma_exp=sigma_exp,
                eps_other=eps_other,
                sigma_other=sigma_other,
            )

            rmse_values.append(rmse)

            rows.append({
                "method": method_label,
                "strain_rate": sr,
                "rmse": rmse,
                "curve_count_for_average": 1 if not np.isnan(rmse) else 0,
            })

        rows.append({
            "method": method_label,
            "strain_rate": "mean",
            "rmse": float(np.nanmean(rmse_values)) if len(rmse_values) > 0 else np.nan,
            "curve_count_for_average": int(np.sum(~np.isnan(rmse_values))),
        })

    rmse_df = pd.DataFrame(rows)

    return rmse_df


def print_rmse_table(rmse_df: pd.DataFrame):
    """
    Красиво печатает таблицу RMSE в консоль.
    """
    if rmse_df.empty:
        print("Таблица RMSE пустая.")
        return

    table = rmse_df.copy()

    table["strain_rate"] = table["strain_rate"].apply(
        lambda x: f"{x:.0e}" if isinstance(x, float) else x
    )

    table["rmse"] = table["rmse"].apply(
        lambda x: f"{x:.6g}" if pd.notna(x) else "nan"
    )

    print("\n" + "=" * 70)
    print("RMSE относительно экспериментальных кривых")
    print("=" * 70)
    print(table.to_string(index=False))
    print("=" * 70 + "\n")


def save_rmse_table(rmse_df: pd.DataFrame, rmse_save_path):
    """
    Сохраняет таблицу RMSE в .xlsx или .csv.
    """
    if rmse_save_path is None:
        return

    rmse_save_path = Path(rmse_save_path)
    rmse_save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = rmse_save_path.suffix.lower()

    if suffix == ".xlsx":
        rmse_df.to_excel(rmse_save_path, index=False)
    elif suffix == ".csv":
        rmse_df.to_csv(rmse_save_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(
            "rmse_save_path должен иметь расширение .xlsx или .csv"
        )

    print(f"Таблица RMSE сохранена: {rmse_save_path}")


# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ ГРУППЫ КРИВЫХ
# ============================================================

def normalize_alt_groups(alt_groups=None, alt_file_paths=None, label_alt="Previous method"):
    """
    Приводит дополнительные группы к единому формату.
    """
    groups = []

    if alt_file_paths is not None:
        groups.append({
            "file_paths": alt_file_paths,
            "label": label_alt,
            "linestyle": "--",
            "linewidth": 2.0,
        })

    if alt_groups is not None:
        for i, group in enumerate(alt_groups):
            if group is None:
                continue

            file_paths = group.get("file_paths", None)

            if file_paths is None or len(file_paths) == 0:
                continue

            groups.append({
                "file_paths": file_paths,
                "label": group.get("label", f"Alt group {i + 1}"),
                "linestyle": group.get("linestyle", "--"),
                "linewidth": group.get("linewidth", 2.0),
            })

    normalized_groups = []

    for group in groups:
        normalized_groups.append({
            "file_paths": [Path(p) for p in group["file_paths"]],
            "label": group["label"],
            "linestyle": group["linestyle"],
            "linewidth": group["linewidth"],
        })

    return normalized_groups


def read_alt_groups(alt_groups):
    """
    Читает все дополнительные группы и возвращает список групп,
    где внутри каждой группы кривые разложены по strain_rate.
    """
    result = []

    for group in alt_groups:
        curves_by_sr = {}

        for path in group["file_paths"]:
            df = read_curve_file(path)
            sr = float(df["strain_rate"].iloc[0])

            curves_by_sr[sr] = {
                "path": path,
                "strain_rate": sr,
                "data": df,
            }

        result.append({
            "label": group["label"],
            "linestyle": group["linestyle"],
            "linewidth": group["linewidth"],
            "curves_by_sr": curves_by_sr,
        })

    return result


# ============================================================
# ГРАФИК СРАВНЕНИЯ
# ============================================================

def plot_material_model_comparison(
    experimental_file_paths,
    coeffs_K=None,
    coeffs_m=None,
    alt_file_paths=None,
    alt_groups=None,
    label_model="Simultaneous polynomial fit",
    label_alt="Previous method",
    save_path=None,
    rmse_save_path=None,
    show=True,

    # ===================== НАСТРОЙКИ ШРИФТА =====================
    font_family="Times New Roman",
    base_fontsize=12,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    legend_title_fontsize=13,
    # ============================================================
):
    """
    Строит сравнение:
    1. экспериментальных кривых;
    2. кривых текущей модели sigma = K(eps) * edot^m(eps), если коэффициенты заданы;
    3. любого количества дополнительных групп кривых.

    Также считает RMSE всех методов относительно эксперимента.
    """

    # ============================================================
    # НАСТРОЙКА ШРИФТА
    # ============================================================

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = base_fontsize
    plt.rcParams["axes.labelsize"] = label_fontsize
    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize
    plt.rcParams["legend.fontsize"] = legend_fontsize
    plt.rcParams["legend.title_fontsize"] = legend_title_fontsize
    plt.rcParams["mathtext.fontset"] = "stix"

    # ============================================================
    # ПОДГОТОВКА ДАННЫХ
    # ============================================================

    experimental_file_paths = [Path(p) for p in experimental_file_paths]

    alt_groups = normalize_alt_groups(
        alt_groups=alt_groups,
        alt_file_paths=alt_file_paths,
        label_alt=label_alt,
    )

    has_model = coefficients_are_defined(coeffs_K, coeffs_m)

    if has_model:
        coeffs_K = np.asarray(coeffs_K, dtype=float)
        coeffs_m = np.asarray(coeffs_m, dtype=float)
    else:
        print("Коэффициенты K или m не заданы — кривые полиномиальной модели не будут построены.")

    # ------------------------------------------------------------
    # Читаем экспериментальные кривые
    # ------------------------------------------------------------

    experimental_curves = []

    for path in experimental_file_paths:
        df = read_curve_file(path)
        sr = float(df["strain_rate"].iloc[0])

        experimental_curves.append({
            "path": path,
            "strain_rate": sr,
            "data": df,
        })

    experimental_curves = sorted(
        experimental_curves,
        key=lambda item: item["strain_rate"]
    )

    # ------------------------------------------------------------
    # Читаем дополнительные группы кривых
    # ------------------------------------------------------------

    alt_groups_data = read_alt_groups(alt_groups)

    # ------------------------------------------------------------
    # Считаем RMSE
    # ------------------------------------------------------------

    rmse_df = build_rmse_table(
        experimental_curves=experimental_curves,
        alt_groups_data=alt_groups_data,
        has_model=has_model,
        coeffs_K=coeffs_K,
        coeffs_m=coeffs_m,
        label_model=label_model,
    )

    print_rmse_table(rmse_df)
    save_rmse_table(rmse_df, rmse_save_path)

    # ------------------------------------------------------------
    # Цвета: одна скорость деформации — один цвет
    # ------------------------------------------------------------

    all_strain_rates = set(item["strain_rate"] for item in experimental_curves)

    for group in alt_groups_data:
        all_strain_rates |= set(group["curves_by_sr"].keys())

    all_strain_rates = sorted(all_strain_rates)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colors = {
        sr: color_cycle[i % len(color_cycle)]
        for i, sr in enumerate(all_strain_rates)
    }

    # ------------------------------------------------------------
    # Построение
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 6))

    for item in experimental_curves:
        df = item["data"]
        sr = item["strain_rate"]
        color = colors[sr]

        eps_exp = df["eps"].to_numpy(dtype=float)
        sigma_exp = df["sigma"].to_numpy(dtype=float)

        ax.plot(
            eps_exp,
            sigma_exp,
            linestyle="None",
            marker="o",
            markersize=4,
            color=color,
            alpha=0.75,
        )

        if has_model:
            eps_grid = np.linspace(eps_exp.min(), eps_exp.max(), 400)

            sigma_model = material_model(
                eps=eps_grid,
                strain_rate=sr,
                coeffs_K=coeffs_K,
                coeffs_m=coeffs_m,
            )

            ax.plot(
                eps_grid,
                sigma_model,
                linestyle="-",
                linewidth=2.0,
                color=color,
            )

    for group in alt_groups_data:
        for sr, item in group["curves_by_sr"].items():
            df_alt = item["data"]
            color = colors[sr]

            eps_alt = df_alt["eps"].to_numpy(dtype=float)
            sigma_alt = df_alt["sigma"].to_numpy(dtype=float)

            ax.plot(
                eps_alt,
                sigma_alt,
                linestyle=group["linestyle"],
                linewidth=group["linewidth"],
                color=color,
            )

    # ------------------------------------------------------------
    # Оформление осей
    # ------------------------------------------------------------

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\sigma$")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Легенда 1: цвет = скорость деформации
    # ------------------------------------------------------------

    color_legend_handles = []

    for sr in all_strain_rates:
        color_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[sr],
                linewidth=3,
                label=rf"$\dot{{\varepsilon}} = {sr:.0e}$",
            )
        )

    legend_colors = ax.legend(
        handles=color_legend_handles,
        title=r"Скорость деформации, $с^{-1}$",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    ax.add_artist(legend_colors)

    # ------------------------------------------------------------
    # Легенда 2: стиль = метод
    # ------------------------------------------------------------

    style_legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Эксперимент",
        )
    ]

    if has_model:
        style_legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=2,
                label=label_model,
            )
        )

    for group in alt_groups_data:
        style_legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=group["linestyle"],
                linewidth=group["linewidth"],
                label=group["label"],
            )
        )

    ax.legend(
        handles=style_legend_handles,
        title="Маркеры",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0.0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, rmse_df


if __name__ == "__main__":

    # ============================================================
    # Экспериментальные кривые
    # ============================================================

    EXPERIMENTAL_FILE_PATHS = [
        r"data/material/test_real_data/cut_data/1e-4.txt",
        r"data/material/test_real_data/cut_data/4e-4.txt",
        r"data/material/test_real_data/cut_data/8e-4.txt",
    ]

    # ============================================================
    # Первая дополнительная группа кривых
    # ============================================================

    ALT_FILE_PATHS = [
        r"data/material/Txt_files/K_m_from_NN3/1e-4.txt",
        r"data/material/Txt_files/K_m_from_NN4/4e-4.txt",
        r"data/material/Txt_files/K_m_from_NN4/8e-4.txt",
    ]

    # ============================================================
    # Вторая дополнительная группа кривых
    # ============================================================

    ALT_2_FILE_PATHS = [
        r"data/material/Txt_files/K_m_from_NN5/1e-4.txt",
        r"data/material/Txt_files/K_m_from_NN5/4e-4.txt",
        r"data/material/Txt_files/K_m_from_NN5/8e-4.txt",
    ]

    ALT_3_FILE_PATHS = [
        r"data/material/Txt_files/K_m_fromNN_poly/1e-4.txt",
        r"data/material/Txt_files/K_m_fromNN_poly/4e-4.txt",
        r"data/material/Txt_files/K_m_fromNN_poly/8e-4.txt",
    ]

    ALT_GROUPS = [
        {
            "file_paths": ALT_FILE_PATHS,
            "label": "Нейронная сеть",
            "linestyle": "--",
            "linewidth": 2.0,
        },
        {
            "file_paths": ALT_2_FILE_PATHS,
            "label": r"Усреднение $K(\varepsilon)$, $m(\varepsilon)$",
            "linestyle": "-.",
            "linewidth": 2.0,
        },
        {
            "file_paths": ALT_3_FILE_PATHS,
            "label": r"$K(\varepsilon)$, $m(\varepsilon)$",
            "linestyle": ":",
            "linewidth": 2.0,
        },
    ]

    # ============================================================
    # Коэффициенты текущей модели
    # ============================================================

    COEFFS_K = [
        # 264.455613254,
        # -177.37322503,
        # 27.0304578037,
    ]

    COEFFS_M = [
        # 0.41365193633,
        # -0.263045279559,
        # 0.128785905521,
    ]

    # ============================================================
    # НАСТРОЙКИ ШРИФТА
    # ============================================================

    FONT_FAMILY = "Times New Roman"
    BASE_FONTSIZE = 16
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 18
    LEGEND_FONTSIZE = 16
    LEGEND_TITLE_FONTSIZE = 13

    # ============================================================
    # Построение графика + RMSE
    # ============================================================

    fig, ax, rmse_df = plot_material_model_comparison(
        experimental_file_paths=EXPERIMENTAL_FILE_PATHS,
        coeffs_K=COEFFS_K,
        coeffs_m=COEFFS_M,
        alt_groups=ALT_GROUPS,
        label_model="Полиномиальная модель",
        save_path=r"data/material/Graphics/comparison_poly_rmse.png",

        # Таблица RMSE
        rmse_save_path=r"data/material/Graphics/comparison_rmse.xlsx",

        show=True,

        # ===== Настройки шрифта =====
        font_family=FONT_FAMILY,
        base_fontsize=BASE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        tick_fontsize=TICK_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE,
        legend_title_fontsize=LEGEND_TITLE_FONTSIZE,
    )