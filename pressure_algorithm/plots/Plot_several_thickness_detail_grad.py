import matplotlib.pyplot as plt
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, r"data/forming/Py_scripts")
from compute_distances import compute_distances_from_folder


def extract_pressure_from_label(label):
    """
    Извлекает значение давления из подписи.

    Примеры:
        '1.0 МПа' -> 1.0
        'pressure_1_2' -> 1.2
        'pressure_2.4' -> 2.4
    """
    label = str(label)

    # Вариант для подписей вида "1.2 МПа"
    match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*МПа", label, flags=re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", "."))

    # Вариант для строк вида "pressure_1_2" или "pressure_1.2"
    match = re.search(r"pressure[_\s-]*([0-9]+)(?:[_.]([0-9]+))?", label, flags=re.IGNORECASE)
    if match:
        whole = match.group(1)
        frac = match.group(2)
        return float(f"{whole}.{frac}") if frac is not None else float(whole)

    return None


def plot_multiple_distances(folders_list, labels_list, output_image_path, n=1,
                            time_round=6, show_time=True,
                            linewidth=2, markersize=4, title=None, style=None,
                            pressure_values=None, cmap_name="coolwarm",
                            show_pressure_colorbar=True,
                            special_color=None,
                            font_family='Times New Roman',
                            base_fontsize=12,
                            label_fontsize=14,
                            title_fontsize=16,
                            legend_fontsize=12,
                            tick_fontsize=12,
                            colorbar_fontsize=12):
    """
    Строит на одном графике минимальные расстояния для нескольких папок.

    Цвета:
        1) Если для кривой удалось определить давление, цвет берётся из
           градиентной шкалы давления.
        2) Если давление определить нельзя, например для подписи
           r'$\\dot{\\varepsilon}_{opt}$', кривая рисуется отдельным цветом
           special_color. Обычно такие кривые задаются штриховкой через style.

    Параметры:
        folders_list: список путей к папкам, каждая содержит top.txt и bottom.txt
        labels_list: список подписей для легенды, должен быть той же длины
        output_image_path: путь для сохранения итогового графика
        n: радиус окна для compute_distances_from_folder
        time_round: число знаков после запятой для времени
        show_time: если True, добавляет значение времени из top.txt в легенду
        linewidth: толщина линии
        markersize: размер маркеров
        title: заголовок графика
        style: список стилей линий, например ['-', '-', '--']
        pressure_values: список значений давления. Для специальных кривых можно
                         поставить None, например [0.5, 0.75, ..., 4.0, None]
        cmap_name: имя matplotlib colormap, например 'coolwarm', 'turbo', 'plasma'
        show_pressure_colorbar: если True, добавляет шкалу давления
        special_color: цвет или список цветов для кривых без численного давления

    Настройки шрифта:
        font_family: семейство шрифта, например 'Times New Roman', 'Arial', 'DejaVu Serif'
        base_fontsize: базовый размер шрифта
        label_fontsize: размер подписей осей
        title_fontsize: размер заголовка
        legend_fontsize: размер шрифта легенды
        tick_fontsize: размер подписей делений осей
        colorbar_fontsize: размер подписи цветовой шкалы
    """
    if len(folders_list) != len(labels_list):
        raise ValueError("folders_list и labels_list должны быть одинаковой длины")

    # ===== Настройка шрифта =====
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = base_fontsize
    plt.rcParams['axes.labelsize'] = label_fontsize
    plt.rcParams['axes.titlesize'] = title_fontsize
    plt.rcParams['legend.fontsize'] = legend_fontsize
    plt.rcParams['xtick.labelsize'] = tick_fontsize
    plt.rcParams['ytick.labelsize'] = tick_fontsize

    # Чтобы математические формулы в легенде выглядели ближе к Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix'

    if style is None:
        style = ['-'] * len(folders_list)

    if len(style) < len(folders_list):
        raise ValueError("Список style должен быть не короче folders_list")

    # Получаем давления: либо из отдельного списка, либо автоматически из labels_list.
    # Для строковых режимов типа r'$\\dot{\\varepsilon}_{opt}$' будет None.
    if pressure_values is None:
        pressure_values = [extract_pressure_from_label(label) for label in labels_list]

    if len(pressure_values) != len(folders_list):
        raise ValueError("pressure_values должен иметь ту же длину, что folders_list")

    # В градиентную шкалу включаем только те кривые, где есть численное давление.
    numeric_pressures = [p for p in pressure_values if p is not None]

    if show_pressure_colorbar and not numeric_pressures:
        raise ValueError(
            "Не найдено ни одного численного давления. "
            "Colorbar давления построить нельзя."
        )

    # Цвета для специальных кривых
    if special_color is None:
        special_color = ["black"]

    if isinstance(special_color, str):
        special_color = [special_color]

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.get_cmap(cmap_name)

    if numeric_pressures:
        p_min = min(numeric_pressures)
        p_max = max(numeric_pressures)

        if p_min == p_max:
            norm = plt.Normalize(vmin=p_min - 0.5, vmax=p_max + 0.5)
        else:
            norm = plt.Normalize(vmin=p_min, vmax=p_max)
    else:
        norm = None

    special_index = 0

    for idx, (folder, label, pressure) in enumerate(zip(folders_list, labels_list, pressure_values)):
        # Вычисляем расстояния
        x_vals, dists = compute_distances_from_folder(folder, n=n)

        # Извлекаем время из top.txt для этой папки
        time_value = None
        if show_time:
            top_path = os.path.join(folder, 'top.txt')
            try:
                with open(top_path, 'r') as f:
                    first_line = f.readline().strip()
                    match = re.search(r"Time:\s*([0-9.eE+-]+)", first_line)
                    if match:
                        time_value = float(match.group(1))
            except Exception:
                pass

        legend_label = label
        if show_time and time_value is not None:
            legend_label += f" ({time_value:.{time_round}f} с)"

        # Если давление численное — цвет по шкале давления.
        # Если давление None — отдельный цвет, стиль берётся из style[idx], например '--'.
        if pressure is None:
            curve_color = special_color[special_index % len(special_color)]
            special_index += 1
        else:
            curve_color = cmap(norm(pressure))

        ax.plot(
            x_vals,
            dists,
            style[idx],
            linewidth=linewidth,
            markersize=markersize,
            color=curve_color,
            label=legend_label
        )

    ax.set_xlabel('x, мм')
    ax.set_ylabel('Толщина, мм')

    if title:
        ax.set_title(title)

    ax.grid(True, linestyle='--', alpha=0.7)

    # Легенда нужна для специальных режимов и времени.
    # Цветовое соответствие давлению показывает colorbar.
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.18, 0.5),
        fontsize=legend_fontsize
    )

    if show_pressure_colorbar and numeric_pressures:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Давление, МПа', fontsize=colorbar_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    fig.tight_layout()
    fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Совмещённый график сохранён: {output_image_path}")


# ======================================================================
# Пример использования
# ======================================================================

folders = [
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_5/increment_11470',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_75/increment_5250',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1/increment_3204',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_25/increment_2254',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_5/increment_1678',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_75/increment_1328',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2/increment_1052',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_25/increment_872',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_5/increment_706',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_75/increment_610',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3/increment_508',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_25/increment_454',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_5/increment_386',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_75/increment_358',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_4/increment_306',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt/increment_3830',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/algorithm_try_0_8/increment_2034',
]

labels = [
    '0.5 МПа',
    '0.75 МПа',
    '1.0 МПа',
    '1.25 МПа',
    '1.5 МПа',
    '1.75 МПа',
    '2.0 МПа',
    '2.25 МПа',
    '2.5 МПа',
    '2.75 МПа',
    '3.0 МПа',
    '3.25 МПа',
    '3.5 МПа',
    '3.75 МПа',
    '4.0 МПа',
    r'$\dot{\varepsilon}_{opt}$',
    "Алгоритм"
]

style = [
    '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
    '--', '--'
]

output = r"data/forming/Free_bulg_test/Graphics/Thickness/smirnov_compare_free_bul_grad_0_8.png"

plot_multiple_distances(
    folders,
    labels,
    output,
    n=1,
    time_round=1,
    show_time=True,
    linewidth=1,
    markersize=4,
    style=style,
    cmap_name="coolwarm",
    special_color=["green", "black"],

    # ===== Настройки шрифта =====
    font_family='Times New Roman',
    base_fontsize=16,
    label_fontsize=18,
    title_fontsize=18,
    legend_fontsize=14,
    tick_fontsize=16,
    colorbar_fontsize=16
)