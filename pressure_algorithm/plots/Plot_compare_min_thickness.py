import numpy as np
import matplotlib.pyplot as plt
from compute_distances import compute_distances_from_folder


def plot_min_thickness_vs_param(folders_list, x_values, n=1,
                                x_l=None, x_r=None,
                                hline_folders=None, hline_labels=None,
                                label='Минимальная толщина',
                                output_image_path=None,
                                xlabel='Параметр', ylabel='Минимальная толщина, мм',
                                title=None, show=True,
                                font_family='Times New Roman',
                                base_fontsize=12,
                                label_fontsize=14,
                                title_fontsize=16,
                                legend_fontsize=12,
                                tick_fontsize=12):
    """
    Строит график зависимости минимальной толщины от параметра для нескольких папок,
    а также может добавлять горизонтальные линии для дополнительных папок.

    Дополнительные параметры шрифта:
        font_family: семейство шрифта, например 'Times New Roman', 'Arial', 'DejaVu Serif'
        base_fontsize: базовый размер шрифта
        label_fontsize: размер подписей осей
        title_fontsize: размер заголовка
        legend_fontsize: размер шрифта легенды
        tick_fontsize: размер подписей делений осей
    """
    if len(folders_list) != len(x_values):
        raise ValueError("Длины folders_list и x_values должны совпадать")

    # ===== Настройка шрифта =====
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = base_fontsize
    plt.rcParams['axes.labelsize'] = label_fontsize
    plt.rcParams['axes.titlesize'] = title_fontsize
    plt.rcParams['legend.fontsize'] = legend_fontsize
    plt.rcParams['xtick.labelsize'] = tick_fontsize
    plt.rcParams['ytick.labelsize'] = tick_fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'  # чтобы формулы выглядели ближе к Times

    def compute_min_thickness(folder):
        try:
            xs, dists = compute_distances_from_folder(folder, n=n)

            if x_l is not None and x_r is not None:
                mask = (xs >= x_l) & (xs <= x_r)
                if np.any(mask):
                    dists = dists[mask]
                else:
                    return np.nan

            return np.min(dists)

        except Exception as e:
            print(f"Ошибка при обработке {folder}: {e}")
            return np.nan

    # Основные точки
    min_thicknesses = []
    for folder in folders_list:
        val = compute_min_thickness(folder)
        min_thicknesses.append(val)

        if not np.isnan(val):
            print(f"{folder}: минимальная толщина = {val:.4f} мм")
        else:
            print(f"{folder}: ошибка")

    # Горизонтальные линии
    hline_values = []
    if hline_folders is not None:
        if hline_labels is None or len(hline_folders) != len(hline_labels):
            raise ValueError("hline_folders и hline_labels должны быть одинаковой длины")

        for folder in hline_folders:
            val = compute_min_thickness(folder)
            hline_values.append(val)

            if not np.isnan(val):
                print(f"Гор. линия {folder}: {val:.4f} мм")
            else:
                print(f"Гор. линия {folder}: ошибка")

    # Построение графика
    plt.figure(figsize=(10, 6))

    valid = ~np.isnan(min_thicknesses)
    if np.any(valid):
        plt.plot(
            np.array(x_values)[valid],
            np.array(min_thicknesses)[valid],
            'o--',
            linewidth=2,
            markersize=8,
            color='b',
            label=label
        )
    else:
        print("Нет данных для построения основной кривой")

    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    linestyles = ['-', '-', '--', ':']
    j = 0
    if hline_folders is not None:
        for i, (val, hlbl) in enumerate(zip(hline_values, hline_labels)):
            if not np.isnan(val):
                color = colors[i % len(colors)]
                plt.axhline(
                    y=val,
                    color=color,
                    linestyle=linestyles[j],
                    linewidth=2,
                    label=hlbl
                )
                j += 1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.grid(True, linestyle='--', alpha=0.7)

    if label is not None or hline_folders is not None:
        plt.legend()

    plt.tight_layout()

    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_image_path}")

    if show:
        plt.show()

    plt.close()

    return min_thicknesses


# ========== Пример использования ==========
if __name__ == "__main__":
    # Основные папки
    folders = [
        # r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_7/increment_6046',
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
    ]
    x_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

    # Дополнительные папки для горизонтальных линий (пример)
    hline_dirs = [
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt/increment_3830',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/algorithm_try_0_8/increment_2034',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/smooth_alg/increment_2136',
    ]
    hline_names = [r'$\dot{\varepsilon}_{opt}$', 'Алгоритм', 'Сглаженный алгоритм']

    # Вызов функции с фильтром по X от 20 до 80 (пример)
    min_thicks = plot_min_thickness_vs_param(
        folders, x_vals, n=1,
        x_l=0, x_r=200,
        hline_folders=hline_dirs,
        hline_labels=hline_names,
        label=None,
        xlabel='Давление, МПа',
        ylabel='Минимальная толщина, мм',
        output_image_path=r"data/forming/Free_bulg_test/Graphics/Min_thickness/smirnov_compare_80_percent_smooth_alg.png",
        show=False,

        # настройки шрифта
        font_family='Times New Roman',
        base_fontsize=16,
        label_fontsize=18,
        title_fontsize=18,
        legend_fontsize=16,
        tick_fontsize=16
    )