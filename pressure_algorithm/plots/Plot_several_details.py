import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import re

def plot_profiles(folder_paths, output_image_path, round_time=6):
    """
    Строит на одном рисунке профили верхней и нижней границ детали
    для нескольких наборов данных (каждый набор – отдельная папка с top.txt и bottom.txt).
    Разные наборы отображаются разными цветами, время каждого выводится в легенде.

    Параметры:
        folder_paths (str или list[str]): путь к папке или список путей к папкам.
        output_image_path (str): путь для сохранения графика.
        round_time (int): количество знаков после запятой для отображения времени.
    """
    # Приводим к списку, если передана одна папка
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    # Цвета для разных наборов
    colors = plt.cm.tab10(np.linspace(0, 1, len(folder_paths)))

    def read_coords(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            raise ValueError(f"Файл {filepath} пуст")
        time_line = lines[0].strip()
        match = re.search(r"Time:\s*([0-9.]+)", time_line)
        time_val = float(match.group(1)) if match else None

        x_vals, y_vals = [], []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    pass
        return time_val, x_vals, y_vals

    plt.figure(figsize=(10, 6))
    legend_patches = []

    for i, folder in enumerate(folder_paths):
        top_path = os.path.join(folder, "top.txt")
        bottom_path = os.path.join(folder, "bottom.txt")

        if not os.path.exists(top_path):
            raise FileNotFoundError(f"Файл {top_path} не найден")
        if not os.path.exists(bottom_path):
            raise FileNotFoundError(f"Файл {bottom_path} не найден")

        time_top, x_top, y_top = read_coords(top_path)
        time_bottom, x_bottom, y_bottom = read_coords(bottom_path)
        time_val = time_top if time_top is not None else time_bottom

        # Сортировка координат
        x_top, y_top = zip(*sorted(zip(x_top, y_top)))
        x_bottom, y_bottom = zip(*sorted(zip(x_bottom, y_bottom)))

        color = colors[i]
        # Верхняя граница – сплошная, нижняя – штриховая (того же цвета)
        plt.plot(x_top, y_top, '-', color=color, linewidth=1)
        plt.plot(x_bottom, y_bottom, '-', color=color, linewidth=1)

        # Метка для легенды: имя папки + время
        folder_label = os.path.basename(folder)
        if time_val is not None:
            label = f"{time_val:.{round_time}f} с"
        else:
            label = f"{folder_label}: время не найдено"
        legend_patches.append(mpatches.Patch(color=color, label=label))

    plt.xlabel('x, мм', fontsize=12)
    plt.ylabel('y, мм', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=legend_patches)
    plt.tight_layout()

    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"График сохранён: {output_image_path}")


# Несколько папок – все на одном рисунке
plot_profiles(
    [
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_7/increment_2910',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1/increment_1630',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_5/increment_840',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2/increment_540',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_5/increment_380',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3/increment_290',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_5/increment_230',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_4/increment_190',
        r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt/increment_2420',
    ],
    r"data/forming/Free_bulg_test/Graphics/Detail/smirnov_profiles_80_mm",
    round_time=2
)