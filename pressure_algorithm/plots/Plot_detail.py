import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re

def plot_profiles(folder_path, output_image_path, round_time=6):
    """
    Строит профили верхней и нижней границ детали по данным из файлов top.txt и bottom.txt.
    Время отображается отдельной строкой в легенде.

    Параметры:
        folder_path (str): путь к папке, содержащей top.txt и bottom.txt
        output_image_path (str): путь для сохранения графика
        round_time (int): количество знаков после запятой для отображения времени
    """
    top_path = os.path.join(folder_path, "top.txt")
    bottom_path = os.path.join(folder_path, "bottom.txt")

    if not os.path.exists(top_path):
        raise FileNotFoundError(f"Файл {top_path} не найден")
    if not os.path.exists(bottom_path):
        raise FileNotFoundError(f"Файл {bottom_path} не найден")

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

    time_top, x_top, y_top = read_coords(top_path)
    time_bottom, x_bottom, y_bottom = read_coords(bottom_path)
    time_val = time_top if time_top is not None else time_bottom

    # Сортировка
    x_top, y_top = zip(*sorted(zip(x_top, y_top)))
    x_bottom, y_bottom = zip(*sorted(zip(x_bottom, y_bottom)))

    plt.figure(figsize=(10, 6))
    line1, = plt.plot(x_top, y_top, '-', linewidth=2, color='b')
    line2, = plt.plot(x_bottom, y_bottom, '-', linewidth=2, color='b')

    # Пустая линия для отображения времени в легенде
    if time_val is not None:
        time_str = f"Время = {time_val:.{round_time}f} с"
        # Создаём фиктивную линию с alpha=0 (невидимая)
        dummy_line = plt.plot([], [], alpha=0, label=time_str)[0]
        # Можно также создать патч
        # dummy_patch = mpatches.Patch(color='none', label=time_str)
        # Легенда автоматически включит dummy_line
    else:
        dummy_line = None

    plt.xlabel('x, мм', fontsize=12)
    plt.ylabel('y, мм', fontsize=12)
    # plt.title('Профиль детали', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"График сохранён: {output_image_path}")


#
plot_profiles(r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_4/increment_6092",
                r"data/forming/Free_bulg_test/Graphics/Detail/sm_pressure_4", 1)