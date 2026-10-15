from pathlib import Path
import numpy as np

# ===================== НАСТРОЙКИ =====================
# Список путей к .txt файлам (каждый содержит два столбца: деформация и напряжение)
LIST_OF_FILES = [
    r"data/material/test_real_data/cut_data/1e-4.txt",
    r"data/material/test_real_data/cut_data/4e-4.txt",
    r"data/material/test_real_data/cut_data/8e-4.txt",
]

# Заданное значение деформации (x), при котором нужно найти напряжение
X_TARGET = 0.3

# Путь к выходному файлу (два столбца: деформация_заданная, напряжение)
OUT_FILE = r"data/material/test_real_data/data_for_find_K_m/values_at_deform_0_35.txt"

# Разделитель в выходном файле (пробел или табуляция)
OUT_SEP = " "

# Разрешить экстраполяцию, если x_target выходит за пределы диапазона данных
EXTRAPOLATE = False

# =====================================================


def read_txt_curve(filepath):
    """
    Читает текстовый файл с двумя столбцами (деформация, напряжение).
    Поддерживает десятичный разделитель как точку, так и запятую.
    Возвращает (массив деформаций, массив напряжений), отсортированные по деформации.
    """
    strains = []
    stresses = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f"Предупреждение: строка {line_num} в {filepath} содержит менее двух чисел, пропущена.")
                    continue
                # Замена запятой на точку, если есть
                x_str = parts[0].replace(',', '.')
                y_str = parts[1].replace(',', '.')
                try:
                    x = float(x_str)
                    y = float(y_str)
                    strains.append(x)
                    stresses.append(y)
                except ValueError:
                    print(f"Предупреждение: строка {line_num} в {filepath} не может быть преобразована в числа, пропущена.")
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return None, None
    if not strains:
        print(f"В файле {filepath} не найдено корректных точек.")
        return None, None

    # Сортировка по деформации (на случай, если данные не упорядочены)
    sorted_indices = np.argsort(strains)
    strains_sorted = np.array(strains)[sorted_indices]
    stresses_sorted = np.array(stresses)[sorted_indices]
    return strains_sorted, stresses_sorted


def find_value_at_x(strain, stress, x_target, extrapolate=False):
    """
    Возвращает интерполированное значение напряжения при заданной деформации x_target.
    Если extrapolate=False и x_target вне диапазона данных, возвращает None.
    """
    if len(strain) == 0:
        return None
    if x_target < strain[0] or x_target > strain[-1]:
        if not extrapolate:
            print(f"Предупреждение: x_target={x_target} вне диапазона данных [{strain[0]}, {strain[-1]}]. Пропускаем.")
            return None
    # Линейная интерполяция (numpy.interp поддерживает экстраполяцию по умолчанию)
    y = np.interp(x_target, strain, stress)
    return y


def main():
    results = []  # список кортежей (x_target, y_value)
    for filepath in LIST_OF_FILES:
        strain, stress = read_txt_curve(filepath)
        if strain is None:
            continue
        y_value = find_value_at_x(strain, stress, X_TARGET, extrapolate=EXTRAPOLATE)
        if y_value is None:
            continue
        results.append((X_TARGET, y_value))
        print(f"{Path(filepath).name}: при x = {X_TARGET:.6f}, y = {y_value:.6f}")

    if not results:
        print("Не найдено ни одного корректного значения.")
        return

    # Запись выходного файла
    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for x_val, y_val in results:
            f.write(f"{x_val}{OUT_SEP}{y_val}\n")
    print(f"\nФайл со значениями сохранён в {out_path}")


if __name__ == "__main__":
    main()