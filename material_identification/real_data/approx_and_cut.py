import numpy as np
import sys

# ==================== НАСТРОЙКИ (измените здесь) ====================
INPUT_FILE = r"data/material/test_real_data/real_data/1e-4.txt"
TRIM_HEAD = 5                     # сколько точек отрезать с начала
TRIM_TAIL = 18                     # сколько точек отрезать с конца
N = 10                           # количество точек для аппроксимированной кривой
OUTPUT_TRIMMED = r"data/material/test_real_data/cut_data/1e-4.txt"
OUTPUT_APPROX = fr"data/material/test_real_data/approx_data/1e-4_{N}_points.txt"
# ====================================================================

def read_points(filepath):
    x_vals = []
    y_vals = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f"Предупреждение: строка {line_num} содержит менее двух чисел, пропускаем.")
                    continue
                try:
                    # Заменяем запятую на точку
                    x_str = parts[0].replace(',', '.')
                    y_str = parts[1].replace(',', '.')
                    x = float(x_str)
                    y = float(y_str)
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    print(f"Предупреждение: строка {line_num} не может быть преобразована в числа, пропускаем.")
    except FileNotFoundError:
        print(f"Ошибка: файл {filepath} не найден.")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)
    if not x_vals:
        print("Ошибка: не удалось прочитать ни одной точки.")
        sys.exit(1)
    return np.array(x_vals), np.array(y_vals)

def write_points(filepath, x_vals, y_vals):
    try:
        with open(filepath, 'w') as f:
            for x, y in zip(x_vals, y_vals):
                f.write(f"{x} {y}\n")
    except Exception as e:
        print(f"Ошибка при записи файла {filepath}: {e}")
        sys.exit(1)

def trim_points(x, y, head, tail):
    total = len(x)
    if head + tail >= total:
        print(f"Ошибка: сумма отрезаемых точек ({head}+{tail}) больше или равна общему числу точек ({total}).")
        sys.exit(1)
    return x[head:total-tail], y[head:total-tail]

def interpolate_curve(x, y, num_points):
    n = len(x)
    if n == 1:
        return np.full(num_points, x[0]), np.full(num_points, y[0])
    t_orig = np.arange(n)
    t_new = np.linspace(0, n-1, num_points)
    x_new = np.interp(t_new, t_orig, x)
    y_new = np.interp(t_new, t_orig, y)
    return x_new, y_new

def main():
    print(f"Чтение файла: {INPUT_FILE}")
    x_full, y_full = read_points(INPUT_FILE)
    print(f"Прочитано {len(x_full)} точек.")

    print(f"Обрезаем {TRIM_HEAD} точек с начала и {TRIM_TAIL} с конца...")
    x_trim, y_trim = trim_points(x_full, y_full, TRIM_HEAD, TRIM_TAIL)
    print(f"После обрезания: {len(x_trim)} точек.")

    write_points(OUTPUT_TRIMMED, x_trim, y_trim)
    print(f"Обрезанный набор сохранён в {OUTPUT_TRIMMED}")

    print(f"Аппроксимируем кривую до {N} точек (линейная интерполяция)...")
    x_approx, y_approx = interpolate_curve(x_trim, y_trim, N)
    print(f"Аппроксимированный набор: {len(x_approx)} точек.")

    write_points(OUTPUT_APPROX, x_approx, y_approx)
    print(f"Аппроксимированный набор сохранён в {OUTPUT_APPROX}")

if __name__ == "__main__":
    main()