from pathlib import Path
import pandas as pd
import numpy as np

# ===================== НАСТРОЙКИ =====================
# Список путей к .txt файлам (каждый файл содержит два столбца: деформация и напряжение)
# Десятичный разделитель в файлах – запятая (автоматически заменяется на точку)
LIST_OF_FILES = [
    r"data/material/test_real_data/approx_data/1e-4_10_points.txt",
    r"data/material/test_real_data/approx_data/4e-4_10_points.txt",
    r"data/material/test_real_data/approx_data/8e-4_10_points.txt",
]

# Значения скорости деформации для каждого файла (порядок соответствует LIST_OF_FILES)
ER_VALUES = [0.0001, 0.0004, 0.0008]   # можно задавать числами или строками (0p0001)
#ER_VALUES = [0.01, 0.1, 0.0001]

# Путь к выходному CSV-файлу
OUT_CSV = r"data/material/test_real_data/dataset/real_data_10_points.csv"

# Количество точек N, которое нужно взять из каждого файла (если None – берётся минимальная длина)
N_POINTS = 10

# Разделитель в выходном CSV (для Excel удобно ";")
CSV_SEP = ";"

# =====================================================


def read_txt_curve(filepath):
    """
    Читает текстовый файл с двумя столбцами (деформация, напряжение).
    Десятичный разделитель – запятая (заменяется на точку).
    Возвращает (список деформаций, список напряжений).
    """
    strains = []
    stresses = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # Замена запятой на точку
                x_str = parts[0].replace(',', '.')
                y_str = parts[1].replace(',', '.')
                try:
                    x = float(x_str)
                    y = float(y_str)
                    strains.append(x)
                    stresses.append(y)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return None, None
    return strains, stresses


def to_p_str(x):
    """Преобразует число в строку с 'p' вместо точки (для имён колонок)."""
    s = str(x)
    return s.replace('.', 'p')


def build_column_names(er_values, n_points):
    """
    Формирует список имён колонок:
    e_1, ..., e_N,
    затем для каждого er:
        s_1_erX, ..., s_N_erX, er_X
    """
    cols = [f"e_{i}" for i in range(1, n_points + 1)]
    for er in er_values:
        er_p = to_p_str(er)
        for i in range(1, n_points + 1):
            cols.append(f"s_{i}_er{er_p}")
        cols.append(f"er_{er_p}")
    return cols


def make_dataset(list_of_files, er_values, output_csv, n_points=None, csv_sep=";"):
    """
    Создаёт CSV-датасет из списка txt-файлов.
    """
    if len(list_of_files) != len(er_values):
        raise ValueError("Длина list_of_files и er_values должна совпадать")

    # Чтение всех файлов
    all_strains = []   # список списков деформаций
    all_stresses = []  # список списков напряжений
    for fpath in list_of_files:
        strains, stresses = read_txt_curve(fpath)
        if strains is None or stresses is None:
            raise ValueError(f"Не удалось прочитать файл {fpath}")
        all_strains.append(strains)
        all_stresses.append(stresses)

    # Определяем количество точек N
    lengths = [len(s) for s in all_strains]
    if n_points is None:
        n_points = min(lengths)  # берём минимальную длину, чтобы не было NaN
        print(f"Автоматически выбрано N = {n_points} (минимальная длина среди файлов)")
        # Обрезаем все до минимальной длины
        all_strains = [s[:n_points] for s in all_strains]
        all_stresses = [s[:n_points] for s in all_stresses]
    else:
        # Обрезаем или дополняем NaN до n_points
        for i in range(len(all_strains)):
            if len(all_strains[i]) < n_points:
                all_strains[i] = all_strains[i] + [np.nan] * (n_points - len(all_strains[i]))
                all_stresses[i] = all_stresses[i] + [np.nan] * (n_points - len(all_stresses[i]))
            elif len(all_strains[i]) > n_points:
                all_strains[i] = all_strains[i][:n_points]
                all_stresses[i] = all_stresses[i][:n_points]

    # Вычисляем средние деформации по всем файлам
    avg_strains = np.mean(all_strains, axis=0).tolist()  # список длины n_points

    # Формируем одну строку данных
    row = avg_strains.copy()
    for stresses in all_stresses:
        row.extend(stresses)        # s_1 ... s_N
    row.extend(er_values)           # er для каждого файла (в конце)

    # Заголовки
    columns = build_column_names(er_values, n_points)

    # Создаём DataFrame (одна строка)
    df_out = pd.DataFrame([row], columns=columns)

    # Сохраняем
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, sep=csv_sep, index=False, encoding="utf-8-sig")
    print(f"CSV сохранён в {out_path}")


if __name__ == "__main__":
    make_dataset(
        list_of_files=LIST_OF_FILES,
        er_values=ER_VALUES,
        output_csv=OUT_CSV,
        n_points=N_POINTS,
        csv_sep=CSV_SEP
    )