from pathlib import Path
import pandas as pd


# генерит датасет вида
# (K, m, e_1_er0p001, ..., e_N_er0p001, s_1_er0p001, ..., s_N_er0p001, er_0p001, e_1_er0p002, ..., e_N_er0p002, s_1_er0p002, ..., s_N_er0p002, er_0p002,
#  ...)

# ===================== НАСТРОЙКИ =====================

# Папка с .txt файлами (Strain, Stress_RF_RP, таб-разделитель)
TXT_DIR = r"data/material/all_txt_10"

# Число точек в каждом файле
N_POINTS = 10

# Путь к выходному .csv
OUT_CSV = fr"data/material/datasets/all_curves_repeat_er_{N_POINTS}.csv"

# Массивы значений K, m, er
# Можно задавать числом или строкой; в имени файла всё равно будет форма с 'p'
K_VALUES = [i for i in range(300, 901, 10)]
M_VALUES = [i / 100 for i in range(30, 81)]
# ER_VALUES = ["0p001", "0p002", "0p004", "0p01", "0p015", "0p03", "0p06", "0p1"]
ER_VALUES = ["0p002","0p01", "0p03", "0p1"]

# Разделитель в итоговом CSV (можно ";" если любишь по-экселовски)
CSV_SEP = ";"


# =====================================================


def to_p_str(x) -> str:
    """
    Преобразует число или строку в формат без точки, с 'p':
    0.3 -> '0p3', '0p01' остаётся '0p01'.
    """
    if isinstance(x, str):
        # если уже в формате с 'p', оставляем
        if "p" in x:
            return x
        return x.replace(".", "p")
    else:
        s = str(x)
        return s.replace(".", "p")


def to_float_maybe(x):
    """
    Для K и m в CSV. Если можем — приводим к float, иначе оставляем строку.
    """
    if isinstance(x, (int, float)):
        return x
    s = str(x)
    if "p" in s:
        s = s.replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return x


def er_to_float(er) -> float:
    """
    Преобразует er в числовое значение скорости деформации.
    '0p001' -> 0.001, '0.01' -> 0.01, 0.03 -> 0.03
    """
    if isinstance(er, (int, float)):
        return float(er)
    s = str(er)
    if "p" in s:
        s = s.replace("p", ".")
    return float(s)


def build_column_names(er_values, n_points: int):
    """
    Формирует список имён колонок:
    K, m,
    e_1_erX, ..., e_N_erX, s_1_erX, ..., s_N_erX, er_X
    для всех er.
    """
    cols = ["K", "m"]
    for er in er_values:
        er_p = to_p_str(er)
        # Strain
        for i in range(1, n_points + 1):
            cols.append(f"e_{i}_er{er_p}")
        # Stress
        for i in range(1, n_points + 1):
            cols.append(f"s_{i}_er{er_p}")
        # Числовое значение скорости деформации для этого блока
        cols.append(f"er_{er_p}")
    return cols


def make_dataset(txt_dir: str,
                 out_csv: str,
                 k_values,
                 m_values,
                 er_values,
                 n_points: int,
                 csv_sep: str = ";") -> None:
    base_path = Path(txt_dir)
    rows = []

    for K in k_values:
        for m in m_values:
            row = []
            # В CSV K и m запишем как числа (по возможности)
            row.append(to_float_maybe(K))
            row.append(to_float_maybe(m))

            for er in er_values:
                K_str_p = to_p_str(K)
                m_str_p = to_p_str(m)
                er_str_p = to_p_str(er)

                filename = f"job_K{K_str_p}_m{m_str_p}_er{er_str_p}.txt"
                file_path = base_path / filename

                # Инициализируем NaN-блоки на случай любых проблем
                strain_vals = [float("nan")] * n_points
                stress_vals = [float("nan")] * n_points

                if not file_path.exists():
                    print(f"⚠ Файл {file_path} не найден, заполняю NaN для этой комбинации.")
                else:
                    try:
                        df = pd.read_csv(file_path, sep="\t")
                    except Exception as e:
                        print(f"⚠ Не удалось прочитать {file_path}: {e}")
                    else:
                        if not {"Strain", "Stress_RF_RP"}.issubset(df.columns):
                            print(f"⚠ В файле {file_path} нет нужных колонок Strain / Stress_RF_RP.")
                        else:
                            if len(df) < n_points:
                                print(
                                    f"⚠ В файле {file_path} только {len(df)} точек, ожидается {n_points}. "
                                    f"Добиваю NaN."
                                )

                            s_vals = df["Strain"].values[:n_points]
                            t_vals = df["Stress_RF_RP"].values[:n_points]

                            # Преобразуем в list и добиваем NaN при необходимости
                            s_vals = list(s_vals)
                            t_vals = list(t_vals)
                            if len(s_vals) < n_points:
                                pad = n_points - len(s_vals)
                                s_vals += [float("nan")] * pad
                                t_vals += [float("nan")] * pad

                            strain_vals = s_vals
                            stress_vals = t_vals

                # Добавляем в строку:
                # - сначала все Strain
                # - затем все Stress
                # - затем числовое значение er
                row.extend(strain_vals)
                row.extend(stress_vals)
                row.append(er_to_float(er))

            rows.append(row)

    # Формируем DataFrame
    columns = build_column_names(er_values, n_points)
    df_out = pd.DataFrame(rows, columns=columns)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, sep=csv_sep, index=False, encoding="utf-8-sig")
    print(f"CSV сохранён в {out_path}")


def main():
    make_dataset(
        txt_dir=TXT_DIR,
        out_csv=OUT_CSV,
        k_values=K_VALUES,
        m_values=M_VALUES,
        er_values=ER_VALUES,
        n_points=N_POINTS,
        csv_sep=CSV_SEP,
    )


if __name__ == "__main__":
    main()
