from pathlib import Path
import re
import pandas as pd

# ===================== НАСТРОЙКИ =====================

# Папка с аппроксимированными данными (.txt с Strain и Stress_RF_RP)
IN_DIR = r"data/material/Txt_files/Resampled_from_modelling_10_points"
N = 10
# Папка, куда сохранять новые .txt файлы
OUT_DIR = r"data/material/Txt_files/Generated_" + str(N) + "_points"

# Старое значение K, относительно которого считались исходные кривые
K_OLD = 500.0

# Набор новых значений K, для которых нужно сгенерировать кривые
NEW_K_VALUES = [i for i in range(300, 901, 10) if i != 500]


# =====================================================


def parse_params(path: Path):
    """
    Разбирает имя файла вида job_K500_m0p3_er0p002.txt
    и возвращает (K, m, er) как строки.
    """
    m = re.match(r"job_K(?P<K>[^_]+)_m(?P<m>[^_]+)_er(?P<er>.+)", path.stem)
    if not m:
        raise ValueError(f"Не удалось разобрать имя файла: {path.name}")
    return m.group("K"), m.group("m"), m.group("er")


def generate_scaled_curves(in_dir: str, out_dir: str,
                           k_old: float, new_k_values: list[float]) -> None:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pattern = "job_K*_m*_er*.txt"

    for txt_file in in_path.glob(pattern):
        print(f"Обрабатываю {txt_file.name}...")

        try:
            K_str, m_str, er_str = parse_params(txt_file)
        except ValueError as e:
            print(e)
            continue

        # Пытаемся прочитать K из имени как число (для проверки)
        try:
            K_in_file = float(K_str.replace("p", "."))
        except ValueError:
            K_in_file = None

        if K_in_file is not None and abs(K_in_file - k_old) > 1e-9:
            print(f"  ⚠ Внимание: в имени файла K={K_in_file}, а K_OLD={k_old}. "
                  f"Использую K_OLD для масштабирования.")
        # Читаем данные
        try:
            df = pd.read_csv(txt_file, sep="\t")
        except Exception as e:
            print(f"  Не удалось прочитать {txt_file.name}: {e}")
            continue

        if not {"Strain", "Stress_RF_RP"}.issubset(df.columns):
            print(f"  В файле {txt_file.name} нет нужных колонок Strain / Stress_RF_RP.")
            continue

        # Для каждого нового K создаём новую кривую и файл
        for K_new in new_k_values:
            scale = K_new / k_old

            df_new = df.copy()
            df_new["Stress_RF_RP"] = df_new["Stress_RF_RP"] * scale

            # Строка K_new для имени файла (оставим с точкой, это Windows нормально ест)
            K_new_str = str(K_new)

            new_name = f"job_K{K_new_str}_m{m_str}_er{er_str}.txt"
            out_file = out_path / new_name

            df_new.to_csv(out_file, sep="\t", index=False, header=True)
            print(f"    → сохранён {out_file.name}")


def main():
    generate_scaled_curves(IN_DIR, OUT_DIR, K_OLD, NEW_K_VALUES)


main()
