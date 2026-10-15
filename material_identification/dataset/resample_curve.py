from pathlib import Path
import pandas as pd
import numpy as np

in_dir = r"data/material/Txt_files/for_article"
out_dir = r"data/material/Txt_files/Resampled_for_article"
N = 10
out_dir = out_dir + f"{N}_points"

def resample_curve_to_n_points(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    """
    Принимает DataFrame с колонками 'Strain' и 'Stress_RF_RP',
    возвращает новый DataFrame с той же структурой, но с n_points точками
    (линейная интерполяция по Strain).
    """
    # Убедимся, что отсортировано по Strain
    df_sorted = df.sort_values("Strain")

    x = df_sorted["Strain"].to_numpy()
    y = df_sorted["Stress_RF_RP"].to_numpy()

    # Новая равномерная сетка по Strain
    x_new = np.linspace(x.min(), x.max(), n_points)

    # Линейная интерполяция
    y_new = np.interp(x_new, x, y)

    return pd.DataFrame({
        "Strain": x_new,
        "Stress_RF_RP": y_new
    })


def resample_folder(in_dir: str, out_dir: str, n_points: int) -> None:
    """
    in_dir  — папка с исходными .txt файлами (Strain, Stress_RF_RP, разделитель табуляция)
    out_dir — папка, куда сохранить аппроксимированные .txt
    n_points — число точек в итоговых файлах
    """
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for txt_file in in_path.glob("*.txt"):
        print(f"Обрабатываю {txt_file.name}...")

        try:
            df = pd.read_csv(txt_file, sep="\t")
        except Exception as e:
            print(f"Не удалось прочитать {txt_file.name}: {e}")
            continue

        # Проверка нужных колонок
        if not {"Strain", "Stress_RF_RP"}.issubset(df.columns):
            print(f"⚠ В файле {txt_file.name} нет нужных колонок.")
            continue

        # Ресемплинг до N точек
        df_new = resample_curve_to_n_points(df[["Strain", "Stress_RF_RP"]], n_points)

        out_file = out_path / txt_file.name
        # сохраняем с заголовками, разделитель — табуляция
        df_new.to_csv(out_file, sep="\t", index=False, header=True)
        print(f"    сохранён {out_file.name}")

resample_folder(in_dir, out_dir, N)
