from pathlib import Path
import pandas as pd
import io


in_dir = r"data/material/Scripts_for_modelling/Backofen_law/Make_report"
out_dir = r"data/material/Txt_files/for_article"

# Нижняя граница по координате Strain.
# Все точки с Strain < STRAIN_MIN будут удалены.
# Если не нужно обрезать слева — поставь None.
STRAIN_MIN = 0.1

# Верхняя граница по координате Strain.
# Все точки с Strain > STRAIN_MAX будут удалены.
# Если не нужно обрезать справа — поставь None.
STRAIN_MAX = 0.87


def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Корректно читает csv-файл, в котором первая строка может быть вида 'sep=;'.
    Возвращает DataFrame.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    if lines and lines[0].lower().startswith("sep="):
        sep = lines[0].split("=", 1)[1].strip()
        csv_text = "\n".join(lines[1:])
        return pd.read_csv(io.StringIO(csv_text), sep=sep)
    else:
        try:
            return pd.read_csv(io.StringIO(text), sep=";")
        except Exception:
            return pd.read_csv(io.StringIO(text), sep=",")


def convert_folder(
    in_dir: str,
    out_dir: str,
    strain_min: float | None = None,
    strain_max: float | None = None
) -> None:
    """
    in_dir      — папка с файлами job_K<1>_m<2>_er<3>.csv
    out_dir     — папка, куда кладём соответствующие .txt
    strain_min  — минимальное значение Strain, начиная с которого сохраняем данные
    strain_max  — максимальное значение Strain, до которого сохраняем данные
    """
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pattern = "job_K*_m*_er*.csv"
    # pattern = "job_er*.csv"

    for csv_file in in_path.glob(pattern):
        print(f"Обрабатываю {csv_file.name}...")

        df = read_csv_smart(csv_file)

        try:
            sub = df[["Strain", "Stress_RF_RP"]].copy()
        except KeyError as e:
            print(f"В файле {csv_file.name} нет нужных колонок: {e}")
            continue

        # На всякий случай приводим Strain и Stress_RF_RP к числам
        sub["Strain"] = pd.to_numeric(sub["Strain"], errors="coerce")
        sub["Stress_RF_RP"] = pd.to_numeric(sub["Stress_RF_RP"], errors="coerce")

        # Удаляем строки, где не удалось прочитать числа
        sub = sub.dropna(subset=["Strain", "Stress_RF_RP"])

        n_before = len(sub)

        # Обрезаем слева по координате Strain
        if strain_min is not None:
            sub = sub[sub["Strain"] >= strain_min]

        # Обрезаем справа по координате Strain
        if strain_max is not None:
            sub = sub[sub["Strain"] <= strain_max]

        sub = sub.reset_index(drop=True)

        if len(sub) == 0:
            print(
                f"  ⚠ После обрезки по Strain "
                f"[{strain_min}, {strain_max}] файл {csv_file.name} пустой, пропускаю."
            )
            continue

        txt_file = out_path / (csv_file.stem + ".txt")

        sub.to_csv(txt_file, sep="\t", index=False, header=True)

        print(
            f"    сохранён {txt_file.name}: "
            f"{n_before} строк → {len(sub)} строк"
        )


convert_folder(
    in_dir,
    out_dir,
    strain_min=STRAIN_MIN,
    strain_max=STRAIN_MAX
)