from pathlib import Path
import shutil

# ===================== НАСТРОЙКИ =====================
N = 10
# Первая папка с .txt файлами
SRC_DIR_1 = r"data/material/Txt_files/Resampled_from_modelling_10_points"

# Вторая папка с .txt файлами
SRC_DIR_2 = r"data/material/Txt_files/Generated_10_points"

# Третья папка, куда всё складываем
DST_DIR = r"data/material/Txt_files/All_txt_files_" + str(N) + "_points"


def copy_with_unique_name(src_file: Path, dst_dir: Path) -> Path:
    """
    Копирует файл src_file в папку dst_dir.
    Если файл с таким именем уже существует, добавляет суффикс _2, _3, ...
    Возвращает путь к созданному файлу.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    base_name = src_file.stem
    suffix = src_file.suffix  # обычно ".txt"
    candidate = dst_dir / (base_name + suffix)

    counter = 2
    while candidate.exists():
        candidate = dst_dir / f"{base_name}_{counter}{suffix}"
        counter += 1

    shutil.copy2(src_file, candidate)
    return candidate


def merge_txt_folders(src1: str, src2: str, dst: str) -> None:
    src_path_1 = Path(src1)
    src_path_2 = Path(src2)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    for folder in [src_path_1, src_path_2]:
        if not folder.exists():
            print(f"Папка {folder} не существует, пропускаю.")
            continue

        for txt_file in folder.glob("*.txt"):
            print(f"Копирую {txt_file} ...")
            new_path = copy_with_unique_name(txt_file, dst_path)
            print(f"    → сохранён как {new_path.name}")


def main():
    merge_txt_folders(SRC_DIR_1, SRC_DIR_2, DST_DIR)


if __name__ == "__main__":
    main()
