import os
import sys
from find_increment_with_touch import find_increment_closest_to_y

def get_closest_increments_for_folders(parent_folders, target_y):
    """
    Для каждой родительской папки находит инкремент, в котором max_y
    ближе всего к target_y.

    Параметры:
        parent_folders: список путей к папкам (каждая содержит increment_X)
        target_y: целевое значение Y

    Возвращает:
        dict: {имя_папки: (номер_инкремента, достигнутое_max_y)}.
              Если инкремент не найден, значение будет (None, None).
    """
    results = {}
    for folder_path in parent_folders:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        print(f"Обработка: {folder_name} -> {folder_path}")
        inc, max_y = find_increment_closest_to_y(folder_path, target_y)
        results[folder_name] = (inc, max_y)
        if inc is not None:
            print(f"  Найден инкремент: {inc} (max_y = {max_y:.4f})")
        else:
            print(f"  Не удалось найти подходящий инкремент")
    return results

base_dir = r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless"
subfolders = ["pressure_0_7", "pressure_1", "pressure_1_5", "pressure_2",
              "pressure_2_5", "pressure_3", "pressure_3_5", "pressure_4", "sr_opt"]

parent_folders = [os.path.join(base_dir, sf) for sf in subfolders
                  if os.path.isdir(os.path.join(base_dir, sf))]

if not parent_folders:
    print("Нет доступных папок для обработки")
    sys.exit(1)

# Запуск с целевым Y = 85
results = get_closest_increments_for_folders(parent_folders, target_y=120)

print("\nРезультаты (инкремент, max_y):")
for name, (inc, max_y) in results.items():
    if inc is not None:
        print(f"{name:15} -> increment_{inc:4d}  (max_y = {max_y:.4f})")
    else:
        print(f"{name:15} -> не найден")