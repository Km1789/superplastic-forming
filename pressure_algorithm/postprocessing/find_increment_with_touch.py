import os
import re

def find_increment_closest_to_y(parent_folder, target_y):
    """
    Находит инкремент, в котором максимальная координата Y в top.txt
    ближе всего к целевому значению target_y.

    Параметры:
        parent_folder: путь к папке, содержащей поддиректории increment_X
        target_y: целевое значение Y

    Возвращает:
        tuple: (номер_инкремента, значение_max_y) или (None, None), если данные отсутствуют
    """
    pattern = re.compile(r'increment_(\d+)')
    increments = []

    # Собираем все существующие папки increment_*
    for name in os.listdir(parent_folder):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(parent_folder, name)):
            increments.append(int(match.group(1)))

    if not increments:
        raise ValueError(f"Нет папок вида 'increment_X' в {parent_folder}")

    increments.sort()

    best_inc = None
    best_diff = float('inf')
    best_max_y = None

    for inc in increments:
        folder_path = os.path.join(parent_folder, f"increment_{inc}")
        top_path = os.path.join(folder_path, "top.txt")

        if not os.path.isfile(top_path):
            continue  # пропускаем, если нет top.txt

        # Чтение координат из top.txt
        try:
            data = []
            with open(top_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line[0].isalpha():
                        continue  # пропускаем строки-заголовки
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            data.append((x, y))
                        except ValueError:
                            continue
            if not data:
                continue  # нет числовых данных

            _, y_vals = zip(*data)
            max_y = max(y_vals)
            diff = abs(max_y - target_y)

            # Выбираем инкремент с минимальным отклонением
            # При равном отклонении оставляем меньший номер (уже при переборе по возрастанию)
            if diff < best_diff:
                best_diff = diff
                best_inc = inc
                best_max_y = max_y

        except Exception as e:
            print(f"Ошибка при чтении {top_path}: {e}")
            continue

    return best_inc, best_max_y