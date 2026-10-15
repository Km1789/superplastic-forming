import sys
import os

def parse_coord_file(input_filename, top_output="top.txt", bottom_output="bottom.txt"):
    # Проверка существования входного файла
    if not os.path.exists(input_filename):
        print(f"Файл не найден: {input_filename}")
        return

    with open(input_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Поиск строки Time
    time_line = None
    for line in lines:
        if line.strip().startswith("Time:"):
            time_line = line.strip()
            break

    if not time_line:
        raise ValueError("Не найдена строка Time в файле")

    # Извлечение таблицы DETAIL-1
    data_lines = []
    in_detail = False
    header_passed = False

    for line in lines:
        if not in_detail and "Field Output reported at nodes for part: DETAIL-1" in line:
            in_detail = True
            continue

        if in_detail and not header_passed:
            if "Node Label" in line or "-----" in line:
                continue

            if line.strip() and line.strip()[0].isdigit():
                header_passed = True

        if in_detail and header_passed:
            stripped = line.strip()

            if not stripped:
                break

            if stripped[0].isdigit():
                data_lines.append(line)

    # -------------------------------------------------
    # Новая логика:
    # top.txt    -> 1, 6, 11, ...
    # bottom.txt -> 5, 10, 15, ...
    # -------------------------------------------------

    top_data = data_lines[0::7]      # строки 1, 6, 11, ... (индексы 0, 5, 10, ...)
    bottom_data = data_lines[6::7]   # строки 5, 10, 15, ... (индексы 4, 9, 14, ...)

    def write_output(filename, selected_lines, time_line):
        # Создаём директорию если её нет
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as out:
            out.write(time_line + "\n")

            for line in selected_lines:
                parts = line.split()

                if len(parts) >= 4:
                    coor1 = parts[2]
                    coor2 = parts[3]
                    out.write(f"{coor1} {coor2}\n")
                else:
                    out.write(line.strip() + "\n")

    write_output(top_output, top_data, time_line)
    write_output(bottom_output, bottom_data, time_line)

    print(f"Файлы '{top_output}' и '{bottom_output}' созданы.")


for i in range(0, 6025, 2):
    parse_coord_file(rf"data/forming/Free_bulg_test/Reports/Smirnov/Frictionless/smooth_alg/increment_{i}.rpt",
                       rf"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/smooth_alg/increment_{i}/top.txt",
                       rf"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/smooth_alg/increment_{i}/bottom.txt")