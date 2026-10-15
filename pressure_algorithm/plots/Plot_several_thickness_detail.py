import matplotlib.pyplot as plt
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, r"data/forming/Py_scripts")
from compute_distances import compute_distances_from_folder


def plot_multiple_distances(folders_list, labels_list, output_image_path, n=1,
                            time_round=6, show_time=True, colors=None,
                            linewidth=2, markersize=4, title=None, style=None):
    """
    Строит на одном графике минимальные расстояния для нескольких папок.

    Параметры:
        folders_list: список путей к папкам (каждая содержит top.txt и bottom.txt)
        labels_list: список подписей для легенды (должен быть той же длины)
        output_image_path: путь для сохранения итогового графика
        n: радиус окна (одинаков для всех)
        time_round: число знаков после запятой для времени
        show_time: если True, добавляет значение времени (из top.txt) в легенду
        colors: список цветов для кривых (если None, используются стандартные)
        linewidth: толщина линии
        markersize: размер маркеров
        title: заголовок графика (если None, то стандартный)
    """
    if len(folders_list) != len(labels_list):
        raise ValueError("folders_list и labels_list должны быть одинаковой длины")

    plt.figure(figsize=(10, 6))

    # Если цвета не заданы, используем цикл по умолчанию
    if colors is None:
        colors = [None] * len(folders_list)  # None означает автоцвет
    i = 0
    for idx, (folder, label) in enumerate(zip(folders_list, labels_list)):
        # Вычисляем расстояния
        x_vals, dists = compute_distances_from_folder(folder, n=n)

        # Извлекаем время из top.txt для этой папки
        time_value = None
        if show_time:
            top_path = os.path.join(folder, 'top.txt')
            try:
                with open(top_path, 'r') as f:
                    first_line = f.readline().strip()
                    match = re.search(r"Time:\s*([0-9.eE+-]+)", first_line)
                    if match:
                        time_value = float(match.group(1))
            except:
                pass

        # Формируем метку для легенды
        legend_label = label
        if show_time and time_value is not None:
            legend_label += f" ({time_value:.{time_round}f} с)"

        # Строим кривую
        color = colors[idx] if colors[idx] is not None else None
        plt.plot(x_vals, dists, style[i], linewidth=linewidth, markersize=markersize,
                 color=color, label=legend_label)

        i += 1

    plt.xlabel('x, мм', fontsize=12)
    plt.ylabel('Толщина, мм', fontsize=12)
    # if title:
    #     plt.title(title, fontsize=14)
    # else:
    #     # plt.title('Минимальное расстояние от нижней границы до верхней для разных инкрементов', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # изменить (было просто tight_layout())
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # добавить
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Совмещённый график сохранён: {output_image_path}")


# parent = r"data/forming/Reports/Backofen_parsed/pressure_1_9"
# increments = [0, 1000, 2000, 3000, 4000, 5000]
# folders = [os.path.join(parent, f"increment_{inc}") for inc in increments]
# labels = [f"incr {inc}" for inc in increments]
#
# output = r"data/forming/Graphics/Length/Lengthes_1_9.png"
# plot_multiple_distances(folders, labels, output, n=1, time_round=1,
#                             show_time=True, linewidth=1)

# folders = [r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_0/increment_11930',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_1/increment_8260',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_2/increment_5980',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_3/increment_4450',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_4/increment_3445',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_5/increment_2725',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_6/increment_2225',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_7/increment_1845',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_8/increment_1575',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_1_9/increment_1335',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/pressure_2_0/increment_1190',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/sr_0_0001/increment_9705',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/sr_0_001/increment_8085',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/sr_0_01/increment_1590',
#            r'data/forming/Reports/Backofen_parsed/Frictionless/sr_0_1/increment_1525'
#            ]

# folders = [r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_0/increment_15675',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_1/increment_7350',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_2/increment_4415',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_3/increment_3225',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_4/increment_2600',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_5/increment_2225',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_6/increment_1925',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_7/increment_1725',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_8/increment_1545',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_1_9/increment_1410',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_2_0/increment_1290',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_2_1/increment_1200',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_2_2/increment_1105',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_2_4/increment_975',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_2_8/increment_745',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_3_4/increment_550',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/pressure_4_0/increment_425',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/sr_opt/increment_675',
#            # r'data/forming/Reports/Smirnov_parsed/Frictionless/sr_0_0001/increment_675',
#            # r'data/forming/Reports/Smirnov_parsed/Frictionless/sr_0_001/increment_675',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/sr_0_01/increment_915',
#            r'data/forming/Reports/Smirnov_parsed/Frictionless/sr_0_1/increment_550'
#            ]

# folders = [
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_1_0/increment_475',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_1_2/increment_485',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_1_4/increment_430',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_1_6/increment_455',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_1_8/increment_425',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_2_0/increment_415',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_2_2/increment_390',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_2_4/increment_355',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_2_6/increment_330',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_2_8/increment_350',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_3_0/increment_360',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_3_4/increment_320',
#     r'data/forming/Reports/Sorgente_parsed/Frictionless/pressure_4_0/increment_310',
# ]

folders = [
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_5/increment_11470',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_75/increment_5250',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1/increment_3204',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_25/increment_2254',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_5/increment_1678',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_75/increment_1328',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2/increment_1052',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_25/increment_872',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_5/increment_706',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_75/increment_610',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3/increment_508',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_25/increment_454',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_5/increment_386',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_75/increment_358',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_4/increment_306',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt/increment_3830',
    r'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/algorithm_try_0_8/increment_2034',
]

labels = ['0.5 МПа', '0.75 МПа','1.0 МПа','1.25 МПа','1.5 МПа', '1.75 МПа', '2.0 МПа', '2.25 МПа', '2.5 МПа', '2.75 МПа', '3.0 МПа', '3.25 МПа', '3.5 МПа', '3.75 МПа','4.0 МПа', r'$\dot{\varepsilon}_{opt}$', "Алгоритм"]
          # r'$\dot{\varepsilon} = 0.0001$', r'$\dot{\varepsilon} = 0.1$']


style = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--']
output = r"data/forming/Free_bulg_test/Graphics/Thickness/smirnov_compare_80_percent.png"
plot_multiple_distances(folders, labels, output, n=1, time_round=1,
                        show_time=True, linewidth=1, style=style)
