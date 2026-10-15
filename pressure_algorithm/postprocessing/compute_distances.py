import numpy as np
import os
import matplotlib.pyplot as plt

def read_profile(filepath):
    """Читает top.txt или bottom.txt, возвращает (x_coords, y_coords)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return [], []
    # Первая строка — время, пропускаем
    data_lines = lines[1:]
    x_vals, y_vals = [], []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                x_vals.append(x)
                y_vals.append(y)
            except ValueError:
                continue
    return np.array(x_vals), np.array(y_vals)

def min_distance_to_top(bottom_points, top_points, n=1):
    N = len(bottom_points)
    M = len(top_points)
    x_bottom = bottom_points[:, 0]
    min_dist = np.zeros(N)

    for i in range(N):
        # --- Расстояния до точек (окно по индексам) ---
        idx_start = max(0, i - n)
        idx_end = min(M, i + n + 1)
        window_points = top_points[idx_start:idx_end]
        if len(window_points) > 0:
            dist_to_points = np.linalg.norm(bottom_points[i] - window_points, axis=1)
            best = np.min(dist_to_points)
        else:
            best = np.inf

        # --- Расстояния до отрезков ---
        # Индексы отрезков от j_start до j_end включительно
        j_start = max(0, i - n)
        j_end = min(M - 2, i + n - 1)   # M-2 — последний индекс отрезка
        for j in range(j_start, j_end + 1):
            A = top_points[j]
            B = top_points[j + 1]
            P = bottom_points[i]

            AB = B - A
            AP = P - A
            t = np.dot(AP, AB) / np.dot(AB, AB)
            if t <= 0:
                closest = A
            elif t >= 1:
                closest = B
            else:
                closest = A + t * AB
            dist = np.linalg.norm(P - closest)
            if dist < best:
                best = dist

        min_dist[i] = best
    return x_bottom, min_dist

def compute_distances_from_folder(folder_path, n=1):
    """Читает top и bottom из папки и вычисляет минимальные расстояния."""
    top_path = os.path.join(folder_path, 'top.txt')
    bottom_path = os.path.join(folder_path, 'bottom.txt')
    if not os.path.exists(top_path) or not os.path.exists(bottom_path):
        raise FileNotFoundError(f"Не найдены top.txt/bottom.txt в {folder_path}")

    x_top, y_top = read_profile(top_path)
    x_bottom, y_bottom = read_profile(bottom_path)
    if len(x_top) == 0 or len(x_bottom) == 0:
        raise ValueError("Один из файлов не содержит данных")

    # Объединяем x и y в массивы точек
    top_points = np.column_stack((x_top, y_top))
    bottom_points = np.column_stack((x_bottom, y_bottom))

    x_vals, dists = min_distance_to_top(bottom_points, top_points, n)
    return x_vals, dists

#
# folder = r"data/forming/Reports/Backofen_parsed/pressure_1_9/increment_0"
# x, dists = compute_distances_from_folder(folder, n=1)
