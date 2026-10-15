import os
import re


# ============================================================
# Geometry helpers
# ============================================================

def read_points(directory, filename="top.txt"):
    """
    Reads a curve from a text file.

    Expected format:
        first line may be a header;
        then at least two numeric columns: x y

    Non-numeric lines are ignored. This makes the function robust to headers
    such as:
        X Y
        Step Time = ...
        Time: ...
    """
    file_path = os.path.join(directory, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError("Файл не найден: %s" % file_path)

    points = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Allow whitespace-separated, comma-separated and semicolon-separated files.
        line_for_split = line.replace(",", " ").replace(";", " ")
        parts = line_for_split.split()

        if len(parts) < 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            points.append((x, y))
        except ValueError:
            continue

    if len(points) < 2:
        raise ValueError("Недостаточно точек для вычисления площади в файле %s" % file_path)

    points.sort(key=lambda p: p[0])
    return points


def compute_area(directory, filename="top.txt"):
    """
    Computes area under a curve using the trapezoidal rule.
    """
    points = read_points(directory, filename)

    area = 0.0
    for i in range(1, len(points)):
        x_prev, y_prev = points[i - 1]
        x_curr, y_curr = points[i]
        area += (x_curr - x_prev) * (y_prev + y_curr) / 2.0

    return area


def compute_fill(area_current, DIE_AREA, area_initial):
    """
    Filling fraction:

        F = (area_current - area_initial) / (DIE_AREA - area_initial)
    """
    denominator = float(DIE_AREA) - float(area_initial)

    if abs(denominator) < 1e-12:
        raise ValueError("DIE_AREA и area_initial не должны быть равны")

    return (float(area_current) - float(area_initial)) / denominator


# ============================================================
# Time reading helpers
# ============================================================

def _extract_first_float(text):
    """
    Extracts the first floating-point number from text.
    """
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
    m = re.search(pattern, text)
    if m:
        return float(m.group(0))
    return None


def _extract_time_from_text(text):
    """
    Tries to extract time from common text fragments:
        Step Time = 114.194466
        frame value = 114.194466
        time: 114.194466
        114.194466
    """
    patterns = [
        r"Step\s*Time\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        r"Frame\s*Value\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        r"frame_value\s*[=:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        r"time\s*[=:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        r"Time\s*[=:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    return _extract_first_float(text)


def read_time_for_increment(
    inc_dir,
    data_filename="top.txt",
    time_filename=None,
    candidate_time_filenames=None,
):
    """
    Reads the physical time of an increment.

    The function tries several options:
      1. If time_filename is given, read time from this file in inc_dir.
      2. Try common files: time.txt, frame_time.txt, step_time.txt, metadata.txt.
      3. Try to read time from the first several lines of data_filename.

    If time is not found, returns None. In that case interpolation will still
    work in terms of increment number, but not in terms of physical time.
    """
    candidates = []

    if time_filename:
        candidates.append(time_filename)

    if candidate_time_filenames is None:
        candidate_time_filenames = [
            "time.txt",
            "frame_time.txt",
            "step_time.txt",
            "metadata.txt",
            "info.txt",
        ]

    for name in candidate_time_filenames:
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        path = os.path.join(inc_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                value = _extract_time_from_text(text)
                if value is not None:
                    return value
            except Exception:
                pass

    data_path = os.path.join(inc_dir, data_filename)
    if os.path.isfile(data_path):
        try:
            lines = []
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
            value = _extract_time_from_text("".join(lines))
            if value is not None:
                return value
        except Exception:
            pass

    return None


# ============================================================
# Series collection
# ============================================================

def collect_fill_series(
    root,
    DIE_AREA=1407.6088,
    area_initial=145.3,
    increment_step=5,
    max_increment=30000,
    filename="top.txt",
    time_filename=None,
):
    """
    Collects area/fill/time values for all increment_* folders.

    Returns a list of dictionaries sorted by increment:
        {
            "increment": int,
            "time": float or None,
            "area": float,
            "fill": float,
            "path": str
        }
    """
    root = os.path.abspath(root)
    series = []

    for inc in range(0, max_increment + 1, increment_step):
        inc_dir = os.path.join(root, "increment_%d" % inc)
        file_path = os.path.join(inc_dir, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            area_current = compute_area(inc_dir, filename)
            fill = compute_fill(
                area_current=area_current,
                DIE_AREA=DIE_AREA,
                area_initial=area_initial,
            )
            time_value = read_time_for_increment(
                inc_dir=inc_dir,
                data_filename=filename,
                time_filename=time_filename,
            )

            series.append({
                "increment": int(inc),
                "time": time_value,
                "area": float(area_current),
                "fill": float(fill),
                "path": file_path,
            })

        except (ValueError, FileNotFoundError):
            continue

    series.sort(key=lambda item: item["increment"])
    return series


# ============================================================
# Search methods
# ============================================================

def find_closest_in_series(series, target_fill):
    """
    Old behavior: find the frame with minimum |F - F_target|.
    """
    if not series:
        return None

    best = None
    best_diff = float("inf")

    for item in series:
        diff = abs(float(item["fill"]) - float(target_fill))
        if diff < best_diff:
            best_diff = diff
            best = item

    if best is None:
        return None

    result = dict(best)
    result.update({
        "method": "closest",
        "target_fill": float(target_fill),
        "diff": abs(float(best["fill"]) - float(target_fill)),
        "crossing_increment": float(best["increment"]),
        "crossing_time": best["time"],
        "crossing_area": float(best["area"]),
        "crossing_fill": float(best["fill"]),
        "alpha": None,
        "prev": None,
        "cur": best,
        "reached": float(best["fill"]) >= float(target_fill),
    })
    return result


def interpolate_crossing(prev_item, cur_item, target_fill):
    """
    Linear interpolation between two neighboring frames:
        F_prev < F_target <= F_cur

    Interpolates:
        increment,
        area,
        time if both neighboring frames have time values.
    """
    f0 = float(prev_item["fill"])
    f1 = float(cur_item["fill"])
    ft = float(target_fill)

    denom = f1 - f0
    if abs(denom) < 1e-14:
        alpha = 0.0
    else:
        alpha = (ft - f0) / denom

    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    inc0 = float(prev_item["increment"])
    inc1 = float(cur_item["increment"])
    crossing_increment = inc0 + alpha * (inc1 - inc0)

    area0 = float(prev_item["area"])
    area1 = float(cur_item["area"])
    crossing_area = area0 + alpha * (area1 - area0)

    time0 = prev_item.get("time", None)
    time1 = cur_item.get("time", None)
    if time0 is not None and time1 is not None:
        crossing_time = float(time0) + alpha * (float(time1) - float(time0))
    else:
        crossing_time = None

    return {
        "method": "first_crossing_interpolated",
        "target_fill": ft,
        "increment": int(cur_item["increment"]),
        "time": cur_item.get("time", None),
        "area": float(cur_item["area"]),
        "fill": float(cur_item["fill"]),
        "path": cur_item["path"],
        "diff": abs(float(cur_item["fill"]) - ft),
        "crossing_increment": crossing_increment,
        "crossing_time": crossing_time,
        "crossing_area": crossing_area,
        "crossing_fill": ft,
        "alpha": alpha,
        "prev": prev_item,
        "cur": cur_item,
        "reached": True,
    }


def find_first_crossing_interpolated(series, target_fill):
    """
    Finds the first upward crossing of target_fill and interpolates its time.

    This is the recommended behavior for comparison with the optimizer:
      - not the closest frame;
      - not just the first saved frame after crossing;
      - but the linearly interpolated crossing time between neighboring frames.

    If target is already reached at the first available frame, returns that frame.
    If target is never reached, returns the closest frame with reached=False.
    """
    if not series:
        return None

    ft = float(target_fill)

    first = series[0]
    if float(first["fill"]) >= ft:
        result = dict(first)
        result.update({
            "method": "first_frame_already_above_target",
            "target_fill": ft,
            "diff": abs(float(first["fill"]) - ft),
            "crossing_increment": float(first["increment"]),
            "crossing_time": first.get("time", None),
            "crossing_area": float(first["area"]),
            "crossing_fill": float(first["fill"]),
            "alpha": 1.0,
            "prev": None,
            "cur": first,
            "reached": True,
        })
        return result

    prev = first

    for cur in series[1:]:
        f_prev = float(prev["fill"])
        f_cur = float(cur["fill"])

        if f_prev < ft <= f_cur:
            return interpolate_crossing(prev, cur, ft)

        prev = cur

    result = find_closest_in_series(series, target_fill)
    if result is not None:
        result["method"] = "not_reached_closest"
        result["reached"] = False
    return result


# ============================================================
# Public functions
# ============================================================

def find_fill_target(
    root_paths,
    target_fill,
    DIE_AREA=1407.6088,
    area_initial=145.3,
    increment_step=5,
    max_increment=30000,
    top_filename="top.txt",
    bottom_filename=None,
    time_filename=None,
    mode="crossing",
):
    """
    Finds target filling for top and optionally bottom files.

    mode:
        "crossing" -> first upward crossing with linear interpolation.
        "closest"  -> old behavior: closest saved frame by |F - F_target|.
    """
    if isinstance(root_paths, str):
        root_paths = [root_paths]

    results = {}

    for root in root_paths:
        root = os.path.abspath(root)
        results[root] = {}

        def process_file(file_name):
            series = collect_fill_series(
                root=root,
                DIE_AREA=DIE_AREA,
                area_initial=area_initial,
                increment_step=increment_step,
                max_increment=max_increment,
                filename=file_name,
                time_filename=time_filename,
            )

            key = file_name.replace(".txt", "")

            if not series:
                results[root][key] = {
                    "error": "Не найден ни один валидный файл %s в %s" % (file_name, root)
                }
                return

            if mode == "closest":
                info = find_closest_in_series(series, target_fill)
            else:
                info = find_first_crossing_interpolated(series, target_fill)

            if info is None:
                results[root][key] = {
                    "error": "Не удалось найти результат для %s в %s" % (file_name, root)
                }
            else:
                info["series_count"] = len(series)
                results[root][key] = info

        process_file(top_filename)

        if bottom_filename:
            process_file(bottom_filename)

    return results


def find_closest_increment(
    root_paths,
    target_fill,
    DIE_AREA=1407.6088,
    area_initial=145.3,
    increment_step=5,
    max_increment=30000,
    top_filename="top.txt",
    bottom_filename=None,
):
    """
    Backward-compatible wrapper with the old behavior.
    """
    return find_fill_target(
        root_paths=root_paths,
        target_fill=target_fill,
        DIE_AREA=DIE_AREA,
        area_initial=area_initial,
        increment_step=increment_step,
        max_increment=max_increment,
        top_filename=top_filename,
        bottom_filename=bottom_filename,
        time_filename=None,
        mode="closest",
    )


def _format_optional_float(value):
    if value is None:
        return "None"
    return "%.8f" % float(value)


def print_result_block(target, result):
    print("\n===== Целевая заполненность: %.6f =====" % float(target))

    for root, data in result.items():
        print("\nКорневая папка: %s" % root)

        for bound in ["top", "bottom"]:
            if bound not in data:
                continue

            info = data[bound]

            if "error" in info:
                print("  %s: ошибка - %s" % (bound, info["error"]))
                continue

            print("  %s:" % bound)
            print("    method               = %s" % info.get("method", ""))
            print("    reached              = %s" % str(info.get("reached", False)))
            print("    selected increment   = %s" % str(info.get("increment", "")))
            print("    selected time        = %s" % _format_optional_float(info.get("time", None)))
            print("    selected fill        = %.8f" % float(info.get("fill", 0.0)))
            print("    selected area        = %.8f" % float(info.get("area", 0.0)))
            print("    selected diff        = %.8f" % float(info.get("diff", 0.0)))

            print("    crossing increment   = %.8f" % float(info.get("crossing_increment", 0.0)))
            print("    crossing time        = %s" % _format_optional_float(info.get("crossing_time", None)))
            print("    crossing fill        = %.8f" % float(info.get("crossing_fill", 0.0)))
            print("    crossing area        = %.8f" % float(info.get("crossing_area", 0.0)))
            print("    alpha                = %s" % _format_optional_float(info.get("alpha", None)))
            print("    path                 = %s" % info.get("path", ""))

            prev = info.get("prev", None)
            cur = info.get("cur", None)

            if prev is not None:
                print("    prev: inc=%d, time=%s, fill=%.8f" % (
                    int(prev["increment"]),
                    _format_optional_float(prev.get("time", None)),
                    float(prev["fill"]),
                ))

            if cur is not None:
                print("    cur : inc=%d, time=%s, fill=%.8f" % (
                    int(cur["increment"]),
                    _format_optional_float(cur.get("time", None)),
                    float(cur["fill"]),
                ))


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    paths = [
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_5",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_0_75",
        r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_25",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_5",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_1_75",
        r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_25",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_5",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_2_75",
        r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_25",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_5",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_3_75",
        r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/pressure_4",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt_center",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/sr_opt_full",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/algo4",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/algorithm_try_0_8",
        # r"data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/smooth_alg",
    ]

    targets = [0.8]

    for target in targets:
        result = find_fill_target(
            root_paths=paths,
            target_fill=target,
            DIE_AREA=15995.90,
            area_initial=800.0,
            increment_step=2,
            max_increment=30000,
            top_filename="top.txt",
            bottom_filename="bottom.txt",

            # If each increment folder has a separate file with physical time,
            # put its name here, for example:
            # time_filename="time.txt"
            #
            # If time is written in the header of top.txt/bottom.txt, leave None.
            time_filename=None,

            # Recommended new mode:
            #   crossing -> first F_target crossing with linear interpolation.
            # Old mode:
            #   closest
            # mode="crossing",
            mode="closest",

        )

        print_result_block(target, result)
