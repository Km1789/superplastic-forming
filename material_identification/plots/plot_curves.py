from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ===================== НАСТРОЙКИ =====================

FULL_DIR = r"data/material/Txt_files/for_article"
APPROX_DIR = r"data/material/Txt_files/Resampled_for_article10_points"

# Что рисуем: "full" / "approx" / "both"
MODE = "both"

# --- КАК ПАРСИТЬ ИМЕНА ФАЙЛОВ ---
# "auto" -> сам определит: если есть job_K*_m*_er* -> режим "km", иначе "er"
# "km"   -> ожидать job_K..._m..._er...
# "er"   -> ожидать job_er...
PARSE_MODE = "auto"

# --- РЕЖИМ ВЫБОРА ГРУПП ---
# Если False — строим ОДИН график:
#   - в режиме "km": для одной пары K_TARGET/M_TARGET
#   - в режиме "er": просто один график со всеми job_er*.txt
# Если True  — строим графики для всех найденных групп:
#   - в режиме "km": для всех (K,m)
#   - в режиме "er": будет ровно один график (групп нет)
USE_ALL_GROUPS = True

# Пара для одиночного режима, актуально только для режима "km"
# K_TARGET = "500"
# M_TARGET = "0p3"

# Необязательные фильтры, актуально только для режима "km" и USE_ALL_GROUPS=True
K_FILTER = ["500"]   # например: ["500"] или ["400", "500"]
M_FILTER = ["0p57"]   # например: ["0p3", "0p4"]

# Фильтр по скоростям деформации для любого режима:
# если пусто — берём все er
ER_FILTER = ['0p0001', '0p0004', '0p0008']
# ER_FILTER = []

SHOW_PLOT = False
SAVE_PLOT = True
OUT_PNG_DIR = r"data/material/Graphics"

# Ось X major ticks
X_MAJOR_STEP = 0.3

# --- НАСТРОЙКИ ШРИФТОВ ---
FONT_FAMILY = "Times New Roman"

BASE_FONT_SIZE = 14
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 14
LEGEND_TITLE_SIZE = 12

# Размер самой картинки
FIG_WIDTH = 9
FIG_HEIGHT = 5

# =====================================================


# --- REGEX под два формата ---
RE_KM = re.compile(r"job_K(?P<K>[^_]+)_m(?P<m>[^_]+)_er(?P<er>.+)$")
# RE_ER = re.compile(r"job_er(?P<er>.+)$")


def apply_plot_style():
    """
    Общие настройки стиля графиков.
    Все размеры шрифтов удобно менять в блоке НАСТРОЙКИ.
    """
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,

        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,

        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_SIZE,

        # Чтобы математические символы выглядели ближе к Times New Roman
        "mathtext.fontset": "stix",
    })


def detect_parse_mode(folders):
    """Определяет режим по наличию файлов с K/m."""
    saw_er = False

    for folder in folders:
        if not folder:
            continue

        p = Path(folder)

        if not p.exists():
            continue

        for f in p.glob("job_*.txt"):
            stem = f.stem

            if RE_KM.match(stem):
                return "km"

            if RE_ER.match(stem):
                saw_er = True

    return "er" if saw_er else "km"


def parse_filename(path: Path, parse_mode: str):
    """
    Возвращает:
      group_key: tuple
        - для "km" это (K, m)
        - для "er" это ("NO_KM",)
      er: str
    """
    stem = path.stem

    if parse_mode == "km":
        match = RE_KM.match(stem)

        if not match:
            raise ValueError(
                f"Не удалось разобрать имя файла, ожидался km-формат: {path.name}"
            )

        return (match.group("K"), match.group("m")), match.group("er")

    if parse_mode == "er":
        match = RE_ER.match(stem)

        if not match:
            raise ValueError(
                f"Не удалось разобрать имя файла, ожидался er-формат: {path.name}"
            )

        return ("NO_KM",), match.group("er")

    raise ValueError(f"Неизвестный parse_mode={parse_mode}")


def plot_curves_from_folder(folder: str, group_key, parse_mode: str, is_approx: bool = False):
    """
    Рисует все кривые из папки для заданной группы.

    parse_mode="km":
        group_key = (K, m)

    parse_mode="er":
        group_key = ("NO_KM",)
        то есть рисуются все job_er*.txt
    """
    if not folder:
        return

    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Папка {folder} не существует, пропускаю.")
        return

    files = sorted(folder_path.glob("job_*.txt"))

    if not files:
        print(f"В папке {folder} не найдено .txt файлов.")
        return

    ax = plt.gca()
    found_any = False

    full_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]

    color_idx = 0

    for f in files:
        try:
            f_group_key, er = parse_filename(f, parse_mode=parse_mode)
        except ValueError:
            continue

        # Фильтр по группе.
        # Для km-режима group_key = (K, m).
        # Для er-режима group_key = ("NO_KM",).
        if f_group_key != group_key:
            continue

        # Фильтр по скорости деформации, если задан
        if ER_FILTER and er not in ER_FILTER:
            continue

        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as e:
            print(f"Не удалось прочитать {f.name}: {e}")
            continue

        if not {"Strain", "Stress_RF_RP"}.issubset(df.columns):
            print(f"В файле {f.name} нет нужных колонок Strain / Stress_RF_RP.")
            continue

        df = df.sort_values("Strain")

        if is_approx:
            ax.plot(
                df["Strain"],
                df["Stress_RF_RP"],
                linestyle="--",
                marker="o",
                linewidth=1.0,
                markersize=3,
                color="gray",
                alpha=1,
                label="_nolegend_",
            )

        else:
            er_display = er.replace("p", ".")
            label = rf"$\dot{{\varepsilon}} = {er_display}\,\mathrm{{s}}^{{-1}}$"

            color = full_colors[color_idx % len(full_colors)]
            color_idx += 1

            ax.plot(
                df["Strain"],
                df["Stress_RF_RP"],
                linestyle="-",
                linewidth=1.5,
                color=color,
                label=label,
            )

        found_any = True

    if not found_any:
        if parse_mode == "km":
            K, m = group_key
            print(f"Для K={K}, m={m} в папке {folder} файлов не найдено.")
        else:
            print(
                f"В папке {folder} не найдено подходящих файлов job_er*.txt "
                f"или всё отфильтровано."
            )


def collect_groups_from_folders(folders, parse_mode: str):
    """Собирает все group_key из файлов."""
    groups = set()

    for folder in folders:
        if not folder:
            continue

        p = Path(folder)

        if not p.exists():
            continue

        for f in p.glob("job_*.txt"):
            try:
                group_key, _ = parse_filename(f, parse_mode=parse_mode)
            except ValueError:
                continue

            groups.add(group_key)

    return groups


def km_sort_key(group_key):
    """Сортировка только для (K, m). Для er-режима не используется."""
    K_str, m_str = group_key

    def parse_k(s):
        try:
            return float(s)
        except Exception:
            return s

    def parse_m(s):
        try:
            return float(s.replace("p", "."))
        except Exception:
            return s

    return parse_k(K_str), parse_m(m_str)


def make_plot_for_group(group_key, parse_mode: str, mode: str):
    apply_plot_style()

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    mode_lower = mode.lower()

    if mode_lower in ("full", "both"):
        plot_curves_from_folder(
            FULL_DIR,
            group_key,
            parse_mode=parse_mode,
            is_approx=False,
        )

    if mode_lower in ("approx", "both"):
        plot_curves_from_folder(
            APPROX_DIR,
            group_key,
            parse_mode=parse_mode,
            is_approx=True,
        )

    ax = plt.gca()

    # Подписи осей
    plt.xlabel(r"$\varepsilon, -$", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r"$\sigma,\,\mathrm{MPa}$", fontsize=AXIS_LABEL_SIZE)

    # Размер шрифта делений осей
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_STEP))

    # Легенду строим только если есть подписанные кривые
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
        )

    plt.grid(True)
    plt.tight_layout()

    if SAVE_PLOT:
        out_dir = Path(OUT_PNG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        if parse_mode == "km":
            K, m = group_key
            out_name = f"curves_K{K}_m{m}_{mode_lower}.png"
        else:
            out_name = f"curves_erOnly_{mode_lower}.png"

        out_path = out_dir / out_name

        # bbox_inches="tight" нужно, чтобы легенда справа не обрезалась
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        print(f"График сохранён в {out_path}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


def main():
    mode_lower = MODE.lower()

    parse_mode = PARSE_MODE.lower()

    if parse_mode == "auto":
        parse_mode = detect_parse_mode([FULL_DIR, APPROX_DIR])
        print(f"[auto] Определён режим парсинга: {parse_mode}")

    if parse_mode == "km":
        if not USE_ALL_GROUPS:
            make_plot_for_group(
                (K_TARGET, M_TARGET),
                parse_mode=parse_mode,
                mode=mode_lower,
            )

        else:
            groups = collect_groups_from_folders(
                [FULL_DIR, APPROX_DIR],
                parse_mode=parse_mode,
            )

            if K_FILTER:
                groups = {g for g in groups if g[0] in K_FILTER}

            if M_FILTER:
                groups = {g for g in groups if g[1] in M_FILTER}

            if not groups:
                print("Не найдено ни одной пары (K, m) для построения графиков.")
                return

            for group_key in sorted(groups, key=km_sort_key):
                K, m = group_key
                print(f"Рисую график для K={K}, m={m}...")

                make_plot_for_group(
                    group_key,
                    parse_mode=parse_mode,
                    mode=mode_lower,
                )

    elif parse_mode == "er":
        # В этом режиме групп нет:
        # один график со всеми job_er*.txt с учётом ER_FILTER, если задан
        make_plot_for_group(
            ("NO_KM",),
            parse_mode=parse_mode,
            mode=mode_lower,
        )

    else:
        raise ValueError(f"Неизвестный parse_mode={parse_mode}")


main()