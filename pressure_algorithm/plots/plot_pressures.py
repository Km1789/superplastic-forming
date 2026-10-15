import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ----------------------------------------------------------------------
# 0. Общие вспомогательные функции
# ----------------------------------------------------------------------
def to_float(value):
    """
    Безопасно переводит строку в float.
    Поддерживает обычную десятичную точку и десятичную запятую.
    """
    if value is None:
        raise ValueError("Empty value")
    return float(str(value).strip().replace(',', '.'))


def trim_curve_by_time(t_arr, p_arr, time_limit=None):
    """
    Обрезает кривую по максимальному времени time_limit.
    Если time_limit попадает внутрь интервала, добавляет точку с линейной интерполяцией давления.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    p_arr = np.asarray(p_arr, dtype=float)

    if len(t_arr) == 0:
        return t_arr, p_arr

    # Сортировка на случай, если в файле точки идут не по времени
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    p_arr = p_arr[order]

    if time_limit is None or time_limit <= 0:
        return t_arr, p_arr

    # Если вся кривая начинается позже time_limit
    if t_arr[0] > time_limit:
        return np.array([time_limit]), np.array([p_arr[0]])

    mask = t_arr <= time_limit
    t_cut = t_arr[mask]
    p_cut = p_arr[mask]

    # Если последняя сохраненная точка меньше time_limit,
    # добавляем интерполированную точку
    if len(t_cut) > 0 and t_cut[-1] < time_limit and time_limit <= t_arr[-1]:
        p_at_limit = np.interp(time_limit, t_arr, p_arr)
        t_cut = np.append(t_cut, time_limit)
        p_cut = np.append(p_cut, p_at_limit)

    return t_cut, p_cut


# ----------------------------------------------------------------------
# 1. Парсер для файлов первого типа: Abaqus report, формат "Step Time = ..."
# ----------------------------------------------------------------------
def parse_pressure_file(filename):
    """
    Парсит файл с отчетами о давлении:
        Step Time = ...
        ...
        ----------------
        ... pressure

    Возвращает два списка: times, pressures.
    """
    times = []
    pressures = []

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        match = re.search(r'Step Time\s*=\s*([\d\.Ee\+\-]+)', line)

        if match:
            time_val = to_float(match.group(1))

            # Пропускаем строки до разделителя "----"
            j = i + 1
            while j < n and not re.match(r'^-+', lines[j].strip()):
                j += 1

            # Перешагиваем разделитель
            j += 1

            # Пропускаем пустые строки
            while j < n and lines[j].strip() == '':
                j += 1

            if j < n:
                data_line = lines[j].strip()
                parts = data_line.split()
                if parts:
                    # Последнее поле считаем давлением
                    p_val = to_float(parts[-1])
                    times.append(time_val)
                    pressures.append(p_val)

            i = j
        else:
            i += 1

    return times, pressures


# ----------------------------------------------------------------------
# 2. Парсер для файлов второго типа: Excel/CSV-таблица time, pressure
# ----------------------------------------------------------------------
def parse_excel_pressure_file(filename):
    """
    Парсит файл второго типа, сохраненный из Excel как CSV.

    Поддерживаемые варианты:
        time,pressure
        0.00000000,0.00000000
        0.50000000,0.30000000

    Также поддерживаются:
        - разделители ',', ';', '\\t';
        - строка Excel вида 'sep=;';
        - десятичная запятая;
        - отсутствие заголовка: тогда считаются первые два столбца time, pressure.

    Возвращает два numpy-массива: t_arr, p_arr.
    Кривая строится напрямую по точкам из файла, без восстановления 3-секундного роста.
    """
    with open(filename, 'r', encoding='utf-8-sig', errors='ignore') as f:
        text = f.read()

    lines = text.splitlines()
    if not lines:
        return np.array([]), np.array([])

    # Excel иногда добавляет первую строку вида sep=;
    forced_delimiter = None
    if lines[0].strip().lower().startswith('sep='):
        forced_delimiter = lines[0].split('=', 1)[1].strip()
        lines = lines[1:]

    csv_text = '\n'.join(lines)
    if not csv_text.strip():
        return np.array([]), np.array([])

    if forced_delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(csv_text[:4096], delimiters=',;\t')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','
    else:
        delimiter = forced_delimiter

    reader = csv.reader(csv_text.splitlines(), delimiter=delimiter)
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        return np.array([]), np.array([])

    first = [cell.strip().lower() for cell in rows[0]]

    # Ищем заголовки более гибко
    time_names = {
        'time',
        't',
        'время',
        'time_s',
        'time, s',
        'time (s)'
    }

    pressure_names = {
        'pressure',
        'p',
        'давление',
        'pressure_mpa',
        'pressure, mpa',
        'pressure (mpa)'
    }

    time_idx = None
    pressure_idx = None

    for idx, name in enumerate(first):
        name_norm = name.replace('ё', 'е').strip()
        if name_norm in time_names:
            time_idx = idx
        if name_norm in pressure_names:
            pressure_idx = idx

    has_header = (time_idx is not None) and (pressure_idx is not None)

    if has_header:
        data_rows = rows[1:]
    else:
        # Если заголовка нет, считаем, что первые два столбца: time, pressure
        time_idx = 0
        pressure_idx = 1
        data_rows = rows

    times = []
    pressures = []

    for row in data_rows:
        if len(row) <= max(time_idx, pressure_idx):
            continue

        try:
            t = to_float(row[time_idx])
            p = to_float(row[pressure_idx])
        except ValueError:
            continue

        times.append(t)
        pressures.append(p)

    if not times:
        return np.array([]), np.array([])

    t_arr = np.array(times, dtype=float)
    p_arr = np.array(pressures, dtype=float)

    # Сортировка на случай, если точки записаны не по возрастанию времени
    order = np.argsort(t_arr)
    return t_arr[order], p_arr[order]


# ----------------------------------------------------------------------
# 3. Подготовка кривых разных типов
# ----------------------------------------------------------------------
def prepare_curve_type1(filename, time_limit=None):
    """
    Загружает данные первого типа через parse_pressure_file и обрезает по time_limit.
    """
    t_raw, p_raw = parse_pressure_file(filename)
    return trim_curve_by_time(t_raw, p_raw, time_limit)


def prepare_curve_type2(filename, time_limit=None):
    """
    Загружает данные второго типа как Excel/CSV-таблицу time-pressure и обрезает по time_limit.
    """
    t_raw, p_raw = parse_excel_pressure_file(filename)
    return trim_curve_by_time(t_raw, p_raw, time_limit)


# ----------------------------------------------------------------------
# 4. Основная функция построения графика
# ----------------------------------------------------------------------
def plot_pressures(
    files_type1,       # список кортежей: (filepath, label, max_time)
    files_type2,       # список кортежей: (filepath, label, max_time)
    out_png,

    # ===================== НАСТРОЙКИ ШРИФТА =====================
    font_family="Times New Roman",
    base_fontsize=12,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    # ============================================================
):
    """
    Строит графики давления от времени для двух типов файлов:
        1) Abaqus report: Step Time = ...
        2) Excel/CSV-таблица: time, pressure

    Важно:
        - файлы первого типа обрабатываются как раньше;
        - файлы второго типа теперь НЕ парсятся как Stage-log;
          они читаются напрямую как таблица из Excel.

    Настройки шрифта:
        font_family: семейство шрифта, например "Times New Roman", "Arial", "DejaVu Serif"
        base_fontsize: базовый размер шрифта
        label_fontsize: размер подписей осей
        tick_fontsize: размер подписей делений осей
        legend_fontsize: размер легенды
    """

    # ============================================================
    # НАСТРОЙКА ШРИФТА
    # ============================================================

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = base_fontsize
    plt.rcParams["axes.labelsize"] = label_fontsize
    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize
    plt.rcParams["legend.fontsize"] = legend_fontsize

    # Чтобы математические формулы выглядели ближе к Times New Roman
    plt.rcParams["mathtext.fontset"] = "stix"

    # ============================================================
    # ПОСТРОЕНИЕ ГРАФИКА
    # ============================================================

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Файлы первого типа: без изменений
    linestyles = ['-', '--']
    i = 0
    for fpath, label, tmax in files_type1:
        t, p = prepare_curve_type1(fpath, tmax)
        if len(t) > 0:
            ax.plot(t, p, linewidth=1.8, label=label, linestyle=linestyles[i])
            i += 1
        else:
            print(f"WARNING: no data parsed from type1 file: {fpath}")

    # Файлы второго типа: Excel/CSV time-pressure
    for fpath, label, tmax in files_type2:
        t, p = prepare_curve_type2(fpath, tmax)
        if len(t) > 0:
            ax.plot(
                t,
                p,
                linewidth=1.8,
                label=label,
                linestyle='--'
            )
        else:
            print(f"WARNING: no data parsed from type2 Excel/CSV file: {fpath}")

    # Настройки графика
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Давление, МПа')

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=tick_fontsize
    )

    ax.legend(
        loc='best',
        fontsize=legend_fontsize
    )

    ax.grid(True, linestyle=':', alpha=0.7)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"График сохранён: {out_png}")


# ----------------------------------------------------------------------
# 5. Точка входа — здесь задаются пути и легенды
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # Файлы ПЕРВОГО типа: формат Step Time = ...
    # Каждый элемент: (путь_к_файлу, метка_в_легенде, максимальное_время)
    # Если ограничение по времени не нужно — поставить None.
    # --------------------------------------------------------------
    files_type1 = [
        # (
        #     r"data/forming/Free_bulg_test/pressure_opt.rpt",
        #     r"$\dot{\varepsilon}_{opt}$",
        #     1907.666260
        # ),
        (
            r"data/forming/Pressures/sr_opt_center.rpt",
            r"$\dot{\varepsilon}_{opt}$ в центре",
            1985.362
        ),
        (
            r"data/forming/Pressures/sr_opt_full.rpt",
            r"$\dot{\varepsilon}_{opt}$ по всей детеали",
            2062.362
        ),

    ]

    # --------------------------------------------------------------
    # Файлы ВТОРОГО типа: Excel/CSV-таблица time,pressure
    # Например файл, сохраненный из Excel:
    #     time,pressure
    #     0.00000000,0.00000000
    #     0.50000000,0.30000000
    #     ...
    # --------------------------------------------------------------
    files_type2 = [
        (
            r"data/forming/Free_bulg_test/LOGS/19_real_test_0_1/fill_pressure_optimizer_pressure_schedule.csv",
            "Алгоритм",
            None
        ),
    ]

    out_png = r"data/forming/Free_bulg_test/Graphics/Pressure/pressure_opt_algo_smirnov_correct_two.png"

    # ============================================================
    # НАСТРОЙКИ ШРИФТА
    # ============================================================

    FONT_FAMILY = "Times New Roman"
    BASE_FONTSIZE = 16
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 18
    LEGEND_FONTSIZE = 16

    # Если Times New Roman не находится matplotlib, можно использовать:
    # FONT_FAMILY = "DejaVu Serif"

    # ============================================================
    # ПОСТРОЕНИЕ ГРАФИКА
    # ============================================================

    plot_pressures(
        files_type1,
        files_type2,
        out_png,

        # ===== Настройки шрифта =====
        font_family=FONT_FAMILY,
        base_fontsize=BASE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        tick_fontsize=TICK_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE
    )