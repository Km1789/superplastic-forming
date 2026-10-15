# run_all_trainings.py
# Последовательно запускает обучение всех моделей (5 скриптов).
# Запуск: python run_all_trainings.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime


# =========================
# ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# =========================

PYTHON_EXE = sys.executable  # или явно: r"C:\Python\Python314\python.exe"

# Папка, где лежат все обучающие скрипты
SCRIPTS_DIR = Path(__file__).resolve().parent

# Какие именно 5 скриптов запускаем (в нужном порядке)
TRAIN_SCRIPTS = [
    "Learning_regression_tree.py",
    "Learning_boosting.py",
    "Learning_neural_network.py",
    "Learning_CNN_2D.py",   # tensor_cnn_sigma_meta
]

# Логи
LOG_DIR = SCRIPTS_DIR / "train_logs"
MERGE_STDERR_TO_STDOUT = True  # чтобы всё было в одном логе

# Если True — если один скрипт упал, прекращаем всю цепочку
STOP_ON_ERROR = True

# ======================================================


def run_one(script_path: Path, log_path: Path) -> int:
    cmd = [PYTHON_EXE, str(script_path)]
    print("\n" + "=" * 90)
    print(f"RUN: {cmd}")
    print(f"LOG: {log_path}")
    print("=" * 90)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(script_path.parent),
            stdout=f,
            stderr=subprocess.STDOUT if MERGE_STDERR_TO_STDOUT else None,
            text=True,
        )
        code = p.wait()

        f.write("\n")
        f.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Exit code: {code}\n")

    return code


def main() -> None:
    # проверим, что файлы существуют
    missing = []
    script_paths = []
    for name in TRAIN_SCRIPTS:
        sp = (SCRIPTS_DIR / name).resolve()
        script_paths.append(sp)
        if not sp.exists():
            missing.append(str(sp))

    if missing:
        print("Не найдены скрипты:")
        for m in missing:
            print("  -", m)
        raise SystemExit(2)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, sp in enumerate(script_paths, start=1):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{i:02d}_{sp.stem}_{stamp}.log"

        code = run_one(sp, log_path)
        summary.append((sp.name, code, str(log_path)))

        if code != 0 and STOP_ON_ERROR:
            print(f"\n[ERROR] Скрипт {sp.name} завершился с кодом {code}. Останов.")
            break

    print("\n" + "=" * 90)
    print("SUMMARY:")
    for name, code, log in summary:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {status:10s}  {name}  -> {log}")
    print("=" * 90)


if __name__ == "__main__":
    main()
