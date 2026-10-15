# Learning_CNN_2D.py
# CNN "sigma + meta": вход = (image напряжений) + (meta = eps + er), выход = (K, m)
# Преобразование целей: log1p -> StandardScaler (сохраняется в модели)
# Сохранение: TorchScript + meta.json (для автоматического предикта)

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from group_split import group_split, groups_from, group_kfold


# ===================== НАСТРОЙКИ =====================

CSV_PATH = r"data/material/Datasets/three_curves_10_type_2_for_real_data.csv"
OUT_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data"
EXPERIMENT_SUBDIR = "tensor_cnn_sigma_meta"

TARGET_COLS = ["K", "m"]

TEST_SIZE = 0.2
VALID_SIZE = 0.2
RANDOM_STATE = 42

USE_LOG_ER = True               # логарифмировать er при формировании мета-признаков
USE_LOG_TARGET = True           # НОВО: логарифмировать K и m перед стандартизацией

BATCH_SIZE = 64
N_EPOCHS = 200        # максимум эпох
PATIENCE = 30         # ранняя остановка: эпох без улучшения на валидации
CONV_CHANNELS = (8, 16)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_PREFIX = "e_"
STRESS_PREFIX = "s_"
ER_PREFIX = "er_"

# имена файлов для сохранения
META_FILENAME = "tensor_cnn_sigma_meta_meta.json"
MODEL_FILENAME = "tensor_cnn_sigma_meta_model.pt"

# =====================================================


def read_csv_smart(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    if first_line.lower().startswith("sep=") and len(first_line.strip()) >= 5:
        sep = first_line.strip()[4]
        df = pd.read_csv(path, sep=sep, skiprows=1, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def rmse_multioutput(y_true, y_pred) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def parse_column_structure(columns, target_cols):
    eps_map = []
    stress_map = {}
    er_map = {}

    for col in columns:
        if col in target_cols:
            continue

        if col.startswith(EPS_PREFIX):
            parts = col.split("_")
            if len(parts) != 2:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            eps_map.append((idx, col))

        elif col.startswith(STRESS_PREFIX):
            try:
                base, suffix = col.split("_er")
            except ValueError:
                continue
            parts = base.split("_")
            if len(parts) != 2:
                continue
            try:
                pt_idx = int(parts[1])
            except ValueError:
                continue

            curve_id = suffix
            stress_map.setdefault(curve_id, {})[pt_idx] = col

        elif col.startswith(ER_PREFIX):
            parts = col.split("er_")
            if len(parts) != 2:
                continue
            curve_id = parts[1]
            er_map[curve_id] = col

    eps_cols = [c for _, c in sorted(eps_map, key=lambda x: x[0])]

    stress_cols_dict = {}
    for curve_id, pts_dict in stress_map.items():
        stress_cols_dict[curve_id] = [pts_dict[k] for k in sorted(pts_dict.keys())]

    for curve_id in stress_cols_dict.keys():
        if curve_id not in er_map:
            raise ValueError(f"Для кривой {curve_id} нет столбца er_{curve_id}")

    def rate_to_float(s: str) -> float:
        return float(s.replace("p", "."))

    curve_ids = sorted(stress_cols_dict.keys(), key=rate_to_float)

    return eps_cols, stress_cols_dict, er_map, curve_ids


def row_to_image_and_meta(
    row: pd.Series,
    eps_cols,
    stress_cols_dict,
    er_cols_dict,
    curve_ids,
    use_log_er: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    eps = row[eps_cols].values.astype(np.float32)
    N = len(eps)
    M = len(curve_ids)

    img = np.zeros((M, N), dtype=np.float32)
    er_feats = np.zeros((M,), dtype=np.float32)

    for i, curve_id in enumerate(curve_ids):
        sigma = row[stress_cols_dict[curve_id]].values.astype(np.float32)
        img[i, :] = sigma

        er_val = float(row[er_cols_dict[curve_id]])
        if use_log_er:
            er_feats[i] = 0.0 if er_val <= 0 else float(np.log10(er_val))
        else:
            er_feats[i] = float(er_val)

    img = img[None, :, :]  # (1, M, N)
    meta = np.concatenate([eps, er_feats], dtype=np.float32)  # (N + M,)

    return img, meta


class CurvesTensorSigmaMetaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols,
        eps_cols,
        stress_cols_dict,
        er_cols_dict,
        curve_ids,
        use_log_er: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.target_cols = list(target_cols)
        self.eps_cols = eps_cols
        self.stress_cols_dict = stress_cols_dict
        self.er_cols_dict = er_cols_dict
        self.curve_ids = curve_ids
        self.use_log_er = use_log_er

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_np, meta_np = row_to_image_and_meta(
            row,
            eps_cols=self.eps_cols,
            stress_cols_dict=self.stress_cols_dict,
            er_cols_dict=self.er_cols_dict,
            curve_ids=self.curve_ids,
            use_log_er=self.use_log_er,
        )
        y_np = row[self.target_cols].values.astype(np.float32)

        return torch.from_numpy(img_np), torch.from_numpy(meta_np), torch.from_numpy(y_np)


class CurvesCNNWithMeta(nn.Module):
    """
    forward_scaled(img, meta) -> предсказание в стандартизованном пространстве логарифмов
    forward(img, meta)        -> (K, m) в исходных единицах (expm1 после стандартизации)
    """

    def __init__(self, in_channels: int, meta_dim: int, target_mean, target_std, use_log_target: bool = True):
        super().__init__()
        self.use_log_target = use_log_target

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, CONV_CHANNELS[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS[0], CONV_CHANNELS[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        conv_out_dim = CONV_CHANNELS[1] * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        target_mean = np.asarray(target_mean, dtype=np.float32)
        target_std = np.asarray(target_std, dtype=np.float32)
        target_std[target_std == 0.0] = 1.0

        self.register_buffer("y_mean", torch.from_numpy(target_mean).view(1, -1))
        self.register_buffer("y_std", torch.from_numpy(target_std).view(1, -1))

    def _core(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = self.conv(img)
        x = torch.flatten(x, 1)
        x = torch.cat([x, meta], dim=1)
        x = self.fc(x)
        return x

    def forward_scaled(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказание в стандартизованном пространстве логарифмов"""
        return self._core(img, meta)

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказание в исходных единицах (K, m)"""
        z = self._core(img, meta)                     # стандартизованный логарифм
        if self.use_log_target:
            # восстанавливаем логарифм -> expm1
            log_pred = z * self.y_std + self.y_mean
            return torch.expm1(log_pred)
        else:
            # прямой случай (без логарифмирования)
            return z * self.y_std + self.y_mean


def train_one_epoch(model, loader, optimizer, criterion, use_log_target: bool) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for img_batch, meta_batch, y_batch in loader:
        img_batch = img_batch.to(DEVICE)
        meta_batch = meta_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # преобразование y в стандартизованные логарифмы
        if use_log_target:
            y_log = torch.log1p(y_batch)
        else:
            y_log = y_batch
        y_scaled = (y_log - model.y_mean) / model.y_std

        optimizer.zero_grad()
        pred_scaled = model.forward_scaled(img_batch, meta_batch)
        loss = criterion(pred_scaled, y_scaled)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_model(model, loader, use_log_target: bool):
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0
    y_true_list = []
    y_pred_list = []

    for img_batch, meta_batch, y_batch in loader:
        img_batch = img_batch.to(DEVICE)
        meta_batch = meta_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # преобразование y в стандартизованные логарифмы для расчёта loss
        if use_log_target:
            y_log = torch.log1p(y_batch)
        else:
            y_log = y_batch
        y_scaled = (y_log - model.y_mean) / model.y_std

        pred_scaled = model.forward_scaled(img_batch, meta_batch)
        loss = criterion(pred_scaled, y_scaled)

        total_loss += float(loss.item())
        n_batches += 1

        # предсказания в исходных единицах (через forward)
        pred = model(img_batch, meta_batch)
        y_true_list.append(y_batch.cpu().numpy())
        y_pred_list.append(pred.cpu().numpy())

    if not y_true_list:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / max(1, n_batches)
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    return avg_loss, float(rmse[0]), float(rmse[1]), float(r2[0]), float(r2[1])


def save_torchscript_with_fallback(model_cpu: nn.Module, example_inputs: tuple[torch.Tensor, ...], model_path: Path) -> str:
    """Пытается torch.jit.script, при неудаче – torch.jit.trace."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not hasattr(model_cpu, "__annotations__"):
            model_cpu.__annotations__ = {}
        scripted = torch.jit.script(model_cpu)
        with open(model_path, "wb") as f:
            torch.jit.save(scripted, f)
        return "script"
    except Exception as e:
        print(f"[WARN] torch.jit.script() не сработал: {e}")
        print("[WARN] Переходим на torch.jit.trace()...")

    traced = torch.jit.trace(model_cpu, example_inputs, strict=False)
    try:
        traced = torch.jit.freeze(traced)
    except Exception:
        pass

    with open(model_path, "wb") as f:
        torch.jit.save(traced, f)
    return "trace"


def train_tensor_cnn_sigma_meta(
    csv_path: str,
    out_base_dir: str | None = None,
    experiment_subdir: str = "tensor_cnn_sigma_meta",
    target_cols: list[str] | tuple[str, str] = ("K", "m"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    base_dir = Path(out_base_dir) if out_base_dir else csv_path.parent
    dataset_dir = base_dir / csv_path.stem
    out_dir = dataset_dir / experiment_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю данные из {csv_path}...")
    df = read_csv_smart(csv_path)

    target_cols = list(target_cols)
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"В датасете нет столбца '{col}'.")

    eps_cols, stress_cols_dict, er_cols_dict, curve_ids = parse_column_structure(
        list(df.columns),
        target_cols,
    )

    print(f"Найдено точек по eps: {len(eps_cols)}")
    print(f"Найдено кривых: {len(curve_ids)} (curve_ids = {curve_ids})")

    # Групповое разбиение по m (см. group_split.py)
    train_full_df, test_df = group_split(df, groups=groups_from(df), test_size=test_size, random_state=random_state)

    # ----- ВЫЧИСЛЕНИЕ СТАТИСТИК ДЛЯ ЦЕЛЕЙ (с логарифмированием, если включено) -----
    y_train_full_np = train_full_df[target_cols].values.astype(np.float32)
    if USE_LOG_TARGET:
        y_transformed = np.log1p(y_train_full_np)
    else:
        y_transformed = y_train_full_np

    y_mean = y_transformed.mean(axis=0)
    y_std = y_transformed.std(axis=0)
    y_std[y_std == 0.0] = 1.0

    print(f"Средние по таргетам (в {'log1p+стандарт' if USE_LOG_TARGET else 'исходном'} пространстве): {dict(zip(target_cols, y_mean))}")
    print(f"Std по таргетам: {dict(zip(target_cols, y_std))}")

    # Разбиение train/valid
    train_df, valid_df = group_split(
        train_full_df, groups=groups_from(train_full_df), test_size=valid_size, random_state=random_state + 1
    )
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}, Test size: {len(test_df)}")

    train_dataset = CurvesTensorSigmaMetaDataset(
        train_df, target_cols, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, USE_LOG_ER
    )
    valid_dataset = CurvesTensorSigmaMetaDataset(
        valid_df, target_cols, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, USE_LOG_ER
    )
    test_dataset = CurvesTensorSigmaMetaDataset(
        test_df, target_cols, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, USE_LOG_ER
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    meta_dim = len(eps_cols) + len(curve_ids)
    in_channels = 1

    model = CurvesCNNWithMeta(
        in_channels=in_channels,
        meta_dim=meta_dim,
        target_mean=y_mean,
        target_std=y_std,
        use_log_target=USE_LOG_TARGET,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    history = {
        "epoch": [],
        "train_loss_scaled": [],
        "valid_loss_scaled": [],
        "valid_rmse_K": [],
        "valid_rmse_m": [],
        "valid_r2_K": [],
        "valid_r2_m": [],
    }

    best_valid_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    print(f"Начинаю обучение на устройстве: {DEVICE}")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, USE_LOG_TARGET)
        valid_loss, rmse_K_v, rmse_m_v, r2_K_v, r2_m_v = evaluate_model(model, valid_loader, USE_LOG_TARGET)

        history["epoch"].append(epoch)
        history["train_loss_scaled"].append(train_loss)
        history["valid_loss_scaled"].append(valid_loss)
        history["valid_rmse_K"].append(rmse_K_v)
        history["valid_rmse_m"].append(rmse_m_v)
        history["valid_r2_K"].append(r2_K_v)
        history["valid_r2_m"].append(r2_m_v)

        print(
            f"Эпоха {epoch:3d}: "
            f"train_loss_scaled={train_loss:.6f} | valid_loss_scaled={valid_loss:.6f} | "
            f"RMSE(K)={rmse_K_v:.4f}, RMSE(m)={rmse_m_v:.4f} | "
            f"R2(K)={r2_K_v:.4f}, R2(m)={r2_m_v:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            print(f"Ранняя остановка: {PATIENCE} эпох без улучшения")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Загружено состояние лучшей модели (эпоха {best_epoch}, valid_loss_scaled={best_valid_loss:.6f})")

    # Оценка на train_full и test
    train_full_dataset = CurvesTensorSigmaMetaDataset(
        train_full_df, target_cols, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, USE_LOG_ER
    )
    train_full_loader = DataLoader(train_full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    _, rmse_K_tr, rmse_m_tr, r2_K_tr, r2_m_tr = evaluate_model(model, train_full_loader, USE_LOG_TARGET)
    _, rmse_K_te, rmse_m_te, r2_K_te, r2_m_te = evaluate_model(model, test_loader, USE_LOG_TARGET)

    metrics = {
        "train": {
            "rmse": [rmse_K_tr, rmse_m_tr],
            "rmse_mean": float(np.mean([rmse_K_tr, rmse_m_tr])),
            "r2": [r2_K_tr, r2_m_tr],
        },
        "test": {
            "rmse": [rmse_K_te, rmse_m_te],
            "rmse_mean": float(np.mean([rmse_K_te, rmse_m_te])),
            "r2": [r2_K_te, r2_m_te],
        },
        "training": {
            "n_epochs": N_EPOCHS,
            "best_epoch": best_epoch,
            "best_valid_loss_scaled": best_valid_loss,
        },
        "tensor_structure": {
            "channels": in_channels,
            "n_curves": len(curve_ids),
            "n_eps_points": len(eps_cols),
            "meta_dim": meta_dim,
        },
        "target_columns": target_cols,
        "curve_ids": curve_ids,
        "eps_columns": eps_cols,
        "use_log_er": USE_LOG_ER,
        "use_log_target": USE_LOG_TARGET,        # НОВО: флаг в метаинформации
        "target_mean": y_mean.tolist(),
        "target_std": y_std.tolist(),
        "n_samples": {
            "total": int(len(df)),
            "train_full": int(len(train_full_df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Метрики сохранены в {metrics_path}")

    # График лосса
    epochs_arr = np.array(history["epoch"], dtype=int)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_arr, history["train_loss_scaled"], label="Train loss (scaled)")
    plt.plot(epochs_arr, history["valid_loss_scaled"], label="Valid loss (scaled)")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE (scaled targets)")
    plt.title("Training curve (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss_scaled.png", dpi=200)
    plt.close()

    # Parity plots по тесту
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for img_batch, meta_batch, y_batch in test_loader:
            img_batch = img_batch.to(DEVICE)
            meta_batch = meta_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(img_batch, meta_batch)
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    for i, col in enumerate(target_cols):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())

        plt.figure(figsize=(5, 5))
        plt.scatter(true_vals, pred_vals, alpha=0.7)
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.xlabel(f"Истинное {col}")
        plt.ylabel(f"Предсказанное {col}")
        plt.title(f"Parity plot для {col} (Tensor CNN σ+meta)")
        plt.tight_layout()
        plt.savefig(out_dir / f"parity_{col}.png", dpi=200)
        plt.close()

    # ---------- СОХРАНЕНИЕ TorchScript + META ----------
    model_cpu = model.to("cpu").eval()
    M = len(curve_ids)
    N = len(eps_cols)
    example_img = torch.zeros(1, 1, M, N, dtype=torch.float32)
    example_meta = torch.zeros(1, meta_dim, dtype=torch.float32)

    model_path = out_dir / MODEL_FILENAME
    mode = save_torchscript_with_fallback(model_cpu, (example_img, example_meta), model_path)
    print(f"TorchScript модель сохранена ({mode}) в {model_path}")

    # Мета-информация для инференса
    meta = {
        "model_type": "tensor_cnn_sigma_meta",
        "use_log_er": USE_LOG_ER,
        "use_log_target": USE_LOG_TARGET,
        "target_columns": target_cols,
        "curve_ids": curve_ids,
        "eps_columns": eps_cols,
        "stress_columns": stress_cols_dict,
        "er_columns": er_cols_dict,
        "meta_dim": int(meta_dim),
        "image_shape": {"C": 1, "M": int(M), "N": int(N)},
        "meta_layout": "meta = [eps_1..eps_N, er_feat_1..er_feat_M]",
    }
    meta_path = out_dir / META_FILENAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Meta сохранена в {meta_path}")

    # Демонстрация предсказания после сохранения
    try:
        img0, meta0, _ = test_dataset[0]
        img0 = img0.unsqueeze(0)
        meta0 = meta0.unsqueeze(0)

        loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
        with torch.no_grad():
            pred0 = loaded(img0, meta0).cpu().numpy()[0]
        print("Demo predict (K,m):", pred0.tolist())
    except Exception as e:
        print(f"[WARN] Demo predict после сохранения не выполнен: {e}")

    print("Готово.")


if __name__ == "__main__":
    train_tensor_cnn_sigma_meta(
        CSV_PATH,
        out_base_dir=OUT_BASE_DIR,
        experiment_subdir=EXPERIMENT_SUBDIR,
        target_cols=TARGET_COLS,
        test_size=TEST_SIZE,
        valid_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )