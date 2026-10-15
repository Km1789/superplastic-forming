# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import csv
import json
import joblib
import numpy as np
import pandas as pd

# Для .pt моделей
import torch


# ============================================================
# SETTINGS
# ============================================================

#MODELS_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data/three_curves_10_type_2_for_real_data"
MODELS_BASE_DIR = r"data/material/Graphics/Models_graphics_3_curves_for_real_data/three_curves_10_type_2_for_real_data"

MODEL_DIR_NAMES = [
    "Classic_GB_log",
    "mlp_optuna_log",
    "tensor_cnn",
    "tensor_cnn_sigma_meta",
    "tree",
]

DATASET_PATH = r"data/material/Datasets/three_polinomial_curves_10_type_2.csv"

OUTPUT_CSV_PATH = r"data/material/run_joblib/results/1_pred_three_curves_10_type_2.csv"
OUTPUT_LOG_CSV_PATH = r"data/material/run_joblib/results/1_log_three_curves_10_type_2.csv"

OUTPUT_CSV_SEP = ";"

DROP_TARGET_COLUMNS_IF_PRESENT = True
TARGET_COLUMNS = ["K", "m"]

FILL_MISSING_FEATURES_WITH_ZERO = False
DEVICE = "cpu"


# ============================================================
# CSV LOADING
# ============================================================

def detect_csv_separator_and_skiprows(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()

    if first_line.startswith("sep=") and len(first_line) >= 5:
        sep = first_line[4:]
        return sep, 1

    with open(file_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter, 0
    except Exception:
        return ",", 0


def load_dataset_raw_as_strings(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file not found: {}".format(dataset_path))

    sep, skiprows = detect_csv_separator_and_skiprows(dataset_path)

    df = pd.read_csv(
        dataset_path,
        sep=sep,
        skiprows=skiprows,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    bad_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if bad_cols:
        df = df.drop(columns=bad_cols)

    if DROP_TARGET_COLUMNS_IF_PRESENT:
        cols_to_drop = [c for c in TARGET_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    return df


# ============================================================
# VALUE CLEANING
# ============================================================

MONTH_MAP = {
    "янв": "01",
    "фев": "02",
    "мар": "03",
    "апр": "04",
    "май": "05",
    "мая": "05",
    "июн": "06",
    "июл": "07",
    "авг": "08",
    "сен": "09",
    "окт": "10",
    "ноя": "11",
    "дек": "12",
}


def clean_single_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    s = s.replace(" ", "")
    s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        pass

    s_low = s.lower()
    for month_name in MONTH_MAP.keys():
        if month_name in s_low:
            left = s_low.split(".")[0].strip()
            if left.isdigit():
                return float("0." + left)

    raise ValueError("Could not convert value to float: {!r}".format(x))


def clean_dataframe_to_numeric(df):
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = df_num[col].apply(clean_single_value)
    return df_num


# ============================================================
# FILE SEARCH HELPERS
# ============================================================

def find_single_file(model_dir, pattern):
    files = glob.glob(os.path.join(model_dir, pattern))
    if len(files) == 0:
        raise FileNotFoundError("No file '{}' found in directory: {}".format(pattern, model_dir))
    if len(files) > 1:
        raise RuntimeError(
            "More than one file '{}' found in directory: {}\nFound:\n{}".format(
                pattern, model_dir, "\n".join(files)
            )
        )
    return files[0]


def ensure_output_dir_exists(output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)


# ============================================================
# COMMON FEATURE ALIGNMENT
# ============================================================

def align_features_to_model(X, expected_features):
    if expected_features is None:
        return X.copy()

    missing = [c for c in expected_features if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_features]

    if missing:
        if FILL_MISSING_FEATURES_WITH_ZERO:
            X_aligned = X.copy()
            for col in missing:
                X_aligned[col] = 0.0
            X_aligned = X_aligned[expected_features]
            return X_aligned
        raise ValueError(
            "columns are missing: {}\nCurrent dataset columns: {}\nExpected columns: {}".format(
                set(missing), list(X.columns), expected_features
            )
        )

    X_aligned = X[expected_features].copy()

    if extra:
        print("  Extra columns ignored:", extra)

    return X_aligned


# ============================================================
# JOBLIB SUPPORT
# ============================================================

def try_extract_predictable_object(obj):
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        common_keys = [
            "model",
            "best_model",
            "estimator",
            "regressor",
            "pipeline",
            "best_estimator",
            "clf",
        ]
        for key in common_keys:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

        if "models" in obj:
            models_obj = obj["models"]
            if hasattr(models_obj, "predict"):
                return models_obj
            if isinstance(models_obj, dict):
                return models_obj

        for key, value in obj.items():
            if hasattr(value, "predict"):
                print("  Model extracted from dict key: {!r}".format(key))
                return value

        raise AttributeError(
            "Loaded object is dict, but no value with predict() was found. Dict keys: {}".format(
                list(obj.keys())
            )
        )

    raise AttributeError(
        "Loaded object does not have predict(). Type: {}".format(type(obj))
    )


def get_expected_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())[::-1]
        for step_name in step_names:
            step_obj = model.named_steps[step_name]
            if hasattr(step_obj, "feature_names_in_"):
                return list(step_obj.feature_names_in_)

            for attr_name in ["regressor", "estimator", "model"]:
                if hasattr(step_obj, attr_name):
                    inner = getattr(step_obj, attr_name)
                    if hasattr(inner, "feature_names_in_"):
                        return list(inner.feature_names_in_)

    return None


def normalize_prediction_to_km(prediction, model_name):
    pred = np.asarray(prediction)

    if pred.ndim == 1:
        if pred.shape[0] == 2:
            return float(pred[0]), float(pred[1])
        raise ValueError(
            "Model '{}' returned 1D prediction with shape {}. Expected 2 values.".format(
                model_name, pred.shape
            )
        )

    if pred.ndim == 2:
        if pred.shape == (1, 2):
            return float(pred[0, 0]), float(pred[0, 1])

        if pred.shape[0] > 1 and pred.shape[1] == 2:
            raise ValueError(
                "Model '{}' returned predictions for multiple rows: shape {}. "
                "This script expects exactly one input row.".format(model_name, pred.shape)
            )

        raise ValueError(
            "Model '{}' returned 2D prediction with shape {}. Expected (1, 2).".format(
                model_name, pred.shape
            )
        )

    raise ValueError(
        "Model '{}' returned unsupported prediction shape: {}".format(
            model_name, pred.shape
        )
    )


def predict_from_models_dict(models_dict, X):
    key_k = None
    key_m = None

    for key in models_dict.keys():
        key_str = str(key).strip().lower()
        if key_str == "k":
            key_k = key
        if key_str == "m":
            key_m = key

    if key_k is None or key_m is None:
        raise KeyError(
            "Could not find target models 'K' and 'm' inside models dict. Available keys: {}".format(
                list(models_dict.keys())
            )
        )

    model_k = models_dict[key_k]
    model_m = models_dict[key_m]

    if not hasattr(model_k, "predict"):
        raise AttributeError("Model for K does not have predict()")
    if not hasattr(model_m, "predict"):
        raise AttributeError("Model for m does not have predict()")

    pred_k = np.asarray(model_k.predict(X)).reshape(-1)
    pred_m = np.asarray(model_m.predict(X)).reshape(-1)

    if len(pred_k) != 1 or len(pred_m) != 1:
        raise ValueError("Models dict returned multiple predictions; expected exactly one row.")

    return float(pred_k[0]), float(pred_m[0])


def predict_with_joblib(model_path, X, model_name):
    loaded_obj = joblib.load(model_path)

    if isinstance(loaded_obj, dict) and "models" in loaded_obj and isinstance(loaded_obj["models"], dict):
        expected_features = loaded_obj.get("feature_columns", None)
        if expected_features is not None:
            print("  Expected feature count:", len(expected_features))
        else:
            print("  Could not determine expected feature names automatically")

        X_prepared = align_features_to_model(X, expected_features)
        print("  Dataset shape used for prediction:", X_prepared.shape)

        return predict_from_models_dict(loaded_obj["models"], X_prepared)

    model = try_extract_predictable_object(loaded_obj)

    expected_features = get_expected_feature_names(model)
    if expected_features is not None:
        print("  Expected feature count:", len(expected_features))
    else:
        print("  Could not determine expected feature names automatically")

    X_prepared = align_features_to_model(X, expected_features)
    print("  Dataset shape used for prediction:", X_prepared.shape)

    prediction = model.predict(X_prepared)
    return normalize_prediction_to_km(prediction, model_name)


# ============================================================
# PT SUPPORT
# ============================================================

def rate_to_float(rate_str):
    return float(str(rate_str).replace("p", "."))


def row_to_tensor_cnn(row, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, use_log_er):
    eps = row[eps_cols].values.astype(np.float32)
    N = len(eps)
    M = len(curve_ids)
    C = 3

    tensor = np.zeros((C, M, N), dtype=np.float32)

    for i, cid in enumerate(curve_ids):
        sigma = row[stress_cols_dict[cid]].values.astype(np.float32)
        er_val = float(row[er_cols_dict[cid]])

        if use_log_er:
            er_encoded = 0.0 if er_val <= 0 else float(np.log10(er_val))
        else:
            er_encoded = float(er_val)

        tensor[0, i, :] = sigma
        tensor[1, i, :] = er_encoded
        tensor[2, i, :] = eps

    return tensor


def row_to_image_and_meta_sigma_meta(row, eps_cols, stress_cols_dict, er_cols_dict, curve_ids, use_log_er):
    eps = row[eps_cols].values.astype(np.float32)
    N = len(eps)
    M = len(curve_ids)

    img = np.zeros((M, N), dtype=np.float32)
    er_feats = np.zeros((M,), dtype=np.float32)

    for i, cid in enumerate(curve_ids):
        sigma = row[stress_cols_dict[cid]].values.astype(np.float32)
        img[i, :] = sigma

        er_val = float(row[er_cols_dict[cid]])
        if use_log_er:
            er_feats[i] = 0.0 if er_val <= 0 else float(np.log10(er_val))
        else:
            er_feats[i] = float(er_val)

    img = img[None, :, :]  # (1,M,N)
    meta = np.concatenate([eps, er_feats]).astype(np.float32)

    return img, meta


def validate_meta_columns(meta, X):
    needed = []
    needed.extend(meta["eps_columns"])

    stress_columns = meta["stress_columns"]
    er_columns = meta["er_columns"]

    for cid, cols in stress_columns.items():
        needed.extend(cols)
        needed.append(er_columns[cid])

    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError("Dataset is missing columns required by PT model: {}".format(sorted(set(missing))))


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_with_tensor_cnn(model_dir, X):
    pt_path = os.path.join(model_dir, "tensor_cnn_model.pt")
    meta_path = os.path.join(model_dir, "tensor_cnn_meta.json")

    if not os.path.exists(pt_path):
        raise FileNotFoundError("File not found: {}".format(pt_path))
    if not os.path.exists(meta_path):
        raise FileNotFoundError("File not found: {}".format(meta_path))

    meta = load_json(meta_path)
    validate_meta_columns(meta, X)

    if len(X) != 1:
        raise ValueError("This script expects exactly one input row for PT models.")

    model = torch.jit.load(pt_path, map_location=DEVICE).eval()

    row = X.iloc[0]
    x_np = row_to_tensor_cnn(
        row=row,
        eps_cols=meta["eps_columns"],
        stress_cols_dict=meta["stress_columns"],
        er_cols_dict=meta["er_columns"],
        curve_ids=meta["curve_ids"],
        use_log_er=bool(meta.get("use_log_er", True)),
    )

    x_t = torch.from_numpy(x_np).unsqueeze(0).to(torch.float32)  # (1,C,M,N)

    with torch.no_grad():
        pred = model(x_t).cpu().numpy()

    return normalize_prediction_to_km(pred, "tensor_cnn")


def predict_with_tensor_cnn_sigma_meta(model_dir, X):
    pt_path = os.path.join(model_dir, "tensor_cnn_sigma_meta_model.pt")
    meta_path = os.path.join(model_dir, "tensor_cnn_sigma_meta_meta.json")

    if not os.path.exists(pt_path):
        raise FileNotFoundError("File not found: {}".format(pt_path))
    if not os.path.exists(meta_path):
        raise FileNotFoundError("File not found: {}".format(meta_path))

    meta = load_json(meta_path)
    validate_meta_columns(meta, X)

    if len(X) != 1:
        raise ValueError("This script expects exactly one input row for PT models.")

    model = torch.jit.load(pt_path, map_location=DEVICE).eval()

    row = X.iloc[0]
    img_np, meta_np = row_to_image_and_meta_sigma_meta(
        row=row,
        eps_cols=meta["eps_columns"],
        stress_cols_dict=meta["stress_columns"],
        er_cols_dict=meta["er_columns"],
        curve_ids=meta["curve_ids"],
        use_log_er=bool(meta.get("use_log_er", True)),
    )

    img_t = torch.from_numpy(img_np).unsqueeze(0).to(torch.float32)   # (1,1,M,N)
    meta_t = torch.from_numpy(meta_np).unsqueeze(0).to(torch.float32) # (1,meta_dim)

    with torch.no_grad():
        pred = model(img_t, meta_t).cpu().numpy()

    return normalize_prediction_to_km(pred, "tensor_cnn_sigma_meta")


# ============================================================
# DISPATCH
# ============================================================

def predict_for_model_dir(model_dir_name, X):
    model_dir = os.path.join(MODELS_BASE_DIR, model_dir_name)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Directory does not exist: {}".format(model_dir))

    if model_dir_name == "tensor_cnn":
        return predict_with_tensor_cnn(model_dir, X)

    if model_dir_name == "tensor_cnn_sigma_meta":
        return predict_with_tensor_cnn_sigma_meta(model_dir, X)

    joblib_path = find_single_file(model_dir, "*.joblib")
    print("  Found model file:", joblib_path)
    return predict_with_joblib(joblib_path, X, model_dir_name)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading dataset from:", DATASET_PATH)
    X_raw = load_dataset_raw_as_strings(DATASET_PATH)
    print("Detected dataset columns:", list(X_raw.columns))
    print("Dataset shape after loading:", X_raw.shape)

    print("\nConverting dataset values to numeric...")
    X = clean_dataframe_to_numeric(X_raw)
    print("Numeric conversion finished successfully.")
    print("Dataset shape after numeric conversion:", X.shape)

    result_dict = {}
    log_rows = []

    for model_dir_name in MODEL_DIR_NAMES:
        print("\nProcessing model directory:", model_dir_name)

        try:
            k_value, m_value = predict_for_model_dir(model_dir_name, X)

            print("  Prediction:")
            print("    K =", k_value)
            print("    m =", m_value)

            result_dict[model_dir_name] = [k_value, m_value]
            log_rows.append({
                "model_name": model_dir_name,
                "status": "ok",
                "message": ""
            })

        except Exception as e:
            print("  Error while processing '{}': {}".format(model_dir_name, e))
            log_rows.append({
                "model_name": model_dir_name,
                "status": "error",
                "message": str(e)
            })

    ensure_output_dir_exists(OUTPUT_CSV_PATH)
    ensure_output_dir_exists(OUTPUT_LOG_CSV_PATH)

    if result_dict:
        result_df = pd.DataFrame(result_dict, index=["K", "m"])
        result_df.to_csv(OUTPUT_CSV_PATH, sep=OUTPUT_CSV_SEP, encoding="utf-8-sig")
        print("\nPrediction table:")
        print(result_df)
        print("\nSaved prediction CSV to:")
        print(OUTPUT_CSV_PATH)
    else:
        print("\nNo successful predictions were obtained.")
        print("Prediction CSV was not created.")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUTPUT_LOG_CSV_PATH, sep=OUTPUT_CSV_SEP, index=False, encoding="utf-8-sig")
    print("\nSaved log CSV to:")
    print(OUTPUT_LOG_CSV_PATH)


if __name__ == "__main__":
    main()