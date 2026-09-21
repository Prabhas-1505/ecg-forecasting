"""
data.py — Data Loading, Preprocessing & PyTorch Dataset
========================================================
Handles CSV ingestion, missing-value treatment, outlier detection
via IQR, mean/median imputation, MinMax scaling, and a reusable
PyTorch Dataset for the ECG multi-step forecasting task.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List, Optional


# ──────────────────────────────────────────────
# 1. RAW DATA LOADING
# ──────────────────────────────────────────────
def load_raw_dataframe(filepath: str) -> pd.DataFrame:
    """Load the ECG CSV and perform basic cleaning.

    Steps
    -----
    1. Read CSV.
    2. Replace non-numeric sentinel strings ('NT') with NaN.
    3. Coerce every column to numeric.
    4. Fill remaining NaNs with 0 (preserves tensor shape).

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. ``data/ecg_raw.csv``).

    Returns
    -------
    pd.DataFrame
        Cleaned, fully numeric DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    df = df.replace("NT", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df


# ──────────────────────────────────────────────
# 2. STRUCTURED ARRAY EXTRACTION
# ──────────────────────────────────────────────
NUM_STEPS = 17
NUM_FEATURES = 8


def _get_col_name(base: str, step_idx: int) -> str:
    """Return the column name for feature *base* at time-step *step_idx*.

    Convention used in the original spreadsheet:
    - Step 0  → ``O1`` … ``O8``
    - Step k  → ``O1.k`` … ``O8.k``
    """
    return base if step_idx == 0 else f"{base}.{step_idx}"


def extract_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the flat DataFrame into 3-D input and output arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from :func:`load_raw_dataframe`.

    Returns
    -------
    X : np.ndarray, shape ``(n_patients, 17, 8)``
        Input features O1–O8 across 17 time-steps.
    Y : np.ndarray, shape ``(n_patients, 17, 1)``
        Target outputs (Output1–Output17).
    """
    n = df.shape[0]
    X = np.zeros((n, NUM_STEPS, NUM_FEATURES), dtype=np.float32)
    Y = np.zeros((n, NUM_STEPS, 1), dtype=np.float32)

    for step in range(NUM_STEPS):
        for f in range(1, NUM_FEATURES + 1):
            col = _get_col_name(f"O{f}", step)
            if col in df.columns:
                X[:, step, f - 1] = df[col].values.astype(np.float32)

        out_col = f"Output{step + 1}"
        if out_col in df.columns:
            Y[:, step, 0] = df[out_col].values.astype(np.float32)

    return X, Y


# ──────────────────────────────────────────────
# 3. OUTLIER DETECTION & IMPUTATION
# ──────────────────────────────────────────────
def detect_outliers_iqr(
    series: np.ndarray, k: float = 1.5
) -> Tuple[np.ndarray, Dict]:
    """Detect outliers using the Inter-Quartile Range method.

    Parameters
    ----------
    series : np.ndarray
        1-D array of values.
    k : float
        IQR multiplier (default 1.5 → standard fence).

    Returns
    -------
    mask : np.ndarray[bool]
        ``True`` where the value is an outlier.
    stats : dict
        Keys: ``q1``, ``q3``, ``iqr``, ``lower``, ``upper``,
        ``n_outliers``, ``outlier_values``.
    """
    # Exclude zeros (missing-value fill) from quartile calculation
    non_zero = series[series != 0]
    if len(non_zero) == 0:
        return np.zeros_like(series, dtype=bool), {}

    q1 = np.percentile(non_zero, 25)
    q3 = np.percentile(non_zero, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (series < lower) | (series > upper)
    # Don't flag zeros as outliers (they are missing-fill)
    mask = mask & (series != 0)

    stats = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "n_outliers": int(mask.sum()),
        "outlier_values": series[mask].tolist(),
    }
    return mask, stats


def impute_outliers(
    series: np.ndarray,
    mask: np.ndarray,
    strategy: str = "mean",
) -> Tuple[np.ndarray, float]:
    """Replace flagged outliers with a summary statistic of the clean data.

    Parameters
    ----------
    series : np.ndarray
        Original 1-D array.
    mask : np.ndarray[bool]
        Outlier mask (``True`` = outlier).
    strategy : str
        ``'mean'`` or ``'median'``.

    Returns
    -------
    imputed : np.ndarray
        Array with outliers replaced.
    fill_value : float
        The imputation value used.
    """
    clean = series[(~mask) & (series != 0)]
    fill_value = float(np.mean(clean) if strategy == "mean" else np.median(clean))
    imputed = series.copy()
    imputed[mask] = fill_value
    return imputed, fill_value


# ──────────────────────────────────────────────
# 4. SCALING
# ──────────────────────────────────────────────
def fit_scalers(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Fit MinMax scalers and return transformed arrays.

    Flattens to 2-D for fitting, then reshapes back.

    Returns
    -------
    X_scaled, Y_scaled, scaler_x, scaler_y
    """
    n, t, f = X.shape
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X.reshape(-1, f)).reshape(n, t, f)
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).reshape(n, t, 1)

    return (
        X_scaled.astype(np.float32),
        Y_scaled.astype(np.float32),
        scaler_x,
        scaler_y,
    )


# ──────────────────────────────────────────────
# 5. PYTORCH DATASET
# ──────────────────────────────────────────────
class ECGDataset(Dataset):
    """PyTorch Dataset wrapping scaled ECG arrays.

    Parameters
    ----------
    X : np.ndarray, shape ``(n, 17, 8)``
    Y : np.ndarray, shape ``(n, 17, 1)``
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────────
# 6. K-FOLD UTILITY
# ──────────────────────────────────────────────
def get_kfold_splits(
    n_samples: int, n_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return reproducible K-Fold train/test index pairs.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int
        Number of folds.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(n_samples)
    return [(train_idx, test_idx) for train_idx, test_idx in kf.split(dummy)]


# ──────────────────────────────────────────────
# 7. CONVENIENCE: FULL PIPELINE
# ──────────────────────────────────────────────
def prepare_data(
    filepath: str,
    impute_output13: bool = False,
) -> Dict:
    """End-to-end data preparation pipeline.

    Parameters
    ----------
    filepath : str
        Path to ``ecg_raw.csv``.
    impute_output13 : bool
        If ``True``, detect and mean-impute outliers in Output13.

    Returns
    -------
    dict with keys:
        ``X_scaled``, ``Y_scaled``, ``scaler_x``, ``scaler_y``,
        ``X_raw``, ``Y_raw``, ``df``, ``n_patients``,
        ``outlier_info`` (if imputation was applied).
    """
    df = load_raw_dataframe(filepath)
    X, Y = extract_arrays(df)

    result: Dict = {
        "df": df,
        "n_patients": X.shape[0],
    }

    if impute_output13:
        output13_raw = Y[:, 12, 0].copy()  # Clone to preserve raw values before mutation
        mask, stats = detect_outliers_iqr(output13_raw, k=3.0)  # Use k=3.0 to isolate true outliers (124)
        imputed, fill_val = impute_outliers(output13_raw, mask, strategy="mean")
        Y[:, 12, 0] = imputed
        result["outlier_info"] = {
            "stats": stats,
            "fill_value": fill_val,
            "variance_before": float(np.var(output13_raw[output13_raw != 0])),
            "variance_after": float(np.var(imputed[imputed != 0])),
        }

    result.update({
        "X_raw": X.copy(),
        "Y_raw": Y.copy(),
    })

    X_s, Y_s, sx, sy = fit_scalers(X, Y)
    result.update(
        {
            "X_scaled": X_s,
            "Y_scaled": Y_s,
            "scaler_x": sx,
            "scaler_y": sy,
        }
    )
    return result
