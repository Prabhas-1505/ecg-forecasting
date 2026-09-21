"""
metrics.py — Evaluation Metrics for ECG Forecasting
====================================================
All metrics operate on NumPy arrays in the **original scale**
(i.e. after inverse-transforming the scaler).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List


# ──────────────────────────────────────────────
# CORE METRICS
# ──────────────────────────────────────────────
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Uses an epsilon floor to avoid division by zero.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all metrics and return as an ordered dictionary."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
    }


# ──────────────────────────────────────────────
# RESULTS FORMATTING
# ──────────────────────────────────────────────
def format_results_table(
    records: List[Dict],
    precision: int = 4,
) -> pd.DataFrame:
    """Convert a list of result dicts into a nicely formatted DataFrame.

    Parameters
    ----------
    records : list of dict
        Each dict should have keys like ``Fold``, ``Model``,
        ``MSE``, ``RMSE``, ``MAE``, etc.
    precision : int
        Number of decimal places for float columns.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(records)
    float_cols = df.select_dtypes(include=[np.floating, float]).columns
    for col in float_cols:
        df[col] = df[col].apply(lambda x: f"{x:.{precision}f}")
    return df


def summary_table(
    records: List[Dict],
    group_cols: List[str] = None,
    metric_cols: List[str] = None,
) -> pd.DataFrame:
    """Group results and compute means.

    Parameters
    ----------
    records : list of dict
    group_cols : list of str
        Columns to group by (default: ``['Scenario', 'Model']``).
    metric_cols : list of str
        Metric columns to average (default: ``['MSE', 'RMSE', 'MAE']``).

    Returns
    -------
    pd.DataFrame  with mean metrics per group.
    """
    if group_cols is None:
        group_cols = ["Scenario", "Model"]
    if metric_cols is None:
        metric_cols = ["MSE", "RMSE", "MAE"]

    df = pd.DataFrame(records)
    return df.groupby(group_cols)[metric_cols].mean()


# ──────────────────────────────────────────────
# STATISTICAL TESTS
# ──────────────────────────────────────────────
def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "MSE",
    alpha: float = 0.05,
) -> Dict:
    """Paired two-sided t-test between two sets of fold scores.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Per-fold metric values for model A and B.
    metric_name : str
        Label for the metric being compared.
    alpha : float
        Significance level.

    Returns
    -------
    dict with ``t_stat``, ``p_value``, ``significant``, ``summary``.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = p_value < alpha

    summary = (
        f"Paired t-test on {metric_name}: "
        f"t = {t_stat:.4f}, p = {p_value:.4f} — "
        f"{'Significant' if significant else 'Not significant'} "
        f"at α = {alpha}"
    )

    return {
        "metric": metric_name,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
        "summary": summary,
    }
