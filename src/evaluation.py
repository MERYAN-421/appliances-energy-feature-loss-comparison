"""CRISP-DM Phase 5: Evaluation & Conformal Prediction Intervals.

Computes:
- Point metrics: MAE, RMSE, R²
- Split-conformal 95% prediction interval calibrated on independent calibration split
- Interval coverage and mean width on held-out test split
"""
from __future__ import annotations

import math
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Computes MAE, RMSE, and R2 regression metrics."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "R2": float(r2_score(y_t, y_p)),
    }


def calibrate_conformal_half_width(y_cal: Any, y_cal_pred: Any, alpha: float = 0.05) -> float:
    """Calibrates conformal prediction interval half-width on independent calibration set.

    half_width W is chosen such that roughly (1 - alpha) of calibration residuals satisfy |e| <= W.
    """
    y_c = np.asarray(y_cal).ravel()
    y_cp = np.asarray(y_cal_pred).ravel()
    residuals = np.abs(y_c - y_cp)
    n = len(residuals)
    # Standard split conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    half_width = float(np.quantile(residuals, quantile_level, method="higher"))
    return half_width


def evaluate_intervals(
    y_test: Any, y_test_pred: Any, half_width: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Generates 95% prediction intervals and computes test coverage and mean width.

    Returns:
        (interval_coverage, mean_interval_width, lower_bounds, upper_bounds)
    """
    y_t = np.asarray(y_test).ravel()
    y_tp = np.asarray(y_test_pred).ravel()
    lower = y_tp - half_width
    upper = y_tp + half_width
    covered = (y_t >= lower) & (y_t <= upper)
    coverage = float(covered.mean())
    mean_width = float(2.0 * half_width)
    return coverage, mean_width, lower, upper
