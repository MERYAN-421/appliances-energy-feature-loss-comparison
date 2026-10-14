"""CRISP-DM Phase 4: Linear Regression Loss Objectives.

Provides 4 linear regression objectives:
1. Squared Error (OLS): Ordinary Least Squares, penalizes large residuals quadratically.
2. Huber: Quadratic near zero, linear beyond delta threshold, robust to outliers.
3. Absolute Error (Median): L1 / quantile loss at median (tau=0.5), robust.
4. Epsilon-insensitive: SGD with epsilon-insensitive tube loss (like Linear SVR).
"""
from __future__ import annotations

from typing import Dict, Any
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_loss_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns the 4 linear regression estimators."""
    return {
        "Squared Error (OLS)": LinearRegression(),
        "Huber": HuberRegressor(max_iter=2000),
        "Absolute Error (Median)": QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
        "Epsilon-insensitive": SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=0.1,
            random_state=random_state,
            max_iter=3000,
            tol=1e-4,
        ),
    }


def build_loss_pipeline(regressor: Any) -> Pipeline:
    """Wraps a regressor in a standard scaling pipeline for fair comparison."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])
