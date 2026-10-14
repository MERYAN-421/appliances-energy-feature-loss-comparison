"""CRISP-DM Phase 4: Linear Regression Loss Objectives.

Provides 4 linear regression objectives:
1. Squared Error (OLS): Ordinary Least Squares, penalizes large residuals quadratically.
2. Huber: Quadratic near zero, linear beyond delta threshold, robust to outliers.
3. Absolute Error (Median): L1 / quantile loss at median (tau=0.5), robust.
4. Epsilon-insensitive: SGD with epsilon-insensitive tube loss (like Linear SVR).

All models are wrapped in a leakage-safe pipeline with SimpleImputer and StandardScaler.
"""
from __future__ import annotations

from typing import Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_loss_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns the 4 linear regression estimators with controlled parameters."""
    return {
        "Squared Error (OLS)": LinearRegression(),
        "Huber": HuberRegressor(max_iter=3000, alpha=1e-4),
        "Absolute Error (Median)": QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
        "Epsilon-insensitive": SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=5.0,  # 5 Wh tube (~5% of mean 97.7 Wh) ignores minor sensor jitter
            alpha=1e-4,   # Moderate L2 regularization
            random_state=random_state,
            max_iter=5000,
            tol=1e-4,
        ),
    }


def build_loss_pipeline(regressor: Any) -> Pipeline:
    """Wraps a regressor in an imputer and standard scaling pipeline for fair comparison."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])
