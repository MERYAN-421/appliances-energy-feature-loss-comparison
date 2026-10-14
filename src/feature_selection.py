"""CRISP-DM Phase 4: Feature Selection Modules.

Provides 7 feature selection methods + 1 baseline:
1. Pearson correlation ranking (top K) - Filter
2. F-test / SelectKBest (top K) - Filter
3. Mutual Information (top K) - Filter (fitted on standardized data)
4. Recursive Feature Elimination - RFE (top K) - Wrapper (fitted on standardized data)
5. Forward Sequential Feature Selection - Forward SFS (top K) - Wrapper (TimeSeriesSplit CV)
6. Backward Sequential Feature Selection - Backward SFS (top K) - Wrapper (TimeSeriesSplit CV)
7. LassoCV coefficient ranking (top K) - Embedded (fitted on standardized data with TimeSeriesSplit)
8. Baseline (all 27 features, passthrough)

All selectors include leakage-safe median imputation and appropriate standardization.
"""
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    RFE,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit


class PearsonTopK(BaseEstimator, TransformerMixin):
    """Rank predictors by absolute Pearson correlation |r(X_i, y)| using training data."""

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        X_arr = imp.fit_transform(np.asarray(X))
        y_arr = np.asarray(y).ravel()
        scores = np.abs(r_regression(X_arr, y_arr, force_finite=True))
        self.scores_ = np.nan_to_num(scores, nan=0.0)
        top = np.argsort(-self.scores_, kind="stable")[: self.k]
        self.support_ = np.zeros(X_arr.shape[1], dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class FTestTopK(BaseEstimator, TransformerMixin):
    """Rank predictors by univariate F-test regression score using training data."""

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        X_arr = imp.fit_transform(np.asarray(X))
        y_arr = np.asarray(y).ravel()
        F, _ = f_regression(X_arr, y_arr, force_finite=True)
        self.scores_ = np.nan_to_num(F, nan=0.0)
        top = np.argsort(-self.scores_, kind="stable")[: self.k]
        self.support_ = np.zeros(X_arr.shape[1], dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class MutualInfoTopK(BaseEstimator, TransformerMixin):
    """Rank predictors by Mutual Information with standardized features to prevent Euclidean scale distortion."""

    def __init__(self, k: int = 15, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imp.fit_transform(np.asarray(X)))
        y_arr = np.asarray(y).ravel()
        self.scores_ = mutual_info_regression(X_scaled, y_arr, random_state=self.random_state)
        top = np.argsort(-self.scores_, kind="stable")[: self.k]
        self.support_ = np.zeros(X_scaled.shape[1], dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class RFETopK(BaseEstimator, TransformerMixin):
    """Recursive Feature Elimination with standardized LinearRegression coefficients for scale-fair rankings."""

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imp.fit_transform(np.asarray(X)))
        y_arr = np.asarray(y).ravel()
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=self.k)
        rfe.fit(X_scaled, y_arr)
        self.support_ = rfe.support_
        self.ranking_ = rfe.ranking_
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class SFSTopK(BaseEstimator, TransformerMixin):
    """Sequential Feature Selector (Forward or Backward) with TimeSeriesSplit and standardized inputs."""

    def __init__(self, k: int = 15, direction: str = "forward", cv_splits: int = 3):
        self.k = k
        self.direction = direction
        self.cv_splits = cv_splits

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imp.fit_transform(np.asarray(X)))
        y_arr = np.asarray(y).ravel()
        sfs = SequentialFeatureSelector(
            estimator=LinearRegression(),
            n_features_to_select=self.k,
            direction=self.direction,
            cv=TimeSeriesSplit(n_splits=self.cv_splits),
            n_jobs=1,
        )
        sfs.fit(X_scaled, y_arr)
        self.support_ = sfs.support_
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class LassoCVTopK(BaseEstimator, TransformerMixin):
    """Rank predictors by standardized LassoCV coefficient magnitude |coef_i| using TimeSeriesSplit."""

    def __init__(self, k: int = 15, cv_splits: int = 3, max_iter: int = 10000, random_state: int = 42):
        self.k = k
        self.cv_splits = cv_splits
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imp.fit_transform(np.asarray(X)))
        y_arr = np.asarray(y).ravel()
        self.estimator_ = LassoCV(
            cv=TimeSeriesSplit(n_splits=self.cv_splits),
            max_iter=self.max_iter,
            random_state=self.random_state,
            alphas=50,
        )
        self.estimator_.fit(X_scaled, y_arr)
        self.importances_ = np.abs(self.estimator_.coef_)
        top = np.argsort(-self.importances_, kind="stable")[: self.k]
        self.support_ = np.zeros(X_scaled.shape[1], dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


class PassthroughSelector(BaseEstimator, TransformerMixin):
    """Passthrough baseline: keeps all candidate features."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.support_ = np.ones(X_arr.shape[1], dtype=bool)
        return self

    def transform(self, X):
        return np.asarray(X)

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()


def get_selectors(k: int = 15, random_state: int = 42) -> Dict[str, Any]:
    """Returns the 7 feature selectors and the baseline."""
    return {
        "Pearson": PearsonTopK(k=k),
        "F-test": FTestTopK(k=k),
        "Mutual Information": MutualInfoTopK(k=k, random_state=random_state),
        "RFE": RFETopK(k=k),
        "Forward SFS": SFSTopK(k=k, direction="forward", cv_splits=3),
        "Backward SFS": SFSTopK(k=k, direction="backward", cv_splits=3),
        "LassoCV ranking": LassoCVTopK(k=k, random_state=random_state),
        "Baseline": PassthroughSelector(),
    }


def extract_selected_features(selector: Any, feature_names: List[str]) -> List[str]:
    """Extracts the names of the selected features given a fitted selector."""
    support = selector.get_support()
    return [name for name, is_sel in zip(feature_names, support) if is_sel]
