"""
Shared preprocessing for training (EDA_and_Preprocessing.ipynb)
and prediction (test/test_model.ipynb).

Keep ALL cleaning / feature-engineering / imputation logic here so that
any change is applied identically to training data and to new data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---- constants ---------------------------------------------------------
ID_COLS = ["Id", "SeriousDlqin2yrs"]          # never used as features
MISSING_FLAG_COLS = ["MonthlyIncome", "NumberOfDependents"]
SENTINEL_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_VALUES = [96, 98]
MIN_AGE = 18


# ---- steps -------------------------------------------------------------
def clean_raw(X):
    """Strip names, add missing/invalid flags, turn invalid values into NaN.
    No imputation here (imputation needs training medians)."""
    df = X.copy()
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    df.columns = df.columns.str.strip()

    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            df[f"{col}Missing"] = df[col].isna().astype(int)

    for col in SENTINEL_COLS:
        if col in df.columns:
            is_sentinel = df[col].isin(SENTINEL_VALUES)
            df[f"{col}InvalidFlag"] = is_sentinel.astype(int)
            df.loc[is_sentinel, col] = np.nan

    if "age" in df.columns:
        df.loc[df["age"] < MIN_AGE, "age"] = np.nan

    return df


def impute(df, medians):
    """Fill NaN with the (training) medians."""
    df = df.copy()
    for col, median_value in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_value)
    return df


def add_features(df):
    """Feature engineering. Must run AFTER imputation (same as training)."""
    df = df.copy()
    df["TotalMonthlyDebt"] = df["MonthlyIncome"] * df["DebtRatio"]
    df["IncomePerDependentAdjusted"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["TotalDelinquencies"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["RealEstateToTotalCredit"] = (
        df["NumberRealEstateLoansOrLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    )
    return df


# ---- the object used by both notebooks --------------------------------
class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """fit() on training data only -> learns medians + final column order.
    transform() on any data -> clean -> impute -> add features -> order columns.
    Works inside sklearn Pipeline, and can be saved/loaded as JSON."""

    def __init__(self):
        self.medians_ = None
        self.feature_columns_ = None

    def fit(self, X, y=None):
        cleaned = clean_raw(X)
        numeric = cleaned.select_dtypes(include="number")
        medians = numeric.median().dropna()
        self.medians_ = {c: float(v) for c, v in medians.items()}
        self.feature_columns_ = list(add_features(impute(cleaned, self.medians_)).columns)
        return self

    def clean_and_impute(self, X):
        """Everything except feature engineering (handy for EDA)."""
        self._check_fitted()
        return impute(clean_raw(X), self.medians_)

    def transform(self, X):
        self._check_fitted()
        df = add_features(self.clean_and_impute(X))
        missing = [c for c in self.feature_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input data: {missing}")
        return df[self.feature_columns_]

    # -- persistence (JSON, so no dependency on pickling this class) --
    def save(self, path):
        self._check_fitted()
        with open(Path(path), "w") as f:
            json.dump({"medians": self.medians_, "feature_columns": self.feature_columns_}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(Path(path), "r") as f:
            params = json.load(f)
        obj = cls()
        obj.medians_ = params["medians"]
        obj.feature_columns_ = params["feature_columns"]
        return obj

    def _check_fitted(self):
        if self.medians_ is None or self.feature_columns_ is None:
            raise RuntimeError("CreditPreprocessor is not fitted. Call fit() or load() first.")
