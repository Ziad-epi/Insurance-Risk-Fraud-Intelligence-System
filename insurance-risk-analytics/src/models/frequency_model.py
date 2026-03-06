"""Poisson GLM for claim frequency modeling."""

import pandas as pd
import statsmodels.api as sm


def _ensure_numeric_frame(X):
    """Ensure exogenous data is purely numeric for statsmodels."""
    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def train_frequency_model(X_train, y_train):
    """Train a Poisson GLM frequency model."""
    X_train = _ensure_numeric_frame(X_train)
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0.0)
    X_train = sm.add_constant(X_train, has_constant="add")
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    return model


def predict_frequency(model, X):
    """Predict expected claim frequency."""
    X = _ensure_numeric_frame(X)
    X = sm.add_constant(X, has_constant="add")
    return model.predict(X)
