"""Gamma GLM for claim severity modeling."""

import numpy as np
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


def train_severity_model(X_train, y_train):
    """Train a Gamma GLM severity model."""
    X_train = _ensure_numeric_frame(X_train)
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0.0)
    X_train = sm.add_constant(X_train, has_constant="add")
    y_train = np.clip(y_train, 1e-6, None)
    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Gamma(sm.families.links.log()),
    ).fit()
    return model


def predict_severity(model, X):
    """Predict expected claim severity."""
    X = _ensure_numeric_frame(X)
    X = sm.add_constant(X, has_constant="add")
    return model.predict(X)
