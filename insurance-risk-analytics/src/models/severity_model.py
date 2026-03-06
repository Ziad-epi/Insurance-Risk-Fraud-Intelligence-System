"""Gamma GLM for claim severity modeling."""

import numpy as np
import statsmodels.api as sm


def train_severity_model(X_train, y_train):
    """Train a Gamma GLM severity model."""
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
    X = sm.add_constant(X, has_constant="add")
    return model.predict(X)
