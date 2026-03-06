"""Poisson GLM for claim frequency modeling."""

import statsmodels.api as sm


def train_frequency_model(X_train, y_train):
    """Train a Poisson GLM frequency model."""
    X_train = sm.add_constant(X_train, has_constant="add")
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    return model


def predict_frequency(model, X):
    """Predict expected claim frequency."""
    X = sm.add_constant(X, has_constant="add")
    return model.predict(X)
