"""Random Forest classifier for fraud detection with SMOTE."""

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def train_fraud_model(X_train, y_train, n_estimators=300, max_depth=10, random_state=42):
    """Train a fraud detection model with SMOTE oversampling."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)
    return model


def predict_fraud_proba(model, X):
    """Predict fraud probability."""
    return model.predict_proba(X)[:, 1]
