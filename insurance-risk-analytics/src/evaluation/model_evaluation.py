"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix


def regression_metrics(y_true, y_pred):
    """Compute MAE and RMSE for regression tasks."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": mae, "rmse": rmse}


def classification_metrics(y_true, y_prob, threshold=0.5):
    """Compute ROC AUC and confusion matrix for classification tasks."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"roc_auc": roc_auc, "confusion_matrix": cm}
