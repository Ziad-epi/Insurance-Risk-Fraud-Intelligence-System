"""Main pipeline for insurance risk analytics."""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.data.data_loader import load_policy_data, load_claim_data
from src.data.data_preprocessing import clean_data, encode_features, train_test_split_data
from src.features.feature_engineering import create_features
from src.models.frequency_model import train_frequency_model, predict_frequency
from src.models.severity_model import train_severity_model, predict_severity
from src.models.fraud_model import train_fraud_model, predict_fraud_proba
from src.models.pricing_model import compute_pure_premium
from src.evaluation.model_evaluation import regression_metrics, classification_metrics
from src.explainability.shap_analysis import generate_shap_plots
from src.utils.helpers import save_model


def load_config(path="config/config.yaml"):
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_policy_key(df):
    for key in ["policy_id", "policy_number", "policy_no"]:
        if key in df.columns:
            return key
    return None


def build_modeling_dataset(policy_df, claim_df):
    """Merge and aggregate raw data into a modeling dataset."""
    policy_df = policy_df.copy()
    claim_df = claim_df.copy()

    policy_key = _find_policy_key(policy_df)
    claim_key = _find_policy_key(claim_df)

    if policy_key is None:
        policy_key = "policy_id"
        policy_df[policy_key] = np.arange(len(policy_df))

    if claim_key is None:
        claim_key = policy_key
        claim_df[claim_key] = policy_df[policy_key].sample(len(claim_df), replace=True, random_state=42).values

    if claim_key != policy_key:
        claim_df = claim_df.rename(columns={claim_key: policy_key})

    agg_dict = {}
    if "claim_id" in claim_df.columns:
        agg_dict["number_of_claims"] = ("claim_id", "count")
    else:
        agg_dict["number_of_claims"] = (claim_df.columns[0], "count")

    if "claim_amount" in claim_df.columns:
        agg_dict["total_claim_amount"] = ("claim_amount", "sum")
        agg_dict["average_claim_amount"] = ("claim_amount", "mean")
    else:
        claim_df["claim_amount"] = 0
        agg_dict["total_claim_amount"] = ("claim_amount", "sum")
        agg_dict["average_claim_amount"] = ("claim_amount", "mean")

    if "fraud_label" in claim_df.columns:
        agg_dict["fraud_label"] = ("fraud_label", "max")

    if "claim_date" in claim_df.columns:
        agg_dict["claim_date"] = ("claim_date", "max")
    if "incident_date" in claim_df.columns:
        agg_dict["incident_date"] = ("incident_date", "max")
    if "claim_report_date" in claim_df.columns:
        agg_dict["claim_report_date"] = ("claim_report_date", "max")

    claim_agg = claim_df.groupby(policy_key).agg(**agg_dict).reset_index()

    df = policy_df.merge(claim_agg, on=policy_key, how="left")
    df["number_of_claims"] = df["number_of_claims"].fillna(0).astype(int)
    df["total_claim_amount"] = df["total_claim_amount"].fillna(0)
    df["average_claim_amount"] = df["average_claim_amount"].fillna(0)
    df["claim_amount"] = df["average_claim_amount"]

    return df


def prepare_features(df):
    """Clean, feature engineer, and encode dataset."""
    df = clean_data(df)
    df = create_features(df)

    if "premium" not in df.columns:
        df["premium"] = 1000.0

    if "fraud_label" not in df.columns:
        df["fraud_label"] = (
            (df["large_and_fast_flag"] == 1) | (df["claim_to_premium_ratio"] > 3)
        ).astype(int)

    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category")
    ]
    df_encoded = encode_features(df, categorical_cols=categorical_cols)

    date_cols = df_encoded.select_dtypes(include=["datetime64[ns]"]).columns
    df_encoded = df_encoded.drop(columns=list(date_cols))

    return df_encoded


def main():
    config = load_config()
    paths = config["paths"]

    raw_dir = Path(paths["raw_data"])
    processed_dir = Path(paths["processed_data"])
    model_dir = Path(paths["models"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    policy_df = load_policy_data(raw_dir=raw_dir)
    claim_df = load_claim_data(raw_dir=raw_dir)

    print("Building modeling dataset...")
    base_df = build_modeling_dataset(policy_df, claim_df)
    df = prepare_features(base_df)

    target_cols = ["number_of_claims", "claim_amount", "fraud_label"]
    leakage_cols = ["total_claim_amount", "average_claim_amount"]
    feature_cols = [c for c in df.columns if c not in target_cols + leakage_cols]

    print("Training frequency model...")
    freq_df = df[feature_cols + ["number_of_claims"]]
    X_train, X_test, y_train, y_test = train_test_split_data(freq_df, "number_of_claims")
    freq_model = train_frequency_model(X_train, y_train)
    freq_pred = predict_frequency(freq_model, X_test)
    freq_metrics = regression_metrics(y_test, freq_pred)

    print("Training severity model...")
    sev_df = df[df["claim_amount"] > 0].copy()
    if len(sev_df) > 10:
        sev_df = sev_df[feature_cols + ["claim_amount"]]
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split_data(sev_df, "claim_amount")
        sev_model = train_severity_model(X_train_s, y_train_s)
        sev_pred = predict_severity(sev_model, X_test_s)
        sev_metrics = regression_metrics(y_test_s, sev_pred)
    else:
        sev_model = None
        sev_metrics = {"mae": float("nan"), "rmse": float("nan")}

    print("Computing pure premium...")
    df["expected_claim_frequency"] = predict_frequency(freq_model, df[feature_cols])
    if sev_model is not None:
        df["expected_claim_severity"] = predict_severity(sev_model, df[feature_cols])
    else:
        df["expected_claim_severity"] = 0

    df = compute_pure_premium(df)

    print("Training fraud model...")
    fraud_df = df[feature_cols + ["fraud_label"]]
    if fraud_df["fraud_label"].nunique() > 1:
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split_data(
            fraud_df, "fraud_label", stratify=True
        )
        fraud_params = config["model_params"]["fraud_model"]
        fraud_model = train_fraud_model(
            X_train_f,
            y_train_f,
            n_estimators=fraud_params.get("n_estimators", 300),
            max_depth=fraud_params.get("max_depth", 10),
            random_state=42,
        )
        fraud_prob = predict_fraud_proba(fraud_model, X_test_f)
        fraud_metrics = classification_metrics(y_test_f, fraud_prob)
        df["fraud_probability"] = predict_fraud_proba(fraud_model, df[feature_cols])
    else:
        fraud_model = None
        fraud_metrics = {"roc_auc": float("nan"), "confusion_matrix": [[0, 0], [0, 0]]}

    print("Generating SHAP explainability...")
    if fraud_model is not None:
        sample_X = df[feature_cols].sample(min(500, len(df)), random_state=42)
        generate_shap_plots(fraud_model, sample_X)

    print("Saving models and artifacts...")
    save_model(freq_model, model_dir / "frequency_model.pkl")
    if sev_model is not None:
        save_model(sev_model, model_dir / "severity_model.pkl")
    if fraud_model is not None:
        save_model(fraud_model, model_dir / "fraud_model.pkl")

    df.to_csv(processed_dir / "modeling_dataset.csv", index=False)

    print("Pipeline completed.")
    print("Frequency metrics:", freq_metrics)
    print("Severity metrics:", sev_metrics)
    print("Fraud metrics:", fraud_metrics)


if __name__ == "__main__":
    main()
