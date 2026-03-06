"""Feature engineering for insurance risk analytics."""

import numpy as np
import pandas as pd


def create_features(df):
    """Create engineered features used across models."""
    df = df.copy()

    if "claim_amount" in df.columns and "premium" in df.columns:
        premium = df["premium"].replace(0, np.nan)
        df["claim_to_premium_ratio"] = (df["claim_amount"] / premium).fillna(0)
    else:
        df["claim_to_premium_ratio"] = 0.0

    if "claim_amount" in df.columns:
        df["log_claim_amount"] = np.log1p(df["claim_amount"].clip(lower=0))
    else:
        df["log_claim_amount"] = 0.0

    if "vehicle_year" in df.columns:
        current_year = pd.Timestamp.today().year
        vehicle_year = pd.to_numeric(df["vehicle_year"], errors="coerce")
        df["vehicle_age"] = (current_year - vehicle_year).clip(lower=0)
        df["vehicle_age"] = df["vehicle_age"].fillna(df["vehicle_age"].median())
    elif "vehicle_age" not in df.columns:
        df["vehicle_age"] = 0.0

    days_to_report = None
    if "claim_report_date" in df.columns and "incident_date" in df.columns:
        df["claim_report_date"] = pd.to_datetime(df["claim_report_date"], errors="coerce")
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
        days_to_report = (df["claim_report_date"] - df["incident_date"]).dt.days
    elif "claim_date" in df.columns and "policy_start_date" in df.columns:
        df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
        df["policy_start_date"] = pd.to_datetime(df["policy_start_date"], errors="coerce")
        days_to_report = (df["claim_date"] - df["policy_start_date"]).dt.days

    if days_to_report is not None:
        df["days_to_report"] = days_to_report.fillna(0)
        df["suspicious_delay_flag"] = (df["days_to_report"] > 30).astype(int)
    else:
        df["days_to_report"] = 0
        df["suspicious_delay_flag"] = 0

    if "claim_amount" in df.columns:
        threshold = df["claim_amount"].quantile(0.90)
        high_claim = df["claim_amount"] > threshold
        fast_report = df["days_to_report"] <= 7
        df["large_and_fast_flag"] = (high_claim & fast_report).astype(int)
    else:
        df["large_and_fast_flag"] = 0

    return df
